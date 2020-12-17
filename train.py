import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import pathlib, sys, os, random, time
import numba, cv2, gc
import glob
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
# %matplotlib inline
from tqdm.notebook import tqdm
import albumentations as A
import rasterio
from rasterio.windows import Window
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as D
import torch.utils.model_zoo as model_zoo
import torchvision
from torchvision import transforms as T
from torchvision.models.resnet import ResNet
from torchvision.models.resnet import BasicBlock
from torchvision.models.resnet import Bottleneck
from pretrainedmodels.models.torchvision_models import pretrained_settings
import functools


# from model import UNet
from dataset import HubDataset



def set_seeds(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
set_seeds();
print('set seeds done')


DATA_PATH = '../hubmap-kidney-segmentation/'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu' 
print(DEVICE)


import logging
logging.basicConfig(filename='log.log',
                    format='%(asctime)s - %(name)s - %(levelname)s -%(module)s:  %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S ',
                    level=logging.INFO)


def get_model():
    model = torchvision.models.segmentation.fcn_resnet50(False)
    
    pth = torch.load('../fcn_resnet50_coco-1167a1af.pth')
    for key in ["aux_classifier.0.weight", "aux_classifier.1.weight", "aux_classifier.1.bias", "aux_classifier.1.running_mean", "aux_classifier.1.running_var", "aux_classifier.1.num_batches_tracked", "aux_classifier.4.weight", "aux_classifier.4.bias"]:
        del pth[key]
    
    model.load_state_dict(pth)
    model.classifier[4] = nn.Conv2d(512, 1, kernel_size=(1, 1), stride=(1, 1))
    return model



def train(model, train_loader, criterion, optimizer):
    losses = []
    for i, (image, target) in enumerate(train_loader):
        image, target = image.to(DEVICE), target.float().to(DEVICE)
        optimizer.zero_grad()
        
        output = model(image)
        loss = criterion(output, target, 1, False)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        print('train, ', loss.item())
    return np.array(losses).mean()


def np_dice_score(probability, mask):
    p = probability.reshape(-1)
    t = mask.reshape(-1)

    p = p>0.5
    t = t>0.5
    uion = p.sum() + t.sum()
    
    overlap = (p*t).sum()
    dice = 2*overlap/(uion+0.001)
    return dice

def validation(model, val_loader, criterion):
    val_probability, val_mask = [], []
    model.eval()
    with torch.no_grad():
        for image, target in val_loader:
            image, target = image.to(DEVICE), target.float().to(DEVICE)
            output = model(image)
            
            output_ny = output.sigmoid().data.cpu().numpy()
            target_np = target.data.cpu().numpy()
            
            val_probability.append(output_ny)
            val_mask.append(target_np)
            
    val_probability = np.concatenate(val_probability)
    val_mask = np.concatenate(val_mask)
    
    return np_dice_score(val_probability, val_mask)
    


class SoftDiceLoss(nn.Module):
    def __init__(self, smooth=1., dims=(-2,-1)):
        super(SoftDiceLoss, self).__init__()
        self.smooth = smooth
        self.dims = dims
    
    def forward(self, x, y):
        tp = (x * y).sum(self.dims)
        fp = (x * (1 - y)).sum(self.dims)
        fn = ((1 - x) * y).sum(self.dims)
        dc = (2 * tp + self.smooth) / (2 * tp + fp + fn + self.smooth)
        dc = dc.mean()
        
        return 1 - dc

bce_fn = nn.BCEWithLogitsLoss()
# bce_fn = nn.BCELoss()
dice_fn = SoftDiceLoss()
    
def loss_fn(y_pred, y_true, ratio=0.8, hard=False):
    bce = bce_fn(y_pred, y_true)
    if hard:
        dice = dice_fn((y_pred.sigmoid()).float() > 0.5, y_true)
    else:
        dice = dice_fn(y_pred.sigmoid(), y_true)
    return ratio*bce + (1-ratio)*dice





EPOCHES = 5
BATCH_SIZE = 8

WINDOW=1024
MIN_OVERLAP=40
NEW_SIZE=256

train_trfm = A.Compose([
    # A.RandomCrop(NEW_SIZE*3, NEW_SIZE*3),
    A.Resize(NEW_SIZE, NEW_SIZE),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(),
    A.OneOf([
        A.RandomContrast(),
        A.RandomGamma(),
        A.RandomBrightness(),
        A.ColorJitter(brightness=0.07, contrast=0.07,
                   saturation=0.1, hue=0.1, always_apply=False, p=0.3),
        ], p=0.3),
#     A.OneOf([
#         A.OpticalDistortion(p=0.5),
#         A.GridDistortion(p=0.5),
#         A.IAAPiecewiseAffine(p=0.5),
#     ], p=0.3),
#     A.ShiftScaleRotate(),
])

val_trfm = A.Compose([
    # A.CenterCrop(NEW_SIZE, NEW_SIZE),
    A.Resize(NEW_SIZE,NEW_SIZE),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(),
#     A.OneOf([
#         A.RandomContrast(),
#         A.RandomGamma(),
#         A.RandomBrightness(),
#         A.ColorJitter(brightness=0.07, contrast=0.07,
#                    saturation=0.1, hue=0.1, always_apply=False, p=0.3),
#         ], p=0.3),
#     A.OneOf([
#         A.OpticalDistortion(p=0.5),
#         A.GridDistortion(p=0.5),
#         A.IAAPiecewiseAffine(p=0.5),
#     ], p=0.3),
#     A.ShiftScaleRotate(),
])

# 每个file单独做一个验证集
print('每个file单独做一个验证集')
tiff_ids = np.array([x.split('/')[-1][:-5] for x in glob.glob('../hubmap-kidney-segmentation/train/*.tiff')])
print(tiff_ids.shape)
skf = KFold(n_splits=8)

for fold_idx, (train_idx, val_idx) in enumerate(skf.split(tiff_ids, tiff_ids)):
    print(tiff_ids[val_idx])
    
    # break
    train_ds = HubDataset(DATA_PATH, tiff_ids[train_idx], window=WINDOW, overlap=MIN_OVERLAP, 
                          threshold=100, transform=train_trfm)
    valid_ds = HubDataset(DATA_PATH, tiff_ids[val_idx], window=WINDOW, overlap=MIN_OVERLAP, 
                          threshold=100, transform=val_trfm, isvalid=False)

    print(len(train_ds), len(valid_ds))
    
    # define training and validation data loaders
    train_loader = D.DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=12)

    val_loader = D.DataLoader(
        valid_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=12)
    
    # model = get_model()
    # model = Unet(encoder_name="resnet34",classes=1,activation=None)
   
    import segmentation_models_pytorch as smp

    model = smp.Unet(
        encoder_name="efficientnet-b1",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights="imagenet",     # use `imagenet` pretreined weights for encoder initialization
        in_channels=3,                  # model input channels (1 for grayscale images, 3 for RGB, etc.)
        classes=1,                      # model output channels (number of classes in your dataset)
    )

    model.to(DEVICE)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    # lr_step = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 2)
    lr_step = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)
    
    header = r'''
            Train | Valid
    Epoch |  Loss |  Dice (Best) | Time
    '''
    print(header)
    #          Epoch         metrics            time
    raw_line = '{:6d}' + '\u2502{:7.4f}'*3 + '\u2502{:6.2f}'
    
    best_dice = 0
    for epoch in range(1, EPOCHES+1):
        start_time = time.time()
        model.train()
        train_loss = train(model, train_loader, loss_fn, optimizer)
        val_dice = validation(model, val_loader, loss_fn)
        lr_step.step(val_dice)

        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), 'fold_{0}.pth'.format(fold_idx))
        
        logging.info(raw_line.format(epoch, train_loss, val_dice, best_dice, (time.time()-start_time)/60**1))
        
    
    del train_loader, val_loader, train_ds, valid_ds
    gc.collect();
    
    break




model.load_state_dict(torch.load("./fold_0.pth"))
model.eval()

valid_ds = HubDataset(DATA_PATH, tiff_ids[val_idx], window=WINDOW, overlap=MIN_OVERLAP, 
                          threshold=100, transform=val_trfm, isvalid=False)



c = 1
for idx in range(80, 200):
    
    image, mask = valid_ds[idx]
    if mask.max() == 0:
        continue
    
    c += 1
    if c > 10:
        continue
    
    plt.figure(figsize=(16,8))
    plt.subplot(141)
    plt.imshow(mask[0], cmap='gray')
    plt.subplot(142)
    plt.imshow(image[0]);

    with torch.no_grad():
        image = image.to(DEVICE)[None]

        score = model(image)[0][0]

        score2 = model(torch.flip(image, [0, 3]))
        score2 = torch.flip(score2, [3, 0])[0][0]

        score3 = model(torch.flip(image, [1, 2]))
        score3 = torch.flip(score3, [2, 1])[0][0]


        score_mean = (score + score2 + score3) / 3.0

        score_sigmoid = score_mean.sigmoid().cpu().numpy()
        score_sigmoid = cv2.resize(score_sigmoid, (WINDOW, WINDOW))

        score = score.sigmoid().cpu().numpy()
        score = cv2.resize(score, (WINDOW, WINDOW))


    plt.subplot(143)
    plt.imshow((score_sigmoid > 0.5).astype(int));

    plt.subplot(144)
    plt.imshow((score > 0.5).astype(int));




