import pandas as pd
import numpy as np
import os
import glob
import cv2
from PIL import Image, ImageDraw
import albumentations as A
import torch
import random
import math
from torch.utils.data.sampler import Sampler
import torch.nn.functional as F
from collections import defaultdict
from bisect import bisect_right
import copy
import timm
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.optim import Optimizer
from albumentations.pytorch import ToTensorV2
from torch.utils.data.dataset import Dataset
from sklearn.model_selection import StratifiedKFold, KFold
from tqdm import tqdm



# 构建训练集，方便5折交叉验证
train_path = 'pepper_diseases/npy/train_resize576/'
test_path = 'pepper_diseases/npy/test_resize512/'

train_df = pd.DataFrame({'path': glob.glob(train_path+'/*/*')})
train_df['label'] = train_df['path'].apply(lambda x: int(x.split('/')[-2][1:]))


train_dict = {}
train_path = []
for i in train_df.path.values:
    train_path.append(i)
    
    
for p in tqdm(train_path):
    train_dict[p] = np.load(p)
    
    
    
# 多折训练
k = 5
folds = StratifiedKFold(n_splits=k, shuffle=True, random_state=2020)
fold = 0
epochs = 50

for fold_, (tr_idx, val_idx) in enumerate(folds.split(train_df, train_df['label'])):
    model = Net().cuda()
    print(val_idx)
    
    train_loader = torch.utils.data.DataLoader(
        Pepper_diseases(train_df.iloc[tr_idx]['path'].values, train_df.iloc[tr_idx]['label'].values, build_transforms(True)),
        batch_size=4, shuffle=True, num_workers=5)

    val_loader = torch.utils.data.DataLoader(
        Pepper_diseases(train_df.iloc[val_idx]['path'].values, train_df.iloc[val_idx]['label'].values, build_transforms(False)),
        batch_size=10, shuffle=False, num_workers=5)
    

    criterion = CrossEntropyLabelSmooth(num_classes=11)
    optimizer = torch.optim.SGD(model.parameters(), 0.005, momentum=0.9)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=5, num_training_steps=epochs*2)
    best_score = 0.

    for epoch in range(epochs):
        print('Epoch: ', epoch)

        train(train_loader, model, criterion, optimizer, epoch)
        score = validate(val_loader, model)
        if score > best_score:
            best_score = score
            torch.save(model.state_dict(), 'model_576_{0}.pt'.format(fold))

        print('验证集准确率', score)
        scheduler.step()
    
    fold += 1
