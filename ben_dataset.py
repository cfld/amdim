#!/usr/bin/env python

"""
    ben_dataset.py
"""

import os
import pickle
import numpy as np

import torch
from torchvision import transforms
from albumentations import Compose as ACompose
from albumentations.augmentations import transforms as atransforms

from torch.utils.data import Dataset

# --
# Helpers

BANDS = ('B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12')

BEN_BAND_STATS = {
    'mean': np.array([
        340.76769064,
        429.9430203,
        614.21682446,
        590.23569706,
        950.68368468,
        1792.46290469,
        2075.46795189,
        2218.94553375,
        2266.46036911,
        2246.0605464,
        1594.42694882,
        1009.32729131
    ]),
    'std': np.array([
        554.81258967,
        572.41639287,
        582.87945694,
        675.88746967,
        729.89827633,
        1096.01480586,
        1273.45393088,
        1365.45589904,
        1356.13789355,
        1302.3292881,
        1079.19066363,
        818.86747235,
    ])
}

# --
# Helpers

def drop_channels(x, **kwargs):
    sel = np.random.uniform(0, 1, x.shape[-1]) < (2 / len(BANDS))
    if sel.sum() == 0:
        return x
    else:
        x[...,sel] = BEN_BAND_STATS['mean'][sel].reshape(1, 1, -1) # Replace 
        return x


class BENTransformTrain:
    def __init__(self):
        
        self.train_transform = ACompose([
            atransforms.HorizontalFlip(p=0.5),
            atransforms.RandomRotate90(p=1.0),
            atransforms.ShiftScaleRotate(p=1.0),
            atransforms.RandomSizedCrop((60, 120), height=128, width=128, interpolation=3),
            
            # Medium aggressive augmentation
            atransforms.GridDistortion(num_steps=5, p=0.5),     # !! Maybe too much noise?
            
            # More aggressive augmentation
            # atransforms.RandomBrightness(p=0.5),             # !! Maybe too much noise?
            # atransforms.Lambda(drop_channels, p=0.5),        # !! Maybe too much noise?
        ])
        
        self.post_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=BEN_BAND_STATS['mean'], std=BEN_BAND_STATS['std'])
        ])
    
    def __call__(self, inp):
        a = self.train_transform(image=inp)['image']
        b = self.train_transform(image=inp)['image']
        
        a = self.post_transform(a)
        b = self.post_transform(b)
        
        return a, b


class BENTransformValid:
    def __init__(self):
        self.transform = ACompose([
            atransforms.Resize(128, 128, interpolation=3)
        ])
        
        self.post_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=BEN_BAND_STATS['mean'], std=BEN_BAND_STATS['std'])
        ])
    
    def __call__(self, inp):
        a = self.transform(image=inp)['image']
        a = self.post_transform(a)
        return a


class BigEarthNet(Dataset):
    def __init__(self, split, root, preshuffle=True):
        self.root        = root
        self.split       = split
        
        self.patch_names = open(f'data/ben_splits/{split}.csv').read().splitlines()
        
        if preshuffle:
            self.patch_names = np.random.permutation(self.patch_names)
        
        self.labels      = pickle.load(open('data/labels_ben_19.pkl', 'rb'))
        self.num_classes = len(np.unique(np.hstack(list(self.labels.values()))))
        
        if split == 'train':
            self.transform = BENTransformTrain()
        elif split in ['val', 'test']:
            self.transform = BENTransformValid()
        else:
            raise Exception
    
    def __len__(self):
        return len(self.patch_names)
    
    def __getitem__(self, idx):
        patch_name = self.patch_names[idx]
        patch_file = os.path.join(self.root, patch_name + '.npy')
        
        X = np.load(patch_file)
        X = X.transpose(1, 2, 0)
        X = X.astype(np.float32)
        X = self.transform(X)
        
        y = torch.zeros(self.num_classes)
        y[self.labels[patch_name]] = 1
        
        return X, y

# root = '/raid/users/bjohnson/projects/benet/data/bigearthnet_patches/'
# b = BigEarthNet(split='val', root=root)
# b[1000]