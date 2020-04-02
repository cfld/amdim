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

BAND_STATS = {
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

def drop_channels(x, p=0.2, **kwargs):
    sel = np.random.uniform(0, 1, x.shape[-1]) < p
    if sel.sum() == 0:
        return x
    else:
        x[...,sel] = BAND_STATS['mean'][sel].reshape(1, 1, -1) # Replace 
        return x


class AugmentBEN:
    def __init__(self):
        
        self.train_transform = ACompose([
            atransforms.HorizontalFlip(p=0.5),
            atransforms.RandomRotate90(p=1.0),
            atransforms.ShiftScaleRotate(p=1.0),
            atransforms.RandomSizedCrop((60, 120), height=128, width=128, interpolation=3),
            # atransforms.Lambda(drop_channels) # !! Maybe too much noise?
            # {jitter color, random greyscale}
        ])
        
        self.valid_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=BAND_STATS['mean'], std=BAND_STATS['std'])
        ])
    
    def __call__(self, inp):
        a = self.train_transform(image=inp)['image']
        b = self.train_transform(image=inp)['image']
        
        a = self.valid_transform(a)
        b = self.valid_transform(b)
        
        return a, b


class BigEarthNet(Dataset):
    def __init__(self, split, root):
        self.root        = root
        self.split       = split
        
        self.patch_names = open(f'data/ben_splits/{split}.csv').read().splitlines()
        self.patch_names = np.random.permutation(self.patch_names)
        
        self.labels      = pickle.load(open('data/labels_ben_19.pkl', 'rb'))
        self.num_classes = len(np.unique(np.hstack(list(self.labels.values()))))
        
        transform = AugmentBEN()
        if split == 'train':
            self.transform = transform
        elif split in ['val', 'test']:
            self.transform = transform.valid_transform
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