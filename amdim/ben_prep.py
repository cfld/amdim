#!/usr/bin/env python

"""
    ben_prep.py
"""

import os
import json
import numpy as np
from glob import glob
from PIL import Image
from tqdm import tqdm
from tifffile import imread as tiffread
from joblib import Parallel, delayed

BANDS = ('B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12')

# --
# Helpers

def get_name2new():
    label_indices = json.load(open('ben_label_indices.json'))
    name2old      = label_indices['original_labels']
    
    old2new = {}
    for new, olds in enumerate(label_indices['label_conversion']):
        for o in olds:
            old2new[o] = new
            
    name2new = {}
    for k,v in name2old.items():
        name2new[k] = old2new.get(v, -1)
    
    return name2new


def bilinear_upsample(x, n=120):
    dtype = x.dtype
    assert len(x.shape) == 2
    if (x.shape[0] == n) and (x.shape[1] == n):
        return x
    else:
        x = x.astype(np.float)
        x = Image.fromarray(x)
        x = x.resize((n, n), Image.BILINEAR)
        x = np.array(x)
        x = x.astype(dtype)
        return x


def load_patch(patch_dir):
    patch_name = os.path.basename(patch_dir)
    patch      = [tiffread(os.path.join(patch_dir, f'{patch_name}_{band}.tif')) for band in BANDS]
    patch      = np.stack([bilinear_upsample(xx) for xx in patch])
    return patch


def load_labels(patch_dir):
    patch_name = os.path.basename(patch_dir)
    meta       = json.load(open(os.path.join(patch_dir, patch_name + '_labels_metadata.json')))
    labels     = [name2new[l] for l in meta['labels']]
    labels     = [l for l in labels if l != -1]
    return labels

# --
# Run

indir  = '/home/ubuntu/projects/benet/data/bigearthnet'
outdir = '/home/ubuntu/projects/benet/data/bigearthnet_patch'

os.makedirs(outdir, exist_ok=True)

patch_dirs = sorted(glob(os.path.join(indir, '*')))

# --
# Prep images

def _f(patch_dir):
    np.save(os.path.join(outdir, os.path.basename(patch_dir) + '.npy'), load_patch(patch_dir))

jobs = [delayed(_f)(patch_dir) for patch_dir in patch_dirs]
res  = Parallel(backend='multiprocessing', n_jobs=8, verbose=10)(jobs)

# --
# Prep metadata

name2new = get_name2new()
labels   = [load_labels(patch_dir) for patch_dir in tqdm(patch_dirs)]


