#!/usr/bin/env python

"""
    train_ben.py
"""

import os
import json
import random
import argparse
import numpy as np
from time import time
from tqdm import tqdm
from sklearn import metrics

import torch
from torch import nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import MultiStepLR

import mixed_precision
from model import Model
from task_self_supervised import train_self_supervised
from datasets import Dataset, build_dataset, get_dataset, get_encoder_size
from utils import _warmup_batchnorm

from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from joblib import Parallel, delayed

# --
# Helpers

def _cv_score(X, y):
    if (y == 0).all():
        return np.zeros(*y.shape)
    elif (y == 1).all():
        return np.ones(*y.shape)
    else:
        try:
            cv = StratifiedKFold(n_splits=2, shuffle=True)
            return cross_val_predict(LinearSVC(), X, y, cv=cv)
        except:
            return np.zeros(*y.shape) + (y.mean() > 0.5)


def cv_score(X, y):
    jobs  = [delayed(_cv_score)(X, y[:,i]) for i in range(y.shape[1])]
    preds = Parallel(backend='multiprocessing', n_jobs=20)(jobs)
    preds = np.row_stack(preds)
    return metrics.f1_score(y.ravel(), preds.ravel())

def to_numpy(x):
    return x.detach().cpu().numpy()

def test_model(model, test_loader, device, max_evals=200000):
    t = time()
    _warmup_batchnorm(model, test_loader, device, batches=50, train_loader=False)
    
    _ = model.eval()
    
    all_mlp_out = []
    all_lin_out = []
    all_labels  = []
    all_feats   = []
    
    n_seen = 0
    for images, labels in tqdm(test_loader, total=len(test_loader)):
        
        images = images.to(device)
        labels = labels.cpu()
        with torch.no_grad():
            res_dict = model(x1=images, class_only=True)
            feats    = res_dict['rkhs_glb']
            mlp_out, lin_out = res_dict['class']
            
            all_mlp_out.append(to_numpy(mlp_out))
            all_lin_out.append(to_numpy(lin_out))
            all_labels.append(to_numpy(labels))
            all_feats.append(to_numpy(feats))
        
        n_seen += labels.shape[0]
        if n_seen > max_evals:
            break
    
    _ = model.train()
    
    all_mlp_out = np.row_stack(all_mlp_out)
    all_lin_out = np.row_stack(all_lin_out)
    all_labels  = np.row_stack(all_labels)
    all_feats   = np.row_stack(all_feats)
    
    return {
        "mlp_auc" : metrics.roc_auc_score(all_labels.ravel(), all_mlp_out.ravel()),
        "lin_auc" : metrics.roc_auc_score(all_labels.ravel(), all_lin_out.ravel()),
        "f1"      : cv_score(all_feats, all_labels),
        "elapsed" : time() - t
    }


def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--input_dir', type=str, default='/raid/users/bjohnson/projects/benet/data/bigearthnet_patches/')
    
    # parameters for general training stuff
    parser.add_argument('--batch_size',    type=int,   default=128)
    parser.add_argument('--learning_rate', type=float, default=0.0002)
    parser.add_argument('--eval_interval', type=int,   default=500)
    parser.add_argument('--seed',          type=int,   default=1)
    parser.add_argument('--amp',           action='store_true', default=False)
    
    # parameters for model and training objective
    parser.add_argument('--classifiers', action='store_true')
    parser.add_argument('--ndf',     type=int, default=128)
    parser.add_argument('--n_rkhs',  type=int, default=1024)
    parser.add_argument('--tclip',   type=float, default=20.0)
    parser.add_argument('--n_depth', type=int, default=3)
    parser.add_argument('--use_bn',  type=int, default=0)
    
    # parameters for output, logging, checkpointing, etc
    parser.add_argument('--output_dir', type=str, default='./runs')
    parser.add_argument('--cpt_load_path', type=str, default=None)
    parser.add_argument('--cpt_name', type=str, default='amdim_cpt.pth')
    parser.add_argument('--run_name', type=str, default='default_run')
    
    return parser.parse_args()

args = parse_args()

# if args.amp: mixed_precision.enable_mixed_precision()

_ = random.seed(args.seed)
_ = np.random.seed(args.seed + 1)
_ = torch.manual_seed(args.seed + 2)
_ = torch.cuda.manual_seed(args.seed + 3)

device = torch.device('cuda')

# --
# Data

dataset = get_dataset('BEN')

train_loader, test_loader, num_classes = build_dataset(
    dataset      = dataset,
    batch_size   = args.batch_size,
    input_dir    = args.input_dir,
    labeled_only = args.classifiers,
    num_workers  = 8
)

# --
# Model

encoder_size = 128
model = Model(
    ndf          = args.ndf, 
    in_channels  = 12,
    n_classes    = num_classes, 
    n_rkhs       = args.n_rkhs,
    tclip        = args.tclip, 
    n_depth      = args.n_depth, 
    encoder_size = encoder_size,
    use_bn       = (args.use_bn == 1)
)

model.init_weights(init_scale=1.0)

model = model.to(device)

# --
# Optimizer

mods_inf    = [m for m in model.info_modules]
mods_cls    = [m for m in model.class_modules]
mods_to_opt = mods_inf + mods_cls

optim_inf = torch.optim.Adam(
    [{'params': mod.parameters(), 'lr': args.learning_rate} for mod in mods_to_opt],
    betas=(0.8, 0.999),
    weight_decay=1e-5, 
    eps=1e-8
)

scheduler = MultiStepLR(optim_inf, milestones=[30, 45], gamma=0.2)
epochs    = 50

model, optim_inf = mixed_precision.initialize(model, optim_inf)
optim_raw        = mixed_precision.get_optimizer(optim_inf)

for p in optim_raw.param_groups:
    lr_real = p['lr']

# --
# Run

_ = torch.cuda.empty_cache()

t = time()
step_counter = 0
for epoch_idx in range(epochs):
    for batch_idx, ((x1, x2), y) in enumerate(train_loader):
        
        # --
        # Forward
        
        y = torch.cat([y, y]).to(device)
        x1 = x1.to(device)
        x2 = x2.to(device)
        
        res_dict = model(x1=x1, x2=x2, class_only=False)
        
        lgt_glb_mlp, lgt_glb_lin = res_dict['class']
        
        loss_g2l = (res_dict['g2l_1t5'] + res_dict['g2l_1t7'] + res_dict['g2l_5t5'])
        loss_inf = loss_g2l + res_dict['lgt_reg']
        loss_cls = (F.binary_cross_entropy_with_logits(lgt_glb_mlp, y) + F.binary_cross_entropy_with_logits(lgt_glb_lin, y))
        
        # --
        # Log
        
        def cov(x, y):
            x_masked = x.clone()
            x_masked[(1 - y).bool()] = float('inf')
            min_score = x_masked.min(axis=-1, keepdims=True).values
            return x.gt(min_score).sum(axis=-1).float().mean()
        
        mlp_cov = cov(lgt_glb_mlp, y)
        lin_cov = cov(lgt_glb_lin, y)
        
        # y_flat  = y.cpu().numpy().ravel()
        # mlp_auc = metrics.roc_auc_score(y_flat, lgt_glb_mlp.data.cpu().numpy().ravel())
        # lin_auc = metrics.roc_auc_score(y_flat, lgt_glb_lin.data.cpu().numpy().ravel())
        
        print(json.dumps({
            'epoch_idx'    : int(epoch_idx),
            'batch_idx'    : int(batch_idx),
            'loss_inf'     : float(loss_inf),
            'loss_cls'     : float(loss_cls),
            # 'loss_g2l'     : float(loss_g2l),
            # 'lgt_reg'      : float(res_dict['lgt_reg']),
            # 'loss_g2l_1t5' : float(res_dict['g2l_1t5']),
            # 'loss_g2l_1t7' : float(res_dict['g2l_1t7']),
            # 'loss_g2l_5t5' : float(res_dict['g2l_5t5']),
            'mlp_cov'      : float(mlp_cov),
            'lin_cov'      : float(lin_cov),
            'elapsed'      : time() - t,
        }))
        sys.stdout.flush()
        
        # --
        # Step
        
        step_counter += 1
        
        # learning rate warmup
        if (step_counter < 500):
            lr_scale = min(1., float(step_counter + 1) / 500.)
            for pg in optim_raw.param_groups:
                pg['lr'] = lr_scale * lr_real
        
        optim_inf.zero_grad()
        mixed_precision.backward(loss_inf + loss_cls, optim_inf)
        optim_inf.step()
        
        # if step_counter % args.eval_interval == 0:
        #     test_scores = test_model(model, test_loader, device, max_evals=args.batch_size * 512)
        #     print(json.dumps(test_scores))
    
    scheduler.step(epoch)
    test_model(model, test_loader, device, max_evals=500000)



