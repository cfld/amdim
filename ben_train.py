#!/usr/bin/env python

"""
    train_ben.py
"""

import os
import sys
import json
import random
import argparse
import numpy as np
from time import time
from tqdm import tqdm
from sklearn import metrics
from tensorboardX import SummaryWriter

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

def to_numpy(x):
    return x.detach().cpu().numpy()

def cov(x, y):
    z = x.copy()
    z[y == 0] = np.inf
    min_pos = z.min(axis=-1, keepdims=True)
    return (x >= min_pos).sum(axis=-1).mean()

def sample_f1(x, y):
    tp = (x != 0) & (y != 0)
    p  = np.mean(tp.sum(axis=-1) / (x.sum(axis=-1) + 1e-10))
    r  = np.mean(tp.sum(axis=-1) / (y.sum(axis=-1) + 1e-10))
    return 2 * (p * r) / (p + r)

def one_error(x, y):
    return 1 - y[(np.arange(y.shape[0]), np.argmax(x, axis=-1))].mean()

def compute_metrics(x, y):
    return {
        "oe"  : one_error(x, y),
        "f1"  : sample_f1(x > 0, y),
        "cov" : cov(x, y),
    }


def do_eval(model, test_loader, device):
    all_mlp_out = []
    all_lin_out = []
    all_labels  = []
    
    counter = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.cpu()
        with torch.no_grad():
            res_dict = model(x1=images, class_only=True)
            mlp_out, lin_out = res_dict['class']
            
            all_mlp_out.append(to_numpy(mlp_out))
            all_lin_out.append(to_numpy(lin_out))
            all_labels.append(to_numpy(labels))
        
        counter += 1
        if counter >= 10:
            break
    
    all_mlp_out = np.row_stack(all_mlp_out)
    all_lin_out = np.row_stack(all_lin_out)
    all_labels  = np.row_stack(all_labels)
    
    return {
        "mlp_metrics" : compute_metrics(all_mlp_out, all_labels),
        "lin_metrics" : compute_metrics(all_lin_out, all_labels),
    }


def dump_feats(model, test_loader, device):
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
    
    _ = model.train()
    
    all_mlp_out = np.row_stack(all_mlp_out)
    all_lin_out = np.row_stack(all_lin_out)
    all_labels  = np.row_stack(all_labels)
    all_feats   = np.row_stack(all_feats)
    
    return all_mlp_out, all_lin_out, all_labels, all_feats

# --
# CLI

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--input_dir', type=str, default='/raid/users/bjohnson/projects/benet/data/bigearthnet_patches/')
    
    # parameters for general training stuff
    parser.add_argument('--batch_size',    type=int,   default=192)
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
    
    parser.add_argument('--output_dir', type=str, default='runs')
    parser.add_argument('--run_name',   type=str, default='run0')
    
    return parser.parse_args()

args = parse_args()

# if args.amp: mixed_precision.enable_mixed_precision()
os.makedirs(os.path.join(args.output_dir, args.run_name), exist_ok=True)

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

scheduler = MultiStepLR(optim_inf, milestones=[10, 20], gamma=0.2)
epochs    = 50

model, optim_inf = mixed_precision.initialize(model, optim_inf)
optim_raw        = mixed_precision.get_optimizer(optim_inf)

for p in optim_raw.param_groups:
    lr_real = p['lr']

# --
# Run

_ = torch.cuda.empty_cache()

writer = SummaryWriter(logdir=os.path.join(os.path.join(args.output_dir, args.run_name), 'logs'))

t = time()
step_counter = 0
for epoch_idx in range(epochs):
    for batch_idx, ((x1, x2), y) in enumerate(train_loader):
        
        # --
        # Forward
        
        y  = torch.cat([y, y]).to(device)
        x1 = x1.to(device)
        x2 = x2.to(device)
        
        res_dict = model(x1=x1, x2=x2, class_only=False)
        
        lgt_glb_mlp, lgt_glb_lin = res_dict['class']
        
        loss_g2l = (res_dict['g2l_1t5'] + res_dict['g2l_1t7'] + res_dict['g2l_5t5'])
        loss_inf = loss_g2l + res_dict['lgt_reg']
        loss_cls = (F.binary_cross_entropy_with_logits(lgt_glb_mlp, y) + F.binary_cross_entropy_with_logits(lgt_glb_lin, y))
        
        # --
        # Log
        
        log = {
            'epoch_idx'    : int(epoch_idx),
            'batch_idx'    : int(batch_idx),
            'loss_inf'     : float(loss_inf),
            'loss_cls'     : float(loss_cls),
            'mlp_metrics'  : compute_metrics(to_numpy(lgt_glb_mlp), to_numpy(y)),
            'lin_metrics'  : compute_metrics(to_numpy(lgt_glb_lin), to_numpy(y)),
            'elapsed'      : time() - t,
        }
        print(json.dumps(log))
        sys.stdout.flush()
        
        writer.add_scalar('loss_inf', log['loss_inf'], step_counter)
        writer.add_scalar('loss_cls', log['loss_cls'], step_counter)
        
        writer.add_scalar('mlp_oe',  log['mlp_metrics']['oe'], step_counter)
        writer.add_scalar('mlp_f1',  log['mlp_metrics']['f1'], step_counter)
        writer.add_scalar('mlp_cov', log['mlp_metrics']['cov'], step_counter)
        
        writer.add_scalar('lin_oe',  log['lin_metrics']['oe'], step_counter)
        writer.add_scalar('lin_f1',  log['lin_metrics']['f1'], step_counter)
        writer.add_scalar('lin_cov', log['lin_metrics']['cov'], step_counter)
        
        if step_counter % 10 == 0:
            valid_log = do_eval(model, test_loader, device)
            
            writer.add_scalar('val_mlp_oe',  valid_log['mlp_metrics']['oe'], step_counter)
            writer.add_scalar('val_mlp_f1',  valid_log['mlp_metrics']['f1'], step_counter)
            writer.add_scalar('val_mlp_cov', valid_log['mlp_metrics']['cov'], step_counter)
            
            writer.add_scalar('val_lin_oe',  valid_log['lin_metrics']['oe'], step_counter)
            writer.add_scalar('val_lin_f1',  valid_log['lin_metrics']['f1'], step_counter)
            writer.add_scalar('val_lin_cov', valid_log['lin_metrics']['cov'], step_counter)
        
        writer.flush()
        
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
    
    scheduler.step(epoch_idx)
    
    mlp_out, lin_out, labels, feats = dump_feats(model, test_loader, device)
    np.save(os.path.join(args.output_dir, args.run_name, f'mlp_out.{epoch_idx}.npy'), mlp_out)
    np.save(os.path.join(args.output_dir, args.run_name, f'lin_out.{epoch_idx}.npy'), lin_out)
    np.save(os.path.join(args.output_dir, args.run_name, f'labels.{epoch_idx}.npy'), labels)
    np.save(os.path.join(args.output_dir, args.run_name, f'feats.{epoch_idx}.npy'), feats)



