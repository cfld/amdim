#!/usr/bin/env python

"""
    train_ben.py
"""

import os
import random
import argparse
import numpy as np
from time import time
from tqdm import tqdm

import torch
from torch import nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import MultiStepLR


import mixed_precision
from stats import StatTracker, AverageMeterSet, update_train_accuracies_multilabel
from datasets import Dataset, build_dataset, get_dataset, get_encoder_size
from model import Model
from checkpoint import Checkpointer
from task_self_supervised import train_self_supervised
from task_classifiers import train_classifiers


def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--input_dir', type=str, default='/raid/users/bjohnson/projects/benet/data/bigearthnet_patches/')
    
    # parameters for general training stuff
    parser.add_argument('--batch_size',    type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=0.0002)
    parser.add_argument('--seed',          type=int, default=1)
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


if not os.path.isdir(args.output_dir):
    os.mkdir(args.output_dir)

# if args.amp:
#     mixed_precision.enable_mixed_precision()


_ = random.seed(args.seed)
_ = np.random.seed(args.seed + 1)
_ = torch.manual_seed(args.seed + 2)
_ = torch.cuda.manual_seed(args.seed + 3)

dataset      = get_dataset('BEN')
encoder_size = get_encoder_size(dataset)

# get a helper object for tensorboard logging
log_dir      = os.path.join(args.output_dir, args.run_name)
stat_tracker = StatTracker(log_dir=log_dir)

# get dataloaders for training and testing
train_loader, test_loader, num_classes = build_dataset(
    dataset=dataset,
    batch_size=args.batch_size,
    input_dir=args.input_dir,
    labeled_only=args.classifiers,
    num_workers=8
)

device = torch.device('cuda')
checkpointer = Checkpointer(args.output_dir)

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
checkpointer.track_new_model(model)

model = model.to(device)

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


torch.cuda.empty_cache()

next_epoch, total_updates = checkpointer.get_current_position()
fast_stats = AverageMeterSet()

for epoch in range(epochs):
    epoch_stats   = AverageMeterSet()
    epoch_updates = 0
    time_start    = time()
    
    gen = tqdm(train_loader, total=len(train_loader))
    for (images1, images2), labels in gen:
        
        labels  = torch.cat([labels, labels]).to(device)
        images1 = images1.to(device)
        images2 = images2.to(device)
        
        res_dict = model(x1=images1, x2=images2, class_only=False)
        
        lgt_glb_mlp, lgt_glb_lin = res_dict['class']
        
        loss_g2l = (res_dict['g2l_1t5'] + res_dict['g2l_1t7'] + res_dict['g2l_5t5'])
        loss_inf = loss_g2l + res_dict['lgt_reg']
        loss_cls = (F.binary_cross_entropy_with_logits(lgt_glb_mlp, labels) + F.binary_cross_entropy_with_logits(lgt_glb_lin, labels))
        
        # do hacky learning rate warmup -- we stop when LR hits lr_real
        if (total_updates < 500):
            lr_scale = min(1., float(total_updates + 1) / 500.)
            for pg in optim_raw.param_groups:
                pg['lr'] = lr_scale * lr_real
                
        # reset gradient accumlators and do backprop
        loss_opt = loss_inf + loss_cls
        optim_inf.zero_grad()
        mixed_precision.backward(loss_opt, optim_inf)  # backwards with fp32/fp16 awareness
        optim_inf.step()
        
        # record loss and accuracy on minibatch
        epoch_stats.update_dict({
            'loss_inf'     : loss_inf.item(),
            'loss_cls'     : loss_cls.item(),
            'loss_g2l'     : loss_g2l.item(),
            'lgt_reg'      : res_dict['lgt_reg'].item(),
            'loss_g2l_1t5' : res_dict['g2l_1t5'].item(),
            'loss_g2l_1t7' : res_dict['g2l_1t7'].item(),
            'loss_g2l_5t5' : res_dict['g2l_5t5'].item()
        }, n=1)
        
        mlp_auc, lin_auc = update_train_accuracies_multilabel(epoch_stats, labels, lgt_glb_mlp, lgt_glb_lin)
        
        gen.set_postfix(
            loss_inf=float(loss_inf),
            mlp_auc=float(mlp_auc),
            lin_auc=float(lin_auc),
        )
        
        # shortcut diagnostics to deal with long epochs
        total_updates += 1
        epoch_updates += 1
        if (total_updates % 100) == 0:
            torch.cuda.empty_cache()
            time_stop = time()
            spu = (time_stop - time_start) / 100.
            print('Epoch {0:d}, {1:d} updates -- {2:.4f} sec/update'.format(epoch, epoch_updates, spu))
            
            time_start = time()
        
        if (total_updates % 500) == 0:
            # record diagnostics
            eval_start = time()
            fast_stats = AverageMeterSet()
            test_model(model, test_loader, device, fast_stats, max_evals=100000)
            stat_tracker.record_stats(fast_stats.averages(total_updates, prefix='fast/'))
            
            eval_time = time() - eval_start
            stat_str = fast_stats.pretty_string(ignore=model.tasks)
            stat_str = '-- {0:d} updates, eval_time {1:.2f}: {2:s}'.format(
                total_updates, eval_time, stat_str)
            print(stat_str)
            
    # update learning rate
    scheduler_inf.step(epoch)
    test_model(model, test_loader, device, epoch_stats, max_evals=500000)
    epoch_str = epoch_stats.pretty_string(ignore=model.tasks)
    diag_str = '{0:d}: {1:s}'.format(epoch, epoch_str)
    print(diag_str)
    sys.stdout.flush()
    stat_tracker.record_stats(epoch_stats.averages(epoch, prefix='costs/'))
    checkpointer.update(epoch + 1, total_updates)



