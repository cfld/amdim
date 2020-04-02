#!/usr/bin/env python

"""

"""

import os
import random
import argparse
import numpy as np

import torch

import mixed_precision
from stats import StatTracker
from datasets import Dataset, build_dataset, get_dataset, get_encoder_size
from model import Model
from checkpoint import Checkpointer
from task_self_supervised import train_self_supervised
from task_classifiers import train_classifiers

def parse_args():
    parser = argparse.ArgumentParser()
    
    # parameters for general training stuff
    parser.add_argument('--batch_size',    type=int, default=200)
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

if args.amp:
    mixed_precision.enable_mixed_precision()


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
    input_dir=None,
    labeled_only=args.classifiers,
    num_workers=8
)

z = next(iter(train_loader))


torch_device = torch.device('cuda')
checkpointer = Checkpointer(args.output_dir)

if args.cpt_load_path:
    raise Exception()
    # model = checkpointer.restore_model_from_checkpoint(args.cpt_load_path, training_classifier=args.classifiers)
else:
    # create new model with random parameters
    model = Model(
        ndf          = args.ndf, 
        n_classes    = num_classes, 
        n_rkhs       = args.n_rkhs,
        tclip        = args.tclip, 
        n_depth      = args.n_depth, 
        encoder_size = encoder_size,
        use_bn       = (args.use_bn == 1)
    )
    
    model.init_weights(init_scale=1.0)
    checkpointer.track_new_model(model)

model = model.to(torch_device)

# select which type of training to do
task = train_classifiers if args.classifiers else train_self_supervised
task(model, args.learning_rate, dataset, train_loader,
     test_loader, stat_tracker, checkpointer, args.output_dir, torch_device)