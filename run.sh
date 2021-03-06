#!/bin/bash

# run.sh

# --
# Setup environment

conda create -y --name amdim_env python=3.7
conda activate amdim_env

# conda activate pytorch_p36

pip install --upgrade pip
pip install --upgrade numpy
pip install --upgrade scikit-image
pip install --upgrade scikit-learn
pip install tifffile
pip install tqdm
pip install albumentations
pip install git+https://github.com/bkj/rsub

conda install -y -c pytorch pytorch=1.4.0

pip install -r requirements.txt

# --

CUDA_VISIBLE_DEVICES=3,4,5,6 python amdim/ben_train.py \
    --batch_size    256    \
    --learning_rate 0.0002 \
    --tclip         100    \
    --run_name orig4_t100 | tee orig4_t100.jl