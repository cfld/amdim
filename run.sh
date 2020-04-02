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
pip install tifffile
pip install tqdm
pip install albumentations
pip install git+https://github.com/bkj/rsub

conda install -y -c pytorch pytorch=1.4.0

pip install -r requirements.txt

# --

