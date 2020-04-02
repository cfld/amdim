#!/bin/bash

# run.sh

# --
# Setup environment

conda activate pytorch_p36

pip install --upgrade pip
pip install --upgrade numpy
pip install --upgrade scikit-image
pip install git+https://github.com/bkj/rsub

pip install tifffile
pip install tqdm
pip install albumentations

pip install -r requirements.txt
