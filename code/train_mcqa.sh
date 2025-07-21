#!/usr/bin/env bash
set -euo pipefail

# Upgrade pip and install dependencies
pip install --upgrade pip
pip install torchvision==0.21.0+cu126 --extra-index-url https://download.pytorch.org/whl/cu126 \
            transformers==4.51.3 \
            datasets \
            accelerate \
            trl==0.17.0 \
            pandas \
            matplotlib \
            seaborn \
            scikit-learn              

# Run training script
echo "Launching training…"
chmod +x train_mcqa.sh
python train_mcqa/train_mcqa.py
