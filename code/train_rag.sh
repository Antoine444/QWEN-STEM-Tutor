#!/bin/bash

# Navigate to the train_rag folder
cd "$(dirname "$0")/train_rag"

# Run the Jupyter notebook
jupyter nbconvert --to notebook --execute pre_processing_data.ipynb --output executed_pre_processing_data.ipynb
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