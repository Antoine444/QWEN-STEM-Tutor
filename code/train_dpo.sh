#!/usr/bin/env bash
set -euo pipefail

# Upgrade pip and install dependencies
pip install --upgrade pip
pip install transformers==4.51.3 \
            torch \
            datasets \
            trl==0.17.0              

# Run training script
echo "Launching training…"
chmod +x train_dpo.sh
python train_dpo/dpo_model.py