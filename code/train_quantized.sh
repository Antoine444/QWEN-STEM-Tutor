#!/bin/bash

# QLoRA Fine-tuning Setup and Launch Script

set -e  # Exit on any error

echo "=== QLoRA Fine-tuning Setup ==="

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python3 is not installed or not in PATH"
    exit 1
fi

# Check if pip is available
if ! command -v pip3 &> /dev/null; then
    echo "Error: pip3 is not installed or not in PATH"
    exit 1
fi

# Check CUDA availability
echo "Checking CUDA availability..."
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA devices: {torch.cuda.device_count()}')"

# Create virtual environment (optional but recommended)
echo "Creating virtual environment..."
python3 -m venv qlora_env
source qlora_env/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch with CUDA support (adjust version as needed)
echo "Installing PyTorch with CUDA support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install requirements
echo "Installing requirements..."
pip install -r train_quantized/requirements.txt

# Verify installations
echo "Verifying installations..."
python3 -c "
import torch
import transformers
import peft
import datasets
import trl
import bitsandbytes
print('All packages installed successfully!')
print(f'PyTorch version: {torch.__version__}')
print(f'Transformers version: {transformers.__version__}')
print(f'PEFT version: {peft.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
"

echo "=== Setup Complete ==="
echo ""
echo "=== Starting QLoRA Training ==="

# Default parameters (you can modify these)
LEARNING_RATE=${1:-5e-4}
BATCH_SIZE=${2:-8}
EPOCHS=${3:-1}
MAX_LENGTH=${4:-512}

echo "Training parameters:"
echo "  Learning rate: $LEARNING_RATE"
echo "  Batch size: $BATCH_SIZE"
echo "  Epochs: $EPOCHS"
echo "  Max length: $MAX_LENGTH"
echo ""

# Launch training
chmod +x train_quantized.sh
python3 train_quantized/train_quantized.py \
    --learning_rate $LEARNING_RATE \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --max_length $MAX_LENGTH

echo "=== Training Complete ==="
