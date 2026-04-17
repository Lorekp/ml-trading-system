#!/bin/bash
echo "============================================"
echo " Kronos Fine-Tuning Setup (Linux/Mac)"
echo "============================================"
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "ERROR: python3 not found. Install Python 3.11+:"
    echo "  Ubuntu/Debian: sudo apt install python3 python3-venv python3-pip"
    echo "  Mac: brew install python3"
    exit 1
fi

echo "[1/4] Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

echo "[2/4] Installing PyTorch with CUDA..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128 --quiet 2>/dev/null
if [ $? -ne 0 ]; then
    echo "  CUDA install failed, trying default PyTorch..."
    pip install torch torchvision --quiet
fi

echo "[3/4] Installing other dependencies..."
pip install numpy pandas einops huggingface_hub tqdm safetensors matplotlib pyyaml --quiet

echo ""
echo "[4/4] Starting training..."
echo "============================================"
echo " Training will take approximately 15-20 hours."
echo " You can safely close this terminal if using tmux/screen."
echo " DO NOT turn off your PC until it finishes."
echo "============================================"
echo ""

cd kronos/finetune_csv
python3 train_sequential.py --config ../../config_3090.yaml --skip-tokenizer

echo ""
echo "============================================"
if [ $? -eq 0 ]; then
    echo " Training COMPLETE!"
    echo " Please send the entire 'output' folder back."
    echo " Location: $(cd ../../output && pwd)"
else
    echo " Training FAILED. Please send a screenshot of the error."
fi
echo "============================================"
