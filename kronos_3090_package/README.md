# Kronos Fine-Tuning Package — RTX 3090 (24GB)

## What This Is
Fine-tunes the Kronos time-series foundation model (102.3M params) on crypto 15-minute OHLCV data.
The tokenizer is already fine-tuned (included). This trains the PREDICTOR only.

## Setup

### 1. Install Python 3.11+ and create venv
```bash
python -m venv kronos-venv
source kronos-venv/bin/activate  # Linux/Mac
# OR: kronos-venv\Scripts\activate  # Windows
```

### 2. Install PyTorch with CUDA
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

### 3. Install dependencies
```bash
pip install numpy pandas einops huggingface_hub tqdm safetensors matplotlib pyyaml
```

### 4. Run training
```bash
cd kronos/finetune_csv
python train_sequential.py --config ../../config_3090.yaml --skip-tokenizer
```

The `--skip-tokenizer` flag skips tokenizer training (already done, saved in `tokenizer_checkpoint/`).

## Expected Behavior
- Downloads Kronos-base (102.3M params) from HuggingFace on first run
- Uses ~8-10 GB VRAM at batch_size=64 (3090 has 24GB — plenty of headroom)
- Trains for 15 epochs on 350K bars (10 crypto assets, 15-minute timeframe)
- Logs loss every 100 steps
- Saves best model checkpoint to `output/basemodel/best_model/`

## Estimated Time
- RTX 3090 at batch_size=64: ~15-20 hours
- If too slow, reduce basemodel_epochs to 5 in config_3090.yaml (first 3-5 epochs capture most improvement)

## After Training
Send back the entire `output/` directory. It will contain:
```
output/
├── basemodel/
│   └── best_model/
│       ├── config.json
│       └── model.safetensors  (~400MB)
└── logs/
```

## What's in the Training Data
- 10 crypto assets: BTC, ETH, SOL, XRP, DOGE, BNB, LINK, AVAX, SUI, ADA
- 15-minute OHLCV bars from March 2024 to February 2026
- ~350,000 bars total
- Format: CSV with columns [symbol, timestamp, open, high, low, close, volume]

## Config Notes (config_3090.yaml)
- batch_size=64 (doubled from 32 — 3090 has 2x the VRAM)
- num_workers=8 (adjust if CPU limited)
- Everything else same as original config
