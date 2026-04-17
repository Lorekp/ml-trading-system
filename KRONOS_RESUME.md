# Kronos Fine-Tuning — Resume Instructions

## Status: TOKENIZER COMPLETE, PREDICTOR NOT STARTED

Tokenizer training was stopped at epoch 14/20 but the best model was saved
at epoch ~5 (validation loss 0.0004). No improvement in epochs 5-14, so
no work was lost.

## Saved Checkpoint
```
C:\Claude\ml-trading-system\data\kronos_finetune\finetuned\crypto_15m_10assets\tokenizer\best_model\
├── config.json (318 bytes)
├── model.safetensors (15.8 MB)
└── README.md
```

## To Resume — Train PREDICTOR Only
```bash
cd C:\Claude\ml-trading-system
source .venv/Scripts/activate
cd vendor/kronos/finetune_csv
python train_sequential.py --config "C:/Claude/ml-trading-system/data/kronos_finetune/config_crypto_15m.yaml" --skip-tokenizer
```

This will:
1. Skip tokenizer training (already done)
2. Load the saved tokenizer from best_model/
3. Train the predictor (Kronos-base, 102.3M params) for 15 epochs
4. Estimated time: ~5-8 hours on RTX 4070 Super at batch_size=32

## Config
```
C:\Claude\ml-trading-system\data\kronos_finetune\config_crypto_15m.yaml
```
- Data: 350K bars, 10 assets, 15m timeframe
- Batch size: 32 (uses ~6.8GB of 12GB VRAM)
- Predictor epochs: 15
- Predictor LR: 0.000001

## After Predictor Training
The fine-tuned model will be at:
```
data/kronos_finetune/finetuned/crypto_15m_10assets/basemodel/best_model/
```

Use it with:
```python
from model import Kronos, KronosTokenizer, KronosPredictor

tokenizer = KronosTokenizer.from_pretrained("data/kronos_finetune/finetuned/crypto_15m_10assets/tokenizer/best_model")
model = Kronos.from_pretrained("data/kronos_finetune/finetuned/crypto_15m_10assets/basemodel/best_model")
predictor = KronosPredictor(model, tokenizer, device="cuda:0", max_context=512)
```
