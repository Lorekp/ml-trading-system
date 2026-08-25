"""Validate BERT pre-training quality: is it learning useful representations?

4 tests:
1. Hard token accuracy — trivial (copy neighbor) vs real predictions
2. RankMe — effective rank of embedding space
3. Cosine similarity — embedding collapse detection
4. Linear probe — trained vs random encoder on direction prediction

Usage:
    PYTHONPATH=src PYTHONUNBUFFERED=1 python experiments/validate_bert_quality.py
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.bbo_bert.model import BBOBert
from models.bbo_bert.config import BBOBertConfig
from models.bbo_bert.tokenizer import tokenize_bbo_updates

PARQUET_DIR = Path("data/bookTicker_parquet/ETHUSDT")
CKPT_PATH = Path("data/bbo_bert_eth_ft/finetuned.pt")
# Also test the 25-asset model if available
CKPT_25 = Path("data/bbo_bert_25asset/epoch1.pt")

N_DATES = 10  # Use 10 dates for validation (fast)
BATCH_SIZE = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model(config, ckpt_path=None, device="cuda"):
    """Load model. If ckpt_path is None, returns random-init model."""
    model = BBOBert(config).to(device)
    if ckpt_path is not None:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def prepare_data(config, n_dates=10):
    """Load and tokenize n_dates of ETH data, create masked sequences."""
    dates = sorted([f.stem for f in PARQUET_DIR.glob("*.parquet")])[:n_dates]
    seq_len = config.seq_len

    all_tokens, all_price, all_vol, all_time, all_ts = [], [], [], [], []

    for date in dates:
        df = pd.read_parquet(PARQUET_DIR / f"{date}.parquet")
        tok = tokenize_bbo_updates(df, config)
        n = len(tok["tokens"])
        n_seqs = (n - seq_len) // seq_len
        if n_seqs < 1:
            continue

        for i in range(min(n_seqs, 500)):  # Cap per date
            s = i * seq_len
            all_tokens.append(tok["tokens"][s:s+seq_len])
            all_price.append(tok["price_cont"][s:s+seq_len])
            all_vol.append(tok["vol_cont"][s:s+seq_len])
            all_time.append(tok["time_cont"][s:s+seq_len])
            all_ts.append(tok["timestamps_ms"][s:s+seq_len])

    tokens = np.stack(all_tokens)
    price = np.stack(all_price)
    vol = np.stack(all_vol)
    time_c = np.stack(all_time)
    ts = np.stack(all_ts)

    print(f"  Prepared {len(tokens)} sequences from {n_dates} dates", flush=True)
    return tokens, price, vol, time_c, ts


def test_hard_token_accuracy(model, config, tokens, price, vol, time_c, device):
    """Test 1: Split accuracy into trivial (copy previous) vs hard (transitions)."""
    print("\n=== Test 1: Hard Token Accuracy ===", flush=True)

    seq_len = config.seq_len
    mask_prob = 0.15
    rng = np.random.RandomState(42)

    total_trivial_correct = 0
    total_trivial = 0
    total_hard_correct = 0
    total_hard = 0
    total_correct = 0
    total_masked = 0

    n_seqs = len(tokens)
    for i in range(0, n_seqs, BATCH_SIZE):
        j = min(i + BATCH_SIZE, n_seqs)
        b_tokens = tokens[i:j].copy()
        b_price = price[i:j]
        b_vol = vol[i:j]
        b_time = time_c[i:j]

        # Create mask (skip position 0)
        mask = rng.random(b_tokens.shape) < mask_prob
        mask[:, 0] = False
        original = b_tokens.copy()

        # Apply mask
        b_tokens[mask] = 0  # MASK token = 0

        with torch.no_grad():
            out = model(
                torch.from_numpy(b_tokens).long().to(device),
                torch.from_numpy(b_price).float().to(device),
                torch.from_numpy(b_vol).float().to(device),
                torch.from_numpy(b_time).float().to(device),
            )
            preds = out["token_logits"].argmax(dim=-1).cpu().numpy()

        # Classify masked positions
        for bi in range(j - i):
            masked_pos = np.where(mask[bi])[0]
            for pos in masked_pos:
                correct = (preds[bi, pos] == original[bi, pos])
                prev_token = original[bi, pos - 1] if pos > 0 else -1
                is_trivial = (original[bi, pos] == prev_token)

                total_masked += 1
                total_correct += int(correct)

                if is_trivial:
                    total_trivial += 1
                    total_trivial_correct += int(correct)
                else:
                    total_hard += 1
                    total_hard_correct += int(correct)

    overall_acc = total_correct / max(total_masked, 1)
    trivial_acc = total_trivial_correct / max(total_trivial, 1)
    hard_acc = total_hard_correct / max(total_hard, 1)
    trivial_frac = total_trivial / max(total_masked, 1)

    print(f"  Overall accuracy: {overall_acc:.1%} ({total_masked:,} masked tokens)")
    print(f"  Trivial tokens (same as prev): {trivial_frac:.1%} of masked positions")
    print(f"  Trivial accuracy: {trivial_acc:.1%} ({total_trivial:,} tokens)")
    print(f"  Hard accuracy:    {hard_acc:.1%} ({total_hard:,} tokens)")

    # Baseline: always copy previous token
    copy_baseline = trivial_frac  # copying gets all trivial right, all hard wrong
    print(f"  Copy-previous baseline: {copy_baseline:.1%}")
    print(f"  SIGNAL = hard_acc - random_guess: {hard_acc:.1%} - {1/config.vocab_size:.1%} = {hard_acc - 1/config.vocab_size:.1%}")

    return {"overall": overall_acc, "trivial": trivial_acc, "hard": hard_acc,
            "trivial_frac": trivial_frac}


def test_rankme(model, config, tokens, price, vol, time_c, device, label=""):
    """Test 2: Effective rank of embeddings (RankMe)."""
    n_samples = min(len(tokens), 2000)

    all_emb = []
    for i in range(0, n_samples, BATCH_SIZE):
        j = min(i + BATCH_SIZE, n_samples)
        with torch.no_grad():
            out = model(
                torch.from_numpy(tokens[i:j]).long().to(device),
                torch.from_numpy(price[i:j]).float().to(device),
                torch.from_numpy(vol[i:j]).float().to(device),
                torch.from_numpy(time_c[i:j]).float().to(device),
            )
            # Mean pool across sequence
            emb = out["hidden"].mean(dim=1).cpu().numpy()  # (B, d_model)
        all_emb.append(emb)

    embeddings = np.concatenate(all_emb)  # (n_samples, d_model)

    # RankMe: Shannon entropy of normalized singular values
    _, S, _ = np.linalg.svd(embeddings, full_matrices=False)
    p = S / S.sum()
    p = p[p > 1e-10]  # avoid log(0)
    rankme = np.exp(-np.sum(p * np.log(p)))

    print(f"  RankMe{label}: {rankme:.1f} (max={embeddings.shape[1]})")
    return rankme


def test_cosine_similarity(model, config, tokens, price, vol, time_c, device, label=""):
    """Test 3: Average cosine similarity between random embedding pairs."""
    n_samples = min(len(tokens), 1000)

    all_emb = []
    for i in range(0, n_samples, BATCH_SIZE):
        j = min(i + BATCH_SIZE, n_samples)
        with torch.no_grad():
            out = model(
                torch.from_numpy(tokens[i:j]).long().to(device),
                torch.from_numpy(price[i:j]).float().to(device),
                torch.from_numpy(vol[i:j]).float().to(device),
                torch.from_numpy(time_c[i:j]).float().to(device),
            )
            emb = out["hidden"].mean(dim=1).cpu().numpy()
        all_emb.append(emb)

    embeddings = np.concatenate(all_emb)

    # Random 1000 pairs
    rng = np.random.RandomState(42)
    n = len(embeddings)
    idx_a = rng.randint(0, n, 1000)
    idx_b = rng.randint(0, n, 1000)
    # Ensure different samples
    idx_b = np.where(idx_a == idx_b, (idx_b + 1) % n, idx_b)

    a = embeddings[idx_a]
    b = embeddings[idx_b]
    cos_sim = np.sum(a * b, axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-10)
    mean_cos = cos_sim.mean()

    print(f"  Avg cosine sim{label}: {mean_cos:.4f} (0=isotropic, 1=collapsed)")
    return mean_cos


def test_linear_probe(model, config, tokens, price, vol, time_c, ts, device, label=""):
    """Test 4: Linear probe — can embeddings predict FUTURE direction?

    Label: did the mid-price go up in the NEXT sequence (not within this one)?
    Uses the timestamp of the last token to find the next sequence's price.
    """
    n_samples = min(len(tokens), 3000)

    all_emb = []
    for i in range(0, n_samples, BATCH_SIZE):
        j = min(i + BATCH_SIZE, n_samples)
        with torch.no_grad():
            out = model(
                torch.from_numpy(tokens[i:j]).long().to(device),
                torch.from_numpy(price[i:j]).float().to(device),
                torch.from_numpy(vol[i:j]).float().to(device),
                torch.from_numpy(time_c[i:j]).float().to(device),
            )
            emb = out["hidden"].mean(dim=1).cpu().numpy()
        all_emb.append(emb)

    embeddings = np.concatenate(all_emb)[:n_samples]

    # Label: compare price_cont at end of NEXT sequence vs end of THIS sequence
    # This is a genuine forward-looking label — predicting what happens AFTER
    # the window the model sees
    labels = []
    valid_idx = []
    for i in range(n_samples - 1):
        # price_cont is cumulative PLGS-encoded price change
        # Positive sum = price went up during that sequence
        this_end = price[i, -1]
        next_end = price[i + 1, -1]
        next_start = price[i + 1, 0]
        # Did price go up during the NEXT sequence?
        direction = 1 if (next_end > next_start) else 0
        labels.append(direction)
        valid_idx.append(i)

    labels = np.array(labels)
    embeddings = embeddings[valid_idx]

    # Check label balance
    balance = labels.mean()
    print(f"  Label balance{label}: {balance:.1%} up / {1-balance:.1%} down", flush=True)

    # Temporal train/test split (80/20 — no shuffle, preserve time order)
    n = len(embeddings)
    split = int(n * 0.8)
    X_train, X_test = embeddings[:split], embeddings[split:]
    y_train, y_test = labels[:split], labels[split:]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    clf = LogisticRegression(max_iter=500, C=1.0, random_state=42)
    clf.fit(X_train_s, y_train)
    acc = clf.score(X_test_s, y_test)

    print(f"  Linear probe accuracy{label}: {acc:.1%} (50% = random)")
    return acc


def main():
    t_start = time.time()
    config = BBOBertConfig()
    print(f"Device: {DEVICE}")
    print(f"Config: d_model={config.d_model}, vocab={config.vocab_size}, seq_len={config.seq_len}")

    # Prepare data
    print("\nPreparing validation data...")
    tokens, price, vol, time_c, ts = prepare_data(config, N_DATES)

    # ============================================================
    # ETH fine-tuned model (844K params)
    # ============================================================
    print("\n" + "=" * 60)
    print("  MODEL A: ETH fine-tuned (844K params)")
    print("=" * 60)

    model_trained = load_model(config, CKPT_PATH, DEVICE)
    print(f"  Loaded: {model_trained.count_parameters():,} params")

    # Test 1: Hard token accuracy
    hard_results = test_hard_token_accuracy(model_trained, config, tokens, price, vol, time_c, DEVICE)

    # Test 2: RankMe
    print("\n=== Test 2: RankMe ===", flush=True)
    rankme_trained = test_rankme(model_trained, config, tokens, price, vol, time_c, DEVICE, " [trained]")

    # Test 3: Cosine similarity
    print("\n=== Test 3: Cosine Similarity ===", flush=True)
    cos_trained = test_cosine_similarity(model_trained, config, tokens, price, vol, time_c, DEVICE, " [trained]")

    # Test 4: Linear probe
    print("\n=== Test 4: Linear Probe ===", flush=True)
    probe_trained = test_linear_probe(model_trained, config, tokens, price, vol, time_c, ts, DEVICE, " [trained]")

    del model_trained
    torch.cuda.empty_cache()

    # ============================================================
    # Random baseline (same architecture, untrained)
    # ============================================================
    print("\n" + "=" * 60)
    print("  MODEL B: Random initialization (untrained)")
    print("=" * 60)

    model_random = load_model(config, None, DEVICE)

    print("\n=== Test 2: RankMe ===", flush=True)
    rankme_random = test_rankme(model_random, config, tokens, price, vol, time_c, DEVICE, " [random]")

    print("\n=== Test 3: Cosine Similarity ===", flush=True)
    cos_random = test_cosine_similarity(model_random, config, tokens, price, vol, time_c, DEVICE, " [random]")

    print("\n=== Test 4: Linear Probe ===", flush=True)
    probe_random = test_linear_probe(model_random, config, tokens, price, vol, time_c, ts, DEVICE, " [random]")

    del model_random
    torch.cuda.empty_cache()

    # ============================================================
    # 25-asset model (if available)
    # ============================================================
    if CKPT_25.exists():
        print("\n" + "=" * 60)
        print("  MODEL C: 25-asset pretrained (4.9M params, epoch 1)")
        print("=" * 60)

        # 25-asset model has d_model=256 — need different config
        config_25 = BBOBertConfig()
        config_25.d_model = 256
        config_25.n_heads = 8
        config_25.n_layers = 6
        config_25.d_ff = 1024

        try:
            model_25 = load_model(config_25, CKPT_25, DEVICE)
            print(f"  Loaded: {model_25.count_parameters():,} params")

            print("\n=== Test 2: RankMe ===", flush=True)
            rankme_25 = test_rankme(model_25, config_25, tokens, price, vol, time_c, DEVICE, " [25-asset]")

            print("\n=== Test 3: Cosine Similarity ===", flush=True)
            cos_25 = test_cosine_similarity(model_25, config_25, tokens, price, vol, time_c, DEVICE, " [25-asset]")

            print("\n=== Test 4: Linear Probe ===", flush=True)
            probe_25 = test_linear_probe(model_25, config_25, tokens, price, vol, time_c, ts, DEVICE, " [25-asset]")

            # Random baseline for 25-asset arch
            print("\n  -- 25-asset random baseline --", flush=True)
            model_25_rand = load_model(config_25, None, DEVICE)
            rankme_25_rand = test_rankme(model_25_rand, config_25, tokens, price, vol, time_c, DEVICE, " [25-asset random]")
            probe_25_rand = test_linear_probe(model_25_rand, config_25, tokens, price, vol, time_c, ts, DEVICE, " [25-asset random]")
            del model_25, model_25_rand
        except Exception as e:
            print(f"  Failed to load 25-asset model: {e}")

    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "=" * 60)
    print("  VALIDATION SUMMARY")
    print("=" * 60)

    print(f"\n  Hard Token Analysis (ETH model):")
    print(f"    Trivial tokens: {hard_results['trivial_frac']:.0%} of all masked")
    print(f"    Trivial accuracy: {hard_results['trivial']:.1%}")
    print(f"    HARD accuracy: {hard_results['hard']:.1%} (random={1/config.vocab_size:.1%})")

    print(f"\n  RankMe (higher=better, max={config.d_model}):")
    print(f"    Trained: {rankme_trained:.1f}")
    print(f"    Random:  {rankme_random:.1f}")

    print(f"\n  Cosine Similarity (lower=better):")
    print(f"    Trained: {cos_trained:.4f}")
    print(f"    Random:  {cos_random:.4f}")

    print(f"\n  Linear Probe (50%=chance):")
    print(f"    Trained: {probe_trained:.1%}")
    print(f"    Random:  {probe_random:.1%}")
    print(f"    SIGNAL:  {probe_trained - probe_random:+.1%}")

    print(f"\n  Decision:")
    signal = probe_trained - probe_random
    if signal < 0.03:
        print(f"    Probe gap {signal:.1%} < 3%: Training NOT adding useful representations")
        print(f"    RECOMMENDATION: Stop Vast training, switch to span masking")
    elif hard_results['hard'] < 2 / config.vocab_size:
        print(f"    Hard token acc near random: masking strategy is broken")
        print(f"    RECOMMENDATION: Switch to span masking before continuing")
    else:
        print(f"    Probe gap {signal:.1%}: Training IS adding value")
        print(f"    Continue training, validate again after epoch 2")

    elapsed = time.time() - t_start
    print(f"\nDone in {elapsed:.0f}s")


if __name__ == "__main__":
    main()
