"""BERT embedding extraction and aggregation to 15m bars.

Extracts hidden states from a fine-tuned BBO-BERT model and aggregates
tick-level embeddings to 15m bar level. No PCA or scaling here — those
must happen inside the walk-forward loop to prevent leakage.

Memory-efficient: aggregates per-date to bar level during extraction,
never holds all 7.8M raw embeddings in memory simultaneously.
"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

log = logging.getLogger(__name__)


def extract_bar_features(bert_model, bert_config, parquet_dir, dates,
                         bar_index, device="cuda", batch_size=512,
                         cache_path=None):
    """Extract BERT embeddings and aggregate to 15m bars in one pass.

    Returns (bar_mean, bar_stats):
        bar_mean: DataFrame (n_bars, d_model) — mean-pooled per bar
        bar_stats: DataFrame (n_bars, 3*d_model) — mean/std/slope per bar

    Memory-efficient: processes one date at a time, accumulates bar-level
    sums/counts, never holds all raw embeddings.
    """
    from models.bbo_bert.tokenizer import tokenize_bbo_updates

    d_model = bert_config.d_model
    seq_len = bert_config.seq_len
    bert_cols = [f"bert_{j}" for j in range(d_model)]

    # Check cache
    if cache_path is not None:
        cache_path = Path(cache_path)
        if cache_path.exists():
            data = np.load(cache_path, allow_pickle=False)
            bar_mean = pd.DataFrame(data["bar_mean"], index=bar_index, columns=bert_cols)
            stat_cols = ([f"bert_mean_{j}" for j in range(d_model)] +
                         [f"bert_std_{j}" for j in range(d_model)] +
                         [f"bert_slope_{j}" for j in range(d_model)])
            bar_stats = pd.DataFrame(data["bar_stats"], index=bar_index, columns=stat_cols)
            print(f"  Loaded cached bar features from {cache_path}", flush=True)
            return bar_mean, bar_stats

    bert_model.eval()
    parquet_dir = Path(parquet_dir)

    # Accumulators: per-bar sum, count, sum_sq, first, last
    n_bars = len(bar_index)
    bar_sum = np.zeros((n_bars, d_model), dtype=np.float64)
    bar_sum_sq = np.zeros((n_bars, d_model), dtype=np.float64)
    bar_count = np.zeros(n_bars, dtype=np.int64)
    bar_first = np.full((n_bars, d_model), np.nan, dtype=np.float64)
    bar_last = np.full((n_bars, d_model), np.nan, dtype=np.float64)

    # Map bar_index to integer positions for fast lookup
    bar_ts = bar_index.values.astype(np.int64)  # nanoseconds
    bar_freq_ns = int(pd.Timedelta("15min").total_seconds() * 1e9)

    total_seqs = 0

    for di, date in enumerate(dates):
        pq_path = parquet_dir / f"{date}.parquet"
        if not pq_path.exists():
            continue

        try:
            df = pd.read_parquet(pq_path)
            if len(df) < 1000:
                continue

            tok = tokenize_bbo_updates(df, bert_config)
            n = len(tok["tokens"])
            n_seqs = (n - seq_len) // seq_len
            if n_seqs < 1:
                continue

            # Vectorized windows
            tok_win = np.lib.stride_tricks.sliding_window_view(
                tok["tokens"], seq_len)[::seq_len][:n_seqs].copy()
            price_win = np.lib.stride_tricks.sliding_window_view(
                tok["price_cont"], seq_len)[::seq_len][:n_seqs].copy()
            vol_win = np.lib.stride_tricks.sliding_window_view(
                tok["vol_cont"], seq_len)[::seq_len][:n_seqs].copy()
            time_win = np.lib.stride_tricks.sliding_window_view(
                tok["time_cont"], seq_len)[::seq_len][:n_seqs].copy()

            # Midpoint timestamps (ms)
            ts_mid_ms = np.array([tok["timestamps_ms"][i * seq_len + seq_len // 2]
                                  for i in range(n_seqs)], dtype=np.int64)

            # GPU inference
            day_hidden = []
            for i in range(0, n_seqs, batch_size):
                j = min(i + batch_size, n_seqs)
                with torch.no_grad():
                    out = bert_model(
                        torch.from_numpy(tok_win[i:j]).long().to(device),
                        torch.from_numpy(price_win[i:j]).float().to(device),
                        torch.from_numpy(vol_win[i:j]).float().to(device),
                        torch.from_numpy(time_win[i:j]).float().to(device),
                    )
                    day_hidden.append(out["hidden"].mean(dim=1).cpu().numpy())

            hidden = np.concatenate(day_hidden)  # (n_seqs, d_model)
            total_seqs += n_seqs

            # Assign to bars: floor ts_mid to 15min bar
            ts_mid_ns = ts_mid_ms * 1_000_000  # ms → ns
            bar_floor_ns = (ts_mid_ns // bar_freq_ns) * bar_freq_ns

            # Find bar indices via searchsorted
            bar_positions = np.searchsorted(bar_ts, bar_floor_ns)
            # Clamp and validate
            valid = (bar_positions < n_bars) & (bar_ts[np.clip(bar_positions, 0, n_bars-1)] == bar_floor_ns)

            # Vectorized bar accumulation (no Python loop)
            v_idx = np.where(valid)[0]
            v_bars = bar_positions[v_idx]
            v_hidden = hidden[v_idx]

            np.add.at(bar_sum, v_bars, v_hidden)
            np.add.at(bar_sum_sq, v_bars, v_hidden * v_hidden)
            np.add.at(bar_count, v_bars, 1)

            # First/last per bar: use sorted order (sequences are chronological)
            for bi in np.unique(v_bars):
                mask_bi = v_bars == bi
                bi_hidden = v_hidden[mask_bi]
                if np.isnan(bar_first[bi, 0]):
                    bar_first[bi] = bi_hidden[0]
                bar_last[bi] = bi_hidden[-1]

        except (MemoryError, KeyboardInterrupt):
            raise
        except Exception as e:
            log.warning("Skip %s: %s", date, e)

        if (di + 1) % 10 == 0:
            print(f"  Extracted {di+1}/{len(dates)} dates ({total_seqs:,} sequences)",
                  flush=True)

    print(f"  Total: {total_seqs:,} sequences -> {(bar_count > 0).sum():,} bars with data",
          flush=True)

    # Compute bar-level features
    safe_count = np.maximum(bar_count, 1)[:, None]

    # Mean
    mean_arr = bar_sum / safe_count

    # Std: sqrt(E[x^2] - E[x]^2)
    var_arr = bar_sum_sq / safe_count - mean_arr ** 2
    std_arr = np.sqrt(np.maximum(var_arr, 0))

    # Slope: (last - first) / count
    slope_arr = np.where(
        bar_count[:, None] > 1,
        (bar_last - bar_first) / safe_count,
        0.0
    )

    # Zero out bars with no data
    no_data = bar_count == 0
    mean_arr[no_data] = 0
    std_arr[no_data] = 0
    slope_arr[no_data] = 0

    # Build DataFrames
    bar_mean = pd.DataFrame(mean_arr.astype(np.float32), index=bar_index, columns=bert_cols)

    stat_cols = ([f"bert_mean_{j}" for j in range(d_model)] +
                 [f"bert_std_{j}" for j in range(d_model)] +
                 [f"bert_slope_{j}" for j in range(d_model)])
    stats_arr = np.hstack([mean_arr, std_arr, slope_arr]).astype(np.float32)
    bar_stats = pd.DataFrame(stats_arr, index=bar_index, columns=stat_cols)

    # Cache
    if cache_path is not None:
        np.savez_compressed(cache_path,
                            bar_mean=bar_mean.values,
                            bar_stats=bar_stats.values)
        size_mb = cache_path.stat().st_size / 1e6
        print(f"  Cached to {cache_path} ({size_mb:.1f} MB)", flush=True)

    return bar_mean, bar_stats
