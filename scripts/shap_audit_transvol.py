"""TransVol per-regime SHAP audit (production-matched).

Mirrors `shap_audit_optuna.py` (LightGBM) for the TransVol encoder:

1. Loads the existing per-month TransVol checkpoint (production-matched
   walk-forward — these are the checkpoints the cycle-9 LightGBM read).
2. For each of 8 audit months (same set as the LightGBM audit), builds
   the test-month input tensor (sliding window of seq_len=32 bars × 101
   features), z-normalizes using checkpoint-saved normalizer.
3. Computes SHAP values via `shap.GradientExplainer` on the model's
   `continuous_var_raw` output (the QLIKE prediction head — the same
   target the model is trained to optimize).
4. Aggregates per-input-feature SHAP across the time dimension and
   batch, producing a (101,) importance vector per month.
5. Aggregates across months to produce KEEP/REVIEW/DROP decisions
   using the same rules as the LightGBM audit.

Why this matches "200/50 Optuna" methodology:
TransVol PRODUCTION does not use Optuna — its hyperparameters are
fixed in `vol_model/transvol/config.py`. The closest production-match
is therefore SHAP on the production-trained checkpoints (no Optuna
search applies). For the pruned 2nd pass, we retrain TransVol with
the same fixed config but on the reduced input set.

Output:
    vol_model/output/shap_audit_transvol_decision.csv
    vol_model/output/shap_audit_transvol_per_month.csv
    vol_model/output/shap_audit_transvol_summary.md

Usage::

    PYTHONPATH=. python scripts/shap_audit_transvol.py
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402
import torch  # noqa: E402

from vol_model.config import VolModelConfig  # noqa: E402
from vol_model.pipeline import build_dataset  # noqa: E402
from vol_model.transvol.config import TransVolConfig  # noqa: E402
from vol_model.transvol.dataset import TIME_COLS, FeatureNormalizer  # noqa: E402
from vol_model.transvol.model import TransVolEncoder  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("shap_audit_transvol")


# Same 8 audit months as LightGBM Optuna audit.
AUDIT_MONTHS = [
    ("2022-06", "LUNA collapse aftermath + 3AC unwind"),
    ("2022-11", "FTX collapse + contagion"),
    ("2023-08", "mid-2023 range-bound calm"),
    ("2023-12", "late-2023 ETF speculation"),
    ("2024-04", "BTC halving cycle / transition"),
    ("2024-08", "yen-carry unwind"),
    ("2025-04", "recent vol shock"),
    ("2026-01", "recent OOS"),
]


# Production TransVol drops 25 microstructure features.
# Mirror the production drop logic from
# run_walkforward_pruned_svi_har_pd.py:
DROP_PREFIXES = ("ofi_", "depth_mid", "oi_change", "oi_zscore", "oi_acceleration")


def _select_transvol_features(X: pd.DataFrame) -> pd.DataFrame:
    """Apply the production TransVol input drop (101 of 126 features)."""
    cols = [c for c in X.columns if not c.startswith(DROP_PREFIXES)]
    return X[cols]


def _build_month_input_tensor(
    X_tv_full: pd.DataFrame,
    normalizer: FeatureNormalizer,
    seq_len: int,
    month_start: pd.Timestamp,
    month_end: pd.Timestamp,
) -> tuple[torch.Tensor, torch.Tensor, pd.DatetimeIndex]:
    """Build (B, T, F) feature tensor for the test month, plus
    (B, T, 2) time tensor and timestamp index of the LAST bar in
    each window. The last-bar timestamp must be in [month_start,
    month_end)."""
    # We need to pull seq_len-1 bars BEFORE the month start so the
    # earliest test-month bar has a full sequence.
    history_start_idx = X_tv_full.index.searchsorted(month_start) - (seq_len - 1)
    history_start_idx = max(history_start_idx, 0)
    test_end_idx = X_tv_full.index.searchsorted(month_end)
    if test_end_idx <= history_start_idx + seq_len:
        raise ValueError(f"Insufficient bars for month {month_start}")

    sub = X_tv_full.iloc[history_start_idx:test_end_idx]
    sub_idx = sub.index
    n_total = len(sub)
    n_windows = n_total - seq_len + 1

    # Z-normalize using checkpoint normalizer fit on TRAIN slice.
    # NOTE: TransVol's feature tensor includes ALL 101 cols (TIME_COLS
    # are in BOTH the feature tensor AND a separate Time2Vec branch).
    X_feats = sub.values.astype(np.float32)
    X_norm = normalizer.transform(X_feats)
    X_time = sub[TIME_COLS].values.astype(np.float32)

    # Sliding windows. Shape (n_windows, seq_len, F).
    X_seq = np.lib.stride_tricks.sliding_window_view(
        X_norm, seq_len, axis=0,
    ).transpose(0, 2, 1).copy()
    T_seq = np.lib.stride_tricks.sliding_window_view(
        X_time, seq_len, axis=0,
    ).transpose(0, 2, 1).copy()
    assert X_seq.shape == (n_windows, seq_len, X_norm.shape[1])

    last_bar_idx = sub_idx[seq_len - 1:]
    # Filter to windows whose LAST bar falls inside [month_start, month_end).
    in_month = (last_bar_idx >= month_start) & (last_bar_idx < month_end)
    X_seq = X_seq[in_month]
    T_seq = T_seq[in_month]
    last_bar_idx = last_bar_idx[in_month]

    return (
        torch.from_numpy(X_seq),
        torch.from_numpy(T_seq),
        last_bar_idx,
    )


class _ContinuousVarOnly(torch.nn.Module):
    """Wraps a TransVolEncoder so `forward(x, t)` returns a single
    scalar tensor (continuous_var_raw) — what the QLIKE head predicts.
    GradientExplainer works on tensor-out models."""
    def __init__(self, encoder: TransVolEncoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, x, t):
        out = self.encoder(x, t)
        # GradientExplainer expects (B, n_outputs); add singleton dim.
        return out["continuous_var_raw"].unsqueeze(-1)


def _audit_one_month(
    X_tv_full: pd.DataFrame,
    month: str,
    device: torch.device,
    bg_size: int = 64,
    test_size_max: int = 512,
) -> tuple[np.ndarray, dict]:
    """Load checkpoint for `month`, compute SHAP on test-month windows.
    Returns (mean_abs_shap_per_feature, meta)."""
    month_start = pd.Timestamp(month, tz="UTC")
    month_end = (month_start + pd.offsets.MonthBegin(1))

    ckpt_path = (
        REPO_ROOT / "vol_model" / "output" / "models"
        / "expanding_hybrid" / f"{month}_transvol.pt"
    )
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)

    ckpt = torch.load(str(ckpt_path), weights_only=False,
                      map_location=device)
    config = ckpt["config"]
    encoder = TransVolEncoder(config).to(device)
    encoder.load_state_dict(ckpt["model_state_dict"])
    encoder.eval()
    wrapped = _ContinuousVarOnly(encoder).to(device).eval()

    # Normalizer is embedded in the checkpoint (per-month, fit on
    # training slice).
    normalizer = FeatureNormalizer()
    normalizer.mean = ckpt["normalizer_mean"]
    normalizer.std = ckpt["normalizer_std"]

    # Build test-month tensor.
    X_seq, T_seq, last_bar_idx = _build_month_input_tensor(
        X_tv_full, normalizer, config.seq_len, month_start, month_end,
    )

    # Sub-sample test windows for SHAP speed (deterministic).
    n_test = X_seq.shape[0]
    if n_test > test_size_max:
        rng = np.random.RandomState(42)
        sel = rng.choice(n_test, test_size_max, replace=False)
        X_seq = X_seq[sel]
        T_seq = T_seq[sel]
    X_seq = X_seq.to(device)
    T_seq = T_seq.to(device)

    # Background sample: random windows from the FULL prior history.
    # Pull from X_tv_full BEFORE month_start.
    prior_start_idx = X_tv_full.index.searchsorted(month_start)
    if prior_start_idx < config.seq_len + bg_size:
        raise ValueError(f"Not enough prior history for {month} background")

    rng2 = np.random.RandomState(13)
    bg_indices = rng2.choice(
        prior_start_idx - config.seq_len, bg_size, replace=False,
    )
    bg_x_list, bg_t_list = [], []
    for ix in bg_indices:
        sub = X_tv_full.iloc[ix:ix + config.seq_len]
        bg_x_list.append(normalizer.transform(
            sub.values.astype(np.float32)
        ))
        bg_t_list.append(sub[TIME_COLS].values.astype(np.float32))
    bg_x = torch.from_numpy(np.stack(bg_x_list)).to(device)
    bg_t = torch.from_numpy(np.stack(bg_t_list)).to(device)

    # SHAP via GradientExplainer.
    t0 = time.time()
    explainer = shap.GradientExplainer(wrapped, [bg_x, bg_t])
    shap_values = explainer.shap_values([X_seq, T_seq])
    shap_s = time.time() - t0

    # shap.GradientExplainer for multi-input model returns either
    # (a) a list of arrays one-per-input, each shape (B, *input_dims, n_outputs)
    # (b) for n_outputs==1, may be shape (B, *input_dims) on some versions.
    # We expect (B, T, F) for the feature input after squeezing the
    # singleton output dim.
    shap_feats = shap_values[0]
    if isinstance(shap_feats, list):
        # Some shap versions wrap each output in a list.
        shap_feats = shap_feats[0]
    if shap_feats.ndim == 4 and shap_feats.shape[-1] == 1:
        shap_feats = shap_feats[..., 0]
    assert shap_feats.ndim == 3, (
        f"expected (B, T, F) but got {shap_feats.shape}"
    )
    # Aggregate across batch + time to per-feature.
    mean_abs_shap = np.abs(shap_feats).mean(axis=(0, 1))

    return mean_abs_shap, {
        "n_test_windows": int(X_seq.shape[0]),
        "shap_s": shap_s,
        "n_features": int(mean_abs_shap.shape[0]),
    }


def _build_decision(
    per_month_shap: dict[str, np.ndarray], feat_idx: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    matrix = pd.DataFrame(index=feat_idx)
    for month, vec in per_month_shap.items():
        matrix[month] = vec

    raw_ranks = matrix.rank(ascending=False, method="min").astype(int)
    mean_shap = matrix.mean(axis=1)
    median_shap = matrix.median(axis=1)
    max_shap = matrix.max(axis=1)
    min_shap = matrix.min(axis=1)
    shap_std = matrix.std(axis=1)
    median_rank = raw_ranks.median(axis=1)
    best_rank = raw_ranks.min(axis=1)
    worst_rank = raw_ranks.max(axis=1)
    rank_range = worst_rank - best_rank
    epsilon_per_col = matrix.max(axis=0) * 0.01
    above_eps = (matrix > epsilon_per_col).sum(axis=1)

    decision = pd.DataFrame({
        "feature": feat_idx,
        "mean_shap": mean_shap.values,
        "median_shap": median_shap.values,
        "max_shap": max_shap.values,
        "min_shap": min_shap.values,
        "shap_std": shap_std.values,
        "median_rank": median_rank.values.astype(int),
        "best_rank": best_rank.values.astype(int),
        "worst_rank": worst_rank.values.astype(int),
        "rank_range": rank_range.values.astype(int),
        "n_months_above_1pct": above_eps.values.astype(int),
    })

    n_months = len(per_month_shap)
    keep_above_pct = max(1, n_months * 4 // 6)  # >=4/6 pattern → scale to n_months

    def _flag(row):
        # KEEP if median_rank <= 35 OR best_rank <= 25 OR
        # n_months_above_1pct >= ~4/6 of audit months.
        if row["median_rank"] <= 35 or row["best_rank"] <= 25 \
                or row["n_months_above_1pct"] >= keep_above_pct:
            return "KEEP"
        if row["best_rank"] > 60 and row["n_months_above_1pct"] == 0:
            return "DROP"
        return "REVIEW"

    decision["decision"] = decision.apply(_flag, axis=1)
    decision = decision.sort_values("mean_shap", ascending=False).reset_index(drop=True)

    matrix_with_rank = matrix.join(raw_ranks.add_suffix("_rank"))
    matrix_with_rank.index.name = "feature"
    return decision, matrix_with_rank


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    config = VolModelConfig(asset="BTCUSDT")
    logger.info("Building dataset (clip=False)...")
    X, _y, _ = build_dataset(config, clip=False)

    # Apply the production TransVol input drop (126 → 101).
    X_tv = _select_transvol_features(X)
    logger.info("TransVol input: %d rows, %d features (%d dropped)",
                len(X_tv), X_tv.shape[1], X.shape[1] - X_tv.shape[1])

    # All 101 TransVol input columns participate in SHAP (TIME_COLS
    # are part of the feature tensor; they ALSO get used by Time2Vec
    # but the Time2Vec branch is gradient-isolated to its 2 inputs).
    feat_idx = list(X_tv.columns)

    per_month_shap: dict[str, np.ndarray] = {}
    per_month_meta: dict[str, dict] = {}

    for month, regime in AUDIT_MONTHS:
        logger.info("=" * 60)
        logger.info("[%s] %s", month, regime)
        try:
            shap_vec, meta = _audit_one_month(X_tv, month, device)
        except (FileNotFoundError, ValueError) as e:
            logger.error("[%s] SKIPPED: %s", month, e)
            continue
        per_month_shap[month] = shap_vec
        per_month_meta[month] = {"regime": regime, **meta}
        logger.info(
            "[%s] n_test=%d  shap=%.0fs  top-3 features by |SHAP|: %s",
            month, meta["n_test_windows"], meta["shap_s"],
            ", ".join(
                f"{feat_idx[i]} ({shap_vec[i]:.4f})"
                for i in shap_vec.argsort()[::-1][:3]
            ),
        )

    if not per_month_shap:
        raise RuntimeError("No months produced SHAP")

    decision, matrix = _build_decision(per_month_shap, feat_idx)

    out_dir = REPO_ROOT / "vol_model" / "output"
    decision.to_csv(out_dir / "shap_audit_transvol_decision.csv", index=False)
    matrix.to_csv(out_dir / "shap_audit_transvol_per_month.csv")

    n_keep = int((decision["decision"] == "KEEP").sum())
    n_drop = int((decision["decision"] == "DROP").sum())
    n_review = int((decision["decision"] == "REVIEW").sum())

    keep_features = decision[decision["decision"] == "KEEP"]["feature"].tolist()
    pruned_path = out_dir / "transvol_pruned_features.csv"
    pruned_path.write_text("\n".join(keep_features) + "\n")
    logger.info("Wrote pruned TransVol input list (%d features) to %s",
                len(keep_features), pruned_path)

    summary = out_dir / "shap_audit_transvol_summary.md"
    lines = [
        "# TransVol per-regime SHAP audit (production-matched)",
        "",
        f"Decision: KEEP={n_keep} / REVIEW={n_review} / DROP={n_drop}",
        "",
        "## Per-month metadata",
        "",
        "| month | regime | n_test_windows | shap_s |",
        "|---|---|---:|---:|",
    ]
    for m, meta in per_month_meta.items():
        lines.append(
            f"| {m} | {meta['regime']} | {meta['n_test_windows']:,} | "
            f"{meta['shap_s']:.0f} |"
        )

    lines += [
        "",
        "## Top 30 features by mean |SHAP|",
        "",
        "| rank | feature | mean SHAP | best rank | worst rank | months>1% | decision |",
        "|---:|---|---:|---:|---:|---:|---|",
    ]
    for i, row in decision.head(30).iterrows():
        lines.append(
            f"| {i+1} | `{row['feature']}` | {row['mean_shap']:.4f} | "
            f"{row['best_rank']} | {row['worst_rank']} | "
            f"{row['n_months_above_1pct']}/{len(per_month_shap)} | "
            f"{row['decision']} |"
        )

    summary.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote %s", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
