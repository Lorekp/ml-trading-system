"""Compare edge magnitude distributions: cycle-9 (pre-retrain) vs cycle-10 (post-retrain).

Tests the hypothesis (made 2026-05-10) that cycle-10's improved
TransVol embeddings + LGBM retrain produce sharper predictions →
larger |edge| → more bars cross the V2 thresholds (12 bps long, 22
bps short) → higher fire rate.

Inputs:
- pre:  ``backups/2026-05-09_pre_transvol_optuna_retrain/BTCUSDT_pruned_svi_har_pd_predictions_aligned_bias.parquet``
- post: ``vol_model/output/BTCUSDT_pruned_svi_har_pd_predictions_aligned_bias.parquet``

The aligned-bias parquets contain ``y_pred = LightGBM_raw + β*_month`` —
the exact prediction the live engine uses. We need a reference
``iv_used_pct`` to compute edge; reuse the V2 trade-log's ``iv_used``
column where available, otherwise fall back to the pre-retrain trade
log's iv_used (since the IV source is independent of the model).

Output: ``vol_arb/output/edge_distribution_cycle10.md`` + JSON.

Usage::

    PYTHONPATH=. python scripts/compare_edge_distributions_cycle10.py
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

logger = logging.getLogger(__name__)

PRE_PRED = REPO_ROOT / "backups" / "2026-05-09_pre_transvol_optuna_retrain" \
    / "BTCUSDT_pruned_svi_har_pd_predictions_aligned_bias.parquet"
POST_PRED = REPO_ROOT / "vol_model" / "output" \
    / "BTCUSDT_pruned_svi_har_pd_predictions_aligned_bias.parquet"
TRADE_LOG = REPO_ROOT / "vol_arb" / "output" / "BTC_v2_trade_log.parquet"

# V2 thresholds (live + backtest both)
THR_LONG_BPS = 12.0
THR_SHORT_BPS = 22.0


def _load_predictions_with_iv(pred_path: Path, iv_lookup: pd.Series) -> pd.DataFrame:
    """Load a predictions parquet + join an iv_used reference."""
    p = pd.read_parquet(pred_path)
    # The aligned-bias parquet has y_pred (corrected) + y_pred_raw + y_true
    if "y_pred" not in p.columns:
        raise ValueError(f"{pred_path} missing y_pred column")
    p["rv_pred_pct"] = np.exp(p["y_pred"]) * 100.0
    # Align iv lookup; iv_lookup is indexed by timestamp.
    if isinstance(p.index, pd.RangeIndex):
        if "timestamp" in p.columns:
            p = p.set_index("timestamp")
        else:
            raise ValueError(f"{pred_path} has no timestamp column")
    p["iv_used_pct"] = p.index.map(iv_lookup)
    p = p.dropna(subset=["iv_used_pct", "rv_pred_pct"])
    p["edge_vp"] = p["rv_pred_pct"] - p["iv_used_pct"]
    return p[["rv_pred_pct", "iv_used_pct", "edge_vp", "y_pred", "y_true"]]


def _summarize(name: str, df: pd.DataFrame) -> dict:
    edge = df["edge_vp"].values
    abs_edge = np.abs(edge)
    n = len(edge)
    metrics = {
        "name": name,
        "n_bars": int(n),
        "edge_mean_vp": float(edge.mean()),
        "edge_median_vp": float(np.median(edge)),
        "edge_std_vp": float(edge.std()),
        "abs_edge_mean_vp": float(abs_edge.mean()),
        "abs_edge_median_vp": float(np.median(abs_edge)),
        "abs_edge_p75_vp": float(np.percentile(abs_edge, 75)),
        "abs_edge_p90_vp": float(np.percentile(abs_edge, 90)),
        "abs_edge_p95_vp": float(np.percentile(abs_edge, 95)),
        "frac_long_candidate": float((edge > THR_LONG_BPS).mean()),
        "frac_short_candidate": float((edge < -THR_SHORT_BPS).mean()),
        "frac_in_band": float(((edge <= THR_LONG_BPS) & (edge >= -THR_SHORT_BPS)).mean()),
    }
    metrics["frac_any_candidate"] = (
        metrics["frac_long_candidate"] + metrics["frac_short_candidate"]
    )
    return metrics


def _delta_table(post: dict, pre: dict) -> str:
    keys = [
        ("n_bars", "{:.0f}", False),
        ("edge_mean_vp", "{:+.3f}", True),
        ("edge_median_vp", "{:+.3f}", True),
        ("abs_edge_mean_vp", "{:.3f}", True),
        ("abs_edge_median_vp", "{:.3f}", True),
        ("abs_edge_p90_vp", "{:.3f}", True),
        ("abs_edge_p95_vp", "{:.3f}", True),
        ("frac_long_candidate", "{:.2%}", False),
        ("frac_short_candidate", "{:.2%}", False),
        ("frac_any_candidate", "{:.2%}", False),
        ("frac_in_band", "{:.2%}", False),
    ]
    rows = ["| metric | pre-retrain | post-retrain | delta |",
            "|---|---:|---:|---:|"]
    for k, fmt, show_pct in keys:
        pre_v, post_v = pre[k], post[k]
        delta = post_v - pre_v
        if k == "n_bars":
            rows.append(f"| `{k}` | {fmt.format(pre_v)} | {fmt.format(post_v)} | {fmt.format(delta)} |")
        else:
            pct_str = ""
            if show_pct and pre_v != 0:
                pct_str = f" ({100*delta/pre_v:+.1f}%)"
            rows.append(
                f"| `{k}` | {fmt.format(pre_v)} | {fmt.format(post_v)} | "
                f"{fmt.format(delta)}{pct_str} |"
            )
    return "\n".join(rows)


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s: %(message)s")
    if not PRE_PRED.exists():
        logger.error("Pre-retrain backup not found at %s", PRE_PRED)
        return 1
    if not POST_PRED.exists():
        logger.error("Post-retrain predictions not yet written: %s", POST_PRED)
        logger.error("Run the orchestrator first or wait for step 2/5 to finish.")
        return 2
    if not TRADE_LOG.exists():
        logger.error("V2 trade log not found at %s — needed for iv_used reference",
                     TRADE_LOG)
        return 3

    logger.info("Loading V2 trade log for iv_used reference...")
    tl = pd.read_parquet(TRADE_LOG)
    if "iv_used" not in tl.columns:
        logger.error("Trade log missing iv_used column")
        return 4
    iv_lookup = tl["iv_used"].dropna()
    logger.info("iv_used reference: %d bars", len(iv_lookup))

    logger.info("Loading pre-retrain predictions: %s", PRE_PRED)
    pre = _load_predictions_with_iv(PRE_PRED, iv_lookup)
    logger.info("Loading post-retrain predictions: %s", POST_PRED)
    post = _load_predictions_with_iv(POST_PRED, iv_lookup)

    pre_metrics = _summarize("pre-retrain (cycle-9)", pre)
    post_metrics = _summarize("post-retrain (cycle-10)", post)

    table = _delta_table(post_metrics, pre_metrics)

    # Predicted fire-rate change (under unchanged jump-fraction filter
    # blocking ~6% from earlier homeserver investigation).
    fire_pre = pre_metrics["frac_any_candidate"] * 0.94
    fire_post = post_metrics["frac_any_candidate"] * 0.94
    multiplier = (fire_post / fire_pre) if fire_pre > 0 else float("nan")

    out_md = REPO_ROOT / "vol_arb" / "output" / "edge_distribution_cycle10.md"
    lines = [
        "# Edge magnitude distribution: cycle-9 vs cycle-10 (2026-05-10)",
        "",
        f"**V2 thresholds:** long > +{THR_LONG_BPS:.0f} bps, short < -{THR_SHORT_BPS:.0f} bps.",
        "",
        f"Hypothesis (formed 2026-05-10): cycle-10 LGBM with sharper TransVol",
        f"embeddings should produce larger |edge|, pushing more bars over the threshold.",
        "",
        "## Metrics",
        "",
        table,
        "",
        "## Implied fire-rate change",
        "",
        f"- Pre-retrain candidate rate (× 0.94 for jump-filter): **{fire_pre:.2%}** of bars",
        f"- Post-retrain candidate rate: **{fire_post:.2%}** of bars",
        f"- **Implied multiplier: {multiplier:.2f}×**",
        "",
        "(0.94 reflects the ~6% jump-fraction circuit-breaker block rate measured",
        "in the 2026-05-09 homeserver investigation.)",
        "",
    ]
    out_md.write_text("\n".join(lines))
    logger.info("Wrote %s", out_md)

    out_json = out_md.with_suffix(".json")
    out_json.write_text(json.dumps({
        "pre": pre_metrics, "post": post_metrics,
        "implied_fire_rate_pre": fire_pre,
        "implied_fire_rate_post": fire_post,
        "implied_fire_rate_multiplier": multiplier,
    }, indent=2, default=float))
    logger.info("Wrote %s", out_json)

    print("=" * 72)
    print("EDGE DISTRIBUTION: CYCLE-9 vs CYCLE-10")
    print("=" * 72)
    print(table)
    print()
    print(f"Implied fire-rate multiplier: {multiplier:.2f}x "
          f"({fire_pre:.2%} -> {fire_post:.2%} of bars)")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
