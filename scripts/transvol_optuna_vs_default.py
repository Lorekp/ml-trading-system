"""Cross-month validation of TransVol Optuna-best hyperparameters
vs production defaults.

The Optuna sanity sweep ran on a SINGLE month (2024-08) and produced
best params (lr=7.5e-5, dropout=0.198, wd=0.0012). To verify these
aren't overfit to the sweep month, we train TransVol on each of the
8 audit months with BOTH:

- production defaults: lr=3e-5, dropout=0.15, wd=0.01
- Optuna-best:        lr=7.5e-5, dropout=0.198, wd=0.0012

with identical training budgets (max_epochs=50, patience=12) and
identical train/val splits per month. We then compare val QLIKE
side-by-side. If Optuna-best dominates ≥6/8 months → confidence high.
If it only wins on the sweep month + 1-2 others → likely overfit and
production defaults are fine.

Output:
    vol_model/output/shap_audit_transvol_optuna_validation.csv
    vol_model/output/shap_audit_transvol_optuna_validation.md

ETA: 8 months × 2 configs × ~3-5 min = ~30-90 min on GPU.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from copy import replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from vol_model.config import VolModelConfig  # noqa: E402
from vol_model.labels import load_labels  # noqa: E402
from vol_model.pipeline import build_dataset  # noqa: E402
from vol_model.transvol.config import TransVolConfig  # noqa: E402

# Reuse training + dataset helpers from the parent audit script.
from scripts.shap_audit_transvol_optuna import (  # noqa: E402
    AUDIT_MONTHS,
    _make_per_month_datasets,
    _select_transvol_features,
    _train_transvol,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("transvol_optuna_val")


PRODUCTION_DEFAULTS = {
    "lr": 3e-5,
    "dropout": 0.15,
    "weight_decay": 0.01,
}


def _train_one(
    X_tv: pd.DataFrame, cont: pd.Series, jump: pd.Series, month: str,
    params: dict, device: torch.device, max_epochs: int = 50,
    patience: int = 12,
) -> tuple[float, float]:
    base = TransVolConfig()
    train_ds, val_ds, _ = _make_per_month_datasets(
        X_tv, cont, jump, pd.Timestamp(month, tz="UTC"), base.seq_len,
    )
    cfg = replace(
        base,
        lr=params["lr"], dropout=params["dropout"],
        weight_decay=params["weight_decay"],
        n_features=X_tv.shape[1],
    )
    t0 = time.time()
    _, val_q = _train_transvol(
        cfg, train_ds, val_ds, device,
        max_epochs=max_epochs, patience=patience,
    )
    return val_q, time.time() - t0


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    out_dir = REPO_ROOT / "vol_model" / "output"
    best_path = out_dir / "shap_audit_transvol_optuna_best_config.json"
    if not best_path.exists():
        raise FileNotFoundError(
            "Run shap_audit_transvol_optuna.py first to produce best config."
        )
    optuna_best = json.loads(best_path.read_text())
    logger.info("Optuna-best params: %s", optuna_best)
    logger.info("Production defaults: %s", PRODUCTION_DEFAULTS)

    config = VolModelConfig(asset="BTCUSDT")
    X, _y, _ = build_dataset(config, clip=False)
    labels = load_labels(config, clip=False)
    common = X.index.intersection(labels.index)
    X = X.loc[common]
    cont = labels.loc[common, "continuous_var"]
    jump = labels.loc[common, "jump_var"]
    X_tv = _select_transvol_features(X)
    logger.info("TransVol input: %d rows, %d features", len(X_tv), X_tv.shape[1])

    rows = []
    for month, regime in AUDIT_MONTHS:
        logger.info("=" * 60)
        logger.info("[%s] %s", month, regime)
        try:
            qlike_default, dt_default = _train_one(
                X_tv, cont, jump, month, PRODUCTION_DEFAULTS, device,
            )
            qlike_optuna, dt_optuna = _train_one(
                X_tv, cont, jump, month, optuna_best, device,
            )
        except (ValueError, RuntimeError) as e:
            logger.error("[%s] SKIPPED: %s", month, e)
            continue
        rows.append({
            "month": month,
            "regime": regime,
            "qlike_default": qlike_default,
            "qlike_optuna": qlike_optuna,
            "delta_qlike": qlike_optuna - qlike_default,
            "delta_pct": 100 * (qlike_optuna - qlike_default) / qlike_default,
            "winner": "optuna" if qlike_optuna < qlike_default else "default",
            "default_train_s": dt_default,
            "optuna_train_s": dt_optuna,
        })
        logger.info(
            "[%s] default qlike=%.4f (%.0fs)  optuna qlike=%.4f (%.0fs)  "
            "delta=%+.4f (%+.2f%%) winner=%s",
            month, qlike_default, dt_default, qlike_optuna, dt_optuna,
            qlike_optuna - qlike_default,
            100 * (qlike_optuna - qlike_default) / qlike_default,
            "optuna" if qlike_optuna < qlike_default else "default",
        )

    df = pd.DataFrame(rows)
    csv_path = out_dir / "shap_audit_transvol_optuna_validation.csv"
    df.to_csv(csv_path, index=False)

    # Summary.
    n_opt_wins = (df["winner"] == "optuna").sum()
    n_def_wins = (df["winner"] == "default").sum()
    avg_delta_pct = df["delta_pct"].mean()
    median_delta_pct = df["delta_pct"].median()

    md_path = out_dir / "shap_audit_transvol_optuna_validation.md"
    lines = [
        "# TransVol Optuna-best vs production defaults — cross-month validation",
        "",
        f"Optuna-best params (from sweep on 2024-08): `{json.dumps(optuna_best)}`",
        f"Production defaults: `{json.dumps(PRODUCTION_DEFAULTS)}`",
        "",
        f"Both trained with max_epochs=50, patience=12, identical train/val splits.",
        "",
        "## Per-month QLIKE",
        "",
        "| month | regime | default | optuna | delta | delta% | winner |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for _, r in df.iterrows():
        lines.append(
            f"| {r['month']} | {r['regime']} | {r['qlike_default']:.4f} | "
            f"{r['qlike_optuna']:.4f} | {r['delta_qlike']:+.4f} | "
            f"{r['delta_pct']:+.2f}% | {r['winner']} |"
        )
    lines += [
        "",
        f"## Verdict",
        f"- Optuna wins in **{n_opt_wins}/{len(df)}** months",
        f"- Default wins in **{n_def_wins}/{len(df)}** months",
        f"- Average delta: **{avg_delta_pct:+.2f}%** (negative = Optuna better)",
        f"- Median delta: **{median_delta_pct:+.2f}%**",
        "",
    ]

    if n_opt_wins >= 6:
        lines.append(
            "**Optuna params are robust** — win in ≥6/8 months, not "
            "just the sweep month."
        )
    elif n_opt_wins >= 5:
        lines.append(
            "**Mixed verdict** — Optuna wins majority but not robustly. "
            "Consider sticking with production defaults for stability."
        )
    else:
        lines.append(
            "**Optuna params likely overfit to 2024-08** — production "
            "defaults match or beat them in ≥4/8 months. Use defaults "
            "for the production retrain."
        )

    md_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote %s", md_path)
    logger.info(
        "Optuna wins %d/%d, avg delta %+.2f%%, median %+.2f%%",
        n_opt_wins, len(df), avg_delta_pct, median_delta_pct,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
