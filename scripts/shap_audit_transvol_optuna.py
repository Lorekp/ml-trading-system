"""TransVol SHAP audit with one-month Optuna sanity sweep + 8-month
production-matched audit (option A+C).

Step 1 (one-month Optuna sanity, 2024-08):
    Optuna 50 trials / 15 patience search over (lr, dropout,
    weight_decay) at max_epochs=30 per trial. Picks best config based
    on validation QLIKE on a representative audit month. ~4h GPU.

Step 2 (full-input audit, 8 months):
    For each audit month, train fresh TransVol with the Optuna-best
    config (capped at max_epochs=50 since hyperparams are tuned), then
    compute SHAP via shap.GradientExplainer on the test month.

Step 3 (pruned-input audit, 8 months):
    Drop features that landed in DROP/REVIEW from Step 2, retrain
    TransVol with Optuna-best config on the reduced input, recompute
    SHAP. Tests stability (which 60/62-style verdict for TransVol).

Output:
    vol_model/output/shap_audit_transvol_optuna_best_config.json
    vol_model/output/shap_audit_transvol_optuna_pass1_decision.csv
    vol_model/output/shap_audit_transvol_optuna_pass1_per_month.csv
    vol_model/output/shap_audit_transvol_optuna_pass2_decision.csv
    vol_model/output/shap_audit_transvol_optuna_pass2_per_month.csv
    vol_model/output/shap_audit_transvol_optuna_summary.md

Total runtime estimate: ~7h on RTX 4070 SUPER.
"""
from __future__ import annotations

import json
import logging
import math
import sys
import time
from copy import replace
from dataclasses import asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import optuna  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

# NOTE: ``shap`` is imported LAZILY inside ``_audit_month`` (the only consumer)
# so the production retrain path — which imports _make_per_month_datasets /
# _train_transvol / _select_transvol_features from this module — does NOT
# require the heavy (and here uninstalled) ``shap`` package. The rehab retrain
# (scripts/retrain_transvol_optuna.py) goes through that path and must run on a
# box without shap.

from vol_model.config import VolModelConfig  # noqa: E402
from vol_model.pipeline import build_dataset  # noqa: E402
from vol_model.transvol.config import TransVolConfig  # noqa: E402
from vol_model.transvol.dataset import (  # noqa: E402
    TIME_COLS, FeatureNormalizer, FeatureWinsorizer, TransVolDataset,
)
from vol_model.transvol.model import TransVolEncoder  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("transvol_optuna")
optuna.logging.set_verbosity(optuna.logging.WARNING)


AUDIT_MONTHS = [
    ("2022-06", "LUNA collapse aftermath + 3AC unwind"),
    ("2022-11", "FTX collapse + contagion"),
    ("2023-08", "mid-2023 range-bound calm"),
    ("2023-12", "late-2023 ETF speculation"),
    ("2024-04", "BTC halving cycle / transition"),
    ("2024-08", "yen-carry unwind"),       # used for Optuna sweep
    ("2025-04", "recent vol shock"),
    ("2026-01", "recent OOS"),
]
OPTUNA_SWEEP_MONTH = "2024-08"

DROP_PREFIXES = ("ofi_", "depth_mid", "oi_change", "oi_zscore", "oi_acceleration")


def _select_transvol_features(X: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in X.columns if not c.startswith(DROP_PREFIXES)]
    return X[cols]


def _make_per_month_datasets(
    X: pd.DataFrame, cont: pd.Series, jump: pd.Series,
    month_start: pd.Timestamp, seq_len: int,
    rehab: bool = False,
) -> tuple[TransVolDataset, TransVolDataset, FeatureNormalizer] | tuple[
    TransVolDataset, TransVolDataset, FeatureNormalizer, dict
]:
    """Build train + val datasets for a single audit month using
    PRIOR-only data. Train = first 85% of prior data; val = last 15%
    of prior data.

    ``rehab=False`` (PRODUCTION DEFAULT): byte-identical to the deployed
    path — feature z-norm fit on the train slice, time z-norm re-fit (not
    returned), raw-variance continuous target. Returns
    ``(train_ds, val_ds, feature_normalizer)``.

    ``rehab=True`` applies the pre-committed R1+R2+R6 clean build, every
    statistic fit on the TRAIN slice only (no leakage):
      * R1 — winsorize features at p0.5/p99.5 (train-fit) BEFORE z-score.
      * R6 — return the time-normalizer (train-fit) so it travels with the
        checkpoint instead of being re-fit on the extraction window.
      * R2 — log-target: ``continuous_var`` -> log; the raw variance is kept
        for the QLIKE validation metric.
    Returns ``(train_ds, val_ds, feature_normalizer, prep)`` where ``prep``
    carries the fitted winsorizer / time-normalizer / log-bias to persist.
    """
    prior = X.index < month_start
    X_prior = X[prior]
    cont_prior = cont[prior]
    jump_prior = jump[prior]
    n = len(X_prior)
    val_start = int(n * 0.85)

    feat_vals = X_prior.values.astype(np.float32)
    time_vals = X_prior[TIME_COLS].values.astype(np.float32)
    cont_vals = cont_prior.values.astype(np.float32)
    jump_vals = jump_prior.values.astype(np.float32)

    # Normalize jump_var to O(1) (identical in both modes; train-fit mean).
    jump_mean = max(float(jump_vals[:val_start].mean()), 1e-15)
    jump_norm = jump_vals / jump_mean

    if not rehab:
        # ── PRODUCTION PATH (unchanged) ──────────────────────────────
        norm = FeatureNormalizer().fit(feat_vals[:val_start])
        feat_norm = norm.transform(feat_vals)

        time_norm_obj = FeatureNormalizer().fit(time_vals[:val_start])
        time_norm = time_norm_obj.transform(time_vals)

        train_ds = TransVolDataset(
            feat_norm[:val_start], time_norm[:val_start],
            cont_vals[:val_start], jump_norm[:val_start], seq_len,
        )
        val_ds = TransVolDataset(
            feat_norm[val_start:], time_norm[val_start:],
            cont_vals[val_start:], jump_norm[val_start:], seq_len,
        )
        return train_ds, val_ds, norm

    # ── REHAB PATH (R1 + R2 + R6) ────────────────────────────────────
    # R1: winsorize features (train-fit p0.5/p99.5) BEFORE z-scoring.
    winsor = FeatureWinsorizer(lo_q=0.005, hi_q=0.995).fit(feat_vals[:val_start])
    feat_clipped = winsor.transform(feat_vals)
    norm = FeatureNormalizer().fit(feat_clipped[:val_start])
    feat_norm = norm.transform(feat_clipped)

    # R6: time-normalizer fit on TRAIN slice only and RETURNED for persistence
    # (production re-fits it on the extraction window -> the A2-M1 leak).
    time_norm_obj = FeatureNormalizer().fit(time_vals[:val_start])
    time_norm = time_norm_obj.transform(time_vals)

    # R2: log-target. continuous_var is strictly positive variance; guard the
    # log with a tiny floor. The head predicts log-variance directly.
    cont_log = np.log(np.clip(cont_vals, 1e-12, None)).astype(np.float32)
    cont_log_bias = float(cont_log[:val_start].mean())

    train_ds = TransVolDataset(
        feat_norm[:val_start], time_norm[:val_start],
        cont_log[:val_start], jump_norm[:val_start], seq_len,
        continuous_var_metric=cont_vals[:val_start],
    )
    val_ds = TransVolDataset(
        feat_norm[val_start:], time_norm[val_start:],
        cont_log[val_start:], jump_norm[val_start:], seq_len,
        continuous_var_metric=cont_vals[val_start:],
    )
    prep = {
        "winsor_lo": winsor.lo,
        "winsor_hi": winsor.hi,
        "winsor_lo_q": winsor.lo_q,
        "winsor_hi_q": winsor.hi_q,
        "time_normalizer_mean": time_norm_obj.mean,
        "time_normalizer_std": time_norm_obj.std,
        "cont_log_bias": cont_log_bias,
        "target_log": True,
    }
    return train_ds, val_ds, norm, prep


def _train_transvol(
    config: TransVolConfig,
    train_ds: TransVolDataset,
    val_ds: TransVolDataset,
    device: torch.device,
    max_epochs: int,
    patience: int,
    log_every: int = 0,
) -> tuple[TransVolEncoder, float]:
    """Train TransVol; return (best model, best val QLIKE)."""
    train_loader = DataLoader(
        train_ds, batch_size=config.batch_size, shuffle=True,
        num_workers=0, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.batch_size * 2, shuffle=False,
        num_workers=0, pin_memory=True,
    )
    model = TransVolEncoder(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay,
    )
    steps_per_epoch = max(len(train_loader) // config.grad_accum_steps, 1)
    total_steps = steps_per_epoch * max_epochs
    warmup = min(config.warmup_steps, max(total_steps // 10, 1))

    def lr_lambda(step):
        if step < warmup:
            return step / max(warmup, 1)
        progress = (step - warmup) / max(total_steps - warmup, 1)
        return max(0.01, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler()

    best_val = float("inf")
    best_state = None
    no_improve = 0

    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(train_loader):
            x = batch["features"].to(device)
            t = batch["time_features"].to(device)
            targets = {
                "continuous_var": batch["continuous_var"].to(device),
                "jump_var": batch["jump_var"].to(device),
            }
            if "continuous_var_raw_for_metric" in batch:
                targets["continuous_var_raw_for_metric"] = (
                    batch["continuous_var_raw_for_metric"].to(device)
                )
            with torch.amp.autocast("cuda"):
                outputs = model(x, t)
                losses = model.compute_loss(outputs, targets)
                loss = losses["total"] / config.grad_accum_steps
            scaler.scale(loss).backward()
            if (step + 1) % config.grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

        model.eval()
        total_q, n = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                x = batch["features"].to(device)
                t = batch["time_features"].to(device)
                tgt = {
                    "continuous_var": batch["continuous_var"].to(device),
                    "jump_var": batch["jump_var"].to(device),
                }
                if "continuous_var_raw_for_metric" in batch:
                    tgt["continuous_var_raw_for_metric"] = (
                        batch["continuous_var_raw_for_metric"].to(device)
                    )
                with torch.amp.autocast("cuda"):
                    out = model(x, t)
                    losses = model.compute_loss(out, tgt)
                total_q += losses["qlike"].item() * x.shape[0]
                n += x.shape[0]
        val_q = total_q / max(n, 1)
        if log_every and (epoch + 1) % log_every == 0:
            logger.info("    epoch %d/%d val_qlike=%.4f", epoch + 1, max_epochs, val_q)
        if val_q < best_val:
            best_val = val_q
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    model.load_state_dict(best_state)
    return model, best_val


# ── Optuna sanity sweep ──────────────────────────────────────────

def _optuna_sweep(
    X_tv: pd.DataFrame, cont: pd.Series, jump: pd.Series,
    sweep_month: str, device: torch.device, n_trials: int = 50,
    patience: int = 15, max_epochs_per_trial: int = 30,
) -> dict:
    base_config = TransVolConfig()
    seq_len = base_config.seq_len
    train_ds, val_ds, _ = _make_per_month_datasets(
        X_tv, cont, jump, pd.Timestamp(sweep_month, tz="UTC"), seq_len,
    )
    logger.info(
        "Optuna sweep on %s: train=%d val=%d", sweep_month,
        len(train_ds), len(val_ds),
    )

    def objective(trial: optuna.Trial) -> float:
        cfg = replace(
            base_config,
            lr=trial.suggest_float("lr", 1e-5, 5e-4, log=True),
            dropout=trial.suggest_float("dropout", 0.05, 0.30),
            weight_decay=trial.suggest_float(
                "weight_decay", 1e-3, 1e-1, log=True,
            ),
            n_features=X_tv.shape[1],
        )
        try:
            _, val_q = _train_transvol(
                cfg, train_ds, val_ds, device,
                max_epochs=max_epochs_per_trial, patience=10,
            )
        except RuntimeError as e:
            logger.warning("Trial %d failed: %s", trial.number, e)
            return float("inf")
        return val_q

    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(sampler=sampler, direction="minimize")

    best_seen = [float("inf")]
    n_since_best = [0]

    def callback(study, _trial):
        if study.best_value < best_seen[0]:
            best_seen[0] = study.best_value
            n_since_best[0] = 0
        else:
            n_since_best[0] += 1
        if n_since_best[0] >= patience:
            study.stop()

    t0 = time.time()
    study.optimize(objective, n_trials=n_trials, callbacks=[callback])
    elapsed = time.time() - t0

    logger.info(
        "Optuna sweep done in %.1fs (n=%d completed). Best val QLIKE=%.4f. "
        "Params: %s",
        elapsed, len(study.trials), study.best_value, study.best_params,
    )
    return study.best_params


# ── SHAP audit (per audit month) ─────────────────────────────────

class _ContinuousVarOnly(torch.nn.Module):
    def __init__(self, encoder: TransVolEncoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, x, t):
        return self.encoder(x, t)["continuous_var_raw"].unsqueeze(-1)


def _build_month_inputs(
    X_tv: pd.DataFrame, normalizer: FeatureNormalizer, seq_len: int,
    month_start: pd.Timestamp, month_end: pd.Timestamp,
):
    history_start = max(X_tv.index.searchsorted(month_start) - (seq_len - 1), 0)
    test_end = X_tv.index.searchsorted(month_end)
    sub = X_tv.iloc[history_start:test_end]
    sub_idx = sub.index
    n_total = len(sub)
    n_windows = n_total - seq_len + 1

    feat_vals = sub.values.astype(np.float32)
    feat_norm = normalizer.transform(feat_vals)
    time_vals = sub[TIME_COLS].values.astype(np.float32)

    X_seq = np.lib.stride_tricks.sliding_window_view(
        feat_norm, seq_len, axis=0,
    ).transpose(0, 2, 1).copy()
    T_seq = np.lib.stride_tricks.sliding_window_view(
        time_vals, seq_len, axis=0,
    ).transpose(0, 2, 1).copy()
    last_idx = sub_idx[seq_len - 1:]
    in_month = (last_idx >= month_start) & (last_idx < month_end)
    return torch.from_numpy(X_seq[in_month]), torch.from_numpy(T_seq[in_month])


def _audit_month(
    X_tv: pd.DataFrame, cont: pd.Series, jump: pd.Series, month: str,
    best_params: dict, device: torch.device,
    full_train_max_epochs: int = 50, full_train_patience: int = 12,
    test_size_max: int = 512, bg_size: int = 64,
) -> tuple[np.ndarray, dict]:
    month_start = pd.Timestamp(month, tz="UTC")
    month_end = month_start + pd.offsets.MonthBegin(1)

    base_config = TransVolConfig()
    seq_len = base_config.seq_len
    train_ds, val_ds, normalizer = _make_per_month_datasets(
        X_tv, cont, jump, month_start, seq_len,
    )
    cfg = replace(
        base_config,
        lr=best_params["lr"],
        dropout=best_params["dropout"],
        weight_decay=best_params["weight_decay"],
        n_features=X_tv.shape[1],
    )
    t0 = time.time()
    encoder, val_q = _train_transvol(
        cfg, train_ds, val_ds, device,
        max_epochs=full_train_max_epochs, patience=full_train_patience,
    )
    train_s = time.time() - t0

    encoder.eval()
    wrapped = _ContinuousVarOnly(encoder).to(device).eval()

    # Build month tensors.
    X_seq, T_seq = _build_month_inputs(
        X_tv, normalizer, seq_len, month_start, month_end,
    )
    if X_seq.shape[0] > test_size_max:
        rng = np.random.RandomState(42)
        sel = rng.choice(X_seq.shape[0], test_size_max, replace=False)
        X_seq = X_seq[sel]
        T_seq = T_seq[sel]
    X_seq = X_seq.to(device)
    T_seq = T_seq.to(device)

    # Background: random prior windows.
    rng2 = np.random.RandomState(13)
    prior_start = X_tv.index.searchsorted(month_start)
    bg_idx = rng2.choice(prior_start - seq_len, bg_size, replace=False)
    bg_x_list, bg_t_list = [], []
    for ix in bg_idx:
        sub = X_tv.iloc[ix:ix + seq_len]
        bg_x_list.append(normalizer.transform(sub.values.astype(np.float32)))
        bg_t_list.append(sub[TIME_COLS].values.astype(np.float32))
    bg_x = torch.from_numpy(np.stack(bg_x_list)).to(device)
    bg_t = torch.from_numpy(np.stack(bg_t_list)).to(device)

    import shap  # lazy: only the SHAP-audit path needs it (see module header)
    t0 = time.time()
    explainer = shap.GradientExplainer(wrapped, [bg_x, bg_t])
    shap_values = explainer.shap_values([X_seq, T_seq])
    shap_s = time.time() - t0

    shap_feats = shap_values[0]
    if isinstance(shap_feats, list):
        shap_feats = shap_feats[0]
    if shap_feats.ndim == 4 and shap_feats.shape[-1] == 1:
        shap_feats = shap_feats[..., 0]
    mean_abs_shap = np.abs(shap_feats).mean(axis=(0, 1))

    return mean_abs_shap, {
        "n_test_windows": int(X_seq.shape[0]),
        "train_s": train_s,
        "shap_s": shap_s,
        "val_qlike": val_q,
    }


# ── Decision aggregation ────────────────────────────────────────

def _build_decision(
    per_month_shap: dict[str, np.ndarray], feat_idx: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    matrix = pd.DataFrame(index=feat_idx)
    for month, vec in per_month_shap.items():
        matrix[month] = vec
    raw_ranks = matrix.rank(ascending=False, method="min").astype(int)

    n_months = len(per_month_shap)
    keep_above_pct = max(1, n_months * 4 // 6)

    decision = pd.DataFrame({
        "feature": feat_idx,
        "mean_shap": matrix.mean(axis=1).values,
        "median_shap": matrix.median(axis=1).values,
        "max_shap": matrix.max(axis=1).values,
        "min_shap": matrix.min(axis=1).values,
        "shap_std": matrix.std(axis=1).values,
        "median_rank": raw_ranks.median(axis=1).values.astype(int),
        "best_rank": raw_ranks.min(axis=1).values.astype(int),
        "worst_rank": raw_ranks.max(axis=1).values.astype(int),
        "n_months_above_1pct": (
            matrix > matrix.max(axis=0) * 0.01
        ).sum(axis=1).values.astype(int),
    })

    def flag(row):
        if (row["median_rank"] <= 35 or row["best_rank"] <= 25
                or row["n_months_above_1pct"] >= keep_above_pct):
            return "KEEP"
        if row["best_rank"] > 60 and row["n_months_above_1pct"] == 0:
            return "DROP"
        return "REVIEW"
    decision["decision"] = decision.apply(flag, axis=1)
    decision = decision.sort_values("mean_shap", ascending=False).reset_index(drop=True)

    matrix_with_rank = matrix.join(raw_ranks.add_suffix("_rank"))
    matrix_with_rank.index.name = "feature"
    return decision, matrix_with_rank


def _run_audit_pass(
    X_tv: pd.DataFrame, cont: pd.Series, jump: pd.Series,
    best_params: dict, device: torch.device, label: str,
) -> pd.DataFrame:
    per_month_shap = {}
    per_month_meta = {}
    for month, regime in AUDIT_MONTHS:
        try:
            logger.info("[%s] [%s] %s", label, month, regime)
            shap_vec, meta = _audit_month(
                X_tv, cont, jump, month, best_params, device,
            )
        except (FileNotFoundError, ValueError) as e:
            logger.error("[%s] [%s] SKIPPED: %s", label, month, e)
            continue
        per_month_shap[month] = shap_vec
        per_month_meta[month] = {"regime": regime, **meta}
        logger.info(
            "[%s] [%s] train=%.0fs shap=%.0fs val_qlike=%.4f n_test=%d",
            label, month, meta["train_s"], meta["shap_s"],
            meta["val_qlike"], meta["n_test_windows"],
        )

    decision, matrix = _build_decision(per_month_shap, list(X_tv.columns))
    out_dir = REPO_ROOT / "vol_model" / "output"
    decision.to_csv(out_dir / f"shap_audit_transvol_optuna_{label}_decision.csv",
                    index=False)
    matrix.to_csv(out_dir / f"shap_audit_transvol_optuna_{label}_per_month.csv")
    return decision


# ── Main ────────────────────────────────────────────────────────

def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    config = VolModelConfig(asset="BTCUSDT")
    logger.info("Building dataset (clip=False)...")
    X, _y, _ = build_dataset(config, clip=False)
    # Need cont + jump too for training; reload labels with clip=False.
    from vol_model.labels import load_labels
    labels = load_labels(config, clip=False)
    common = X.index.intersection(labels.index)
    X = X.loc[common]
    cont = labels.loc[common, "continuous_var"]
    jump = labels.loc[common, "jump_var"]

    X_tv_full = _select_transvol_features(X)
    logger.info("TransVol input (full): %d rows, %d features",
                len(X_tv_full), X_tv_full.shape[1])

    out_dir = REPO_ROOT / "vol_model" / "output"
    best_path = out_dir / "shap_audit_transvol_optuna_best_config.json"

    # Step 1: Optuna sanity sweep on one month.
    if best_path.exists():
        best_params = json.loads(best_path.read_text())
        logger.info("Loaded cached best Optuna params: %s", best_params)
    else:
        logger.info("=" * 60)
        logger.info("Step 1: Optuna sanity sweep on %s (compressed budget)",
                    OPTUNA_SWEEP_MONTH)
        logger.info("=" * 60)
        best_params = _optuna_sweep(
            X_tv_full, cont, jump, OPTUNA_SWEEP_MONTH, device,
        )
        best_path.write_text(json.dumps(best_params, indent=2))
        logger.info("Saved best config to %s", best_path)

    # Step 2: Pass 1 SHAP audit on 101-feature input.
    logger.info("=" * 60)
    logger.info("Step 2: Pass 1 SHAP audit (full 101-feature input)")
    logger.info("=" * 60)
    pass1 = _run_audit_pass(
        X_tv_full, cont, jump, best_params, device, "pass1",
    )
    n_keep1 = int((pass1["decision"] == "KEEP").sum())
    logger.info("Pass 1: KEEP=%d / total=%d", n_keep1, len(pass1))

    keep_features = pass1[pass1["decision"] == "KEEP"]["feature"].tolist()
    pruned_path = out_dir / "transvol_pruned_features_optuna.csv"
    pruned_path.write_text("\n".join(keep_features) + "\n")

    if len(keep_features) == X_tv_full.shape[1]:
        logger.info(
            "Pass 1 kept all %d features → Pass 2 would be redundant. "
            "Skipping.", len(keep_features),
        )
        pass2_status = "skipped (Pass 1 dropped 0 features)"
    else:
        logger.info("=" * 60)
        logger.info("Step 3: Pass 2 SHAP audit on %d-feature pruned input",
                    len(keep_features))
        logger.info("=" * 60)
        X_tv_pruned = X_tv_full[keep_features]
        pass2 = _run_audit_pass(
            X_tv_pruned, cont, jump, best_params, device, "pass2",
        )
        n_keep2 = int((pass2["decision"] == "KEEP").sum())
        pass2_status = f"KEEP={n_keep2} / total={len(pass2)}"
        logger.info("Pass 2: %s", pass2_status)

    # Summary doc.
    summary = out_dir / "shap_audit_transvol_optuna_summary.md"
    summary.write_text(
        "# TransVol SHAP audit (Optuna A+C compressed)\n\n"
        f"Optuna best params: `{json.dumps(best_params, indent=2)}`\n\n"
        f"Pass 1 (full 101-feature input): KEEP={n_keep1} / total={len(pass1)}\n\n"
        f"Pass 2 ({len(keep_features)}-feature pruned input): {pass2_status}\n",
        encoding="utf-8",
    )
    logger.info("Wrote %s", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
