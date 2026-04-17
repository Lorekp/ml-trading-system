"""Load the production signal + hedging models from disk.

Three artifacts, loaded once at engine startup and reused for the
lifetime of the process:

1. **LightGBM signal model** — ``{YYYY-MM}.txt`` + ``{YYYY-MM}_features.json``
   from ``vol_model/output/models/expanding_hybrid/``.  We pick the
   most recent month whose walk-forward train window ends ≤ today.
2. **TransVol encoder** — ``{YYYY-MM}_transvol.pt`` from the same
   directory.  Loaded with ``weights_only=False`` because the checkpoint
   contains normaliser state objects.
3. **DRL delta-hedging policy** — ``production_sac_v3_final.zip`` +
   ``production_sac_v3_final_vecnormalize.pkl`` from
   ``vol_arb/drl_hedging/models/``.  Loaded via ``stable_baselines3.SAC``
   with CPU device (inference is cheap; GPU not needed).

The loader returns a :class:`ModelBundle` dataclass holding all three
plus the feature list + month key so downstream code has everything
it needs to do a forward pass.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


_WALK_FORWARD_MONTH_RE = re.compile(r"^(\d{4}-\d{2})\.txt$")


@dataclass
class ModelBundle:
    """All loaded signal + hedging artifacts for the paper-trading engine.

    Passed around as an opaque bundle to avoid threading four
    separate parameters through every function that needs model state.
    """
    month_key: str                     # e.g., "2026-03"
    lgbm_model: Any                    # lightgbm.Booster
    lgbm_features: list[str]           # 51 feature names in the order LightGBM expects
    transvol_model: Any                # TransVolEncoder in eval mode
    transvol_config: Any               # TransVolConfig used to build the encoder
    transvol_feat_mean: Any            # normaliser mean (numpy array)
    transvol_feat_std: Any             # normaliser std  (numpy array)
    transvol_time_mean: Any
    transvol_time_std: Any
    drl_policy: Any                    # stable_baselines3.SAC (or subclass)
    drl_vec_normalize_mean: Any | None  # numpy array or None if VecNorm not used
    drl_vec_normalize_var: Any | None
    drl_vec_normalize_clip: float


def _find_latest_month(
    models_dir: Path,
    transvol_dir: Path | None = None,
) -> str:
    """Return the latest YYYY-MM month with a complete artifact set.

    A complete set is ``{month}.txt`` + ``{month}_features.json`` in
    ``models_dir``, AND ``{month}_transvol.pt`` in ``transvol_dir``
    (defaults to ``models_dir``).  When the two directories differ
    (e.g., T2.1 pruned+SVI variant reuses hybrid's TransVol weights),
    the loader checks both paths independently so an incomplete
    TransVol rollout surfaces as a warning rather than a silent
    half-working month pick.
    """
    tv_dir = transvol_dir if transvol_dir is not None else models_dir
    months: list[str] = []
    for path in models_dir.glob("*.txt"):
        match = _WALK_FORWARD_MONTH_RE.match(path.name)
        if match is None:
            continue
        month = match.group(1)
        features_path = models_dir / f"{month}_features.json"
        transvol_path = tv_dir / f"{month}_transvol.pt"
        if features_path.exists() and transvol_path.exists():
            months.append(month)
    if not months:
        raise FileNotFoundError(
            f"No complete (txt + features.json in {models_dir}, "
            f"transvol.pt in {tv_dir}) walk-forward artifact found",
        )
    months.sort()  # lexicographic works for YYYY-MM
    return months[-1]


def load_model_bundle(
    *,
    lgbm_models_dir: str | Path = "vol_model/output/models/expanding_pruned_svi_v1",
    transvol_models_dir: str | Path | None = "vol_model/output/models/expanding_hybrid",
    drl_policy_path: str | Path = "vol_arb/drl_hedging/models/production_sac_v4_final.zip",
    drl_vecnormalize_path: str | Path = "vol_arb/drl_hedging/models/production_sac_v4_final_vecnormalize.pkl",
    month_override: str | None = None,
) -> ModelBundle:
    """Load the full LightGBM + TransVol + DRL artifact bundle.

    Parameters
    ----------
    lgbm_models_dir : str | Path
        Directory holding ``{YYYY-MM}.txt`` + ``{YYYY-MM}_features.json``
        for each walk-forward month.  The T2.1 pruned+SVI variant lives
        at ``vol_model/output/models/expanding_pruned_svi_v1``.
    transvol_models_dir : str | Path or None
        Directory holding ``{YYYY-MM}_transvol.pt`` TransVol encoder
        checkpoints.  Defaults to the hybrid run (``expanding_hybrid``)
        because T2.1 reuses those weights unchanged.  Pass ``None`` to
        use ``lgbm_models_dir`` for TransVol as well (single-dir layout).
    drl_policy_path : str | Path
        Path to the production SAC policy zip (from
        :mod:`vol_arb.drl_hedging.train_sac`).
    drl_vecnormalize_path : str | Path
        Path to the VecNormalize stats pickle saved alongside the DRL
        policy.  Optional — if missing, the DRL hedger runs without
        observation normalisation (OK for policies trained without it).
    month_override : str, optional
        Force loading a specific month's LightGBM + TransVol artifacts
        instead of the auto-detected latest.  Useful for testing +
        reproducing specific historical signals.
    """
    import lightgbm as lgb
    import numpy as np
    import torch

    models_dir = Path(lgbm_models_dir)
    if not models_dir.exists():
        raise FileNotFoundError(f"LightGBM models directory not found: {models_dir}")

    tv_models_dir = (
        Path(transvol_models_dir) if transvol_models_dir is not None else models_dir
    )
    if not tv_models_dir.exists():
        raise FileNotFoundError(
            f"TransVol models directory not found: {tv_models_dir}",
        )

    month = month_override or _find_latest_month(models_dir, tv_models_dir)
    logger.info(
        "Loading walk-forward artifacts for month %s "
        "(lgbm from %s, transvol from %s)",
        month, models_dir, tv_models_dir,
    )

    # ── LightGBM ────────────────────────────────────────────────────
    lgbm_txt = models_dir / f"{month}.txt"
    lgbm_features_json = models_dir / f"{month}_features.json"
    lgbm_model = lgb.Booster(model_file=str(lgbm_txt))
    lgbm_features = json.loads(lgbm_features_json.read_text())
    if len(lgbm_features) != lgbm_model.num_feature():
        raise ValueError(
            f"Feature count mismatch: JSON has {len(lgbm_features)}, "
            f"LightGBM expects {lgbm_model.num_feature()}",
        )
    logger.info("  LightGBM loaded: %d features", len(lgbm_features))

    # ── TransVol ────────────────────────────────────────────────────
    from vol_model.transvol.config import TransVolConfig
    from vol_model.transvol.model import TransVolEncoder

    transvol_path = tv_models_dir / f"{month}_transvol.pt"
    ckpt = torch.load(str(transvol_path), map_location="cpu", weights_only=False)

    # The checkpoint format has evolved over time:
    # - Newest format stores a ``config`` object + ``normalizer_mean/std``
    #   for the joint feature+time normaliser
    # - Older format stores separate ``feat_mean/std`` and ``time_mean/std``
    # We accept both.
    if "config" in ckpt and ckpt["config"] is not None:
        config = ckpt["config"]
        if "normalizer_mean" in ckpt:
            feat_mean = np.asarray(ckpt["normalizer_mean"])
            feat_std = np.asarray(ckpt["normalizer_std"])
        else:
            feat_mean = np.asarray(ckpt.get("feat_mean", [])) if ckpt.get("feat_mean") is not None else np.zeros(0)
            feat_std = np.asarray(ckpt.get("feat_std", [])) if ckpt.get("feat_std") is not None else np.zeros(0)
        time_mean = np.asarray(ckpt.get("time_mean", [])) if ckpt.get("time_mean") is not None else np.zeros(0)
        time_std = np.asarray(ckpt.get("time_std", [])) if ckpt.get("time_std") is not None else np.zeros(0)
    else:
        feat_mean_raw = ckpt.get("feat_mean")
        feat_std_raw = ckpt.get("feat_std")
        if feat_mean_raw is None:
            raise ValueError(
                f"TransVol checkpoint {transvol_path} has neither 'config' "
                "nor 'feat_mean' — cannot infer n_features",
            )
        feat_mean = np.asarray(feat_mean_raw)
        feat_std = np.asarray(feat_std_raw)
        n_features = int(feat_mean.shape[0])
        config = TransVolConfig(n_features=n_features)
        time_mean = np.asarray(ckpt.get("time_mean", [])) if ckpt.get("time_mean") is not None else np.zeros(0)
        time_std = np.asarray(ckpt.get("time_std", [])) if ckpt.get("time_std") is not None else np.zeros(0)

    encoder = TransVolEncoder(config)
    encoder.load_state_dict(ckpt["model_state_dict"])
    encoder.eval()
    logger.info(
        "  TransVol loaded: n_features=%d, seq_len=%d",
        getattr(config, "n_features", 0),
        getattr(config, "seq_len", 0),
    )

    # ── DRL policy ──────────────────────────────────────────────────
    import faulthandler
    faulthandler.enable()  # segfault → traceback instead of silent crash

    from stable_baselines3 import SAC

    drl_policy_path = Path(drl_policy_path)
    if not drl_policy_path.exists():
        raise FileNotFoundError(f"DRL policy not found: {drl_policy_path}")
    drl_policy = SAC.load(str(drl_policy_path), device="cpu")
    logger.info("  DRL policy loaded: %s", drl_policy_path.name)

    # Smoke test: one forward pass must succeed (catches cross-platform pickle issues)
    _obs = np.zeros(drl_policy.observation_space.shape, dtype=np.float32)
    _act, _ = drl_policy.predict(_obs, deterministic=True)
    logger.info("  DRL smoke test OK (action=%s)", _act)

    # ── VecNormalize stats ──────────────────────────────────────────
    vn_mean: Any | None = None
    vn_var: Any | None = None
    vn_clip: float = 10.0

    vn_path = Path(drl_vecnormalize_path)
    if vn_path.exists():
        import pickle
        with open(vn_path, "rb") as f:
            payload = pickle.load(f)
        obs_rms = getattr(payload, "obs_rms", None)
        if obs_rms is not None:
            vn_mean = np.asarray(obs_rms.mean, dtype=np.float64)
            vn_var = np.asarray(obs_rms.var, dtype=np.float64)
            vn_clip = float(getattr(payload, "clip_obs", 10.0))
            logger.info(
                "  VecNormalize loaded: mean shape %s, clip=%.1f",
                vn_mean.shape, vn_clip,
            )
    else:
        logger.warning(
            "VecNormalize stats not found at %s — DRL policy will "
            "receive unnormalised observations (may be incorrect)",
            vn_path,
        )

    return ModelBundle(
        month_key=month,
        lgbm_model=lgbm_model,
        lgbm_features=lgbm_features,
        transvol_model=encoder,
        transvol_config=config,
        transvol_feat_mean=feat_mean,
        transvol_feat_std=feat_std,
        transvol_time_mean=time_mean,
        transvol_time_std=time_std,
        drl_policy=drl_policy,
        drl_vec_normalize_mean=vn_mean,
        drl_vec_normalize_var=vn_var,
        drl_vec_normalize_clip=vn_clip,
    )
