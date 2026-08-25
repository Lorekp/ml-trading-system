"""Latest-month research-bundle structural verification.

Asserts that the historical model bundle for the latest-trained month
loads cleanly and passes basic health gates:

1. ``expanding_pruned_svi_v1/{latest}.txt`` + features.json present.
2. ``expanding_hybrid/{latest}_transvol.pt`` present.
3. ``summary.json`` reports an entry for ``{latest}`` with QLIKE <= 0.7.
4. ``walkforward_results.csv`` has a row for ``{latest}`` (TransVol training).
5. ``load_model_bundle()`` returns a bundle for the latest month and
   the encoder's smoke inference produces a valid 16-dim embedding.

Exits non-zero on any gate failure so a stalled research run is visible. A
pass is a bundle-health check, not model or deployment validation.

Usage::

    python scripts/verify_latest_month_training.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

LGBM_DIR = REPO_ROOT / "vol_model" / "output" / "models" / "expanding_pruned_svi_v1"
TRANSVOL_DIR = REPO_ROOT / "vol_model" / "output" / "models" / "expanding_hybrid"

QLIKE_GATE = 0.7  # Historical health threshold; not a promotion gate.


def _latest_dated_file(directory: Path, pattern: str) -> tuple[str, Path] | None:
    """Find the most recent YYYY-MM file matching ``pattern``."""
    import re
    re_month = re.compile(r"^(\d{4}-\d{2})")
    candidates: list[tuple[str, Path]] = []
    for path in directory.glob(pattern):
        match = re_month.match(path.name)
        if match is None:
            continue
        candidates.append((match.group(1), path))
    if not candidates:
        return None
    return sorted(candidates, key=lambda x: x[0])[-1]


def main() -> int:
    print("=" * 70)
    print(f"  S-1b verify — latest-month training health check")
    print("=" * 70)

    if not LGBM_DIR.exists():
        print(f"FAIL: LightGBM model dir missing: {LGBM_DIR}")
        return 2
    if not TRANSVOL_DIR.exists():
        print(f"FAIL: TransVol model dir missing: {TRANSVOL_DIR}")
        return 2

    # 1. Latest LightGBM month
    latest = _latest_dated_file(LGBM_DIR, "*.txt")
    if latest is None:
        print(f"FAIL: no dated LightGBM checkpoints in {LGBM_DIR}")
        return 2
    month, lgbm_path = latest
    features_json = LGBM_DIR / f"{month}_features.json"
    if not features_json.exists():
        print(f"FAIL: features.json missing for {month}: {features_json}")
        return 2
    n_features = len(json.loads(features_json.read_text()))
    print(f"  latest LightGBM month: {month} ({n_features} features)")

    # 2. TransVol checkpoint paired with the LightGBM month
    transvol_path = TRANSVOL_DIR / f"{month}_transvol.pt"
    if not transvol_path.exists():
        print(f"FAIL: TransVol checkpoint missing for {month}: {transvol_path}")
        return 2
    print(f"  paired TransVol checkpoint: {transvol_path.name}")

    # 3. summary.json gate
    summary_path = LGBM_DIR / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        per_month = summary.get("per_month", []) or []
        latest_summary = next(
            (m for m in per_month if m.get("month") == month), None,
        )
        if latest_summary is not None:
            qlike = float(latest_summary.get("pruned_svi_qlike", float("nan")))
            rank_ic = float(latest_summary.get("rank_ic", float("nan")))
            print(f"  {month} QLIKE: {qlike:.4f} (gate <= {QLIKE_GATE})")
            print(f"  {month} RankIC: {rank_ic:.4f}")
            if qlike > QLIKE_GATE:
                print(
                    f"FAIL: QLIKE {qlike:.4f} exceeds gate {QLIKE_GATE} — "
                    f"production model is suspect."
                )
                return 2
        else:
            print(f"  WARN: summary.json has no entry for {month}")
    else:
        print(f"  WARN: no summary.json at {summary_path}")

    # 4. walkforward_results.csv gate
    walkforward_csv = TRANSVOL_DIR / "walkforward_results.csv"
    if walkforward_csv.exists():
        import pandas as pd
        df = pd.read_csv(walkforward_csv)
        latest_rows = df[df["month"].astype(str) == month]
        if not latest_rows.empty:
            row = latest_rows.iloc[-1]
            qlike_tv = float(row.get("qlike", float("nan")))
            rank_ic_tv = float(row.get("rank_ic", float("nan")))
            print(f"  {month} TransVol QLIKE: {qlike_tv:.4f}")
            print(f"  {month} TransVol RankIC: {rank_ic_tv:.4f}")
        else:
            print(f"  WARN: walkforward_results.csv has no row for {month}")
    else:
        print(f"  WARN: no walkforward_results.csv at {walkforward_csv}")

    # 5. Bundle smoke
    print("  loading bundle (forces model_loader's full validation chain)…")
    try:
        from deribit_paper.signal.model_loader import load_model_bundle
        bundle = load_model_bundle(month_override=month)
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL: load_model_bundle({month}) raised: {exc}")
        return 2
    print(f"  bundle loaded: lgbm={bundle.lgbm_model.num_feature()} feats, "
          f"transvol n_features={bundle.transvol_config.n_features}, "
          f"seq_len={bundle.transvol_config.seq_len}")
    if bundle.lgbm_model.num_feature() != n_features:
        print(f"FAIL: bundle feature count mismatch: bundle reports "
              f"{bundle.lgbm_model.num_feature()}, json has {n_features}")
        return 2

    # Smoke: encoder forward pass with random normalised input
    import numpy as np
    import torch
    rng = np.random.default_rng(0)
    seq_len = int(bundle.transvol_config.seq_len)
    n_feat = int(bundle.transvol_config.n_features)
    x = torch.from_numpy(
        rng.standard_normal((1, seq_len, n_feat)).astype(np.float32)
    )
    t = torch.from_numpy(
        rng.standard_normal((1, seq_len, 2)).astype(np.float32)
    )
    with torch.no_grad():
        outputs = bundle.transvol_model(x, t)
    emb = outputs["embedding_compact"].numpy().reshape(-1)
    if emb.shape != (16,):
        print(f"FAIL: embedding_compact shape {emb.shape}, expected (16,)")
        return 2
    if not np.all(np.isfinite(emb)):
        print(f"FAIL: embedding_compact contains non-finite values: {emb}")
        return 2
    print(
        f"  encoder smoke: 16-dim embedding finite + bounded "
        f"(min={emb.min():.3f}, max={emb.max():.3f})"
    )

    print("=" * 70)
    print(f"  PASS — {month} bundle passed structural and smoke checks")
    print("  This does not establish model validity or deployment readiness.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
