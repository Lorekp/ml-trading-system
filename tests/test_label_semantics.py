"""Regression tests locking in Stage 18 label semantics.

At any label key T, `load_labels(VolModelConfig()).ln_vol[T]` must equal
`log(vol of [T, T+freq))` — the 1-bar-ahead forecast target. This invariant
was silently broken on 2026-04-11 when the BTC TSRV parquet was migrated
from START-stamped to END-stamped without updating the downstream alignment
logic (full diagnosis in `vol_model/output/LABEL_SHIFT_AUDIT.md`). The fix
lives in `vol_model/labels.py::load_labels` (conditional -freq shift) +
`_PER_ASSET_CALIBRATION["BTCUSDT"]["tsrv_end_stamped"] = True`.

Rationale for these tests:
- If someone re-migrates TSRV or flips the config flag, these tests break
  loudly instead of silently producing a nowcast target.
- If the saved Stage 18 predictions parquet moves or the labels pipeline is
  further refactored, the "retrained model reproduces saved y_true" contract
  is preserved.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from vol_model.config import VolModelConfig
from vol_model.labels import load_labels


STAGE18_PREDS = Path("vol_model/output/BTCUSDT_pruned_svi_predictions.parquet")


@pytest.mark.skipif(
    not STAGE18_PREDS.exists(),
    reason=f"{STAGE18_PREDS} not available — skip byte-match regression",
)
def test_btc_labels_match_stage18_predictions():
    """Under default BTC config, load_labels().ln_vol byte-matches saved y_true.

    This is the definitive regression for Stage 18 label semantics. The saved
    predictions parquet was written 2026-04-11 11:15 against START-stamped
    TSRV labels. The post-migration end-stamped TSRV is compensated by the
    -freq shift in load_labels when tsrv_end_stamped=True.
    """
    cfg = VolModelConfig()
    assert cfg.asset == "BTCUSDT"
    assert cfg.tsrv_end_stamped is True, (
        "BTC per-asset default must set tsrv_end_stamped=True — TSRV is end-stamped "
        "since 2026-04-11 migration; see LABEL_SHIFT_AUDIT.md."
    )

    labels = load_labels(cfg)
    saved = pd.read_parquet(STAGE18_PREDS)
    common = labels.index.intersection(saved.index)
    assert len(common) > 140_000, f"common index too sparse: {len(common)}"

    diff = (labels.loc[common, "ln_vol"] - saved.loc[common, "y_true"]).abs()
    n_exact = int((diff == 0).sum())
    max_diff = float(diff.max())
    # Expect 100% exact match. A max|Δ| > 0 means label semantics drifted.
    assert n_exact == len(common), (
        f"Only {n_exact} / {len(common)} exact matches; max|Δ|={max_diff:.6e}. "
        f"Label semantics no longer match Stage 18 — investigate TSRV stamp "
        f"convention or labels.py."
    )


@pytest.mark.skipif(
    not STAGE18_PREDS.exists(),
    reason=f"{STAGE18_PREDS} not available — skip nowcast-detector regression",
)
def test_btc_labels_are_not_nowcast_when_end_stamped_flag_true():
    """With the fix ON, labels must NOT match the backward-looking nowcast.

    This complements the byte-match test by explicitly verifying we are in the
    correct target (forward forecast), not the incorrect target (backward
    nowcast). The nowcast bug signature is `current[T] == saved[T-15min]` for
    100% of rows — see LABEL_SHIFT_AUDIT.md Section 1.
    """
    cfg = VolModelConfig()  # tsrv_end_stamped=True via per-asset default
    labels = load_labels(cfg)
    saved = pd.read_parquet(STAGE18_PREDS)

    # Under the nowcast bug: labels at T == saved at T-15min. Under the fix,
    # that shifted-alignment should NOT match (mean |Δ| roughly equal to the
    # autocorrelation gap of the series, ~0.25-0.5 log units).
    freq = pd.Timedelta(seconds=cfg.freq_s)
    shifted_saved = saved.copy()
    shifted_saved.index = shifted_saved.index + freq
    common_shift = labels.index.intersection(shifted_saved.index)
    if len(common_shift) == 0:
        pytest.skip("no overlap after +freq shift")
    diff_shift = (labels.loc[common_shift, "ln_vol"]
                  - shifted_saved.loc[common_shift, "y_true"]).abs()
    frac_exact_nowcast = float((diff_shift < 1e-9).mean())
    # If > 99% of rows match the shifted (nowcast) target, the fix is broken.
    assert frac_exact_nowcast < 0.5, (
        f"Labels match nowcast target on {frac_exact_nowcast * 100:.1f}% of rows "
        f"— tsrv_end_stamped flag / labels.py shift is probably broken."
    )


def test_load_labels_end_stamp_shift_is_applied():
    """Directly verify the shift: labels.index[0] == TSRV.index[0] - freq under
    tsrv_end_stamped=True, and == TSRV.index[0] under False.

    This is a unit-level check that the flag controls the shift, independent
    of the saved Stage 18 predictions.
    """
    cfg_on = VolModelConfig(asset="BTCUSDT", tsrv_end_stamped=True)
    cfg_off = VolModelConfig(asset="BTCUSDT", tsrv_end_stamped=False)
    tsrv = pd.read_parquet(cfg_on.get_tsrv_path())

    labels_on = load_labels(cfg_on)
    labels_off = load_labels(cfg_off)

    freq = pd.Timedelta(seconds=cfg_on.freq_s)
    # With shift: labels index = tsrv index - freq
    # Without shift: labels index = tsrv index
    # Compare first 100 indices to avoid any filter-side edge cases
    on_prefix = labels_on.index[:100]
    off_prefix = labels_off.index[:100]

    # The shift should be exactly +freq between the two
    delta = (off_prefix - on_prefix).unique()
    assert len(delta) == 1, f"non-uniform shift: {delta}"
    assert delta[0] == freq, f"shift magnitude wrong: {delta[0]} != {freq}"


def test_eth_labels_no_shift_under_default():
    """ETH TSRV is still START-stamped; no shift should be applied."""
    cfg = VolModelConfig(asset="ETHUSDT")
    assert cfg.tsrv_end_stamped is False, (
        "ETH per-asset default must set tsrv_end_stamped=False — ETH TSRV is "
        "still start-stamped as of 2026-04-17 (verified via backfill_rq --verify-only)."
    )
    # Skip if ETH TSRV not present on this machine
    if not Path(cfg.get_tsrv_path()).exists():
        pytest.skip(f"{cfg.get_tsrv_path()} missing; skip ETH regression")
    labels = load_labels(cfg)
    tsrv = pd.read_parquet(cfg.get_tsrv_path())
    # Labels index must equal tsrv index (after the vol_ann>0 filter) — no shift
    # We check by verifying the first labels index matches the first tsrv index
    # where vol_ann > 0.
    tsrv_valid = tsrv[tsrv["vol_ann"] > 0]
    assert labels.index[0] == tsrv_valid.index[0], (
        f"ETH labels shifted unexpectedly: {labels.index[0]} vs {tsrv_valid.index[0]}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
