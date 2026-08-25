# Feature contribution contract

Every new feature family in this directory must satisfy the checklist below
**before** it enters a walk-forward training run. A feature that skips the
causality and provenance checks must not support a promoted QLIKE result.

## 1. Implementation checklist

- [ ] Pure function `compute_<family>_features(raw_df) -> pd.DataFrame` that
      is deterministic and has no I/O side effects (loads live in the
      per-asset loader, not the compute).
- [ ] Causal by construction. Rolling/EMA windows are trailing and use no
      row later than the decision timestamp; any current-bar usage is
      documented with a rationale.
- [ ] Output DataFrame has a `DatetimeIndex` matching the expected stamp
      convention of the downstream pipeline (END-stamped bars for TSRV-
      derived features; see `vol_model/labels.py`).
- [ ] Alignment shift (if any) applied in the per-asset *loader*, not the
      compute. The loader's docstring documents the shift direction and
      its rationale — compare with `run_walkforward_pruned_svi_har_pd.py::_load_har_pd`
      (-15m shift) and `run_walkforward_pruned_svi.py::_load_svi_atm_1d`
      (+15m shift).

## 2. Leakage-probe battery

Every new feature family MUST have a unit test that calls the gold-standard
causality probe:

```python
from vol_model.leakage_probes import (
    probe_causality_by_mutation,
    probe_current_bar_dependence,
)

def test_<family>_is_causal():
    """Byte-identical under mutation of all future raw rows."""
    raw = _load_raw_fixture()  # synthetic or real, at least 200 rows
    T = raw.index[len(raw) // 2]
    # Wrap the full load pipeline — compute + any alignment shift — into
    # a single callable. This is what the probe verifies end-to-end.
    def feature_fn(r):
        f = compute_<family>_features(r)
        f.index = f.index + ALIGNMENT_SHIFT   # if your family applies one
        return f

    r = probe_causality_by_mutation(feature_fn, raw, T)
    assert r.passed, r.detail


def test_<family>_uses_current_bar():
    """Not wastefully conservative — raw at T actually influences feature at T."""
    raw = _load_raw_fixture()
    T = raw.index[len(raw) // 2]
    def feature_fn(r): ...  # as above
    r = probe_current_bar_dependence(feature_fn, raw, T)
    # Fail this test if the alignment shift is eating the most recent bar
    # you could legally use (HAR-PD Session D lesson).
    assert r.passed, r.detail
```

A feature that fails `probe_causality_by_mutation` is **quarantined** —
do not include it in a training run. A feature that fails
`probe_current_bar_dependence` is a warning — you are wasting information
you could legally use, but the feature is not leaky.

## 3. Documentation checklist

Add a section to this README covering:

- **Raw source** — which parquet(s) the feature reads from, and the stamp
  convention of each source.
- **Alignment shift** — direction (+ or -), magnitude, and rationale.
- **Expected NaN pattern** — warmup rows, missing-data policy. Target is
  < 5% of rows dropped at training time.
- **Reference** — academic citation (arxiv/DOI) if the feature family
  comes from a paper. Fabricated citations are unacceptable.
- **Leakage-probe test path** — `tests/test_features_<family>.py::test_<family>_is_causal`.

## 4. Registration

To make the family available to the walk-forward trainer:

1. Add the feature list as a module-level constant in
   `vol_model/run_walkforward_<variant>.py` (e.g. `ADD_<FAMILY>_FEATURES`).
2. Implement a per-asset loader `_load_<family>(asset)` that reads the
   sidecar parquet, applies any alignment shift, and returns a DataFrame.
3. Concatenate the loader's output inside the training loop's
   `X_month` assembly.
4. Drop NaN rows for the new columns via `dropna(subset=ADD_<FAMILY>_FEATURES)`.
5. Or, for gauntlet-only registration, extend `TIER_FEATURE_MAP` in
   `vol_model/signal_research.py` with a new tier key.

## 5. Existing families (reference examples)

| Family | Compute | Loader | Shift | Probe test |
|---|---|---|---|---|
| HAR-PD | `har_pd.py::compute_har_pd_features` | `run_walkforward_pruned_svi_har_pd.py::_load_har_pd` | -15m | `test_walkforward_pruned_svi_har_pd.py::test_no_lookahead_after_shift` |
| SVI-ATM | (sidecar) | `run_walkforward_pruned_svi.py::_load_svi_atm_1d` | +15m | `tests/test_svi_spot_feature.py` |
| Microstructure (OFI etc.) | `microstructure.py` | `engineer.py` | 0 | **Not yet covered by the required causal probe** |

Candidate implementations currently present but not promoted:

- SVI surface PCA (`svi_surface.py`)
- Hawkes-OFI branching ratio (`hawkes_ofi.py`)
- Rough-Heston spot volatility (`rough_heston.py`)

Their presence in the package is not evidence that they passed the contract.
