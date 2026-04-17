"""Signal generation + DRL hedging inference.

Phase 4 deliverables:

- :mod:`deribit_paper.signal.model_loader` — loads LightGBM (latest
  walk-forward month), TransVol encoder (matching month), and the
  production DRL policy + VecNormalize stats.
- :mod:`deribit_paper.signal.feature_assembler` — builds the 51-dim
  feature row at bar close.  Uses the historical features parquet as
  a rolling buffer, overrides the fresh fields (TSRV, jump_fraction,
  n_trades, duration_s, vol_ann) with live values, runs
  ``apply_windowing`` to regenerate lag/EMA columns, and extracts the
  latest row.  Pragmatic MVP: the "hard" base features
  (noise_level, parkinson_1d, ffd_rv, vpin, vov, imbalance,
  bpv_tsrv_ratio, time_to_funding) are taken from the most recent
  historical row and not recomputed live.  See Phase 7 for the
  parity validation that decides whether this is acceptable.
- :mod:`deribit_paper.signal.predictor` — per-bar signal generator:
  assembles feature row → LightGBM predict → edge vs iv_used →
  long/short/flat via the same threshold logic as the backtester.
- :mod:`deribit_paper.signal.drl_hedger_live` — NEW scalar-inference
  wrapper around the production DRL policy.  Maintains per-trade
  state (inventory, frozen SVI slice) and exposes
  ``begin_trade`` / ``step`` / ``close_trade`` — the live equivalent
  of ``DRLHedger.simulate()``'s vectorised batch API.
"""

from deribit_paper.signal.drl_hedger_live import LiveDRLHedger
from deribit_paper.signal.feature_assembler import FeatureAssembler, FeatureRow
from deribit_paper.signal.model_loader import ModelBundle, load_model_bundle
from deribit_paper.signal.predictor import Predictor, Signal

__all__ = [
    "FeatureAssembler",
    "FeatureRow",
    "LiveDRLHedger",
    "ModelBundle",
    "load_model_bundle",
    "Predictor",
    "Signal",
]
