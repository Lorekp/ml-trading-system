"""Live feature pipeline — thin wrappers reusing the batch code.

Three reusable pieces (built in Phase 3):

- :mod:`deribit_paper.features.tsrv_live` — wraps
  ``TSRV.tsrv_series.compute_tsrv_window`` to compute the 12 TSRV
  fields for the trades in a single bar.
- :mod:`deribit_paper.features.svi_live` — wraps
  ``vol_arb.svi.calibrate_surface`` with caching + scheduling so the
  expensive per-hour calibration doesn't block the bar loop.
- :mod:`deribit_paper.features.straddle` — ATM-strike picker that
  queries Deribit's instrument list and chooses the call/put pair
  closest to the current index price within a DTE band.

The full 51-dimensional feature row assembly (needed by the hybrid
LightGBM + TransVol signal model) lives in Phase 4, bundled with
model inference in :mod:`deribit_paper.signal`.

- :mod:`deribit_paper.features.vpin_live` — streams Binance perp
  aggTrades via WebSocket and computes real-time bucketed VPIN
  (Easley-Prado-O'Hara 2012) for the DRL hedger observation.
"""

from deribit_paper.features.straddle import StrikePicker, StraddleSelection
from deribit_paper.features.svi_live import LiveSVICalibrator
from deribit_paper.features.tsrv_live import BarTSRV, compute_bar_tsrv
from deribit_paper.features.vpin_live import LiveVPINStream

__all__ = [
    "BarTSRV",
    "compute_bar_tsrv",
    "LiveSVICalibrator",
    "LiveVPINStream",
    "StrikePicker",
    "StraddleSelection",
]
