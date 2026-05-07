"""Live HAR-PD path-dependent feature computation for the paper engine.

C (parity 2026-05-07) — Tier 3 model alignment. The live engine pre-this
loaded the 44-feature ``_pruned_svi_v1`` model while the V2 backtest
trained / aligner-β*-calibrated the 50-feature ``_pruned_svi_har_pd_v1``
variant. The 6 extra features are HAR-PD path-dependent kernels
documented in ``vol_model/features/har_pd.py`` (Xiao et al. 2025,
arxiv:2503.00851):

    R1(t; alpha) = sum_{k=1..K} w_k(alpha) * signed_flow_{t-k}
    R2(t; beta)  = sum_{k=1..K} w_k(beta)  * realized_var_{t-k}

with the exponential kernel ``w_k(p) = p * (1-p)^(k-1)`` normalised so
``sum w_k = 1``, ``signed_flow = sqrt(rs_up) - sqrt(rs_down)``,
``realized_var = rv_ann``, ``K = 96 bars (= 24h @ 15m)``.

Lifted byte-for-byte from the offline ``vol_model.features.har_pd``:
``ALPHA_GRID``, ``BETA_GRID``, ``K_BARS_DEFAULT``,
``_exponential_kernel_weights``. The only difference is that the offline
implementation uses ``np.convolve`` over a full Series; the live path
uses a per-bar dot product against the kernel-reversed buffer (cheap,
O(K)). Both produce identical values to within float rounding.

Strict-past contract: ``compute_features(state)`` reads only from the
buffer; the caller must call ``append(...)`` AFTER consuming the
features so that the current bar's TSRV does NOT leak into the
features it would feed into the LightGBM input. Verified by the
byte-identity test ``tests/test_har_pd_live_byte_identity.py``.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np


# ── Constants (lifted from vol_model.features.har_pd, single source of truth)


ALPHA_GRID: tuple[float, ...] = (0.05, 0.1, 0.3)
BETA_GRID: tuple[float, ...] = (0.05, 0.1, 0.3)
K_BARS_DEFAULT: int = 96
HAR_PD_RAW_FEATURES: tuple[str, ...] = (
    "har_pd_R1_a005", "har_pd_R1_a01", "har_pd_R1_a03",
    "har_pd_R2_b005", "har_pd_R2_b01", "har_pd_R2_b03",
)


def _exponential_kernel_weights(alpha: float, k_bars: int = K_BARS_DEFAULT) -> np.ndarray:
    """Exponential decay weights ``w_k = alpha*(1-alpha)^(k-1)`` for k=1..k_bars.

    Normalised so ``sum(w_k) == 1`` (removes dependence on the
    sum-truncation). Lifted byte-for-byte from
    ``vol_model.features.har_pd._exponential_kernel_weights`` so the
    offline parquet and live computation share weights exactly.
    """
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    k = np.arange(1, k_bars + 1)
    w = alpha * (1 - alpha) ** (k - 1)
    return w / w.sum()


@dataclass
class HARPDState:
    """Circular buffer of past (rs_up, rs_down, rv_ann) for HAR-PD live.

    Maintains exactly the last ``k_bars`` bars' values. The buffer is
    ordered oldest-first → newest-last so:

    * ``buf[0]`` is the bar at ``t - k_bars`` (oldest in window)
    * ``buf[-1]`` is the bar at ``t - 1`` (most recent in strict past)

    The current bar (``t``) is NOT in the buffer until ``append(...)``
    is called AFTER consuming the features.

    Pre-warmed instances accept an iterable of (rs_up, rs_down, rv_ann)
    triples to avoid the 96-bar cold-start at engine launch.
    """

    k_bars: int = K_BARS_DEFAULT
    _rs_up: deque[float] = field(init=False, default=None)  # type: ignore[assignment]
    _rs_down: deque[float] = field(init=False, default=None)  # type: ignore[assignment]
    _rv_ann: deque[float] = field(init=False, default=None)  # type: ignore[assignment]
    # Pre-computed weights cached at construction so each bar costs
    # only the dot product (no per-bar exp/clip computation).
    _alpha_weights_rev: tuple[np.ndarray, ...] = field(init=False, default=())
    _beta_weights_rev: tuple[np.ndarray, ...] = field(init=False, default=())

    def __post_init__(self) -> None:
        self._rs_up = deque(maxlen=self.k_bars)
        self._rs_down = deque(maxlen=self.k_bars)
        self._rv_ann = deque(maxlen=self.k_bars)
        # Weights reversed at construction so we can dot against the
        # oldest-first buffer directly. ``weights[0]`` (largest weight)
        # is for the MOST RECENT past bar (buf[-1]), so reversed,
        # ``weights[::-1][0]`` aligns with ``buf[0]`` (oldest).
        # Test ``test_weight_orientation_matches_offline`` locks this.
        self._alpha_weights_rev = tuple(
            _exponential_kernel_weights(a, k_bars=self.k_bars)[::-1]
            for a in ALPHA_GRID
        )
        self._beta_weights_rev = tuple(
            _exponential_kernel_weights(b, k_bars=self.k_bars)[::-1]
            for b in BETA_GRID
        )

    @property
    def is_warm(self) -> bool:
        """True once ``k_bars`` strict-past entries have been observed."""
        return len(self._rs_up) >= self.k_bars

    @property
    def n_observed(self) -> int:
        """Count of bars in the buffer (≤ ``k_bars``)."""
        return len(self._rs_up)

    def append(self, rs_up: float, rs_down: float, rv_ann: float) -> None:
        """Add the latest bar's TSRV scalars to the buffer.

        Call AFTER ``compute_features(...)`` has been read out for the
        current bar so the strict-past contract is preserved (current
        bar must not appear in its own HAR-PD inputs).
        """
        self._rs_up.append(float(rs_up))
        self._rs_down.append(float(rs_down))
        self._rv_ann.append(float(rv_ann))

    def prewarm(self, rows: Iterable[tuple[float, float, float]]) -> int:
        """Bulk-feed historical (rs_up, rs_down, rv_ann) rows.

        Used at engine startup so the first live bar can produce
        non-NaN HAR-PD features instead of waiting 96 bars (24h) of
        live data accumulation. Returns the number of rows appended.
        """
        n = 0
        for rs_up, rs_down, rv_ann in rows:
            self.append(rs_up, rs_down, rv_ann)
            n += 1
        return n

    def compute_features(self) -> dict[str, float]:
        """Return the 6 HAR-PD features computed from the strict-past buffer.

        Returns NaN for all features when the buffer hasn't yet
        accumulated ``k_bars`` entries — caller treats this as warmup
        and should fall back to the historical baseline row (or refuse
        to trade if the manifest validation requires HAR-PD to be live).
        """
        if not self.is_warm:
            return {name: float("nan") for name in HAR_PD_RAW_FEATURES}

        rs_up = np.fromiter(self._rs_up, dtype=np.float64, count=self.k_bars)
        rs_down = np.fromiter(self._rs_down, dtype=np.float64, count=self.k_bars)
        rv_ann = np.fromiter(self._rv_ann, dtype=np.float64, count=self.k_bars)

        signed_flow = (
            np.sqrt(np.clip(rs_up, 0.0, None))
            - np.sqrt(np.clip(rs_down, 0.0, None))
        )

        out: dict[str, float] = {}
        for w_rev, suffix in zip(
            self._alpha_weights_rev, ("a005", "a01", "a03"),
        ):
            out[f"har_pd_R1_{suffix}"] = float(np.dot(w_rev, signed_flow))
        for w_rev, suffix in zip(
            self._beta_weights_rev, ("b005", "b01", "b03"),
        ):
            out[f"har_pd_R2_{suffix}"] = float(np.dot(w_rev, rv_ann))
        return out


__all__ = [
    "ALPHA_GRID",
    "BETA_GRID",
    "K_BARS_DEFAULT",
    "HAR_PD_RAW_FEATURES",
    "HARPDState",
]
