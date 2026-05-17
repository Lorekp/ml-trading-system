"""Per-bar feature trace — cycle-33 (2026-05-13).

Captures every intermediate value the live signal pipeline produces
for a single bar, so the offline backtester's parallel trace can be
diff'd field-by-field by ``scripts/feature_trace_diff.py`` to pinpoint
where the residual ~3.78vp Δedge between live and offline originates.

The schema is a flat dataclass with eight named checkpoint groups:

    C0  raw bar inputs
    C1  TSRV outputs
    C2  SVI surface state (which bucket, params)
    C3  SVI evaluation at the ATM strike (iv chain)
    C4  77-named-feature LGBM input vector (dict)
    C5  TransVol embedding (16 floats)
    C6  LGBM raw y_pred (ln-vol space)
    C7  Aligner β* + exp() → rv_pred_pct
    C8  Edge + signal decision

The comparator imports :data:`TOLERANCES` so the same per-field
rtol/atol are used on both sides; never duplicate.

Wire-up:
    trace = BarTrace(bar_close_ts=bar.timestamp)
    # ... predictor / feature_assembler mutate `trace` if non-None ...
    bar_trace_writer.record(trace)

When the writer is ``enabled=False`` it is a no-op (zero file IO);
production engines run with ``feature_trace_path=None`` and pay
nothing for the instrumentation hooks.
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


# Per-field tolerances. Comparator imports this dict; both pipelines
# agree by construction. Keys are checkpoint-prefixed; ``default`` is
# the fallback. Tightened later checkpoints (C4..C8) on the principle
# that everything earlier must already be identical or the divergence
# would surface there first.
TOLERANCES: dict[str, dict[str, float]] = {
    "default":   {"rtol": 1e-3, "atol": 1e-6},
    # C0 raw inputs — index_price_close is sub-bar-sampled differently
    # on each side (live: WS deribit_price_index push at bar close;
    # offline: vol_arb parquet's per-bar index). They track the same
    # series but never match to the penny — a 0.5% band confirms
    # "same bar" without masking the genuine downstream divergence.
    "c0":        {"rtol": 5e-3, "atol": 1.0},
    # C1 TSRV — derived from trade streams which may have tiny windowing differences.
    "c1":        {"rtol": 1e-3, "atol": 1e-6},
    # C2 SVI surface params — exact match if same bucket_ts; comparator flags bucket_ts drift separately.
    "c2":        {"rtol": 1e-6, "atol": 1e-8},
    # C3 SVI eval at ATM.
    "c3":        {"rtol": 1e-4, "atol": 1e-6},
    # C4 77-feature vector — LGBM deterministic float64.
    "c4":        {"rtol": 1e-5, "atol": 1e-6},
    # C5 TransVol embedding — Torch float32, marginal drift acceptable.
    "c5":        {"rtol": 1e-3, "atol": 1e-5},
    # C6 LGBM raw y_pred.
    "c6":        {"rtol": 1e-5, "atol": 1e-6},
    # C7 β* exact (same JSON file), rv_pred_pct rtol=1e-4.
    "c7":        {"rtol": 1e-4, "atol": 1e-6},
    # C8 signal decision — edge to rtol=1e-4, signal_final must match exactly.
    "c8":        {"rtol": 1e-4, "atol": 1e-6},
}


@dataclass
class BarTrace:
    """Per-bar intermediate values, mutated in-place through the pipeline.

    Every field except ``bar_close_ts`` defaults to ``None``; only the
    checkpoints actually executed for a given bar are populated. The
    JSONL serialiser drops ``None`` fields so empty checkpoints don't
    pollute downstream diff output.

    ``side`` is set to ``"live"`` by the engine; the offline trace
    builder sets ``"offline"`` for symmetry.
    """
    bar_close_ts: pd.Timestamp
    side: str = "live"

    # ── C0 raw inputs ─────────────────────────────────────────────
    bar_open_ts: pd.Timestamp | None = None
    index_price_close: float | None = None
    index_price_open: float | None = None
    n_perp_trades: int | None = None
    n_binance_fut_trades: int | None = None
    bar_volume_usd: float | None = None
    dollar_bars_in_tsrv_window: int | None = None

    # ── C1 TSRV ───────────────────────────────────────────────────
    tsrv: float | None = None
    rv_fast: float | None = None
    rv_slow: float | None = None
    bpv: float | None = None
    medrv: float | None = None
    continuous_var: float | None = None
    jump_var: float | None = None
    jump_fraction: float | None = None
    vol_ann_continuous: float | None = None
    vol_ann: float | None = None

    # ── C2 SVI surface state ──────────────────────────────────────
    svi_bucket_ts: pd.Timestamp | None = None
    svi_age_seconds: float | None = None
    svi_expiry_dt: pd.Timestamp | None = None
    svi_a: float | None = None
    svi_b: float | None = None
    svi_rho: float | None = None
    svi_m: float | None = None
    svi_sigma: float | None = None
    svi_T_years: float | None = None
    svi_F: float | None = None
    svi_rmse: float | None = None
    svi_n_strikes: int | None = None
    svi_level: float | None = None

    # ── C3 SVI eval @ ATM ─────────────────────────────────────────
    atm_strike: float | None = None
    svi_iv_at_atm_pct: float | None = None
    fallback_iv_pct: float | None = None
    fallback_dvol_pct: float | None = None
    iv_used_pct: float | None = None
    iv_source: str | None = None

    # ── C4 LGBM input vector ──────────────────────────────────────
    feature_vector: dict[str, float] | None = None

    # ── C5 TransVol embedding ─────────────────────────────────────
    transvol_embedding: list[float] | None = None
    transvol_status: str | None = None  # "computed" / "warmup_fallback" / "frozen_baseline"
    transvol_window_size: int | None = None

    # ── C6 LGBM raw output ────────────────────────────────────────
    y_pred_raw: float | None = None
    lgbm_model_sha256: str | None = None

    # ── C7 Aligner + exp ──────────────────────────────────────────
    aligner_month_key: str | None = None
    beta_star: float | None = None
    y_pred_aligned: float | None = None
    rv_pred_pct: float | None = None

    # ── C8 Signal decision ────────────────────────────────────────
    edge_vol_pts: float | None = None
    base_spread_iv: float | None = None
    spread_jump_k: float | None = None
    half_spread: float | None = None
    effective_long_thresh: float | None = None
    effective_short_thresh: float | None = None
    signal_raw: int | None = None
    signal_final: int | None = None
    filter_reason: str | None = None


def _coerce(v: Any) -> Any:
    """Make a field JSONL-safe.

    - ``pd.Timestamp`` → ISO 8601 string
    - ``float('nan')`` / ``inf`` → ``None`` (drop from output)
    - ``numpy`` scalars → native Python via ``.item()`` where possible
    - ``dict``/``list`` → recursively coerced
    """
    if v is None:
        return None
    if isinstance(v, pd.Timestamp):
        return v.isoformat()
    if isinstance(v, float):
        if not math.isfinite(v):
            return None
        return v
    if isinstance(v, dict):
        return {k: _coerce(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_coerce(x) for x in v]
    # numpy scalar fallback
    item = getattr(v, "item", None)
    if callable(item):
        try:
            return _coerce(item())
        except Exception:  # noqa: BLE001
            pass
    return v


def trace_to_jsonl_dict(trace: BarTrace) -> dict[str, Any]:
    """Serialise a :class:`BarTrace` to a dict ready for ``json.dumps``.

    Drops any field whose coerced value is ``None`` (keeps the JSONL
    line compact when only a subset of checkpoints was reached — e.g.,
    a flat signal that bailed before SVI eval).

    Always retains ``bar_close_ts`` and ``side`` so the comparator can
    join.
    """
    raw = asdict(trace)
    out: dict[str, Any] = {}
    for k, v in raw.items():
        coerced = _coerce(v)
        if coerced is None and k not in {"bar_close_ts", "side"}:
            continue
        out[k] = coerced
    return out


class BarTraceWriter:
    """Append-only JSONL writer for :class:`BarTrace` rows.

    Crash-safe by virtue of line-level flushes (each ``record`` opens,
    appends one line, closes). The trace is typically a few KB per bar
    — write amplification is negligible at 15-minute bar cadence.

    Construct with ``path=None`` (or ``enabled=False``) for a no-op
    instance — production engines do this by default; the operator
    opts in via the engine config field ``feature_trace_path``.
    """

    def __init__(
        self,
        path: str | Path | None,
        *,
        enabled: bool | None = None,
    ) -> None:
        self.path: Path | None = Path(path) if path else None
        if enabled is None:
            self.enabled = self.path is not None
        else:
            self.enabled = bool(enabled and self.path is not None)
        if self.enabled and self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            logger.info(
                "BarTraceWriter enabled — appending per-bar JSONL → %s",
                self.path,
            )

    def record(self, trace: BarTrace) -> None:
        if not self.enabled or self.path is None:
            return
        try:
            payload = trace_to_jsonl_dict(trace)
            line = json.dumps(payload, separators=(",", ":"))
            with self.path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception:  # noqa: BLE001
            logger.exception(
                "BarTraceWriter.record failed at %s",
                trace.bar_close_ts,
            )
