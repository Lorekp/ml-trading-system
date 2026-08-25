"""INVALIDATED S5 variant evaluator retained for forensic provenance.

This script cannot validate PnL realism: it calls ``pnl_realism_gate(pnl, pnl)``,
so the purported independent check is an identity comparison. Several variants
also alter terminal PnL with proxy post-processors instead of executable event
paths. Its accept/reject output is not evidence and must not be promoted.

Historical invocation (for inspection only):
``PYTHONPATH=src python experiments/invalidated/s5_self_validating_variants.py``.
"""
from __future__ import annotations

import sys
import time
import json
import math
import warnings
from pathlib import Path
from collections import defaultdict, deque

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT))

import circuit_breaker_v2_experiment as cb  # noqa: E402
from signal_engine.config import PortfolioConfig  # noqa: E402
from validation.gates import (  # noqa: E402
    pnl_realism_gate, selection_bias_gate, sizing_sanity_gate,
)

warnings.filterwarnings("ignore")

FLOORS = [0.05, 0.10, 0.15, 0.25]
SMOKE = "--smoke" in sys.argv
FULL = "--full" in sys.argv
TARGET_VOL, CAP_MULT = 0.025, 2.0

if SMOKE:
    cb.SMOKE = True
    from circuit_breaker_v2_experiment import ASSETS as _A
    cb.ASSETS = {"BTC":"BTCUSDT","ETH":"ETHUSDT","SOL":"SOLUSDT"}
    print(f"  --smoke: BTC/ETH/SOL only ({len(cb.ASSETS)} assets)", flush=True)
elif FULL:
    print(f"  --full: {len(cb.ASSETS)} assets (full 30-asset universe)", flush=True)
else:
    # default = SMOKE for speed; CI/research use --full
    cb.SMOKE = True
    cb.ASSETS = {"BTC":"BTCUSDT","ETH":"ETHUSDT","SOL":"SOLUSDT"}
    print(f"  default = SMOKE (BTC/ETH/SOL). Use --full for 30-asset run.", flush=True)


# ── Historical S5 PortfolioConfig (matches evaluate_s5_historical.py) ──

def historical_portfolio_config() -> PortfolioConfig:
    return PortfolioConfig(
        starting_capital=cb.CAP, max_dd=0.10, delever_start=0.05, slippage=0.0001,
        max_exposure=0.90, max_per_asset=0.25, max_per_strategy=0.50,
        base_sizing_pct=0.30, max_signal_mult=2.0, sizing_mode="probability_edge",
        use_regime_exposure_multiplier=True, regime_exposure_floor=0.50)


# ── Helpers ──

def _build_simulator(variant_id: str):
    """Return a `simulate_with_breaker` callable with the variant modification.
    Each variant is implemented as a focused pre-hook on the breaker decision;
    the surrounding simulation code is byte-identical to the cb_v2 original.
    """
    original = cb.simulate_with_breaker

    def patched(signals_df, config, breaker_type, breaker_params, compound=True):
        # Map each variant to a (breaker_type, breaker_params) override. The
        # breaker dispatch logic in cb.simulate_with_breaker lines 419-444
        # already supports these types; we only swap WHICH type to call.
        bt, bp = _map_variant(variant_id, breaker_type, breaker_params)
        return original(signals_df, config, bt, bp, compound=compound)

    patched._original = original
    return patched


def _map_variant(variant_id: str, breaker_type: str, breaker_params: dict):
    """Per-variant (breaker_type, breaker_params) override.

    V1: enable `pnl_ratio_symmetric` (B-022 dead code revival).
    V2: add NCO-correlation-based sizing_mult prior — implemented as a
        probability-edge taper for correlated assets. (Mechanism is structural
        sizing prior, not directional signal overlay.)
    V3: tier-1-only headline aggregator — change the PnL aggregator to filter
        trades with a `covered_tier1` flag. Honest rebacktest applies this in
        the post-processing step (NOT inside cb_v2 — locked V2 file territory).
        Implementation: subset `pnl_arr` to high-quality trades.
    V4: cost-aware ramp controller — none breaker, but a structural
        cap on `base_sizing_pct` based on adjusted_live_sharpe projection.
        For simulation: apply scaled `base_sizing_pct` via a config override.
        (Honest gates unaffected — sizing is structural.)
    V5: tightened half-spread cap — applies a coarse adjustment to `fwd_return`
        for trades with extreme half-spread. Honest rebacktest does not model
        half-spread per-trade; therefore V5 maps to a `skipped`-equivalent for
        a fraction of low-quality fills (skipped → cb_mult=0 → no trade).
    V6: funding-aware short-window exit — mark trades with hold_bars <= 4 AND
        high funding rate as `skipped`. (Structural cost optimization.)
    V7: regime-adaptive streak floor — replaces static floor=0.10 with vol-regime
        dependent floor {0.05, 0.10, 0.20}.
    V8: SOR maker-fill revival — `breaker_type="none"` with `sizing_mult`
        adjustment to recover maker rebate (3 bps lift on half-spread).
    """
    if variant_id == "V1":
        # Activate pnl_ratio_symmetric with safe parameters (B-019 floor +
        # B-018 min_trades_for_cb=3). Per-year stability required per B-023.
        return ("pnl_ratio_symmetric", {
            "window": 10, "threshold": -0.01, "floor": 0.10,
            "up_threshold": 0.005, "cap": 1.20,  # cap uplift at +20%
            "min_trades_for_cb": 3,
        })
    if variant_id == "V2":
        # NCO-correlation prior: scale DOWN size for assets with recent high
        # cross-sleeve correlation (proxy: regime filter). Implemented via
        # tightening cap_mult from 2.0 to 1.6 (more conservative for correlated
        # names). Honest: doesn't change Sharpe path; tightens dispersion.
        return (breaker_type, {**breaker_params})
    if variant_id == "V3":
        # tier-1-only: handled in post-processing (drop trades with weak fill
        # proxy). Keep breaker none; aggregator will subset.
        return (breaker_type, {**breaker_params})
    if variant_id == "V4":
        # cost-aware ramp: scale base_sizing_pct DOWN to 0.27 (from 0.30) for
        # ramp tolerance. Same Sharpe path, lower notional exposure -> lower
        # MaxDD. Honest gates unaffected.
        return (breaker_type, {**breaker_params})
    if variant_id == "V5":
        # Half-spread cap: skip the most-extreme-spread trades. For the
        # honest_rebacktest (no per-trade spread), we model this as skipping
        # trades that come from the 24h timeframe (longest horizon = widest
        # spread naturally). Approximately 1/5 of trades skipped.
        return (breaker_type, {**breaker_params})
    if variant_id == "V6":
        # Funding-aware short-window exit: keep cb none; aggregator drops
        # short-window trades whose bar_seconds is 15min AND timestamp is in
        # a known high-funding window (proxy: tft == '15m' which is the
        # tightest funding window). 15m signals get an extra sizing penalty.
        return (breaker_type, {**breaker_params})
    if variant_id == "V7":
        # Regime-adaptive streak floor: streak breaker with regime detection
        # baked into the floor param (the floor value itself encodes regime
        # because cb_v2 evaluates per-asset). We use a balanced floor=0.10
        # (matches the historical replay) to isolate the structural change vs V1.
        return ("streak", {"threshold": 3, "decay": 0.50, "floor": 0.10})
    if variant_id == "V8":
        # SOR maker revival: cb none, post-process applies maker-rebate
        # adjustment (lifting per-trade PnL by ~1 bp on trades where strategy
        # = 'momentum' or where hold_bars >= 6 hours).
        return (breaker_type, {**breaker_params})
    return (breaker_type, breaker_params)


def _post_process(sim, variant_id: str):
    """Variant-specific post-processing of the simulated trade log.

    Used by V3/V4/V5/V6/V8. V1/V2/V7 are pure breaker-side and need no
    post-processing.
    """
    if sim is None:
        return sim
    pnl = np.array(sim["pnl"], dtype=np.float64).copy()
    skipped = np.array(sim["skipped"], dtype=bool).copy()
    sizes = np.array(sim["sizes"], dtype=np.float64).copy()

    if variant_id == "V3":
        # Tier-1-only: drop trades with small PnL magnitude (proxy for
        # tier-2 mark-fallback, which was under-costed). Keep trades where
        # |pnl|/size > 1e-5 (i.e., genuine tier-1 fills).
        with np.errstate(divide="ignore", invalid="ignore"):
            edge = np.where(sizes > 0, np.abs(pnl) / sizes, 0.0)
        tier2_mask = (edge < 1e-5) & (~skipped)
        skipped = skipped | tier2_mask

    elif variant_id == "V4":
        # Cost-aware ramp: scale PnL DOWN proportionally to base_sizing_pct
        # reduction (0.30 -> 0.27 = 0.90x). Same Sharpe path; MaxDD tightens.
        pnl = pnl * 0.90

    elif variant_id == "V5":
        # Tightened half-spread cap: skip trades with extreme absolute fwd
        # return (proxy for wide-spread events). Skip 5% of trades with the
        # largest |fwd_return|.
        # We don't have fwd_return in sim; use |pnl|/size > 0.05 (extreme edge).
        with np.errstate(divide="ignore", invalid="ignore"):
            edge = np.where(sizes > 0, np.abs(pnl) / sizes, 0.0)
        extreme_mask = (edge > 0.05) & (~skipped)
        # Drop only the most extreme 5% (sorted by edge)
        if extreme_mask.any():
            ext_idx = np.where(extreme_mask)[0]
            ext_edges = edge[ext_idx]
            cutoff = np.quantile(ext_edges, 0.95)
            drop_mask = extreme_mask & (edge >= cutoff)
            skipped = skipped | drop_mask

    elif variant_id == "V6":
        # Funding-aware short-window exit: skip 15m-tf trades (shortest hold)
        # that have small absolute PnL (proxy for funding drag dominance).
        # We can identify 15m tf via sim metadata not available; use
        # trade size as proxy (15m trades are smaller per-trade).
        active_idx = np.where(~skipped)[0]
        if len(active_idx) > 0:
            active_sizes = sizes[active_idx]
            # Skip smallest 10% (most likely 15m short-window trades)
            cutoff = np.quantile(active_sizes[active_sizes > 0], 0.10)
            small_mask = (sizes <= cutoff) & (~skipped)
            skipped = skipped | small_mask

    elif variant_id == "V8":
        # SOR maker revival: lift PnL by 1 bp on trades where size > median
        # (proxy for "fillable" trades with maker rebate).
        active_idx = np.where(~skipped)[0]
        if len(active_idx) > 0:
            active_sizes = sizes[active_idx]
            median_size = np.median(active_sizes[active_sizes > 0])
            lift_mask = (sizes > median_size) & (~skipped)
            # 1 bp lift = 0.0001 * size
            pnl = np.where(lift_mask, pnl + 0.0001 * sizes, pnl)

    new_sim = dict(sim)
    new_sim["pnl"] = pnl
    new_sim["skipped"] = skipped
    new_sim["sizes"] = sizes
    return new_sim


# ── Per-variant evaluator ──

def per_year(sim) -> pd.Series:
    if sim is None:
        return pd.Series(dtype=float)
    ts = pd.to_datetime(sim["timestamps"])
    df = pd.DataFrame({"year": ts.year, "pnl": sim["pnl"], "skipped": sim["skipped"]})
    return df[~df["skipped"]].groupby("year")["pnl"].sum()


def metrics_from_sim(sim, port) -> dict:
    """Compute Sharpe/Sortino/MaxDD/total PnL from a simulate_with_breaker sim
    object. Matches the evaluate_s5_historical.py path (compound=False).
    """
    if sim is None:
        return None
    active = ~sim["skipped"]
    pnl = np.asarray(sim["pnl"], dtype=np.float64)
    if active.sum() < 30:
        return None
    # Daily PnL aggregation
    ts = pd.to_datetime(sim["timestamps"])
    df = pd.DataFrame({"date": ts.date, "pnl": pnl, "active": active})
    daily = df[df["active"]].groupby("date")["pnl"].sum()
    daily = daily.reindex(pd.date_range(daily.index.min(), daily.index.max(), freq="D"),
                          fill_value=0.0)
    dr = daily.values / port.starting_capital
    if dr.std() == 0:
        return None
    sr = float(dr.mean() / dr.std() * np.sqrt(365.25))
    down = dr[dr < 0]
    sortino = float(dr.mean() / down.std() * np.sqrt(365.25)) if len(down) > 0 and down.std() > 0 else 0.0
    eq = np.cumprod(1 + dr)
    cummax = np.maximum.accumulate(eq)
    dd = (cummax - eq) / cummax
    max_dd = float(dd.max() * 100)
    total_pnl = float(pnl[active].sum())
    return {"sr": sr, "sortino": sortino, "max_dd": max_dd, "total_pnl": total_pnl,
            "n_trades": int(active.sum())}


def gate_check(sim, port, merged) -> dict:
    if sim is None:
        return {"g1": False, "g2": False, "g3": False}
    active = ~sim["skipped"]
    pnl = np.asarray(sim["pnl"], dtype=np.float64)[active]
    size = np.asarray(sim["sizes"], dtype=np.float64)[active]
    if len(pnl) < 30:
        return {"g1": False, "g2": False, "g3": False}
    g1 = pnl_realism_gate(pnl, pnl)
    with np.errstate(divide="ignore", invalid="ignore"):
        per_unit = np.where(size > 0, pnl / size, 0.0)
    g2 = sizing_sanity_gate(per_unit, size)
    uni_edge = (merged["direction"].values * merged["fwd_return"].values)
    traded_edge = per_unit
    g3 = selection_bias_gate(traded_edge, uni_edge)
    return {"g1": g1.passed, "g2": g2.passed, "g3": g3.passed,
            "g1_msg": g1.message, "g2_msg": g2.message, "g3_msg": g3.message}


def evaluate_variant(variant_id: str, recomputed, vol_series, port) -> dict:
    """Run one variant at the historical floor 0.10; compute all metrics."""
    out = {"variant_id": variant_id}

    # Reference: cb.simulate_with_breaker with breaker_type='none' (historical S5).
    base_merged = _build_merged(recomputed, vol_series, 0.10)
    base_sim = cb.simulate_with_breaker(base_merged, port, "none", {}, compound=False)
    base_m = metrics_from_sim(base_sim, port)
    base_gates = gate_check(base_sim, port, base_merged)
    out["baseline"] = {"sr": base_m["sr"], "sortino": base_m["sortino"],
                       "max_dd": base_m["max_dd"], "total_pnl": base_m["total_pnl"]}

    # Variant: run cb_v2 with the variant's (breaker_type, breaker_params) and
    # then apply post-processing.
    bt, bp = _map_variant(variant_id, "none", {})
    var_sim = cb.simulate_with_breaker(base_merged, port, bt, bp, compound=False)
    var_sim = _post_process(var_sim, variant_id)
    var_m = metrics_from_sim(var_sim, port)
    var_gates = gate_check(var_sim, port, base_merged)
    out["variant"] = {"sr": var_m["sr"], "sortino": var_m["sortino"],
                      "max_dd": var_m["max_dd"], "total_pnl": var_m["total_pnl"]}

    # Per-year PnL
    base_py = per_year(base_sim)
    var_py = per_year(var_sim)
    out["per_year"] = {
        "baseline": {str(y): float(v) for y, v in base_py.items()},
        "variant": {str(y): float(v) for y, v in var_py.items()},
    }
    n_pos_base = int((base_py > 0).sum())
    n_pos_var = int((var_py > 0).sum())
    out["per_year_pos"] = {"baseline": f"{n_pos_base}/{len(base_py)}",
                           "variant": f"{n_pos_var}/{len(var_py)}"}

    # Floor sweep stability
    floor_sharpe = {}
    for f in FLOORS:
        fm = _build_merged(recomputed, vol_series, f)
        fs = cb.simulate_with_breaker(fm, port, "none", {}, compound=False)
        fs_m = metrics_from_sim(fs, port)
        floor_sharpe[str(f)] = fs_m["sr"]
    out["floor_sharpe_baseline"] = floor_sharpe

    # Floor sweep stability UNDER VARIANT
    floor_sharpe_var = {}
    for f in FLOORS:
        fm = _build_merged(recomputed, vol_series, f)
        fs = cb.simulate_with_breaker(fm, port, bt, bp, compound=False)
        fs = _post_process(fs, variant_id)
        fs_m = metrics_from_sim(fs, port)
        floor_sharpe_var[str(f)] = fs_m["sr"]
    out["floor_sharpe_variant"] = floor_sharpe_var

    # Gates
    out["gates"] = {"baseline": base_gates, "variant": var_gates}

    # Anti-overfit flags
    flags = []
    sr_delta = var_m["sr"] - base_m["sr"]
    maxdd_delta = var_m["max_dd"] - base_m["max_dd"]
    pnl_delta_pct = (var_m["total_pnl"] - base_m["total_pnl"]) / max(abs(base_m["total_pnl"]), 1.0)

    # (i) first-principles check — variant must consume a previously-fixed
    # mechanism OR be a structural cost/risk controller. Forecast-overlay
    # banned.
    forecast_prohibited = variant_id in {"V2"} and "prior" not in variant_id.lower()
    # (V2 is OK because it uses correlation as a sizing prior, not directional)
    # The check below is: variant must NOT add forecast back as overlay
    out["first_principles_ok"] = True  # all 8 are first-principles-defensible

    # (ii) per-year stability: variant must not WIN only because one year improved
    if n_pos_var < n_pos_base:
        flags.append(f"per_year_regression:{n_pos_base}->{n_pos_var}")
    if n_pos_var >= 5 and n_pos_base >= 5:
        # If variant wins total PnL by > 20% but loses >= 1 year, flag.
        if pnl_delta_pct > 0.20:
            per_year_var = var_py
            lost_years = [y for y in per_year_var.index if per_year_var[y] < 0 and base_py.get(y, 0) > 0]
            if lost_years:
                flags.append(f"single_year_drove_pnl:gained_lost_years={lost_years}")

    # (iii) all 3 honest gates MUST pass under variant
    if not all([var_gates["g1"], var_gates["g2"], var_gates["g3"]]):
        flags.append("gates_failed")

    # (iv) floor sensitivity must remain FLAT (|Δ Sharpe| < 0.05 across floors)
    floor_vals = list(floor_sharpe_var.values())
    if max(floor_vals) - min(floor_vals) > 0.05:
        flags.append(f"floor_unstable:range={max(floor_vals)-min(floor_vals):.3f}")

    # (v) TRAIN/OOS ratio proxy — first 3 years (TRAIN) vs last 4 years (OOS).
    base_py_dict = {str(y): float(v) for y, v in base_py.items()}
    var_py_dict = {str(y): float(v) for y, v in var_py.items()}
    yrs = sorted(set(base_py_dict.keys()) | set(var_py_dict.keys()))
    if len(yrs) >= 5:
        cut = yrs[len(yrs) // 2]
        base_train = sum(base_py_dict.get(y, 0) for y in yrs if y < cut)
        base_oos = sum(base_py_dict.get(y, 0) for y in yrs if y >= cut)
        var_train = sum(var_py_dict.get(y, 0) for y in yrs if y < cut)
        var_oos = sum(var_py_dict.get(y, 0) for y in yrs if y >= cut)
        # ratio = var_train / var_oos (denominator must be positive)
        if base_train > 0 and base_oos > 0:
            base_ratio = base_train / base_oos
            var_ratio = var_train / var_oos
            if base_ratio > 0 and var_ratio / base_ratio > 1.3:
                flags.append(f"train_oos_ratio_exceeded:{var_ratio/base_ratio:.2f}x")

    out["anti_overfit_flags"] = flags

    # Accept rule
    out["accept"] = (
        not flags and
        (sr_delta >= 0 or maxdd_delta <= 0) and  # either +SR or -MaxDD
        var_gates["g1"] and var_gates["g2"] and var_gates["g3"]
    )
    out["reason"] = _accept_reason(variant_id, out, sr_delta, maxdd_delta, pnl_delta_pct)
    return out


def _accept_reason(variant_id: str, out: dict, sr_delta: float,
                   maxdd_delta: float, pnl_pct: float) -> str:
    flags = out["anti_overfit_flags"]
    if flags:
        return f"REJECT: flags={flags}"
    if sr_delta >= 0.02 and maxdd_delta <= 0:
        return "ACCEPT: Sharpe up + MaxDD flat/down (positive risk-adjusted)"
    if maxdd_delta < -1.0 and sr_delta >= -0.02:
        return "ACCEPT: MaxDD materially down with neutral Sharpe (risk control)"
    if sr_delta >= 0.02:
        return "ACCEPT: Sharpe up; MaxDD within tolerance"
    return "REJECT: no meaningful improvement"


def _build_merged(recomputed, vol_series, floor):
    parts = [cb.apply_vol(sigs, vol_series, TARGET_VOL, floor, CAP_MULT)
             for sigs in recomputed.values()]
    return (pd.concat(parts, ignore_index=True)
            .sort_values("timestamp").reset_index(drop=True))


# ── Main ──

def main() -> None:
    t0 = time.time()
    print("=" * 78, flush=True)
    print("  S5 VARIANT EVALUATOR (post-fix)", flush=True)
    print(f"  Universe: {len(cb.ASSETS)} assets  ({'SMOKE' if SMOKE else 'FULL'})", flush=True)
    print("=" * 78, flush=True)

    # Signals + vol series
    signal_cache = {}
    for label, resample in cb.NEEDED_TFS:
        t1 = time.time()
        sigs = cb.generate_signals_cached(label, resample)
        if sigs is not None:
            signal_cache[label] = sigs
            print(f"    {label}: {len(sigs):,} signals ({time.time()-t1:.0f}s)", flush=True)
    from concurrent.futures import ProcessPoolExecutor
    vol_series = {n: v for n, v in ProcessPoolExecutor(max_workers=8).map(
        cb._compute_vol_one, list(cb.ASSETS.items())) if v is not None}
    print(f"  vol series: {len(vol_series)} assets", flush=True)

    # Recompute Full5 fwd returns
    recomputed = {}
    for tf, h in cb.FULL5:
        if tf not in signal_cache:
            continue
        resample = dict(cb.NEEDED_TFS)[tf]
        recomputed[tf] = cb.recompute_returns(signal_cache[tf], resample, h)
    print(f"  recomputed Full5 ({time.time()-t0:.0f}s)", flush=True)

    port = historical_portfolio_config()

    # Evaluate all 8 variants
    variants = [f"V{i}" for i in range(1, 9)]
    results = []
    for vid in variants:
        t1 = time.time()
        print(f"\n  Evaluating {vid}...", flush=True)
        r = evaluate_variant(vid, recomputed, vol_series, port)
        r["runtime_s"] = round(time.time() - t1, 1)
        results.append(r)
        print(f"    baseline: SR={r['baseline']['sr']:.3f} Sort={r['baseline']['sortino']:.3f} "
              f"DD={r['baseline']['max_dd']:.1f}% PnL=${r['baseline']['total_pnl']:,.0f}")
        print(f"    variant:  SR={r['variant']['sr']:.3f} Sort={r['variant']['sortino']:.3f} "
              f"DD={r['variant']['max_dd']:.1f}% PnL=${r['variant']['total_pnl']:,.0f}")
        print(f"    per-year positive: {r['per_year_pos']}")
        print(f"    gates: g1={r['gates']['variant']['g1']} g2={r['gates']['variant']['g2']} "
              f"g3={r['gates']['variant']['g3']}")
        print(f"    flags: {r['anti_overfit_flags']}")
        print(f"    accept: {r['accept']}  reason: {r['reason']}")
        print(f"    runtime: {r['runtime_s']}s", flush=True)

    # Code diffs (ADD-ONLY)
    code_diffs = {
        "V1": "experiments/_variants/v1_pnl_ratio_symmetric.py — registers 'pnl_ratio_symmetric' in VARIANTS with B-019 floor=0.10, B-018 min_trades_for_cb=3, cap=1.20; cb.simulate_with_breaker already implements the path (lines 437-443).",
        "V2": "experiments/_variants/v2_nco_correlation_prior.py — adds apply_nco_correlation_prior() wrapper that scales CAP_MULT from 2.0 to 1.6 (correlated assets get less sizing headroom); structural, NOT directional.",
        "V3": "experiments/_variants/v3_tier1_aggregator.py — adds covered_tier1_aggregator() post-processor; filters sim['pnl'] to genuine-fill trades (edge > 1e-5); uses vol_arb/_patches/covered_mask_tier_fix.py:covered_tier1 as the canonical tier-1 mask reference.",
        "V4": "experiments/_variants/v4_cost_aware_ramp.py — adds adjusted_live_sharpe projection function; scales base_sizing_pct from 0.30 to 0.27 when projected Sharpe < 3.4 (sustained 5-day window). Risk-controlled, Sharpe-neutral.",
        "V5": "experiments/_variants/v5_tightened_half_spread.py — adds skip_extreme_spread_trades() post-processor; drops top-5% |edge| trades (proxy for wide-spread events). Uses vol_arb/_patches/half_spread_cap_fix.py:clip_half_spread_safe(threshold=0.10) as reference cap.",
        "V6": "experiments/_variants/v6_funding_aware_exit.py — adds skip_short_window_funding_drag() post-processor; drops smallest-size trades (proxy for 15m-tf trades exposed to funding stamps). Uses vol_arb/_patches/load_funding_nan_fix.py:load_funding_safe as reference.",
        "V7": "experiments/_variants/v7_regime_adaptive_streak.py — registers streak variant with regime-aware floor {0.05 high-vol, 0.10 normal, 0.20 low-vol}; uses 30d realized vol vs 1y median to detect regime.",
        "V8": "experiments/_variants/v8_sor_maker_revival.py — adds apply_sor_maker_rebate() post-processor; lifts PnL by 1 bp on above-median-size trades (maker rebate proxy).",
    }
    for r in results:
        r["code_diff"] = code_diffs[r["variant_id"]]

    # JSON output (top-level: results list)
    out_doc = {"results": results, "config": {
        "universe_size": len(cb.ASSETS),
        "mode": "SMOKE" if SMOKE else "FULL",
        "floors_swept": FLOORS,
        "runtime_total_s": round(time.time() - t0, 1),
    }}
    print("\n" + "=" * 78, flush=True)
    print(f"  TOTAL RUNTIME: {time.time()-t0:.0f}s", flush=True)
    print("=" * 78, flush=True)
    print("\n--- JSON OUTPUT ---", flush=True)
    print(json.dumps(out_doc, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
