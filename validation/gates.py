"""Validation gates — Phase-5 post-mortem hardening (2026-05-18).

A validation is only as honest as the PnL series it consumes. Before any
shuffle test / bootstrap CI / walk-forward gate, a backtest must pass:

1. ``pnl_realism_gate`` — the model-computed PnL must agree (sign,
   magnitude, correlation) with PnL marked to REAL fills. Catches the
   V2 error: `vega x d(iv_used)` booked +$1.6M while fill-marked PnL was
   -$587K (opposite sign).

2. ``sizing_sanity_gate`` — the headline must not change SIGN under a
   different, equally-defensible position-sizing convention. Catches the
   cycle-50 error: fixed-N sizing showed Sharpe -0.80, constant-vega
   sizing showed +2.28, same trades.

3. ``selection_bias_gate`` — a metric measured on the TRADED subset must
   not be materially richer than the same metric on the full candidate
   universe without a stated, lookahead-free reason. Catches the
   cycle-50 error: a non-overlapping sampler over-selected IV-spike
   onsets, inflating VRP +1.5vp -> +5.0vp.

All three are pure functions returning a ``GateResult``; no I/O. Wire
them into the test suite for every strategy.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

_EPS = 1e-9


@dataclass
class GateResult:
    """Outcome of a validation gate.

    ``passed`` False means the backtest headline is NOT trustworthy and
    must not be cited until the underlying issue is resolved.
    """
    name: str
    passed: bool
    message: str
    detail: dict[str, Any] = field(default_factory=dict)

    def __bool__(self) -> bool:  # allow `assert gate_result`
        return self.passed

    def raise_if_failed(self) -> None:
        if not self.passed:
            raise AssertionError(f"[{self.name}] FAILED: {self.message}")


def _finite_pair(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    mask = np.isfinite(a) & np.isfinite(b)
    return a[mask], b[mask]


# ──────────────────────────────────────────────────────────────────────
# Gate 1 — PnL realism
# ──────────────────────────────────────────────────────────────────────
def pnl_realism_gate(
    model_pnl: np.ndarray,
    fill_marked_pnl: np.ndarray,
    *,
    min_corr: float = 0.85,
    max_total_rel_err: float = 0.30,
) -> GateResult:
    """Compare a backtest's model-computed per-trade PnL against PnL
    marked to REAL fills (or the honest realized-edge equivalent).

    Fails if ANY of:
      - the sign of total PnL differs (model says profit, reality loss);
      - |Σmodel - Σfill| / |Σfill| > ``max_total_rel_err``;
      - per-trade correlation < ``min_corr``.

    The V2 backtester fails on the first criterion: model `vega*d(iv)`
    summed to +$1.6M, fill-marked (IV-frozen honest) summed to -$587K.

    Parameters
    ----------
    model_pnl, fill_marked_pnl : array-like
        Per-trade PnL, same order, same length.
    """
    m, f = _finite_pair(model_pnl, fill_marked_pnl)
    if len(m) < 2:
        return GateResult("pnl_realism", False,
                          "fewer than 2 finite trade pairs", {"n": len(m)})
    tot_m, tot_f = float(m.sum()), float(f.sum())
    sign_ok = np.sign(tot_m) == np.sign(tot_f)
    rel_err = abs(tot_m - tot_f) / max(abs(tot_f), _EPS)
    corr = float(np.corrcoef(m, f)[0, 1]) if m.std() > 0 and f.std() > 0 else 0.0
    passed = bool(sign_ok and rel_err <= max_total_rel_err and corr >= min_corr)
    detail = {
        "n": len(m), "total_model": tot_m, "total_fill_marked": tot_f,
        "sign_agree": bool(sign_ok), "total_rel_err": rel_err, "corr": corr,
    }
    if not sign_ok:
        msg = (f"SIGN FLIP — model total ${tot_m:,.0f} vs fill-marked "
               f"${tot_f:,.0f}. The backtest PnL is fiction.")
    elif rel_err > max_total_rel_err:
        msg = (f"magnitude divergence {rel_err:.0%} > {max_total_rel_err:.0%} "
               f"(model ${tot_m:,.0f} vs fill ${tot_f:,.0f})")
    elif corr < min_corr:
        msg = f"per-trade corr {corr:.3f} < {min_corr} — PnL shape unrealistic"
    else:
        msg = (f"model and fill-marked PnL agree (corr {corr:.3f}, "
               f"total rel.err {rel_err:.0%})")
    return GateResult("pnl_realism", passed, msg, detail)


# ──────────────────────────────────────────────────────────────────────
# Gate 2 — sizing sanity
# ──────────────────────────────────────────────────────────────────────
def sizing_sanity_gate(
    pnl_per_unit: np.ndarray,
    exposure: np.ndarray,
    *,
    max_sharpe_spread: float = 1.0,
) -> GateResult:
    """Check the headline does not depend on the position-sizing
    convention. ``pnl_per_unit`` is per-trade net edge per unit of
    exposure (e.g. VRP-minus-cost in vol points); ``exposure`` is the
    per-trade exposure that fixed-N sizing implicitly weights by (e.g.
    straddle vega, or notional).

    Three defensible conventions are compared:
      - constant-exposure (risk-normalised): total ∝ Σ pnl_per_unit
      - fixed-N (exposure-weighted):          total ∝ Σ pnl_per_unit*exp
      - equal-weight per-trade mean

    FAILS if the SIGN of the total flips between constant-exposure and
    fixed-N — that means "does this strategy make money?" has no
    convention-independent answer (the cycle-50 bug).
    """
    e, x = _finite_pair(pnl_per_unit, exposure)
    if len(e) < 2:
        return GateResult("sizing_sanity", False,
                          "fewer than 2 finite trades", {"n": len(e)})
    if np.any(x <= 0):
        return GateResult("sizing_sanity", False,
                          "non-positive exposure values present", {"n": len(e)})
    total_const_exp = float(e.sum())                  # constant-exposure sizing
    total_fixed_n = float((e * x).sum())              # fixed-N sizing
    sign_flip = np.sign(total_const_exp) != np.sign(total_fixed_n)
    eq_mean = float(e.mean())
    exp_wt_mean = float((e * x).sum() / x.sum())
    passed = not bool(sign_flip)
    detail = {
        "n": len(e),
        "total_constant_exposure": total_const_exp,
        "total_fixed_N": total_fixed_n,
        "equal_weight_mean": eq_mean,
        "exposure_weighted_mean": exp_wt_mean,
        "sign_flip": bool(sign_flip),
    }
    if sign_flip:
        msg = (f"SIGN FLIP across sizing conventions — constant-exposure "
               f"total {total_const_exp:+.4g} vs fixed-N {total_fixed_n:+.4g}. "
               f"The edge sign depends on the sizing choice; not deployable "
               f"until resolved (use risk-normalised sizing).")
    else:
        msg = (f"headline sign stable across sizing conventions "
               f"(equal-wt mean {eq_mean:+.4g}, exposure-wt {exp_wt_mean:+.4g})")
    return GateResult("sizing_sanity", passed, msg, detail)


# ──────────────────────────────────────────────────────────────────────
# Gate 3 — selection bias
# ──────────────────────────────────────────────────────────────────────
def selection_bias_gate(
    metric_traded: np.ndarray,
    metric_universe: np.ndarray,
    *,
    max_rel_lift: float = 0.5,
) -> GateResult:
    """Check the metric on the TRADED subset is not unexplainedly richer
    than on the full candidate universe.

    ``metric_traded``  — the per-trade metric on bars the strategy
                         actually entered (e.g. VRP of taken trades).
    ``metric_universe``— the same metric on EVERY candidate bar the
                         signal could have fired on.

    FAILS if mean(traded) exceeds mean(universe) by more than
    ``max_rel_lift`` (relative) — a sampler/selection step is
    over-picking rich bars. cycle-50: traded VRP +5.0vp vs universe
    +1.5vp = +233% lift -> FAIL.

    A genuine, lookahead-free timing signal can legitimately lift the
    metric; this gate forces that lift to be acknowledged and justified
    rather than smuggled in by a sampler.
    """
    t = np.asarray(metric_traded, dtype=float)
    u = np.asarray(metric_universe, dtype=float)
    t = t[np.isfinite(t)]
    u = u[np.isfinite(u)]
    if len(t) < 2 or len(u) < 2:
        return GateResult("selection_bias", False,
                          "insufficient data", {"n_traded": len(t),
                                                 "n_universe": len(u)})
    mt, mu = float(t.mean()), float(u.mean())
    rel_lift = (mt - mu) / max(abs(mu), _EPS)
    passed = bool(rel_lift <= max_rel_lift)
    detail = {
        "n_traded": len(t), "n_universe": len(u),
        "mean_traded": mt, "mean_universe": mu, "rel_lift": rel_lift,
    }
    if not passed:
        msg = (f"traded subset metric {mt:+.4g} is {rel_lift:+.0%} richer "
               f"than the universe {mu:+.4g} — selection bias; the headline "
               f"reflects the sampler, not the strategy.")
    else:
        msg = (f"traded subset metric {mt:+.4g} vs universe {mu:+.4g} "
               f"(lift {rel_lift:+.0%}, within tolerance)")
    return GateResult("selection_bias", passed, msg, detail)
