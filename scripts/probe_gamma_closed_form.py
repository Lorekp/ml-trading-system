"""Adversarial gamma-closed-form probe.

Construct synthetic BTC price paths with KNOWN realized variance and drive the
ACTUAL engine function `compute_gamma_leg_realpath` through a constructed
PricePanel. Verify the engine gamma leg matches the closed form

    gamma_pnl_long = 0.5 * Gamma * S^2 * (RV_total - IV^2 * T)

where RV_total = sum_t r_t^2 (realized variance over the hold) and IV^2*T is the
implied variance over the hold (T = n_bars * dt_bar). SHORT flips the sign.

We test HIGH-RV (realized >> implied) and LOW-RV (realized << implied) paths,
both LONG and SHORT, and check the engine value to within discretization error.
Because the engine uses S_t^2 with the *time-varying* S along the path, a fully
exact closed form requires summing 0.5*Gamma*S_t^2*(r_t^2 - IV^2*dt) bar by bar.
We verify BOTH:
  (1) the per-bar exact closed form == engine (machine precision), and
  (2) the constant-S approximation 0.5*Gamma*S0^2*(RV - IV^2*T) is close
      (this is the economically-meaningful headline; matches to O(path drift)).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vol_arb.realistic_eval import compute_gamma_leg_realpath, PricePanel, RealisticEvalConfig, _YEAR_MIN


def build_panel_and_trade(rets, S0, t0_ns, dt_ns):
    """Given a 1-D array of per-bar log returns, build a PricePanel whose closes
    realize EXACTLY those returns, and an entry/exit window covering all bars.

    rets has length m -> path has m+1 prices (S0 .. S_m). Bars are stamped at
    t0, t0+dt, ... The engine's searchsorted uses side='left' for entry and
    'right' for exit; we place entry exactly on the first stamp and exit exactly
    on the last stamp so lo..hi spans the whole path.
    """
    m = len(rets)
    logS = np.concatenate([[np.log(S0)], np.log(S0) + np.cumsum(rets)])
    S = np.exp(logS)
    ts = t0_ns + np.arange(m + 1) * dt_ns
    panel = PricePanel(ts.astype(np.int64), S)
    return panel, ts, S


def run_engine(panel, ts, S, gamma_long, iv_dec, signal, T_years):
    """Drive compute_gamma_leg_realpath for ONE trade spanning the whole panel."""
    cfg = RealisticEvalConfig()
    entry_ts = np.array([ts[0]], dtype="datetime64[ns]")
    exit_ts = np.array([ts[-1]], dtype="datetime64[ns]")
    gpb, hedge_fee, n_reb, mean_abs, pos_delta, resid = compute_gamma_leg_realpath(
        entry_ts, exit_ts,
        S_entry=np.array([S[0]]),
        delta_long=np.array([0.0]),
        gamma_long=np.array([gamma_long]),
        signal=np.array([float(signal)]),
        iv_used=np.array([iv_dec]),      # decimal (< 5) -> NOT auto-divided
        T_years=np.array([T_years]),
        panel=panel, cfg=cfg,
    )
    return float(gpb[0])


def closed_form_perbar(S, rets, gamma_long, iv_dec, signal, dt_bar):
    """Exact per-bar closed form the engine should reproduce:
    signal * sum_t 0.5*Gamma*S_t^2*(r_t^2 - IV^2*dt). S_t is the bar-START price."""
    S_start = S[:-1]
    r = rets
    per_bar = 0.5 * gamma_long * S_start**2 * (r**2 - iv_dec**2 * dt_bar)
    return signal * per_bar.sum()


def closed_form_constS(S0, rets, gamma_long, iv_dec, signal, dt_bar):
    """Economic headline: 0.5*Gamma*S0^2*(RV_total - IV^2*T)."""
    RV = np.sum(rets**2)
    m = len(rets)
    T = m * dt_bar
    return signal * 0.5 * gamma_long * S0**2 * (RV - iv_dec**2 * T)


def main():
    dt_bar = 1.0 / _YEAR_MIN
    t0_ns = np.int64(pd.Timestamp("2024-01-01", tz="UTC").value)
    dt_ns = np.int64(60_000_000_000)  # 1 minute in ns
    S0 = 40000.0
    gamma_long = 1.0e-4   # per-BTC gamma; magnitude arbitrary, scales linearly
    m = 15                # 15-minute hold, 15 one-minute bars

    rng = np.random.RandomState(7)

    cases = []

    # ---- HIGH-RV path: realized vol ~ 1.2 (annualized) >> IV 0.5 ----
    iv_hi = 0.50
    sigma_real_hi = 1.20
    step_hi = sigma_real_hi * np.sqrt(dt_bar)
    rets_hi = rng.normal(0.0, step_hi, m)

    # ---- LOW-RV path: realized vol ~ 0.15 << IV 0.80 ----
    iv_lo = 0.80
    sigma_real_lo = 0.15
    step_lo = sigma_real_lo * np.sqrt(dt_bar)
    rets_lo = rng.normal(0.0, step_lo, m)

    specs = [
        ("HIGH-RV LONG ", rets_hi, iv_hi, +1),
        ("HIGH-RV SHORT", rets_hi, iv_hi, -1),
        ("LOW-RV  LONG ", rets_lo, iv_lo, +1),
        ("LOW-RV  SHORT", rets_lo, iv_lo, -1),
    ]

    print(f"{'case':14s} {'engine':>14s} {'cf_perbar':>14s} {'cf_constS':>14s} "
          f"{'relerr_pb':>11s} {'relerr_cS':>11s}  sign_ok")
    print("-" * 100)

    all_pb_relerr = []
    all_signs_ok = True
    economic_sign_ok = True

    for name, rets, iv, sig in specs:
        T_years = m * dt_bar
        panel, ts, S = build_panel_and_trade(rets, S0, t0_ns, dt_ns)
        eng = run_engine(panel, ts, S, gamma_long, iv, sig, T_years)
        cf_pb = closed_form_perbar(S, rets, gamma_long, iv, sig, dt_bar)
        cf_cs = closed_form_constS(S0, rets, gamma_long, iv, sig, dt_bar)

        relerr_pb = abs(eng - cf_pb) / max(abs(cf_pb), 1e-12)
        relerr_cs = abs(eng - cf_cs) / max(abs(cf_cs), 1e-12)
        all_pb_relerr.append(relerr_pb)

        # economic-sign check: HIGH-RV long should PROFIT (RV>IV); short should LOSE.
        RV = np.sum(rets**2)
        IVvar = iv**2 * T_years
        if RV > IVvar:   # realized > implied
            want = +1 if sig > 0 else -1  # long profits, short loses
        else:
            want = -1 if sig > 0 else +1  # long loses, short profits
        got = np.sign(eng)
        sign_ok = (got == want) or abs(eng) < 1e-9
        economic_sign_ok = economic_sign_ok and sign_ok

        print(f"{name:14s} {eng:14.4f} {cf_pb:14.4f} {cf_cs:14.4f} "
              f"{relerr_pb:11.2e} {relerr_cs:11.2e}  {sign_ok}")

    print("-" * 100)
    max_pb = max(all_pb_relerr)
    print(f"max per-bar closed-form relerr = {max_pb:.3e}")
    print(f"economic sign all correct      = {economic_sign_ok}")

    # ---- Long + short must be exact negatives of each other ----
    panel_hi, ts_hi, S_hi = build_panel_and_trade(rets_hi, S0, t0_ns, dt_ns)
    eng_long = run_engine(panel_hi, ts_hi, S_hi, gamma_long, iv_hi, +1, m * dt_bar)
    eng_short = run_engine(panel_hi, ts_hi, S_hi, gamma_long, iv_hi, -1, m * dt_bar)
    antisym_err = abs(eng_long + eng_short)
    print(f"long+short antisymmetry resid  = {antisym_err:.3e} "
          f"(should be ~0; long={eng_long:.4f} short={eng_short:.4f})")

    # ---- DEGENERATE: realized == implied EXACTLY -> gamma ~ 0 ----
    # construct returns whose r^2 == iv^2*dt exactly every bar (alternating sign)
    iv_deg = 0.6
    r_deg_mag = iv_deg * np.sqrt(dt_bar)
    rets_deg = r_deg_mag * np.where(np.arange(m) % 2 == 0, 1.0, -1.0)
    panel_d, ts_d, S_d = build_panel_and_trade(rets_deg, S0, t0_ns, dt_ns)
    eng_deg = run_engine(panel_d, ts_d, S_d, gamma_long, iv_deg, -1, m * dt_bar)
    # per-bar (r^2 - iv^2 dt) == 0 exactly, so engine must be ~0 to machine eps
    # (scaled by 0.5*gamma*S^2 ~ 0.5*1e-4*1.6e9 = 8e4, so tol relative to that)
    scale = 0.5 * gamma_long * S0**2
    print(f"degenerate RV==IV gamma        = {eng_deg:.6e} "
          f"(scale={scale:.3e}, ratio={abs(eng_deg)/scale:.3e})")

    # ---- VERDICT ----
    pass_pb = max_pb < 1e-9
    pass_sign = economic_sign_ok
    pass_antisym = antisym_err < 1e-6 * abs(eng_long)
    pass_deg = abs(eng_deg) / scale < 1e-12
    print()
    print(f"PASS per-bar closed form (<1e-9)   : {pass_pb}")
    print(f"PASS economic signs                : {pass_sign}")
    print(f"PASS long/short antisymmetry       : {pass_antisym}")
    print(f"PASS degenerate RV==IV ~0          : {pass_deg}")
    overall = pass_pb and pass_sign and pass_antisym and pass_deg
    print(f"OVERALL                            : {overall}")
    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())
