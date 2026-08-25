"""ATTACK 5 — HONEST Deflated Sharpe Ratio (multiple-testing hurdle).

K configs were tried on this strategy family: holds {15m,1h,4h,1d,expiry}=5 +
bands {0.05,0.1,0.2,0.35,0.5,0.75}=6. Honest K>=11 (arguably more with greedy
add-back). Compute the multiple-testing-adjusted significance hurdle for K=11 and
K=20 and test whether the best band's NON-OVERLAPPING t-stat and Newey-West t-stat
clear it UNCONDITIONALLY.

We treat the script's reported Sharpe numbers as t-stat analogues (they are
mean/sd*sqrt(n)). We compute:
  - the standard Bonferroni / Sidak two-sided z-hurdle for K tests,
  - the Bailey-Lopez de Prado Deflated Sharpe Ratio expected-max-Sharpe hurdle,
  - the implied DSR p-value of the best config.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

# From the band sweep log (hold=expiry). nonov n_keep=99 throughout.
# nw_sharpe and nonov_sharpe are mean/sd*sqrt(n)-style t-stat analogues already.
BANDS = {
    0.05: dict(nw=-0.341, nonov=-1.374, n_keep=99, naive=-1.086, n=945),
    0.10: dict(nw=0.679, nonov=-0.167, n_keep=99, naive=2.193, n=945),
    0.20: dict(nw=1.335, nonov=0.712, n_keep=99, naive=4.411, n=945),
    0.35: dict(nw=1.886, nonov=1.048, n_keep=99, naive=5.937, n=945),
    0.50: dict(nw=1.455, nonov=1.261, n_keep=99, naive=4.935, n=945),
    0.75: dict(nw=1.876, nonov=1.034, n_keep=99, naive=6.477, n=945),
}


def expected_max_sharpe(K, var_sr):
    """Bailey-LdP expected maximum of K independent Sharpe estimates (the DSR
    benchmark SR0). E[max] ~ sqrt(var_sr) * [(1-g)*Phi^-1(1-1/K) + g*Phi^-1(1-1/(K*e))],
    g = Euler-Mascheroni. var_sr is the variance of the Sharpe ESTIMATE."""
    g = 0.5772156649
    z1 = norm.ppf(1 - 1.0 / K)
    z2 = norm.ppf(1 - 1.0 / (K * np.e))
    return np.sqrt(var_sr) * ((1 - g) * z1 + g * z2)


def main():
    # best by NW Sharpe
    best_band = max(BANDS, key=lambda b: BANDS[b]["nw"])
    b = BANDS[best_band]
    print(f"best band by NW Sharpe = {best_band:.2f}: nw={b['nw']:.3f} "
          f"nonov={b['nonov']:.3f} (n_keep={b['n_keep']})")

    # These "Sharpe" numbers ARE t-stats (mean/sd*sqrt(n)); convert to a per-obs SR
    # for the non-overlapping series: SR_perobs = t / sqrt(n_keep).
    n_keep = b["n_keep"]
    t_nonov = b["nonov"]
    t_nw = b["nw"]
    sr_perobs_nonov = t_nonov / np.sqrt(n_keep)
    print(f"\nnon-overlapping t-stat = {t_nonov:.3f} over n_keep={n_keep} "
          f"-> per-obs SR = {sr_perobs_nonov:.4f}")
    print(f"Newey-West t-stat      = {t_nw:.3f}")

    print("\n" + "=" * 70)
    print("MULTIPLE-TESTING HURDLES (two-sided)")
    print("=" * 70)
    for K in (6, 11, 20):
        # Bonferroni alpha=0.05 two-sided
        z_bonf = norm.ppf(1 - 0.05 / (2 * K))
        # Sidak
        z_sidak = norm.ppf(1 - (1 - (1 - 0.05) ** (1.0 / K)) / 2)
        print(f"  K={K:>2}: Bonferroni z-hurdle(5%)={z_bonf:.3f}   Sidak={z_sidak:.3f}")

    print("\n" + "=" * 70)
    print("DEFLATED SHARPE RATIO (Bailey & Lopez de Prado)")
    print("=" * 70)
    # Variance of the Sharpe estimate for an IID series of length n: ~ (1 + 0.5 SR^2)/n.
    # Use the non-overlapping per-obs SR and n_keep. Skew/kurt unknown -> Gaussian.
    for K in (11, 20):
        # var of SR estimate (annualization-free, per-obs convention)
        var_sr = (1.0 + 0.5 * sr_perobs_nonov ** 2) / n_keep
        sr0 = expected_max_sharpe(K, var_sr)       # benchmark per-obs SR
        # DSR = Phi( (SR_hat - SR0) / sd(SR_hat) )
        sd_sr = np.sqrt(var_sr)
        dsr = norm.cdf((sr_perobs_nonov - sr0) / sd_sr)
        # equivalently as t-stats:
        t_hurdle = sr0 / sd_sr   # the t-stat the best config must EXCEED
        print(f"  K={K}: E[max SR0](per-obs)={sr0:.4f}  sd(SR)={sd_sr:.4f}  "
              f"-> t-hurdle={t_hurdle:.3f}")
        print(f"        best non-overlap t={t_nonov:.3f}  DSR p(skill)={dsr:.3f}  "
              f"{'CLEARS' if t_nonov > t_hurdle else 'FAILS'} the K={K} DSR hurdle")

    print("\n" + "=" * 70)
    print("VERDICT INPUTS")
    print("=" * 70)
    print(f"  NW t = {t_nw:.3f}  vs  Bonferroni K=11 hurdle ~2.54, K=20 ~2.81")
    print(f"  nonov t = {t_nonov:.3f}  vs  same hurdles")
    print("  (A t-stat must exceed the hurdle to pass this historical screen.)")


if __name__ == "__main__":
    main()
