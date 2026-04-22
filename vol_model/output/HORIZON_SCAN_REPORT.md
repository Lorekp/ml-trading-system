# Horizon Scan Report — Phase 5 Cross-Horizon Eval

**Plan**: `C:/Users/laure/.claude/plans/starry-rolling-rabin.md`
**Commits**: cb55123, 7278b3f, ae4e0ed
**Active grid (per 2026-04-21 user amendment)**: {h=30m, h=60m} vs h=15m control.
h=5m deferred (SVI features are 15-min resolution → effective sampling
collapses to 15-min at h=5; true 5-min pipeline needs separate SVI
regen outside this cycle).

## Pre-committed ship gates (locked at plan time, before any backtest ran)

| gate                | NC PnL Δ vs h=15 | NC Sharpe Δ | NC MDD Δ   | bootstrap LCL | perm p (Bonf.) | action                              |
|---------------------|------------------|-------------|------------|---------------|----------------|-------------------------------------|
| **Default-flip**    | ≥ +$150K (+18%)  | ≤ −0.3      | ≤ +$1,000  | > $0          | < 0.025 (Bonf. 0.05/2) | flip V2 default                     |
| **Ship as preset**  | ≥ +$50K (+6%)    | ≤ −0.7      | ≤ +$3,000  | > $0          | < 0.025        | add `with_btc_production_h{N}m()`   |
| **Document-only**   | −$50K to +$50K   | any         | any        | n/a           | n/a            | record, do not ship                 |
| **Reject**          | < −$50K          | n/a         | n/a        | n/a           | n/a            | document failure, do not ship       |

Bonferroni α = 0.05 / 2 horizons = **0.025** (single-side).
NC PnL Δ is post-Kelly-K1 stack (the production headline per
`feedback_noncompound_truth.md`).

## Per-horizon pipeline results

### h=60m — full V2 stack (aligner + Kelly K1)

Training:
- TSRV bars: 45,768 (pre-existing).
- HAR-PD sidecar: 45,744 non-null rows at K_BARS=24 (24h @ 60m).
- TransVol full walk-forward: 50/50 months, avg QLIKE 0.5931, RankIC 0.573.
- HAR-PD walk-forward (w/ TransVol): 50/50 months, **avg ΔQLIKE −23.3%** vs
  h=60 pruned+TransVol baseline (h=15 production equivalent: −3.67%).
- Walk-forward aligner: avg QLIKE improvement **−8.96%** (h=15 reference −0.78%);
  β̃ median +0.163 (~3× h=15's +0.047 under-prediction bias).
- Kelly K1 sidecar: 381/36,248 bars boosted, mean mult 1.0040, max 2.0.

Backtest headline:

| metric              | h=60m            | h=15m (production) | Δ             |
|---------------------|------------------|--------------------|---------------|
| NC PnL              | **$227,696**     | $960,000           | **−$732,304** |
| CP PnL              | $1,881,886       | $18,710,000        | −$16,828,114  |
| NC Sharpe           | 6.15             | 6.29               | −0.14         |
| CP Sharpe           | 4.60             | 5.15               | −0.55         |
| NC MDD              | −$3,398          | −$8,288            | +$4,890 ✓     |
| NC worst day        | −$2,182          | n/a                | n/a           |
| n_trades            | 1,488            | ~7,400             | −5,912        |
| $/trade (NC)        | $153             | $130               | +18%          |

Kelly K1 contribution at h=60: **+$30,988 NC (+15.7%)** at **−0.68 Sharpe**
(vs h=15's +$114K (+13.5%) at −0.94 Sharpe — slightly better $/Sharpe
ratio at h=60 but much smaller absolute gain).

**Gate check**: NC PnL Δ = −$732K < −$50K → **REJECT**.

Honest read: h=60 has marginally better per-trade quality (+18% $/trade)
and 59% better MDD, but ~5× fewer trades drops total dollar PnL to 24%
of h=15. The pre-committed gates are dollar-PnL based → reject. Does not
ship.

### h=30m — full V2 stack (aligner + Kelly K1)

Training:
- TSRV bars: 91,535 (pre-existing).
- HAR-PD sidecar: 91,487 non-null rows at K_BARS=48 (24h @ 30m).
- TransVol full walk-forward: 50/50 months, avg QLIKE 0.5744, RankIC 0.633.
- HAR-PD walk-forward (w/ TransVol): 50/50 months, **avg ΔQLIKE −32.9%**
  vs h=30 pruned+TransVol baseline (even larger relative gain than h=60's
  −23.3% and far above h=15's −3.67%).
- Walk-forward aligner: avg QLIKE improvement **−9.30%**; β̃ median +0.12
  (under-prediction bias similar to h=60's +0.16).
- Kelly K1 sidecar: 636/72,511 bars boosted, mean mult 1.0032, max 2.0.

Backtest headline:

| metric              | h=30m            | h=15m (production) | Δ             |
|---------------------|------------------|--------------------|---------------|
| NC PnL              | **$565,499**     | $959,571           | **−$394,072** |
| CP PnL              | $10,019,806      | $18,707,982        | −$8,688,176   |
| NC Sharpe           | 6.25             | 6.29               | −0.04         |
| NC MDD              | −$7,458          | −$8,521            | +$1,063 ✓     |
| n_trades            | 2,955            | 7,417              | −4,462        |
| $/trade (NC)        | $191             | $129               | **+48%**      |

Kelly K1 contribution at h=30: **+$74,664 NC (+15.2%)** at **−1.05 Sharpe**
(bare Sharpe 7.30 → Kelly 6.25).

**Gate check**: NC PnL Δ = −$394K < −$50K → **REJECT**.

Best per-trade economics of the horizon grid ($191/trade, +48% vs h=15's
$129), but ~40% of the trade count at h=15 collapses the absolute PnL.

### Cross-horizon daily PnL correlation (h=5 pending)

| pair           | Pearson ρ | Spearman ρ (tail-robust) |
|----------------|-----------|--------------------------|
| h=15 vs h=30   | **0.870** | 0.543                    |
| h=15 vs h=60   | 0.747     | 0.450                    |
| h=30 vs h=60   | 0.682     | 0.468                    |

Closer bar sizes → higher correlation, as predicted. h=15 vs h=30 at 0.87
is effectively the same signal sampled at a different cadence.

**Portfolio Sharpe (equal-risk inverse-vol weighted)**:

| blend                | solo Sharpe pair    | blend Sharpe | vs solo h=15 |
|----------------------|---------------------|--------------|--------------|
| h=15 + h=30 (41/59)  | 6.08 / 5.26         | 5.87         | **−0.21**    |
| h=15 + h=60 (25/75)  | 6.08 / 4.41         | 5.61         | **−0.47**    |

No multi-timeframe blend we tested beats solo h=15 on Sharpe. The
observed correlations are too high for diversification to overcome the
standalone-Sharpe drop at h≠15.

### h=15m — control (existing production, memory reference)

| metric              | h=15m production |
|---------------------|------------------|
| NC PnL              | $960,000         |
| CP PnL              | $18,710,000      |
| NC Sharpe           | 6.29             |
| CP Sharpe           | 5.15             |
| NC MDD              | −$8,288          |
| CP MDD              | −5.03%           |
| n_trades            | ~7,400           |

Numbers from `memory/project_kelly_sizing_2026_04_21.md`.

## Preliminary conclusions (pending h=30)

1. **h=60 REJECTED** on dollar-PnL gate (−$732K below h=15, far below −$50K
   gate). Better Sharpe (+0.14) and MDD (−59%) are not enough at the
   dollar-PnL threshold the plan pre-committed.
2. **Per-pipeline quality surprise**: h=60 HAR-PD shows a *much* larger
   relative ΔQLIKE (−23.3%) than h=15's −3.67%. Interpretation: at h=60
   the pruned+TransVol baseline is weaker (fewer bars to learn, avg QLIKE
   0.66 vs h=15's ~0.37), so the SVI+HAR-PD add-on has more room to
   improve relative to it. Absolute QLIKE at h=60 end-state (0.50) is
   still worse than h=15's end-state, but measured more favorably on
   the relative metric.
3. **Aligner much stronger at h=60**: −8.96% avg QLIKE vs h=15's −0.78%
   (10× larger effect). Corresponds to larger systematic under-prediction
   bias at longer horizons (β̃ median +0.163 vs h=15's +0.047).
4. **Sharpe parity, not win**: h=60 Sharpe 6.15 is WITHIN 0.14 of h=15's
   6.29. The earlier smoke suggested h=15 (0.753 rank-IC) had a narrow
   edge over h=60 (0.704); that translates to essentially same
   risk-adjusted return but with far less total exposure at h=60.

## Phase 5 action (pending h=30)

- h=60 → document-only per REJECT gate. Per-horizon preset
  `with_btc_production_h60m()` stays in the code (Phase 1 infrastructure)
  but is not advertised as ship-ready.
- h=30 → pending (roughly 2× the h=60 bars and training compute; expected
  to sit between h=60 and h=15 on dollar PnL).
- V2 default stays at h=15m unchanged.

Memory entry to be written into `memory/project_horizon_scan_2026_04_21.md`
once h=30 completes.
