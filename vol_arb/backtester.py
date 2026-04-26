"""Vectorized delta-neutral vol-arb execution simulator.

Simulates straddle trading where the edge is model-predicted RV vs market IV.
All computations are fully vectorized (no for loops, no .apply).
Position sizing uses constant capital (initial_equity) for instant array computation.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ── Configuration ──────────────────────────────────────────────

class BacktestConfig(BaseModel):
    # Data paths — defaults to the SVI variant which has `svi_iv_matched`
    # populated (+26% Sharpe over non-SVI) and to the pruned+svi_atm_1d
    # predictions (T2.1, +8.3% Sharpe over OOF baseline).  See:
    #   - Stage 15.5 #1 of PIPELINE_DOCUMENTATION.md
    #   - vol_arb/output/SVI_BACKTEST_NOTE.md (parquet switch)
    #   - project_t2_1_retrain.md in memory (T2.1 pruned+SVI retrain result)
    vol_arb_path: str = "vol_arb/output/BTC_vol_arb_svi.parquet"
    # pruned_svi_har_pd_v1 + walk-forward QLIKE-optimal scalar bias alignment.
    # The underlying predictor is pruned_svi_v1 + 6 HAR-PD path-dependent
    # features (Xiao 2025, arxiv:2503.00851), certified 2026-04-18 with
    # -3.67% avg ΔQLIKE across 50 months. The "aligned_bias" parquet
    # additionally applies a per-month scalar shift β* = 0.5·log(mean(
    # exp(2·residual))) fit on trailing 3-month OOS residuals, strictly
    # non-leaking. β median +0.047 (model systematically under-predicts
    # log-vol by ~5% OOS); row-weighted QLIKE -0.78%. V2 backtest impact:
    # Compound PnL +6.4% ($14.92M → $15.87M), NC PnL +3.7% ($815K → $845K),
    # Compound Sharpe +0.07 (5.27 → 5.35), Compound MDD -0.28pp better.
    # NC Sharpe -0.42 on the non-compounding view — concentration shift
    # toward long straddles — but the compound axis (what matters for real
    # capital deployment) is neutral-to-positive on Sharpe with meaningful
    # PnL gain. Adopted as production default 2026-04-20. See
    # memory/project_prediction_aligner_2026_04_20.md. Override via
    # --predictions-path to reproduce pre-aligner results.
    predictions_path: str = "vol_model/output/BTCUSDT_pruned_svi_har_pd_predictions_aligned_bias.parquet"

    # Dynamic spread model
    base_spread_iv: float = 1.5   # base IV bid-ask spread (vol points)
    spread_jump_k: float = 3.0    # jump_fraction sensitivity

    # Signal thresholds (vol points)
    entry_threshold: float = 25.0  # additional edge required beyond half-spread

    # Fees (Deribit: 0.03% option both sides, 0.05% perp taker, -0.01% perp maker)
    option_maker_fee: float = 0.0003  # Deribit: 0.03% of underlying (same maker & taker)
    perp_taker_fee: float = 0.0005    # Deribit: 0.05% taker (maker gets -0.01% rebate)
    perp_slippage_bps: float = 2.0    # order book slippage on gamma rebalance (bps)

    # Discrete delta hedging
    delta_hedge_band: float = 0.05    # BTC delta threshold to trigger rebalance
    hedge_sim_steps: int = 10         # price path steps per 15m window
    hedge_sim_seed: int = 42          # reproducible Monte Carlo

    # Adverse selection on maker fills
    maker_adverse_selection_iv: float = 0.5  # vol points penalty per fill

    # VRP baseline (off by default — inflates edge and creates false long signals)
    use_vrp: bool = False
    # Bar-count constants below are LEGACY 15m defaults. For other horizons
    # use ``BacktestConfig.for_timeframe(bar_minutes=N)`` which derives
    # ``vrp_window=bpd*7`` and ``vrp_min_periods=bpd`` from N — preserves
    # the "7 days / 1 day" wall-clock semantics across horizons. Per-horizon
    # presets ``with_btc_production_h{N}m()`` chain through that factory.
    vrp_window: int = 672             # rolling window for VRP (672 = 7 days @ 15m)
    vrp_min_periods: int = 96         # minimum periods (96 = 1 day @ 15m)

    # Position sizing: Deribit Portfolio Margin utilization
    initial_equity: float = 100_000.0
    target_margin_utilization: float = 0.05  # fractional Kelly (5% of equity as IM)
    im_fraction: float = 0.20               # SHORT straddle IM ≈ 20% (Deribit: max(0.15-OTM/S,0.10)+mark)
    use_directional_im: bool = False          # Long IM = premium; Short IM = im_fraction
    long_im_buffer: float = 0.02             # Extra IM for perp hedge margin on longs
    long_im_floor: float = 0.03              # Minimum long IM (fraction of notional)
    max_participation_rate: float = 0.10     # 10% of trailing 1h ATM volume
    short_sizing_penalty: float = 0.50       # halve N for short straddles (tail risk)

    # Vol-targeted sizing: scale margin inversely to DVOL
    baseline_target_dvol: float = 50.0       # DVOL at which full margin deployed

    # Circuit breaker: kill short signals during liquidation cascades
    max_jump_fraction_for_shorts: float = 0.15

    # ── Slippage model ──────────────────────────────────────────
    # "grid" = legacy 4x3 lookup, "realistic_taker" = per-bar empirical from trade tape
    slippage_model: str = "realistic_taker"
    bar_exec_quality_path: str = "vol_arb/output/bar_execution_quality.parquet"
    # Legacy continuous impact model (kept for research)
    use_continuous_impact: bool = False
    impact_params_path: str = "vol_arb/output/impact_model_params.parquet"

    # ── Position sizing model ────────────────────────────────────
    # Pluggable strategy — see vol_arb/sizing.py for details.
    sizing_model: str = "composite_v2"    # baseline|edge|vega_risk|kelly|dd|composite|liq|cost_edge|composite_v2
    sizing_edge_cap: float = 3.0          # max edge multiplier (edge/vega_risk/composite)
    sizing_dd_floor: float = 0.25         # min scale at deep drawdown (dd/composite)
    sizing_risk_fraction: float = 0.005   # equity fraction as target risk (vega_risk)
    sizing_kelly_fraction: float = 0.25   # quarter Kelly (kelly)
    sizing_liq_cap: float = 1.5           # max upscale in liquid markets (liq/composite_v2)

    # IOC cap multiplier for realistic_taker model (max slip = N * bar_slip)
    ioc_cap_multiplier: float = 2.0

    # Asymmetric entry threshold: if set > 0, longs require this edge instead of entry_threshold
    entry_threshold_long: float = 30.0  # longs require more edge (higher slippage)

    # Hard BTC position cap (applied after all sizing logic)
    max_position_btc: float = 0.0  # 0 = no hard cap (use dynamic_btc_cap only)

    # Execution delay: seconds between signal (bar close) and order submission.
    # Models realistic latency — spot drifts by vol*sqrt(delay) during the gap.
    execution_delay_s: float = 1.0

    # Vega stop-loss (lookahead version — DO NOT use for performance claims)
    vega_stop_usd: float = 0.0  # 0 = disabled

    # Honest minute-level vega stop: checks actual 1-min IV from trade tape.
    # Requires preprocessing: PYTHONPATH=. python scripts/build_minute_iv.py
    vega_stop_minute_usd: float = 0.0  # 0 = disabled; >0 = per-BTC loss cap ($)
    minute_iv_path: str = "vol_arb/output/minute_atm_iv.parquet"

    # ── Slippage-aware trade filters (applied inside PnL loop) ────
    # A1: Pre-trade expected profit filter — skip if estimated net < ratio * gross
    use_profit_filter: bool = True
    profit_filter_min_ratio: float = 0.0    # min est_net / |est_gross|
    profit_filter_edge_discount: float = 0.5  # model captures ~50% of predicted edge
    profit_filter_min_abs: float = 0.0      # min est_net_per_btc in $ (0 = ratio-only)
    # Empirical-Bayes shrinkage for est_gross: blend the realized per-BTC gross
    # EMA with the per-signal theoretical prior (edge*vega*discount). The blend
    # weight λ = count / (count + τ) grows smoothly with the number of realized
    # trades. This replaces the old hard cliff (count>=200 switch) which could
    # create a self-reinforcing deadlock: if the realized EMA landed below the
    # gate at warmup-end, the filter blocked all trades, which froze the EMA,
    # which kept the filter blocking. τ=200 matches the former warmup threshold
    # semantically (50/50 blend at 200 obs) and degenerates smoothly thereafter.
    profit_filter_shrinkage_tau: float = 200.0

    # A2: Slippage-adjusted dynamic threshold — adds slip-in-vol-pts to threshold
    use_slip_adjusted_threshold: bool = False
    slip_threshold_base_short: float = 25.0
    slip_threshold_base_long: float = 30.0

    # A3: Profit-ratio position scaling — scale N by expected profitability
    use_profit_scaling: bool = False
    profit_scaling_floor: float = 0.1       # min scale factor
    profit_scaling_cap: float = 1.0         # max scale factor

    # A4: Liquidity-regime filter — skip when slip > Pth percentile of recent window
    use_liquidity_regime_filter: bool = False
    liquidity_regime_percentile: float = 90.0
    liquidity_regime_window: int = 200

    # Deterministic SOR (print-through)
    use_sor: bool = True          # check taker prints for maker fills; False = always taker

    # Dynamic instrument routing (ATM straddle / bear strangle / bull strangle)
    # Two routing modes:
    #   "jump_skew" — legacy: gate on (down_jump - up_jump), units in [0, 1]
    #   "svi_skew"  — T2.2: gate on svi_25d_skew_1w from the SVI surface,
    #                  units in vol points (mean 3.8, std 6.0 on BTC 2022-2026)
    use_instrument_routing: bool = False  # ATM outperforms jump-skew routing per Stage 8.4
    routing_mode: str = "jump_skew"       # "jump_skew" | "svi_skew"
    jump_skew_threshold: float = 0.05     # |jump_skew| must exceed this
    svi_skew_threshold: float = 5.0       # |svi_25d_skew_1w| must exceed this (vol points)

    # Holding period
    holding_minutes: int = 15

    # ── Entry filters (W5.X production) ───────────────────────────
    # These filters mute signals BEFORE downstream sizing/exit logic
    # runs, so a muted row incurs zero fees/slippage. Defaults are
    # no-op so the reference BTC tearsheet reproduces byte-identical.
    #
    # min_dte_days: skip trades whose option DTE < this floor. Short
    # -dated trades have vega ∝ √DTE but per-trade fees are roughly
    # flat, so DTE<5 bleeds in 5/5 historical years (-$20/trade).
    # Set to 5.0 for BTC production; 0.0 disables.
    min_dte_days: float = 0.0

    # skip_long_hours_utc: mute ONLY long (+1) signals at these UTC
    # hours. Motivated by US equity cash-open (14:30 UTC DST) volatility
    # dislocation, where long-vol bets systematically fade 4/5 years
    # during 14-16 UTC. Shorts at those hours actually OUTPERFORM
    # the average — so this is strictly a long-side filter. Empty
    # tuple disables. Set to (14, 15, 16) for BTC production.
    skip_long_hours_utc: tuple[int, ...] = ()

    # skip_long_if_term_slope_gt: mute LONG signals when the SVI
    # term-slope feature (``svi_term_slope_1w_2w``, positive = IV
    # contango / 2w > 1w) exceeds this threshold. Motivated by audit
    # finding: longs in contango lose 5/5 years (−$87/trade), longs
    # in backwardation win 5/5 years (+$303/trade). Natural boundary
    # is the 0.0 contango/backwardation sign flip; no tuning needed.
    # None disables (default — reproduces legacy behavior).
    skip_long_if_term_slope_gt: float | None = None

    # ── DFL (Decision-Focused Learning) head filter ─────────────
    # When ``dfl_scores_path`` is set, load walk-forward P(win) scores
    # (per trade-fired bar) and MUTE bars whose P(win) falls below
    # ``dfl_filter_quantile`` of prior-year scores. Year-by-year
    # walk-forward threshold; bars in the first scored year always
    # pass through (no prior score history). See
    # ``vol_arb/dfl_head.py`` and
    # ``vol_model/output/DFL_HEAD_PHASE3_DECISION.md``. Defaults
    # preserve legacy behavior when ``dfl_scores_path is None``.
    dfl_scores_path: str | None = None
    dfl_filter_quantile: float = 0.25

    # ── Kelly-shrinkage sizing boost (Phase 2 from 2026-04-21) ───
    # When ``kelly_mult_path`` is set, load per-bar size multipliers
    # (parquet with a ``size_mult`` column indexed by bar timestamp;
    # always >= 1.0 — upside-only asymmetric boost on long-side high
    # |pred_edge| bars). Multiplier applies to ``N[i]`` AFTER
    # profit_scaling and BEFORE the dynamic_btc_cap clip so Kelly can
    # never breach the margin envelope. Walk-forward multipliers are
    # generated offline by ``scripts/generate_kelly_sidecar.py``
    # (wrapping phase1 math: Rising-Wyner shrinkage on |pred_edge|/sigma^2
    # with trailing-12m median normalization). Defaults to None ->
    # pass-through (byte-identical V2). See plan:
    # C:/Users/laure/.claude/plans/starry-rolling-rabin.md
    kelly_mult_path: str | None = None

    # ── DRL delta-hedging (experimental) ─────────────────────────
    # When True, ``compute_pnl`` routes the delta-hedge simulation through
    # ``vol_arb.drl_hedging.DRLHedger`` (a trained SAC policy) instead of
    # the rule-based ``simulate_discrete_hedging``.  Defaults to False so
    # existing backtests are unaffected.  See the DRL plan file for
    # details: C:/Users/laure/.claude/plans/flickering-wandering-locket.md
    use_drl_hedging: bool = False
    drl_policy_path: str = "vol_arb/drl_hedging/models/production_sac_v4_final.zip"
    drl_vecnormalize_path: str = "vol_arb/drl_hedging/models/production_sac_v4_final_vecnormalize.pkl"
    drl_svi_params_path: str = "vol_arb/output/svi_surface_params_1h.parquet"
    drl_fallback_on_error: bool = True  # fall back to rule-based on any DRL error

    # ── Directional classifier head gate (W4.1) ─────────────────
    # Optional sibling classifier that predicts sign(y_true - log IV) from
    # the same 44-feature matrix as T2.1. When ``classifier_path`` is set,
    # compute_pnl mutes bars whose primary signal is not confirmed by the
    # classifier's edge-direction prediction:
    #   - long-vol (+1): require direction_prob >= classifier_threshold
    #   - short-vol (-1): require direction_prob <= 1 - classifier_threshold
    # Bars with NaN classifier predictions (e.g. dates outside the
    # classifier's training window) are controlled by ``classifier_nan_is_pass``:
    # when True (default), they bypass the gate and trade as usual;
    # when False, they are muted — used for validation A/B comparisons.
    # The gate is applied after routing, before sizing, so muted bars
    # propagate as ``signal == 0`` with no downstream changes required.
    classifier_path: str | None = None
    classifier_threshold: float = Field(
        default=0.55, gt=0.5, lt=1.0,
        description="Classifier confidence threshold; must be in (0.5, 1.0).",
    )
    classifier_nan_is_pass: bool = True

    # ── Non-compounding path ─────────────────────────────────────
    # When True, compute_pnl attaches a second set of "_nc" columns that
    # represent a non-compounding run: every trade sized against the
    # fixed ``initial_equity`` reference notional rather than the running
    # equity.  Lets you separate signal quality from compounding effects.
    # Adds ~10-30 ms to a full backtest (vectorized post-hoc block).
    report_non_compounding: bool = True

    # ── Excess-logvol target (Phase 5) ───────────────────────────
    # When True, ``load_data`` interprets the ``y_pred`` column of the
    # predictions parquet as excess log-vol (i.e., log(RV/IV)) rather than
    # raw log-RV. Edge is then computed as ``(exp(y_pred) - 1) * iv_used``
    # in vol points. Default False preserves BTC identity — BTC predictions
    # remain raw log-RV and edge = ``exp(y_pred) * 100 - iv_used``.
    # Typically set via ``for_currency("ETH").with_excess_predictions(path)``
    # or by explicit kwarg on ETH runs whose predictions parquet is
    # ``{asset}_pruned_svi_predictions_excess.parquet``.
    predictions_are_excess_logvol: bool = False

    # ── Exit-IV attribution mode (Phase A — REVERTED to default-legacy) ──
    # The strategy is documented (TSRV/PIPELINE_DOCUMENTATION.md §17.3) as
    # ROLLING-ATM exposure: at each bar we are exposed to the current ATM
    # straddle, and at exit we are exposed to whatever is ATM then. The
    # original ``dIV = iv_exit_row - iv_entry_row`` is described in the doc
    # as "the original formula is exact for ATM straddles" under the
    # second-order "skew change in 15m is small" assumption. The default is
    # therefore the legacy (rolling-ATM) attribution.
    #
    # When ``use_pinned_exit_iv=True``, exit IV is reconstructed for the
    # ENTRY contract's exact (K, D) via a three-tier cascade:
    #   (1) Real same-contract trade within ``exit_iv_trade_tolerance_minutes``
    #   (2) SVI surface evaluated at (K_entry, D_entry, exit_target)
    #   (3) Fallback to legacy exit-row IV
    # This is the FIXED-CONTRACT interpretation — appropriate if you intend
    # to model the realized PnL of holding a specific contract for the full
    # 15min window with no re-strike. Use only with explicit awareness that
    # this changes the strategy's economic interpretation.
    use_pinned_exit_iv: bool = False
    # Tier-1 backward-tolerance (min) for same-contract exit trade.
    exit_iv_trade_tolerance_minutes: int = 15
    # Raw Deribit trades directory (for tier-1 exit IV lookup).
    option_trades_dir: str = "data/deribit/trades/BTC_option"

    @classmethod
    def for_timeframe(cls, bar_minutes: int = 15, **overrides) -> "BacktestConfig":
        """Construct a BacktestConfig with bar-count defaults derived from timeframe.

        Usage::

            cfg = BacktestConfig.for_timeframe(bar_minutes=5, initial_equity=200_000)
        """
        bpd = 1440 // bar_minutes
        defaults: dict = dict(
            holding_minutes=bar_minutes,
            vrp_window=bpd * 7,
            vrp_min_periods=bpd,
        )
        defaults.update(overrides)
        return cls(**defaults)

    @classmethod
    def for_currency(cls, currency: str = "BTC", **overrides) -> "BacktestConfig":
        """Build a BacktestConfig with per-currency path + calibration defaults.

        BTC overrides nothing (field defaults reproduce the legacy BTC run
        byte-identical).  ETH shifts vol-baseline-dependent knobs and points
        every data path at the corresponding ETH artifact via the resolver.

        Extra kwargs override anything the factory set — use for sweeps.
        """
        from vol_arb.paths import (
            bar_exec_quality_path, minute_atm_iv_path, option_trade_dir,
            pruned_svi_predictions_path, sac_model_dir, svi_surface_params_path,
            vol_arb_path,
        )

        ccy = currency.upper()
        if ccy == "BTC":
            # BTC field defaults reproduce legacy BTC behavior exactly.
            # The one explicit path needed is ``option_trades_dir`` for the
            # tier-1 trade-level exit IV lookup (added Phase A2b); already
            # defaults to the BTC directory via the class-level default.
            return cls(**overrides)

        if ccy == "ETH":
            asset = "ETHUSDT"
            defaults: dict = dict(
                # Paths
                vol_arb_path=str(vol_arb_path("ETH", svi=True)),
                predictions_path=str(pruned_svi_predictions_path(asset)),
                bar_exec_quality_path=str(bar_exec_quality_path("ETH")),
                drl_svi_params_path=str(svi_surface_params_path("ETH")),
                minute_iv_path=str(minute_atm_iv_path("ETH")),
                drl_policy_path=str(sac_model_dir(asset) / "best_model.zip"),
                drl_vecnormalize_path=str(sac_model_dir(asset) / "vecnormalize.pkl"),
                option_trades_dir=str(option_trade_dir("ETH")),
                # ETH-calibrated vol-dependent knobs
                baseline_target_dvol=75.0,
                max_jump_fraction_for_shorts=0.20,
            )
            defaults.update(overrides)
            return cls(**defaults)

        raise ValueError(f"Unsupported currency for BacktestConfig.for_currency: {currency!r}")

    def with_realistic_costs(self) -> "BacktestConfig":
        """Return a copy of this config with realistic Deribit 2026 cost widening.

        The default config fields reflect Deribit's nominal fee schedule (0.03%
        option both sides, 0.05% perp taker, 2 bps perp slippage, 1.5 vol pts
        base IV spread). Those numbers are the *tight-quote* assumption — real
        production execution pays more: wider BBO spreads on ATM options,
        higher market impact on perp hedges, and systematic adverse selection
        on maker fills.

        This method widens the cost knobs to a realistic baseline without
        touching fee field defaults (so the BTC reference tearsheet generated
        with ``run_backtest.py`` continues to reproduce byte-identical).

        Usage::

            cfg = BacktestConfig.for_currency("BTC").with_realistic_costs()
            cfg = BacktestConfig.for_currency("ETH").with_realistic_costs()

        Deltas from defaults (first-principles, moderate):

        - ``perp_slippage_bps``     2.0  → 3.0   (modest widen; 3 bps is
          around the 60th-percentile market impact for $1-10K BTC perp taker
          orders in normal conditions, up from the 2 bps "tight" baseline)
        - ``base_spread_iv``        1.5  → 2.0   (reflects typical ATM
          short-dated BBO of 2 vol pts rather than tight-quote 1.5; wider in
          stress is handled by ``spread_jump_k`` dynamic widening)
        - ``maker_adverse_selection_iv`` 0.5 → 0.75 (modest widen reflecting
          that maker fills preferentially happen when the direction was right)
        - ``use_profit_filter``     True → False (under realistic costs the
          filter's built-in 50% edge-capture assumption creates pathological
          adverse selection — it lets through only the highest-edge trades
          but those systematically over-predict edge, so the surviving set
          underperforms the unfiltered set. Live execution cannot perfectly
          pre-estimate net edge per bar; disabling this filter is the honest
          production view.)
        """
        return self.model_copy(update=dict(
            perp_slippage_bps=3.0,
            base_spread_iv=2.0,
            maker_adverse_selection_iv=0.75,
            use_profit_filter=False,
        ))

    def with_btc_production_filters(self) -> "BacktestConfig":
        """Return a copy with the production BTC filter/threshold stack.

        Discovered 2026-04-16 via exhaustive trade-log audit + walk-forward
        OOS validation on 2022-2026:

        - ``entry_threshold=22.0``, ``entry_threshold_long=30.0`` —
          asymmetric thresholds (shorts show consistently lower fee break-
          even than longs at the same edge magnitude).
        - ``min_dte_days=5.0`` — DTE<5 bleeds 5/5 years on fee-vs-vega
          economics (flat fees, vega ∝ √DTE).
        - ``skip_long_hours_utc=(14, 15, 16)`` — long-only skip for US
          cash-open regime dislocation. Shorts in those hours outperform
          and are kept.
        - ``target_margin_utilization=0.075`` — scales PnL to 10%
          peak-to-trough DD cap (was 5% default, leaves capacity on
          the table).
        - ``use_sor=False`` — taker-only. Maker fills observed ~1.6% rate
          and add only marginal Sharpe, not worth the adverse-selection
          risk for live execution.

        Expected vs baseline (under `with_realistic_costs()` + filters):
          ~$310K NC PnL (5y) vs $128K baseline, Sharpe 8.3 vs 4.1,
          MDD ~9% vs unbounded-scale, all 5 years positive.
        """
        return self.model_copy(update=dict(
            entry_threshold=22.0,
            entry_threshold_long=30.0,
            min_dte_days=5.0,
            skip_long_hours_utc=(14, 15, 16),
            target_margin_utilization=0.075,
            use_sor=False,
        ))

    def with_btc_production_filters_v2(self) -> "BacktestConfig":
        """Return a copy with the v2 production stack (2026-04-16).

        Builds on v1 by adding four independently-validated improvements,
        each with stated mechanism and walk-forward OOS confirmation:

        1. ``use_drl_hedging=True`` — SAC v4 policy cuts hedge cost 58%
           (Phase A.1, this session; model already adopted per
           ``project_w5_v4_adopted.md``).
        2. ``use_profit_scaling=True``, ``profit_scaling_floor=0.3`` —
           conviction-weighted position sizing (Phase A.4).
        3. ``target_margin_utilization=0.15`` (up from 0.075) — capital-
           scale against lower MDD from profit_scaling, recovered PnL
           to exceed v1 (Phase A.4 combo).
        4. ``holding_minutes=15`` — matches the model's prediction
           horizon ``[D, D+15m]`` exactly, so the trade's exposure
           window is the SAME 15-min bar the model was trained to
           forecast. A prior iteration used ``holding_minutes=30``
           (Phase B.2, +43% PnL / +14% OOS Sharpe) but that gain was
           dominated by theta carry over the second (un-forecast)
           15-min bar, not by signal edge extraction. Holding to
           15m disentangles signal quality from theta-carry and
           gives a foundation where PnL directly reflects forecast
           skill. Foundation fix 2026-04-17; see
           ``vol_model/output/FOUNDATION_ALIGNMENT_AUDIT.md``.
        5. ``skip_long_if_term_slope_gt=0.0`` — drop LONG signals in IV
           contango (svi_term_slope_1w_2w > 0). 5/5 years negative in
           contango longs vs 5/5 positive in backwardation. Natural
           0.0 boundary (no tuning). (Phase C.5).
        6. ``entry_threshold_long=12.0`` (down from v1's 30.0) — the
           contango-long filter already scrubbed the bad-regime
           longs, so the remaining (backwardation-only) longs have
           structural positive edge and can fire at a lower edge
           floor. Sweep showed monotonic PnL rise as thr_L drops
           30 → 12 with MDD staying at ~10%; below 12 the MDD cap
           binds (thr_L=10 → 10.33%, thr_L=8 → 10.52%). The asymmetry
           thr_S=22 / thr_L=12 LOOKS inverted but is NOT — the long
           filter stack has already removed the bad regimes, so the
           surviving long bar has a higher conditional edge than an
           unfiltered short bar at the same edge magnitude.

        Sizing model kept at ``composite_v2`` (v1 default). Kelly
        was tested in Phase B.3 and marginally beat composite_v2 on
        NC Sharpe (+0.30) but interacted badly with compound-mode
        equity scaling (compound end-equity collapsed from ~40× to
        ~2× while NC numbers looked better). composite_v2 preserves
        the compound-path growth, which matters for real capital
        deployment.

        Expected v2 vs v1 baseline:
          v1: $308K NC over 5y, Sharpe 8.31, MDD 9.01%,
              compound $4.06M (41.6x on $100K)
          v2 (pre-foundation-fix, hold=30m, historical): $726K NC,
              Sharpe 12.56, MDD 9.91%. Superseded by foundation fix.
          v2 (post-foundation-fix, hold=15m, current): numbers land
              after regen of BTC_v2_trade_log.parquet; see
              ``vol_model/output/FOUNDATION_ALIGNMENT_AUDIT.md``.
        """
        return self.with_btc_production_filters().model_copy(update=dict(
            **self._v2_horizon_invariants(),
            holding_minutes=15,
            # Kelly-shrinkage long-side boost adopted 2026-04-21 as
            # an accepted Sharpe-for-PnL tradeoff. Phase 2 backtest:
            # NC PnL +$114K (+13.5%) and Compound PnL +$2.84M (+17.9%)
            # at NC Sharpe -0.94 and Compound Sharpe -0.20 (Compound
            # MDD actually improved 0.62pp). $/Sharpe ratio is 1.7x
            # better than the aligner-adoption template the user
            # accepted on 2026-04-20. See
            # vol_arb/output/KELLY_SHRINKAGE_PHASE2_REPORT.md and
            # memory/project_kelly_sizing_2026_04_21.md.
            kelly_mult_path=(
                "vol_arb/output/BTC_v2_kelly_mult_K1.parquet"
            ),
        ))

    @staticmethod
    def _v2_horizon_invariants() -> dict:
        """V2-stack fields that DO NOT depend on prediction horizon.

        Factored out so per-horizon presets (`with_btc_production_h{N}m`)
        can apply the same V2 stack without duplicating the field list.

        Excluded (because horizon-coupled): ``holding_minutes``,
        ``predictions_path``, ``kelly_mult_path``, ``vrp_window``,
        ``vrp_min_periods``. Those are set by `for_timeframe()` (vrp) and
        per-horizon path helpers (predictions, kelly).
        """
        return dict(
            entry_threshold_long=12.0,
            use_drl_hedging=True,
            use_profit_scaling=True,
            profit_scaling_floor=0.3,
            target_margin_utilization=0.15,
            # sizing_model stays "composite_v2" (the v1 default) —
            # native Kelly strategy tested but broke compound-mode
            # path. The Kelly-shrinkage sidecar (kelly_mult_path) is a
            # different mechanism: a per-bar UPSIDE-ONLY multiplier
            # (>=1.0) applied AFTER profit_scaling and BEFORE
            # dynamic_btc_cap clip. Walk-forward, leakage-tested.
            skip_long_if_term_slope_gt=0.0,
        )

    @classmethod
    def with_btc_production_for_horizon(cls, horizon_minutes: int) -> "BacktestConfig":
        """Compose the V2 BTC stack at an arbitrary prediction horizon.

        Used by `with_btc_production_h{5,30,60}m()` per-horizon presets.
        Chain:
          1. `for_timeframe(N)` → holding_minutes / vrp_window / vrp_min_periods.
          2. `with_btc_production_filters()` → V1 filter stack
             (thresholds, DTE, hour skip, util floor, taker-only).
          3. `_v2_horizon_invariants()` → V2 risk stack
             (DRL, profit_scaling, util=0.15, contango filter, thr_L=12).
          4. Horizon-aware path overrides: aligned-bias HAR-PD predictions
             + Kelly K1 sidecar resolved through `vol_arb.paths` helpers.

        Sidecars at h≠15 must be regenerated separately; this preset
        merely points at the convention-named locations. A missing file
        will surface as a backtester load error rather than silently
        falling back to a wrong-horizon sidecar.

        See plan starry-rolling-rabin.md Phase 1f.
        """
        from vol_arb.paths import (
            bar_exec_quality_path, har_pd_predictions_path, kelly_sidecar_path,
            vol_arb_path,
        )

        if 1440 % horizon_minutes != 0:
            raise ValueError(
                f"horizon_minutes={horizon_minutes} does not divide 1440; "
                f"choose a value such that bars_per_day is integer."
            )
        base = cls.for_timeframe(horizon_minutes).with_btc_production_filters()
        # ``.as_posix()`` (not ``str()``) so paths land in the same
        # forward-slash form the legacy literal field defaults use; this
        # keeps stored paths cross-platform identical (Windows would
        # otherwise stringify Path objects with backslashes and silently
        # diverge from the literal "vol_model/output/..." defaults).
        #
        # VPIN: NOT threaded here because the backtester computes VPIN
        # on-the-fly from returns (see compute_vpin_proxy_from_returns
        # call at L1501). VPIN parquets at per-horizon paths exist only
        # for DRL *training*, not for backtest-time inference.
        return base.model_copy(update=dict(
            **cls._v2_horizon_invariants(),
            # holding_minutes already set by for_timeframe(); do NOT override.
            predictions_path=har_pd_predictions_path(
                "BTCUSDT", horizon_minutes, aligned_bias=True,
            ).as_posix(),
            kelly_mult_path=kelly_sidecar_path(
                "BTC", horizon_minutes, variant="K1",
            ).as_posix(),
            bar_exec_quality_path=bar_exec_quality_path(
                "BTC", horizon_minutes=horizon_minutes,
            ).as_posix(),
            vol_arb_path=vol_arb_path(
                "BTC", svi=True, horizon_minutes=horizon_minutes,
            ).as_posix(),
        ))

    @classmethod
    def with_btc_production_h5m(cls) -> "BacktestConfig":
        """V2 BTC stack at h=5m. Sidecars must be regenerated per Phase 4."""
        return cls.with_btc_production_for_horizon(5)

    @classmethod
    def with_btc_production_h30m(cls) -> "BacktestConfig":
        """V2 BTC stack at h=30m. Sidecars must be regenerated per Phase 4."""
        return cls.with_btc_production_for_horizon(30)

    @classmethod
    def with_btc_production_h60m(cls) -> "BacktestConfig":
        """V2 BTC stack at h=60m. Sidecars must be regenerated per Phase 4."""
        return cls.with_btc_production_for_horizon(60)


# ── Discrete Delta Hedging Simulator ──────────────────────────

def simulate_discrete_hedging(
    S_entry: np.ndarray,
    K: np.ndarray,
    sigma: np.ndarray,
    T_years: np.ndarray,
    vol_ann: np.ndarray,
    signal: np.ndarray,
    n_steps: int = 10,
    delta_hedge_band: float = 0.05,
    perp_fee: float = 0.0004,
    perp_slippage_bps: float = 2.0,
    seed: int = 42,
    *,
    delta_entry: np.ndarray | None = None,
    gamma_entry: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate path-dependent discrete delta hedging for all windows.

    For each 15-minute window, generates a random walk price path scaled by
    the window's realized vol (from TSRV). At each step, updates the
    position's delta via the constant-gamma approximation and rebalances
    when residual exceeds ``delta_hedge_band``.

    Two modes:

    - **Legacy** (``delta_entry=None``, ``gamma_entry=None``): computes
      delta from the ATM-straddle BS formula ``2*N(d1) - 1`` using K and
      sigma.  Correct for ATM straddles at strike K with IV sigma.
      Incorrect for non-ATM composite instruments (e.g. bear/bull
      strangles) because the "position" isn't actually a straddle at K.

    - **Greeks-explicit** (``delta_entry``, ``gamma_entry`` provided): uses
      the position's precomputed BS Greeks directly, then updates the
      delta along the simulated path via the constant-gamma linear
      approximation ``delta(t) = delta_0 + gamma * (S(t) - S_entry)``.
      This is the correct formulation for ANY composite (straddle,
      strangle, risk reversal) because it doesn't assume the position is
      a single option at K.  Over 15-minute windows the constant-gamma
      approximation is a <1% higher-order error.

    Args:
        S_entry: Entry spot price per window (n,)
        K: Strike per window (n,) — only used when ``delta_entry`` is None
        sigma: Entry IV in decimal per window (n,) — only used when
            ``delta_entry`` is None
        T_years: Time to expiry in years per window (n,) — only used when
            ``delta_entry`` is None
        vol_ann: Realized annualized vol in decimal per window (n,)
        signal: Trade direction +1/-1/0 per window (n,)
        n_steps: Price path steps per 15m window
        delta_hedge_band: BTC delta trigger threshold
        perp_fee: Taker fee on perp hedges (fraction)
        perp_slippage_bps: Slippage on perp hedges (bps)
        seed: RNG seed for reproducibility
        delta_entry: Position delta per BTC at entry (n,), optional. If
            provided (together with ``gamma_entry``), bypasses the
            legacy BS recomputation and uses this as the starting delta.
            Convention: LONG position's delta (e.g., +0.0 for ATM
            straddle, +0.25 for bear strangle).  The function applies
            ``signal`` to get the signed position delta.
        gamma_entry: Position gamma per BTC at entry (n,), optional.
            Required iff ``delta_entry`` is provided.  Convention: LONG
            position's gamma (positive).

    Returns:
        net_delta_pnl: Path-dependent delta/gamma P&L per BTC (n,)
        hedge_costs: Total discrete hedge transaction costs per BTC (n,)
    """
    from scipy.stats import norm

    n = len(S_entry)
    rng = np.random.RandomState(seed)

    # Time per step (15 min / n_steps, in years)
    dt_step = (15.0 / n_steps) / (365.25 * 24 * 60)
    sqrt_dt = np.sqrt(dt_step)

    # Per-step volatility from realized vol
    step_sigma = vol_ann * sqrt_dt  # (n,)

    # Random increments for the full path: (n, n_steps)
    dW = rng.standard_normal((n, n_steps))

    # Decide delta trajectory mode up-front
    use_greeks_explicit = (
        delta_entry is not None and gamma_entry is not None
    )

    if use_greeks_explicit:
        # Explicit mode: caller provides per-BTC Greeks.  delta_entry/gamma_entry
        # are the LONG position's values; signal is applied here to get the
        # signed position delta.
        d0_long = np.asarray(delta_entry, dtype=np.float64)
        g0_long = np.asarray(gamma_entry, dtype=np.float64)
        # No BS recomputation needed — skip the sigma_sqrt_T / half_sigma2_T
        # precompute that the legacy path uses.
        sigma_sqrt_T = None  # sentinel for safety
        half_sigma2_T = None
    else:
        # Legacy mode: recompute delta from BS straddle formula at strike K.
        safe_sigma = np.where(sigma > 0, sigma, 0.01)
        safe_T = np.where(T_years > 0, T_years, 0.001)
        sigma_sqrt_T = safe_sigma * np.sqrt(safe_T)
        half_sigma2_T = 0.5 * safe_sigma**2 * safe_T

    # Cost rate per hedge trade
    cost_rate = perp_fee + perp_slippage_bps / 10_000

    # State arrays
    S_current = S_entry.copy()
    hedge_position = np.zeros(n)       # BTC of perps held (signed)
    net_delta_pnl = np.zeros(n)        # cumulative delta P&L from residual exposure
    hedge_costs = np.zeros(n)          # cumulative transaction costs
    active = signal != 0               # mask for active trades

    # Initial hedge: hedge the entry position's delta
    if use_greeks_explicit:
        init_delta = signal * d0_long
    else:
        d1_init = (np.log(S_entry / K) + half_sigma2_T) / sigma_sqrt_T
        init_delta = signal * (2.0 * norm.cdf(d1_init) - 1.0)
    # Hedge to zero: short perps equal to position delta
    hedge_position = np.where(active, init_delta, 0.0)
    hedge_costs += np.where(active, np.abs(init_delta) * S_entry * cost_rate, 0.0)

    for j in range(n_steps):
        # Price step (geometric Brownian motion)
        S_new = S_current * np.exp(step_sigma * dW[:, j] - 0.5 * step_sigma**2)

        dS = S_new - S_current

        # New position delta at updated price
        if use_greeks_explicit:
            # Linear constant-gamma update: delta(t) = delta_0 + gamma * (S-S_0)
            current_delta = signal * (d0_long + g0_long * (S_new - S_entry))
        else:
            d1 = (np.log(S_new / K) + half_sigma2_T) / sigma_sqrt_T
            current_delta = signal * (2.0 * norm.cdf(d1) - 1.0)

        # Residual (unhedged) delta = position delta - our hedge position
        residual = current_delta - hedge_position

        # P&L from residual delta exposure during this price step
        net_delta_pnl += np.where(active, residual * dS, 0.0)

        # Check if rebalance needed
        needs_hedge = np.abs(residual) > delta_hedge_band

        # Execute hedge: trade the residual to bring net delta back to zero
        trade_size = np.where(needs_hedge & active, residual, 0.0)
        hedge_costs += np.abs(trade_size) * S_new * cost_rate
        hedge_position += trade_size

        S_current = S_new

    return net_delta_pnl, hedge_costs


# ── Backtester ─────────────────────────────────────────────────

class VolArbBacktester:
    """Fully vectorized delta-neutral straddle vol-arb simulator."""

    def __init__(self, config: BacktestConfig | None = None):
        self.cfg = config or BacktestConfig()

    # ── Step 1: Load & prepare data ────────────────────────────

    def load_data(self) -> pd.DataFrame:
        """Load vol_arb merged dataset + join OOF predictions."""
        # Load merged dataset
        df = pd.read_parquet(self.cfg.vol_arb_path)
        logger.info("Loaded vol_arb: %d rows x %d cols", len(df), df.shape[1])

        # Join OOF predictions
        oof = pd.read_parquet(self.cfg.predictions_path)
        df = df.join(oof[["y_pred"]], how="left")
        n_pred = df["y_pred"].notna().sum()
        logger.info("Joined predictions: %d / %d bars (%.1f%%)",
                    n_pred, len(df), 100 * n_pred / len(df))

        # Optional classifier head predictions (W4.1). Rename to ``clf_prob``
        # to namespace it in the working DataFrame and avoid any collision
        # with the vol predictions' ``y_pred``.
        if self.cfg.classifier_path:
            clf = pd.read_parquet(self.cfg.classifier_path)
            if "direction_prob" not in clf.columns:
                raise KeyError(
                    f"Classifier parquet at {self.cfg.classifier_path} "
                    "missing required column 'direction_prob'.",
                )
            df = df.join(
                clf[["direction_prob"]].rename(
                    columns={"direction_prob": "clf_prob"},
                ),
                how="left",
            )
            n_clf = df["clf_prob"].notna().sum()
            logger.info(
                "Joined classifier predictions: %d / %d bars (%.1f%%)",
                n_clf, len(df), 100 * n_clf / len(df),
            )

        # IV fallback chain: maturity-matched SVI IV → trade IV → DVOL
        # svi_iv_matched is SVI surface evaluated at the straddle's exact (k, T),
        # so it's directly comparable to trade IV (unlike nearest-expiry svi_atm_vol).
        if "svi_iv_matched" in df.columns:
            df["iv_used"] = df["svi_iv_matched"].fillna(df["iv"]).fillna(df["dvol_close"])
        else:
            df["iv_used"] = df["iv"].fillna(df["dvol_close"])

        # Convert log prediction to annualized RV percentage.
        # Two target conventions:
        #   - raw log-RV (default): y_pred = log(RV_future_decimal)
        #                           rv_pred_pct = exp(y_pred) * 100
        #                           edge       = rv_pred_pct - iv_used  (computed below)
        #   - excess log-RV (Phase 5): y_pred = log(RV_future / IV_at_T)
        #                              rv_pred_pct = iv_used * exp(y_pred)
        #                              (edge = rv_pred_pct - iv_used naturally reduces to
        #                               (exp(y_pred) - 1) * iv_used, i.e., vol-points edge)
        # Selecting between the two is driven by cfg.predictions_are_excess_logvol
        # so no separate code path is needed downstream.
        if self.cfg.predictions_are_excess_logvol:
            df["rv_pred_pct"] = df["iv_used"] * np.exp(df["y_pred"])
            logger.info(
                "Excess-logvol predictions: rv_pred_pct = iv_used * exp(y_pred). "
                "Edge = (exp(y_pred) - 1) * iv_used.",
            )
        else:
            df["rv_pred_pct"] = np.exp(df["y_pred"]) * 100

        # VRP baseline (optional)
        if self.cfg.use_vrp:
            # Lag vol_ann by one bar to avoid lookahead: vol_ann[T] is the
            # realized vol OF bar T (open-stamped), so using it at decision
            # time T would be forward-looking.  shift(1) uses the previous
            # bar's realized vol, which is known at time T.
            df["hist_vol_pct"] = df["vol_ann"].shift(1) * 100
            df["vrp_raw"] = df["iv_used"] - df["hist_vol_pct"]
            df["vrp_baseline"] = df["vrp_raw"].rolling(
                self.cfg.vrp_window, min_periods=self.cfg.vrp_min_periods
            ).mean()
            logger.info("VRP baseline: mean=%.2f, median=%.2f vol pts",
                        df["vrp_baseline"].mean(), df["vrp_baseline"].median())
        else:
            df["vrp_baseline"] = 0.0

        # Drop rows without signal inputs
        mask = (
            df["rv_pred_pct"].notna()
            & df["iv_used"].notna()
            & df["vega"].notna()
        )
        if self.cfg.use_vrp:
            mask = mask & df["vrp_baseline"].notna()
        df = df[mask].copy()
        logger.info("Tradeable bars after filtering: %d", len(df))

        return df

    # ── Step 2: Dynamic spread ─────────────────────────────────

    def compute_spread(self, jump_fraction: np.ndarray) -> np.ndarray:
        """Dynamic IV spread that widens during market stress."""
        jf = np.clip(jump_fraction, 0.0, 1.0)
        return self.cfg.base_spread_iv * np.exp(self.cfg.spread_jump_k * jf)

    # ── Step 3: Signal generation ──────────────────────────────

    def generate_signals(
        self,
        edge: np.ndarray,
        half_spread: np.ndarray,
        jump_fraction: np.ndarray,
    ) -> np.ndarray:
        """Vectorized signal: +1 long straddle, -1 short straddle, 0 flat.

        Circuit breaker: short signals killed when jump_fraction exceeds
        max_jump_fraction_for_shorts (liquidation cascade protection).
        """
        threshold_short = self.cfg.entry_threshold
        threshold_long = (self.cfg.entry_threshold_long
                          if self.cfg.entry_threshold_long > 0
                          else self.cfg.entry_threshold)
        signal = np.where(
            edge > half_spread + threshold_long, 1,
            np.where(edge < -(half_spread + threshold_short), -1, 0),
        ).astype(np.int8)

        # Circuit breaker: kill short vol during high-jump regimes
        signal = np.where(
            (signal == -1) & (jump_fraction > self.cfg.max_jump_fraction_for_shorts),
            0, signal,
        ).astype(np.int8)

        return signal

    # ── Step 3b: Entry filters (W5.X) ──────────────────────────

    def _apply_entry_filters(self, df: pd.DataFrame) -> np.ndarray:
        """Mute signals per cfg.min_dte_days and cfg.skip_long_hours_utc.

        Runs AFTER generate_signals, so a filtered bar appears as
        ``signal == 0`` downstream (no fees/slippage incurred). When both
        config fields are at their no-op defaults the returned array is
        identical to ``df["signal"]`` so the BTC reference tearsheet
        reproduces byte-identical.

        Filters applied (each guarded on config):

        - ``min_dte_days`` > 0: mute ALL signals (long and short) on bars
          whose option DTE is below the floor. Rationale: per-trade option
          fees are roughly DTE-invariant (% of notional) while vega scales
          as √DTE, so short-dated straddles need outsized IV moves to
          clear costs. Empirically DTE<5 bleeds in 5/5 years on BTC.

        - ``skip_long_hours_utc`` non-empty: mute LONG (+1) signals when
          the bar's UTC hour is in the set. Shorts at those hours are
          KEPT (they outperform). Rationale: US cash-market open
          (14:30 UTC during DST) systematically fades long-vol bets in
          4/5 historical years.
        """
        signal = df["signal"].values.copy().astype(np.int8)

        if self.cfg.min_dte_days > 0:
            if "dte_days" in df.columns:
                dte_mask = df["dte_days"].values < float(self.cfg.min_dte_days)
                n_muted = int((dte_mask & (signal != 0)).sum())
                if n_muted > 0:
                    signal = np.where(dte_mask, np.int8(0), signal).astype(np.int8)
                    logger.info(
                        "Entry filter (DTE): muted %d signals (dte < %.1f days)",
                        n_muted, float(self.cfg.min_dte_days),
                    )
            else:
                logger.warning(
                    "min_dte_days=%.1f set but 'dte_days' column absent — skipping",
                    float(self.cfg.min_dte_days),
                )

        if self.cfg.skip_long_hours_utc:
            hours = pd.DatetimeIndex(df.index).hour.values
            hour_set = np.asarray(sorted(set(self.cfg.skip_long_hours_utc)))
            hour_mask = np.isin(hours, hour_set)
            long_at_bad_hour = hour_mask & (signal == 1)
            n_muted = int(long_at_bad_hour.sum())
            if n_muted > 0:
                signal = np.where(long_at_bad_hour, np.int8(0), signal).astype(np.int8)
                logger.info(
                    "Entry filter (long hours): muted %d longs (UTC hour in %s)",
                    n_muted, list(hour_set),
                )

        if self.cfg.skip_long_if_term_slope_gt is not None:
            if "svi_term_slope_1w_2w" in df.columns:
                slope = df["svi_term_slope_1w_2w"].values
                thr = float(self.cfg.skip_long_if_term_slope_gt)
                # NaN-safe: NaN > thr is False, so NaN slope rows pass through.
                slope_mask = np.where(np.isnan(slope), False, slope > thr)
                long_at_contango = slope_mask & (signal == 1)
                n_muted = int(long_at_contango.sum())
                if n_muted > 0:
                    signal = np.where(long_at_contango, np.int8(0), signal).astype(np.int8)
                    logger.info(
                        "Entry filter (term slope): muted %d longs "
                        "(svi_term_slope_1w_2w > %.2f)",
                        n_muted, thr,
                    )
            else:
                logger.warning(
                    "skip_long_if_term_slope_gt=%.2f set but "
                    "'svi_term_slope_1w_2w' column absent — skipping",
                    float(self.cfg.skip_long_if_term_slope_gt),
                )

        # DFL head filter — walk-forward per-year quantile threshold.
        # Mutes both longs and shorts whose P(win) falls below the
        # prior-year empirical quantile. Bars in the first scored
        # year pass through (no prior history). No lookahead: the
        # scores parquet is a product of walk-forward training per
        # ``vol_arb/dfl_head.compute_walkforward_scores``.
        if self.cfg.dfl_scores_path is not None:
            scores_path = Path(self.cfg.dfl_scores_path)
            if not scores_path.exists():
                logger.warning(
                    "dfl_scores_path=%s does not exist — skipping DFL filter",
                    scores_path,
                )
            else:
                dfl = pd.read_parquet(scores_path)
                # Normalize: single-column 'p_win' or one-column frame
                if "p_win" in dfl.columns:
                    p_win = dfl["p_win"]
                else:
                    p_win = dfl.iloc[:, 0]
                # Align to df.index
                p_aligned = p_win.reindex(df.index)
                q = float(self.cfg.dfl_filter_quantile)
                years = sorted(set(df.index.year))
                muted_mask = np.zeros(len(df), dtype=bool)
                for y in years:
                    train_mask = p_aligned.index.year < y
                    train_scores = p_aligned.loc[train_mask].dropna()
                    if len(train_scores) < 100:
                        continue  # pass through first scored year
                    thr_q = float(train_scores.quantile(q))
                    year_mask = np.asarray(p_aligned.index.year == y)
                    year_scores = p_aligned.to_numpy()
                    below = (year_scores < thr_q) & year_mask & ~np.isnan(year_scores)
                    muted_mask |= below
                dfl_mute = muted_mask & (signal != 0)
                n_muted = int(dfl_mute.sum())
                if n_muted > 0:
                    signal = np.where(dfl_mute, np.int8(0), signal).astype(np.int8)
                    logger.info(
                        "Entry filter (DFL head): muted %d bars "
                        "(P(win) < train_q%d, scores from %s)",
                        n_muted, int(q * 100), scores_path.name,
                    )

        return signal

    # ── Step 4: Chronological exit alignment ───────────────────

    def align_exits(self, df: pd.DataFrame) -> pd.DataFrame:
        """merge_asof: find market state exactly holding_minutes forward.

        Default: rolling-ATM attribution per TSRV/PIPELINE_DOCUMENTATION.md
        §17.3 — ``iv_exit`` is the next bar's ``iv_used`` (the ATM IV at
        exit time, possibly at a different K than entry). The doc states
        this formulation is "exact for ATM straddles" under the second-
        order "skew change in 15m is small" assumption.

        When ``cfg.use_pinned_exit_iv=True``, the fixed-contract three-tier
        cascade fires (real same-contract trade → SVI surface at entry K →
        legacy fallback). This represents the alternative interpretation
        where the trader holds a specific (K, D) contract for the full
        holding period and we want to model that contract's evolved IV.
        """
        # Build exit lookup table — always computes the legacy exit IV so
        # we can fall back to it if the surface reconstruction misses.
        exit_lookup = pd.DataFrame({
            "ts": df.index,
            "iv_exit_legacy": df["iv_used"].values,
            "price_exit": df["index_price"].values,
        })

        # Entry with target exit timestamp
        entry = df.reset_index()
        entry["exit_target"] = entry["timestamp"] + pd.Timedelta(
            minutes=self.cfg.holding_minutes
        )

        # merge_asof backward: closest exit_lookup.ts <= exit_target
        aligned = pd.merge_asof(
            entry.sort_values("exit_target"),
            exit_lookup.sort_values("ts"),
            left_on="exit_target",
            right_on="ts",
            direction="backward",
        )

        # Restore original order
        aligned = aligned.set_index("timestamp").sort_index()

        # Default = rolling-ATM (legacy iv_exit). Opt-in pinned via flag.
        if self.cfg.use_pinned_exit_iv:
            aligned["iv_exit"] = self._reconstruct_exit_iv(aligned)
        else:
            aligned["iv_exit"] = aligned["iv_exit_legacy"]

        # Drop last bar / any remaining trades with no valid exit IV
        aligned = aligned[aligned["iv_exit"].notna()]

        return aligned

    def _reconstruct_exit_iv(self, aligned: pd.DataFrame) -> pd.Series:
        """Three-tier exit IV cascade for the ENTRY contract at T+holding_minutes.

        Tier 1 — Same-contract trade-level IV (``_exit_iv_from_trades``):
          Find the most recent real trade of the ENTRY contract's exact
          (strike, expiry_dt, option_type) at or before ``exit_target``,
          within ``cfg.exit_iv_trade_tolerance_minutes``. Uses raw Deribit
          trades via a vectorized ``merge_asof(by=[...], direction='backward',
          tolerance=...)``. Ground-truth market mid. Preferred when
          available.

        Tier 2 — SVI-surface reconstruction (``_exit_iv_from_surface``):
          Evaluate the calibrated SVI surface at ``exit_target`` for the
          entry (strike, expiry_dt). Works even when no trade fired for the
          specific contract in the exit window. Coarseness limited by the
          surface's bucket_freq — within-bucket evaluations yield the same
          IV as entry, so ``dIV → 0`` under coarse-bucket surfaces. Best
          complement to Tier 1 for surface-confident but trade-sparse exits.

        Tier 3 — Legacy exit-row IV (``iv_exit_legacy`` column):
          Pre-fix behavior; the exit IV may reference a DIFFERENT contract
          than the one entered. Used only when both Tier 1 and Tier 2 miss.

        Fully vectorized end to end. Logs per-tier fill counts.
        """
        legacy = aligned["iv_exit_legacy"]

        # Tier 1
        iv_trade = self._exit_iv_from_trades(aligned)
        # Tier 2
        iv_surface = self._exit_iv_from_surface(aligned)

        # Combine: trade > surface > legacy
        filled = iv_trade.fillna(iv_surface).fillna(legacy)

        n_trade = iv_trade.notna().sum()
        n_surface_add = (iv_trade.isna() & iv_surface.notna()).sum()
        n_legacy_fallback = (iv_trade.isna() & iv_surface.isna() & legacy.notna()).sum()
        n_total = len(aligned)
        logger.info(
            "Exit IV cascade: tier1 trade %d/%d (%.1f%%), tier2 surface +%d "
            "(%.1f%%), tier3 legacy +%d (%.1f%%).",
            n_trade, n_total, 100 * n_trade / max(n_total, 1),
            n_surface_add, 100 * n_surface_add / max(n_total, 1),
            n_legacy_fallback, 100 * n_legacy_fallback / max(n_total, 1),
        )
        return filled

    def _exit_iv_from_trades(self, aligned: pd.DataFrame) -> pd.Series:
        """Tier 1: same-contract trade-level exit IV.

        Loads raw trades once (cached on the backtester instance), then for
        each entry row does a single ``merge_asof`` keyed by
        (strike, expiry_dt, option_type) with ``direction='backward'`` +
        tolerance to find the most recent trade of THAT contract at or
        before ``exit_target``. Returns NaN where no trade is within
        tolerance.

        Tolerance is ``cfg.exit_iv_trade_tolerance_minutes``. At default 60
        min and a 15m holding period, this allows exit lookup of trades up
        to 60 min after entry. Within this window the IV estimate is the
        real market mid, not a model reconstruction.
        """
        trades_dir = Path(self.cfg.option_trades_dir)
        if not trades_dir.exists():
            logger.warning(
                "Exit-IV tier 1 (same-contract trade) disabled: trades dir "
                "%s does not exist. Using tier 2/3 only.", trades_dir,
            )
            return pd.Series(np.nan, index=aligned.index)

        # Lazy-load + cache the prepared trades lookup.
        if not hasattr(self, "_trades_lookup_cache") or self._trades_lookup_cache is None:
            from vol_arb.options_loader import load_option_trades
            logger.info("Loading raw option trades for exit-IV tier 1 from %s...", trades_dir)
            trades = load_option_trades(trades_dir)
            # Minimal columns: (timestamp, strike, expiry_dt, option_type, iv)
            keep = ["timestamp", "strike", "expiry_dt", "option_type", "iv"]
            missing = [c for c in keep if c not in trades.columns]
            if missing:
                logger.warning(
                    "load_option_trades missing columns %s; tier 1 disabled.",
                    missing,
                )
                self._trades_lookup_cache = None
                return pd.Series(np.nan, index=aligned.index)
            trades = trades[keep].dropna()
            # merge_asof requires sorted by the `on` column (timestamp) globally
            # — pandas handles the `by` groups internally.
            trades["timestamp"] = pd.to_datetime(trades["timestamp"], utc=True)
            trades["expiry_dt"] = pd.to_datetime(trades["expiry_dt"], utc=True)
            trades = trades.sort_values("timestamp").reset_index(drop=True)
            self._trades_lookup_cache = trades
            logger.info("Cached %d option trades for exit-IV lookup.", len(trades))
        trades = self._trades_lookup_cache
        if trades is None:
            return pd.Series(np.nan, index=aligned.index)

        # Backtester's straddle is ATM call+put at same (K, D). Call and put
        # ATM IVs are equal by put-call parity; use calls for the lookup
        # (arbitrary — could average both if desired).
        # Cast strike to float64 to match the trade-side dtype produced by
        # parse_instrument_names (int strikes break merge_asof's strict
        # dtype requirement on the by= keys).
        left = pd.DataFrame({
            "timestamp": pd.to_datetime(aligned["exit_target"].values, utc=True),
            "strike": np.asarray(aligned["strike"].values, dtype=np.float64),
            "expiry_dt": pd.to_datetime(aligned["expiry_dt"].values, utc=True),
            "option_type": "C",  # query calls; put IV equals call IV at ATM
            "_row_idx": np.arange(len(aligned)),
        }).sort_values("timestamp").reset_index(drop=True)

        calls = trades[trades["option_type"] == "C"].copy()
        calls["strike"] = calls["strike"].astype(np.float64)
        tol = pd.Timedelta(minutes=int(self.cfg.exit_iv_trade_tolerance_minutes))
        merged = pd.merge_asof(
            left,
            calls[["timestamp", "strike", "expiry_dt", "iv"]],
            on="timestamp",
            by=["strike", "expiry_dt"],
            direction="backward",
            tolerance=tol,
            suffixes=("", "_tr"),
        )
        # merged["iv"] is NaN where no same-contract trade within tolerance
        out = pd.Series(np.nan, index=aligned.index, dtype=float)
        # Map merged's iv back to aligned's original index via _row_idx
        row_idx = merged["_row_idx"].values
        iv_vals = merged["iv"].values
        idx_array = aligned.index.to_numpy()
        out.loc[idx_array[row_idx]] = iv_vals
        return out

    def _exit_iv_from_surface(self, aligned: pd.DataFrame) -> pd.Series:
        """Tier 2: SVI-surface-reconstructed exit IV for the ENTRY contract.

        Evaluates the SVI surface at ``exit_target`` for the entry's exact
        (strike, expiry_dt). NaN where the surface slice is missing.
        """
        from vol_arb.svi import evaluate_surface_batch

        svi_path = Path(self.cfg.drl_svi_params_path)
        if not svi_path.exists():
            return pd.Series(np.nan, index=aligned.index)

        svi_params = pd.read_parquet(svi_path)
        exit_ts = pd.to_datetime(aligned["exit_target"].values, utc=True)
        expiry_ts = pd.to_datetime(aligned["expiry_dt"].values, utc=True)
        query = pd.DataFrame({
            "timestamp": exit_ts,
            "strike": aligned["strike"].values,
            "expiry_dt": expiry_ts,
            "index_price": aligned["price_exit"].values,
        })
        iv_surf = evaluate_surface_batch(svi_params, query)
        return pd.Series(iv_surf.values, index=aligned.index)

    # ── Step 5+6: P&L with discrete delta hedging ───────────────

    def compute_pnl(self, df: pd.DataFrame) -> pd.DataFrame:
        """P&L with path-dependent discrete delta hedging.

        Replaces the continuous gamma formula with a Monte Carlo simulation
        of the intra-window price path. Delta hedges fire when unhedged
        exposure exceeds delta_hedge_band.

        P&L = Vega + Theta + Discrete_Hedge_PnL - Option_Slippage - Hedge_Costs
        """
        cfg = self.cfg
        dt_days = cfg.holding_minutes / 1440.0

        S = df["index_price"].values
        signal = df["signal"].values

        # ── Dynamic instrument routing ───────────────────────────
        # Gating variable depends on routing_mode:
        #   "jump_skew": (down_jump_fraction - up_jump_fraction), threshold in [0, 1]
        #   "svi_skew":  svi_25d_skew_1w, threshold in vol points
        # Positive skew (either kind) → market sees more downside → use bear strangle
        # Negative skew → more upside → use bull strangle
        if cfg.routing_mode == "svi_skew":
            skew_available = (
                cfg.use_instrument_routing
                and "svi_25d_skew_1w" in df.columns
                and "vega_bear" in df.columns
            )
        else:
            skew_available = (
                cfg.use_instrument_routing
                and "down_jump_fraction" in df.columns
                and "vega_bear" in df.columns
            )

        if skew_available:
            if cfg.routing_mode == "svi_skew":
                skew_vals = df["svi_25d_skew_1w"].values
                skew_threshold = cfg.svi_skew_threshold
                # NaN-safe: treat missing SVI skew rows as "no routing" (ATM)
                skew_vals = np.where(np.isfinite(skew_vals), skew_vals, 0.0)
            else:
                skew_vals = (df["down_jump_fraction"].values
                             - df["up_jump_fraction"].values)
                skew_threshold = cfg.jump_skew_threshold
            use_bear = skew_vals > skew_threshold
            use_bull = skew_vals < -skew_threshold

            def _route(atm, bear, bull):
                bear_safe = np.where(np.isfinite(bear), bear, atm)
                bull_safe = np.where(np.isfinite(bull), bull, atm)
                return np.select([use_bear, use_bull],
                                 [bear_safe, bull_safe], default=atm)

            vega_s = _route(df["vega"].values,
                            df["vega_bear"].values, df["vega_bull"].values)
            theta_s = _route(df["theta"].values,
                             df["theta_bear"].values, df["theta_bull"].values)
            delta_s = _route(df["delta"].values,
                             df["delta_bear"].values, df["delta_bull"].values)
            # Gamma is needed for the strangle-correct delta hedge path
            # (linear constant-gamma delta update in simulate_discrete_hedging).
            # Falls back to NaN → ATM gamma if bear/bull gamma isn't present.
            if "gamma_bear" in df.columns and "gamma_bull" in df.columns:
                gamma_s = _route(df["gamma"].values,
                                 df["gamma_bear"].values,
                                 df["gamma_bull"].values)
            else:
                gamma_s = df["gamma"].values.copy()
            K = _route(df["strike"].values,
                       df["strike_bear"].values, df["strike_bull"].values)
            iv_entry = _route(df["iv"].values,
                              df["iv_bear"].values, df["iv_bull"].values)
            # Fall back to iv_used (DVOL fallback) where routed IV is NaN
            iv_entry = np.where(np.isfinite(iv_entry), iv_entry,
                                df["iv_used"].values)

            # Track instrument type: 1=bear, -1=bull, 0=ATM
            instrument_type = np.select(
                [use_bear & np.isfinite(df["vega_bear"].values),
                 use_bull & np.isfinite(df["vega_bull"].values)],
                [np.int8(1), np.int8(-1)],
                default=np.int8(0),
            )
            n_bear = (instrument_type == 1).sum()
            n_bull = (instrument_type == -1).sum()
            n_atm = (instrument_type == 0).sum()
            logger.info("Instrument routing: %d ATM, %d bear, %d bull",
                        n_atm, n_bear, n_bull)
        else:
            # No routing — use ATM columns directly
            vega_s = df["vega"].values
            theta_s = df["theta"].values
            delta_s = df["delta"].values
            gamma_s = df["gamma"].values
            K = df["strike"].values
            iv_entry = df["iv_used"].values
            instrument_type = np.zeros(len(df), dtype=np.int8)

        # ── Classifier-head gate (W4.1) ─────────────────────────
        # Apply the optional directional classifier filter AFTER routing
        # and BEFORE any sizing/pricing so muted bars propagate as
        # ``signal == 0`` with no further downstream changes.
        if cfg.classifier_path:
            if "clf_prob" not in df.columns:
                raise RuntimeError(
                    "classifier_path set but clf_prob column missing — "
                    "did load_data run?",
                )
            thr = cfg.classifier_threshold  # pydantic-validated in (0.5, 1.0)
            clf_prob = df["clf_prob"].to_numpy()
            prob_valid = np.isfinite(clf_prob)
            # Bars without classifier coverage (NaN prob) are treated per
            # the config flag: pass-through lets them trade as usual, mute
            # kills them (used for A/B validation where we want to compare
            # only the classifier-covered window).
            nan_pass = bool(cfg.classifier_nan_is_pass)
            nan_agree = (~prob_valid) if nan_pass else np.zeros_like(
                prob_valid
            )
            long_agree = (signal == 1) & (
                (prob_valid & (clf_prob >= thr)) | nan_agree
            )
            short_agree = (signal == -1) & (
                (prob_valid & (clf_prob <= 1.0 - thr)) | nan_agree
            )
            agree = long_agree | short_agree
            active = signal != 0
            muted = active & ~agree
            n_muted = int(muted.sum())
            n_active = int(active.sum())
            n_nan_active = int((active & ~prob_valid).sum())
            if n_active:
                logger.info(
                    "Classifier gate (thr=%.2f, nan_pass=%s): muted %d / %d "
                    "active bars (%.1f%%); %d active NaN bars",
                    thr, nan_pass, n_muted, n_active,
                    100.0 * n_muted / n_active, n_nan_active,
                )
            if n_muted:
                signal = np.where(muted, np.int8(0), signal).astype(np.int8)
                # Keep df["signal"] in sync so the downstream trade-log
                # filter in run_backtest (`result["signal"] != 0`) sees
                # the gated state.
                df["signal"] = signal

        T_years = df["T_years"].values
        sigma_dec = iv_entry / 100.0  # IV in decimal for BS delta

        # Maker adverse-selection penalty is the MAKER's cost — a fill that
        # earns spread but is adversely selected by informed counterparties.
        # Taker fills pay half-spread via the slippage model (bar_taker_slip
        # × size_mult, added to straddle_slip in the per-bar loop) and should
        # NOT also be charged this penalty. The up-front vega_pnl_btc is
        # computed WITHOUT the penalty; it is subtracted inside the PnL loop
        # ONLY on bars where `maker_filled_arr[i] == True` (SOR confirmed a
        # maker fill). Fix 2026-04-18 — previously the penalty was applied
        # unconditionally to every trade, double-counting spread cost on
        # taker fills (100% of current trades) by 0.5 * Σ|vega|*N ≈ $1.48M NC.

        # Realized vol (from TSRV, decimal) — drives the simulated price path.
        # Use lagged vol_ann: vol_ann[T] is the realized vol OF bar T (open-stamped),
        # so at decision time T only vol_ann[T-1] is known.  shift(1) avoids lookahead.
        vol_ann = df["vol_ann"].shift(1).bfill().values

        # ── Execution delay: spot drifts between signal and order ──
        # At bar close we observe S; the order hits the exchange ~delay seconds
        # later.  Model the micro price move as a symmetric random walk step
        # scaled by realized vol.  Affects hedge entry price and fee $ amounts.
        if cfg.execution_delay_s > 0:
            dt_delay = cfg.execution_delay_s / (365.25 * 86400)
            delay_sigma = vol_ann * np.sqrt(dt_delay)  # per-bar vol over delay
            rng_delay = np.random.RandomState(cfg.hedge_sim_seed + 99)
            S_exec = S * np.exp(rng_delay.normal(0, delay_sigma))
            logger.info(
                "Execution delay %.1fs: median spot drift $%.1f (%.1f bps)",
                cfg.execution_delay_s,
                float(np.median(np.abs(S_exec - S))),
                float(np.median(np.abs(S_exec - S) / S * 1e4)),
            )
        else:
            S_exec = S

        # Exit values.  iv_exit comes from align_exits() which looks up the
        # ATM iv_used at T+holding_minutes — it's an ATM-strike IV.
        iv_exit = df["iv_exit"].values

        # ── Per-1-BTC vega PnL — skew-consistent formulation ──
        # For ATM straddles: iv_entry and iv_exit are both ATM, so
        #     dIV = iv_exit - effective_entry_iv
        # is the realized ATM IV move, and vega * dIV is correct.
        #
        # For routed strangles (bear/bull): iv_entry is the VEGA-WEIGHTED
        # strangle IV (on average iv_atm + skew), while iv_exit is still
        # the ATM IV at exit time.  Taking their direct difference introduces
        # a systematic bias of ~skew_entry per trade — a surface-mismatch bug:
        # the entry IV and exit IV live on different points of the vol surface.
        # A correct simulation would re-query the SVI surface at the strangle's
        # (strike, T_exit) for the exit IV, but that infrastructure doesn't
        # exist yet.  As a second-order approximation (consistent with the
        # existing "skew change in 15m is second-order" comment that already
        # justifies using ATM iv_exit for strangles), we assume skew is stable
        # within the 15-minute holding window:
        #     iv_strangle_exit ≈ iv_atm_exit + (iv_strangle_entry - iv_atm_entry)
        # which gives
        #     dIV_strangle = iv_strangle_exit - iv_strangle_entry
        #                  = iv_atm_exit - iv_atm_entry
        # — i.e., for strangles, use the ATM-to-ATM realized IV move, still
        # scaled by the strangle's actual vega.
        atm_iv_entry = df["iv_used"].values
        atm_iv_exit = iv_exit  # iv_exit is already ATM (from align_exits on iv_used)
        # dIV: strangles use ATM-to-ATM move; ATM straddles use the original
        # formula. Both use UNPENALIZED entry IVs — the maker adverse-selection
        # cost is handled per-bar inside the loop (see docstring above).
        is_routed = instrument_type != 0
        dIV_atm_consistent = atm_iv_exit - atm_iv_entry
        dIV_original = iv_exit - iv_entry
        dIV = np.where(is_routed, dIV_atm_consistent, dIV_original)

        vega_pnl_btc = signal * vega_s * dIV
        theta_pnl_btc = signal * theta_s * dt_days

        # ── Delta hedging: DRL policy or rule-based fallback ──
        delta_pnl_btc = None
        hedge_costs_btc = None
        if cfg.use_drl_hedging:
            try:
                from vol_arb.drl_hedging.env import DRLHedger  # lazy
                # Horizon-cadence audit (Phase 2h5.7.B). The shipped SAC
                # policy (production_sac_v4_final) was trained at the
                # default DRLHedgingConfig.bar_minutes=15. When the
                # backtester runs at a different holding cadence the
                # policy still emits actions but its observation regime
                # is off-distribution. Log a single warning so the
                # mismatch is loud rather than silent.
                if cfg.holding_minutes != 15:
                    logger.warning(
                        "DRL hedger horizon mismatch: policy trained at "
                        "bar_minutes=15 but backtester holding_minutes=%d. "
                        "Hedge actions are off-distribution; consider "
                        "use_drl_hedging=False for this horizon. See plan "
                        "starry-rolling-rabin.md Phase 2h5.7.B.",
                        cfg.holding_minutes,
                    )
                if not hasattr(self, "_drl_hedger") or self._drl_hedger is None:
                    logger.info("Loading DRL hedging policy from %s", cfg.drl_policy_path)
                    self._drl_hedger = DRLHedger(
                        policy_path=cfg.drl_policy_path,
                        svi_params_path=cfg.drl_svi_params_path,
                    )
                logger.info("Running DRL delta hedging (%d steps)...", cfg.hedge_sim_steps)
                # Extra context for the DRL agent's observation vector.
                # Missing columns → arrays of NaN → DRLHedger substitutes safe defaults.
                jump_arr = df["jump_fraction"].values if "jump_fraction" in df.columns else None
                trail_arr = (
                    df["trailing_1h_volume"].values
                    if "trailing_1h_volume" in df.columns else None
                )
                # VPIN: compute on-the-fly from vol_ann via the returns-proxy.
                # Phase 2 upgrade: read a precomputed vpin column from df.
                from vol_arb.drl_hedging.vpin import compute_vpin_proxy_from_returns
                vpin_arr = compute_vpin_proxy_from_returns(df).values
                delta_pnl_btc, hedge_costs_btc = self._drl_hedger.simulate(
                    S_entry=S_exec,
                    K=K,
                    sigma=sigma_dec,
                    T_years=T_years,
                    vol_ann=vol_ann,
                    signal=signal,
                    n_steps=cfg.hedge_sim_steps,
                    perp_fee=cfg.perp_taker_fee,
                    perp_slippage_bps=cfg.perp_slippage_bps,
                    seed=cfg.hedge_sim_seed,
                    vpin=vpin_arr,
                    jump_fraction=jump_arr,
                    trailing_volume=trail_arr,
                )
            except Exception as exc:
                if not cfg.drl_fallback_on_error:
                    raise
                logger.warning("DRL hedger failed (%s); falling back to rule-based", exc)
                delta_pnl_btc = None
                hedge_costs_btc = None

        if delta_pnl_btc is None:
            logger.info("Simulating discrete delta hedging (%d steps, band=%.3f)...",
                        cfg.hedge_sim_steps, cfg.delta_hedge_band)
            # Delta-hedge simulation mode selection:
            #
            # - When routing is OFF (the production default), every position
            #   is an ATM straddle and the legacy BS-recomputation path
            #   (``delta_entry=None``) is mathematically exact: it recomputes
            #   ``2*N(d1) - 1`` at each tick using the current S, K_atm, and
            #   sigma_atm, which is the true BS straddle delta.  We use this
            #   path to preserve bit-identical behavior for all historical
            #   production runs.
            #
            # - When routing is ON, routed positions (bear/bull strangles)
            #   are NOT straddles at strike K — they're composites where
            #   one leg is ATM and the other is 25-delta.  The legacy BS
            #   formula is wrong for those.  We switch to the greeks-explicit
            #   path which uses the precomputed delta/gamma from the trade
            #   log and updates delta via a linear constant-gamma
            #   approximation.  This introduces a small (~1%) Taylor error
            #   for the ATM bars in a routing run, but it's the only way to
            #   get physically correct hedging for the strangle bars, and
            #   routing runs are experimental anyway.
            if cfg.use_instrument_routing and gamma_s is not None:
                de = delta_s.copy() if delta_s is not None else None
                ge = gamma_s.copy() if gamma_s is not None else None
                if (de is not None and ge is not None
                        and np.isfinite(de).all() and np.isfinite(ge).all()):
                    delta_kw = {"delta_entry": de, "gamma_entry": ge}
                else:
                    # NaN safety: if any routed Greek is missing, fall back
                    # to legacy (which uses the ATM BS formula — correct for
                    # ATM bars and the same wrong behavior we had before for
                    # strangle bars with missing data).
                    delta_kw = {}
            else:
                delta_kw = {}  # legacy BS path — exact for ATM straddles
            delta_pnl_btc, hedge_costs_btc = simulate_discrete_hedging(
                S_entry=S_exec,
                K=K,
                sigma=sigma_dec,
                T_years=T_years,
                vol_ann=vol_ann,
                signal=signal,
                n_steps=cfg.hedge_sim_steps,
                delta_hedge_band=cfg.delta_hedge_band,
                perp_fee=cfg.perp_taker_fee,
                perp_slippage_bps=cfg.perp_slippage_bps,
                seed=cfg.hedge_sim_seed,
                **delta_kw,
            )

        # Option maker fees (entry + exit, 4 legs) with Deribit 12.5% cap
        # Fee per leg = MIN(0.0003 BTC, 0.125 * option_price_btc)
        # ATM call price in BTC ≈ 0.4 * sigma * sqrt(T) (Brenner-Subrahmanyam)
        _opt_price_btc = 0.4 * sigma_dec * np.sqrt(np.maximum(T_years, 1e-6))
        _fee_flat = cfg.option_maker_fee  # 0.0003 BTC per contract
        _fee_capped = np.minimum(_fee_flat, 0.125 * _opt_price_btc)
        option_fee_btc = 4 * _fee_capped * S_exec  # USD per BTC (4 legs)

        # ── Vega stop: minute-level (honest) or terminal (lookahead) ──
        n_bars = len(S)

        # Direction-dependent IM: long straddle IM = premium, short = Deribit formula
        if cfg.use_directional_im:
            # Long straddle IM = straddle premium / S + hedge buffer
            # Straddle premium ≈ 2 * call = 2 * 0.4 * S * sigma * sqrt(T)
            # As fraction of notional: 0.8 * sigma * sqrt(T)
            _long_im = np.clip(
                0.8 * sigma_dec * np.sqrt(np.maximum(T_years, 1e-6)) + cfg.long_im_buffer,
                cfg.long_im_floor, cfg.im_fraction,
            )
            im_per_bar = np.where(signal == 1, _long_im, cfg.im_fraction)
        else:
            im_per_bar = np.full(n_bars, cfg.im_fraction)
        stop_triggered = np.zeros(n_bars, dtype=bool)
        exit_frac = np.ones(n_bars)
        _exit_spread_minute = np.zeros(n_bars)

        if cfg.vega_stop_minute_usd > 0:
            # Honest minute-level vega stop from actual trade-tape IV
            min_iv_path = Path(cfg.minute_iv_path)
            if not min_iv_path.exists():
                raise FileNotFoundError(
                    f"{min_iv_path} not found — run: "
                    "PYTHONPATH=. python scripts/build_minute_iv.py"
                )
            min_iv_df = pd.read_parquet(min_iv_path)
            min_iv_series = min_iv_df["median_iv"]
            min_spread_series = min_iv_df["exit_spread_usd"]

            bar_ts = df.index
            iv_entry_arr = df["iv_used"].values

            # Build (n_bars, 14) matrix: IV at T+1m, T+2m, ..., T+14m
            iv_matrix = np.empty((n_bars, 14))
            spread_matrix = np.empty((n_bars, 14))
            for j in range(14):
                target_ts = bar_ts + pd.Timedelta(minutes=j + 1)
                iv_matrix[:, j] = (
                    min_iv_series
                    .reindex(target_ts, method="ffill")
                    .values
                )
                spread_matrix[:, j] = (
                    min_spread_series
                    .reindex(target_ts, method="ffill")
                    .fillna(12.0)
                    .values
                )

            # dIV from entry at each minute checkpoint
            dIV_matrix = iv_matrix - iv_entry_arr[:, np.newaxis]

            # Per-BTC vega loss at each checkpoint (positive = losing)
            loss_matrix = (
                -signal[:, np.newaxis]
                * vega_s[:, np.newaxis]
                * dIV_matrix
            )

            # First minute where loss exceeds threshold
            exceeds = loss_matrix > cfg.vega_stop_minute_usd
            has_any = exceeds.any(axis=1) & (signal != 0)
            first_k = np.argmax(exceeds, axis=1)  # 0-based index

            stop_triggered = has_any
            exit_minute = first_k + 1  # 1-indexed: minute 1-14
            exit_frac = np.where(stop_triggered, exit_minute / 15.0, 1.0)

            # Exit spread at the stop minute
            _exit_spread_minute = np.where(
                stop_triggered,
                spread_matrix[np.arange(n_bars), first_k],
                0.0,
            )

            n_stopped = int(stop_triggered.sum())
            if n_stopped > 0:
                # Use actual dIV at stop minute (not terminal dIV)
                stopped_dIV = dIV_matrix[np.arange(n_bars), first_k]
                vega_pnl_btc = np.where(
                    stop_triggered,
                    signal * vega_s * stopped_dIV,
                    vega_pnl_btc,
                )
                # Safety cap (minute IV is median, not exact fill)
                vega_pnl_btc = np.where(
                    stop_triggered,
                    np.maximum(vega_pnl_btc, -cfg.vega_stop_minute_usd),
                    vega_pnl_btc,
                )
                theta_pnl_btc = theta_pnl_btc * exit_frac
                delta_pnl_btc = delta_pnl_btc * exit_frac
                hedge_costs_btc = hedge_costs_btc * exit_frac
                logger.info(
                    "Minute vega stop ($%.0f/BTC): %d stopped (%.1f%%), "
                    "avg exit at minute %.1f",
                    cfg.vega_stop_minute_usd, n_stopped,
                    100 * n_stopped / max(int((signal != 0).sum()), 1),
                    exit_minute[stop_triggered].mean(),
                )

        elif cfg.vega_stop_usd > 0:
            adverse = -vega_pnl_btc  # positive when losing on vega
            stop_triggered = adverse > cfg.vega_stop_usd
            n_stopped = int(stop_triggered.sum())
            if n_stopped > 0:
                exit_frac = np.where(
                    stop_triggered,
                    np.clip(
                        cfg.vega_stop_usd / np.maximum(adverse, 1e-6),
                        0.1, 1.0,
                    ),
                    1.0,
                )
                vega_pnl_btc = np.maximum(vega_pnl_btc, -cfg.vega_stop_usd)
                theta_pnl_btc = theta_pnl_btc * exit_frac
                delta_pnl_btc = delta_pnl_btc * exit_frac
                hedge_costs_btc = hedge_costs_btc * exit_frac
                logger.info(
                    "Vega stop ($%.0f/BTC): %d stopped (%.1f%%)",
                    cfg.vega_stop_usd, n_stopped,
                    100 * n_stopped / max(int((signal != 0).sum()), 1),
                )

        # ── 2D slippage surface lookup ──
        slip_grid = np.zeros((4, 3, n_bars))
        for s in range(4):
            for v in range(3):
                col = f"slip_s{s}_v{v}"
                if col in df.columns:
                    slip_grid[s, v] = df[col].values
                else:
                    slip_grid[s, v] = 10.0

        dvol_raw = df["dvol_close"].values
        vol_bucket = np.digitize(dvol_raw, [45, 65])
        vol_bucket = np.clip(vol_bucket, 0, 2)
        jump_frac = df["jump_fraction"].values

        # ── Vol-targeted dynamic margin ──
        dvol = df["dvol_close"].values.copy()
        dvol = np.where(np.isfinite(dvol) & (dvol > 0), dvol, cfg.baseline_target_dvol)
        dynamic_margin = cfg.target_margin_utilization * (cfg.baseline_target_dvol / dvol)
        dynamic_margin = np.clip(dynamic_margin, 0.05, cfg.target_margin_utilization)

        # ── Dynamic capacity cap from trailing ATM volume ──
        trailing_vol = df["trailing_1h_volume"].values
        dynamic_btc_cap = np.clip(
            trailing_vol * cfg.max_participation_rate, 1.0, 50.0,
        )

        # ── Deterministic SOR arrays ──
        if cfg.use_sor and "next_sell_price_5s" in df.columns:
            next_sell_price = df["next_sell_price_5s"].values
            next_sell_mark = df["next_sell_mark_5s"].values
            next_buy_price = df["next_buy_price_5s"].values
            next_buy_mark = df["next_buy_mark_5s"].values
            has_sor = True
        else:
            has_sor = False

        # ── Per-bar taker slippage from trade tape (realistic_taker) ──
        _bar_taker_slip = None
        if cfg.slippage_model == "realistic_taker":
            eq_path = Path(cfg.bar_exec_quality_path)
            if eq_path.exists():
                _eq_df = pd.read_parquet(eq_path)
                _eq_df.index = pd.to_datetime(_eq_df.index, utc=True)
                # Align to vol_arb bars via forward-fill (trailing, no lookahead)
                _bar_taker_slip = (
                    _eq_df["bar_taker_slip_usd"]
                    .reindex(df.index, method="ffill")
                    .fillna(12.0)  # fallback to empirical median
                    .values
                )
                logger.info(
                    "Loaded realistic_taker slippage: %d bars, median $%.1f/BTC",
                    len(_bar_taker_slip), float(np.nanmedian(_bar_taker_slip)),
                )
            else:
                logger.warning(
                    "bar_exec_quality_path %s not found, falling back to grid",
                    eq_path,
                )

        # ── Continuous impact model (optional) ──
        _impact_model = None
        _impact_date_idx = None
        if cfg.use_continuous_impact:
            from vol_arb.impact import ImpactModel
            _impact_model = ImpactModel(cfg.impact_params_path)
            _impact_date_idx = _impact_model.align_to_bars(df.index)

        # ── Pre-compute edge array for sizing strategies ──
        edge_arr = np.abs(df["rv_pred_pct"].values - df["iv_used"].values)

        # ── Sequential equity scan with capacity cap + slippage ──
        from vol_arb.sizing import compute_position_size

        equity_arr = np.empty(n_bars)
        N = np.empty(n_bars)
        trade_pnl = np.empty(n_bars)
        slippage_arr = np.empty(n_bars)
        maker_filled_arr = np.zeros(n_bars, dtype=bool)

        eq = cfg.initial_equity
        peak_eq = cfg.initial_equity
        size_edges = [1.0, 3.0, 10.0]

        # Rolling statistics for sizing (no lookahead)
        _edge_ema = float(np.nanmedian(edge_arr[:96])) if n_bars > 96 else 35.0
        _edge_alpha = 2.0 / (500 + 1)  # 500-bar EMA for median edge
        _win_count = 0
        _trade_count = 0
        _win_sum = 0.0
        _loss_sum = 0.0
        _iv_changes = []
        # Rolling taker slippage EMA for liquidity-conditioned sizing
        _slip_ema = (
            float(np.nanmedian(_bar_taker_slip[:96]))
            if (_bar_taker_slip is not None and n_bars > 96)
            else 12.0
        )
        _slip_alpha = 2.0 / (200 + 1)  # 200-bar EMA (~50 hours at 15m)

        # ── Slippage filter state (A1-A4) ──
        _hedge_cost_ema = 0.0
        _hedge_cost_alpha = 2.0 / (200 + 1)
        _gross_pnl_per_btc_ema = 0.0  # rolling EMA of actual per-BTC gross PnL
        _gross_pnl_alpha = 2.0 / (500 + 1)
        _gross_pnl_count = 0  # trades seen (for warmup)
        _slip_buffer: list[float] = []  # rolling window for A4 regime filter
        profit_filtered_arr = np.zeros(n_bars, dtype=bool)
        slip_thresh_filtered_arr = np.zeros(n_bars, dtype=bool)
        regime_filtered_arr = np.zeros(n_bars, dtype=bool)
        profit_scale_arr = np.ones(n_bars)
        _any_filter_active = (
            cfg.use_profit_filter or cfg.use_slip_adjusted_threshold
            or cfg.use_profit_scaling or cfg.use_liquidity_regime_filter
        )

        # ── Kelly-shrinkage multiplier sidecar (Phase 2 2026-04-21) ──
        # Loaded once, aligned to df.index. Defaults None -> no-op.
        _kelly_mult_arr: np.ndarray | None = None
        _kelly_applied_arr = np.ones(n_bars)
        if cfg.kelly_mult_path is not None:
            kpath = Path(cfg.kelly_mult_path)
            if not kpath.exists():
                logger.warning(
                    "kelly_mult_path=%s does not exist -- skipping Kelly boost",
                    kpath,
                )
            else:
                km_df = pd.read_parquet(kpath)
                if "size_mult" in km_df.columns:
                    km_ser = km_df["size_mult"]
                else:
                    km_ser = km_df.iloc[:, 0]
                km_aligned = km_ser.reindex(df.index)
                arr = km_aligned.to_numpy(dtype=float)
                # Safety: floor at 1.0 (upside-only), cap at 2.0 (hard
                # risk bound), and pass-through NaN -> 1.0.
                arr = np.where(np.isnan(arr), 1.0, arr)
                arr = np.clip(arr, 1.0, 2.0)
                _kelly_mult_arr = arr
                n_boost = int((arr > 1.0 + 1e-9).sum())
                logger.info(
                    "Kelly sidecar loaded (%s): %d bars with mult>1.0, "
                    "mean mult=%.4f, max mult=%.4f",
                    kpath.name, n_boost,
                    float(arr.mean()), float(arr.max()),
                )

        for i in range(n_bars):
            if signal[i] == 0 or eq <= 0.0:
                N[i] = 0.0
                trade_pnl[i] = 0.0
                slippage_arr[i] = 0.0
            else:
                # Rolling IV std (last 200 bars of IV changes)
                rolling_iv_std = max(np.std(_iv_changes[-200:]), 0.5) if len(_iv_changes) > 20 else 2.0
                rolling_wr = _win_count / max(_trade_count, 1)
                rolling_payoff = (_win_sum / max(_win_count, 1)) / max(_loss_sum / max(_trade_count - _win_count, 1), 0.01) if _trade_count > 10 else 1.0

                n_i = compute_position_size(
                    strategy=cfg.sizing_model,
                    equity=eq, spot=S[i], signal=signal[i],
                    dynamic_margin=dynamic_margin[i],
                    dynamic_btc_cap=dynamic_btc_cap[i],
                    cfg=cfg,
                    edge_vp=edge_arr[i],
                    rolling_median_edge=_edge_ema,
                    vega=vega_s[i],
                    rolling_iv_std=rolling_iv_std,
                    peak_equity=peak_eq,
                    rolling_win_rate=rolling_wr,
                    rolling_payoff_ratio=rolling_payoff,
                    bar_taker_slip=_bar_taker_slip[i] if _bar_taker_slip is not None else 12.0,
                    rolling_slip_ema=_slip_ema,
                    vega_per_pt=abs(vega_s[i]) if np.isfinite(vega_s[i]) else 0.0,
                    im_override=im_per_bar[i],
                )
                N[i] = n_i

                # Hard BTC cap (if configured)
                if cfg.max_position_btc > 0:
                    N[i] = min(N[i], cfg.max_position_btc)

                # ── Slippage-aware trade filters (A1-A4) ──
                skip_trade = False

                if _any_filter_active and N[i] > 0:
                    # Pre-compute slippage estimate (shared by A1/A2/A3)
                    _est_bar_slip = (
                        _bar_taker_slip[i] if _bar_taker_slip is not None else 12.0
                    )
                    _est_size_mult = np.sqrt(max(N[i], 0.1))
                    _est_full_slip = min(
                        2.0 * _est_bar_slip * _est_size_mult * np.exp(
                            cfg.spread_jump_k * jump_frac[i]
                        ),
                        cfg.ioc_cap_multiplier * _est_bar_slip,
                    )
                    _vega_abs = (
                        abs(vega_s[i]) if np.isfinite(vega_s[i]) else 1e-6
                    )

                    # A4: Liquidity-regime filter (cheapest — no PnL estimation)
                    if (cfg.use_liquidity_regime_filter
                            and not skip_trade
                            and _bar_taker_slip is not None
                            and len(_slip_buffer) >= 96):
                        pctile = np.percentile(
                            _slip_buffer[-cfg.liquidity_regime_window:],
                            cfg.liquidity_regime_percentile,
                        )
                        if _est_bar_slip > pctile:
                            skip_trade = True
                            regime_filtered_arr[i] = True

                    # A2: Slippage-adjusted dynamic threshold
                    if cfg.use_slip_adjusted_threshold and not skip_trade:
                        slip_vol_pts = (
                            (2.0 * _est_bar_slip) / max(_vega_abs * S[i], 1e-6)
                        )
                        eff_thresh = (
                            cfg.slip_threshold_base_long + slip_vol_pts
                            if signal[i] == 1
                            else cfg.slip_threshold_base_short + slip_vol_pts
                        )
                        if edge_arr[i] < eff_thresh:
                            skip_trade = True
                            slip_thresh_filtered_arr[i] = True

                    # A1: Pre-trade expected profit filter.
                    # Uses an empirical-Bayes shrinkage estimator for expected
                    # per-BTC gross PnL: blend the rolling EMA of realized gross
                    # (observed only on executed trades) with a per-signal
                    # theoretical prior (edge*vega*discount, always fresh from
                    # the current bar). Shrinkage weight λ = n/(n+τ) where n is
                    # the count of executed trades and τ=profit_filter_shrinkage_tau.
                    # Because the theoretical prior never goes stale, the estimator
                    # cannot self-reinforce a lockout even if the realized EMA
                    # drops below the gate.
                    if cfg.use_profit_filter and not skip_trade:
                        est_hedge = (
                            _hedge_cost_ema
                            if _hedge_cost_ema > 0
                            else 2.0 * cfg.perp_taker_fee * S[i]
                        )
                        est_costs = _est_full_slip + est_hedge + option_fee_btc[i]
                        est_theo = (
                            edge_arr[i] * _vega_abs
                            * cfg.profit_filter_edge_discount
                        )
                        _tau = max(cfg.profit_filter_shrinkage_tau, 1e-6)
                        _lam = _gross_pnl_count / (_gross_pnl_count + _tau)
                        est_gross = (
                            _lam * _gross_pnl_per_btc_ema
                            + (1.0 - _lam) * est_theo
                        )
                        est_net = est_gross - est_costs
                        est_ratio = est_net / max(abs(est_gross), 1e-6)

                        if est_ratio < cfg.profit_filter_min_ratio:
                            skip_trade = True
                            profit_filtered_arr[i] = True
                        elif (cfg.profit_filter_min_abs > 0
                              and est_net < cfg.profit_filter_min_abs):
                            skip_trade = True
                            profit_filtered_arr[i] = True

                    # A3: Profit-ratio position scaling (soft — doesn't skip).
                    # Same empirical-Bayes shrinkage estimator as A1 so the two
                    # stay coherent — otherwise A3 could scale down to the floor
                    # on a frozen EMA while A1 (if disabled) passes the trade.
                    if cfg.use_profit_scaling and not skip_trade and N[i] > 0:
                        est_hedge = (
                            _hedge_cost_ema
                            if _hedge_cost_ema > 0
                            else 2.0 * cfg.perp_taker_fee * S[i]
                        )
                        est_costs = _est_full_slip + est_hedge + option_fee_btc[i]
                        est_theo = (
                            edge_arr[i] * _vega_abs
                            * cfg.profit_filter_edge_discount
                        )
                        _tau = max(cfg.profit_filter_shrinkage_tau, 1e-6)
                        _lam = _gross_pnl_count / (_gross_pnl_count + _tau)
                        est_gross = (
                            _lam * _gross_pnl_per_btc_ema
                            + (1.0 - _lam) * est_theo
                        )
                        est_net = est_gross - est_costs
                        est_ratio = est_net / max(abs(est_gross), 1e-6)

                        ps = np.clip(
                            est_ratio, cfg.profit_scaling_floor,
                            cfg.profit_scaling_cap,
                        )
                        if est_ratio <= 0:
                            ps = cfg.profit_scaling_floor
                        profit_scale_arr[i] = ps
                        N[i] *= ps
                        N[i] = min(N[i], dynamic_btc_cap[i])

                # ── Kelly-shrinkage boost (Phase 2, 2026-04-21) ──
                # Apply precomputed per-bar multiplier AFTER profit_scaling
                # AND AFTER the dynamic_btc_cap clip above, then re-clip so
                # Kelly can never breach the margin envelope. Floor 1.0 is
                # already enforced at load time (upside-only asymmetric).
                if _kelly_mult_arr is not None and not skip_trade and N[i] > 0:
                    km = _kelly_mult_arr[i]
                    if km > 1.0 + 1e-12:
                        N[i] *= km
                        N[i] = min(N[i], dynamic_btc_cap[i])
                        _kelly_applied_arr[i] = km

                if skip_trade:
                    N[i] = 0.0
                    trade_pnl[i] = 0.0
                    slippage_arr[i] = 0.0
                else:
                    # Option slippage (taker cost)
                    if _bar_taker_slip is not None:
                        # Realistic per-bar taker cost from trade tape
                        bar_slip = _bar_taker_slip[i]
                        base_slip = bar_slip
                        # Size impact: sqrt scaling (floor at 0.1 BTC, not 1.0)
                        size_mult = np.sqrt(max(N[i], 0.1))
                        full_straddle_slip = 2.0 * bar_slip * size_mult * np.exp(
                            cfg.spread_jump_k * jump_frac[i]
                        )
                        # IOC cap: configurable multiplier prevents catastrophic fills
                        full_straddle_slip = min(
                            full_straddle_slip, cfg.ioc_cap_multiplier * bar_slip,
                        )
                    elif _impact_model is not None:
                        # Continuous sqrt model — returns per-BTC slippage
                        # for both legs combined (matching legacy convention)
                        per_btc_slip = _impact_model.slippage_usd_per_btc(
                            n_i, date_idx=_impact_date_idx[i],
                            trailing_volume=trailing_vol[i] if np.isfinite(trailing_vol[i]) else None,
                        )
                        base_slip = per_btc_slip
                        full_straddle_slip = 2.0 * per_btc_slip * np.exp(
                            cfg.spread_jump_k * jump_frac[i]
                        )
                    else:
                        # Legacy 2D grid
                        sb = min(int(np.searchsorted(size_edges, n_i)), 3)
                        base_slip = slip_grid[sb, vol_bucket[i], i]
                        full_straddle_slip = 2.0 * base_slip * np.exp(
                            cfg.spread_jump_k * jump_frac[i]
                        )

                    # SOR: did a taker print through our limit within 5s?
                    if has_sor:
                        if signal[i] == 1:  # Long: posting bids
                            if (np.isfinite(next_sell_price[i])
                                    and np.isfinite(next_sell_mark[i])):
                                taker_slip_usd = (
                                    (next_sell_mark[i] - next_sell_price[i]) * S[i]
                                )
                                maker_filled_arr[i] = taker_slip_usd >= base_slip
                        elif signal[i] == -1:  # Short: posting asks
                            if (np.isfinite(next_buy_price[i])
                                    and np.isfinite(next_buy_mark[i])):
                                taker_slip_usd = (
                                    (next_buy_price[i] - next_buy_mark[i]) * S[i]
                                )
                                maker_filled_arr[i] = taker_slip_usd >= base_slip

                    straddle_slip = (
                        0.0 if maker_filled_arr[i] else full_straddle_slip
                    )
                    # Emergency taker exit slippage for vega stop-loss
                    if stop_triggered[i]:
                        if _exit_spread_minute[i] > 0:
                            # Minute-level spread from actual trade tape
                            straddle_slip += 2.0 * _exit_spread_minute[i]
                        else:
                            _exit_bar_slip = (
                                _bar_taker_slip[i]
                                if _bar_taker_slip is not None
                                else 12.0
                            )
                            straddle_slip += 2.0 * _exit_bar_slip
                    slippage_arr[i] = straddle_slip

                    # Maker adverse-selection penalty — applied ONLY when a
                    # maker fill was confirmed by SOR (maker_filled_arr[i]).
                    # Taker fills already paid half-spread via straddle_slip
                    # and must not be charged this penalty. Because straddle
                    # vega is always >=0 for a long straddle and the penalty
                    # shifts entry IV against the trade direction (signal*penalty),
                    # the per-BTC cost is `vega_s * penalty` regardless of signal.
                    maker_adv_sel_penalty = (
                        vega_s[i] * cfg.maker_adverse_selection_iv
                        if maker_filled_arr[i]
                        else 0.0
                    )

                    # Per-BTC net PnL: vega + theta + discrete hedge PnL
                    #                  - option fees - option slippage - hedge costs
                    #                  - maker adverse selection (maker fills only)
                    pnl_per_btc = (
                        vega_pnl_btc[i]
                        + theta_pnl_btc[i]
                        + delta_pnl_btc[i]
                        - option_fee_btc[i]
                        - straddle_slip
                        - hedge_costs_btc[i]
                        - maker_adv_sel_penalty
                    )
                    trade_pnl[i] = N[i] * pnl_per_btc

            eq += trade_pnl[i]
            eq = max(eq, 0.0)
            peak_eq = max(peak_eq, eq)
            equity_arr[i] = eq

            # Update rolling stats for sizing (no lookahead)
            if signal[i] != 0 and N[i] > 0:
                _edge_ema += _edge_alpha * (edge_arr[i] - _edge_ema)
                _trade_count += 1
                if trade_pnl[i] > 0:
                    _win_count += 1
                    _win_sum += trade_pnl[i]
                else:
                    _loss_sum += abs(trade_pnl[i])
            # Track IV changes for vega_risk sizing
            if i > 0 and np.isfinite(iv_entry[i]) and np.isfinite(iv_entry[i - 1]):
                _iv_changes.append(abs(iv_entry[i] - iv_entry[i - 1]))
            # Update rolling slip EMA (every bar, not just trades)
            if _bar_taker_slip is not None:
                _slip_ema += _slip_alpha * (_bar_taker_slip[i] - _slip_ema)
            # Update rolling hedge cost EMA (for A1/A3 profit estimation)
            if signal[i] != 0 and N[i] > 0 and np.isfinite(hedge_costs_btc[i]):
                _hedge_cost_ema += _hedge_cost_alpha * (
                    hedge_costs_btc[i] - _hedge_cost_ema
                )
            # Update rolling gross PnL per BTC (for A1/A3 profit estimation)
            if signal[i] != 0 and N[i] > 0:
                gross_per_btc_i = (
                    vega_pnl_btc[i] + theta_pnl_btc[i] + delta_pnl_btc[i]
                )
                if np.isfinite(gross_per_btc_i):
                    _gross_pnl_per_btc_ema += _gross_pnl_alpha * (
                        gross_per_btc_i - _gross_pnl_per_btc_ema
                    )
                    _gross_pnl_count += 1
            # Update slip buffer for A4 regime filter (every bar)
            if _bar_taker_slip is not None:
                _slip_buffer.append(float(_bar_taker_slip[i]))

        if _any_filter_active:
            n_pf = int(profit_filtered_arr.sum())
            n_st = int(slip_thresh_filtered_arr.sum())
            n_rf = int(regime_filtered_arr.sum())
            n_ps = int((profit_scale_arr < 1.0).sum())
            logger.info(
                "Slippage filters: profit=%d, slip_thresh=%d, regime=%d, "
                "scaled=%d (of %d signalled)",
                n_pf, n_st, n_rf, n_ps, int((signal != 0).sum()),
            )

        # Derive r_t for metrics
        equity_prev = np.empty(n_bars)
        equity_prev[0] = cfg.initial_equity
        equity_prev[1:] = equity_arr[:-1]
        r_t = np.where(equity_prev > 0, trade_pnl / equity_prev, 0.0)

        # ── Non-compounding (fixed-notional) path ──────────────────
        # Post-hoc vectorized computation using the SAME per-BTC quantities,
        # slippage surface, dynamic_margin and capacity cap that the compound
        # loop used.  Only difference: N is sized against cfg.initial_equity
        # (fixed) instead of running equity.  Equity evolves additively.
        # This isolates signal quality from compounding amplification.
        if cfg.report_non_compounding:
            # Vectorized sizing at fixed reference equity
            short_pen = cfg.short_sizing_penalty
            n_target_nc = (cfg.initial_equity * dynamic_margin) / (im_per_bar * S)
            n_target_nc = np.where(signal == -1, n_target_nc * short_pen, n_target_nc)
            N_nc = np.minimum(n_target_nc, dynamic_btc_cap)
            N_nc = np.where(signal == 0, 0.0, N_nc)

            # Apply slippage filters to NC path (matching compound loop decisions)
            any_filtered = (
                profit_filtered_arr | slip_thresh_filtered_arr | regime_filtered_arr
            )
            if any_filtered.any():
                N_nc = np.where(any_filtered, 0.0, N_nc)
            if cfg.use_profit_scaling:
                N_nc = N_nc * profit_scale_arr
                N_nc = np.minimum(N_nc, dynamic_btc_cap)
                N_nc = np.where(signal == 0, 0.0, N_nc)

            # ── Kelly-shrinkage boost on NC path ──
            # Mirror the compound-loop application: multiply N_nc by the
            # same per-bar Kelly multiplier that was applied to N. Without
            # this, NC PnL would not reflect the Kelly upsizing and the
            # compound-vs-NC comparison would be misleading.
            if _kelly_mult_arr is not None:
                N_nc = N_nc * _kelly_applied_arr
                N_nc = np.minimum(N_nc, dynamic_btc_cap)
                N_nc = np.where(signal == 0, 0.0, N_nc)

            # Slippage: branch on model type (matching compound loop)
            bar_idx = np.arange(n_bars)
            if _bar_taker_slip is not None:
                # Realistic taker: vectorized version of compound loop
                size_mult_nc = np.sqrt(np.maximum(N_nc, 0.1))
                base_slip_nc = _bar_taker_slip
                full_slip_nc = 2.0 * _bar_taker_slip * size_mult_nc * np.exp(
                    cfg.spread_jump_k * jump_frac
                )
                ioc_cap_nc = cfg.ioc_cap_multiplier * _bar_taker_slip
                full_slip_nc = np.minimum(full_slip_nc, ioc_cap_nc)
                # Liquidity-conditioned sizing for NC path
                if cfg.sizing_model in ("liq", "cost_edge", "composite_v2"):
                    slip_series = pd.Series(_bar_taker_slip)
                    slip_ema_nc = slip_series.ewm(span=200, adjust=False).mean().values
                    liq_scale_nc = np.clip(
                        slip_ema_nc / np.maximum(_bar_taker_slip, 0.01),
                        0.25, cfg.sizing_liq_cap,
                    )
                    N_nc = N_nc * liq_scale_nc
                    N_nc = np.minimum(N_nc, dynamic_btc_cap)
                    N_nc = np.where(signal == 0, 0.0, N_nc)
                    # Recompute slip with updated N
                    size_mult_nc = np.sqrt(np.maximum(N_nc, 0.1))
                    full_slip_nc = 2.0 * _bar_taker_slip * size_mult_nc * np.exp(
                        cfg.spread_jump_k * jump_frac
                    )
                    full_slip_nc = np.minimum(full_slip_nc, ioc_cap_nc)
            elif _impact_model is not None:
                # Continuous impact: vectorized
                base_slip_nc = np.array([
                    _impact_model.slippage_usd_per_btc(
                        N_nc[j], date_idx=_impact_date_idx[j],
                        trailing_volume=trailing_vol[j] if np.isfinite(trailing_vol[j]) else None,
                    ) for j in range(n_bars)
                ])
                full_slip_nc = 2.0 * base_slip_nc * np.exp(
                    cfg.spread_jump_k * jump_frac
                )
            else:
                # Legacy grid
                sb_nc = np.minimum(
                    np.searchsorted(size_edges, N_nc).astype(np.int64), 3,
                )
                base_slip_nc = slip_grid[sb_nc, vol_bucket, bar_idx]
                full_slip_nc = 2.0 * base_slip_nc * np.exp(
                    cfg.spread_jump_k * jump_frac
                )

            # SOR maker-fill detection — vectorized version of the loop logic
            if has_sor:
                long_maker_nc = (
                    (signal == 1)
                    & np.isfinite(next_sell_price)
                    & np.isfinite(next_sell_mark)
                    & (((next_sell_mark - next_sell_price) * S) >= base_slip_nc)
                )
                short_maker_nc = (
                    (signal == -1)
                    & np.isfinite(next_buy_price)
                    & np.isfinite(next_buy_mark)
                    & (((next_buy_price - next_buy_mark) * S) >= base_slip_nc)
                )
                maker_filled_nc = long_maker_nc | short_maker_nc
            else:
                maker_filled_nc = np.zeros(n_bars, dtype=bool)

            slip_nc = np.where(maker_filled_nc, 0.0, full_slip_nc)
            slip_nc = np.where(signal == 0, 0.0, slip_nc)
            # Emergency exit slippage for vega stop-loss
            if cfg.vega_stop_usd > 0 and _bar_taker_slip is not None:
                slip_nc = np.where(
                    stop_triggered, slip_nc + 2.0 * _bar_taker_slip, slip_nc,
                )

            # Per-BTC net PnL (identical structure to the compound loop body)
            pnl_per_btc_nc = (
                vega_pnl_btc
                + theta_pnl_btc
                + delta_pnl_btc
                - option_fee_btc
                - slip_nc
                - hedge_costs_btc
            )
            trade_pnl_nc = N_nc * pnl_per_btc_nc

            # Additive equity: initial + cumulative PnL.  No halting rule —
            # non-compound is a theoretical signal-quality measure that
            # assumes you'd top up capital after losses.
            equity_nc = cfg.initial_equity + np.cumsum(trade_pnl_nc)

            # Returns normalized against fixed reference notional.  This
            # makes r_t_nc comparable across bars regardless of when they
            # occurred — the whole point of the non-compound view.
            r_t_nc = trade_pnl_nc / cfg.initial_equity
        else:
            # Placeholders so downstream code can uniformly reference _nc
            N_nc = np.zeros(n_bars)
            trade_pnl_nc = np.zeros(n_bars)
            slip_nc = np.zeros(n_bars)
            maker_filled_nc = np.zeros(n_bars, dtype=bool)
            equity_nc = np.full(n_bars, cfg.initial_equity)
            r_t_nc = np.zeros(n_bars)

        # Attach to DataFrame
        df = df.copy()
        df["vega_pnl"] = N * vega_pnl_btc
        df["theta_pnl"] = N * theta_pnl_btc
        df["delta_hedge_pnl"] = N * delta_pnl_btc
        df["hedge_costs"] = N * hedge_costs_btc
        df["option_fees"] = N * option_fee_btc
        df["slippage"] = N * slippage_arr
        df["N_contracts"] = N
        df["dynamic_margin"] = dynamic_margin
        df["im_per_bar"] = im_per_bar
        df["r_t"] = r_t
        df["equity"] = equity_arr
        df["gross_pnl"] = N * (vega_pnl_btc + theta_pnl_btc + delta_pnl_btc)
        df["net_pnl"] = trade_pnl
        df["maker_filled"] = maker_filled_arr
        df["instrument_type"] = instrument_type
        # Slippage filter diagnostics
        df["profit_filtered"] = profit_filtered_arr
        df["slip_thresh_filtered"] = slip_thresh_filtered_arr
        df["regime_filtered"] = regime_filtered_arr
        df["profit_scale"] = profit_scale_arr
        df["vega_stopped"] = stop_triggered
        df["exit_frac"] = exit_frac

        # Non-compound columns (always present; zeroed if the flag is off)
        if cfg.report_non_compounding:
            df["vega_pnl_nc"] = N_nc * vega_pnl_btc
            df["theta_pnl_nc"] = N_nc * theta_pnl_btc
            df["delta_hedge_pnl_nc"] = N_nc * delta_pnl_btc
            df["hedge_costs_nc"] = N_nc * hedge_costs_btc
            df["option_fees_nc"] = N_nc * option_fee_btc
            df["slippage_nc"] = N_nc * slip_nc
            df["N_contracts_nc"] = N_nc
            df["r_t_nc"] = r_t_nc
            df["equity_nc"] = equity_nc
            df["gross_pnl_nc"] = N_nc * (vega_pnl_btc + theta_pnl_btc + delta_pnl_btc)
            df["net_pnl_nc"] = trade_pnl_nc
            df["maker_filled_nc"] = maker_filled_nc

        return df

    # ── Step 7: Performance metrics ────────────────────────────

    @staticmethod
    def compute_metrics(
        equity_curve: np.ndarray,
        r_t: np.ndarray,
        trade_pnl: np.ndarray,
        initial_equity: float,
    ) -> dict:
        """Compute tearsheet metrics from compounded equity curve."""
        trade_returns = r_t[r_t != 0]
        trade_dollars = trade_pnl[trade_pnl != 0]
        n_trades = len(trade_returns)

        if n_trades == 0:
            return {"n_trades": 0, "total_pnl": 0.0}

        total_pnl = float(equity_curve[-1] - initial_equity)
        total_return = (equity_curve[-1] / initial_equity - 1.0)

        # Sharpe/Sortino from trade returns (percentage, not dollars)
        bars_per_year = 365.25 * 24 * 4
        mean_r = float(trade_returns.mean())
        std_r = float(trade_returns.std(ddof=1)) if n_trades > 1 else 1e-10

        sharpe = (mean_r / std_r) * np.sqrt(bars_per_year) if std_r > 0 else 0.0

        downside_r = trade_returns[trade_returns < 0]
        down_std = float(downside_r.std(ddof=1)) if len(downside_r) > 1 else 1e-10
        sortino = (mean_r / down_std) * np.sqrt(bars_per_year) if down_std > 0 else 0.0

        # Max drawdown from compounded equity curve
        running_max = np.maximum.accumulate(equity_curve)
        drawdowns = (equity_curve - running_max) / running_max
        max_dd = float(drawdowns.min())

        # Win rate and profit factor (from dollar trade PnL)
        wins = trade_dollars[trade_dollars > 0]
        losses = trade_dollars[trade_dollars < 0]
        win_rate = len(wins) / n_trades if n_trades > 0 else 0.0

        gross_profit = float(wins.sum()) if len(wins) > 0 else 0.0
        gross_loss = float(abs(losses.sum())) if len(losses) > 0 else 1e-10
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        return {
            "n_trades": n_trades,
            "total_pnl": total_pnl,
            "total_return_pct": total_return * 100,
            "sharpe_ratio": float(sharpe),
            "sortino_ratio": float(sortino),
            "max_drawdown_pct": float(max_dd * 100),
            "win_rate_pct": float(win_rate * 100),
            "profit_factor": float(profit_factor),
            "avg_trade_pnl": float(trade_dollars.mean()),
            "avg_win": float(wins.mean()) if len(wins) > 0 else 0.0,
            "avg_loss": float(losses.mean()) if len(losses) > 0 else 0.0,
            "final_equity": float(equity_curve[-1]),
        }

    # ── Prepare data (steps 1-4, reusable) ──────────────────────

    def _prepare(self) -> pd.DataFrame:
        """Run steps 1-4 (load, spread, signals, exit alignment).

        Returns the DataFrame ready for ``compute_pnl``.
        """
        df = self.load_data()

        spread = self.compute_spread(df["jump_fraction"].values)
        df["spread_iv"] = spread

        adjusted_iv = df["iv_used"].values - df["vrp_baseline"].values
        edge = df["rv_pred_pct"].values - adjusted_iv
        df["adjusted_iv"] = adjusted_iv
        df["edge"] = edge
        df["signal"] = self.generate_signals(
            edge, spread / 2.0, df["jump_fraction"].values,
        )
        # Entry filters (W5.X production). No-op when both config fields
        # are default (min_dte_days=0.0, skip_long_hours_utc=()).
        df["signal"] = self._apply_entry_filters(df)

        n_long = (df["signal"] == 1).sum()
        n_short = (df["signal"] == -1).sum()
        n_flat = (df["signal"] == 0).sum()
        logger.info("Signals: %d long, %d short, %d flat (%.1f%% active)",
                    n_long, n_short, n_flat,
                    100 * (n_long + n_short) / len(df))

        df = self.align_exits(df)
        return df

    # ── Delta hedge band sweep ────────────────────────────────

    def optimize_hedge_band(
        self,
        bands: list[float] | None = None,
    ) -> pd.DataFrame:
        """Sweep delta_hedge_band values and compare execution outcomes.

        Runs steps 1-4 once, then loops ``compute_pnl`` per band value.
        Returns a comparison DataFrame.
        """
        if bands is None:
            bands = [0.02, 0.04, 0.06, 0.08, 0.10, 0.15]

        df_base = self._prepare()

        original_band = self.cfg.delta_hedge_band
        rows = []

        for band in bands:
            self.cfg.delta_hedge_band = band
            result = self.compute_pnl(df_base.copy())
            metrics = self.compute_metrics(
                result["equity"].values, result["r_t"].values,
                result["net_pnl"].values, self.cfg.initial_equity,
            )
            active = result[result["signal"] != 0]
            rows.append({
                "band": band,
                "total_pnl": metrics["total_pnl"],
                "sharpe": metrics["sharpe_ratio"],
                "sortino": metrics["sortino_ratio"],
                "max_dd_pct": metrics["max_drawdown_pct"],
                "delta_hedge_pnl": active["delta_hedge_pnl"].sum(),
                "hedge_costs": active["hedge_costs"].sum(),
                "maker_fill_pct": active["maker_filled"].mean() * 100
                                  if "maker_filled" in active.columns else 0.0,
            })
            logger.info("Band %.3f: PnL=$%.0f  Sharpe=%.2f  HedgeCost=$%.0f",
                        band, rows[-1]["total_pnl"], rows[-1]["sharpe"],
                        rows[-1]["hedge_costs"])

        self.cfg.delta_hedge_band = original_band
        return pd.DataFrame(rows)

    # ── Run backtest ───────────────────────────────────────────

    def run_backtest(self) -> tuple[dict, pd.DataFrame]:
        """Execute the full backtest pipeline. Returns (metrics, trade_log).

        When ``cfg.report_non_compounding`` is True, the returned metrics
        dict gains a ``"non_compounding"`` key with the same structure
        (computed from the ``*_nc`` columns of the trade log).
        """
        # 1-4. Load, spread, signals, exit alignment
        df = self._prepare()

        # 5-6. P&L (fees computed inside, both compound and non-compound paths)
        df = self.compute_pnl(df)

        # 7. Metrics — compound (original, from running-equity curve)
        metrics = self.compute_metrics(
            df["equity"].values, df["r_t"].values,
            df["net_pnl"].values, self.cfg.initial_equity,
        )

        # 7b. Metrics — non-compound (fixed-notional view)
        if self.cfg.report_non_compounding and "net_pnl_nc" in df.columns:
            metrics["non_compounding"] = self.compute_metrics(
                df["equity_nc"].values, df["r_t_nc"].values,
                df["net_pnl_nc"].values, self.cfg.initial_equity,
            )

        return metrics, df


# ── Tearsheet printer ──────────────────────────────────────────

def print_tearsheet(metrics: dict, config: BacktestConfig) -> None:
    """Print formatted backtest results."""
    print("\n" + "=" * 55)
    print("  VOL-ARB V7 EXECUTION SIMULATOR — TEARSHEET")
    print("=" * 55)
    print(f"  Initial Equity:     ${config.initial_equity:>12,.0f}")
    print(f"  Margin Util:        {config.target_margin_utilization*100:>12.1f}%")
    print(f"  Participation Cap:  {config.max_participation_rate*100:>12.0f}%")
    print(f"  Short Penalty:      {config.short_sizing_penalty*100:>12.0f}%")
    print(f"  Baseline DVOL:      {config.baseline_target_dvol:>12.1f}")
    print(f"  Adverse Selection:  {config.maker_adverse_selection_iv:>12.1f} vol pts")
    print(f"  Hedge Slippage:     {config.perp_slippage_bps:>12.1f} bps")
    print(f"  Jump Breaker:       {config.max_jump_fraction_for_shorts*100:>12.1f}%")
    print(f"  Entry Threshold:    {config.entry_threshold:>12.1f} vol pts")
    print("-" * 55)

    if metrics["n_trades"] == 0:
        print("  NO TRADES GENERATED")
        print("=" * 55)
        return

    print(f"  Total Trades:       {metrics['n_trades']:>12,d}")
    print(f"  Final Equity:       ${metrics.get('final_equity', 0):>12,.0f}")
    print(f"  Total PnL:          ${metrics['total_pnl']:>12,.2f}")
    print(f"  Total Return:       {metrics['total_return_pct']:>12.2f}%")
    print("-" * 55)
    print(f"  Sharpe Ratio:       {metrics['sharpe_ratio']:>12.2f}")
    print(f"  Sortino Ratio:      {metrics['sortino_ratio']:>12.2f}")
    print(f"  Max Drawdown:       {metrics['max_drawdown_pct']:>12.2f}%")
    print("-" * 55)
    print(f"  Win Rate:           {metrics['win_rate_pct']:>12.1f}%")
    print(f"  Profit Factor:      {metrics['profit_factor']:>12.2f}")
    print(f"  Avg Trade PnL:      ${metrics['avg_trade_pnl']:>12.4f}")
    print(f"  Avg Win:            ${metrics['avg_win']:>12.4f}")
    print(f"  Avg Loss:           ${metrics['avg_loss']:>12.4f}")
    print("=" * 55 + "\n")

    # ── Non-compounding tearsheet (fixed-notional signal-quality view) ──
    if "non_compounding" in metrics and metrics["non_compounding"].get("n_trades", 0) > 0:
        nc = metrics["non_compounding"]
        print("=" * 55)
        print(f"  NON-COMPOUNDING (fixed ${config.initial_equity:,.0f} reference notional)")
        print("=" * 55)
        print(f"  Total Trades:       {nc['n_trades']:>12,d}")
        print(f"  Final Equity:       ${nc.get('final_equity', 0):>12,.0f}")
        print(f"  Total PnL:          ${nc['total_pnl']:>12,.2f}")
        print(f"  Total Return:       {nc['total_return_pct']:>12.2f}%")
        print("-" * 55)
        print(f"  Sharpe Ratio:       {nc['sharpe_ratio']:>12.2f}")
        print(f"  Sortino Ratio:      {nc['sortino_ratio']:>12.2f}")
        print(f"  Max Drawdown:       {nc['max_drawdown_pct']:>12.2f}%")
        print("-" * 55)
        print(f"  Win Rate:           {nc['win_rate_pct']:>12.1f}%")
        print(f"  Profit Factor:      {nc['profit_factor']:>12.2f}")
        print(f"  Avg Trade PnL:      ${nc['avg_trade_pnl']:>12.4f}")
        print(f"  Avg Win:            ${nc['avg_win']:>12.4f}")
        print(f"  Avg Loss:           ${nc['avg_loss']:>12.4f}")
        print("=" * 55)

        # Headline delta: how much of the compound PnL is from compounding?
        nc_pnl = nc['total_pnl']
        cp_pnl = metrics['total_pnl']
        delta = cp_pnl - nc_pnl
        ratio = (cp_pnl / nc_pnl) if nc_pnl != 0 else float("nan")
        print(f"\n  Compounding lift:   ${delta:>+14,.2f}")
        if nc_pnl != 0:
            print(f"  Lift ratio:         {ratio:>12.2f}x  "
                  f"(compound / non-compound total PnL)")
        print("=" * 55 + "\n")
