"""Configuration for the volatility prediction model."""

from pydantic import BaseModel, Field, model_validator


# Per-asset microstructure calibration. Populated empirically via
# ``TSRV/run_signature_plot.py --asset {ASSET}``. The calibration is the knee
# detected on the script's RV-versus-threshold curve; retain that result with
# any generated calibration artifact rather than relying on a deleted report.
#
# Fields:
#   - tick_size: Binance USD-M perp tick in USD (from exchange spec).
#   - dollar_bar_threshold: USD notional per dollar bar at the RV-vs-threshold
#     knee. Re-run calibration per-asset; don't extrapolate from BTC.
#   - vpin_bucket_size: USD notional per VPIN bucket (feature-side; distinct
#     from the DRL env VPIN bucket in native units). BTC uses 2× dollar bar
#     as the historical ratio; kept at that ratio for new assets unless a
#     separate VPIN signature plot justifies otherwise.
_PER_ASSET_CALIBRATION: dict[str, dict[str, float]] = {
    "BTCUSDT": {
        "tick_size": 0.10,
        "dollar_bar_threshold": 500_000,  # Stage 0 knee calibration
        "vpin_bucket_size": 1_000_000,
        # BTC TSRV parquet migrated START-stamped → END-stamped on 2026-04-11.
        # load_labels() compensates by
        # shifting the index by -freq so Stage 18 forecast semantics
        # (y[T] = log(vol of [T, T+freq))) are preserved.
        "tsrv_end_stamped": True,
    },
    # ETHUSDT calibrated via `TSRV/run_signature_plot.py --asset ETHUSDT`
    # on 60 evenly-spaced dates (2023-05 → 2026-04). The signature-plot
    # knee detector reports $5M, but RV is essentially flat from $1M onward
    # (161.2% → 160.3% over the $1M–$10M range, well inside 1σ of per-day
    # variation). Using $1M keeps feature granularity high while still
    # attenuating microstructure noise — same trade-off BTC makes at $500K.
    "ETHUSDT": {
        "tick_size": 0.01,  # Binance ETHUSDT perp tick (exchange spec)
        # TSRV threshold verified at $500K via TSRV/backfill_rq --verify-only
        # (2026-04-17) — the $1M dollar_bar_threshold here is a *feature-side*
        # dollar-bar parameter, distinct from the TSRV sub-bar threshold.
        "dollar_bar_threshold": 1_000_000,
        "vpin_bucket_size": 2_000_000,  # 2× dollar_bar per BTC ratio
        # ETH TSRV parquet still START-stamped (not migrated); no shift needed.
        "tsrv_end_stamped": False,
    },
}


class VolModelConfig(BaseModel):
    # Data paths
    tick_dir: str = "data/bookTicker_parquet"
    trade_dir: str = "data/binance-aggTrades-parquet"
    tsrv_path: str = ""  # auto-resolved from asset + freq_s if empty
    funding_dir: str = "data/funding-rates"
    oi_dir: str = "data/oi"
    output_dir: str = "vol_model/output"

    # Asset and timeframe
    asset: str = "BTCUSDT"
    freq_s: int = 900  # 15m = 900s

    # Feature: liquidity (tick scaling) — overridden per-asset
    # via _PER_ASSET_CALIBRATION unless caller passes an explicit value.
    tick_size: float = 0.10

    # Feature: toxicity (dollar bars within windows) — overridden per-asset.
    dollar_bar_threshold: float = 500_000
    vpin_bucket_size: float = 1_000_000
    vpin_n_buckets: int = 50

    # TSRV timestamp convention. Default True matches the post-2026-04-11 BTC
    # migration (bar at index T = vol of [T-freq, T), END-stamped). When True,
    # load_labels() shifts the TSRV index by -freq so downstream label
    # semantics remain "y[T] = log(vol of [T, T+freq))" (Stage 18 forecast
    # semantics). Per-asset default is injected from _PER_ASSET_CALIBRATION.
    # ETH is still START-stamped and sets this to False.
    tsrv_end_stamped: bool = True

    # ── Signal-research feature flags (2026-04-17 Tier 1/2/3 sweep) ──
    # All default False so existing `pruned_svi_v1` control feature set
    # (44 features) is unchanged — flip per-candidate in signal_research.py.

    # Include cross-asset ETH-TSRV features (eth_tsrv_mean_{1d,1w},
    # eth_tsrv_mean_1d_ema50). Mirrors the BTC-TSRV pattern with strict
    # shift(1) point-in-time lag. Zhang-Cucuringu-Qian 2024 JFE intraday
    # commonality thesis; ETH TSRV parquet is still START-stamped so its
    # shift(1) in engineer.py is directly causal.
    include_eth_tsrv: bool = False

    # Include Realized Quarticity features (rq_1d, rq_1w). Fourth-moment
    # analog of RV, vol-of-vol proxy. Bollerslev-Patton-Quaedvlieg 2016
    # (HARQ). Computed inside TSRV compute_tsrv_window on sub-bar returns;
    # loaded from the `rq` column added by TSRV.backfill_rq.
    include_rq: bool = False

    # Include SVI-surface-implied instantaneous ATM spot vol as a feature
    # (svi_atm_spot_vol). Nearest-calibrated-slice ATM IV, no T→0
    # extrapolation. Fan-Wang-Ye 2025 rough-Heston result suggests ~6%
    # QLIKE reduction via options-implied spot-vol (SPX evidence).
    include_svi_spot_vol: bool = False

    @model_validator(mode="before")
    @classmethod
    def _apply_asset_defaults(cls, data):
        """Inject per-asset calibrated defaults when caller didn't pass them.

        Only fields that are still absent from the input dict get the asset
        default — an explicit kwarg always wins. Keeps BTC callers producing
        byte-identical configs while ETH (and future assets) auto-calibrate.
        """
        if isinstance(data, dict):
            asset = data.get("asset", "BTCUSDT")
            defaults = _PER_ASSET_CALIBRATION.get(asset, {})
            for k, v in defaults.items():
                if k not in data:
                    data[k] = v
        return data

    # Feature: microstructure (LOB + VPIN)
    l2_data_dir: str = "data/hl-bbo-raw"
    l2_cache_dir: str = "data/hl-bbo-raw/_cache"
    ofi_n_levels: int = 10
    vpin_window: int = 50

    # Feature: rough memory
    hurst_window: int = 100
    fracdiff_d: float = 0.4
    fracdiff_threshold: float = 1e-5

    # Feature: derivatives
    funding_window_short: int = 50
    funding_window_long: int = 200

    # Windowing / contextualization
    lags: list[int] = Field(default=[1, 3])
    ema_spans: list[int] = Field(default=[10, 50])

    # Labels
    label_clip_quantile: float = 0.999

    # Target reparameterization: when True, the regression target becomes
    # y = log(RV_future) - log(IV_at_T) (i.e., excess log-vol) rather than
    # y = log(RV_future). This reframes the prediction as the model's edge
    # over market IV directly, which:
    #   - Factors out vol-regime level: target distribution stays centered
    #     near zero whether market IV is 20% or 80%, so the model doesn't
    #     need to learn the IV level implicitly.
    #   - Is lookahead-free: IV_at_T is observed at time T, label RV is
    #     measured over [T, T+15min], so subtraction of a known-at-T
    #     quantity from a future quantity is a clean label transform.
    #   - Integrates with backtester via one-line edge change:
    #     ``edge = (exp(y_pred_excess) - 1) * iv_used`` instead of
    #     ``edge = exp(y_pred) * 100 - iv_used``.
    # IV at T is read from {CCY}_vol_arb_svi.parquet::iv_used fallback
    # chain (same as the backtester) so training and inference see the same
    # IV definition. Default False preserves BTC identity.
    predict_excess_logvol: bool = False

    # LightGBM hyperparameters
    n_estimators: int = 1000
    max_depth: int = 6
    num_leaves: int = 31
    learning_rate: float = 0.03
    subsample: float = 0.8
    colsample: float = 0.8
    min_child_samples: int = 20
    reg_alpha: float = 0.1
    reg_lambda: float = 1.0
    early_stopping_rounds: int = 50

    # Cross-validation
    n_splits: int = 5
    pct_embargo: float = 0.01

    # Misc
    seed: int = 42

    def get_tsrv_path(self) -> str:
        """Resolve TSRV path from asset + freq_s if not explicitly set.

        Uses minutes for naming (matches TSRV/run_tsrv_series.py convention).
        """
        if self.tsrv_path:
            return self.tsrv_path
        tf_min = self.freq_s // 60
        return f"TSRV/output/{self.asset}_tsrv_{tf_min}m.parquet"

    # ────────────────────────────────────────────────────────────────────
    # Horizon SSOT (single source of truth for bar frequency)
    # ────────────────────────────────────────────────────────────────────
    # The pipeline supports multiple prediction horizons (15m default; 5m,
    # 30m, 60m via per-horizon presets). Every horizon-dependent constant
    # downstream — feature lookback windows, backtester holding_minutes,
    # sidecar paths — derives from these helpers, NOT from a hardcoded "15"
    # somewhere. ``tests/test_horizon_parametrization.py`` locks the
    # historical h=15 behavior.
    #
    # Why properties (not fields)? `freq_s` already encodes the horizon in
    # seconds; adding a parallel `horizon_minutes` field would create the
    # exact "two sources of truth" failure mode the foundation fix
    # (2026-04-17) was designed to prevent. Properties stay derived and
    # cannot drift.
    @property
    def horizon_minutes(self) -> int:
        """Prediction horizon in minutes. Derived from freq_s."""
        return self.freq_s // 60

    @property
    def bars_per_day(self) -> int:
        """Number of bars in a calendar day at the current horizon."""
        return 1440 // self.horizon_minutes

    @property
    def bars_per_24h(self) -> int:
        """Alias for bars_per_day; clearer when used as a feature lookback."""
        return self.bars_per_day

    @property
    def tf_str(self) -> str:
        """pandas resample/Timedelta-compatible frequency string, e.g. '15min'."""
        return f"{self.horizon_minutes}min"

    @property
    def horizon_suffix(self) -> str:
        """Filename suffix used by horizon-aware path helpers, e.g. 'h60m'."""
        return f"h{self.horizon_minutes}m"

    winsorize_lo: float = 0.01
    winsorize_hi: float = 0.99
