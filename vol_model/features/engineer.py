"""Dollar-Bar-Native Feature Engineer.

ALL features on the dollar-bar grid. No chronological time
dependencies except exchange-driven funding rates.

Memory management:
    - Bar series: sequential by month, cached to parquet
    - Features: computed on full bar series (~630MB)
    - Windowing: per-column with immediate 15m aggregation (no 6.6M × 70 explosion)
    - Explicit del + gc.collect after heavy operations
"""

from __future__ import annotations

import gc
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from vol_model.config import VolModelConfig

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────
# BAR SERIES CONSTRUCTION (sequential, memory-safe, cached)
# ────────────────────────────────────────────────────────────

def _build_bars_sequential(trade_dir: str, asset: str, threshold: float, cache_path: Path) -> pd.DataFrame:
    """Build dollar-bar series one month at a time. Cached to parquet."""
    from TSRV.data_loader import load_trades, available_months
    from TSRV.sampling import _accumulate_bars

    if cache_path.exists():
        bars = pd.read_parquet(cache_path)
        logger.info("  Cached bar series: %d bars", len(bars))
        return bars

    months = available_months(trade_dir, asset)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = cache_path.parent / "_bar_tmp"
    tmp_dir.mkdir(exist_ok=True)

    for month in months:
        tmp_path = tmp_dir / f"{month}.parquet"
        if tmp_path.exists():
            logger.info("    %s: cached", month)
            continue

        try:
            mdf = load_trades(trade_dir, asset, month)
        except Exception:
            continue

        prices = mdf["price"].values.astype(np.float64)
        dvol = mdf["dollar_volume"].values.astype(np.float64)
        buyer = mdf["is_buyer_maker"].values
        ts_vals = mdf.index.values  # numpy datetime64

        bar_idx = _accumulate_bars(dvol, threshold)
        del mdf
        gc.collect()

        if len(bar_idx) < 2:
            continue

        # Vectorized extraction
        starts = np.concatenate([[0], bar_idx[:-1]])
        ends = bar_idx
        n_bars = len(starts)

        out_ts = np.empty(n_bars, dtype="datetime64[ns]")
        out = np.empty((n_bars, 10), dtype=np.float64)

        for i in range(n_bars):
            s, e = starts[i], ends[i]
            if e <= s:
                continue
            seg_p = prices[s:e]
            seg_d = dvol[s:e]
            seg_b = buyer[s:e]
            buy_v = seg_d[~seg_b].sum()
            sell_v = seg_d[seg_b].sum()

            # VoV: intra-bar return volatility (how smoothly the bar formed)
            if e - s > 2:
                seg_lr = np.diff(np.log(seg_p))
                vov = np.std(seg_lr, ddof=1)
            else:
                vov = 0.0

            out_ts[i] = ts_vals[e - 1]
            out[i] = [
                seg_p[-1], seg_p[0], seg_p.max(), seg_p.min(),
                (ts_vals[e - 1] - ts_vals[s]).astype("timedelta64[ms]").astype(np.float64) / 1000,
                e - s, buy_v + sell_v, buy_v, sell_v, vov,
            ]

        cols = ["close", "open", "high", "low", "duration_s",
                "n_trades", "dollar_volume", "buy_volume", "sell_volume", "vov"]
        month_bars = pd.DataFrame(out, columns=cols,
                                  index=pd.DatetimeIndex(out_ts, tz="UTC", name="timestamp"))
        month_bars.to_parquet(tmp_path)
        logger.info("    %s: %d bars", month, len(month_bars))

        del prices, dvol, buyer, ts_vals, out, out_ts, month_bars, bar_idx
        gc.collect()

    # Concat from disk (one file at a time)
    parts = []
    for f in sorted(tmp_dir.glob("*.parquet")):
        parts.append(pd.read_parquet(f))
    bars = pd.concat(parts).sort_index()
    del parts
    gc.collect()

    # Make timestamps unique: add monotonic nanosecond offsets within each ms
    # Multiple bars at the same ms are real (BTC trades sub-ms). Don't dedup.
    dup_offsets = bars.groupby(bars.index).cumcount()
    bars.index = bars.index + pd.to_timedelta(dup_offsets, unit="ns")

    bars["imbalance"] = (bars["buy_volume"] - bars["sell_volume"]) / bars["dollar_volume"].clip(lower=1.0)
    bars["log_return"] = np.log(bars["close"] / bars["close"].shift(1))
    bars["rv_bar"] = bars["log_return"] ** 2

    bars.to_parquet(cache_path)
    logger.info("  Built & cached: %d bars -> %s", len(bars), cache_path)

    # Clean up temp files
    for f in tmp_dir.glob("*.parquet"):
        f.unlink()
    tmp_dir.rmdir()

    return bars


# ────────────────────────────────────────────────────────────
# HURST (subsampled, interpolated — no 6.6M Python loop)
# ────────────────────────────────────────────────────────────

def _hurst_subsampled(returns: np.ndarray, window: int = 100, step: int = 50) -> np.ndarray:
    """Rolling Hurst via R/S, computed every `step` bars, linearly interpolated."""
    n = len(returns)
    result = np.full(n, np.nan)
    if n < window + step:
        return result

    positions = np.arange(window, n, step)
    hurst_vals = np.full(len(positions), np.nan)

    sub_sizes = [s for s in [8, 16, 32, 64] if s <= window // 2]
    if len(sub_sizes) < 2:
        return result

    for idx, pos in enumerate(positions):
        chunk = returns[pos - window: pos]
        valid = chunk[~np.isnan(chunk)]
        if len(valid) < 20:
            continue

        log_rs, log_n = [], []
        for size in sub_sizes:
            nc = len(valid) // size
            if nc < 1:
                continue
            rs_list = []
            for j in range(nc):
                seg = valid[j * size: (j + 1) * size]
                std = seg.std(ddof=1)
                if std < 1e-15:
                    continue
                cd = np.cumsum(seg - seg.mean())
                rs_list.append((cd.max() - cd.min()) / std)
            if rs_list:
                log_rs.append(np.log(np.mean(rs_list)))
                log_n.append(np.log(size))

        if len(log_rs) >= 2:
            hurst_vals[idx] = np.clip(np.polyfit(log_n, log_rs, 1)[0], 0.0, 1.0)

    valid_mask = ~np.isnan(hurst_vals)
    if valid_mask.sum() >= 2:
        vp = positions[valid_mask]
        vv = hurst_vals[valid_mask]
        result = np.interp(np.arange(n), vp, vv)
        result[:vp[0]] = np.nan

    return result


# ────────────────────────────────────────────────────────────
# FEATURE ENGINEER
# ────────────────────────────────────────────────────────────

class FeatureEngineer:

    def __init__(self, config: VolModelConfig):
        self.config = config

    @staticmethod
    def avg_bars_per_day(bar_index: pd.DatetimeIndex, trailing_days: int = 30) -> int:
        cutoff = bar_index[-1] - pd.Timedelta(days=trailing_days)
        recent = bar_index[bar_index >= cutoff]
        n_days = max((recent[-1] - recent[0]).days, 1)
        return max(int(len(recent) / n_days), 10)

    # ── RAW FEATURES (bar-level, ~14 columns on 6.6M rows) ─

    def _compute_raw_features(self, bars: pd.DataFrame, bpd: int) -> pd.DataFrame:
        """Compute all raw features on the bar grid.

        Memory: bars (630MB) + features (740MB) + intermediates (~200MB peak) = ~1.6GB
        """
        from src.afml.fracdiff import frac_diff_ffd

        feats = pd.DataFrame(index=bars.index)
        returns = bars["log_return"].values
        rv_bar = bars["rv_bar"]

        # ── Memory features ──
        logger.info("    Hurst (%d bars, step=50)...", len(returns))
        feats["hurst_100"] = _hurst_subsampled(returns, window=100, step=50)

        logger.info("    FFD (d=%.1f)...", self.config.fracdiff_d)
        lp = np.log(bars["close"]).to_frame("lp")
        ffd = frac_diff_ffd(lp, d=self.config.fracdiff_d, threshold=self.config.fracdiff_threshold)
        feats["ffd_rv"] = ffd["lp"] if "lp" in ffd.columns else np.nan
        del lp, ffd

        # ── RV Z-scores at dollar-bar horizons ──
        for mult, label, cap in [(1, "1d", 50_000), (7, "1w", 50_000), (30, "1m", 200_000)]:
            w = min(max(bpd * mult, 10), cap)
            mp = max(w // 4, 10)
            logger.info("    RV Z-score %s (w=%d)...", label, w)
            rv_roll = rv_bar.rolling(w, min_periods=mp).sum()
            mu = rv_roll.rolling(w, min_periods=mp).mean()
            sigma = rv_roll.rolling(w, min_periods=mp).std()
            feats[f"rv_zscore_{label}"] = (rv_roll - mu) / sigma.clip(lower=1e-15)
            del rv_roll, mu, sigma

        # ── Asynchronous TSRV features (true lagged label, forward-filled to bar grid) ──
        # Shift by one window so bars only see the PREVIOUS completed TSRV — strict
        # point-in-time. Without shift, bars at 12:05 would see TSRV(12:00) which
        # includes trades from 12:05-12:15 (look-ahead at bar level).
        # FFD strips persistence so the tree can't collapse to a lazy AR.
        logger.info("    TSRV forward-fill + FFD (shifted by 1 window)...")
        tsrv_df = pd.read_parquet(self.config.get_tsrv_path())
        ln_vol = np.log(tsrv_df["vol_ann"].clip(lower=1e-10))
        ln_vol_lagged = ln_vol.shift(1)  # TSRV(t-1) at index t: strict point-in-time
        ci = ln_vol_lagged.dropna().index.union(bars.index).sort_values()
        tsrv_at_bar = ln_vol_lagged.reindex(ci).ffill().reindex(bars.index)
        del ln_vol, ln_vol_lagged, ci

        # FFD of TSRV: vol shocks (strips persistence, d=0.4)
        tsrv_for_ffd = tsrv_at_bar.to_frame("tsrv")
        tsrv_ffd = frac_diff_ffd(tsrv_for_ffd, d=self.config.fracdiff_d,
                                  threshold=self.config.fracdiff_threshold)
        feats["tsrv_ffd"] = tsrv_ffd["tsrv"] if "tsrv" in tsrv_ffd.columns else np.nan
        del tsrv_for_ffd, tsrv_ffd

        # Rolling mean TSRV at dollar-bar horizons (vol level context)
        for mult, label, cap in [(1, "1d", 50_000), (7, "1w", 50_000)]:
            w = min(max(bpd * mult, 10), cap)
            mp = max(w // 4, 10)
            feats[f"tsrv_mean_{label}"] = tsrv_at_bar.rolling(w, min_periods=mp).mean()

        # EMA50 of TSRV level — complements rolling means, Stage 18 uses it
        # at the 1d horizon. Used when include_eth_tsrv mirrors this pattern.
        # (BTC TSRV EMA50 is actually computed further below in the EMA block
        # for the feature row, so we don't duplicate here.)
        del tsrv_at_bar

        # ── Realized Quarticity features (HARQ, BPQ 2016) ──
        # `rq` = (n_sub_returns/3) * Σ r^4 on sub-bar log-returns, computed in
        # TSRV.tsrv_series.compute_tsrv_window. Added to TSRV parquet by
        # TSRV/backfill_rq.py on 2026-04-17. Gated on include_rq so existing
        # 44-feature control stays byte-identical when disabled.
        if getattr(self.config, "include_rq", False) and "rq" in tsrv_df.columns:
            logger.info("    HARQ features rq_{1d,1w} (shifted by 1 window, FFD)...")
            rq_series = tsrv_df["rq"].clip(lower=0.0)
            # Log transform for scale — RQ spans many orders of magnitude
            # (4th moment of returns). Use log1p(RQ * scaler) for numerical stability.
            log_rq = np.log1p(rq_series * 1e6)
            log_rq_lagged = log_rq.shift(1)  # strict point-in-time
            ci_rq = log_rq_lagged.dropna().index.union(bars.index).sort_values()
            rq_at_bar = log_rq_lagged.reindex(ci_rq).ffill().reindex(bars.index)
            del log_rq, log_rq_lagged, ci_rq
            for mult, label, cap in [(1, "1d", 50_000), (7, "1w", 50_000)]:
                w = min(max(bpd * mult, 10), cap)
                mp = max(w // 4, 10)
                feats[f"rq_{label}"] = rq_at_bar.rolling(w, min_periods=mp).mean()
            del rq_at_bar
        elif getattr(self.config, "include_rq", False):
            logger.warning(
                "    include_rq=True but 'rq' column missing from TSRV parquet; "
                "run `python -m TSRV.backfill_rq --asset %s --threshold …` first",
                self.config.asset,
            )

        # ── ETH cross-asset TSRV features (Zhang/Cucuringu/Qian 2024 JFE) ──
        # eth_tsrv_mean_{1d,1w} mirror the BTC-TSRV pattern. ETH TSRV parquet
        # is START-stamped (not migrated as of 2026-04-17); shift(1) is
        # directly causal. Gated on include_eth_tsrv.
        if getattr(self.config, "include_eth_tsrv", False):
            eth_path = "TSRV/output/ETHUSDT_tsrv_15m.parquet"
            if Path(eth_path).exists():
                logger.info("    ETH-TSRV features eth_tsrv_mean_{1d,1w} (shifted by 1 window)...")
                eth_df = pd.read_parquet(eth_path)
                # If ETH TSRV ever gets migrated to END-stamped in the future,
                # honor that by shifting its index before the shift(1) lag.
                # Currently ETH is START-stamped so no normalization needed.
                eth_ln = np.log(eth_df["vol_ann"].clip(lower=1e-10))
                eth_ln_lagged = eth_ln.shift(1)  # TSRV(t-1) at index t
                ci_eth = eth_ln_lagged.dropna().index.union(bars.index).sort_values()
                eth_at_bar = eth_ln_lagged.reindex(ci_eth).ffill().reindex(bars.index)
                del eth_ln, eth_ln_lagged, ci_eth, eth_df
                for mult, label, cap in [(1, "1d", 50_000), (7, "1w", 50_000)]:
                    w = min(max(bpd * mult, 10), cap)
                    mp = max(w // 4, 10)
                    feats[f"eth_tsrv_mean_{label}"] = eth_at_bar.rolling(w, min_periods=mp).mean()
                # EMA50 of ETH TSRV level — mirrors the pattern that made
                # `svi_atm_1d_ema50` SHAP-useful in Stage 18.
                feats["eth_tsrv_ema50"] = eth_at_bar.ewm(span=50, min_periods=10).mean()
                del eth_at_bar
            else:
                logger.warning(
                    "    include_eth_tsrv=True but %s missing; disabling ETH block",
                    eth_path,
                )

        # ── Jump decomposition features (from enriched TSRV output) ──
        # Forward-fill jump_fraction, noise proxy, and BPV ratio to bar grid
        # with 1-window shift for strict point-in-time
        if "down_jump_fraction" in tsrv_df.columns:
            logger.info("    Jump/noise features from enriched TSRV...")

            def _ffill_to_bars(series: pd.Series) -> pd.Series:
                """Shift by 1 window and forward-fill to bar grid."""
                lagged = series.shift(1)
                idx = lagged.dropna().index.union(bars.index).sort_values()
                return lagged.reindex(idx).ffill().reindex(bars.index)

            # Directional jump fractions — asymmetric risk in crypto:
            # down-jumps = liquidation cascades (sustained vol)
            # up-jumps = sell-wall absorption (quick mean-revert)
            feats["down_jump_frac"] = _ffill_to_bars(tsrv_df["down_jump_fraction"])
            feats["up_jump_frac"] = _ffill_to_bars(tsrv_df["up_jump_fraction"])
            feats["noise_level"] = _ffill_to_bars(
                tsrv_df["rv_fast"] / (2 * tsrv_df["n_ticks"].clip(lower=1)))
            feats["bpv_tsrv_ratio"] = _ffill_to_bars(
                tsrv_df["bpv"] / tsrv_df["rv_slow"].clip(lower=1e-20))
        del tsrv_df

        # ── Parkinson volatility (range-based, 5x efficient vs close-to-close) ──
        # Only on bars with meaningful range (duration > 1s, H != L)
        logger.info("    Parkinson volatility...")
        hl_ratio = (bars["high"] / bars["low"]).clip(lower=1.0 + 1e-10)
        parkinson_bar = np.log(hl_ratio) ** 2 / (4.0 * np.log(2.0))
        w_park = min(max(bpd, 10), 50_000)
        feats["parkinson_1d"] = pd.Series(parkinson_bar, index=bars.index).rolling(
            w_park, min_periods=max(w_park // 4, 10)).mean()

        # ── Toxicity features ──
        feats["duration_s"] = bars["duration_s"]
        feats["n_trades"] = bars["n_trades"]
        feats["imbalance"] = bars["imbalance"]
        feats["vov"] = bars["vov"]  # intra-bar return volatility

        # ── Microstructure features (VPIN + OFI) ──
        logger.info("    Microstructure (VPIN + OFI)...")
        from vol_model.features.microstructure import MicrostructureFeatures
        ms = MicrostructureFeatures(self.config)

        # VPIN: Numba ring buffer (correct formula, replaces naive rolling mean)
        feats["vpin"] = ms.compute_vpin(bars)

        # OFI: multi-level from L2 (NaN outside L2 coverage window)
        ofi_series, wmid_series = ms.compute_ofi(bars)
        feats["ofi_normalized"] = ofi_series
        feats["depth_mid_spread"] = (wmid_series - bars["close"]).abs() / bars["close"]
        del ofi_series, wmid_series, ms

        # ── OI features ──
        coin = self.config.asset.replace("USDT", "")
        hl_file = Path("data/hl-asset-ctxs/cache_15m") / f"{coin}.parquet"
        if hl_file.exists():
            hl = pd.read_parquet(hl_file)
            if "open_interest" in hl.columns and "mark_px" in hl.columns:
                oi_usd = (hl["open_interest"] * hl["mark_px"]).dropna()
                ci = oi_usd.index.union(bars.index).sort_values()
                oi = oi_usd.reindex(ci).ffill().reindex(bars.index)
                del hl, oi_usd, ci

                w = min(max(bpd, 10), 50_000)
                feats["oi_change_5"] = oi.pct_change(5)
                oi_mu = oi.rolling(w, min_periods=w // 4).mean()
                oi_sig = oi.rolling(w, min_periods=w // 4).std()
                feats["oi_zscore"] = (oi - oi_mu) / oi_sig.clip(lower=1e-15)
                feats["oi_acceleration"] = feats["oi_change_5"].diff(5)
                del oi, oi_mu, oi_sig
            else:
                del hl

        # ── Funding features ──
        fr_file = Path(self.config.funding_dir) / f"{self.config.asset}_funding.parquet"
        if fr_file.exists():
            raw = pd.read_parquet(fr_file)
            fr = raw["fundingRate"]
            ci = fr.index.union(bars.index).sort_values()
            feats["funding_basis"] = fr.reindex(ci).ffill().reindex(bars.index)
            del raw, fr, ci
        else:
            feats["funding_basis"] = 0.0

        h = bars.index.hour + bars.index.minute / 60.0 + bars.index.second / 3600.0
        feats["time_to_funding"] = np.where(h < 8, 8.0 - h,
                                   np.where(h < 16, 16.0 - h, 24.0 - h)) * 3600.0

        gc.collect()
        return feats

    # ── WINDOWING + AGGREGATION (column-by-column, memory-safe) ─

    def _window_and_aggregate(self, raw_feats: pd.DataFrame, freq: str) -> pd.DataFrame:
        """Apply bar-grid lags/EMAs and aggregate to 15m — ONE COLUMN AT A TIME.

        Instead of creating 6.6M × 70 DataFrame (3.7GB OOM), processes each
        raw column independently. Peak memory: ~100MB per column.

        For each column:
            1. Compute shift(1), shift(3), ewm(10), ewm(50) on 6.6M bar series
            2. groupby(window).last() → 32K rows
            3. del the 6.6M temporary immediately
        """
        window_labels = raw_feats.index.floor(freq)

        # Pre-compute n_bars_in_window (count of bars per 15m window)
        n_bars = pd.Series(1, index=raw_feats.index).groupby(window_labels).sum()
        n_bars.name = "n_bars_in_window"

        result_cols = {"n_bars_in_window": n_bars}
        total = len(raw_feats.columns)

        for i, col in enumerate(raw_feats.columns):
            series = raw_feats[col]

            # t(0): last bar's value per window
            result_cols[col] = series.groupby(window_labels).last()

            # Lags on bar grid, then take last per window
            for lag in self.config.lags:
                shifted = series.shift(lag)
                result_cols[f"{col}_lag{lag}"] = shifted.groupby(window_labels).last()
                del shifted

            # EMAs on bar grid, then take last per window
            for span in self.config.ema_spans:
                ema = series.ewm(span=span, min_periods=1).mean()
                result_cols[f"{col}_ema{span}"] = ema.groupby(window_labels).last()
                del ema

            if (i + 1) % 5 == 0:
                logger.info("    Windowed %d/%d columns", i + 1, total)

        result = pd.DataFrame(result_cols)
        result.index.name = "timestamp"

        logger.info("    Done: %d windows × %d features", len(result), len(result.columns))
        return result

    # ── FULL PIPELINE ───────────────────────────────────────

    def build(self) -> pd.DataFrame:
        """Full pipeline: bars → features → window+aggregate → 15m."""
        cache_path = Path(self.config.output_dir) / f"{self.config.asset}_bar_series.parquet"

        logger.info("Step 1: Dollar-bar series...")
        bars = _build_bars_sequential(
            self.config.trade_dir, self.config.asset,
            self.config.dollar_bar_threshold, cache_path)
        logger.info("  %d bars (%s to %s)", len(bars), bars.index[0].date(), bars.index[-1].date())

        bpd = self.avg_bars_per_day(bars.index)
        logger.info("  avg bars/day: %d (1d=%d, 1w=%d, 1m=%d)", bpd, bpd, bpd * 7, bpd * 30)

        logger.info("Step 2: Raw features on bar grid...")
        raw = self._compute_raw_features(bars, bpd)
        logger.info("  %d raw features on %d bars", len(raw.columns), len(raw))

        # Free the heavy bars DataFrame — we only need raw features from here
        del bars
        gc.collect()

        logger.info("Step 3: Window (bar-grid) + aggregate (15m)...")
        freq = f"{self.config.freq_s}s"
        result = self._window_and_aggregate(raw, freq)

        # Free bar-level features
        del raw
        gc.collect()

        return result
