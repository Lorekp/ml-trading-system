#!/usr/bin/env python
"""Validate paper trader: full engine + portfolio simulation.

Aligned with yolo_compound.py (PROD config). Applies the same sizing, portfolio
caps, F1 breaker, deleveraging, and HyroTrader compliance on the BT side so
paper-vs-BT mismatches only appear on genuine divergence.

- Loads config from the paper trader's YAML (default: configs/paper_binance.yaml)
- Seeds breaker state from strategy_state table (per-asset recent pnl/notional)
- Seeds open positions with their actual entry_price/size from the DB
- Tracks BT equity, peak, drawdown, and per-asset/total exposure
- Reports BT rejections per reason (halt, caps, breaker, etc.)

Usage:
    PYTHONPATH="src;." python experiments/validate_paper_v3.py \\
        [--db path/to.db] [--since YYYY-MM-DDTHH:MM] \\
        [--config configs/paper_binance.yaml]
"""

import sys, os, warnings, sqlite3, argparse, json, time
import numpy as np
import pandas as pd
import requests
from pathlib import Path
from collections import defaultdict

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, "src"); sys.path.insert(0, ".")
warnings.filterwarnings("ignore")

from signal_engine.data import load_binance_1m, resample_ohlcv
from signal_engine.features import compute_features
from signal_engine.config import FeatureConfig
from live.signal_generator import SklearnBackend
from live.breaker import F1SymmetricBreaker
from live.config import load_live_config, BreakerConfig, PortfolioConfig

try:
    from afml.bet_sizing import get_signal, discretize_signal
except ImportError:
    def get_signal(prob, num_classes=2): return 2.0 * prob - 1.0
    def discretize_signal(sig, step_size=0.1): return round(sig / step_size) * step_size

MODEL_DIR = Path("data/live/models")
DATA_DIRS = [Path("data/binance-1m")]


def _edge_from_probability(prob: float, sig: float) -> float:
    """Same as experiments/drawdown_improvements.py:40."""
    if np.isfinite(prob):
        return float(np.clip(abs(2.0 * prob - 1.0), 0.0, 1.0))
    return float(np.clip(abs(sig), 0.0, 1.0))


BINANCE_KLINES_URL = "https://fapi.binance.com/fapi/v1/klines"


def _download_day(symbol: str, date: pd.Timestamp, out_path: Path,
                  cap_end_ms: int | None = None) -> int:
    """Download one day of 1m klines for symbol, write to CSV. Returns rows.

    If cap_end_ms is set, only fetch bars with open_time < cap_end_ms (for
    the partial final day when the target end time is mid-day).
    """
    day_start = date.normalize()
    start_ms = int(day_start.timestamp() * 1000)
    end_ms = int((day_start + pd.Timedelta(days=1)).timestamp() * 1000)
    if cap_end_ms is not None:
        end_ms = min(end_ms, cap_end_ms)
    if start_ms >= end_ms:
        return 0

    all_rows: list = []
    fetch_start = start_ms
    while fetch_start < end_ms:
        params = {
            "symbol": symbol, "interval": "1m",
            "startTime": fetch_start, "endTime": end_ms, "limit": 1500,
        }
        data = None
        for attempt in range(3):
            try:
                r = requests.get(BINANCE_KLINES_URL, params=params, timeout=15)
                if r.status_code == 429:
                    time.sleep(2 ** attempt)
                    continue
                if r.status_code == 418:
                    time.sleep(60)
                    continue
                if r.status_code != 200:
                    time.sleep(1 + attempt)
                    continue
                data = r.json()
                break
            except requests.RequestException:
                time.sleep(1 + attempt)
                continue
        if data is None or not isinstance(data, list):
            return 0  # gave up
        if not data:
            break
        all_rows.extend(data)
        fetch_start = data[-1][0] + 60_000
        time.sleep(0.1)

    if not all_rows:
        return 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for k in all_rows:
            f.write(",".join(str(x) for x in k[:6]) + "\n")
    return len(all_rows)


def _count_csv_rows(path: Path) -> int:
    """Count rows in a CSV file (no header). Returns 0 if missing."""
    if not path.exists():
        return 0
    try:
        with open(path, "rb") as f:
            return sum(1 for _ in f)
    except OSError:
        return 0


def _fetch_vol_series(asset: str, end_time: pd.Timestamp) -> pd.Series | None:
    """Fetch 1d klines from Binance and compute 20-day rolling std.

    Mirrors src/live/core/engine.py:_ensure_vol_series: uses native 1d
    klines fetched via REST. The returned series includes the partial
    current-day bar up to `end_time`, matching what the live engine sees
    when it caches vol at that wall-clock time.
    """
    # Binance returns up to `limit` bars ending at the current time.
    # We ask for 200 (same as the engine) and use `endTime` to cap the
    # last bar at our target snapshot time — this gives us the same
    # "partial last day" that the live engine had at --since time.
    end_ms = int(end_time.timestamp() * 1000)
    params = {
        "symbol": f"{asset}USDT", "interval": "1d",
        "endTime": end_ms, "limit": 200,
    }
    try:
        r = requests.get(BINANCE_KLINES_URL, params=params, timeout=15)
        if r.status_code != 200:
            return None
        data = r.json()
    except requests.RequestException:
        return None
    if not isinstance(data, list) or len(data) < 30:
        return None
    rows = []
    for k in data:
        rows.append({
            "ts": pd.Timestamp(int(k[0]), unit="ms"),
            "close": float(k[4]),
        })
    df = pd.DataFrame(rows).set_index("ts")
    return df["close"].pct_change().rolling(20).std()


def ensure_data_coverage(
    assets: list[str],
    start_bar: pd.Timestamp,
    end_bar: pd.Timestamp,
    data_dir: Path = Path("data/binance-1m"),
) -> dict:
    """Ensure each asset has complete daily 1m CSVs for the validation range.

    For each asset × day in [start_bar.date(), end_bar.date()], check that
    the daily CSV (`{SYMBOL}-1m-YYYY-MM-DD.csv`) has the expected number of
    bars and re-download from Binance if missing or short.

    Only touches the date range of the run — historical months (which are
    archived as monthly CSVs) are never modified.

    Returns:
        {asset: {"downloaded": int, "skipped": int, "failed": int}}
    """
    report: dict = {}
    # Need data up to at least end_bar + 1 minute so the `< cutoff` slice
    # at end_bar has the final bar available.
    target_end = (end_bar + pd.Timedelta(minutes=1)).ceil("1min")
    target_end_ms = int(target_end.timestamp() * 1000)
    now_ms = int(pd.Timestamp.utcnow().tz_localize(None).timestamp() * 1000)

    first_date = pd.Timestamp(start_bar.date())
    last_date = pd.Timestamp(end_bar.date())
    date_range = pd.date_range(first_date, last_date, freq="D")

    for asset in assets:
        sym = f"{asset}USDT"
        sym_dir = data_dir / sym
        r = {"downloaded": 0, "skipped": 0, "failed": 0}
        for date in date_range:
            day_end_ms = int((date + pd.Timedelta(days=1)).timestamp() * 1000)
            # Effective end for this day's download: min of day end, target end,
            # and now (can't fetch future minutes).
            eff_end_ms = min(day_end_ms, target_end_ms, now_ms)
            day_start_ms = int(date.timestamp() * 1000)
            if eff_end_ms <= day_start_ms:
                continue  # entirely in the future
            expected = int((eff_end_ms - day_start_ms) / 60_000)
            fname = f"{sym}-1m-{date.strftime('%Y-%m-%d')}.csv"
            out = sym_dir / fname
            have = _count_csv_rows(out)
            # Allow small tolerance (Binance occasionally drops 1-2 bars on
            # brand-new listings); only re-download if we're meaningfully short.
            if have >= expected - 2:
                r["skipped"] += 1
                continue
            wrote = _download_day(sym, date, out, cap_end_ms=eff_end_ms)
            if wrote > 0:
                r["downloaded"] += 1
            else:
                r["failed"] += 1
        report[asset] = r
    return report

OHLCV_FEATURES = [
    "ret_1", "ret_3", "ret_7", "ret_14", "bar_range", "vol_zscore",
    "rsi_14", "stoch_rsi", "macd_hist", "bb_pct_b", "atr_14",
    "dist_swing_high", "dist_swing_low",
]

# Must match src/live/config.py TF_MINUTES exactly
TF_MINUTES = {"15m": 15, "1h": 60, "3h": 180, "4h": 240, "24h": 1440}

TF_CONFIG = [
    ("15m", "15min", 15,  2),
    ("1h",  "1h",    60,  4),
    ("3h",  "3h",    180, 16),
    ("4h",  "4h",    240, 6),
    ("24h", "1D",    1440, 12),
]

HOLD_MAP = {tf: h for tf, _, _, h in TF_CONFIG}


class BTPortfolio:
    """BT-side portfolio simulator mirroring simulate_with_improvements().

    Applies the same sizing, portfolio caps, F1 breaker, deleveraging, and
    HyroTrader compliance as experiments/drawdown_improvements.py so the BT
    accepts/rejects signals on the same criteria as the live engine.
    """

    def __init__(
        self,
        port_cfg: PortfolioConfig,
        breaker_cfg: BreakerConfig,
        initial_balance: float,
        leverage: int,
        max_risk_per_trade: float,
        sl_cap: float,
        vol_series: dict | None = None,
        vol_target: float = 0.025,
        vol_floor: float = 0.3,
        vol_cap: float = 1.5,
    ) -> None:
        self.cfg = port_cfg
        self.initial_balance = float(initial_balance)
        self.leverage = max(1, int(leverage))
        self.max_risk_per_trade = float(max_risk_per_trade or 0.03)
        self.sl_cap = float(sl_cap)
        self.equity = self.initial_balance
        self.peak_equity = self.initial_balance
        self.day_start_equity = self.initial_balance
        self.last_day_key = None
        self.halted = False
        self.day_halted = False
        self.positions: dict = {}  # (ts, asset, tf) -> position dict
        self.breaker = F1SymmetricBreaker.from_config(breaker_cfg)
        self.reject_counts: dict = defaultdict(int)
        self.rejection_log: list = []  # (ts, asset, tf, side, reason, detail)
        self.closed_trades: list = []  # trade-level PnL ledger
        # Vol overlay — exact formula from experiments/circuit_breaker_v2_experiment.py:174-187
        self.vol_series = vol_series or {}
        self.vol_target = float(vol_target)
        self.vol_floor = float(vol_floor)
        self.vol_cap = float(vol_cap)

    def sizing_mult(self, asset: str, bar_time: pd.Timestamp) -> float:
        """Vol overlay multiplier — clip(target_vol / realized_vol, floor, cap).

        Mirrors the live engine exactly (src/live/strategies/s8_sym15.py:363):
          v = vs.iloc[-1]
          return clip(target / v, floor, cap)

        The live engine fetches 1d klines per asset with a 24h TTL cache, so
        the vol value is frozen at the point of first lookup during a session.
        Our vol_series is built once at validator startup from 1d klines
        fetched via REST (see _fetch_vol_series), so iloc[-1] matches the
        paper trader's snapshot-time vol value.
        """
        vs = self.vol_series.get(asset)
        if vs is None or len(vs) == 0:
            return 1.0
        v = vs.iloc[-1]
        if not np.isfinite(v) or v <= 0:
            return 1.0
        return float(np.clip(self.vol_target / v, self.vol_floor, self.vol_cap))

    @property
    def drawdown(self) -> float:
        if self.peak_equity <= 0:
            return 0.0
        return max(0.0, (self.peak_equity - self.equity) / self.peak_equity)

    def roll_day(self, bar_time: pd.Timestamp) -> None:
        day_key = bar_time.strftime("%Y-%m-%d")
        if self.last_day_key != day_key:
            self.last_day_key = day_key
            self.day_start_equity = self.equity
            self.day_halted = False

    def _check_halts(self) -> None:
        if self.drawdown >= self.cfg.max_dd:
            self.halted = True
        abs_floor = self.initial_balance * (1 - self.cfg.max_dd)
        if self.equity <= abs_floor:
            self.halted = True
        daily_floor = (
            self.day_start_equity - self.initial_balance * self.cfg.daily_dd_limit
        )
        if self.equity <= daily_floor:
            self.day_halted = True

    def total_exposure(self) -> float:
        return sum(p["size"] for p in self.positions.values())

    def asset_exposure(self, asset: str) -> float:
        return sum(p["size"] for p in self.positions.values() if p["asset"] == asset)

    def compute_size(
        self, asset: str, prob: float, signal_strength: float,
        bar_time: pd.Timestamp,
    ) -> tuple[float, str]:
        """Return (size_usd, reason). size=0 with reason if rejected.

        Mirrors experiments/drawdown_improvements.py:simulate_with_improvements()
        sizing block (lines 323-338 + 411-441 + 443-455):
          sig_abs = edge_from_probability(prob, strength)
          sig_abs *= clip(regime_mult, regime_exposure_floor, 1.0)  # PROD: regime_mult=1.0
          raw_size = equity * base_sizing_pct * sig_abs * size_mult * sizing_mult
          raw_size = min(raw_size, equity * base_sizing_pct * max_signal_mult * size_mult)
          raw_size *= breaker_mult
          [cap by exposure / margin / notional]
        """
        if self.halted:
            return 0.0, "halt"
        if self.day_halted:
            return 0.0, "day_halted"

        # Deleveraging ramp (drawdown_improvements.py:328-331)
        dd = self.drawdown
        if dd > self.cfg.delever_start:
            size_mult = max(
                1.0 - (dd - self.cfg.delever_start)
                / (self.cfg.max_dd - self.cfg.delever_start),
                0.0,
            )
        else:
            size_mult = 1.0

        # Probability edge sizing (drawdown_improvements.py:334-337)
        sig_abs = _edge_from_probability(prob, signal_strength)
        # regime_mult = 1.0 for PROD (no cached regime_exposure_mult column),
        # still clipped by floor for parity with sim line 335
        regime_mult = 1.0
        sig_abs *= float(np.clip(regime_mult, self.cfg.regime_exposure_floor, 1.0))
        # Vol overlay — exact apply_vol() formula, clip(target_vol/realized, floor, cap)
        vol_mult = self.sizing_mult(asset, bar_time)
        raw = self.equity * self.cfg.base_sizing_pct * sig_abs * size_mult * vol_mult
        raw = min(
            raw,
            self.equity * self.cfg.base_sizing_pct
            * self.cfg.max_signal_mult * size_mult,
        )

        # F1 breaker
        breaker_mult = self.breaker.get_multiplier(asset)
        raw *= breaker_mult

        # Per-asset exposure cap
        asset_exp = self.asset_exposure(asset)
        asset_cap = self.equity * self.cfg.max_per_asset
        if asset_exp + raw > asset_cap:
            raw = max(asset_cap - asset_exp, 0.0)
            if raw <= 0:
                return 0.0, "asset_cap"

        # Total exposure cap
        total_exp = self.total_exposure()
        total_cap = self.equity * self.cfg.max_exposure
        if total_exp + raw > total_cap:
            raw = max(total_cap - total_exp, 0.0)
            if raw <= 0:
                return 0.0, "exposure_cap"

        # HyroTrader margin cap
        total_margin = total_exp / self.leverage
        margin_limit = self.initial_balance * self.cfg.max_margin_pct
        new_margin = raw / self.leverage
        if total_margin + new_margin > margin_limit:
            raw = max((margin_limit - total_margin) * self.leverage, 0.0)
            if raw <= 0:
                return 0.0, "margin_cap"

        # HyroTrader notional cap
        notional_limit = self.initial_balance * self.cfg.max_notional_mult
        if total_exp + raw > notional_limit:
            raw = max(notional_limit - total_exp, 0.0)
            if raw <= 0:
                return 0.0, "notional_cap"

        if raw < 1.0:
            return 0.0, "breaker_floor" if breaker_mult <= self.breaker._floor else "size_too_small"

        return raw, "ok"

    def open_position(
        self, key, asset: str, tf: str, side: str,
        size: float, entry_price: float, hold: int, bars_held: int = 0,
    ) -> None:
        self.positions[key] = {
            "asset": asset, "tf": tf, "side": side, "size": size,
            "entry_price": entry_price, "hold": hold, "bars_held": bars_held,
        }

    def close_position(self, key, exit_price: float) -> float:
        p = self.positions.pop(key)
        if p["entry_price"] <= 0 or p["size"] <= 0:
            return 0.0
        direction = 1.0 if p["side"] == "Buy" else -1.0
        trade_return = direction * (exit_price - p["entry_price"]) / p["entry_price"]
        # Dynamic SL cap (max_risk_per_trade × initial / size, capped at sl_cap)
        dynamic_sl = min(
            self.max_risk_per_trade * self.initial_balance / p["size"], self.sl_cap
        )
        if trade_return < -dynamic_sl:
            trade_return = -dynamic_sl
        pnl = p["size"] * trade_return
        self.equity = max(self.equity + pnl, 0.0)
        if self.equity > self.peak_equity:
            self.peak_equity = self.equity
        self.breaker.record_trade(p["asset"], pnl, p["size"])
        self.closed_trades.append({
            "asset": p["asset"], "tf": p["tf"], "side": p["side"],
            "size": p["size"], "entry_price": p["entry_price"],
            "exit_price": exit_price, "pnl": pnl, "return": trade_return,
            "entry_key": key,
        })
        self._check_halts()
        return pnl


def load_all_trades(db_path, since=None):
    """Load open AND close trades from paper trader DB."""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    where = "WHERE strategy_id='s8_sym15'"
    if since:
        where += f" AND timestamp >= '{since}'"
    c.execute(f"""
        SELECT timestamp, asset, side, timeframe, action, size_usd
        FROM trades {where} ORDER BY timestamp
    """)
    trades = []
    for row in c.fetchall():
        ts = row[0][:16].replace("T", " ")
        trades.append({
            "ts": ts, "asset": row[1], "side": row[2],
            "tf": row[3], "action": row[4], "size_usd": row[5],
        })
    conn.close()
    return trades


def load_seed_positions(db_path, since):
    """Load positions that were open at `since` time (carried over from prior session).

    Returns list of dicts with entry_time, asset, tf, hold_bars, bars_held_at_since,
    plus size_usd and entry_price for BT portfolio seeding.
    """
    if not since:
        return []
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    # Positions opened before `since` that are still open OR closed after `since`
    c.execute("""
        SELECT asset, timeframe, side, entry_time, hold_bars, bars_held,
               exit_time, status, size_usd, entry_price
        FROM positions
        WHERE entry_time < ? AND strategy_id='s8_sym15'
          AND (status='open' OR exit_time >= ?)
        ORDER BY entry_time
    """, (since, since))
    seeds = []
    for row in c.fetchall():
        (asset, tf, side, entry_time, hold_bars, bars_held_now,
         exit_time, status, size_usd, entry_price) = row
        # Compute how many bars had elapsed at `since`
        entry_ts = pd.Timestamp(entry_time[:19].replace("T", " "))
        since_ts = pd.Timestamp(since.replace("T", " "))
        tf_min = TF_MINUTES.get(tf, 15)
        bars_elapsed = int((since_ts - entry_ts).total_seconds() / 60 / tf_min)
        seeds.append({
            "asset": asset, "tf": tf, "side": side,
            "entry_ts": str(entry_ts)[:16],
            "hold_bars": hold_bars, "bars_held": bars_elapsed,
            "size_usd": float(size_usd or 0.0),
            "entry_price": float(entry_price or 0.0),
        })
    conn.close()
    return seeds


def load_strategy_state(db_path: str) -> dict:
    """Load the strategy_state.state_json blob for s8_sym15 (breaker/universe/etc)."""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("SELECT state_json FROM strategy_state WHERE strategy_id='s8_sym15'")
    row = c.fetchone()
    conn.close()
    if not row:
        return {}
    try:
        return json.loads(row[0])
    except (json.JSONDecodeError, TypeError):
        return {}


def check_regime(direction, ohlcv_4h, ema_span=20):
    if len(ohlcv_4h) < ema_span + 5:
        return True
    close = ohlcv_4h["close"]
    ema = close.ewm(span=ema_span, adjust=False).mean()
    regime_raw = np.where(close > ema, 1.0, -1.0)
    regime_shifted = pd.Series(regime_raw, index=ohlcv_4h.index).shift(1)
    latest = regime_shifted.iloc[-1]
    if pd.isna(latest):
        return True
    if direction > 0 and latest < 0:
        return False
    if direction < 0 and latest > 0:
        return False
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="data/live/paper_binance_latest.db")
    parser.add_argument("--since", default=None,
                        help="Only validate trades after this timestamp (e.g. 2026-04-08T20:22)")
    parser.add_argument("--config", default="configs/paper_binance.yaml",
                        help="Paper trader config YAML to align BT with")
    args = parser.parse_args()

    print("=" * 120)
    print("  PAPER TRADER VALIDATION v3 — Full Portfolio Simulation (aligned with PROD)")
    print("=" * 120)

    # Load live config — single source of truth for sizing/caps/breaker
    cfg = load_live_config(Path(args.config))
    strat_cfg = cfg.strategies.get("s8_sym15", {})
    # Prefer strategy-level breaker/portfolio if present, fall back to top-level
    breaker_raw = strat_cfg.get("breaker") or cfg.breaker.model_dump()
    breaker_cfg = BreakerConfig(**breaker_raw) if isinstance(breaker_raw, dict) else cfg.breaker
    port_raw = strat_cfg.get("portfolio") or {}
    port_cfg = cfg.portfolio.model_copy(update=port_raw) if port_raw else cfg.portfolio
    # signal_generator config (conf threshold)
    sg_raw = strat_cfg.get("signal_generator") or {}
    conf_threshold = sg_raw.get("conf_threshold", 0.08)

    initial_balance = cfg.capital
    leverage = cfg.leverage
    max_risk_per_trade = cfg.max_risk_per_trade or 0.03
    sl_cap = cfg.sl_cap
    max_signals_per_bar = cfg.max_signals_per_bar or 20

    print(f"  Config:        {args.config}")
    print(f"  Capital:       ${initial_balance:,.0f}  Leverage: {leverage}×")
    print(f"  base_sizing:   {port_cfg.base_sizing_pct:.3f}  max_signal_mult: {port_cfg.max_signal_mult}")
    print(f"  Caps:          asset={port_cfg.max_per_asset:.2f}  total={port_cfg.max_exposure:.2f}  "
          f"max_dd={port_cfg.max_dd:.2f}  delever={port_cfg.delever_start:.2f}")
    print(f"  HyroTrader:    daily_dd={port_cfg.daily_dd_limit:.2f}  "
          f"margin={port_cfg.max_margin_pct:.2f}  notional={port_cfg.max_notional_mult}×")
    print(f"  Breaker:       window={breaker_cfg.window}  thr={breaker_cfg.threshold}  "
          f"floor={breaker_cfg.floor}  cap={breaker_cfg.cap}")
    print(f"  conf_threshold: {conf_threshold}  max_signals_per_bar: {max_signals_per_bar}")

    # Load models
    backend = SklearnBackend()
    n_models = backend.load_models(MODEL_DIR)
    print(f"  Models: {n_models}")

    fc = FeatureConfig(
        feature_list=OHLCV_FEATURES, winsor_lo=0.01, winsor_hi=0.99,
        vol_filter_quantile=0.50, vol_zscore_window=30, swing_lookback=20,
    )

    # Load all trades (opens + closes)
    all_trades = load_all_trades(args.db, args.since)
    opens = [t for t in all_trades if t["action"] == "open"]
    closes = [t for t in all_trades if t["action"] == "close"]
    print(f"  Paper trades: {len(opens)} opens, {len(closes)} closes")

    # Load strategy state (breaker + universe)
    state = load_strategy_state(args.db)
    universe = state.get("universe", [])
    traded_assets = sorted(set(t["asset"] for t in all_trades))
    # Union: must include every traded asset + full active universe
    assets = sorted(set(universe) | set(traded_assets))
    print(f"  Universe: {len(assets)} assets ({len(traded_assets)} traded, {len(universe)} active)")

    # Build lookups
    paper_open_lookup = defaultdict(list)  # ts -> [(asset, tf, side)]
    for t in opens:
        paper_open_lookup[t["ts"]].append((t["asset"], t["tf"], t["side"]))

    paper_close_lookup = defaultdict(list)  # ts -> [(asset, tf)]
    for t in closes:
        paper_close_lookup[t["ts"]].append((t["asset"], t["tf"]))

    # Determine time range first (needed for data coverage check)
    trade_times = sorted(set(t["ts"] for t in all_trades))
    first_bar = pd.Timestamp(trade_times[0])
    last_bar = pd.Timestamp(trade_times[-1]) + pd.Timedelta(hours=2)  # extend past last trade
    bar_times = []
    t = first_bar
    while t <= last_bar:
        bar_times.append(str(t).replace("-", "-")[:16])
        t += pd.Timedelta(minutes=15)

    # ── Auto-download any missing daily data before loading ──
    print(f"  Checking data coverage [{first_bar} -> {last_bar}]...")
    dl_report = ensure_data_coverage(assets, first_bar, last_bar, DATA_DIRS[0])
    total_dl = sum(r["downloaded"] for r in dl_report.values())
    total_ok = sum(r["skipped"] for r in dl_report.values())
    total_fail = sum(r["failed"] for r in dl_report.values())
    print(f"    Coverage: {total_ok} files complete, {total_dl} downloaded, "
          f"{total_fail} failed")
    if total_dl > 0:
        for asset, r in sorted(dl_report.items()):
            if r["downloaded"] > 0:
                print(f"      {asset}: +{r['downloaded']} file(s)")
    if total_fail > 0:
        fail_assets = [a for a, r in dl_report.items() if r["failed"] > 0]
        print(f"    WARN: failed downloads for: {', '.join(fail_assets[:10])}")

    # Load + resample all data (after ensuring coverage)
    print("  Loading data...")
    ohlcv = {}
    ohlcv_4h = {}
    for asset in assets:
        sym = f"{asset}USDT"
        try:
            raw = load_binance_1m(sym, DATA_DIRS)
            if raw.index.tz is not None:
                raw.index = raw.index.tz_localize(None)
        except Exception as e:
            print(f"    {asset}: no data ({e})")
            continue
        for tf_label, tf_resample, _, _ in TF_CONFIG:
            try:
                ohlcv[(asset, tf_label)] = resample_ohlcv(raw, tf_resample)
            except Exception:
                pass
        try:
            ohlcv_4h[asset] = resample_ohlcv(raw, "4h")
        except Exception:
            pass

    # Build vol_series by fetching 1d klines from Binance, capped at --since time.
    # This mirrors src/live/core/engine.py:_ensure_vol_series which fetches 1d
    # bars from the exchange with a 24h TTL cache — the live engine sees the
    # partial current-day bar at the moment of first lookup.
    since_ts = pd.Timestamp(args.since.replace("T", " ")) if args.since else first_bar
    print(f"  Fetching vol series (1d klines, capped at {since_ts})...")
    vol_series: dict = {}
    for asset in assets:
        vs = _fetch_vol_series(asset, since_ts)
        if vs is not None:
            vol_series[asset] = vs

    print(f"    {len(ohlcv)} asset/TF pairs loaded, {len(vol_series)} vol series\n")

    # Load vol overlay params from config (strategy-level, falling back to top-level)
    vol_overlay_raw = strat_cfg.get("vol_overlay") or {}
    vol_target = vol_overlay_raw.get("target_vol", cfg.vol_overlay.target_vol)
    vol_floor = vol_overlay_raw.get("floor", cfg.vol_overlay.floor)
    vol_cap = vol_overlay_raw.get("cap", cfg.vol_overlay.cap)
    print(f"  Vol overlay: target={vol_target} floor={vol_floor} cap={vol_cap}")

    # ── BT portfolio (full simulation with caps/breaker/sizing) ──
    portfolio = BTPortfolio(
        port_cfg=port_cfg, breaker_cfg=breaker_cfg,
        initial_balance=initial_balance, leverage=leverage,
        max_risk_per_trade=max_risk_per_trade, sl_cap=sl_cap,
        vol_series=vol_series, vol_target=vol_target,
        vol_floor=vol_floor, vol_cap=vol_cap,
    )

    # Reconstruct breaker state from positions that CLOSED before --since.
    # The live strategy_state blob is a running snapshot at "now", so it
    # includes trades from within the validation window — seeding from it
    # would double-count. Instead, pull closed positions with exit_time
    # strictly before args.since and replay them into a fresh breaker.
    conn_seed = sqlite3.connect(args.db)
    cs = conn_seed.cursor()
    cs.execute(
        """SELECT asset, size_usd, pnl FROM positions
           WHERE strategy_id='s8_sym15' AND status='closed'
             AND exit_time < ?
           ORDER BY exit_time""",
        (args.since,),
    )
    seed_trades = cs.fetchall()
    conn_seed.close()
    for asset, size, pnl in seed_trades:
        if pnl is None or size is None:
            continue
        portfolio.breaker.record_trade(asset, float(pnl), float(size))
    seed_assets = set(t[0] for t in seed_trades)
    print(f"  Seeded breaker: {len(seed_trades)} trades across "
          f"{len(seed_assets)} assets (pre-{args.since})")

    # Paper position tracking for exit comparison (lightweight)
    paper_positions = {}
    paper_pos_by_key = defaultdict(list)

    # Seed with positions carried over from prior session — with actual sizes/prices
    seed_positions = load_seed_positions(args.db, args.since)
    if seed_positions:
        print(f"  Seeded {len(seed_positions)} carried-over positions:")
        for sp in seed_positions:
            key = (sp["entry_ts"], sp["asset"], sp["tf"])
            portfolio.open_position(
                key=key, asset=sp["asset"], tf=sp["tf"], side=sp["side"],
                size=sp["size_usd"], entry_price=sp["entry_price"],
                hold=sp["hold_bars"], bars_held=sp["bars_held"],
            )
            paper_positions[key] = {
                "bars_held": sp["bars_held"], "hold_bars": sp["hold_bars"],
                "side": sp["side"],
            }
            paper_pos_by_key[(sp["asset"], sp["tf"])].append(sp["entry_ts"])
            remaining = sp["hold_bars"] - sp["bars_held"]
            print(f"    {sp['asset']:>6s}/{sp['tf']:<3s}  opened {sp['entry_ts']}  "
                  f"held={sp['bars_held']}/{sp['hold_bars']}  remaining={remaining}  "
                  f"${sp['size_usd']:,.0f} @ {sp['entry_price']:.4f}")
        print()

    # Results
    open_rows = []
    exit_rows = []
    open_match = open_mismatch = open_paper_only = open_bt_only = 0

    # Cap signal generation at the last paper trade time — beyond that, the
    # paper trader snapshot hasn't processed bars yet, so any BT signals at
    # later bars would be spurious "BT_ONLY" entries. Exits still get tracked
    # in the extended window so hold-expired exits are compared correctly.
    last_paper_open_ts = pd.Timestamp(max(t["ts"] for t in opens)) if opens else first_bar

    def _price_at(asset: str, tf: str, bar_cutoff: pd.Timestamp) -> float | None:
        df = ohlcv.get((asset, tf))
        if df is None or len(df) == 0:
            return None
        # Use the last bar strictly before the cutoff (matches how signals see data)
        sliced = df[df.index < bar_cutoff]
        if len(sliced) == 0:
            return None
        return float(sliced["close"].iloc[-1])

    for ts_str in bar_times:
        bar_time = pd.Timestamp(ts_str)
        total_min = bar_time.hour * 60 + bar_time.minute
        generate_signals = bar_time <= last_paper_open_ts

        # Roll day at UTC midnight
        portfolio.roll_day(bar_time)

        active_tfs = []
        for tf_label, _, tf_min, hold in TF_CONFIG:
            if total_min % tf_min == 0:
                active_tfs.append((tf_label, hold, tf_min))

        # ── Tick BT positions and collect exits ──
        bt_exits_this_bar = set()
        for tf_label, _, tf_min in active_tfs:
            expired = []
            for pos_key, pos in portfolio.positions.items():
                if pos["tf"] == tf_label:
                    pos["bars_held"] += 1
                    if pos["bars_held"] >= pos["hold"]:
                        expired.append(pos_key)
            for pk in expired:
                # Exit price = close of current bar (last completed bar for this TF)
                cutoff_min = ((total_min + tf_min) // tf_min) * tf_min
                bar_cut = bar_time.replace(
                    hour=(cutoff_min // 60) % 24,
                    minute=cutoff_min % 60,
                    second=0, microsecond=0)
                if cutoff_min >= 1440:
                    bar_cut = bar_cut + pd.Timedelta(days=1) - pd.Timedelta(hours=bar_cut.hour, minutes=bar_cut.minute)
                    bar_cut = bar_cut.replace(hour=(cutoff_min - 1440) // 60, minute=(cutoff_min - 1440) % 60)
                asset, tf = pk[1], pk[2]
                exit_price = _price_at(asset, tf, bar_cut) or portfolio.positions[pk]["entry_price"]
                portfolio.close_position(pk, exit_price)
                bt_exits_this_bar.add((asset, tf))

        # ── Paper closes at this bar ──
        paper_closes_this_bar = set()
        for asset, tf in paper_close_lookup.get(ts_str, []):
            paper_closes_this_bar.add((asset, tf))
            pkey_list = paper_pos_by_key.get((asset, tf), [])
            if pkey_list:
                removed_ts = pkey_list.pop(0)  # FIFO
                paper_positions.pop((removed_ts, asset, tf), None)

        # ── Compare exits ──
        all_exit_keys = paper_closes_this_bar | bt_exits_this_bar
        for (asset, tf) in sorted(all_exit_keys):
            p_exit = (asset, tf) in paper_closes_this_bar
            bt_exit = (asset, tf) in bt_exits_this_bar
            if p_exit and bt_exit:
                exit_rows.append((ts_str, asset, tf, "EXIT", "EXIT", "MATCH"))
            elif p_exit and not bt_exit:
                exit_rows.append((ts_str, asset, tf, "EXIT", "---", "PAPER_ONLY"))
            elif bt_exit and not p_exit:
                exit_rows.append((ts_str, asset, tf, "---", "EXIT", "BT_ONLY"))

        # ── Generate BT entry signals (only up to last paper trade time) ──
        bt_signals = []
        if not generate_signals:
            # Past last paper trade — skip signal generation, continue to next bar
            # so exits can still be tracked.
            continue
        for tf_label, hold, tf_min_val in active_tfs:
            for asset in assets:
                df = ohlcv.get((asset, tf_label))
                if df is None or len(df) < 30:
                    continue
                if not backend.has_model(asset, tf_label):
                    continue

                # Per-TF bar cutoff matching engine.py
                cutoff_min = (total_min // tf_min_val) * tf_min_val
                bar_cutoff = bar_time.replace(
                    hour=cutoff_min // 60, minute=cutoff_min % 60,
                    second=0, microsecond=0)
                df_slice = df[df.index < bar_cutoff]
                if len(df_slice) < 30:
                    continue

                try:
                    feat = compute_features(df_slice, fc)
                except Exception:
                    continue
                if len(feat) == 0:
                    continue

                last = feat.iloc[[-1]]
                cols = [c for c in fc.feature_list if c in last.columns]
                if len(cols) < len(fc.feature_list) * 0.8:
                    continue
                vals = last[cols]
                if vals.isna().any().any():
                    continue

                prob = backend.predict(asset, tf_label, vals.values)
                if prob is None:
                    continue

                direction = 1.0 if prob > 0.5 else -1.0
                conf = abs(prob - 0.5)

                if conf < conf_threshold:
                    continue

                # Regime filter — 4H cutoff
                df_4h = ohlcv_4h.get(asset)
                if df_4h is not None:
                    cutoff_4h_min = (total_min // 240) * 240
                    bar_cutoff_4h = bar_time.replace(
                        hour=cutoff_4h_min // 60, minute=cutoff_4h_min % 60,
                        second=0, microsecond=0)
                    df_4h_slice = df_4h[df_4h.index < bar_cutoff_4h]
                    if not check_regime(direction, df_4h_slice):
                        continue

                sig = get_signal(prob, num_classes=2)
                sig_disc = discretize_signal(sig, step_size=0.1)
                strength = abs(float(sig_disc))

                side = "Buy" if direction > 0 else "Sell"
                entry_price = float(df_slice["close"].iloc[-1])
                bt_signals.append({
                    "asset": asset, "tf": tf_label, "side": side,
                    "prob": prob, "conf": conf, "strength": strength,
                    "hold": hold, "entry_price": entry_price,
                })

        # Sort by strength; truncate to max_signals_per_bar
        bt_signals.sort(key=lambda s: -s["strength"])
        bt_signals_capped = bt_signals[:max_signals_per_bar]
        bt_max_sig_rejected = bt_signals[max_signals_per_bar:]
        for s in bt_max_sig_rejected:
            portfolio.reject_counts["max_signals"] += 1
            portfolio.rejection_log.append((
                ts_str, s["asset"], s["tf"], s["side"], "max_signals",
                f"rank>{max_signals_per_bar}"
            ))

        # Run portfolio sizing — reject or accept each signal in rank order
        bt_at_bar = {}
        for s in bt_signals_capped:
            size, reason = portfolio.compute_size(
                s["asset"], s["prob"], s["strength"], bar_time)
            if size > 0:
                pos_key = (ts_str, s["asset"], s["tf"])
                portfolio.open_position(
                    key=pos_key, asset=s["asset"], tf=s["tf"], side=s["side"],
                    size=size, entry_price=s["entry_price"], hold=s["hold"],
                )
                bt_at_bar[(s["asset"], s["tf"])] = {**s, "bt_size": size}
            else:
                portfolio.reject_counts[reason] += 1
                portfolio.rejection_log.append((
                    ts_str, s["asset"], s["tf"], s["side"], reason,
                    f"eq=${portfolio.equity:,.0f} exp=${portfolio.total_exposure():,.0f}"
                ))

        # Paper opens at this bar
        paper_at_bar = {}
        for asset, tf, side in paper_open_lookup.get(ts_str, []):
            paper_at_bar[(asset, tf)] = side
            pos_key = (ts_str, asset, tf)
            paper_positions[pos_key] = {
                "bars_held": 0, "hold_bars": HOLD_MAP.get(tf, 4), "side": side,
            }
            paper_pos_by_key[(asset, tf)].append(pos_key[0])

        # ── Compare opens ──
        all_keys = set(paper_at_bar.keys()) | set(bt_at_bar.keys())
        for (asset, tf) in sorted(all_keys):
            p_side = paper_at_bar.get((asset, tf))
            bt_sig = bt_at_bar.get((asset, tf))

            if p_side and bt_sig:
                status = "MATCH" if p_side == bt_sig["side"] else "MISMATCH"
                if status == "MATCH":
                    open_match += 1
                else:
                    open_mismatch += 1
                open_rows.append((ts_str, asset, tf, p_side, bt_sig["side"],
                                  f"{bt_sig['prob']:.4f}", f"{bt_sig['conf']:.4f}", status))
            elif p_side and not bt_sig:
                open_paper_only += 1
                open_rows.append((ts_str, asset, tf, p_side, "---", "---", "---", "PAPER_ONLY"))
            elif bt_sig and not p_side:
                open_bt_only += 1
                open_rows.append((ts_str, asset, tf, "---", bt_sig["side"],
                                  f"{bt_sig['prob']:.4f}", f"{bt_sig['conf']:.4f}", "BT_ONLY"))

    # ── Print ENTRY results ──
    print(f"  {'':>3s} {'Time':>16s} {'Asset':>8s} {'TF':>4s} {'Paper':>6s} {'BT':>6s} "
          f"{'Prob':>8s} {'Conf':>8s} {'Status':>12s}")
    print(f"  {'-'*85}")
    for row in open_rows:
        print(f"  {'':>3s} {row[0]:>16s} {row[1]:>8s} {row[2]:>4s} {row[3]:>6s} {row[4]:>6s} "
              f"{row[5]:>8s} {row[6]:>8s} {row[7]:>12s}")

    # ── Print EXIT results ──
    exit_match = sum(1 for r in exit_rows if r[5] == "MATCH")
    exit_paper_only = sum(1 for r in exit_rows if r[5] == "PAPER_ONLY")
    exit_bt_only = sum(1 for r in exit_rows if r[5] == "BT_ONLY")

    print(f"\n  {'':>3s} {'Time':>16s} {'Asset':>8s} {'TF':>4s} {'Paper':>6s} {'BT':>6s} {'Status':>12s}")
    print(f"  {'-'*65}")
    for row in exit_rows:
        print(f"  {'':>3s} {row[0]:>16s} {row[1]:>8s} {row[2]:>4s} {row[3]:>6s} {row[4]:>6s} {row[5]:>12s}")

    # ── Summary ──
    total_opens = len(opens)
    total_closes = len(closes)
    print(f"\n{'='*120}")
    print(f"  ENTRY SUMMARY")
    print(f"{'='*120}")
    print(f"  Paper opens:                    {total_opens}")
    print(f"  Match (both agree):             {open_match:>4d}  "
          f"({open_match/max(total_opens,1)*100:.0f}%)")
    print(f"  Direction mismatch:             {open_mismatch:>4d}  "
          f"({open_mismatch/max(total_opens,1)*100:.0f}%)")
    print(f"  Paper-only (BT no signal):      {open_paper_only:>4d}  "
          f"({open_paper_only/max(total_opens,1)*100:.0f}%)")
    print(f"  BT-only (paper didn't open):    {open_bt_only:>4d}")

    print(f"\n{'='*120}")
    print(f"  EXIT SUMMARY")
    print(f"{'='*120}")
    print(f"  Paper closes:                   {total_closes}")
    print(f"  Exit match (same bar):          {exit_match:>4d}  "
          f"({exit_match/max(total_closes,1)*100:.0f}%)")
    print(f"  Paper-only exit:                {exit_paper_only:>4d}  "
          f"({exit_paper_only/max(total_closes,1)*100:.0f}%)")
    print(f"  BT-only exit:                   {exit_bt_only:>4d}")

    # ── BT Rejection diagnostic ──
    if portfolio.rejection_log:
        print(f"\n{'='*120}")
        print(f"  BT REJECTIONS (signals generated but not opened)")
        print(f"{'='*120}")
        print(f"  {'Time':>16s} {'Asset':>8s} {'TF':>4s} {'Side':>6s} {'Reason':>16s}  {'Detail'}")
        print(f"  {'-'*110}")
        for r in portfolio.rejection_log:
            print(f"  {r[0]:>16s} {r[1]:>8s} {r[2]:>4s} {r[3]:>6s} {r[4]:>16s}  {r[5]}")
        print(f"\n  Reject counts by reason:")
        for reason, count in sorted(portfolio.reject_counts.items(), key=lambda x: -x[1]):
            print(f"    {reason:>16s}: {count}")

    # ── Per-trade PnL comparison: paper vs BT ──
    # Load paper PnLs by (entry_ts, asset, tf) from positions table
    conn_p = sqlite3.connect(args.db)
    cp = conn_p.cursor()
    cp.execute(
        """SELECT entry_time, asset, timeframe, size_usd, entry_price,
                  exit_price, pnl, status
           FROM positions WHERE strategy_id='s8_sym15' AND entry_time >= ?
             AND status='closed'""",
        (args.since,),
    )
    paper_closed = {}
    for r in cp.fetchall():
        entry_time, a, tf, sz, ep, xp, pnl, st = r
        # Normalize to YYYY-MM-DD HH:MM (matches BT bar_times format)
        norm = entry_time[:16].replace("T", " ")
        paper_closed[(norm, a, tf)] = {
            "size": sz, "entry_price": ep, "exit_price": xp, "pnl": pnl,
        }
    conn_p.close()

    print(f"\n{'='*120}")
    print(f"  PER-TRADE PnL COMPARISON (closed trades)")
    print(f"{'='*120}")
    print(f"  {'Time':<17s} {'Asset':>5s} {'TF':<4s} {'Side':<5s} "
          f"{'P_Entry':>10s} {'BT_Entry':>10s} "
          f"{'EntDrift':>10s} "
          f"{'P_PnL':>8s} {'B_PnL':>8s}")
    print(f"  {'-'*105}")
    total_ppnl = total_bpnl = 0
    n_matched = 0
    entry_drifts_long_bps = []
    entry_drifts_short_bps = []
    for t in portfolio.closed_trades:
        key = t["entry_key"]
        pp = paper_closed.get(key)
        if pp is None:
            continue
        n_matched += 1
        total_ppnl += pp["pnl"] or 0
        total_bpnl += t["pnl"]
        # Entry drift in bps (paper - BT) / BT * 10000
        if t["entry_price"] > 0:
            drift_raw_bps = (pp["entry_price"] - t["entry_price"]) / t["entry_price"] * 10000
        else:
            drift_raw_bps = 0
        # Sign convention: positive drift = paper got WORSE entry
        # For long: paying more = worse = positive when paper > BT
        # For short: receiving less = worse = negative when paper > BT (paper sold higher = better, so flip)
        if t["side"] == "Buy":
            signed_drift = drift_raw_bps  # paying more = worse
            entry_drifts_long_bps.append(signed_drift)
        else:
            signed_drift = -drift_raw_bps  # selling lower = worse
            entry_drifts_short_bps.append(signed_drift)
        print(f"  {key[0]:<17s} {t['asset']:>5s} {t['tf']:<4s} {t['side']:<5s} "
              f"{pp['entry_price']:>10.4f} {t['entry_price']:>10.4f} "
              f"{signed_drift:>+8.1f}bp "
              f"${pp['pnl'] or 0:>+7,.0f} ${t['pnl']:>+7,.0f}")
    print(f"  {'-'*105}")
    print(f"  TOTALS ({n_matched} matched trades):"
          f"                                              "
          f"${total_ppnl:>+7,.0f} ${total_bpnl:>+7,.0f}")

    # Entry fill drift stats
    all_drifts = entry_drifts_long_bps + entry_drifts_short_bps
    if all_drifts:
        print()
        print(f"  ENTRY FILL DRIFT (paper vs BT bar-close, in bps — positive = paper got WORSE fill)")
        print(f"  {'-'*100}")
        print(f"  All trades:   n={len(all_drifts):>3d}  "
              f"mean={np.mean(all_drifts):>+7.1f}bp  "
              f"median={np.median(all_drifts):>+7.1f}bp  "
              f"std={np.std(all_drifts):>6.1f}bp  "
              f"min={min(all_drifts):>+7.1f}bp  max={max(all_drifts):>+7.1f}bp")
        if entry_drifts_long_bps:
            print(f"  Longs:        n={len(entry_drifts_long_bps):>3d}  "
                  f"mean={np.mean(entry_drifts_long_bps):>+7.1f}bp  "
                  f"median={np.median(entry_drifts_long_bps):>+7.1f}bp")
        if entry_drifts_short_bps:
            print(f"  Shorts:       n={len(entry_drifts_short_bps):>3d}  "
                  f"mean={np.mean(entry_drifts_short_bps):>+7.1f}bp  "
                  f"median={np.median(entry_drifts_short_bps):>+7.1f}bp")

    # ── BT portfolio state ──
    print(f"\n{'='*120}")
    print(f"  BT PORTFOLIO STATE (end of run)")
    print(f"{'='*120}")
    print(f"  Initial capital:   ${portfolio.initial_balance:>12,.2f}")
    print(f"  Final equity:      ${portfolio.equity:>12,.2f}  "
          f"(PnL: ${portfolio.equity - portfolio.initial_balance:+,.2f})")
    print(f"  Paper equity:      ${initial_balance + total_ppnl:>12,.2f}  "
          f"(realized from closed trades only)")
    print(f"  Peak equity:       ${portfolio.peak_equity:>12,.2f}")
    print(f"  Drawdown:          {portfolio.drawdown*100:>12.2f}%")
    print(f"  Halted:            {portfolio.halted}")
    print(f"  Open BT positions: {len(portfolio.positions)}")
    print(f"  Total exposure:    ${portfolio.total_exposure():>12,.2f}")
    # Breaker per-asset state
    print(f"\n  Breaker state per asset (recent window):")
    for asset, data in portfolio.breaker.get_state().items():
        pnl_list = data.get("pnl", [])
        not_list = data.get("notional", [])
        if not pnl_list or not not_list:
            continue
        total_pnl = sum(pnl_list)
        total_not = sum(not_list)
        ratio = total_pnl / total_not * 100 if total_not > 0 else 0
        mult = portfolio.breaker.get_multiplier(asset)
        print(f"    {asset:>6s}: n={len(pnl_list):>2d}  pnl=${total_pnl:>+9,.2f}  "
              f"ratio={ratio:>+6.2f}%  mult={mult:.2f}")
    print(f"{'='*120}")


if __name__ == "__main__":
    main()
