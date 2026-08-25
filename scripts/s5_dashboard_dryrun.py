"""Dry-run harness for S5 dashboard alerts.

Forces each primary metric into a value that crosses one alert threshold, then
calls compute_metrics() + _alerts() and prints the firing. Does NOT touch
data/live/paper_state.db — synthetic state is built in-memory and fed into the
engine's alert logic via the same call signatures.

Usage:
    python scripts/s5_dashboard_dryrun.py            # run all 8 cases
    python scripts/s5_dashboard_dryrun.py --case sev1_dd  # one case
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "s5_dashboard.py"

# Load s5_dashboard.py as a module (it has no `if __name__ == "__main__"` import
# side effects, so it's safe to importlib-load).
_spec = importlib.util.spec_from_file_location("s5_dashboard", SCRIPT)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["s5_dashboard"] = _mod
_spec.loader.exec_module(_mod)


# ----------------------------- synthetic state ----------------------------- #

class FakeCon:
    """Duck-typed sqlite3.Connection that supports only the queries the
    dashboard actually issues. Each case installs the rows it needs."""

    def __init__(self, tables: dict):
        # tables: { table_name: [ (row, ...), ... ] }
        self._tables = tables
        self._schema = {t: True for t in tables}  # every table "exists"

    def execute(self, sql, params=()):
        s = sql.strip()
        # sqlite_master existence check (parameterized: name is in params)
        if s.startswith("SELECT 1 FROM sqlite_master"):
            name = params[0] if params else None
            return _Cursor([[1]] if self._schema.get(name) else [])
        # SELECT MAX(equity) FROM daily_equity
        if s.startswith("SELECT MAX(equity)"):
            rows = self._tables.get("daily_equity", [])
            v = max((r[1] for r in rows if r[1] is not None), default=None)
            return _Cursor([[v]])
        # SELECT equity FROM daily_equity ORDER BY date DESC LIMIT 1
        if "ORDER BY date DESC LIMIT 1" in s:
            rows = self._tables.get("daily_equity", [])
            return _Cursor([rows[0][1:2]] if rows else [])
        # SELECT date, equity, pnl FROM daily_equity ORDER BY date DESC LIMIT 2
        if "FROM daily_equity ORDER BY date DESC LIMIT 2" in s:
            rows = self._tables.get("daily_equity", [])[:2]
            return _Cursor(rows)
        # SELECT SUM(pnl) FROM daily_equity ...
        if s.startswith("SELECT SUM(pnl)") and "daily_equity" in s:
            rows = self._tables.get("daily_equity", [])
            total = sum((r[2] or 0) for r in rows)
            return _Cursor([[total]])
        # SELECT SUM(pnl) FROM trades ...
        if s.startswith("SELECT SUM(pnl)") and "trades" in s:
            rows = self._tables.get("trades", [])
            total = sum((r[1] if len(r) > 1 and r[1] is not None else 0) for r in rows)
            return _Cursor([[total]])
        # SELECT asset, side FROM positions
        if "FROM positions" in s:
            return _Cursor(self._tables.get("positions", []))
        # SELECT strategy_id, state_json FROM strategy_state
        if "FROM strategy_state" in s:
            return _Cursor(self._tables.get("strategy_state", []))
        # SELECT impl_shortfall_bps, spread_cost_bps, arrival_cost_bps, timing_cost_bps FROM tca_records
        if "impl_shortfall_bps, spread_cost_bps" in s:
            cutoff = params[0] if params else "1970-01-01"
            rows = [r for r in self._tables.get("tca_records", [])
                    if r[0] is not None]
            return _Cursor(rows)
        # SELECT impl_shortfall_bps FROM tca_records WHERE timestamp >= ?
        if "FROM tca_records" in s:
            cutoff = params[0] if params else "1970-01-01"
            rows = [r for r in self._tables.get("tca_records", [])
                    if r[0] is not None]
            return _Cursor(rows)
        # SELECT COUNT(*) FROM trades WHERE timestamp LIKE ?
        if s.startswith("SELECT COUNT(*) FROM trades"):
            today = (params[0] if params else "").rstrip("%")
            rows = [r for r in self._tables.get("trades", [])
                    if r[0] and r[0].startswith(today)]
            return _Cursor([[len(rows)]])
        # SELECT value FROM checkpoints WHERE key='vol_overlay'
        if "FROM checkpoints" in s:
            rows = self._tables.get("checkpoints", [])
            for r in rows:
                if r[0] == "vol_overlay":
                    return _Cursor([[r[1]]])
            return _Cursor([])
        # Fallback: empty
        return _Cursor([])


class _Cursor:
    def __init__(self, rows): self._rows = rows
    def fetchone(self): return self._rows[0] if self._rows else None
    def fetchall(self): return list(self._rows)


# ----------------------------- case builders ------------------------------- #

def _now_iso(): return _mod._now().isoformat()


def _tca_row(slip_bps: float) -> tuple:
    """Single TCA record inside the lookback window.
    Schema matches the SELECT in _fills_1h/_reconciler_24h:
    impl_shortfall_bps, spread_cost_bps, arrival_cost_bps, timing_cost_bps.
    """
    return (slip_bps, 0.0, 0.0, 0.0)


def case_sev1_dd():
    """Drawdown >= 20% (hard kill)."""
    peak, cur = 300_000.0, 230_000.0  # 23.3% DD
    return FakeCon({"daily_equity": [("2026-06-27", cur, -1500.0),
                                       ("2026-06-26", peak, 0.0)]}), \
           "drawdown >= HK_DD (20%)", {"drawdown_pct": (peak - cur) / peak * 100}


def case_sev1_daily():
    """Daily drawdown <= -5% (hard kill)."""
    prev = 300_000.0
    pnl_today = -20_000.0  # -6.67% daily
    cur = prev + pnl_today
    return FakeCon({"daily_equity": [("2026-06-27", cur, pnl_today),
                                       ("2026-06-26", prev, 0.0)]}), \
           "daily_pnl <= -D_DD (-5%)", {"daily_pnl_pct": pnl_today / prev * 100}


def case_sev1_cost():
    """Cost delta 24h >= +1.5 bps (hard kill)."""
    rows = [_tca_row(5.6)] * 20  # avg slip = 5.6 bps, delta = +1.7 bps
    return FakeCon({"tca_records": rows}), \
           "cost_delta_24h >= K_BPS (+1.5 bps)", {"delta_bps": 5.6 - 3.9}


def case_sev2_dd():
    """Drawdown in [15%, 20%) (warning)."""
    peak, cur = 300_000.0, 250_000.0  # 16.67% DD
    return FakeCon({"daily_equity": [("2026-06-27", cur, -500.0),
                                       ("2026-06-26", peak, 0.0)]}), \
           "drawdown in [G_DD, HK_DD) -> SEV2", {"drawdown_pct": (peak - cur) / peak * 100}


def case_sev2_cost():
    """Cost delta in [+1.0, +1.5) bps (warning)."""
    rows = [_tca_row(4.9)] * 20  # avg slip = 4.9 bps, delta = +1.0 bps exactly
    return FakeCon({"tca_records": rows}), \
           "cost_delta_24h in [A_BPS, K_BPS) -> SEV2", {"delta_bps": 4.9 - 3.9}


def case_sev4_trades():
    """n_signals_today > 0 (daily summary)."""
    today = _mod._now().strftime("%Y-%m-%d")
    trades = [(f"{today}T12:00:00+00:00", 100.0, "BTC", "long")]
    return FakeCon({"trades": trades}), \
           "trades today > 0 -> SEV4 daily summary", {"n": len(trades)}


def case_healthy():
    """All green: nothing should fire."""
    peak = cur = 250_000.0
    return FakeCon({"daily_equity": [("2026-06-27", cur, 500.0),
                                       ("2026-06-26", peak, 200.0)]}), \
           "healthy: no firings", {}


# ----------------------------- runner -------------------------------------- #

CASES = {
    "sev1_dd": case_sev1_dd,
    "sev1_daily": case_sev1_daily,
    "sev1_cost": case_sev1_cost,
    "sev2_dd": case_sev2_dd,
    "sev2_cost": case_sev2_cost,
    "sev4_trades": case_sev4_trades,
    "healthy": case_healthy,
}


def run(case_name: str | None) -> int:
    cases = [case_name] if case_name else list(CASES.keys())
    print(f"=== S5 dashboard dry-run | {len(cases)} case(s) ===\n")
    summary = []
    for name in cases:
        con, desc, meta = CASES[name]()
        # patch _reconciler_24h: it falls back to _tca_window(con, 24) which our
        # FakeCon already serves correctly via tca_records. For sev1_dd /
        # sev1_daily / sev4_trades / healthy, no tca rows -> reconciler None
        # (not a breach). For sev1_cost / sev2_cost, tca rows drive it.
        m = _mod.compute(con)
        fired = m.alerts
        ok = (name == "healthy" and not fired) or (name != "healthy" and fired)
        verdict = "PASS" if ok else "FAIL"
        print(f"[{verdict}] {name}: {desc}")
        if meta:
            print(f"         forced value(s): {meta}")
        if fired:
            for a in fired:
                print(f"         FIRED {a['tier']} {a['metric']}={a['value']} (threshold {a['threshold']})")
        else:
            print("         (no firings)")
        summary.append({"case": name, "verdict": verdict, "fired": len(fired),
                        "tiers": [a["tier"] for a in fired]})
        print()
    print("=== SUMMARY ===")
    print(json.dumps(summary, indent=2))
    fails = [s for s in summary if s["verdict"] == "FAIL"]
    return 0 if not fails else 1


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--case", choices=list(CASES.keys()))
    args = p.parse_args()
    return run(args.case)


if __name__ == "__main__":
    sys.exit(main())
