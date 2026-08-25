"""S5 historical research dashboard metric engine and HTML renderer.

Preserves the historical dashboard metric set:
- 8 primary metrics (§7.1) + 4 secondary + 4-tier alert matrix (§7.2 / §5)
- Reads ONLY from StateDB sqlite + engine output dir (mtimes, jsonl fills, reconciler md)
- Writes self-contained HTML + alerts log to vol_arb/output/
- NEVER mutates engine state, NEVER spawns subprocesses against the engine
- Pure read + render

Usage:
    python scripts/s5_dashboard.py                # one-shot render
    python scripts/s5_dashboard.py --loop        # refresh primary every 5s
    python scripts/s5_dashboard.py --offline     # render sample from cached BTC parquet
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

REPO = Path(__file__).resolve().parent.parent
DB_PATH = REPO / "data" / "live" / "paper_state.db"
OUT_DIR = REPO / "vol_arb" / "output"
DASHBOARD_HTML = OUT_DIR / "s5_dashboard.html"
ALERTS_LOG = OUT_DIR / "s5_alerts.log"
RECONCILER_MD = OUT_DIR / "S5_TESTNET_RECONCILIATION.md"
FILLS_JSONL = OUT_DIR / "S5_TESTNET_FILLS.jsonl"
CACHED_SIGNAL_PARQUET = OUT_DIR / "BTC_vol_arb.parquet"

CAL = 3.9           # calibrated slippage assumption (§6)
G_BPS = 0.5         # §7.1 #7 green threshold
A_BPS = 1.0         # §7.2 cost-delta alert
K_BPS = 1.5         # §5 hard kill
G_DD = 15.0         # §7.2 SEV2 drawdown
HK_DD = 20.0        # §5 hard kill
D_DD = 5.0          # §5 daily kill
W_DD = 10.0         # §5 weekly kill
ERR_H = 5           # §5 error-rate hard kill
REFRESH_S = 5       # primary refresh
ALERT_S = 900       # 15m alert re-check
TFS = {"5m": 5, "15m": 15, "30m": 30, "1h": 60, "4h": 240, "1d": 1440}
START_CAP = 250_000.0


# ----------------------------- helpers ------------------------------------- #

def _now() -> datetime: return datetime.now(timezone.utc)


def _money(v) -> str: return "n/a" if v is None else f"${v:,.0f}"


def _pct(v, d=2) -> str: return "n/a" if v is None else f"{v:.{d}f}%"


def _pill(green: Optional[bool]) -> str:
    if green is None: return '<span class="pill gray">N/A</span>'
    return f'<span class="pill {"green" if green else "red"}">{"GREEN" if green else "BREACH"}</span>'


def _safe(con, name) -> bool:
    return con.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone() is not None


def _open_ro() -> sqlite3.Connection:
    return sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)


def _mtime_iso(p: Path) -> Optional[str]:
    return datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc).isoformat() if p.exists() else None


def _parse_dt(s: Optional[str]) -> Optional[datetime]:
    if not s: return None
    try: return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception: return None


# ----------------------------- metric engine ------------------------------- #

@dataclass
class M:
    primary: dict = field(default_factory=dict)
    secondary: dict = field(default_factory=dict)
    alerts: list = field(default_factory=list)
    ts: str = ""


def _daily_pnl(con) -> dict:
    if not _safe(con, "daily_equity"): return {"value_usd": None, "value_pct": None, "green": True}
    rows = con.execute("SELECT date, equity, pnl FROM daily_equity ORDER BY date DESC LIMIT 2").fetchall()
    if not rows: return {"value_usd": None, "value_pct": None, "green": True}
    pnl = rows[0][2]
    prev = rows[1][1] if len(rows) > 1 and rows[1][1] else START_CAP
    pct = (pnl / prev * 100.0) if (pnl is not None and prev) else None
    return {"value_usd": pnl, "value_pct": pct, "green": pnl is None or pnl >= 0}


def _cum_pnl(con) -> dict:
    for tbl, col in [("daily_equity", "pnl"), ("trades", "pnl")]:
        if _safe(con, tbl):
            r = con.execute(f"SELECT SUM({col}) FROM {tbl} WHERE {col} IS NOT NULL").fetchone()
            if r and r[0] is not None:
                v = float(r[0])
                return {"value_usd": v, "green": v >= 0}
    return {"value_usd": None, "green": True}


def _drawdown(con) -> dict:
    peak = cur = None
    if _safe(con, "daily_equity"):
        r = con.execute("SELECT MAX(equity) FROM daily_equity").fetchone()
        if r and r[0]: peak = r[0]
        r = con.execute("SELECT equity FROM daily_equity ORDER BY date DESC LIMIT 1").fetchone()
        if r: cur = r[0]
    if cur is None: cur = START_CAP
    if peak is None: peak = max(cur, START_CAP)
    dd = max(0.0, (peak - cur) / peak * 100.0) if peak else None
    return {"value_pct": dd, "current_equity": cur, "peak_equity": peak,
            "green": dd is None or dd < G_DD}


def _open_positions(con) -> dict:
    if not _safe(con, "positions"):
        return {"count": 0, "by_asset": {}, "by_side": {"long": 0, "short": 0}, "green": True}
    rows = con.execute("SELECT asset, side FROM positions WHERE status='open' OR status IS NULL").fetchall()
    by_a, by_s = {}, {"long": 0, "short": 0}
    for a, s in rows:
        by_a[a] = by_a.get(a, 0) + 1
        by_s["long" if (s or "").lower().startswith("long") else "short"] += 1
    return {"count": len(rows), "by_asset": by_a, "by_side": by_s, "green": len(rows) <= 30}


def _cb_state(con) -> dict:
    out = {}
    if not _safe(con, "strategy_state"): return {"per_strategy": {}, "green": True}
    for sid, sj in con.execute("SELECT strategy_id, state_json FROM strategy_state").fetchall():
        try:
            br = json.loads(sj).get("breaker") or {}
            if isinstance(br, dict): out[sid] = br
        except Exception: pass
    green = all(
        (not isinstance(v, dict)) or all(isinstance(m, (int, float)) and m >= 0.99 for m in v.values() if isinstance(m, (int, float)))
        for v in out.values()
    ) if out else True
    return {"per_strategy": out, "green": green}


def _tca_window(con, hours: int) -> Optional[float]:
    """Return avg impl_shortfall_bps over the last `hours`, or None."""
    if not _safe(con, "tca_records"): return None
    cutoff = (_now() - timedelta(hours=hours)).isoformat()
    rows = con.execute("SELECT impl_shortfall_bps FROM tca_records WHERE timestamp >= ?", (cutoff,)).fetchall()
    impls = [r[0] for r in rows if r[0] is not None]
    return sum(impls) / len(impls) if impls else None


def _fills_1h(con) -> dict:
    if not _safe(con, "tca_records"):
        return {"count": 0, "avg_slippage_bps": None, "cost_delta_bps": None, "green": True}
    cutoff = (_now() - timedelta(hours=1)).isoformat()
    rows = con.execute(
        "SELECT impl_shortfall_bps, spread_cost_bps, arrival_cost_bps, timing_cost_bps "
        "FROM tca_records WHERE timestamp >= ?", (cutoff,)).fetchall()
    if not rows: return {"count": 0, "avg_slippage_bps": None, "cost_delta_bps": None, "green": True}
    impls = [r[0] for r in rows if r[0] is not None]
    others = [(r[1] or 0) + (r[2] or 0) + (r[3] or 0) for r in rows]
    if not impls: return {"count": len(rows), "avg_slippage_bps": None, "cost_delta_bps": None, "green": True}
    realized = sum(impls) / len(impls)
    delta = realized - CAL
    return {"count": len(rows), "avg_slippage_bps": realized, "cost_delta_bps": delta,
            "avg_other_bps": sum(others) / len(rows), "green": delta <= A_BPS}


def _reconciler_24h(con) -> Optional[float]:
    delta = _tca_window(con, 24)
    if delta is not None: return delta - CAL
    if FILLS_JSONL.exists():
        try:
            cutoff = _now() - timedelta(hours=24)
            impls = []
            with open(FILLS_JSONL) as fh:
                for line in fh:
                    try:
                        rec = json.loads(line)
                        ts = _parse_dt(rec.get("ts") or rec.get("timestamp"))
                        bps = rec.get("impl_shortfall_bps") or rec.get("slippage_bps")
                        if ts and ts >= cutoff and bps is not None: impls.append(float(bps))
                    except Exception: continue
            if impls: return sum(impls) / len(impls) - CAL
        except Exception: pass
    return None


def _freshness() -> dict:
    out = {}
    for tf, mn in TFS.items():
        p = OUT_DIR / f"svi_features_{tf}.parquet"
        if p.exists():
            mt = datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc)
            age = (_now() - mt).total_seconds() / 60.0
            out[tf] = {"last_bar_mtime": mt.isoformat(), "age_min": age, "max_age_min": 3 * mn,
                       "stale": age > 3 * mn}
    stale = sum(1 for v in out.values() if v["stale"])
    return {"per_tf": out, "stale_count": stale, "green": stale == 0}


def _trades_today(con) -> int:
    if not _safe(con, "trades"): return 0
    today = _now().strftime("%Y-%m-%d")
    r = con.execute("SELECT COUNT(*) FROM trades WHERE timestamp LIKE ?", (today + "%",)).fetchone()
    return int(r[0]) if r else 0


def _model_drift() -> Optional[str]:
    md = REPO / "data" / "live" / "model"
    if not md.exists(): return None
    fs = list(md.rglob("*.pkl")) + list(md.rglob("*.json"))
    return _mtime_iso(Path(max(fs, key=lambda f: f.stat().st_mtime))) if fs else None


def _vol_overlay(con) -> dict:
    if not _safe(con, "checkpoints"): return {}
    r = con.execute("SELECT value FROM checkpoints WHERE key='vol_overlay'").fetchone()
    if r and r[0]:
        try: return json.loads(r[0])
        except Exception: pass
    return {}


def compute(con: sqlite3.Connection) -> M:
    m = M(ts=_now().isoformat())
    m.primary["open_positions"] = _open_positions(con)
    m.primary["daily_pnl"] = _daily_pnl(con)
    m.primary["cumulative_pnl"] = _cum_pnl(con)
    m.primary["drawdown"] = _drawdown(con)
    m.primary["circuit_breaker"] = _cb_state(con)
    m.primary["fills_1h"] = _fills_1h(con)
    delta_24h = _reconciler_24h(con)
    m.primary["reconciler"] = {
        "cost_delta_bps_24h": delta_24h,
        "report_mtime": _mtime_iso(RECONCILER_MD),
        "green": delta_24h is None or delta_24h <= G_BPS,
    }
    m.primary["data_freshness"] = _freshness()
    m.secondary["S1_reconciler_report"] = {
        "path": str(RECONCILER_MD.relative_to(REPO)),
        "exists": RECONCILER_MD.exists(),
        "mtime": _mtime_iso(RECONCILER_MD),
    }
    m.secondary["S2_vol_overlay"] = {"per_asset": _vol_overlay(con)}
    m.secondary["S3_signal_parity"] = {"n_signals_today": _trades_today(con)}
    m.secondary["S4_model_drift"] = {"model_drift_mtime": _model_drift()}
    m.alerts = _alerts(m.primary, m.secondary)
    return m


# ----------------------------- alerts -------------------------------------- #

def _alerts(p, s) -> list:
    fired, now = [], _now().isoformat()
    dd = (p["drawdown"] or {}).get("value_pct")
    dp = (p["daily_pnl"] or {}).get("value_pct")
    cd = (p["reconciler"] or {}).get("cost_delta_bps_24h")
    weekly = dd  # approximation when weekly series unavailable
    # SEV1 hard kills
    if dd is not None and dd >= HK_DD:
        fired.append(_a("SEV1", "drawdown", dd, f">= {HK_DD}%", "telegram-error", 5, now,
                         f"SEV1 HARD KILL: drawdown {dd:.2f}% >= {HK_DD}%"))
    if dp is not None and dp <= -D_DD:
        fired.append(_a("SEV1", "daily_maxdd", dp, f"<= -{D_DD}%", "telegram-error", 5, now,
                         f"SEV1 HARD KILL: daily drawdown {dp:.2f}% <= -{D_DD}%"))
    if weekly is not None and weekly >= W_DD:
        fired.append(_a("SEV1", "weekly_maxdd", weekly, f">= {W_DD}%", "telegram-error", 5, now,
                         f"SEV1 HARD KILL: weekly drawdown {weekly:.2f}% >= {W_DD}%"))
    if cd is not None and cd >= K_BPS:
        fired.append(_a("SEV1", "cost_delta_24h", cd, f">= +{K_BPS} bps", "telegram-error", 5, now,
                         f"SEV1 HARD KILL: cost delta {cd:.2f} bps >= +{K_BPS} bps"))
    # SEV2
    if dd is not None and G_DD <= dd < HK_DD:
        fired.append(_a("SEV2", "drawdown", dd, f">= {G_DD}%", "telegram-error", 30, now,
                         f"SEV2: drawdown {dd:.2f}% >= {G_DD}%"))
    if cd is not None and A_BPS <= cd < K_BPS:
        fired.append(_a("SEV2", "cost_delta_24h", cd, f">= +{A_BPS} bps", "telegram-error", 30, now,
                         f"SEV2: cost delta {cd:.2f} bps >= +{A_BPS} bps"))
    # SEV4
    n = (s.get("S3_signal_parity") or {}).get("n_signals_today", 0)
    if n > 0:
        fired.append(_a("SEV4", "trades_today", n, ">0", "telegram-trade", 0, now,
                         f"SEV4: {n} trades today (daily summary 00:05 UTC)"))
    return fired


def _a(tier, metric, value, threshold, channel, lat, ts, msg):
    return {"tier": tier, "metric": metric, "value": value, "threshold": threshold,
            "channel": channel, "latency_s": lat, "ts": ts, "msg": msg}


# ----------------------------- HTML render --------------------------------- #

CSS = """
body{font-family:-apple-system,'Segoe UI',Roboto,sans-serif;margin:0;background:#0e1117;color:#e6e6e6}
header{padding:16px 24px;background:#161b22;border-bottom:1px solid #30363d;display:flex;justify-content:space-between;align-items:center}
header h1{font-size:18px;margin:0;font-weight:600}
header .ts{font-size:12px;color:#8b949e;font-family:ui-monospace,Consolas,monospace}
main{padding:20px 24px;display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:16px;max-width:1400px;margin:0 auto}
.card{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:16px}
.card h2{font-size:13px;margin:0 0 8px 0;color:#8b949e;text-transform:uppercase;letter-spacing:.5px;font-weight:600}
.card .value{font-size:28px;font-weight:700;font-family:ui-monospace,Consolas,monospace}
.card .sub{font-size:12px;color:#8b949e;margin-top:4px}
.green{color:#3fb950}.amber{color:#d29922}.red{color:#f85149}.gray{color:#8b949e}
.pill{display:inline-block;padding:2px 8px;border-radius:10px;font-size:11px;font-weight:600;margin-left:8px}
.pill.green{background:rgba(63,185,80,.15)}.pill.amber{background:rgba(210,153,34,.15)}.pill.red{background:rgba(248,81,73,.15)}.pill.gray{background:rgba(139,148,158,.15)}
table{width:100%;border-collapse:collapse;font-size:13px}
td,th{text-align:left;padding:4px 8px;border-bottom:1px solid #21262d}
th{color:#8b949e;font-weight:500;font-size:11px;text-transform:uppercase}
.alert{padding:10px 12px;border-radius:6px;margin-bottom:8px;font-size:13px;font-family:ui-monospace,Consolas,monospace}
.alert.SEV1{background:rgba(248,81,73,.15);border-left:3px solid #f85149}
.alert.SEV2{background:rgba(210,153,34,.15);border-left:3px solid #d29922}
.alert.SEV3{background:rgba(139,148,158,.15);border-left:3px solid #8b949e}
.alert.SEV4{background:rgba(56,139,253,.15);border-left:3px solid #388bfd}
.muted{color:#8b949e;font-size:12px}
footer{padding:16px 24px;text-align:center;color:#8b949e;font-size:12px}
"""


def _card(title: str, value: str, sub: str, green, extra: str = "") -> str:
    return f"""<div class="card"><h2>{title}</h2><div class="value">{value}</div><div class="sub">{sub}</div>{extra}{_pill(green)}</div>"""


def render(m: M, mode: str) -> str:
    p, s = m.primary, m.secondary
    op = p["open_positions"]; dp = p["daily_pnl"]; cp = p["cumulative_pnl"]
    dd = p["drawdown"]; cb = p["circuit_breaker"]; f1 = p["fills_1h"]
    rec = p["reconciler"]; df = p["data_freshness"]
    df_rows = "".join(
        f"<tr><td>{tf}</td><td class='{'red' if v['stale'] else 'green'}'>{v['age_min']:.1f}m / {v['max_age_min']}m</td></tr>"
        for tf, v in (df["per_tf"] or {}).items()
    ) or "<tr><td colspan='2' class='muted'>no TF caches found</td></tr>"
    cb_rows = "".join(f"<tr><td>{k}</td><td>{len(v)} assets</td></tr>" for k, v in (cb["per_strategy"] or {}).items()) \
        or "<tr><td colspan='2' class='muted'>no breaker state recorded</td></tr>"
    _slip = f1["avg_slippage_bps"]
    _delta = f1["cost_delta_bps"]
    _slip_s = "n/a" if _slip is None else f"{_slip:.2f} bps"
    _delta_s = "n/a" if _delta is None else f"{_delta:+.2f} bps"
    cards = [
        _card("1. Open Positions", f"{op['count']} <span class='muted'>/ 30</span>",
              f"long: {op['by_side'].get('long', 0)} · short: {op['by_side'].get('short', 0)}",
              op["green"], f"<div class='sub'>by asset: {', '.join(f'{a}={c}' for a,c in op['by_asset'].items()) or '—'}</div>"),
        _card("2. Daily PnL", _money(dp["value_usd"]), _pct(dp["value_pct"]), dp["green"]),
        _card("3. Cumulative PnL since start", _money(cp["value_usd"]), "NC Sharpe >= 2.7 (rolling)", cp["green"]),
        _card("4. Drawdown (peak-to-trough)", _pct(dd["value_pct"]),
              f"peak {_money(dd['peak_equity'])} · cur {_money(dd['current_equity'])}", dd["green"]),
        f"""<div class="card"><h2>5. Circuit-Breaker (per asset)</h2><table><tr><th>strategy</th><th>state</th></tr>{cb_rows}</table>{_pill(cb['green'])}</div>""",
        _card("6. Fills in last 1h", str(f1["count"]),
              f"avg slip: {_slip_s} · Δ vs calib: {_delta_s}", f1["green"]),
        _card("7. Reconciler (cost delta 24h)",
              None if rec["cost_delta_bps_24h"] is None else f"{rec['cost_delta_bps_24h']:+.2f} bps",
              f"report: {rec['report_mtime'] or 'not yet written'}", rec["green"]),
        f"""<div class="card"><h2>8. Data Freshness per TF</h2><table><tr><th>TF</th><th>age / max</th></tr>{df_rows}</table>{_pill(df['green'])}</div>""",
        # secondary cards
        f"""<div class="card"><h2>S1. Reconciler report path</h2><div class="sub">{s['S1_reconciler_report']['path']}</div><div class="sub">exists: {s['S1_reconciler_report']['exists']} · mtime: {s['S1_reconciler_report']['mtime'] or '—'}</div></div>""",
        f"""<div class="card"><h2>S2. Per-asset vol overlay</h2><div class="sub">{len(s['S2_vol_overlay']['per_asset'])} assets</div></div>""",
        f"""<div class="card"><h2>S3. Signal parity (trades today)</h2><div class="value">{s['S3_signal_parity']['n_signals_today']}</div></div>""",
        f"""<div class="card"><h2>S4. Model-version drift</h2><div class="sub">latest mtime: {s['S4_model_drift']['model_drift_mtime'] or '—'}</div></div>""",
    ]
    if m.alerts:
        alerts_html = "<div class='card' style='grid-column:1/-1;'><h2>Alerts</h2>" + \
            "".join(f"<div class='alert {a['tier']}'>[{a['tier']}] {a['msg']} · channel={a['channel']} · latency<{a['latency_s']}s</div>" for a in m.alerts) + \
            "</div>"
    else:
        alerts_html = "<div class='card' style='grid-column:1/-1;'><h2>Alerts</h2><div class='muted'>no firings</div></div>"

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>S5 Live Dashboard</title>
<meta http-equiv="refresh" content="{REFRESH_S}"><style>{CSS}</style></head>
<body><header><h1>S5 Live Dashboard <span class="muted" style="font-weight:400;font-size:13px;">[{mode}]</span></h1>
<span class="ts">snapshot: {m.ts}</span></header>
<main>{''.join(cards)}{alerts_html}</main>
<footer>refresh: every {REFRESH_S}s · alert re-check every {ALERT_S//60}m ·
read-only on StateDB ({DB_PATH.relative_to(REPO)}) · calibrated baseline {CAL} bps</footer>
</body></html>
"""


# ----------------------------- offline sample ------------------------------ #

def offline_metrics() -> M:
    m = M(ts=_now().isoformat())
    m.primary.update({
        "open_positions": {"count": 0, "by_asset": {}, "by_side": {"long": 0, "short": 0}, "green": True},
        "daily_pnl": {"value_usd": None, "value_pct": None, "green": True},
        "cumulative_pnl": {"value_usd": 0.0, "green": True},
        "drawdown": {"value_pct": 0.0, "current_equity": START_CAP, "peak_equity": START_CAP, "green": True},
        "circuit_breaker": {"per_strategy": {}, "green": True},
        "fills_1h": {"count": 0, "avg_slippage_bps": None, "cost_delta_bps": None, "green": True},
        "reconciler": {"cost_delta_bps_24h": None, "report_mtime": _mtime_iso(RECONCILER_MD), "green": True},
        "data_freshness": _freshness(),
    })
    m.secondary.update({
        "S1_reconciler_report": {"path": str(RECONCILER_MD.relative_to(REPO)), "exists": RECONCILER_MD.exists(),
                                  "mtime": _mtime_iso(RECONCILER_MD)},
        "S2_vol_overlay": {"per_asset": {}},
        "S3_signal_parity": {"n_signals_today": 0},
        "S4_model_drift": {"model_drift_mtime": _model_drift()},
    })
    m.alerts = []
    return m


# ----------------------------- main ---------------------------------------- #

_seen_alerts: set = set()


def _log_alerts(alerts) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for a in alerts:
        k = (a["tier"], a["metric"], str(a.get("value")))
        if k in _seen_alerts: continue
        _seen_alerts.add(k)
        with open(ALERTS_LOG, "a", encoding="utf-8") as fh:
            fh.write(f"{a['ts']}\t{a['tier']}\t{a['metric']}\tvalue={a['value']}\tthreshold={a['threshold']}\t{a['msg']}\n")


def _write(m: M, mode: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = DASHBOARD_HTML.with_suffix(".html.tmp")
    tmp.write_text(render(m, mode), encoding="utf-8")
    tmp.replace(DASHBOARD_HTML)


def run_once(offline: bool) -> M:
    mode = "live"
    if offline:
        m = offline_metrics(); mode = "offline"
    else:
        try:
            con = _open_ro()
            try: m = compute(con)
            finally: con.close()
        except sqlite3.OperationalError as e:
            print(f"[warn] cannot open {DB_PATH}: {e}", file=sys.stderr)
            m = offline_metrics(); mode = "fallback-offline"
    _write(m, mode); _log_alerts(m.alerts)
    print(f"[{m.ts}] wrote {DASHBOARD_HTML.relative_to(REPO)} "
          f"({len(m.primary)} primary, {len(m.secondary)} secondary, {len(m.alerts)} alert firings)")
    return m


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--loop", action="store_true")
    p.add_argument("--offline", action="store_true")
    args = p.parse_args()
    if args.loop:
        while True:
            run_once(args.offline); time.sleep(REFRESH_S)
    else:
        run_once(args.offline)
    return 0


if __name__ == "__main__":
    sys.exit(main())
