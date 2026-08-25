#!/usr/bin/env python
"""Render HTML dashboard for logreg backtest results."""
import sys, os
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, "src"); sys.path.insert(0, ".")

import numpy as np
import pandas as pd
import json
from pathlib import Path

CAP = 200_000.0

def main():
    df = pd.read_csv("experiments/trade_logs/trade_log_yolo.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"] = df["timestamp"].dt.date

    # ── Core metrics ──
    final_eq = df["equity"].iloc[-1]
    total_pnl = final_eq - CAP
    n_trades = len(df)
    n_days = df["date"].nunique()
    win_rate = (df["pnl"] > 0).mean() * 100
    profit_factor = df.loc[df["pnl"] > 0, "pnl"].sum() / abs(df.loc[df["pnl"] < 0, "pnl"].sum())

    # Daily PnL
    daily = df.groupby("date").agg(
        pnl=("pnl", "sum"), n_trades=("pnl", "count"),
        equity=("equity", "last"),
    )
    daily.index = pd.to_datetime(daily.index)
    daily_rets = daily["pnl"] / daily["equity"].shift(1).fillna(CAP)
    sharpe = daily_rets.mean() / daily_rets.std() * np.sqrt(365.25)

    # Drawdown
    eq = daily["equity"].values
    peak = np.maximum.accumulate(eq)
    dd_pct = np.where(peak > 0, (peak - eq) / peak * 100, 0)
    max_dd = dd_pct.max()

    # CAGR
    years = n_days / 365.25
    cagr = ((final_eq / CAP) ** (1 / years) - 1) * 100

    # Monthly returns
    monthly = daily["pnl"].resample("MS").sum()
    monthly_eq_start = daily["equity"].resample("MS").first().shift(1).fillna(CAP)
    monthly_ret = (monthly / monthly_eq_start * 100)

    # By strategy
    strat_stats = df.groupby("strategy").agg(
        n=("pnl", "count"), total_pnl=("pnl", "sum"),
        win_rate=("pnl", lambda x: (x > 0).mean() * 100),
        avg_pnl=("pnl", "mean"), avg_size=("size_usd", "mean"),
    ).round(2)

    # By asset (top 20)
    asset_stats = df.groupby("asset").agg(
        n=("pnl", "count"), total_pnl=("pnl", "sum"),
        win_rate=("pnl", lambda x: (x > 0).mean() * 100),
    ).sort_values("total_pnl", ascending=False)

    # Yearly
    df["year"] = df["timestamp"].dt.year
    yearly = df.groupby("year").agg(
        n=("pnl", "count"), total_pnl=("pnl", "sum"),
        win_rate=("pnl", lambda x: (x > 0).mean() * 100),
        eq_end=("equity", "last"),
    )
    yearly["ret_pct"] = 0.0
    prev = CAP
    for yr in yearly.index:
        yearly.loc[yr, "ret_pct"] = (yearly.loc[yr, "eq_end"] / prev - 1) * 100
        prev = yearly.loc[yr, "eq_end"]

    # ── Build chart data ──
    eq_dates = daily.index.strftime("%Y-%m-%d").tolist()
    eq_values = daily["equity"].tolist()
    dd_values = dd_pct.tolist()
    daily_pnl_values = daily["pnl"].tolist()

    # Monthly heatmap data
    monthly_data = []
    for dt, ret in monthly_ret.items():
        monthly_data.append({"year": dt.year, "month": dt.month, "ret": round(ret, 1)})

    # Asset bar chart (top 20)
    top20 = asset_stats.head(20)
    asset_labels = top20.index.tolist()
    asset_pnl = top20["total_pnl"].tolist()

    bottom10 = asset_stats.tail(10)
    bottom_labels = bottom10.index.tolist()
    bottom_pnl = bottom10["total_pnl"].tolist()

    # Strategy pie
    strat_labels = strat_stats.index.tolist()
    strat_pnl = strat_stats["total_pnl"].tolist()
    strat_trades = strat_stats["n"].tolist()

    # Rolling Sharpe (60-day)
    rolling_sr = daily_rets.rolling(60).apply(lambda x: x.mean()/x.std()*np.sqrt(365.25) if x.std()>0 else 0)
    rs_dates = rolling_sr.dropna().index.strftime("%Y-%m-%d").tolist()
    rs_values = rolling_sr.dropna().tolist()

    # ── Render HTML ──
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>LogReg Backtest Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #0d1117; color: #c9d1d9; }}
  .header {{ background: linear-gradient(135deg, #161b22 0%, #0d1117 100%); padding: 24px 32px; border-bottom: 1px solid #30363d; }}
  .header h1 {{ font-size: 24px; color: #f0f6fc; }}
  .header .subtitle {{ color: #8b949e; font-size: 14px; margin-top: 4px; }}
  .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 16px; padding: 24px 32px; }}
  .metric {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 16px; }}
  .metric .label {{ font-size: 12px; color: #8b949e; text-transform: uppercase; letter-spacing: 0.5px; }}
  .metric .value {{ font-size: 28px; font-weight: 700; margin-top: 4px; }}
  .metric .value.green {{ color: #3fb950; }}
  .metric .value.red {{ color: #f85149; }}
  .metric .value.blue {{ color: #58a6ff; }}
  .charts {{ padding: 0 32px 32px; display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
  .chart-box {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 16px; }}
  .chart-box.full {{ grid-column: 1 / -1; }}
  .chart-box h3 {{ font-size: 14px; color: #f0f6fc; margin-bottom: 12px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  th {{ text-align: left; padding: 8px 12px; border-bottom: 1px solid #30363d; color: #8b949e; font-weight: 600; }}
  td {{ padding: 6px 12px; border-bottom: 1px solid #21262d; }}
  tr:hover {{ background: #1c2128; }}
  .pos {{ color: #3fb950; }} .neg {{ color: #f85149; }}
  canvas {{ max-height: 300px; }}
  .heatmap {{ display: grid; grid-template-columns: 60px repeat(12, 1fr); gap: 2px; font-size: 11px; text-align: center; }}
  .heatmap .cell {{ padding: 6px 2px; border-radius: 3px; }}
  .heatmap .header-cell {{ color: #8b949e; font-weight: 600; }}
</style>
</head>
<body>
<div class="header">
  <h1>LogReg Perp Trader — YOLO Backtest</h1>
  <div class="subtitle">$200K compounding | {df['timestamp'].min().strftime('%Y-%m-%d')} to {df['timestamp'].max().strftime('%Y-%m-%d')} | {n_trades:,} trades | {df['asset'].nunique()} assets</div>
</div>

<div class="metrics">
  <div class="metric"><div class="label">Final Equity</div><div class="value green">${final_eq:,.0f}</div></div>
  <div class="metric"><div class="label">Total PnL</div><div class="value green">${total_pnl:,.0f}</div></div>
  <div class="metric"><div class="label">CAGR</div><div class="value green">{cagr:.0f}%</div></div>
  <div class="metric"><div class="label">Sharpe Ratio</div><div class="value blue">{sharpe:.2f}</div></div>
  <div class="metric"><div class="label">Max Drawdown</div><div class="value red">{max_dd:.1f}%</div></div>
  <div class="metric"><div class="label">Win Rate</div><div class="value blue">{win_rate:.1f}%</div></div>
  <div class="metric"><div class="label">Profit Factor</div><div class="value blue">{profit_factor:.2f}</div></div>
  <div class="metric"><div class="label">Trades / Day</div><div class="value blue">{n_trades/n_days:.1f}</div></div>
</div>

<div class="charts">
  <div class="chart-box full"><h3>Equity Curve</h3><canvas id="eqChart"></canvas></div>
  <div class="chart-box full"><h3>Drawdown</h3><canvas id="ddChart"></canvas></div>
  <div class="chart-box"><h3>Daily PnL</h3><canvas id="dailyChart"></canvas></div>
  <div class="chart-box"><h3>Rolling 60-Day Sharpe</h3><canvas id="srChart"></canvas></div>
  <div class="chart-box"><h3>PnL by Strategy</h3><canvas id="stratChart"></canvas></div>
  <div class="chart-box"><h3>Top 20 Assets by PnL</h3><canvas id="assetChart"></canvas></div>

  <div class="chart-box full">
    <h3>Monthly Returns (%)</h3>
    <div class="heatmap" id="heatmap"></div>
  </div>

  <div class="chart-box">
    <h3>Yearly Performance</h3>
    <table>
      <tr><th>Year</th><th>Trades</th><th>PnL</th><th>Return</th><th>Win%</th></tr>
      {"".join(f'<tr><td>{yr}</td><td>{int(row.n):,}</td><td class="{"pos" if row.total_pnl>0 else "neg"}">${row.total_pnl:,.0f}</td><td class="{"pos" if row.ret_pct>0 else "neg"}">{row.ret_pct:+.0f}%</td><td>{row.win_rate:.0f}%</td></tr>' for yr, row in yearly.iterrows())}
    </table>
  </div>

  <div class="chart-box">
    <h3>Strategy Breakdown</h3>
    <table>
      <tr><th>Strategy</th><th>Trades</th><th>PnL</th><th>Win%</th><th>Avg PnL</th></tr>
      {"".join(f'<tr><td>{s}</td><td>{int(row.n):,}</td><td class="{"pos" if row.total_pnl>0 else "neg"}">${row.total_pnl:,.0f}</td><td>{row.win_rate:.0f}%</td><td>${row.avg_pnl:,.0f}</td></tr>' for s, row in strat_stats.iterrows())}
    </table>
  </div>

  <div class="chart-box full">
    <h3>Bottom 10 Assets</h3>
    <table>
      <tr><th>Asset</th><th>Trades</th><th>PnL</th><th>Win%</th></tr>
      {"".join(f'<tr><td>{a}</td><td>{int(row.n):,}</td><td class="neg">${row.total_pnl:,.0f}</td><td>{row.win_rate:.0f}%</td></tr>' for a, row in bottom10.iterrows())}
    </table>
  </div>
</div>

<script>
const eq_dates = {json.dumps(eq_dates)};
const eq_values = {json.dumps(eq_values)};
const dd_values = {json.dumps(dd_values)};
const daily_pnl = {json.dumps(daily_pnl_values)};
const rs_dates = {json.dumps(rs_dates)};
const rs_values = {json.dumps(rs_values)};
const strat_labels = {json.dumps(strat_labels)};
const strat_pnl = {json.dumps(strat_pnl)};
const asset_labels = {json.dumps(asset_labels)};
const asset_pnl = {json.dumps(asset_pnl)};
const monthly_data = {json.dumps(monthly_data)};

const gridColor = '#21262d';
const txtColor = '#8b949e';
const defaults = {{ responsive: true, maintainAspectRatio: false, plugins: {{ legend: {{ display: false }} }},
  scales: {{ x: {{ ticks: {{ color: txtColor, maxTicksLimit: 12 }}, grid: {{ color: gridColor }} }},
             y: {{ ticks: {{ color: txtColor }}, grid: {{ color: gridColor }} }} }} }};

// Equity
new Chart(document.getElementById('eqChart'), {{
  type: 'line', data: {{ labels: eq_dates, datasets: [{{ data: eq_values, borderColor: '#3fb950', borderWidth: 1.5, pointRadius: 0, fill: {{ target: 'origin', above: 'rgba(63,185,80,0.08)' }} }}] }},
  options: {{ ...defaults, scales: {{ ...defaults.scales, y: {{ ...defaults.scales.y, ticks: {{ ...defaults.scales.y.ticks, callback: v => '$'+v.toLocaleString() }} }} }} }}
}});

// Drawdown
new Chart(document.getElementById('ddChart'), {{
  type: 'line', data: {{ labels: eq_dates, datasets: [{{ data: dd_values.map(v => -v), borderColor: '#f85149', borderWidth: 1.5, pointRadius: 0, fill: {{ target: 'origin', below: 'rgba(248,81,73,0.15)' }} }}] }},
  options: {{ ...defaults, scales: {{ ...defaults.scales, y: {{ ...defaults.scales.y, ticks: {{ ...defaults.scales.y.ticks, callback: v => v.toFixed(1)+'%' }} }} }} }}
}});

// Daily PnL
new Chart(document.getElementById('dailyChart'), {{
  type: 'bar', data: {{ labels: eq_dates, datasets: [{{ data: daily_pnl, backgroundColor: daily_pnl.map(v => v >= 0 ? 'rgba(63,185,80,0.6)' : 'rgba(248,81,73,0.6)') }}] }},
  options: {{ ...defaults, scales: {{ ...defaults.scales, x: {{ display: false }} }} }}
}});

// Rolling Sharpe
new Chart(document.getElementById('srChart'), {{
  type: 'line', data: {{ labels: rs_dates, datasets: [{{ data: rs_values, borderColor: '#58a6ff', borderWidth: 1.5, pointRadius: 0 }}, {{ data: rs_dates.map(() => 0), borderColor: '#30363d', borderWidth: 1, borderDash: [4,4], pointRadius: 0 }}] }},
  options: defaults
}});

// Strategy PnL
new Chart(document.getElementById('stratChart'), {{
  type: 'bar', data: {{ labels: strat_labels, datasets: [{{ data: strat_pnl, backgroundColor: strat_pnl.map(v => v >= 0 ? '#3fb950' : '#f85149') }}] }},
  options: {{ ...defaults, indexAxis: 'y' }}
}});

// Asset PnL
new Chart(document.getElementById('assetChart'), {{
  type: 'bar', data: {{ labels: asset_labels, datasets: [{{ data: asset_pnl, backgroundColor: '#3fb950' }}] }},
  options: {{ ...defaults, indexAxis: 'y' }}
}});

// Monthly heatmap
const hm = document.getElementById('heatmap');
const months = ['','Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
months.forEach(m => {{ const d = document.createElement('div'); d.className = 'cell header-cell'; d.textContent = m; hm.appendChild(d); }});
const years = [...new Set(monthly_data.map(d => d.year))].sort();
years.forEach(yr => {{
  const yd = document.createElement('div'); yd.className = 'cell header-cell'; yd.textContent = yr; hm.appendChild(yd);
  for (let m = 1; m <= 12; m++) {{
    const entry = monthly_data.find(d => d.year === yr && d.month === m);
    const cell = document.createElement('div'); cell.className = 'cell';
    if (entry) {{
      const v = entry.ret;
      const intensity = Math.min(Math.abs(v) / 30, 1);
      cell.style.background = v >= 0 ? `rgba(63,185,80,${{0.15+intensity*0.6}})` : `rgba(248,81,73,${{0.15+intensity*0.6}})`;
      cell.textContent = v.toFixed(0) + '%';
      cell.style.color = '#f0f6fc';
    }} else {{ cell.style.background = '#0d1117'; }}
    hm.appendChild(cell);
  }}
}});
</script>
</body>
</html>"""

    out_path = Path("experiments/dashboard_yolo.html")
    out_path.write_text(html, encoding="utf-8")
    print(f"Dashboard saved to {out_path}")
    print(f"Open in browser: file:///{out_path.resolve()}")


if __name__ == "__main__":
    main()
