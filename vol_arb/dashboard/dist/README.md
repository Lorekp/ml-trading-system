# BTC Vol-Arb Backtest Dashboard — Static Bundle

Pre-built React/Vite dashboard for the BTC volatility-arbitrage
strategy backtest. Self-contained: ships with both the JS/CSS bundle
AND the data file (`report.json`, ~15 MB) — no npm install, no
network calls beyond Google Fonts (the dashboard works offline if
fonts are cached).

**Privacy**: contains aggregate backtest performance numbers from the
production V2 preset (Stage 24 + Stage 29 in the pipeline doc). No
credentials, no API keys, no live trading data, no raw market data.
Suitable for a private trusted demo.

## Run it

**Mac / Linux**:
```bash
bash serve.sh
```

**Windows**:
```cmd
serve.bat
```

Both spin up a local Python HTTP server on `http://127.0.0.1:8765`.
Open that URL in any modern browser (Chrome, Firefox, Safari, Edge —
last 2 years). Use `Ctrl-C` to stop the server.

Custom port: `bash serve.sh 8080` / `serve.bat 8080`.

## Requirements

- **Python 3** (any 3.x) on PATH.
- That's it. The bundle is a static SPA; the server only serves
  files. No Node.js, no npm, no Python packages.

## Why not just open `index.html` directly?

The dashboard fetches `report.json` at runtime via `fetch("/report.json")`.
Browsers block `fetch` on `file://` URLs as a security policy
(CORS / same-origin). A local HTTP server resolves the issue with
no security tradeoff (it only listens on `127.0.0.1`, never
exposed to the network).

## What's in here

| File | Size | Purpose |
|---|---:|---|
| `index.html` | 11 KB | App shell + inline CSS theme |
| `assets/index-*.js` | 196 KB (63 KB gzip) | React app + visx charts |
| `report.json` | 15 MB | Pre-computed backtest stats (5,496 episodes, 36,480 1h bars) |
| `report.json.gz` | 4.6 MB | gzipped variant — not used by the bundle but kept for reference |
| `serve.sh` / `serve.bat` | — | Helper scripts |

## What the dashboard shows

Headline metrics + 6 tabs of detail:

- **Headline**: NC Sharpe 7.51, NC PnL $959.6K (959% on $100K), 5,496
  episodes — V2 production preset, post Kelly K1 adoption (Stage 29).
- **Edge**: bootstrap Sharpe CI, permutation null, walk-forward folds,
  rolling 90-day Sharpe + 180-day IC.
- **Attribution**: PnL decomposition (vega / theta / delta hedge),
  monthly returns calendar, decile tables.
- **Drawdown**: max-DD chronology, recovery times, stress events.
- **Cost sensitivity**: PnL grid across slip / fee perturbations.
- **Other**: distribution charts, scatter plots, raw episode browser.

## Provenance

Generated 2026-05-01 from `vol_arb/output/BTC_v2_trade_log.parquet`
(the canonical V2 + Kelly K1 + bias-aligned predictions trade log,
last regenerated 2026-04-21 per Stage 29 in `TSRV/PIPELINE_DOCUMENTATION.md`).

To regenerate from a fresh trade log:
```bash
cd vol_arb/dashboard
python etl/build_report.py --in ../output/BTC_v2_trade_log.parquet --out public/report.json
npm install   # one-time
npm run build
# new dist/ produced
```
