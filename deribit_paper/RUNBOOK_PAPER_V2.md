# Runbook — Live Engine V2-Parity Paper Run

I5 in plan `C:\Users\laure\.claude\plans\live-engine-h15-alignment-2026-04-26\`.

This runbook documents the standard 1-2 week testnet paper run with the
V2 production-parity stack. It assumes Phase A-D code has landed
(commits `32cd0ce` … `25bc2d4`) and that the operator has Deribit
testnet credentials in `.env`.

## Pre-flight

1. Verify Phase A-D code is on `master`:
   ```sh
   git log --oneline | head -12   # should show feat(live-engine) Phase A..I4 commits
   ```

2. Run the full live-engine + V2-parity test suite:
   ```sh
   cd C:/Claude/ml-trading-system
   PYTHONPATH=. python -m pytest \
     deribit_paper/tests/ \
     tests/test_sidecars.py \
     tests/test_h15_alignment_harness.py \
     tests/test_refresh_live_sidecars.py \
     -q
   ```
   Expect 100+ tests green. Pre-existing `test_limits.py` rot is
   tracked in a separate spawned task.

3. Confirm sidecars in place:
   ```sh
   ls -la \
     vol_model/output/aligner_betas_by_month.json \
     vol_arb/output/BTC_v2_kelly_mult_K1.parquet \
     vol_arb/output/bar_execution_quality.parquet
   ```

4. Verify the canonical V2 trade-log baseline (the harness compares
   against this):
   ```sh
   ls -la vol_arb/output/BTC_v2_trade_log.parquet   # ~84MB, 145041 rows
   ```

## Monthly aligner β\* refresh

The aligner β\* sidecar (`vol_model/output/aligner_betas_by_month.json`)
must be regenerated each calendar month. Engine startup (E.2.3) refuses
to start under `--config-preset v2` when the sidecar is ≥ 2 months stale,
unless `--allow-stale-aligner` is passed.

**When**: run on the 5th of each month (Binance Vision typically
publishes the prior month's aggTrades by day 3-5).

**Command**:
```sh
cd C:/Claude/ml-trading-system
PYTHONPATH=. python scripts/refresh_aligner_monthly.py
# Auto-picks the most recent complete month. Override with --month 2026-04.
```

**What it does** (orchestrates existing tooling — see the script header
for full details):

1. Phase 1: gates on data availability (BTCUSDT aggTrades, Deribit
   option trades ≥ 80% calendar coverage, DVOL extending into the
   target month). Refuses cleanly if any input is missing.
2. Phase 2-4: rebuilds features, runs single-month LightGBM walk-
   forward, runs the aligner.
3. Phase 5: re-extracts the β\* JSON to a tmp path.
4. Phase 6: sanity gate — refuses to ship when the new month's β\* is
   outside `[-0.5, +0.5]` or when the new JSON drops historical months.
5. Phase 7: atomic swap (tmp + `os.replace`) into the live sidecar
   path; backs up the old JSON beside it.

**After a successful refresh**: redeploy + restart the engine, and
**drop** the `--allow-stale-aligner` flag from the launch command:

```sh
bash scripts/deploy_to_homeserver.sh --skip-tests
ssh monitor "tmux kill-session -t deribit; \
    tmux new-session -d -s deribit -c /hdd/ml-trading-system \
    'PYTHONPATH=. python3 -m deribit_paper --run-paper \
       --config-preset v2 --venue testnet --duration-minutes 0 \
       --log-level INFO 2>&1 | tee deribit_paper/logs/paper_v2_\$(date +%Y%m%d_%H%M).log'"
```

**Cron template** (Linux):
```cron
# 06:30 UTC on the 5th of each month
30 6 5 * * cd /path/to/ml-trading-system && PYTHONPATH=. python3 scripts/refresh_aligner_monthly.py >> logs/aligner_refresh.log 2>&1
```

## Launch (testnet, dry-run)

```sh
cd C:/Claude/ml-trading-system
PYTHONPATH=. python -m deribit_paper \
    --run-paper \
    --config-preset v2 \
    --venue testnet \
    --duration-minutes 0 \
    --log-level INFO \
    > deribit_paper/logs/paper_v2_$(date +%Y%m%d).log 2>&1 &
```

* `--config-preset v2` activates the production parity stack:
  `entry_threshold_long=12`, `target_margin_utilization=0.15`, contango
  filter, β\*, Kelly K1, realistic_taker slippage. Logged at startup
  as `Applied --config-preset=v2 (production parity stack)`.
* `--duration-minutes 0` runs forever (until SIGTERM / Ctrl-C).
* Dry-run is the default; the engine talks to Deribit testnet but the
  REST client short-circuits orders to stub fills. To enable real
  testnet orders pass `--live` (requires the `GO LIVE` passphrase).

Output paths:

| path | role |
|---|---|
| `deribit_paper/state/paper.db` | SQLite trade + equity snapshots |
| `deribit_paper/state/live_metrics.parquet` | per-bar signal log |
| `deribit_paper/logs/deribit_paper.jsonl` | structured JSON log |
| `vol_arb/output/live_trades_harness.parquet` | per-trade harness rows (C6) |

## Monitor

Telegram heartbeat every 60 minutes (configured via `cfg.heartbeat_minutes`).

For ad-hoc status without interrupting the engine:
```sh
sqlite3 deribit_paper/state/paper.db \
    "SELECT COUNT(*), AVG(realized_pnl_usd) FROM trades WHERE status='completed';"

PYTHONPATH=. python -c "
import pandas as pd
df = pd.read_parquet('vol_arb/output/live_trades_harness.parquet')
print('rows:', len(df))
print('signal counts:', df['signal'].value_counts().to_dict())
print('mean kelly_mult on longs:',
      df.loc[df['signal']==1, 'kelly_mult'].mean())
"
```

## Mid-run alignment check

After ~1 day or whenever ≥10 trades have closed, run the harness in
preview mode:

```sh
PYTHONPATH=. python scripts/h15_alignment_harness.py
cat vol_arb/output/h15_alignment/H15_ALIGNMENT_REPORT.md
```

Use the report to spot early divergence in:

* `config_drift_signal_mismatch` — should be 0 (otherwise the contango
  filter or threshold isn't being respected).
* `pnl_med_abs_diff_usd` — first sanity number; growing systematically
  suggests slippage drift, strike-rounding edge, or DRL hedge cadence
  mismatch.

## End-of-window analysis

After 1-2 weeks (≥30 trades is the soft floor; extend up to 4 weeks
if testnet fills sparse):

1. **Stop the engine cleanly:**
   ```sh
   kill -TERM <pid>
   ```
   (or Ctrl-C if running interactively).

2. **Run the final alignment report:**
   ```sh
   PYTHONPATH=. python scripts/h15_alignment_harness.py
   ```

3. **Read the report:**
   ```sh
   cat vol_arb/output/h15_alignment/H15_ALIGNMENT_REPORT.md
   ```

4. **Compare against acceptance gates** (per plan §I5):

   | gate | threshold | status |
   |---|---|---|
   | Stage A median \|Δ NC PnL/trade\| / \|backtest NC PnL/trade\| | <5% (post strike-rounding) | review |
   | Stage A 90th percentile \|Δ\| | <25% | review |
   | β\* applied to every prediction | yes | logged at startup |
   | ≥1 trade with kelly_mult > 1.0 | yes | check via SQL |
   | 0 unhandled exceptions | yes | grep `deribit_paper.jsonl` |

5. **Document findings** in
   `memory/project_h15_alignment_2026_04_2X.md` and update
   `memory/MEMORY.md` with a 1-line index entry.

## Kill criteria (during the run)

| trigger | action |
|---|---|
| Engine cannot complete a 24h cycle | STOP; check `deribit_paper.jsonl` for the offending exception |
| `kill_switch_consecutive_errors` trips (3+ consecutive errors) | STOP; engine auto-halts |
| Daily DD > `cfg.kill_switch_daily_dd_pct` (default 2%) | STOP; engine auto-halts |
| WebSocket outage > `cfg.network_outage_seconds` (default 120s) | STOP; engine auto-halts |

If any of these fire, capture the last 200 lines of the JSON log and
the SQLite `kill_switch_state` table before restarting.

## Out of scope (for this run)

* Mainnet bring-up (separate cycle).
* Multi-asset (ETH/SOL/AVAX) — testnet validates BTC-only path.
* Order fragmentation by size (deferred; testnet liquidity is too
  artificial to validate).
* Real-time harness streaming (post-hoc only).
