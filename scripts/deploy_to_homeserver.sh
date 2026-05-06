#!/bin/bash
# Deploy ml-trading-system code + sidecars to homeserver.
# Validates locally, syncs, then validates remotely.
#
# Coverage post-cycle E (2026-05-01): includes the full V2-alignment
# stack from E.1 (P0), E.2 (P1 mainnet pre-reqs), and E.3 (P2). Every
# new file or behaviour-affecting modification needs to be on this list
# or it won't ship.
#
# Categories:
#   1. Backtester core (vol_arb/) — sizing, sidecars, pipeline.
#   2. Live engine (deribit_paper/) — config, execution, signal,
#      monitoring, risk.
#   3. Models + manifests (vol_arb/drl_hedging/, vol_model/output/).
#   4. Sidecars + features (vol_arb/output/, vol_model/output/).
#   5. Scripts (scripts/) — sidecar refresh, harness, beta extract.
#
# Usage: bash scripts/deploy_to_homeserver.sh
#        bash scripts/deploy_to_homeserver.sh --skip-tests       # skip pytest
#        bash scripts/deploy_to_homeserver.sh --skip-data        # skip parquet sidecars (large)

set -euo pipefail

HOST="monitor"
REMOTE_DIR="/hdd/ml-trading-system"

SKIP_TESTS=0
SKIP_DATA=0
for arg in "$@"; do
    case "$arg" in
        --skip-tests) SKIP_TESTS=1 ;;
        --skip-data)  SKIP_DATA=1 ;;
        *) echo "Unknown arg: $arg"; exit 1 ;;
    esac
done

# ── 1. Local validation ─────────────────────────────────────────────
if [ "$SKIP_TESTS" = "0" ]; then
    echo "=== Local validation ==="
    PYTHONPATH=. python -m pytest \
        deribit_paper/tests/ \
        tests/test_sidecars.py \
        tests/test_h15_alignment_harness.py \
        tests/test_refresh_live_sidecars.py \
        tests/test_backtester_noncomp.py \
        -x -q || {
        echo "ABORT: local tests failed"
        exit 1
    }
    echo ""
fi

# ── 2. File manifests ──────────────────────────────────────────────

# Source files (always sync — small, fast, frequently-changed).
SOURCE_FILES=(
    # Backtester core
    "vol_arb/backtester.py"
    "vol_arb/sizing.py"
    "vol_arb/sidecars.py"

    # DRL training module — referenced by SAC v4 pickle's _schedule
    # callable; needs to be importable on the homeserver even though
    # the live engine doesn't call it directly.
    "vol_arb/drl_hedging/train_sac.py"

    # Live engine — config + entry point
    "deribit_paper/__init__.py"
    "deribit_paper/__main__.py"
    "deribit_paper/cli.py"
    "deribit_paper/config.py"
    "deribit_paper/engine.py"

    # Live engine — execution layer
    "deribit_paper/execution/__init__.py"
    "deribit_paper/execution/amounts.py"
    "deribit_paper/execution/idempotency.py"
    "deribit_paper/execution/order_manager.py"
    "deribit_paper/execution/perp_hedger.py"
    "deribit_paper/execution/profit_scale.py"          # E.1.1 NEW
    "deribit_paper/execution/slippage_calc.py"         # C4
    "deribit_paper/execution/straddle_flow.py"

    # Live engine — signal / features
    "deribit_paper/signal/__init__.py"
    "deribit_paper/signal/predictor.py"
    "deribit_paper/signal/feature_assembler.py"
    "deribit_paper/signal/model_loader.py"
    "deribit_paper/signal/drl_hedger_live.py"
    "deribit_paper/features/__init__.py"
    "deribit_paper/features/svi_live.py"
    "deribit_paper/features/tsrv_live.py"
    "deribit_paper/features/straddle.py"
    "deribit_paper/features/vpin_live.py"

    # Live engine — Deribit client (REST + WS + auth)
    "deribit_paper/client/__init__.py"
    "deribit_paper/client/auth.py"
    "deribit_paper/client/errors.py"
    "deribit_paper/client/rest.py"
    "deribit_paper/client/ws.py"

    # Live engine — market data (bar clock, aggregator, buffers, subscriber)
    "deribit_paper/market_data/__init__.py"
    "deribit_paper/market_data/aggregator.py"
    "deribit_paper/market_data/bar_clock.py"
    "deribit_paper/market_data/buffer.py"
    "deribit_paper/market_data/subscriber.py"

    # Live engine — position state (equity, store, tracker)
    "deribit_paper/position/__init__.py"
    "deribit_paper/position/equity.py"
    "deribit_paper/position/store.py"
    "deribit_paper/position/tracker.py"

    # Live engine — risk + monitoring
    "deribit_paper/risk/__init__.py"
    "deribit_paper/risk/dry_run.py"
    "deribit_paper/risk/kill_switch.py"
    "deribit_paper/risk/limits.py"
    "deribit_paper/risk/rnd_monitor.py"
    "deribit_paper/monitoring/__init__.py"
    "deribit_paper/monitoring/harness_writer.py"        # C6 + E.1.3
    "deribit_paper/monitoring/logger.py"
    "deribit_paper/monitoring/metrics_sink.py"
    "deribit_paper/monitoring/telegram.py"

    # Live engine — engine-internal scripts
    "deribit_paper/scripts/__init__.py"
    "deribit_paper/scripts/validate_dry_run.py"
    "deribit_paper/scripts/run_engine.sh"               # H2: auto-restart wrapper

    # Scripts (sidecar refresh + harness + beta extract)
    "scripts/extract_aligner_betas.py"                 # C2
    "scripts/h15_alignment_harness.py"                  # I4
    "scripts/refresh_live_sidecars.py"                  # I6 + E.2.9
    "scripts/export_model_portable.py"
    "scripts/deploy_to_homeserver.sh"                  # self
)

# Model + manifest files (small, ship every deploy).
MODEL_FILES=(
    "vol_arb/drl_hedging/models/production_sac_v4_final.zip"
    "vol_arb/drl_hedging/models/production_sac_v4_final_vecnormalize.pkl"
    "vol_arb/drl_hedging/models/production_sac_v4_final.manifest.json"   # E.2.2 NEW
)

# Heavy data + sidecars (large; toggleable via --skip-data when only
# code changed). Order matters: smaller files first so a failed sync
# doesn't force a 6-MB resync of bar_execution_quality.
DATA_FILES=(
    "vol_model/output/aligner_betas_by_month.json"     # ~2KB
    "vol_arb/output/BTC_v2_kelly_mult_K1.parquet"      # ~1.3MB
    "vol_arb/output/bar_execution_quality.parquet"     # ~5.5MB
    "vol_arb/output/svi_surface_params_1h.parquet"     # ~22MB
    "vol_model/output/aligner_betas_by_month.json"      # dup is fine; idempotent
    "vol_model/output/BTCUSDT_pruned_svi_har_pd_predictions_aligned_bias.parquet"   # ~5.6MB
)

# ── 3. Sync ────────────────────────────────────────────────────────

sync_one() {
    local f="$1"
    if [ -f "$f" ]; then
        ssh "$HOST" "mkdir -p $REMOTE_DIR/$(dirname $f)" 2>/dev/null
        if scp -q "$f" "$HOST:$REMOTE_DIR/$f"; then
            echo "  OK $f"
        else
            echo "  FAIL $f"
            return 1
        fi
    else
        echo "  SKIP $f (not present locally)"
    fi
}

echo "=== Syncing source files ($HOST:$REMOTE_DIR) ==="
for f in "${SOURCE_FILES[@]}"; do
    sync_one "$f"
done

echo ""
echo "=== Syncing model files ==="
for f in "${MODEL_FILES[@]}"; do
    sync_one "$f"
done

if [ "$SKIP_DATA" = "0" ]; then
    echo ""
    echo "=== Syncing data + sidecars (large, --skip-data to skip) ==="
    for f in "${DATA_FILES[@]}"; do
        sync_one "$f"
    done
fi

# ── 4. Remote validation ────────────────────────────────────────────

echo ""
echo "=== Remote validation ==="
ssh "$HOST" "cd $REMOTE_DIR && PYTHONPATH=. python3 -c '
import faulthandler; faulthandler.enable()
import json, os

# E.1.1 — ProfitScaleEstimator
from deribit_paper.execution.profit_scale import (
    ProfitScaleEstimator, bsm_vega_per_vol_pt,
)
est = ProfitScaleEstimator()
ps = est.estimate(edge_vp=15.0, vega_per_vol_pt_usd=200.0, spot=60_000.0,
                   jump_fraction=0.0, slip_usd_per_btc=12.0, n_btc=0.5)
assert 0.5 < ps <= 1.0, f\"profit_scale {ps} out of band\"
print(f\"ProfitScaleEstimator: OK (cold-start ps={ps:.3f})\")

# E.1.5 — Kelly walk-forward + short-side enforcement
import pandas as pd
from vol_arb.sidecars import lookup_kelly_mult_at
ser = pd.Series([1.0, 1.5, 2.0],
                 index=pd.date_range(\"2026-01-01\", periods=3, freq=\"15min\", tz=\"UTC\"))
ts_after = pd.Timestamp(\"2027-01-01\", tz=\"UTC\")
assert lookup_kelly_mult_at(ser, ts_after, signal_direction=1) == 2.0
assert lookup_kelly_mult_at(ser, ts_after, signal_direction=-1) == 1.0
print(\"Kelly walk-forward + short defence: OK\")

# E.2.2 — DRL manifest
with open(\"vol_arb/drl_hedging/models/production_sac_v4_final.manifest.json\") as f:
    m = json.load(f)
assert m[\"bar_minutes\"] == 15
print(\"DRL manifest: OK (bar_minutes=\" + str(m[\"bar_minutes\"]) + \")\")

# DRL model file presence (we deliberately do NOT call SAC.load() here:
# it has been observed to segfault in some standalone-process contexts
# even when the running engine loads it cleanly. The engine startup
# self-test exercises the actual load + smoke prediction.)
sac_zip = \"vol_arb/drl_hedging/models/production_sac_v4_final.zip\"
sac_size = os.path.getsize(sac_zip)
assert sac_size > 1_000_000, \"SAC zip suspiciously small\"
print(\"SAC v4 zip: OK (\" + str(sac_size) + \" bytes; load-test deferred)\")

# V2 preset wires all E.1+E.2 invariants
from deribit_paper.config import PaperConfig
v2 = PaperConfig.with_v2_preset()
assert v2.use_profit_scaling is True              # E.1.1
assert v2.use_tiered_taker is True                # E.1.6
assert v2.entry_threshold_long == 12.0
assert v2.skip_long_if_term_slope_gt == 0.0
assert v2.target_margin_utilization == 0.15
print(\"V2 preset: OK\")

# CLI imports (incl --allow-stale-aligner flag)
from deribit_paper.cli import _parse_args, main as _cli_main
print(\"CLI: OK\")

print(\"All remote validations passed.\")
'"

echo ""
echo "=== Deploy complete ==="
echo ""
echo "To restart paper trader (mainnet quotes, simulated fills, auto-restart):"
echo "  ssh $HOST \"tmux kill-session -t deribit 2>/dev/null; \\"
echo "    tmux new-session -d -s deribit -c $REMOTE_DIR \\"
echo "    'PYTHONPATH=. bash deribit_paper/scripts/run_engine.sh \\"
echo "       --paper --config-preset v2 --duration-minutes 0 \\"
echo "       --log-level INFO \\"
echo "       2>&1 | tee deribit_paper/logs/paper_v2_\$(date +%Y%m%d_%H%M).log'\""
echo ""
echo "(run_engine.sh auto-restarts on crash with exponential backoff and"
echo "stops cleanly on rc=0 or rc=42. Use --allow-stale-aligner if the"
echo "aligner_betas_by_month.json is older than 1 month.)"
