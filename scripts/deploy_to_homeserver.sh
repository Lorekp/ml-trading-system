#!/bin/bash
# Deploy ml-trading-system code + models to homeserver.
# Validates locally, syncs, then validates remotely.
#
# Usage: bash scripts/deploy_to_homeserver.sh

set -euo pipefail

HOST="monitor"
REMOTE_DIR="/hdd/ml-trading-system"

echo "=== Local validation ==="
PYTHONPATH=. python -m pytest tests/test_backtester_noncomp.py -x -q || {
    echo "ABORT: local tests failed"
    exit 1
}

echo ""
echo "=== Syncing to $HOST:$REMOTE_DIR ==="
scp -r \
    vol_arb/backtester.py \
    vol_arb/sizing.py \
    vol_arb/drl_hedging/models/production_sac_v4_final.zip \
    deribit_paper/cli.py \
    deribit_paper/engine.py \
    deribit_paper/execution/order_manager.py \
    deribit_paper/signal/model_loader.py \
    "$HOST:$REMOTE_DIR/_deploy_staging/" 2>/dev/null || true

# Use scp for individual files (rsync may not be available)
for f in \
    vol_arb/backtester.py \
    vol_arb/sizing.py \
    vol_arb/drl_hedging/models/production_sac_v4_final.zip \
    deribit_paper/cli.py \
    deribit_paper/engine.py \
    deribit_paper/execution/order_manager.py \
    deribit_paper/signal/model_loader.py \
    scripts/export_model_portable.py \
    scripts/deploy_to_homeserver.sh
do
    if [ -f "$f" ]; then
        ssh "$HOST" "mkdir -p $REMOTE_DIR/$(dirname $f)" 2>/dev/null
        scp "$f" "$HOST:$REMOTE_DIR/$f" && echo "  ✓ $f"
    fi
done

echo ""
echo "=== Remote validation ==="
ssh "$HOST" "cd $REMOTE_DIR && PYTHONPATH=. python3 -c '
import faulthandler; faulthandler.enable()
# Model load test
from stable_baselines3 import SAC
m = SAC.load(\"vol_arb/drl_hedging/models/production_sac_v4_final.zip\", device=\"cpu\")
import numpy as np
a, _ = m.predict(np.zeros(8, dtype=np.float32), deterministic=True)
print(f\"Model:  OK (action={a})\")

# Order manager import test
from deribit_paper.execution.order_manager import OrderManager
print(\"Orders: OK\")

# CLI import test
from deribit_paper.cli import main
print(\"CLI:    OK\")
print(\"All remote validations passed.\")
'"

echo ""
echo "=== Deploy complete ==="
echo "To restart paper trader:"
echo "  ssh $HOST \"tmux kill-session -t deribit 2>/dev/null; tmux new-session -d -s deribit 'cd $REMOTE_DIR && PYTHONPATH=. python3 -m deribit_paper --run-paper --live --venue testnet'\""
