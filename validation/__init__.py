"""Cross-cutting validation gates (Phase-5, 2026-05-18).

Built after the Phase-5 post-mortem found the V2 vol-arb backtest's
~$1.6M headline was an artifact: PnL booked as `vega x d(iv_used)` on a
noisy non-tradeable estimate, never marked to real fills; and a
follow-on sizing artifact (fixed-N vs constant-vega) that flipped a
verdict. NONE of the repo's prior validation (shuffle test, bootstrap
CI, IV oracle, leakage audit) could catch either, because they all run
INSIDE the PnL model.

These gates check the things that machinery missed:
  - pnl_realism_gate    — model PnL vs fill-marked PnL must agree.
  - sizing_sanity_gate  — headline must not flip under sizing convention.
  - selection_bias_gate — traded subset must not be richer than the
                          candidate universe by an unexplained margin.

Use them on ANY new strategy before trusting a backtest headline.
"""
from validation.gates import (
    GateResult,
    pnl_realism_gate,
    selection_bias_gate,
    sizing_sanity_gate,
)

__all__ = [
    "GateResult",
    "pnl_realism_gate",
    "sizing_sanity_gate",
    "selection_bias_gate",
]
