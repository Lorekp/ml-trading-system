"""Deep Reinforcement Learning delta-hedging for vol-arb.

Replaces the fixed-band rule in ``vol_arb.backtester.simulate_discrete_hedging``
with a SAC-trained policy that operates inside a Gymnasium environment with:

- **Sigmoid-adjusted square-root law** market impact (pro-level, corrects
  SRL underestimation at participation rates < 0.5%).
- **SVI-interpolated BSM repricing** of the option leg at every step
  (no flat-IV assumption).
- **Risk-averse dual-Q objective** — second critic tracks E[R^2] so the
  actor can optimise E[R] - lambda * sqrt(Var[R]).

Public entry points:

- :class:`GymnasiumHedgingEnv` — the custom env
- :class:`RiskAverseSAC` — SAC subclass with second critic
- :class:`DRLHedger` — runtime adapter exposing the same signature as
  ``simulate_discrete_hedging`` so it drops into ``compute_pnl`` unchanged

See ``TSRV/PIPELINE_DOCUMENTATION.md`` Stage 16 for the full protocol.
"""

from vol_arb.drl_hedging.config import DRLHedgingConfig
from vol_arb.drl_hedging.impact import sigmoid_sqrt_impact, calibrate_kappa

__all__ = [
    "DRLHedgingConfig",
    "sigmoid_sqrt_impact",
    "calibrate_kappa",
]
