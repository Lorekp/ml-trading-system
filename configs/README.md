# Configurations

The maintained profiles in this directory are for research, paper trading, or
exchange demo/testnet use. No checked-in profile authorizes real capital.

- `default.yaml` contains shared research defaults.
- `live_paper.yaml` is a simulated-execution profile.
- `bybit_demo.yaml` targets Bybit's demo environment.
- `historical/` preserves obsolete parameter sets for provenance. Their names
  and values are not validation evidence and they must not be used for orders.
- `risk_*.yaml` are optional paper/testnet risk overlays.

Credentials come from environment variables or an ignored local `.env` file.
The former real-order MEXC profile was removed; create no replacement until a
strategy is independently validated and an explicit deployment review records
the authorization and safeguards.
