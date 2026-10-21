# ACRS

ACRS is a research and shadow-execution repository for the frozen
`RANGE_ONLY_NO_MOMENTUM` Spot strategy.

## Current state

- DEV freeze: `5C9-M2`, baseline exact and hash-verified.
- Binance Spot Demo integration: read-only market/account preflight and
  public market-data snapshots.
- Live-universe reconstruction and signal shadow are implemented.
- Execution gate: `CLOSED`; order endpoints are not implemented or called.
- Historical datasets and local Demo outputs remain outside version control.

The current live shadow is intentionally fail-closed. If the global regime is
not causally continuous or sufficiently confident, it produces `NO_SIGNAL` and
does not submit an order.

## Local validation

From the repository root:

```text
uv run pytest -q
uv run ruff check scripts/run_binance_demo_signal_shadow.py scripts/run_binance_demo_market_snapshot.py scripts/build_binance_demo_live_universe.py src/acrs/brokers tests/test_binance_demo_client.py
uv run python scripts/freeze_range_strategy_dev.py
```

## Binance Demo preflight

Credentials are supplied only through the local environment:

```text
export BINANCE_DEMO_API_KEY=...
export BINANCE_DEMO_API_SECRET=...
uv run python scripts/binance_demo_preflight.py
```

The preflight is read-only and does not call an order endpoint. Never commit
credentials, `.env` files, or generated files under `data/paper/`.

## Local shadow workflow

```text
uv run python scripts/build_binance_demo_live_universe.py
uv run python scripts/run_binance_demo_signal_shadow.py
```

Generated snapshots are local-only. GitHub synchronization and server
deployment happen only after local validation is complete and explicitly
reviewed.
