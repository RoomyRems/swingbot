# Migration from the original project

The original implementation remains available on `main`. The v2 branch is a
replacement, not an in-place compatibility layer.

## Credentials

The following local values can be copied from the old `.env`:

| Old variable | v2 |
|---|---|
| `ALPACA_API_KEY` | Reused; must be a paper-account key |
| `ALPACA_API_SECRET` | Reused; must be a paper-account secret |
| `ALPACA_PAPER_ENDPOINT` | May remain in `.env` but is ignored; the SDK uses `paper=True` |
| `FMP_API_KEY` | Not used |
| `MARKETAUX_API_KEY` | Not used |

Copy values locally; never commit `.env`. The repository includes only
`.env.example` placeholders.

## Configuration

Do not copy `config.yaml`. Its duplicated sections and large option surface are
part of what v2 removes. Start with `swingbot.toml`; unknown settings fail instead
of being silently ignored.

## Data and reports

Old caches and backtest CSVs are not compatible. Create a new snapshot with the
v2 `fetch` command. Snapshot and report directories are ignored by Git.

## Paper safety boundary

There is no `paper=false`, endpoint switch, or live-order command. Paper submission
requires both `--submit` and the literal `--confirm PAPER`. Existing positions and
orders are reconciled before new plans are submitted, and reconciliation errors
stop the run. Submission is disabled during the regular-session safety window and
whenever Alpaca's market clock reports that the market is open.

`burns-book-v1` submits a DAY buy stop-limit bracket: the entry trigger is one
tick above the completed hook bar, the entry limit caps gap/slippage exposure,
and the attached stop-market leg protects the active cycle low.
