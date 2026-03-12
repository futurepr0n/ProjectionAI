# Production Runbook

This project should run in production with a DB-first data flow and a pinned file-based model package.

## Operating Model

Use the database for:
- raw source tables
- derived feature tables
- daily lineups
- historical weather
- audit and review tables

Use file artifacts for:
- trained model packages
- feature lists and medians
- training summaries
- serving manifest

The web app should:
- read current data from Postgres
- read the promoted model package from `models/artifacts/`
- never retrain inside the dashboard

## Daily Refresh

Run this when you want fresh lineups, weather, derived feature tables, and batch outputs without promoting new models.

```bash
./scripts/refresh_daily_data.sh --date 2025-09-28
```

What it does:
- fetches daily lineups into `daily_lineups`
- backfills historical weather for the requested date
- syncs derived feature snapshots into DB tables
- optionally generates a daily prediction export

## Full Retrain And Promote

Run this when you want to produce a new model package and replace what production serves.

```bash
./scripts/retrain_and_publish.sh --recent-lookback-games 20
```

What it does:
- rebuilds hitter and starter datasets
- retrains hitter and starter models
- refreshes derived DB snapshot tables
- rewrites `models/artifacts/serving_manifest.json`
- optionally commits the updated artifact package

Important:
- retraining is not part of normal dashboard usage
- only promote artifacts after reviewing holdout metrics and tier behavior

## Runtime Health Check

Use this after deploys or restarts.

```bash
./scripts/check_runtime_health.sh
```

What it checks:
- dashboard process can start
- model stats endpoint responds
- serving mode is pinned by manifest
- DB-backed datasets are available

## Promotion Checklist

Before changing the manifest to a new package:
- verify holdout metrics in `models/artifacts/training_results.json`
- verify strikeout metrics in `models/artifacts/pitcher_strikeout_training_results.json`
- review `holdout_tiers` to confirm `STRONG_BUY` materially outperforms the full pool
- verify dashboard loads and prediction cards render correctly
- verify daily batch export still works

## Current Production Serving

Production serving is currently pinned by:
- [models/artifacts/serving_manifest.json](/Users/futurepr0n/Development/Capping.Pro/Github/ProjectionAI/models/artifacts/serving_manifest.json)

That manifest should be treated as the source of truth for which artifact package the app serves.

## Deployment Notes

For the host server:
- keep the app code and `models/artifacts/` in the deployed checkout
- keep Postgres reachable from the app host
- run refresh scripts from the project root with the project virtualenv
- use a process manager in production instead of Flask debug mode

Recommended production process pattern:
- `systemd`
- `supervisord`
- or a container entrypoint

## Recommended Cadence

Daily:
- refresh lineups
- refresh weather
- sync derived DB features
- export predictions

As needed:
- rerun name-audit workflows
- review alias audit tables

Promotion only:
- rebuild datasets
- retrain models
- validate metrics
- update serving manifest
- restart app
