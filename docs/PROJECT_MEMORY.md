# ProjectionAI Project Memory

Purpose: a short working reference for future coding passes in this repository. This file is based on the current codebase, not just older planning docs.

## What This Project Is

ProjectionAI is an MLB betting projection system centered on batter prop prediction. The current code supports at least three binary targets:

- `hr`: home run
- `hit`: at least one hit
- `so`: starting pitcher strikeout threshold probability (`3+`, `4+`, `5+`, `6+`)

The active stack is Python + PostgreSQL + Flask, with model artifacts stored on disk under `models/artifacts/`.

## Current Architecture

### 1. Data preparation

Primary file: [data/build_training_dataset.py](/Users/futurepr0n/Development/Capping.Pro/Github/ProjectionAI/data/build_training_dataset.py)

This is the most important data pipeline in the repo right now.

- Builds a training dataset from `play_by_play_plays` joined to `games`
- Creates one row per `(game_id, batter)`
- Derives labels:
  - `label` for HR
  - `label_hit` for hit
  - `label_so` for strikeout
- Enriches with:
  - hitter EV / barrel / sweet spot
  - batter xStats
  - recent 14-day rates
  - opponent pitching 30-day rolling stats
  - hitter vs pitcher pitch-type matchup features
  - prior batter-vs-pitcher history
  - park/travel features
  - placeholder weather defaults

Important newer behavior:

- the PBP batter spine now cleans action-text suffixes out of batter names
- `Unknown` placeholder batters are excluded from the hitter training spine
- batter-team assignment prefers opposing pitcher team context over raw `inning_half`

Inference helpers also live here:

- `load_todays_picks()`
- `build_for_prediction()`

Those methods enrich rows from `hellraiser_picks` for live scoring.

### 2. Feature utilities

Primary file: [data/feature_engineering.py](/Users/futurepr0n/Development/Capping.Pro/Github/ProjectionAI/data/feature_engineering.py)

Important responsibilities:

- hardcoded fallback park factors
- stadium location metadata
- travel fatigue features
- DB-backed enrichment helpers

This file mixes static reference data with DB queries. It is a utility layer, not a full feature store abstraction.

### 3. Name normalization

Primary file: [data/name_utils.py](/Users/futurepr0n/Development/Capping.Pro/Github/ProjectionAI/data/name_utils.py)

Supporting file: [data/migrate_player_names.py](/Users/futurepr0n/Development/Capping.Pro/Github/ProjectionAI/data/migrate_player_names.py)

The newer pipeline depends on canonical player-name mapping to join across inconsistent source systems.

### 4. Model training

Primary file: [models/train_models_v4.py](/Users/futurepr0n/Development/Capping.Pro/Github/ProjectionAI/models/train_models_v4.py)

This is the current hitter-model training path.

- Reads `data/complete_dataset.csv`
- Trains separate models for `hr`, `hit`, and legacy hitter-side `so`
- Uses XGBoost + LightGBM + logistic-regression meta learner
- Saves artifacts to `models/artifacts/`

Important note:

- the product `SO` path in the dashboard is no longer driven by this hitter-side `label_so` target
- the hitter-side `so` target is effectively legacy and now close to degenerate after dataset cleanup

Artifact pattern:

- `{target}_xgb.json`
- `{target}_lgb.txt`
- `{target}_meta.pkl`

Older training code still exists in [models/train.py](/Users/futurepr0n/Development/Capping.Pro/Github/ProjectionAI/models/train.py) and related files. Treat that path as legacy unless the user says otherwise.

### 5. Serving / dashboard

Primary file: [dashboards/app.py](/Users/futurepr0n/Development/Capping.Pro/Github/ProjectionAI/dashboards/app.py)

This is the main application entry point.

- Flask app
- loads ensemble artifacts for all three targets
- opens a PostgreSQL connection on startup
- loads `data/complete_dataset.csv` on startup
- loads `data/pitcher_strikeout_dataset.csv` for starter strikeout serving
- computes signal thresholds from model output percentiles
- exposes the live prediction/dashboard path through templates in `dashboards/templates/`

Important serving behavior:

- hitter rows are deduped before display/export
- dashboard results summary now separates hit rate from odds-aware ROI
- if odds are unavailable, the summary shows ROI as `N/A` instead of implying a betting return
- filter changes auto-refresh predictions; there is no longer a separate load button
- the historical stat cards are classification-aware
- clicking a prediction card opens a modal with:
  - a per-row prediction summary
  - explanation components
  - row-level XGBoost driver summaries
  - result breakdown when the outcome exists
- some explanation components are intentionally matchup-level and can repeat across players in the same game
- explanation formatting has been corrected for percentage-point fields such as `barrel_rate`

### 6. Batch prediction

Primary file: [scripts/generate_daily_predictions.py](/Users/futurepr0n/Development/Capping.Pro/Github/ProjectionAI/scripts/generate_daily_predictions.py)

This script now matches the app direction more closely:

- supports HR, Hit, and SO exports
- supports strikeout-threshold selection
- writes per-date JSON output to `output/`

## Likely Primary Workflow

For training:

1. Build dataset with [data/build_training_dataset.py](/Users/futurepr0n/Development/Capping.Pro/Github/ProjectionAI/data/build_training_dataset.py)
2. Save to `data/complete_dataset.csv`
3. Train ensembles with [models/train_models_v4.py](/Users/futurepr0n/Development/Capping.Pro/Github/ProjectionAI/models/train_models_v4.py)
4. Serve or reload via [dashboards/app.py](/Users/futurepr0n/Development/Capping.Pro/Github/ProjectionAI/dashboards/app.py)

For live inference:

1. Pull rows from `hellraiser_picks`
2. Enrich via `build_for_prediction()`
3. Score with saved artifacts
4. Surface via Flask dashboard or JSON output

## Runtime Dependencies

Primary dependency file: [requirements.txt](/Users/futurepr0n/Development/Capping.Pro/Github/ProjectionAI/requirements.txt)

Observed key dependencies:

- `Flask`
- `psycopg2-binary`
- `pandas`
- `numpy`
- `xgboost`
- `lightgbm`
- `scikit-learn`
- `joblib`

## Database Assumptions

The code frequently assumes a PostgreSQL database named `baseball_migration_test`.

Observed tables referenced in active code include:

- `play_by_play_plays`
- `games`
- `hitter_exit_velocity`
- `custom_batter_2025`
- `pitching_stats`
- `hellraiser_picks`
- `stadiums`
- `player_name_map`

The repo currently contains hardcoded default DB connection values in active files. That is an operational and security smell and should be treated as technical debt.

## Current State Notes

- The repository contains many planning / summary markdown files at the root.
- Some of those docs describe older files such as `matchup_model_v3.py` as the main path.
- The codebase has shifted toward:
  - `data/build_training_dataset.py`
  - `data/build_pitcher_strikeout_dataset.py`
  - `models/train_models_v4.py`
  - `models/train_pitcher_strikeout_models.py`
  - `dashboards/app.py`

When docs and code disagree, trust the code first.

## High-Signal Files

- [dashboards/app.py](/Users/futurepr0n/Development/Capping.Pro/Github/ProjectionAI/dashboards/app.py)
- [data/build_training_dataset.py](/Users/futurepr0n/Development/Capping.Pro/Github/ProjectionAI/data/build_training_dataset.py)
- [data/feature_engineering.py](/Users/futurepr0n/Development/Capping.Pro/Github/ProjectionAI/data/feature_engineering.py)
- [data/name_utils.py](/Users/futurepr0n/Development/Capping.Pro/Github/ProjectionAI/data/name_utils.py)
- [data/migrate_player_names.py](/Users/futurepr0n/Development/Capping.Pro/Github/ProjectionAI/data/migrate_player_names.py)
- [models/train_models_v4.py](/Users/futurepr0n/Development/Capping.Pro/Github/ProjectionAI/models/train_models_v4.py)
- [scripts/generate_daily_predictions.py](/Users/futurepr0n/Development/Capping.Pro/Github/ProjectionAI/scripts/generate_daily_predictions.py)

## Known Risks / Things To Verify Before Major Changes

- `dashboards/app.py` opens the DB and dataset at startup, so many changes can have immediate runtime side effects.
- Training and serving appear to depend on exact feature-name alignment from `meta.pkl`.
- `requirements.txt` has duplicate entries and may not be tightly curated.
- The worktree is already dirty; do not assume a clean baseline.
- Several top-level docs look stale relative to current code.
- Weather features are still placeholder defaults in the newer dataset builder.
- The dashboard runtime on a no-DB path can still differ from the live DB-backed app because it falls back to local CSV datasets.
- The hitter-side `label_so` path in `train_models_v4.py` should probably be retired or explicitly separated from the real starter strikeout product.

## Practical Starting Points For Future Work

If the task is about:

- model quality: start with [models/train_models_v4.py](/Users/futurepr0n/Development/Capping.Pro/Github/ProjectionAI/models/train_models_v4.py) and [data/build_training_dataset.py](/Users/futurepr0n/Development/Capping.Pro/Github/ProjectionAI/data/build_training_dataset.py)
- live prediction bugs: start with [dashboards/app.py](/Users/futurepr0n/Development/Capping.Pro/Github/ProjectionAI/dashboards/app.py)
- prediction explanation / modal issues: start with [dashboards/app.py](/Users/futurepr0n/Development/Capping.Pro/Github/ProjectionAI/dashboards/app.py) and [dashboard.html](/Users/futurepr0n/Development/Capping.Pro/Github/ProjectionAI/dashboards/templates/dashboard.html)
- missing player matches: start with [data/name_utils.py](/Users/futurepr0n/Development/Capping.Pro/Github/ProjectionAI/data/name_utils.py) and [data/migrate_player_names.py](/Users/futurepr0n/Development/Capping.Pro/Github/ProjectionAI/data/migrate_player_names.py)
- feature coverage gaps: start with [data/feature_engineering.py](/Users/futurepr0n/Development/Capping.Pro/Github/ProjectionAI/data/feature_engineering.py)
- daily automation: start with [scripts/generate_daily_predictions.py](/Users/futurepr0n/Development/Capping.Pro/Github/ProjectionAI/scripts/generate_daily_predictions.py)
- name/alias review: start with [data/migrate_player_names.py](/Users/futurepr0n/Development/Capping.Pro/Github/ProjectionAI/data/migrate_player_names.py) and the generated files under `output/`

## Minimal Command Reference

Typical local commands inferred from the repo:

```bash
python data/build_training_dataset.py
python models/train_models_v4.py
python scripts/generate_daily_predictions.py
python dashboards/app.py
```

## Bottom Line

The repo is in a transition state from older HR-only experiments toward a fuller multi-target pipeline. The clearest current backbone is:

`build_training_dataset.py -> train_models_v4.py -> dashboards/app.py`

For starter strikeouts, the parallel backbone is:

`build_pitcher_strikeout_dataset.py -> train_pitcher_strikeout_models.py -> dashboards/app.py`
