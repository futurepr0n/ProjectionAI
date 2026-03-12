# ProjectionAI

ProjectionAI is a locally run MLB prediction project built around three probability products:

- hitter home run probability
- hitter any-hit probability
- starting pitcher strikeout probability versus the opposing team

The current system uses tabular ensembles for training and a Flask dashboard for exploration, review, and batch export.

## Current Scope

The active code path is:

`data/build_training_dataset.py` -> `models/train_models_v4.py` -> `dashboards/app.py`

Starter strikeouts now have their own dataset and model flow:

`data/build_pitcher_strikeout_dataset.py` -> `models/train_pitcher_strikeout_models.py` -> `dashboards/app.py`

The dashboard supports:

- `HR` predictions
- `Hit` predictions
- `SO` predictions with a slider for `3+`, `4+`, `5+`, and `6+` strikeouts
- `HR/Hit` recent-form control using last `5`, `10`, `15`, or `20` games
- per-day team filtering
- optional opponent inclusion for matchup review
- starter override controls for hitter-side scenario analysis
- cleaned results summaries that separate hit rate from odds-aware ROI
- clickable card modals with per-prediction breakdowns
- row-level XGBoost driver summaries for individual cards
- result breakdowns inside the same modal when outcomes are available

## Main Files

```text
ProjectionAI/
├── dashboards/
│   ├── app.py
│   └── templates/
│       ├── dashboard.html
│       ├── analysis.html
│       └── master_list.html
├── data/
│   ├── build_training_dataset.py
│   ├── build_pitcher_strikeout_dataset.py
│   ├── build_player_team_history.py
│   ├── fetch_daily_lineups.py
│   ├── migrate_player_names.py
│   ├── name_utils.py
│   ├── complete_dataset.csv
│   └── pitcher_strikeout_dataset.csv
├── docs/
│   ├── IMPLEMENTATION_BACKLOG.md
│   ├── NAME_MATCHING_INVESTIGATION.md
│   ├── MODEL_DOCUMENTATION.md
│   └── PROJECT_MEMORY.md
├── models/
│   ├── train_models_v4.py
│   ├── train_pitcher_strikeout_models.py
│   └── artifacts/
└── scripts/
    └── generate_daily_predictions.py
```

## Modeling Summary

### HR / Hit

- built from hitter-game rows in `data/complete_dataset.csv`
- trained with XGBoost + LightGBM + logistic meta model
- evaluated with temporal holdout instead of random final split
- includes:
  - cleaned play-by-play batter spine with `Unknown`/action-text rows removed
  - hitter pitch-type matchup features
  - prior batter-vs-pitcher history with leakage-safe cumulative features
  - handedness and lineup-slot / projected-PA context
  - historical game-weather backfill from `historical_game_weather`
  - target-aware weather gating so `HR` and `Hit` do not consume the exact same weather feature bundle
  - faster team/date-based travel fatigue enrichment
- live serving now replaces the old 14-day recent-form fallback with last-N-games rates from `hitting_stats`

### Starter Strikeouts

- built from `daily_lineups`, `games`, `pitching_stats`, `play_by_play_plays`, `play_by_play_pitches`
- one row per starter-game
- supports labels for `3+`, `4+`, `5+`, and `6+` strikeouts
- includes:
  - recent starter form
  - recent opponent team strikeout form
  - starter pitch-mix / arsenal features
  - opponent team performance versus the starter's pitch types
  - recent workload / leash features such as pitch counts and batters faced
  - prior team-vs-starter matchup history

## Name Resolution / Data Integration

This project joins multiple baseball data sources that do not always use the same player naming conventions.

Current protections:

- centralized name normalization in `data/name_utils.py`
- audit and review workflow in `data/migrate_player_names.py`
- grouped pending-review exports under `output/`
- team-aware validation and official roster / transaction history support
- bulk alias review workflow for repeated play-by-play variants

Important docs:

- [docs/NAME_MATCHING_INVESTIGATION.md](/Users/futurepr0n/Development/Capping.Pro/Github/ProjectionAI/docs/NAME_MATCHING_INVESTIGATION.md)
- [docs/IMPLEMENTATION_BACKLOG.md](/Users/futurepr0n/Development/Capping.Pro/Github/ProjectionAI/docs/IMPLEMENTATION_BACKLOG.md)
- [docs/HITTER_RECENCY_OPTIMIZATION_PLAN.md](/Users/futurepr0n/Development/Capping.Pro/Github/ProjectionAI/docs/HITTER_RECENCY_OPTIMIZATION_PLAN.md)
- [docs/PRODUCTION_RUNBOOK.md](/Users/futurepr0n/Development/Capping.Pro/Github/ProjectionAI/docs/PRODUCTION_RUNBOOK.md)

## Running Locally

### Environment

This repo is intended to be run locally with the project virtualenv.

```bash
source venv/bin/activate
pip install -r requirements.txt
```

### Start the dashboard

```bash
source venv/bin/activate
python dashboards/app.py
```

Dashboard URL:

`http://127.0.0.1:5002`

### Rebuild datasets

Hitter dataset:

```bash
source venv/bin/activate
python data/build_training_dataset.py
```

Fetch / refresh daily lineups into `daily_lineups`:

```bash
source venv/bin/activate
python data/fetch_daily_lineups.py --date 2025-09-03
```

Backfill a date range:

```bash
source venv/bin/activate
python data/fetch_daily_lineups.py --start-date 2025-03-27 --end-date 2025-09-28
```

Historical weather backfill:

```bash
source venv/bin/activate
python data/backfill_historical_weather.py
```

Starter strikeout dataset:

```bash
source venv/bin/activate
python data/build_pitcher_strikeout_dataset.py
```

### Train models

HR / Hit ensembles:

```bash
source venv/bin/activate
python models/train_models_v4.py
```

Starter strikeout threshold models:

```bash
source venv/bin/activate
python models/train_pitcher_strikeout_models.py
```

Training is intended to be run from the terminal by a developer or agent. The dashboard serves saved artifacts and should not be treated as the training surface.

### Serving artifacts

- live serving now uses an explicit manifest at [models/artifacts/serving_manifest.json](/Users/futurepr0n/Development/Capping.Pro/Github/ProjectionAI/models/artifacts/serving_manifest.json)
- this pins the production artifact package per target instead of inferring serving behavior from whatever local artifact files happen to exist
- current production configuration is pinned to the committed `LightGBM` + `meta` packages
- local experimental `xgb.json` files should not affect runtime unless the manifest is intentionally changed

### Dashboard behavior notes

- prediction controls auto-refresh; there is no manual `Load Predictions` step
- model training is not triggered from the UI; retraining is a terminal workflow
- the historical stat cards reflect the currently selected classification scope
- `HR/Hit` live scoring uses a user-selectable last-N-games recent-form window
- hitter-side starter overrides rerun pitcher-dependent matchup features and rescore the slate
- clicking a prediction card opens an individual modal with:
  - summary and classification context
  - underlying feature breakdown
  - model-driver breakdown for that exact row
  - result breakdown when available
  - weather context where available for hitter props

### Generate batch predictions

All targets:

```bash
source venv/bin/activate
python scripts/generate_daily_predictions.py 2025-09-02
```

Starter strikeouts only with threshold selection:

```bash
source venv/bin/activate
python scripts/generate_daily_predictions.py 2025-09-02 --targets so --so-threshold 5
```

## Current Artifacts

Model artifacts are saved under `models/artifacts/`.

Current verified holdout META AUCs after the latest weather-integrated retrain:

- `HR`: `0.6850`
- `Hit`: `0.6935`
- legacy hitter-side `SO`: `0.5000` and should be treated as obsolete

Important starter strikeout artifact prefixes:

- `pitcher_so_3_plus`
- `pitcher_so_4_plus`
- `pitcher_so_5_plus`
- `pitcher_so_6_plus`

## Project Notes

- This project is locally run and currently assumes local database access.
- Hardcoded DB credentials are still used in parts of the codebase by design for the current workflow.
- The backlog for active implementation work is tracked in [docs/IMPLEMENTATION_BACKLOG.md](/Users/futurepr0n/Development/Capping.Pro/Github/ProjectionAI/docs/IMPLEMENTATION_BACKLOG.md).
- The repo contains older exploratory scripts and docs; the files listed above reflect the current working path.

## Next Priorities

See [docs/IMPLEMENTATION_BACKLOG.md](/Users/futurepr0n/Development/Capping.Pro/Github/ProjectionAI/docs/IMPLEMENTATION_BACKLOG.md), but the current highest-value open items are:

- target-specific weather refinement for HR vs Hit
- threshold-specific feature gating and calibration work
- sportsbook-optional odds ingestion for future Hit and SO seasons
- calibration and betting-oriented backtesting
- deeper park-direction / roof-status refinement for weather features
- deeper model-explanation quality beyond current XGBoost driver summaries
