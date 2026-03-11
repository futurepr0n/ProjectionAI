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
- per-day team filtering
- optional opponent inclusion for matchup review
- cleaned results summaries that separate hit rate from odds-aware ROI

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
  - faster team/date-based travel fatigue enrichment

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

- weather / park / environment improvements
- threshold-specific feature gating and calibration work
- sportsbook-optional odds ingestion for future Hit and SO seasons
- calibration and betting-oriented backtesting
- reviewed team-mismatch badges in the dashboard
