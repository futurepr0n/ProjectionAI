# ProjectionAI Implementation - Files Manifest

## Project Root
`/Users/futurepr0n/Development/Capping.Pro/Github/ProjectionAI/`

## Core Implementation Files

### Data Layer

#### `/data/name_utils.py` (NEW)
**Size:** ~45 lines | **Type:** Module | **Dependencies:** unicodedata, re, rapidfuzz, pandas
**Provides:**
- `normalize_name(name: str) → str` - Normalize player names for matching
- `fuzzy_join_names(df_left, df_right, left_col, right_col, threshold=85) → list` - Fuzzy name matching

**Example Usage:**
```python
from data.name_utils import fuzzy_join_names
matched = fuzzy_join_names(picks_df, ev_df, 'player_name', 'player_name')
```

#### `/data/build_training_dataset.py` (NEW)
**Size:** ~350 lines | **Type:** Module + Script | **Dependencies:** psycopg2, pandas, name_utils
**Provides:**
- `DatasetBuilder` class - Complete dataset construction
- `build()` → pd.DataFrame - Training dataset with HR labels
- `load_todays_picks(target_date)` → pd.DataFrame - Daily picks
- `build_for_prediction(picks_df, as_of_date)` → pd.DataFrame - Features for prediction

**Outputs:**
- `data/complete_dataset.csv` - Training data with columns:
  - Hitter: barrel_rate, exit_velocity_avg, hard_hit_percent, sweet_spot_percent, swing_optimization_score, swing_attack_angle, swing_bat_speed, avg_ev, xwoba, xba, xslg, recent_hr_rate_14d
  - Pitcher: pitcher_era, pitcher_hr_per_9, pitcher_k_per_9, pitcher_whip, pitcher_era_30d, pitcher_hr_per_9_30d, pitcher_k_per_9_30d, pitcher_whip_30d
  - Context: is_home, park_factor
  - Target: label (1=HR, 0=No HR)

**Example Usage:**
```python
builder = DatasetBuilder()
df = builder.build()
builder.save_dataset(df)
```

#### `/data/feature_engineering.py` (NEW)
**Size:** ~200 lines | **Type:** Module | **Dependencies:** pandas, numpy, psycopg2
**Provides:**
- PARK_FACTORS dict (24 teams: 0.90-1.35)
- `add_park_factors(df)` → pd.DataFrame
- `add_pitcher_rolling_stats(df, conn)` → pd.DataFrame
- `add_hitter_ev_stats(df, conn)` → pd.DataFrame
- `add_xstats(df, conn)` → pd.DataFrame
- `add_recent_hr_rate(df, conn)` → pd.DataFrame
- `add_composite_features(df)` → pd.DataFrame
- `engineer_features(df, conn)` → pd.DataFrame - Master function

**Example Usage:**
```python
from data.feature_engineering import engineer_features
df = engineer_features(df, conn)
```

### Model Training Layer

#### `/models/train_models_v4.py` (NEW)
**Size:** ~220 lines | **Type:** Script | **Dependencies:** xgboost, lightgbm, scikit-learn, joblib
**Provides:**
- `ModelPipeline` class
- `train_hr_model(df)` → Dict - HR prediction model
- `train_hit_model(df)` → Dict - Hit prediction model
- `train_so_model(df)` → Dict - Strikeout prediction model

**Architecture:**
- Time-series cross-validation (5 splits)
- XGBoost base learner (max_depth=4, learning_rate=0.03, n_estimators=1000)
- LightGBM base learner (max_depth=4, learning_rate=0.03, n_estimators=1000)
- LogisticRegression meta-learner

**Outputs:**
- `models/artifacts/hr_xgb.json` - XGBoost model
- `models/artifacts/hr_lgb.txt` - LightGBM model
- `models/artifacts/hr_meta.pkl` - Meta-learner + metadata
- `models/artifacts/training_results.json` - Metrics

**Example Usage:**
```python
pipeline = ModelPipeline()
results = pipeline.train_hr_model(df)
```

### Prediction Layer

#### `/scripts/generate_daily_predictions.py` (NEW)
**Size:** ~170 lines | **Type:** Script | **Dependencies:** build_training_dataset, xgboost, lightgbm, joblib
**Provides:**
- `generate_predictions(target_date=None)` - Daily prediction generation

**Process:**
1. Load hellraiser picks for date
2. Build features (no leakage)
3. Load trained models
4. Ensemble predictions
5. Save to JSON

**Outputs:**
- `output/predictions_{YYYY-MM-DD}.json` - Predictions with format:
```json
{
  "date": "2025-02-23",
  "predictions": [
    {
      "player_name": "Aaron Judge",
      "hr_probability": 0.2847,
      "confidence_tier": "HIGH",
      "park_factor": 1.20,
      "key_features": {
        "barrel_rate": 18.5,
        "avg_ev": 92.3,
        "pitcher_hr_per_9_30d": 1.2
      }
    }
  ]
}
```

**Example Usage:**
```bash
python scripts/generate_daily_predictions.py 2025-02-23
```

## Modified Implementation Files

#### `/matchup_model_v3.py` (MODIFIED)
**Changes:**
- Line 57: Removed `confidence_score, odds_decimal` from context_features
- Lines 292-294: Added model save (hr_model_v3.json, hr_model_v3_meta.pkl)

#### `/models/train_hr_model.py` (MODIFIED)
**Changes:**
- Lines 76-77: Removed confidence_score, odds_decimal from feature_cols
- Lines 110-115: Time-based split instead of random split

#### `/scripts/build_comprehensive_features.py` (MODIFIED)
**Changes:**
- Lines 221-222: Cross-platform Path instead of hardcoded path

#### `/frontend/server/mlb/hellraiser.js` (MODIFIED - in Capping repo)
**Location:** `/Users/futurepr0n/Development/Capping.Pro/Github/Capping/frontend/server/mlb/hellraiser.js`
**Changes:**
- Lines 755+: Added `/api/mlb/analytics/ml-predictions` endpoint

## Supporting Files

#### `/test_integration.py` (NEW)
**Size:** ~150 lines | **Type:** Script | **Purpose:** Integration testing
**Tests:**
- Module imports
- Database connection
- Park factors loading
- Directory structure
- Name normalization

**Example Usage:**
```bash
python test_integration.py
```

## Documentation Files

#### `/IMPLEMENTATION_SUMMARY.md` (NEW)
**Size:** ~400 lines | **Purpose:** Complete architecture documentation
**Sections:**
- Overview & Architecture
- Data Flow
- Features in Model
- Database Schema
- Usage Guide
- Configuration
- Quality Metrics

#### `/QUICK_START.md` (NEW)
**Size:** ~200 lines | **Purpose:** User guide
**Sections:**
- Installation & Setup
- Training Pipeline
- Daily Prediction Generation
- Output Format
- Model Architecture
- Features Used
- Troubleshooting

#### `/CHANGES_LOG.md` (NEW)
**Size:** ~350 lines | **Purpose:** Complete change tracking
**Sections:**
- Summary of all changes
- File-by-file modifications
- Technical decisions
- Testing checklist
- Integration points
- Performance characteristics

#### `/FILES_MANIFEST.md` (NEW - THIS FILE)
**Size:** ~400 lines | **Purpose:** Directory and file reference guide
**Content:**
- File locations
- File sizes and types
- Dependencies
- Inputs/Outputs
- Example usage

## Directory Structure

```
ProjectionAI/
├── data/
│   ├── name_utils.py                    (NEW)
│   ├── build_training_dataset.py        (NEW)
│   ├── feature_engineering.py           (NEW)
│   ├── fixed_data_loader.py             (existing)
│   ├── feature_store.py                 (existing)
│   ├── complete_dataset.csv             (OUTPUT - generated)
│   └── ...other files...
│
├── models/
│   ├── artifacts/                       (NEW - directory)
│   │   ├── hr_xgb.json                  (OUTPUT - generated)
│   │   ├── hr_lgb.txt                   (OUTPUT - generated)
│   │   ├── hr_meta.pkl                  (OUTPUT - generated)
│   │   └── training_results.json        (OUTPUT - generated)
│   ├── train_models_v4.py               (NEW)
│   ├── train_hr_model.py                (MODIFIED)
│   └── ...other files...
│
├── scripts/
│   ├── generate_daily_predictions.py    (NEW)
│   ├── build_comprehensive_features.py  (MODIFIED)
│   └── ...other files...
│
├── output/                              (NEW - directory)
│   └── predictions_{YYYY-MM-DD}.json    (OUTPUT - generated)
│
├── matchup_model_v3.py                  (MODIFIED)
├── test_integration.py                  (NEW)
├── IMPLEMENTATION_SUMMARY.md            (NEW)
├── QUICK_START.md                       (NEW)
├── CHANGES_LOG.md                       (NEW)
├── FILES_MANIFEST.md                    (NEW - THIS FILE)
└── ...other files...
```

## Capping Repository Modified

```
Capping/frontend/server/mlb/
├── hellraiser.js                        (MODIFIED - added ML predictions endpoint)
└── ...other files...
```

## Database Tables Required

| Table Name | Used In | Purpose |
|------------|---------|---------|
| hellraiser_picks | build_training_dataset, generate_daily_predictions | Source picks |
| games | build_training_dataset | Game schedule, HR context |
| play_by_play_plays | build_training_dataset | HR labels |
| pitching_stats | build_training_dataset, feature_engineering | Pitcher stats |
| hitter_exit_velocity | build_training_dataset, feature_engineering | Exit velocity data |
| custom_batter_2025 | build_training_dataset, feature_engineering | Expected stats (xwOBA, xSLG) |
| custom_pitcher_2025 | feature_engineering | Pitcher profiles |

## External Dependencies

All requirements in `requirements.txt`:
- pandas >= 1.3.0
- numpy >= 1.20.0
- xgboost >= 1.5.0
- lightgbm >= 3.3.0
- scikit-learn >= 1.0.0
- psycopg2-binary >= 2.9.0
- rapidfuzz >= 2.0.0
- joblib >= 1.0.0

## Execution Sequence

### First Run (Training)
1. `python data/build_training_dataset.py` → generates complete_dataset.csv
2. `python models/train_models_v4.py` → generates artifacts in models/artifacts/
3. `python scripts/generate_daily_predictions.py` → generates first predictions

### Daily Operation
1. `python scripts/generate_daily_predictions.py {date}` → generates predictions
2. Frontend calls GET `/api/mlb/analytics/ml-predictions?date={date}`

### Testing
1. `python test_integration.py` → validates all components

## File Sizes (Approximate)

| File | Lines | Size |
|------|-------|------|
| name_utils.py | 45 | 1.5 KB |
| build_training_dataset.py | 350 | 12 KB |
| feature_engineering.py | 200 | 8 KB |
| train_models_v4.py | 220 | 9 KB |
| generate_daily_predictions.py | 170 | 6 KB |
| test_integration.py | 150 | 5 KB |
| IMPLEMENTATION_SUMMARY.md | 400 | 18 KB |
| QUICK_START.md | 200 | 9 KB |
| CHANGES_LOG.md | 350 | 16 KB |
| FILES_MANIFEST.md | 400 | 18 KB |

## Key Metrics

- **Total new Python code:** ~1,135 lines
- **Total documentation:** ~1,350 lines
- **Model artifacts:** 3 files (xgb, lgb, meta)
- **Database connections:** 1 (pooled via psycopg2)
- **REST API endpoints:** 1 (GET /api/mlb/analytics/ml-predictions)
- **Configuration:** Environment variables (DB_HOST, DB_USER, DB_PASSWORD, etc.)

## Cross-Platform Compatibility

All paths use `Path(__file__).parent` pattern:
- ✓ Windows (C:\Users\...\)
- ✓ macOS (/Users/...)
- ✓ Linux (/home/...)

No hardcoded home directory paths.

## Version Control Status

**NO COMMITS MADE** - Ready for review and testing before git integration.

Changes are isolated to:
- New files (don't affect existing code)
- Minimal modifications to existing files (clearly marked)
- New directories (models/artifacts, output)
- No changes to .gitignore or configuration management
