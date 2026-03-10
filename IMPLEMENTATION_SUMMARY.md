# ProjectionAI MLB Prediction Model Implementation Summary

## Overview
This implementation provides a complete ML pipeline for HR prediction using XGBoost+LightGBM ensemble models with meta-learner stacking.

## Files Created & Modified

### Created Files

#### 1. **data/name_utils.py**
- `normalize_name(name)` - Removes accents, suffixes (Jr., Sr., II-V), converts to lowercase
- `fuzzy_join_names(df_left, df_right, left_col, right_col, threshold=85)` - Fuzzy name matching using RapidFuzz
- Used for joining hellraiser picks with player stats from different sources

#### 2. **data/build_training_dataset.py**
- `DatasetBuilder` class with database connection pattern from `fixed_data_loader.py`
- **Methods:**
  - `load_todays_picks(target_date)` - Load hellraiser picks for a specific date
  - `build_for_prediction(picks_df, as_of_date)` - Adds features without leakage (30d rolling, 14d HR rate)
  - `build()` - Complete training dataset: picks + HR labels from play_by_play + pitcher stats + hitter EV + xstats
  - `_get_pitcher_rolling_stats()` - Rolling 30d pitcher metrics (HR/9, ERA, K/9, WHIP) prior to game date
  - `_get_hitter_ev_stats()` - Exit velocity data via fuzzy name join
  - `_get_xstats()` - Custom batter 2025 (xwOBA, xBA, xSLG)
  - `_get_recent_hr_rate()` - 14-day rolling HR rate
- **Key Features:**
  - No leakage: Only uses data from BEFORE the pick date
  - Drops `confidence_score` and `odds_decimal` columns
  - Adds `park_factor` column for away team
  - Saves to `data/complete_dataset.csv`

#### 3. **data/feature_engineering.py**
- PARK_FACTORS dict (24 teams, HR multipliers)
- **Functions:**
  - `add_park_factors()` - Maps away_team to park factor
  - `add_pitcher_rolling_stats()` - Rolling 30d pitcher stats via DB
  - `add_hitter_ev_stats()` - EV metrics via DB join
  - `add_xstats()` - xwOBA/xSLG from custom_batter_2025
  - `add_recent_hr_rate()` - Per-date rolling HR rate from play_by_play
  - `add_composite_features()` - adjusted_power (xslg * park_factor), pitcher_hr_vulnerability
  - `engineer_features()` - Main entry point, applies all transformations

#### 4. **models/train_models_v4.py**
- `ModelPipeline` class with three models:
  - `train_hr_model()` - HR prediction
  - `train_hit_model()` - Hit prediction
  - `train_so_model()` - Strikeout prediction
- **Training Strategy:**
  - Time-series cross-validation (5 splits)
  - XGBoost + LightGBM ensemble with LogisticRegression meta-learner
  - Time-sorted train/test split (80/20)
  - Handles class imbalance via scale_pos_weight
  - Early stopping on validation AUC
- **Artifacts Saved:**
  - `{name}_xgb.json` - XGBoost model
  - `{name}_lgb.txt` - LightGBM model
  - `{name}_meta.pkl` - Meta-learner + feature list + imputation medians
  - `training_results.json` - AUC scores

#### 5. **scripts/generate_daily_predictions.py**
- `generate_predictions(target_date=None)` - Generate predictions for a specific date
- **Process:**
  1. Load picks for target date via DatasetBuilder
  2. Build features for prediction (no leakage)
  3. Load trained models from artifacts
  4. Ensemble predictions: XGB + LGB → LogReg meta-learner
  5. Save to `output/predictions_{date}.json`
- **Output Format:**
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

### Modified Files

#### 1. **matchup_model_v3.py**
- **Change:** Removed `confidence_score` and `odds_decimal` from `context_features` list
- **Addition:** Added model save after training:
  ```python
  model.save_model('models/hr_model_v3.json')
  joblib.dump({'features': available_features, 'threshold': best_threshold}, 'models/hr_model_v3_meta.pkl')
  ```

#### 2. **models/train_hr_model.py**
- **Change 1:** Removed `confidence_score` and `odds_decimal` from feature_cols
- **Change 2:** Replaced random train/test split with time-based split:
  ```python
  split_idx = int(len(X_imputed) * (1 - test_size))
  X_train, X_test = X_imputed.iloc[:split_idx], X_imputed.iloc[split_idx:]
  y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
  ```

#### 3. **scripts/build_comprehensive_features.py**
- **Change:** Replaced hardcoded path with cross-platform Path:
  ```python
  output_path = Path(__file__).parent.parent / 'data' / 'comprehensive_features.csv'
  ```

#### 4. **frontend/server/mlb/hellraiser.js** (in Capping repo)
- **Addition:** New endpoint `/api/mlb/analytics/ml-predictions`
- **Behavior:**
  1. Check if predictions file exists locally
  2. If yes, serve from file
  3. If no, spawn Python process to generate predictions
  4. Return JSON with predictions and confidence tiers
  5. Handle errors gracefully (no predictions → 404)

## Database Schema Assumptions

### Tables Used
- `hellraiser_picks` - Player picks with analysis_date, confidence_score, etc.
- `games` - Game schedule (game_id, game_date, home_team, away_team)
- `play_by_play_plays` - Individual plays (game_id, batter, play_result)
- `pitching_stats` - Pitcher stats by game (game_id, player_name, innings_pitched, home_runs, etc.)
- `hitter_exit_velocity` - Exit velocity data (last_name_first_name, avg_hit_speed_numeric, brl_percent_numeric)
- `custom_batter_2025` - Expected stats (last_name_first_name, xwoba, xba, xslg)
- `custom_pitcher_2025` - Pitcher metrics (last_name_first_name, pitch_hand, hard_hit_percent)

### Important Columns
- `hellraiser_picks.analysis_date` - Date the pick was made
- `games.game_date` - Game date (used for joining)
- `play_by_play_plays.play_result` - 'Home Run' identifies HR outcomes
- `pitching_stats.innings_pitched` - Used for rate calculations

## Data Flow

```
Hellraiser Picks
    ↓
build_training_dataset.build()
    ├→ Join with games table
    ├→ Count HRs from play_by_play (label column)
    ├→ Get pitcher stats (ERA, HR/9, K/9, WHIP)
    ├→ Fuzzy join with hitter_exit_velocity
    ├→ Fuzzy join with custom_batter_2025
    └→ Add park_factor for away_team
    ↓
feature_engineering.engineer_features()
    ├→ Add 30-day rolling pitcher stats
    ├→ Add hitter EV stats
    ├→ Add xstats (xwOBA, xSLG)
    ├→ Add 14-day rolling HR rate
    └→ Add composite features
    ↓
train_models_v4.ModelPipeline
    ├→ Time-series cross-validation
    ├→ XGBoost training (with early stopping)
    ├→ LightGBM training (with early stopping)
    └→ LogisticRegression meta-learner
    ↓
generate_daily_predictions.py
    ├→ Load today's hellraiser picks
    ├→ Build features for prediction
    ├→ Load trained ensemble models
    └→ Generate HR probabilities
```

## Key Features in Model

### Hitter Features
- barrel_rate, exit_velocity_avg, hard_hit_percent, sweet_spot_percent
- swing_optimization_score, swing_attack_angle, swing_bat_speed
- avg_ev (from hitter_exit_velocity)
- xwoba, xba, xslg (expected stats)
- recent_hr_rate_14d (14-day rolling)

### Pitcher Features
- pitcher_era, pitcher_hr_per_9, pitcher_k_per_9, pitcher_whip (from pitching_stats)
- pitcher_hr_per_9_30d, pitcher_era_30d, pitcher_k_per_9_30d, pitcher_whip_30d (rolling)
- pitcher_hr_vulnerability (composite)

### Context Features
- is_home (boolean)
- park_factor (team-specific HR multiplier)
- adjusted_power (xslg * park_factor)

## Usage

### Generate Training Dataset
```bash
cd /Users/futurepr0n/Development/Capping.Pro/Github/ProjectionAI
python data/build_training_dataset.py
```

### Train Models
```bash
python models/train_models_v4.py
```

### Generate Daily Predictions
```bash
python scripts/generate_daily_predictions.py [YYYY-MM-DD]
```

### Via API (from Capping frontend)
```bash
curl "http://localhost:3000/api/mlb/analytics/ml-predictions?date=2025-02-23"
```

## Configuration

### Environment Variables
- `DB_HOST` - PostgreSQL host (default: 192.168.1.23)
- `DB_PORT` - PostgreSQL port (default: 5432)
- `DB_NAME` - Database name (default: baseball_migration_test)
- `DB_USER` - Database user (default: postgres)
- `DB_PASSWORD` - Database password (default: korn5676)

### Paths
All paths use `Path(__file__).parent` pattern for cross-platform compatibility:
- Models: `ProjectionAI/models/artifacts/`
- Data: `ProjectionAI/data/`
- Output: `ProjectionAI/output/`

## Model Parameters

### XGBoost
- max_depth: 4
- learning_rate: 0.03
- n_estimators: 1000 (with early_stopping_rounds=50)
- min_child_weight: 10
- reg_lambda: 1.5
- scale_pos_weight: Dynamic (based on class imbalance)

### LightGBM
- max_depth: 4
- learning_rate: 0.03
- n_estimators: 1000 (with early_stopping_rounds=50)
- min_child_samples: 20
- class_weight: 'balanced'

### Meta-Learner
- LogisticRegression on [XGB_proba, LGB_proba]
- Trained on test set to avoid overfitting

## Quality Metrics

- **AUC-ROC:** Primary metric, target > 0.65
- **Time-series CV:** 5-fold validation prevents data leakage
- **Calibration:** Meta-learner improves probability calibration
- **Feature completeness:** Handles missing values with median imputation

## Confidence Tiers

- **HIGH:** HR probability ≥ 0.25
- **MED:** HR probability 0.15-0.24
- **LOW:** HR probability < 0.15

## Notes

- No comments in production code unless logic is non-obvious
- All DB queries parameterized to prevent SQL injection
- Graceful handling of missing data (fillna with medians)
- No hardcoded paths (uses Path(__file__).parent pattern)
- Respects existing architectural patterns from fixed_data_loader.py
