# ProjectionAI — Complete Model Documentation

**Last Updated:** 2026-02-24
**System:** MLB Home Run / Hit / Strikeout Prediction Engine
**Architecture:** XGBoost + LightGBM ensemble with per-date rank-based signal classification

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Database Schema](#2-database-schema)
3. [Data Pipeline — Training](#3-data-pipeline--training)
4. [Feature Engineering](#4-feature-engineering)
5. [Model Architecture](#5-model-architecture)
6. [Training Methodology](#6-training-methodology)
7. [Inference Pipeline](#7-inference-pipeline)
8. [Signal Classification](#8-signal-classification)
9. [Composite Score](#9-composite-score)
10. [API Endpoints](#10-api-endpoints)
11. [Key Design Decisions](#11-key-design-decisions)
12. [File Manifest](#12-file-manifest)

---

## 1. System Overview

ProjectionAI generates per-player probability predictions for three MLB betting outcomes:

| Target | Label Column | Base Rate | What it predicts |
|--------|-------------|-----------|-----------------|
| **HR** | `label` | ~10% | Batter hits ≥ 1 home run |
| **Hit** | `label_hit` | ~55% | Batter gets ≥ 1 hit of any kind |
| **SO** | `label_so` | ~9% | Batter strikes out ≥ 1 time |

Each target has its own independently trained ensemble. Predictions rank players on a given date from best to worst matchup and classify them into five signal tiers relative to the day's slate.

### High-Level Data Flow

```
PostgreSQL DB (baseball_migration_test @ 192.168.1.23)
        │
        ▼
build_training_dataset.py
  ├─ Spine: play_by_play_plays aggregated per (game_id, batter)
  ├─ EV stats from hitter_exit_velocity
  ├─ xStats from custom_batter_2025
  ├─ 14-day rolling rates from play_by_play_plays
  ├─ 30-day pitcher rolling stats from pitching_stats + games
  └─ Park / travel / weather context
        │
        ▼
data/complete_dataset.csv  (40,438 rows, one per batter-game)
        │
        ▼
models/train_models_v4.py
  ├─ XGBoost (max_depth=4, lr=0.03, n_estimators=1000)
  ├─ LightGBM (max_depth=4, lr=0.03, n_estimators=1000)
  └─ LogisticRegression meta-learner (stacks XGB + LGB)
        │
        ▼
models/artifacts/
  ├─ {hr,hit,so}_xgb.json
  ├─ {hr,hit,so}_lgb.txt
  └─ {hr,hit,so}_meta.pkl  (meta-learner + feature list + train medians)
        │
        ▼
dashboards/app.py  (Flask API on port 5002)
  ├─ Load hitting_stats for date from DB (or fallback to CSV)
  ├─ Enrich rows with pitcher stats, park, weather defaults
  ├─ Run XGB.predict_proba() per target
  ├─ Sort by probability descending
  └─ Assign rank-based signals (top 10% = STRONG_BUY, etc.)
```

---

## 2. Database Schema

**Host:** `192.168.1.23:5432`
**Database:** `baseball_migration_test`

### Core Tables

#### `games`
| Column | Type | Description |
|--------|------|-------------|
| game_id | VARCHAR | Unique game identifier (e.g. ESPN game ID) |
| home_team | VARCHAR | Home team code (e.g. `NYY`) |
| away_team | VARCHAR | Away team code |
| home_pitcher | VARCHAR | Starting home pitcher name |
| away_pitcher | VARCHAR | Starting away pitcher name |
| game_date | DATE | Date of game |
| game_time | TIME | Start time |
| weather_temp | FLOAT | Temperature at game time |
| weather_wind | FLOAT | Wind speed at game time |
| home_lineup | JSONB | Home batting order |
| away_lineup | JSONB | Away batting order |
| home_score | INTEGER | Final home score |
| away_score | INTEGER | Final away score |

#### `play_by_play_plays`
| Column | Type | Description |
|--------|------|-------------|
| game_id | VARCHAR | FK → games.game_id |
| batter | VARCHAR | Full batter name |
| pitcher | VARCHAR | Full pitcher name |
| inning_half | VARCHAR | `Top X` (away bats) or `Bottom X` (home bats) |
| play_result | VARCHAR | `Home Run`, `Single`, `Double`, `Triple`, `Strikeout`, etc. |

**This is the primary training spine source.** Every row is one plate appearance. The dataset builder aggregates per (game_id, batter) to produce one row per player-game.

#### `hitting_stats`
| Column | Type | Description |
|--------|------|-------------|
| game_id | VARCHAR | FK → games |
| player_name | VARCHAR | Batter name |
| team | VARCHAR | Team code |
| home_runs | INTEGER | HRs in game |
| hits | INTEGER | Hits in game |
| strikeouts | INTEGER | Ks in game |
| at_bats | INTEGER | ABs |
| runs | INTEGER | Runs scored |
| rbi | INTEGER | RBIs |
| walks | INTEGER | BB |
| avg / obp / slg | FLOAT | Season-level slash line |

Used for inference (daily predictions) — provides the player roster for a given date along with season-level batting stats.

#### `pitching_stats`
| Column | Type | Description |
|--------|------|-------------|
| game_id | VARCHAR | FK → games |
| player_name | VARCHAR | Pitcher name |
| team | VARCHAR | Pitcher's team |
| earned_runs | INTEGER | ER in game |
| home_runs | INTEGER | HR allowed |
| strikeouts | INTEGER | K's recorded |
| walks | INTEGER | BB issued |
| hits | INTEGER | Hits allowed |
| innings_pitched | FLOAT | IP |

Used to compute rolling pitcher stats (ERA, HR/9, K/9, WHIP) over the 30 days preceding each game.

#### `hitter_exit_velocity`
| Column | Type | Description |
|--------|------|-------------|
| last_name_first_name | VARCHAR | Player key (e.g. `Judge, Aaron`) |
| avg_hit_speed_numeric | FLOAT | Average exit velocity (mph) → `avg_ev` |
| brl_percent_numeric | FLOAT | Barrel rate (%) → `barrel_rate` |
| anglesweetspotpercent_numeric | FLOAT | Sweet spot % → `sweet_spot_percent` |

Season-level Statcast data. Joined at training and inference time via canonical name matching.

#### `custom_batter_2025`
| Column | Type | Description |
|--------|------|-------------|
| last_name_first_name | VARCHAR | Player key |
| xwoba | FLOAT | Expected weighted on-base average |
| xba | FLOAT | Expected batting average |
| xslg | FLOAT | Expected slugging percentage |
| barrel_batted_rate | FLOAT | Barrel % (alternate source) |
| sweet_spot_percent | FLOAT | Sweet spot % (alternate source) |
| pa | INTEGER | Plate appearances (sample size) |

Season-level expected stats. Used as quality-of-contact features.

#### `hellraiser_picks`
| Column | Type | Description |
|--------|------|-------------|
| analysis_date | DATE | Pick date |
| player_name | VARCHAR | Player |
| team | VARCHAR | Team code |
| pitcher_name | VARCHAR | Opposing pitcher |
| is_home | BOOLEAN | Home/away |
| confidence_score | FLOAT | Hellraiser system score (0–100) |
| odds_decimal | FLOAT | Bookmaker decimal odds |
| barrel_rate | FLOAT | From Hellraiser data |
| exit_velocity_avg | FLOAT | From Hellraiser data |
| hard_hit_percent | FLOAT | From Hellraiser data |
| sweet_spot_percent | FLOAT | From Hellraiser data |
| swing_optimization_score | FLOAT | Swing quality composite |
| swing_attack_angle | FLOAT | Swing plane metric |
| swing_bat_speed | FLOAT | Bat speed (mph) |

External signal source. Used only at inference — **not** in training data to avoid leakage.

---

## 3. Data Pipeline — Training

**Script:** `data/build_training_dataset.py`
**Output:** `data/complete_dataset.csv` (~40,438 rows)

### Step 1: Build Spine

One row per (game_id, batter) aggregated from `play_by_play_plays`:

```sql
SELECT
    pp.game_id,
    g.game_date,
    g.home_team,
    g.away_team,
    pp.batter AS player_name,
    -- Inning half determines home/away: Top = away bats, Bottom = home bats
    CASE WHEN MAX(pp.inning_half) ILIKE 'Bottom%' THEN g.home_team
         ELSE g.away_team END AS team,
    CASE WHEN MAX(pp.inning_half) ILIKE 'Bottom%' THEN true
         ELSE false END AS is_home,
    -- Primary pitcher faced (modal value across all PAs)
    MODE() WITHIN GROUP (ORDER BY pp.pitcher) AS pitcher_name,
    -- Outcome counts → binary labels
    COUNT(CASE WHEN pp.play_result = 'Home Run' THEN 1 END) AS hr_count,
    COUNT(CASE WHEN pp.play_result IN ('Single','Double','Triple','Home Run') THEN 1 END) AS hit_count,
    COUNT(CASE WHEN pp.play_result = 'Strikeout' THEN 1 END) AS so_count
FROM play_by_play_plays pp
JOIN games g ON pp.game_id = g.game_id
GROUP BY pp.game_id, pp.batter, g.game_date, g.home_team, g.away_team
```

Labels derived:
- `label` = 1 if hr_count > 0
- `label_hit` = 1 if hit_count > 0
- `label_so` = 1 if so_count > 0

### Step 2: Name Canonicalization

All player names are normalized through `name_utils.normalize_to_canonical()`:
1. Unicode decomposition (strips accents: é → e)
2. Remove suffixes: Jr., Sr., II, III, IV, V
3. Handle `Lastname, Firstname` → `Firstname Lastname`
4. Query `player_name_map` table for exact or fuzzy match
5. Fallback: normalized form as-is

This unifies name formats across PBP, EV tables, and xStats tables.

### Step 3: Attach EV Stats (Season-Level)

From `hitter_exit_velocity`, joined via canonical name:

| Feature | Source Column | Description |
|---------|--------------|-------------|
| `avg_ev` | avg_hit_speed_numeric | Average exit velocity (mph) |
| `barrel_rate` | brl_percent_numeric | Barrel rate (% of batted balls) |
| `sweet_spot_percent` | anglesweetspotpercent_numeric | Launch angle 8–32° rate |

Match rate: ~80–90% of training rows.

### Step 4: Attach xStats (Season-Level)

From `custom_batter_2025`, joined via canonical name:

| Feature | Description |
|---------|-------------|
| `xwoba` | Expected wOBA (overall quality of contact) |
| `xba` | Expected batting average |
| `xslg` | Expected slugging percentage |

Match rate: ~85–95%.

### Step 5: Rolling 14-Day Rates (Per Date, No Leakage)

For every unique game_date in the spine, compute batter outcome rates from the 14 days **before** that date:

```sql
WHERE g.game_date >= {game_date} - INTERVAL '14 days'
  AND g.game_date <  {game_date}
```

Features produced:
- `recent_hr_rate_14d` — fraction of PAs resulting in HR in last 14 days
- `recent_hit_rate_14d` — fraction resulting in any hit
- `recent_so_rate_14d` — fraction resulting in strikeout

Strictly excludes the current game date to prevent leakage.

### Step 6: Rolling 30-Day Pitcher Stats (Per Game, No Leakage)

For each (game_id, opposing_team) pair, computes the pitching staff's aggregate stats from the 30 days before the game:

```sql
WHERE ps.team = {opp_team}
  AND g.game_date >= {game_date} - INTERVAL '30 days'
  AND g.game_date <  {game_date}
```

Features produced:
- `pitcher_era_30d` — ERA over last 30 days
- `pitcher_hr_per_9_30d` — HR allowed per 9 innings
- `pitcher_k_per_9_30d` — strikeouts per 9 innings
- `pitcher_whip_30d` — (hits + walks) / innings pitched

Keyed on (game_id, opp_team) to avoid cartesian product ambiguity.

### Step 7: Context Features

Added with league-average defaults when DB data is unavailable:

- **Park factor** — HR likelihood multiplier per home team (see full table below)
- **Travel fatigue** — Haversine distance from last game + timezone changes → composite 0–100 score
- **Weather** — Defaults set to 0 wind, 70°F, 0 precip. Historical weather via Open-Meteo API is available but defaults are used for training to avoid API dependency.

---

## 4. Feature Engineering

### Complete Feature List (Training)

All features are numeric. Non-numeric columns and label columns are excluded automatically.

**Explicitly excluded from training (to prevent leakage):**
```
label, label_hit, label_so,
game_date, player_name, game_id, confidence_score, odds_decimal, pick_id,
hr_count, hit_count, batter_so, so_count, batter_hit,
avg_ev, barrel_rate.1, sweet_spot_percent.1, player_name.1, player_name.2
```

**Active model features (all numeric columns remaining):**

| Feature | Category | Description |
|---------|----------|-------------|
| `is_home` | Context | 1 = batting at home |
| `pa_count` | Volume | Total plate appearances in game |
| `avg_ev` | Statcast | Average exit velocity (mph), ~85–95 range |
| `barrel_rate` | Statcast | Barrel rate %, higher = more optimal contact |
| `sweet_spot_percent` | Statcast | Launch angle 8–32° rate |
| `xwoba` | Expected | Expected weighted OBA, ~.280–.420 for regulars |
| `xba` | Expected | Expected batting average |
| `xslg` | Expected | Expected slugging %, proxy for power |
| `recent_hr_rate_14d` | Form | HR rate over last 14 days |
| `recent_hit_rate_14d` | Form | Hit rate over last 14 days |
| `recent_so_rate_14d` | Form | Strikeout rate over last 14 days |
| `pitcher_era_30d` | Matchup | Opposing staff ERA over last 30 days |
| `pitcher_hr_per_9_30d` | Matchup | Opposing staff HR/9 over last 30 days |
| `pitcher_k_per_9_30d` | Matchup | Opposing staff K/9 over last 30 days |
| `pitcher_whip_30d` | Matchup | Opposing staff WHIP over last 30 days |
| `park_factor` | Environment | HR likelihood multiplier per stadium |
| `travel_distance_miles` | Fatigue | Miles traveled since last game |
| `timezone_changes` | Fatigue | Timezone hours shifted since last game |
| `travel_fatigue_score` | Fatigue | Composite: (dist/3000×25) + (tz×10), clipped 0–100 |
| `wind_speed_mph` | Weather | Wind speed at game time |
| `temp_f` | Weather | Temperature (°F) |
| `precip_prob` | Weather | Precipitation probability |
| `wind_out_factor` | Weather | 0.95 if wind > 10 mph else 1.0 |
| `adjusted_power` | Composite | xslg × park_factor |
| `pitcher_hr_vulnerability` | Composite | mean(pitcher_hr_per_9_30d, pitcher_era_30d) |

### Park Factor Table (HR Multiplier)

| Factor | Teams |
|--------|-------|
| 1.35 | COL (Coors Field) |
| 1.20 | NYY |
| 1.15 | BOS, CIN |
| 1.10 | PHI, HOU |
| 1.05 | BAL, TEX, ARI |
| 1.00 | ATL, CHC, LAD, MIL, SDP, WSN |
| 0.95 | CLE, DET, KCR, MIN |
| 0.90 | OAK, MIA, PIT, SEA, SFG, TBR |

Source: hardcoded in `data/feature_engineering.py`. Can be overridden from a `stadiums` DB table if it contains ≥ 20 entries.

### Travel Fatigue Calculation

```python
travel_fatigue_score = (
    (haversine_distance(prev_stadium, current_stadium) / 3000 * 25) +
    (abs(timezone_offset_prev - timezone_offset_current) * 10)
).clip(0, 100)
```

Stadium coordinates for all 30 MLB teams are hardcoded in `STADIUM_LOCATIONS` dict (lat/lon pairs).

### Dome Stadiums (No Weather Impact)

Tropicana Field, Rogers Centre, Chase Field, Minute Maid Park, American Family Field, Globe Life Field, T-Mobile Park, loanDepot Park.

---

## 5. Model Architecture

**Script:** `models/train_models_v4.py`
**Class:** `ModelPipeline`

Three independent pipelines — one per target (HR, Hit, SO). Each produces the same architecture.

### Layer 1: XGBoost Classifier

```python
XGBClassifier(
    max_depth=4,           # Shallow trees to limit overfitting
    learning_rate=0.03,    # Conservative step size
    n_estimators=1000,     # Max trees (early stopping typically halts earlier)
    min_child_weight=10,   # Require ≥10 samples at each leaf
    reg_lambda=1.5,        # L2 regularization
    scale_pos_weight=spw,  # Dynamic: negatives/positives (e.g. ~9 for HR)
    eval_metric='auc',
    early_stopping_rounds=50,
    random_state=42,
    n_jobs=-1
)
```

**scale_pos_weight** is computed per fold/split from actual class ratios:
- HR: ~90 negatives : 10 positives → spw ≈ 9
- Hit: ~45 negatives : 55 positives → spw ≈ 0.8
- SO: ~91 negatives : 9 positives → spw ≈ 10

### Layer 2: LightGBM Classifier

```python
LGBMClassifier(
    max_depth=4,
    learning_rate=0.03,
    n_estimators=1000,
    min_child_samples=20,     # Minimum samples per leaf (LGB equivalent of min_child_weight)
    class_weight='balanced',  # Automatic class reweighting
    random_state=42
)
```

Early stopping: 50 rounds with LGB callback `lgb.early_stopping(50)`.

### Layer 3: Meta-Learner (Stacking)

```python
# Generate base predictions on held-out test set (20%)
xgb_proba = final_xgb.predict_proba(X_test_imp)[:, 1]
lgb_proba  = final_lgb.predict_proba(X_test_imp)[:, 1]

# Stack as 2-feature input
meta_X = pd.DataFrame({'xgb': xgb_proba, 'lgb': lgb_proba})

# Fit logistic regression on stacked probabilities
meta = LogisticRegression().fit(meta_X, y_test)
```

**Important caveat:** The meta-learner is trained on only 20% of data (the test split) and learns coefficients for weighting XGB vs LGB. In practice, because XGB and LGB are highly correlated on this task, the meta-learner's intercept becomes very negative (compensating for the low base rate), suppressing outputs to ~2%. As a result, **app.py uses XGB probabilities directly** rather than the meta-learner output for signal generation.

### Artifact Files

```
models/artifacts/
├── hr_xgb.json      ← XGB model binary (HR)
├── hr_lgb.txt       ← LGB booster model (HR)
├── hr_meta.pkl      ← {'meta': LR, 'features': [...], 'train_medians': {...}}
├── hit_xgb.json
├── hit_lgb.txt
├── hit_meta.pkl
├── so_xgb.json
├── so_lgb.txt
└── so_meta.pkl
```

`train_medians` in the `.pkl` is a dict mapping feature_name → median value computed on the training split. Used at inference to impute missing features.

---

## 6. Training Methodology

### Time Series Cross-Validation (5 Folds)

Training data is sorted by `game_date` and split with `TimeSeriesSplit(n_splits=5)`. Each fold uses earlier dates for training and later dates for validation — strictly no future data leaks into training.

```
Fold 1: [dates 1–20%] train → [dates 20–36%] validate
Fold 2: [dates 1–36%] train → [dates 36–52%] validate
Fold 3: [dates 1–52%] train → [dates 52–68%] validate
Fold 4: [dates 1–68%] train → [dates 68–84%] validate
Fold 5: [dates 1–84%] train → [dates 84–100%] validate
```

CV AUC is reported per fold. Final model is then trained on an 80/20 stratified split.

### Final Model Training

After CV evaluation, a final model is trained on `train_test_split(test_size=0.2, stratify=y)`:
- Training set (80%) → fit XGB + LGB
- Test set (20%) → evaluate + fit meta-learner
- Medians computed from training set only, applied to test set

### Missing Value Imputation

Per-fold median imputation:
1. Compute column medians on training fold only
2. Apply same medians to validation fold
3. Final medians stored in `train_medians` dict in `{target}_meta.pkl`
4. Applied at inference time for any missing features

---

## 7. Inference Pipeline

**Path:** `dashboards/app.py` → `PredictionEngine`

### At Startup

1. `load_model()` — loads XGB, LGB, meta artifacts for all 3 targets
2. `connect_db()` — connects to `baseball_migration_test`
3. `_load_dataset()` — loads `complete_dataset.csv` into memory
4. `_compute_thresholds()` — runs XGB on full training set to compute percentile thresholds (used for header stats but **not** for daily signal assignment — see §8)

### Per Request (`generate_daily_predictions_with_results`)

**Step 1: Load players for date**
- Primary: `hitting_stats JOIN games` for target date from DB
- Fallback: filter `complete_dataset.csv` by game_date

**Step 2: Feature row construction**
For each player row, defaults are set for any missing features:

```python
feature_row.setdefault('pitcher_era_30d', 4.5)
feature_row.setdefault('pitcher_hr_per_9_30d', 1.2)
feature_row.setdefault('pitcher_k_per_9_30d', 20.0)
feature_row.setdefault('pitcher_whip_30d', 1.3)
feature_row.setdefault('park_factor', get_park_factor(home_team))
feature_row.setdefault('travel_fatigue_score', 72.0)
feature_row.setdefault('wind_speed_mph', 0.0)
feature_row.setdefault('temp_f', 72.0)
feature_row.setdefault('recent_hr_rate_14d', 0.03)
feature_row.setdefault('recent_hit_rate_14d', 0.25)
feature_row.setdefault('recent_so_rate_14d', 0.20)
```

**Step 3: Model inference (`predict` method)**
```python
# Build 1-row DataFrame with all training features
X = pd.DataFrame([features])[model_data['features']]

# Impute any remaining NaN with training medians
X = X.fillna(model_data['train_medians'])

# XGB probability (primary signal source)
prob = model_data['xgb'].predict_proba(X)[:, 1][0]

# LGB computed but not used for signal (consistency check only)
lgb_proba = model_data['lgb'].predict(X)
```

**Step 4: Sort and rank signals**
All players for the date sorted descending by XGB probability, then signals assigned by rank position (see §8).

**Step 5: Composite score**
`compute_composite_score()` normalizes each player's key features against the day's pool and produces a 0–100 score (see §9).

---

## 8. Signal Classification

### Current Method: Per-Date Rank-Based (as of 2026-02-24)

Signals are assigned **after sorting all predictions by probability descending**, based on rank position within the day's slate. This guarantees consistent, meaningful distribution regardless of the model's absolute output range.

```python
cutoffs = [
    (0.10, 'STRONG_BUY'),   # Top 10% of today's slate
    (0.25, 'BUY'),          # Next 15% (75th–90th percentile)
    (0.50, 'MODERATE'),     # Next 25% (50th–75th percentile)
    (0.75, 'AVOID'),        # Next 25% (25th–50th percentile)
    (1.00, 'STRONG_SELL'),  # Bottom 25%
]

for i, pred in enumerate(sorted_predictions):
    rank_pct = i / len(predictions)
    for threshold, label in cutoffs:
        if rank_pct < threshold:
            pred['signal_label'] = label
            break
```

### Why Not Fixed Thresholds?

XGBoost outputs are calibrated relative to the training distribution, not to raw base-rate probabilities. The output range differs dramatically per target:
- HR model output: ~0.33–0.74
- Hit model output: ~0.61–0.90
- SO model output: varies

Fixed thresholds computed from training-data percentiles fail on gameday slates because gameday players are a selected active subset with a systematically shifted probability distribution. For example, if training 90th percentile for Hit = 0.728, but all gameday Hit predictions fall between 0.61–0.90, then 92%+ of gameday players get classified as STRONG_BUY — meaningless.

Per-date rank-based classification means STRONG_BUY always means "elite pick relative to today's options."

### Signal Meaning

| Signal | Rank Position | Interpretation |
|--------|--------------|----------------|
| STRONG_BUY | Top 10% | Elite matchup — strongest model signal available today |
| BUY | Top 10–25% | Above-average signal — strong pick |
| MODERATE | Top 25–50% | Average signal — consider with other factors |
| AVOID | Bottom 25–50% | Below-average model signal |
| STRONG_SELL | Bottom 25% | Worst matchups — filtered from display by default |

### Header Stats (`/api/stats/all-targets`)

The model-stats header counts historical "picks" as the **top 25% per date** (STRONG_BUY + BUY equivalent) across all dates in `complete_dataset.csv`:

```python
# For each date in dataset, rank by XGB probability
# Count top 25% as "picks" and check if label = 1
for date_group in df.groupby('game_date'):
    top25 = date_group.sort_values('_proba', ascending=False).head(25%)
    picks += len(top25)
    hits += top25[label_col].sum()
```

This mirrors exactly what the daily predictions endpoint shows.

---

## 9. Composite Score

**Method:** `compute_composite_score(row, all_rows)`

The composite score (0–100) is a **separate interpretability metric** from the model probability. It measures evidence strength across five dimensions, normalizing each player's values against the day's full pool. It is used for display and ranking purposes alongside model probability.

### Components and Weights

| Component | Weight | Features Used | Direction |
|-----------|--------|--------------|-----------|
| Power | 30% | xwoba (45%), barrel_rate (35%), avg_ev (20%) | Higher = better |
| Pitcher Matchup | 25% | pitcher_hr_per_9_30d (60%), pitcher_era_30d (40%) | Higher = better (vulnerable pitcher) |
| Recent Form | 20% | recent_hr_rate_14d | Higher = better |
| Park + Environment | 15% | park_factor (50%), wind_out_factor (30%), temp_f (20%) | Higher = better |
| Freshness | 10% | travel_fatigue_score | **Inverted** — lower fatigue = better |

### Normalization

Each metric is min-max normalized against the day's prediction pool:

```python
def normalize(val, vals, invert=False):
    lo, hi = min(vals), max(vals)
    if hi == lo: return 0.5
    n = (val - lo) / (hi - lo)
    return 1.0 - n if invert else n
```

### Formula

```python
power      = xwoba_n * 0.45 + barrel_n * 0.35 + ev_n * 0.20
matchup    = p_hr9_n * 0.60 + p_era_n * 0.40
form       = hr_rate_14d_n
env        = pf_n * 0.50 + wind_n * 0.30 + temp_n * 0.20
freshness  = 1.0 - fatigue_n    # inverted

composite  = power*0.30 + matchup*0.25 + form*0.20 + env*0.15 + freshness*0.10
score      = composite * 100  # → 0–100
```

Default values used when features are missing:
- xwoba: 0.320, barrel_rate: 0.06, avg_ev: 88.0
- pitcher_hr_per_9: 1.2, pitcher_era: 4.0
- recent_hr_rate: 0.03, park_factor: 1.0
- wind_out: 1.0, temp: 72°F, fatigue: 0

---

## 10. API Endpoints

**Base URL:** `http://localhost:5002`

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/predictions/today` | GET | Today's predictions (all 3 targets) |
| `/api/predictions/<date>` | GET | Predictions for specific date (YYYY-MM-DD) |
| `/api/predictions/<date>?target=hr` | GET | Predictions for specific target |
| `/api/predictions/all` | GET | All predictions across all dates |
| `/api/stats/all-targets` | GET | Header panel stats: picks count + hit rate per target |
| `/api/dates` | GET | All available dates with player counts |
| `/api/reload-models` | POST | Hot-reload model artifacts from disk |

### `/api/predictions/<date>` Response Structure

```json
{
  "date": "2025-09-03",
  "target": "hr",
  "predictions": [
    {
      "player_name": "Aaron Judge",
      "team": "NYY",
      "opponent": "BOS",
      "pitcher_name": "Brayan Bello",
      "is_home": true,
      "probability": 0.651,
      "signal": "STRONG_BUY",
      "signal_label": "STRONG_BUY",
      "implied_probability": 0.286,
      "edge": 0.365,
      "edge_pct": 36.5,
      "score": 84.2,
      "actual_hr": true,
      "actual_hr_count": 1
    }
  ],
  "stats": {
    "total_picks": 311,
    "total_hrs": 28,
    "hit_rate": 9.0
  }
}
```

### `/api/stats/all-targets` Response Structure

```json
{
  "hr":  { "picks": 10053, "hit_rate": 20.5 },
  "hit": { "picks": 10053, "hit_rate": 77.0 },
  "so":  { "picks": 10053, "hit_rate": 14.3 }
}
```

`picks` = total rows classified as STRONG_BUY or BUY across all historical dates.
`hit_rate` = percentage of those picks where the outcome occurred.

---

## 11. Key Design Decisions

### No Leakage Guarantees

| Risk | Mitigation |
|------|-----------|
| Future pitcher stats in training | 30-day window excludes game_date: `game_date < {date}` |
| Future batter rates in training | 14-day window excludes game_date: `game_date < {date}` |
| Hellraiser confidence score | Excluded from training features (external signal, inference only) |
| Outcome columns in training | Explicitly excluded: hr_count, hit_count, so_count, batter_hit, etc. |
| Cross-fold contamination | TimeSeriesSplit ensures folds never have future data in training |
| Median imputation leakage | Medians computed on training fold only, applied to validation fold |

### Spine Design Choice

Using `play_by_play_plays` as the training spine (not `hitting_stats`) provides:
- Full player names (vs. abbreviated names in hitting_stats)
- Outcome derivation from raw play data (more reliable than aggregated stats)
- Direct pitcher assignment per PA → modal pitcher = primary matchup

### Pitcher Stats Coverage

Early versions matched pitcher stats via player_name → player_name join (0.2% coverage). Current approach joins via `pitching_stats.game_id → games.game_id → team` to get the opposing team's staff stats, achieving ~92% coverage.

### Meta-Learner Bypass

The meta-learner (logistic regression stacking XGB + LGB) is trained on only the 20% test split. Its large negative intercept suppresses predictions to ~2% for rare events (HR, SO), making signals meaningless. **XGB probability is used directly** for all signal generation. LGB is retained as an artifact but serves as a consistency check only.

### Signal Design: Relative Not Absolute

Per-date rank-based signals (vs. fixed thresholds) because:
1. XGBoost outputs are not calibrated to real-world probabilities
2. Different targets have very different output ranges (HR: 0.33–0.74, Hit: 0.61–0.90)
3. Gameday slate is a selected subset of active players, systematically shifted from training distribution
4. "Top 10% today" is always a semantically meaningful label regardless of absolute output level

---

## 12. File Manifest

```
ProjectionAI/
├── data/
│   ├── build_training_dataset.py   ← Builds complete_dataset.csv from DB
│   ├── feature_engineering.py      ← Park/travel/weather/pitcher/EV feature functions
│   ├── name_utils.py               ← Canonical name normalization
│   ├── database.py                 ← Schema definitions + connection pool
│   ├── config.py                   ← DB connection config
│   ├── complete_dataset.csv        ← Training data (~40,438 rows)
│   └── weather_cache.json          ← Cached Open-Meteo API responses
│
├── models/
│   ├── train_models_v4.py          ← Current ensemble trainer (XGB + LGB + meta)
│   ├── train.py                    ← Legacy XGB-only trainer
│   ├── train_hr_model.py           ← Legacy HR-specific trainer
│   └── artifacts/
│       ├── {hr,hit,so}_xgb.json   ← XGB model binaries
│       ├── {hr,hit,so}_lgb.txt    ← LGB booster models
│       └── {hr,hit,so}_meta.pkl   ← Meta-learner + feature list + train medians
│
├── dashboards/
│   └── app.py                      ← Flask API server (port 5002)
│                                      PredictionEngine class:
│                                        load_model(), connect_db(), _load_dataset()
│                                        _compute_thresholds()
│                                        predict(), generate_daily_predictions_with_results()
│                                        compute_composite_score()
│                                        get_all_target_stats()
│
├── matchup_model_v3.py             ← Analysis + betting simulation scripts (legacy)
├── matchup_model_v2.py             ← Legacy
├── matchup_model.py                ← Legacy
├── research_analysis.py            ← Ad-hoc analysis scripts
└── requirements.txt
```

### Key Dependencies

```
xgboost>=2.0
lightgbm>=4.1
scikit-learn>=1.3
pandas>=2.1
numpy>=1.26
Flask>=3.0
psycopg2-binary>=2.9
requests>=2.31
```

---

## Retraining Procedure

```bash
# 1. Rebuild training dataset from DB
cd data
python build_training_dataset.py
# → writes data/complete_dataset.csv

# 2. Train all three models
cd ../models
python train_models_v4.py
# → writes artifacts/{hr,hit,so}_{xgb,lgb,meta}.*

# 3. Hot-reload running server
curl -X POST http://localhost:5002/api/reload-models
# → PredictionEngine.reload_models() re-reads all artifacts

# 4. Verify thresholds recomputed and signals look correct
grep "Thresholds" /tmp/pai.log
curl -s "http://localhost:5002/api/stats/all-targets"
```

---

*Generated from source code audit of ProjectionAI codebase.*
*For questions about individual components, cross-reference the source files listed in §12.*
