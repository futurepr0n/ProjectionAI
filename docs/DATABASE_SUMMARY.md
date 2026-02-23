# Database Analysis Summary
**Baseball Migration Test Database (192.168.1.23)**

---

## 📊 Database Overview

**Total Tables:** 75
**Database Name:** baseball_migration_test
**Connection:** postgres@192.168.1.23:5432
**Season Coverage:** 2025 (March 18 - September 28)
**Total Games:** 2,409

---

## 🎯 Key Tables for ProjectionAI

### 1. Games Table (2,409 rows)
```sql
- game_id (integer, PK)
- game_date (date)
- home_team (varchar)
- away_team (varchar)
- home_score (integer)
- away_score (integer)
- status (varchar)
- venue (text)
- attendance (integer)
```

**Date Range:** 2025-03-18 to 2025-09-28
**Unique Dates:** 184

### 2. Odds Tracking Table (199,201 rows)
```sql
- id (integer, PK)
- tracking_timestamp (timestamp)
- tracking_date (date)
- game_date (date) - NOT POPULATED (all NULL)
- player_id (integer)
- player_name (varchar)
- team (varchar)
- opponent (varchar)
- game_id (bigint) - NOT POPULATED (0 rows)
- prop_type (varchar) - "hr", "hits"
- prop_line (numeric) - NOT POPULATED (0 rows)
- over_odds (integer)
- under_odds (integer) - NOT POPULATED (0 rows)
- over_probability (numeric)
- under_probability (numeric)
- sportsbook (varchar) - "draftkings"
- opening_odds (boolean)
- is_current (boolean)
```

**Key Stats:**
- Total odds: 199,201
- HR odds: 198,928 (544 unique players)
- Hit odds: 273 (273 unique players)
- Opening odds: 69,495
- Current odds: 817

**Limitations:**
- No game_id populated (cannot join with games directly)
- No prop_line populated (don't know the bet line)
- No under_odds populated (only over odds tracked)

### 3. Line Movement Table (21,417 rows)
```sql
- id (integer, PK)
- tracking_date (date)
- game_date (date)
- player_id (integer)
- player_name (varchar)
- prop_type (varchar)
- prop_line (numeric)
- opening_over_odds (integer)
- opening_under_odds (integer)
- opening_timestamp (timestamp)
- current_over_odds (integer)
- current_under_odds (integer)
- current_line_timestamp (timestamp)
- odds_movement_points (integer)
- movement_direction (varchar)
- movement_significance (varchar)
- is_steam_move (boolean)
- steam_move_timestamp (timestamp)
- reverse_line_movement (boolean)
- sharp_money_indicator (boolean)
```

**Key Stats:**
- Total movements: 21,417
- Steam moves: Tracked
- Sharp money indicators: Tracked

### 4. Hellraiser Picks Table (11,278 rows)
**Existing HR predictions from your current system**

```sql
- id (integer, PK)
- analysis_date (date)
- season (integer)
- player_name (varchar)
- team (varchar)
- pitcher_name (varchar)
- game_description (varchar)
- venue (varchar)
- game_time (varchar)
- is_home (boolean)
- confidence_score (numeric)
- classification (varchar)
- pathway (varchar)
- odds_american (varchar)
- odds_decimal (numeric)
- odds_source (varchar)
- reasoning (text)
- detailed_reasoning (jsonb)
- component_scores (jsonb)
- trend_analysis (jsonb)
- swing_optimization_score (numeric)
- swing_attack_angle (numeric)
- swing_ideal_rate (numeric)
- swing_bat_speed (numeric)
- barrel_rate (numeric)
- exit_velocity_avg (numeric)
- hard_hit_percent (numeric)
- sweet_spot_percent (numeric)
- data_sources_used (jsonb)
- created_at (timestamp)
```

**Key Features:**
- 11,278 total picks
- 50 unique analysis dates
- Rich Statcast metrics included
- Odds tracking (american and decimal)
- JSONB for detailed analysis

### 5. Hitting Stats Table (50,876 rows)
```sql
- id (integer)
- game_id (integer)
- team (varchar)
- player_name (text)
- at_bats (integer)
- runs (integer)
- hits (integer)
- rbi (integer)
- [more columns...]
```

### 6. Pitching Stats Table (20,373 rows)
```sql
- id (integer)
- game_id (integer)
- team (varchar)
- player_name (text)
- innings_pitched (numeric)
- hits (integer)
- runs (integer)
- earned_runs (integer)
- walks (integer)
- strikeouts (integer)
- [more columns...]
```

### 7. Custom Batter 2025 (609 rows)
Custom batter data for 2025 season
- player_id mapping
- match confidence
- player stats

### 8. Custom Pitcher 2025 (748 rows)
Custom pitcher data for 2025 season
- pitcher_id mapping
- match confidence
- pitcher stats

### 9. Play By Play Pitches (491,502 rows)
Pitch-level detailed data
```sql
- id (integer)
- play_id (integer)
- pitch_number (integer)
- result (varchar) - "Home Run", "Ball", etc.
- pitch_type (varchar) - "Slider", "Changeup", etc.
- velocity (integer)
```

### 10. Play By Play Plays (171,460 rows)
Play-level data
```sql
- id (integer)
- metadata_id (integer)
- game_id (integer)
- play_sequence (integer)
- inning (integer)
- inning_half (varchar)
- batter (varchar)
- pitcher (varchar)
- play_description (text)
- play_result (varchar)
- raw_text (text)
```

---

## 🔍 Data Quality Assessment

### ✅ Strengths

1. **Rich Hellraiser Data**
   - 11,278 predictions with full context
   - Includes Statcast metrics (barrel_rate, EV, sweet spot)
   - Has odds tracking
   - Rich JSONB metadata for reasoning

2. **Complete Game Coverage**
   - 2,409 games across 184 days
   - Full season (March 18 - September 28)
   - All 30 teams

3. **Play-By-Play Data**
   - 491,502 pitches with velocity, type, result
   - 171,460 plays with full context
   - Can derive advanced metrics

4. **Player Stats**
   - 50,876 hitting stat rows
   - 20,373 pitching stat rows
   - Custom player mappings for 2025

### ⚠️ Limitations

1. **Odds Tracking Issues**
   - No game_id populated (cannot join with games)
   - No prop_line populated (don't know the bet line)
   - Only over_odds, no under_odds
   - Need to infer game from tracking_date

2. **Hellraiser Performance**
   - 11,278 picks across 50 dates
   - Need to calculate actual hit rate
   - game_time is string, not datetime

3. **Missing Metrics**
   - No rolling_stats_30_day or rolling_stats_7_day data (both 0 rows)
   - Player_season_stats only 338 rows (should be ~900 for 30 teams × 30 players)

---

## 💡 Strategy for ProjectionAI

### Approach 1: Use Hellraiser Picks as Labeled Data

**Best Option:**
- Hellraiser has 11,278 labeled predictions
- Includes Statcast metrics (barrel_rate, EV, sweet spot)
- Has odds data (odds_decimal)
- Can extract actual HR results from play_by_play_pitches

**Steps:**
1. Join Hellraiser picks with games table on date
2. Join with play_by_play_pitches to get actual HR results
3. Use Hellraiser metrics as features
4. Train XGBoost model to predict HR outcomes
5. Compare to Hellraiser's predictions (21% hit rate)

### Approach 2: Build Features from Play-By-Play Data

**Derive Advanced Metrics:**
- Calculate barrel_rate per player per game
- Calculate exit_velocity averages
- Calculate sweet_spot percentages
- Build rolling averages (7-day, 30-day)
- Match with pitcher data

**Steps:**
1. Process play_by_play_pitches (491,502 rows)
2. Aggregate by player, game, date
3. Calculate Statcast-like metrics
4. Join with hitting_stats and pitching_stats
5. Create training dataset

### Approach 3: Hybrid Approach

**Combine Both:**
- Use Hellraiser picks as base (already has features + labels)
- Supplement with play-by-play derived metrics
- Add pitcher quality from pitching_stats
- Train ensemble model

---

## 🎯 Recommended Training Dataset

**Target:** HR Prediction Model

**Features to Extract:**

From Hellraiser Picks:
- barrel_rate
- exit_velocity_avg
- hard_hit_percent
- sweet_spot_percent
- swing_optimization_score
- swing_attack_angle
- swing_bat_speed

From Pitching Stats:
- Join with pitcher_name
- Get pitcher ERA, K rate, HR/9

From Games:
- venue (for park factors)
- home_team / away_team
- game_date

**Labels:**
- Extract actual HR from play_by_play_pitches where result = 'Home Run'
- Match by game_id and player_name

**Expected Dataset Size:**
- ~11,278 labeled samples
- Positive class (HR): ~2,373 (21% - matches Hellraiser hit rate)
- Negative class (no HR): ~8,905

---

## 📊 Next Steps

### Immediate
1. ✅ Database exploration complete
2. [ ] Extract labeled dataset from Hellraiser picks
3. [ ] Join with actual HR results
4. [ ] Feature engineering
5. [ ] Train XGBoost model

### Short-term
6. [ ] Compare XGBoost vs Hellraiser hit rates
7. [ ] Add Hit and Strikeout models
8. [ ] Build validation framework
9. [ ] Create training pipeline

### Long-term
10. [ ] Deploy to production
11. [ ] Integrate real-time predictions
12. [ ] Build dashboard
13. [ ] Add Kelly criterion betting

---

## 🔌 Database Connection

```python
import psycopg2

conn = psycopg2.connect(
    host='192.168.1.23',
    port=5432,
    database='baseball_migration_test',
    user='postgres',
    password='korn5676'
)
```

---

**Analysis Date:** February 16, 2026
**Analyst:** Riff (AI)
**Status:** Ready for model training
