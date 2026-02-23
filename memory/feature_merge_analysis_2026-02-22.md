# Feature Merge Analysis - February 22, 2026

## Summary

Investigated feature building pipeline after new pitcher data was loaded to tables. Found critical name normalization issues preventing proper feature matching.

---

## Current State

### Database Tables
- **custom_pitcher_2025**: 755 records, 2025 season data with spin rates and stats
- **hitter_exit_velocity**: 1,017 records, EV metrics stored as `last_name_first_name` format
- **custom_batter_2025**: 609 records, x-stats stored as `last_name_first_name` format
- **daily_batted_ball_tracking**: 32,126 records, 253 unique batters in `last_name_first_name` format

### Feature Coverage (comprehensive_features.csv - 44,137 records)

| Feature Group | Matched | Coverage | Status |
|--------------|---------|----------|--------|
| Pitcher Stats (K%, ERA) | 17,651 | 40.0% | ⚠️ Partial |
| Pitcher Spin Rates | 14,193-17,651 | 32.2-40.0% | ⚠️ Partial |
| Hitter EV | 0 | 0.0% | ❌ COMPLETE FAILURE |
| Hitter x-Stats | Not analyzed | Likely 0% | ❌ COMPLETE FAILURE |
| Batted Ball Trends | 0 | 0.0% | ❌ COMPLETE FAILURE |

---

## Root Cause: Name Format Mismatch

### The Problem

| Data Source | Name Format | Example |
|-------------|-------------|---------|
| `hellraiser_picks.player_name` | "First Last" | "Aaron Judge", "Cal Raleigh" |
| `hellraiser_picks.pitcher_name` | "First Last" | "Jack Leiter", "Robbie Ray" |
| `custom_pitcher_2025.last_name_first_name` | "Last, First" | "Leiter, Jack", "Ray, Robbie" |
| `hitter_exit_velocity.last_name_first_name` | "Last, First" | "Judge, Aaron", "Raleigh, Cal" |
| `custom_batter_2025.last_name_first_name` | "Last, First" | "Judge, Aaron" |
| `daily_batted_ball_tracking.player_name` | "Last, First" | "Raleigh, Cal" |

### Why Pitcher Features Partially Work (40%)

The merge script (`build_comprehensive_features.py`) includes a `normalize_pitcher_name()` function:

```python
def normalize_pitcher_name(name):
    """Normalize pitcher name to 'Last, First' format"""
    if not name or pd.isna(name):
        return None
    parts = str(name).strip().split(' ')
    if len(parts) >= 2:
        return f"{parts[-1]}, {' '.join(parts[:-1])}"
    return name
```

This is applied to hellraiser_picks.pitcher_name before merging, which is why **17,651 out of 44,137** records got matched.

**However**, only **4,103 out of 11,278** hellraiser_picks records have a pitcher_name assigned to begin with.

### Why Hitter Features Completely Fail (0%)

The merge code does NOT normalize player names:

```python
# Current code (WRONG)
df = df.merge(
    hitter_df,
    left_on='player_name',              # "Aaron Judge"
    right_on='last_name_first_name',     # "Judge, Aaron"
    how='left',
    suffixes=('', '_ev')
)
```

This tries to match "Aaron Judge" with "Judge, Aaron" → **NO MATCHES**.

Same issue for:
- Hitter EV features (0%)
- Batter x-stats (likely 0%)
- Batted ball trends (0%)

---

## Proposed Fix

Add a similar normalization function for player names and use it for all hitter merges:

```python
def normalize_player_name(name):
    """Normalize player name to 'Last, First' format"""
    if not name or pd.isna(name):
        return None
    parts = str(name).strip().split(' ')
    if len(parts) >= 2:
        return f"{parts[-1]}, {' '.join(parts[:-1])}"
    return name
```

Then update merges:

```python
# Normalize player names before merging
df['player_normalized'] = df['player_name'].apply(normalize_player_name)

# Merge hitter EV features
df = df.merge(
    hitter_df,
    left_on='player_normalized',        # Now "Judge, Aaron"
    right_on='last_name_first_name',   # Matches "Judge, Aaron"
    how='left',
    suffixes=('', '_ev')
)

# Merge x-stats
df = df.merge(
    xstats_df,
    left_on='player_normalized',
    right_on='last_name_first_name',
    how='left',
    suffixes=('', '_xstats')
)

# Merge batted ball trends
df = df.merge(
    trends_df,
    left_on='player_normalized',
    right_on='batter_name',  # Which is already "Last, First"
    how='left'
)
```

---

## Expected Impact After Fix

| Feature Group | Current | Expected | Improvement |
|--------------|---------|----------|-------------|
| Pitcher Stats | 17,651 (40%) | ~17,651 (40%) | ✅ No change (limited by missing pitcher_name) |
| Hitter EV | 0 (0%) | ~35,000-40,000 (80-90%) | 🚀 MASSIVE GAIN |
| Batter x-Stats | 0 (0%) | ~35,000-40,000 (80-90%) | 🚀 MASSIVE GAIN |
| Batted Ball Trends | 0 (0%) | ~2,500-5,000 (5-11%) | 📈 Moderate (limited to 253 batters in trends data) |

---

## Data Quality Issues Beyond Names

### 1. Missing Pitcher Names in Source
- Only 4,103 of 11,278 hellraiser_picks have pitcher_name assigned (36%)
- This is a data collection issue, not a normalization issue

### 2. Limited Batter Trend Coverage
- Only 253 unique batters in daily_batted_ball_tracking
- 44,137 predictions may cover many more players
- Batted ball trend feature will have sparse coverage even after fix

### 3. Pitcher Data Only 2025 Season
- custom_pitcher_2025 contains 2025 data only
- hellraiser_picks may span multiple seasons
- May cause mismatches for historical data

---

## Next Steps

1. **Fix the normalization bug** in `build_comprehensive_features.py`
2. **Re-run feature building** to generate corrected comprehensive_features.csv
3. **Analyze feature coverage** after fix to confirm improvements
4. **Investigate missing pitcher_name** in hellraiser_picks (why only 36% have values?)
5. **Consider expanding batted ball trend data** to cover more batters

---

**Analysis completed:** February 22, 2026
**Status:** Root cause identified, fix ready to implement
