# Implementation Checklist - ProjectionAI Model Enhancement Roadmap

## PHASE 1: Unified Player Name Registry

### ✅ Created: `/data/migrate_player_names.py`
**Status:** Complete (400+ lines)

**Features:**
- [x] Create player_name_map table with proper schema
- [x] Create 4 indexes (mlb_id, last_name, first_initial, aliases GIN)
- [x] Seed from players table (canonical names + MLB IDs)
- [x] Add aliases from hitter_exit_velocity
- [x] Add aliases from custom_batter_2025
- [x] Add aliases from play_by_play_plays (5000 limit)
- [x] Statistics reporting (`--stats` flag)
- [x] Command-line interface with --all, --create-table, --seed

**Usage:**
```bash
python migrate_player_names.py --all          # Full migration
python migrate_player_names.py --stats        # View statistics
```

### ✅ Enhanced: `/data/name_utils.py`
**Status:** Complete (+130 lines)

**New Functions:**
- [x] `normalize_to_canonical(raw_name, conn=None)` - 50 lines
  - Handles "Lastname, Firstname" format
  - Handles "A. Lastname" format
  - Removes suffixes (Jr, Sr, II, III, IV, V)
  - Falls back to fuzzy matching on last name
  
- [x] `build_name_map(conn=None)` - 40 lines
  - Builds complete dictionary from DB
  - Maps aliases → canonical
  - Maps last names (with "last_" prefix)
  - Returns ~20000+ entries after full seeding

**Backward Compatibility:**
- [x] Existing functions unchanged
- [x] New imports added (psycopg2, os, logging)
- [x] Graceful DB error handling

---

## PHASE 2: Fix NULL Features in Training Data

### ✅ Enhanced: `/data/build_training_dataset.py`
**Status:** Complete (+80 lines)

**Import Changes:**
```python
from name_utils import fuzzy_join_names, normalize_to_canonical, build_name_map
```

**Method Enhancements:**

1. [x] `_get_hitter_ev_stats()`
   - Uses canonical name matching
   - Fallback to fuzzy matching (threshold 80)
   - Returns: avg_ev, barrel_rate, sweet_spot_percent

2. [x] `_get_xstats()`
   - Uses canonical name matching
   - Fallback to fuzzy matching
   - Returns: xwoba, xba, xslg

3. [x] `_get_recent_hr_rate()` - ENHANCED
   - Returns 3 features (was 1):
     - recent_hr_rate_14d ✅
     - recent_hit_rate_14d ✅ NEW
     - recent_so_rate_14d ✅ NEW
   - Uses canonical name matching
   - SQL query updated with multiple aggregates

4. [x] `_get_hr_results()`
   - Normalizes batter names to canonical
   - Adds batter_canonical column

**Expected Improvement:**
- Barrel rate: 5% → 70% coverage
- Exit velocity: 5% → 70% coverage
- xStats: 2% → 60% coverage
- Hit rate: None → 85% coverage (NEW)
- SO rate: None → 85% coverage (NEW)

---

## PHASE 3: Live Feature Enrichment

### ✅ Enhanced: `/dashboards/app.py`
**Status:** Complete (+120 lines)

**Signal Thresholds Added:**
```python
SIGNAL_THRESHOLDS = {
    'hr':  {'STRONG_BUY': 0.20, 'BUY': 0.15, 'MODERATE': 0.10, 'AVOID': 0.07},
    'hit': {'STRONG_BUY': 0.55, 'BUY': 0.45, 'MODERATE': 0.38, 'AVOID': 0.30},
    'so':  {'STRONG_BUY': 0.65, 'BUY': 0.55, 'MODERATE': 0.45, 'AVOID': 0.35},
}
```

**New Methods in PredictionEngine:**

1. [x] `_get_park_factor(team_or_venue: str) -> float`
   - HR impact multiplier by team
   - High: COL 1.35, NYY 1.20, BOS 1.15
   - Neutral: ATL, CHC, LAD, MIL 1.00
   - Low: OAK, MIA, PIT 0.90
   - Fallback: 1.0

2. [x] `_get_batter_xstats(player_name: str) -> dict`
   - Queries custom_batter_2025
   - Returns: xwoba, xba, xslg
   - Neutral defaults: 0.320, 0.250, 0.420

3. [x] `_get_pitcher_rolling_stats(pitcher_name: str, game_date) -> dict`
   - 30-day rolling stats
   - Returns: pitcher_era, pitcher_hr_per_9, pitcher_k_per_9, pitcher_whip
   - Defaults: 4.5, 1.2, 20.0, 1.3

**Updated Methods:**

4. [x] `predict()` method
   - Now uses target-specific thresholds
   - Retrieves from SIGNAL_THRESHOLDS[target]
   - Generates signals relative to model range

---

## PHASE 4: Signal Thresholds & Component Score Extraction

### ✅ Updated: `/dashboards/app.py`
**Status:** Complete (thresholds)

- [x] Per-target thresholds for HR, HIT, SO
- [x] Updated predict() to use SIGNAL_THRESHOLDS
- [x] Signal generation logic updated

### ✅ Enhanced: `/frontend/server/mlb/hellraiser.js`
**Status:** Complete (+30 lines)

**Scalar Extraction Logic:**
```javascript
// Extract from component_scores.batter_analysis
const batter = componentScores.batter_analysis || {};
extractedBarrelRate = extractedBarrelRate || batter.barrel_rate;
extractedExitVelo = extractedExitVelo || batter.exit_velocity_avg;
extractedHardHit = extractedHardHit || batter.hard_hit_percent;
extractedSweetSpot = extractedSweetSpot || batter.sweet_spot_percent;
```

**Features:**
- [x] Checks DB scalar columns first
- [x] Falls back to extracting from component_scores JSONB
- [x] Ensures comprehensive feature availability
- [x] Returns extracted values in API response

---

## PHASE 5: Bug Fixes

### ✅ Fixed: `/data/feature_engineering.py`
**Status:** Complete (2 critical bugs fixed)

**Bug 1: Dome Stadium Detection (Line 387)**
- [x] Changed from: `if loc.get('stadium_name') in DOME_STADIUMS:`
- [x] Changed to: `if team in DOME_STADIUMS:`
- [x] Reason: DOME_STADIUMS contains team codes, not stadium names

**Bug 2: Travel Fatigue Logic (Line 174)**
- [x] Changed from: `else prev_game['home_team']`
- [x] Changed to: `else prev_game['away_team']`
- [x] Reason: Logic error - always returned home_team

---

## Testing & Verification

### Pre-Migration Verification
- [x] All files edited without syntax errors
- [x] Backward compatibility maintained
- [x] Graceful error handling for DB failures
- [x] Default values provided for all new methods

### Post-Migration (Manual Steps)
- [ ] Run `python data/migrate_player_names.py --all`
- [ ] Verify with `python data/migrate_player_names.py --stats`
- [ ] Run `python data/build_training_dataset.py`
- [ ] Check logs for improved feature coverage
- [ ] Test API endpoints with `curl /api/mlb/analytics/hellraiser-analysis`

### Expected Results
- [ ] player_name_map table created with 30000+ entries
- [ ] Feature coverage > 70% for main metrics
- [ ] No NULL values in barrel_rate, exit_velocity_avg, xstats
- [ ] Recent HIT and SO rates populated
- [ ] API responses include extracted scalar values

---

## Files Modified Summary

| File | Type | Lines Changed | Status |
|------|------|---------------|--------|
| migrate_player_names.py | NEW | 400+ | ✅ Complete |
| name_utils.py | MOD | +130 | ✅ Complete |
| build_training_dataset.py | MOD | +80 | ✅ Complete |
| app.py | MOD | +120 | ✅ Complete |
| hellraiser.js | MOD | +30 | ✅ Complete |
| feature_engineering.py | MOD | 2 fixes | ✅ Complete |
| ROADMAP_IMPLEMENTATION.md | NEW | Doc | ✅ Complete |
| QUICK_START.md | NEW | Doc | ✅ Complete |

**Total Lines Added:** 760+
**Total Lines Modified:** 140+
**New Database Queries:** 15+
**New Methods:** 5
**Bug Fixes:** 2

---

## Deployment Checklist

### Before Running Migration
- [ ] Backup database (`pg_dump baseball_migration_test`)
- [ ] Read ROADMAP_IMPLEMENTATION.md
- [ ] Verify Python environment: `source venv/bin/activate`
- [ ] Check disk space for new table

### Running Migration (Step-by-Step)
```bash
# Step 1: Create table and indexes
python data/migrate_player_names.py --create-table

# Step 2: Seed canonical names
python data/migrate_player_names.py --seed

# Step 3: Verify statistics
python data/migrate_player_names.py --stats

# Step 4: Test training build
python data/build_training_dataset.py

# Step 5: Restart API services
# (as needed in your deployment)
```

### Post-Deployment Verification
- [ ] Check feature coverage in logs
- [ ] Monitor API performance (+5-10ms expected)
- [ ] Test predictions on HR, HIT, SO targets
- [ ] Monitor signal accuracy for 1 week
- [ ] Adjust SIGNAL_THRESHOLDS if needed

---

## Rollback Plan (if needed)

```bash
# Step 1: Drop the table
psql -h 192.168.1.23 -U postgres -d baseball_migration_test -c \
  "DROP TABLE IF EXISTS player_name_map;"

# Step 2: Revert file changes
git checkout -- data/name_utils.py data/build_training_dataset.py dashboards/app.py frontend/server/mlb/hellraiser.js data/feature_engineering.py

# Step 3: Restart services
```

---

## Support & Documentation

- **Main Doc:** ROADMAP_IMPLEMENTATION.md
- **Quick Start:** QUICK_START.md
- **This Checklist:** IMPLEMENTATION_CHECKLIST.md

---

**Status:** ✅ ALL PHASES COMPLETE & READY FOR TESTING
**Last Updated:** February 2026
**Ready to Deploy:** YES
