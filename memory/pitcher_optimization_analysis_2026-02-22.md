# Daily Lineups Duplication Analysis

**Date:** February 22, 2026
**Issue:** Attempting to join hellraiser_picks to daily_lineups created duplicate records (198k vs expected 44k)

---

## Findings

### No True Duplicates

At the **game_id level**, there are NO duplicates:
```sql
SELECT game_id, home_team, away_team, game_date, COUNT(*) as dup_count
FROM daily_lineups
GROUP BY game_id, home_team, away_team, game_date
HAVING COUNT(*) > 1;
```
**Result:** 0 rows

### Multiple Games Per Matchup

The same team matchup has multiple **different game_ids** over the season:

| Matchup | Unique Games | Total Rows | First Date | Last Date |
|----------|--------------|-------------|-------------|------------|
| CHC vs MIL | 7 | 9 | 2025-06-17 | 2025-08-21 |
| BOS vs BAL | 6 | 8 | 2025-05-22 | 2025-08-19 |
| CWS vs CLE | 7 | 8 | 2025-07-10 | 2025-08-10 |
| NYY vs TB | 8 | 8 | 2025-03-23 | 2025-07-31 |
| WSH vs NYM | 8 | 8 | 2025-03-20 | 2025-08-21 |

### Why This Happens

1. **Multiple games in season:** Teams play each other 6-8 times per season
2. **Lineup updates:** Games get updated over time (probable starters change, injuries, etc.)
3. **Daily tracking:** daily_lineups tracks lineup status for each game date

### Root Cause of Join Explosion

When joining hellraiser_picks to daily_lineups using **only team names**:
```sql
-- BAD: Matches ANY game with same teams
LEFT JOIN daily_lineups dl ON
    g.home_team = dl.home_team AND g.away_team = dl.away_team
```

A hellraiser_picks record for "NYY @ TB" would match **all 8 games** between those teams.

---

## Solution

### Option 1: Add Date to Join (Recommended)

```sql
-- GOOD: Matches specific game on specific date
LEFT JOIN daily_lineups dl ON
    g.home_team = dl.home_team AND
    g.away_team = dl.away_team AND
    g.game_date = dl.game_date
```

**Issue:** games table may not have accurate game_date, hellraiser_picks has no date field.

### Option 2: Use game_id Direct Join (Best)

```sql
-- BEST: Uses unique game_id
LEFT JOIN daily_lineups dl ON
    g.game_id = dl.game_id
```

**Issue:** Requires games.game_id to match daily_lineups.game_id perfectly.

### Option 3: Use Window Function to Get Latest

```sql
-- Get only the most recent lineup for each game
WITH ranked_lineups AS (
    SELECT
        dl.*,
        ROW_NUMBER() OVER (PARTITION BY dl.home_team, dl.away_team, dl.game_date ORDER BY dl.last_updated DESC) as rn
    FROM daily_lineups dl
)
SELECT * FROM ranked_lineups WHERE rn = 1;
```

---

## Current Status

The daily_lineups join approach is **not viable** without:
1. Accurate game_id matching between games and daily_lineups tables
2. OR Adding game_date to the join condition

**Recommended:** Continue using hellraiser_picks.pitcher_name for now (40% coverage) until games table is updated with reliable game_id references.

---

**Analysis completed:** February 22, 2026
