# Matchup Memory Analysis

This script analyzes historical batter-pitcher matchups to identify patterns where predictions failed and calculates "revenge factors" for future adjustments.

## Purpose

Find "Failed K" instances (predicted Strikeout, but result was Hit/HR) and determine if this creates a persistent advantage for the batter.

## What It Does

1. **Load Data**: Reads play-by-play actuals (`play_by_play_plays`) and predictions (`hellraiser_picks`)
2. **Find Failed Predictions**: Identifies matchups where we predicted Strikeout but the result was Hit or HR
3. **Track Rematches**: For each failed prediction, checks if the batter and pitcher met again later in the season
4. **Calculate Revenge Factor**: Determines how often the batter won again in subsequent matchups
5. **Generate Adjustment Formula**: Outputs a percentage-based adjustment for future predictions

## Usage

```bash
python scripts/analyze_matchup_history.py [OPTIONS]
```

### Options

- `--data-dir DIR`: Directory containing data files (default: `data`)
- `--output-dir DIR`: Directory to save results (default: `output`)
- `--min-sample N`: Minimum sample size for reliable revenge stats (default: 3)

## Output

The script generates:

1. **Console Output**: Summary statistics and adjustment formula
2. **failed_strikeout_matchups.csv**: All failed strikeout matchups with full details
3. **revenge_factor_stats.json**: Revenge statistics for each batter-pitcher pair

## Example Output Formula

```
SCORE ADJUSTMENT RULE:
  If Batter beat Pitcher last time despite Strikeout prediction:
    → Adjust Batter Hit Probability by +15.2%

IMPLEMENTATION:
  1. Check matchup history for (batter_id, pitcher_id) pair
  2. If last encounter was a 'Failed K' (predicted K, actual Hit/HR)
  3. AND revenge_rate >= 0.5 (sample size >= 3):
       new_hit_prob = base_hit_prob * (1 + 0.152)
```

## Data Requirements

The script expects the following data structure:

### play_by_play_plays (actuals)
- `game_id`, `game_date`: Game identification
- `batter_id`, `pitcher_id`: Player IDs
- `result`, `event`, or `play_result`: Actual outcome (Hit, HR, Strikeout, etc.)

### hellraiser_picks (predictions)
- `game_id`: Game identification
- `batter_id`, `pitcher_id`: Player IDs
- `prediction`: Predicted outcome (Strikeout, Hit, etc.)

### Join Key

The script merges on these columns by default (adjust as needed):
- `game_id`
- `batter_id`
- `pitcher_id`

## Key Concepts

### Failed K
A matchup where the model predicted a Strikeout but the batter achieved a Hit or Home Run.

### Revenge Factor
The rate at which a batter continues to beat the same pitcher in future encounters after a Failed K.

### Adjustment Formula
A percentage boost/reduction to apply to Hit Probability based on historical matchup patterns.

## Integration

To integrate into your prediction pipeline:

1. Load `revenge_factor_stats.json` at model startup
2. Before each prediction, check if batter-pitcher pair exists in stats
3. If found and conditions met, apply the adjustment:
   ```python
   def adjust_for_matchup_memory(batter_id, pitcher_id, base_hit_prob):
       key = f"{batter_id}|{pitcher_id}"
       if key in matchup_memory and matchup_memory[key]['future_matchups'] >= 3:
           revenge_rate = matchup_memory[key]['revenge_rate']
           adjustment = (revenge_rate - 0.5) * 2  # Scale factor
           return base_hit_prob * (1 + adjustment)
       return base_hit_prob
   ```

## Notes

- This is a prototype script. Column names and data structure may need adjustment to match your actual schema.
- Sample size matters: Fewer than 3 rematches = low confidence.
- The adjustment is multiplicative, not additive, to maintain probability bounds.
- Consider capping adjustments at ±20% to avoid extreme values.

## Future Enhancements

- Add time decay (older failed Ks should matter less)
- Factor in pitcher fatigue, game context, ballpark factors
- Separate by pitch type (some batters crush certain pitches)
- Add streak detection (consecutive failed Ks = higher confidence)
