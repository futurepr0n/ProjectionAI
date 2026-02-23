#!/usr/bin/env python3
"""
Matchup Memory Analysis - Prototype

This script analyzes historical batter-pitcher matchups to identify patterns
where our predictions failed and calculate revenge factors for future matchups.

Goal: Find "Failed K" instances (predicted Strikeout, but result was Hit/HR) and
determine if this creates a persistent advantage for the batter.
"""

import pandas as pd
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple
import argparse
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from data.database import Database


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load predictions and results from database.

    Returns:
        Tuple of (plays_df, picks_df)
    """
    db = Database(database="projectionai")
    if not db.connect():
        raise Exception("Failed to connect to database 'projectionai'")

    # Load predictions from hellraiser_picks
    picks_query = """
    SELECT
        run_date as prediction_date,
        player_name,
        prediction_signal as prediction_type,
        game_id
    FROM hellraiser_picks
    WHERE prediction_signal IS NOT NULL
      AND run_date IS NOT NULL
      AND player_name IS NOT NULL
    """

    picks_data = db.execute_query(picks_query)
    if not picks_data:
        raise Exception("Failed to load data from hellraiser_picks")
    picks_df = pd.DataFrame(picks_data)

    # Load results from play_by_play_plays
    plays_query = """
    SELECT
        g.game_date,
        pp.batter as player_name,
        pp.pitcher,
        pp.play_result,
        pp.game_id
    FROM play_by_play_plays pp
    JOIN games g ON pp.game_id = g.game_id
    WHERE g.game_date IS NOT NULL
      AND pp.batter IS NOT NULL
      AND pp.play_result IS NOT NULL
    """

    plays_data = db.execute_query(plays_query)
    if not plays_data:
        raise Exception("Failed to load data from play_by_play_plays")
    plays_df = pd.DataFrame(plays_data)

    db.close()

    return plays_df, picks_df


def find_failed_strikeouts(
    plays_df: pd.DataFrame,
    picks_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Find instances where we predicted Strikeout but the result was Hit or HR.

    Merge on player_name + game_date to find Failed Ks.

    Args:
        plays_df: Actual play-by-play data (game_date, player_name, play_result)
        picks_df: Prediction data (prediction_date, player_name, prediction_type)

    Returns:
        DataFrame with failed strikeout matchups
    """
    # Merge on player_name and date
    merged = plays_df.merge(
        picks_df,
        left_on=['player_name', 'game_date'],
        right_on=['player_name', 'prediction_date'],
        how='inner'
    )

    # Filter for predicted strikeouts (prediction_type='Strikeout')
    predicted_k = merged[
        (merged['prediction_type'].str.contains('Strikeout', case=False, na=False)) |
        (merged['prediction_type'] == 'SO')
    ].copy()

    # Filter for actual hits or home runs (play_result='Hit' OR play_result='Home Run')
    failed_k = predicted_k[
        (predicted_k['play_result'].str.contains('Hit', case=False, na=False)) |
        (predicted_k['play_result'].str.contains('Home Run', case=False, na=False))
    ].copy()

    print(f"Found {len(failed_k)} failed strikeout matchups")
    return failed_k


def calculate_revenge_factor(
    failed_k_df: pd.DataFrame,
    plays_df: pd.DataFrame
) -> Dict[Tuple[str, str], Dict]:
    """
    Calculate revenge factor for each batter-pitcher pair.

    For each Failed K, check if they met again later (same batter, same pitcher).
    Calculate 'Revenge Wins' / 'Total Rematches'.

    Returns:
        Dictionary mapping (batter_name, pitcher_name) to revenge statistics
    """
    matchup_stats = {}

    for _, row in failed_k_df.iterrows():
        batter_name = str(row.get('player_name', ''))
        pitcher_name = str(row.get('pitcher', ''))
        failed_date = row.get('game_date')

        if not batter_name or not pitcher_name or not failed_date:
            continue

        # Find all future matchups between this batter and pitcher
        future_matchups = plays_df[
            (plays_df['player_name'] == batter_name) &
            (plays_df['pitcher'] == pitcher_name) &
            (plays_df['game_date'] > failed_date)
        ]

        if len(future_matchups) == 0:
            # No future meetings
            matchup_stats[(batter_name, pitcher_name)] = {
                'initial_fail_date': str(failed_date),
                'future_matchups': 0,
                'revenge_wins': 0,
                'revenge_rate': 0.0,
                'sample_size': 0
            }
            continue

        # Count revenge wins (batter hits again or hits HR)
        revenge_wins = future_matchups[
            (future_matchups['play_result'].str.contains('Hit', case=False, na=False)) |
            (future_matchups['play_result'].str.contains('Home Run', case=False, na=False))
        ]
        revenge_wins_count = len(revenge_wins)

        matchup_stats[(batter_name, pitcher_name)] = {
            'initial_fail_date': str(failed_date),
            'future_matchups': len(future_matchups),
            'revenge_wins': revenge_wins_count,
            'revenge_rate': revenge_wins_count / len(future_matchups),
            'sample_size': len(future_matchups)
        }

    return matchup_stats


def generate_adjustment_formula(matchup_stats: Dict, min_sample_size: int = 3) -> str:
    """
    Generate a score adjustment formula based on revenge factors.

    If 'Revenge Wins' / 'Total Rematches' > 0.5, output formula:
    'Adjustment: Hit Probability *= 1.15'

    Args:
        matchup_stats: Dictionary of matchup statistics
        min_sample_size: Minimum number of future matchups to consider reliable

    Returns:
        String describing the adjustment formula
    """
    # Calculate average revenge rate for reliable matchups
    reliable_matchups = {
        k: v for k, v in matchup_stats.items()
        if v['future_matchups'] >= min_sample_size
    }

    if not reliable_matchups:
        return "Insufficient data to generate formula (need more rematch matchups)"

    avg_revenge_rate = sum(v['revenge_rate'] for v in reliable_matchups.values()) / len(reliable_matchups)
    total_matchups = len(reliable_matchups)

    # Determine if adjustment should be applied
    # If 'Revenge Wins' / 'Total Rematches' > 0.5
    adjustment_factor = None
    if avg_revenge_rate > 0.5:
        adjustment_factor = 1.15

    formula_lines = [
        "=== MATCHUP MEMORY ADJUSTMENT FORMULA ===",
        f"\nBased on {total_matchups} reliable batter-pitcher pairs with {min_sample_size}+ rematches:",
        f"  - Average 'Revenge Rate': {avg_revenge_rate:.1%} (Revenge Wins / Total Rematches)",
        f"  - Initial prediction failures: {len(matchup_stats)}",
        "",
    ]

    if adjustment_factor:
        formula_lines.extend([
            "SCORE ADJUSTMENT RULE:",
            "  If Revenge Rate > 0.5 (batter wins more often in rematches):",
            f"    Adjustment: Hit Probability *= {adjustment_factor}",
            "",
            "IMPLEMENTATION:",
            "  1. Check matchup history for (batter_name, pitcher_name) pair",
            "  2. If last encounter was a 'Failed K' (predicted K, actual Hit/HR)",
            "  3. AND revenge_rate > 0.5 (sample size >= {min_sample_size}):",
            f"       new_hit_prob = base_hit_prob * {adjustment_factor}",
            "",
        ])
    else:
        formula_lines.extend([
            "SCORE ADJUSTMENT RULE:",
            "  No adjustment needed (revenge_rate <= 0.5)",
            "",
        ])

    formula_lines.extend([
        "RELIABILITY METRICS:",
        "  - High confidence: 5+ rematches with revenge_rate > 0.5",
        "  - Medium confidence: 3-4 rematches with revenge_rate > 0.5",
        "  - Low confidence: < 3 rematches (use base prediction)",
        "",
        "NOTES:",
        "  - This adjustment captures 'hitter has pitcher's number' patterns",
        "  - Apply as a multiplicative factor to the existing Hit probability",
    ])

    return "\n".join(formula_lines)


def save_results(
    failed_k_df: pd.DataFrame,
    matchup_stats: Dict
):
    """Save analysis results to scripts/matchup_revenge_factors.json."""
    output_path = Path(__file__).parent / "matchup_revenge_factors.json"

    # Convert tuple keys to strings for JSON serialization
    serializable_stats = {
        f"{batter_name}|{pitcher_name}": stats
        for (batter_name, pitcher_name), stats in matchup_stats.items()
    }

    result = {
        'matchup_revenge_factors': serializable_stats,
        'summary': {
            'total_failed_ks': len(failed_k_df),
            'total_matchup_pairs': len(matchup_stats),
            'pairs_with_rematches': sum(1 for v in matchup_stats.values() if v['future_matchups'] > 0),
            'avg_revenge_rate': sum(v['revenge_rate'] for v in matchup_stats.values()) / max(len(matchup_stats), 1)
        }
    }

    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"Saved revenge factor statistics to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze matchup memory and revenge factors")
    parser.add_argument("--min-sample", type=int, default=3,
                        help="Minimum sample size for reliable revenge stats")
    args = parser.parse_args()

    print("=" * 60)
    print("MATCHUP MEMORY ANALYSIS")
    print("=" * 60)

    # Load data from database
    print("\n[1] Loading data from database...")
    try:
        plays_df, picks_df = load_data()
        print(f"  Loaded {len(plays_df)} play-by-play records")
        print(f"  Loaded {len(picks_df)} prediction records")
    except Exception as e:
        print(f"  ERROR: {e}")
        return

    # Find failed strikeout predictions
    print("\n[2] Finding failed strikeout predictions...")
    try:
        failed_k_df = find_failed_strikeouts(plays_df, picks_df)
        print(f"  Found {len(failed_k_df)} instances where we predicted K but got Hit/HR")
    except Exception as e:
        print(f"  ERROR: {e}")
        return

    # Calculate revenge factors
    print("\n[3] Calculating revenge factors...")
    matchup_stats = calculate_revenge_factor(failed_k_df, plays_df)

    # Print summary
    print("\n[4] Summary Statistics:")
    with_rematch = sum(1 for v in matchup_stats.values() if v['future_matchups'] > 0)
    avg_future = sum(v['future_matchups'] for v in matchup_stats.values()) / max(len(matchup_stats), 1)
    avg_revenge = sum(v['revenge_rate'] for v in matchup_stats.values()) / max(len(matchup_stats), 1)

    print(f"  Total failed K matchups: {len(matchup_stats)}")
    print(f"  With future rematches: {with_rematch}")
    print(f"  Avg future matchups per pair: {avg_future:.2f}")
    print(f"  Avg revenge rate: {avg_revenge:.1%}")

    # Generate adjustment formula
    print("\n[5] Generating adjustment formula...")
    formula = generate_adjustment_formula(matchup_stats, args.min_sample)
    print("\n" + formula)

    # Save results
    print("\n[6] Saving results...")
    save_results(failed_k_df, matchup_stats)

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
