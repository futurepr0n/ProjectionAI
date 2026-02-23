#!/usr/bin/env python3
"""
Generate Pitcher Matchup Features

This script calculates repertoire_match_score for each hellraiser_picks prediction
based on pitcher spin rates and prediction type.

Rules:
- If prediction_type is 'Strikeout' and pitcher's sl_avg_spin is high (> 0.30), score = 0.0 (Bad)
- If prediction_type is 'Home Run' and pitcher's ch_avg_spin is high (> 0.30), score = 0.0 (Bad)
- Otherwise, score = 0.5 (Neutral)

Data Flow:
1. Load hellraiser_picks and join with games to get game_id and team info
2. Load daily_lineups to get pitcher names (via JSON fields)
3. Join on game_date + teams to match games across different data sources
4. Load custom_pitcher_2025 to get spin rate data
5. Match pitcher names (normalized to "Last, First" format)
6. Calculate matchup scores based on spin rates and prediction type
7. Save results to pitcher_matchup_scores.csv

Note: Currently all spin rates in custom_pitcher_2025 are NULL, so all matchups
default to 0.5 (Neutral). The script is ready to use once spin rate data is populated.
"""

import psycopg2
from psycopg2.extras import RealDictCursor
import pandas as pd
import logging
import os
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Database connection parameters
DB_PARAMS = {
    'host': '192.168.1.23',
    'port': 5432,
    'database': 'baseball_migration_test',
    'user': 'postgres',
    'password': 'korn5676'
}


def connect_db():
    """Create database connection"""
    try:
        conn = psycopg2.connect(**DB_PARAMS)
        logger.info(f"✅ Connected to database: {DB_PARAMS['database']}")
        return conn
    except Exception as e:
        logger.error(f"❌ Failed to connect to database: {e}")
        raise


def load_hellraiser_picks(conn):
    """Load hellraiser_picks data and join with games to get game_id and team info"""
    query = """
    SELECT
        g.game_id::text as game_id,
        g.game_date,
        g.home_team as game_home_team,
        g.away_team as game_away_team,
        hp.player_name,
        hp.game_description,
        hp.team,
        hp.is_home
    FROM hellraiser_picks hp
    LEFT JOIN games g ON
        hp.game_description = (g.away_team || ' @ ' || g.home_team) OR
        hp.game_description = (g.home_team || ' @ ' || g.away_team);
    """

    cursor = conn.cursor(cursor_factory=RealDictCursor)
    cursor.execute(query)
    results = cursor.fetchall()

    df = pd.DataFrame(results)
    logger.info(f"✅ Loaded {len(df)} hellraiser_picks records")
    logger.info(f"   Game IDs found: {df['game_id'].notna().sum()}/{len(df)}")
    return df


def load_custom_pitcher_2025(conn):
    """Load custom_pitcher_2025 data with spin rates"""
    query = """
    SELECT
        last_name_first_name,
        sl_avg_spin,
        ch_avg_spin,
        cu_avg_spin,
        si_avg_spin
    FROM custom_pitcher_2025;
    """

    cursor = conn.cursor(cursor_factory=RealDictCursor)
    cursor.execute(query)
    results = cursor.fetchall()

    df = pd.DataFrame(results)
    logger.info(f"✅ Loaded {len(df)} custom_pitcher_2025 records")
    return df


def load_daily_lineups(conn):
    """Load daily_lineups data and extract pitcher names"""
    query = """
    SELECT
        game_id,
        game_date,
        home_team,
        away_team,
        home_pitcher->>'name' as home_pitcher_name,
        away_pitcher->>'name' as away_pitcher_name
    FROM daily_lineups
    WHERE (home_pitcher->>'name') IS NOT NULL OR (away_pitcher->>'name') IS NOT NULL;
    """

    cursor = conn.cursor(cursor_factory=RealDictCursor)
    cursor.execute(query)
    results = cursor.fetchall()

    df = pd.DataFrame(results)
    logger.info(f"✅ Loaded {len(df)} daily_lineups records with pitcher info")
    return df


def normalize_pitcher_name(name):
    """
    Convert pitcher name from "First Last" to "Last, First" format
    Example: "Brady Singer" -> "Singer, Brady"
    """
    if pd.isna(name) or name is None:
        return None
    parts = name.strip().split()
    if len(parts) >= 2:
        return f"{parts[-1]}, {' '.join(parts[:-1])}"
    return name


def calculate_repertoire_match_score(df_picks, df_pitchers, df_lineups):
    """
    Calculate repertoire_match_score for each prediction

    Rules:
    - If prediction_type is 'Strikeout' and pitcher's sl_avg_spin is high (> 0.30), score = 0.0 (Bad)
    - If prediction_type is 'Home Run' and pitcher's ch_avg_spin is high (> 0.30), score = 0.0 (Bad)
    - Otherwise, score = 0.5 (Neutral)
    """
    logger.info("🔄 Calculating repertoire_match_score...")

    # Make copies to avoid SettingWithCopyWarning
    df_picks = df_picks.copy()
    df_lineups = df_lineups.copy()

    # Merge picks with lineups on game_date and team information
    df_merged = df_picks.merge(
        df_lineups,
        left_on=['game_date', 'game_home_team', 'game_away_team'],
        right_on=['game_date', 'home_team', 'away_team'],
        how='left',
        suffixes=('_pick', '_lineup')
    )

    # Determine which pitcher the batter is facing
    df_merged['pitcher_name'] = df_merged.apply(
        lambda row: row['away_pitcher_name'] if not row['is_home'] else row['home_pitcher_name'],
        axis=1
    )

    # Transform pitcher name to match custom_pitcher_2025 format (Last, First)
    df_merged['pitcher_name_normalized'] = df_merged['pitcher_name'].apply(normalize_pitcher_name)

    # Merge with pitcher data
    df_final = df_merged.merge(
        df_pitchers,
        left_on='pitcher_name_normalized',
        right_on='last_name_first_name',
        how='left'
    )

    # The game_id from picks gets renamed to game_id_pick during merge
    logger.info(f"   Found game_id (from picks): {df_final['game_id_pick'].notna().sum()}/{len(df_final)}")
    logger.info(f"   Found pitcher name: {df_final['pitcher_name'].notna().sum()}/{len(df_final)}")
    logger.info(f"   Matched to pitcher data: {df_final['sl_avg_spin'].notna().sum()}/{len(df_final)}")

    # Since hellraiser_picks doesn't have prediction_type, default to 'Home Run'
    # (hellraiser = HR prediction system)
    df_final['prediction_type'] = 'Home Run'

    # Initialize score column to 0.5 (Neutral)
    df_final['matchup_score'] = 0.5

    # Apply rules
    # Note: Spin rate data in custom_pitcher_2025 is currently NULL for all pitchers
    # All matchups will default to 0.5 (Neutral) until spin rate data is populated

    # Rule 1: Strikeout + high sl_avg_spin (> 0.30) = 0.0 (Bad)
    # (Not applicable since all predictions are 'Home Run' for hellraiser_picks)
    # mask_so_high_spin = (
    #     (df_final['prediction_type'] == 'Strikeout') &
    #     (df_final['sl_avg_spin'] > 0.30)
    # )
    # df_final.loc[mask_so_high_spin, 'matchup_score'] = 0.0
    # logger.info(f"   Rule 1 (SO + high sl_avg_spin): {mask_so_high_spin.sum()} bad matchups")

    # Rule 2: Home Run + high ch_avg_spin (> 0.30) = 0.0 (Bad)
    mask_hr_high_spin = (
        (df_final['prediction_type'] == 'Home Run') &
        (df_final['ch_avg_spin'] > 0.30)
    )
    df_final.loc[mask_hr_high_spin, 'matchup_score'] = 0.0
    logger.info(f"   Rule 2 (HR + high ch_avg_spin): {mask_hr_high_spin.sum()} bad matchups")

    # Count neutral scores
    mask_neutral = df_final['matchup_score'] == 0.5
    logger.info(f"   Neutral matchups: {mask_neutral.sum()}")

    # Return only the required columns
    result = df_final[['game_id_pick', 'player_name', 'pitcher_name', 'matchup_score']].copy()
    result = result.rename(columns={'game_id_pick': 'game_id'})

    # Log score distribution
    score_dist = result['matchup_score'].value_counts().sort_index()
    logger.info(f"\n📊 Score Distribution:")
    for score, count in score_dist.items():
        logger.info(f"   {score:.2f}: {count} ({count/len(result)*100:.1f}%)")

    return result


def save_results(df, output_path):
    """Save results to CSV"""
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df.to_csv(output_path, index=False)
    logger.info(f"✅ Results saved to: {output_path}")
    logger.info(f"   Total records: {len(df)}")


def main():
    """Main execution function"""
    logger.info("=" * 60)
    logger.info("Pitcher Matchup Feature Generation")
    logger.info("=" * 60)

    conn = None
    try:
        # Connect to database
        conn = connect_db()

        # Load data
        df_picks = load_hellraiser_picks(conn)
        df_pitchers = load_custom_pitcher_2025(conn)
        df_lineups = load_daily_lineups(conn)

        # Calculate matchup scores
        df_scores = calculate_repertoire_match_score(df_picks, df_pitchers, df_lineups)

        # Save results
        output_path = '/home/futurepr0n/Development/ProjectionAI/scripts/pitcher_matchup_scores.csv'
        save_results(df_scores, output_path)

        # Display sample results
        logger.info("\n📋 Sample Results (first 10):")
        print(df_scores.head(10).to_string(index=False))

        # Summary statistics
        logger.info("\n📊 Summary Statistics:")
        print(df_scores['matchup_score'].describe())

        logger.info("\n✅ Script completed successfully!")

    except Exception as e:
        logger.error(f"❌ Error during execution: {e}")
        raise
    finally:
        if conn:
            conn.close()
            logger.info("✅ Database connection closed")


if __name__ == "__main__":
    main()
