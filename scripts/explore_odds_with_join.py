#!/usr/bin/env python3
"""
Explore odds with proper game join
"""

import psycopg2
from psycopg2.extras import RealDictCursor
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def connect_db():
    return psycopg2.connect(
        host='192.168.1.23',
        port=5432,
        database='baseball_migration_test',
        user='postgres',
        password='korn5676'
    )


def explore_odds_joined_with_games(conn):
    """Get odds joined with games to get game dates"""
    query = """
    SELECT
        g.game_date,
        COUNT(DISTINCT ot.id) as odds_count,
        COUNT(DISTINCT ot.player_id) as unique_players,
        COUNT(DISTINCT ot.game_id) as unique_games
    FROM odds_tracking ot
    JOIN games g ON ot.game_id = g.game_id
    WHERE g.game_date IS NOT NULL
    GROUP BY g.game_date
    ORDER BY g.game_date DESC
    LIMIT 20;
    """

    cursor = conn.cursor()
    cursor.execute(query)
    results = cursor.fetchall()

    logger.info(f"\n📊 Odds Per Game Date (last 20 dates):")
    logger.info(f"   {'Date':<12} {'Odds':>8} {'Players':>10} {'Games':>8}")
    logger.info(f"   {'-'*45}")
    for row in results:
        logger.info(f"   {str(row[0]):<12} {row[1]:>8,} {row[2]:>10,} {row[3]:>8,}")


def explore_hellraiser_schema(conn):
    """Explore hellraiser_picks table schema"""
    query = """
    SELECT column_name, data_type
    FROM information_schema.columns
    WHERE table_name = 'hellraiser_picks' AND table_schema = 'public'
    ORDER BY ordinal_position;
    """

    cursor = conn.cursor()
    cursor.execute(query)
    columns = cursor.fetchall()

    logger.info(f"\n📋 Hellraiser Picks Schema:")
    for col_name, data_type in columns:
        logger.info(f"   - {col_name}: {data_type}")


def explore_hellraiser_picks(conn):
    """Explore hellraiser picks properly"""
    query = """
    SELECT
        COUNT(*) as total_picks,
        COUNT(DISTINCT game_time) as unique_dates,
        MIN(game_time) as earliest_date,
        MAX(game_time) as latest_date
    FROM hellraiser_picks;
    """

    cursor = conn.cursor()
    cursor.execute(query)
    result = cursor.fetchone()

    logger.info(f"\n🎯 Hellraiser Picks Summary:")
    logger.info(f"   Total picks: {result[0]:,}")
    logger.info(f"   Unique dates: {result[1]:,}")
    logger.info(f"   Date range: {result[2]} to {result[3]}")

    # Get sample picks
    sample_query = """
    SELECT
        game_time,
        player_name,
        team,
        opponent,
        prediction_signal,
        confidence_score,
        actual_hr_hit
    FROM hellraiser_picks
    LIMIT 10;
    """

    cursor.execute(sample_query)
    samples = cursor.fetchall()

    logger.info(f"\n📋 Sample Picks:")
    logger.info(f"   {'Date':<12} {'Player':<20} {'Signal':<10} {'Confidence':>10} {'Hit':>5}")
    logger.info(f"   {'-'*70}")
    for row in samples:
        logger.info(f"   {str(row[0]):<12} {str(row[1])[:20]:<20} {str(row[4]):<10} {row[5]:>10} {row[6]:>5}")


def analyze_odds_data_quality(conn):
    """Analyze odds data quality"""
    queries = [
        ("Total odds", "SELECT COUNT(*) FROM odds_tracking"),
        ("Odds with game_id", "SELECT COUNT(*) FROM odds_tracking WHERE game_id IS NOT NULL"),
        ("Odds with player_id", "SELECT COUNT(*) FROM odds_tracking WHERE player_id IS NOT NULL"),
        ("Odds with prop_line", "SELECT COUNT(*) FROM odds_tracking WHERE prop_line IS NOT NULL"),
        ("Odds with over_odds", "SELECT COUNT(*) FROM odds_tracking WHERE over_odds IS NOT NULL"),
        ("Odds with under_odds", "SELECT COUNT(*) FROM odds_tracking WHERE under_odds IS NOT NULL"),
        ("Opening odds", "SELECT COUNT(*) FROM odds_tracking WHERE opening_odds = TRUE"),
        ("Current odds", "SELECT COUNT(*) FROM odds_tracking WHERE is_current = TRUE"),
    ]

    logger.info(f"\n📊 Odds Data Quality:")
    for name, query in queries:
        cursor = conn.cursor()
        cursor.execute(query)
        count = cursor.fetchone()[0]
        logger.info(f"   {name:<25} {count:>12,}")


def explore_games_date_range(conn):
    """Get game date range"""
    query = """
    SELECT
        MIN(game_date) as earliest_game,
        MAX(game_date) as latest_game,
        COUNT(*) as total_games,
        COUNT(DISTINCT game_date) as unique_dates
    FROM games;
    """

    cursor = conn.cursor()
    cursor.execute(query)
    result = cursor.fetchone()

    logger.info(f"\n📅 Games Summary:")
    logger.info(f"   Earliest game: {result[0]}")
    logger.info(f"   Latest game: {result[1]}")
    logger.info(f"   Total games: {result[2]:,}")
    logger.info(f"   Unique dates: {result[3]:,}")


def get_sample_complete_odds(conn):
    """Get sample of complete odds data"""
    query = """
    SELECT
        g.game_date,
        g.home_team,
        g.away_team,
        ot.player_name,
        ot.team,
        ot.opponent,
        ot.prop_type,
        ot.prop_line,
        ot.over_odds,
        ot.under_odds,
        ot.opening_odds,
        lm.movement_direction,
        lm.sharp_money_indicator
    FROM odds_tracking ot
    JOIN games g ON ot.game_id = g.game_id
    LEFT JOIN line_movement lm ON ot.game_id = lm.game_id AND ot.player_id = lm.player_id
    WHERE g.game_date IS NOT NULL
      AND ot.prop_line IS NOT NULL
      AND ot.over_odds IS NOT NULL
    ORDER BY g.game_date DESC
    LIMIT 5;
    """

    cursor = conn.cursor(cursor_factory=RealDictCursor)
    cursor.execute(query)
    results = cursor.fetchall()

    logger.info(f"\n📋 Sample Complete Odds:")
    for i, row in enumerate(results, 1):
        logger.info(f"\n   Sample {i}:")
        logger.info(f"     Date: {row['game_date']}")
        logger.info(f"     Game: {row['away_team']} @ {row['home_team']}")
        logger.info(f"     Player: {row['player_name']} ({row['team']}) vs {row['opponent']}")
        logger.info(f"     Bet: {row['prop_type'].upper()} {row['prop_line']} - Over {row['over_odds']} / Under {row['under_odds']}")
        logger.info(f"     Opening: {row['opening_odds']}")
        logger.info(f"     Movement: {row.get('movement_direction', 'N/A')}")
        logger.info(f"     Sharp money: {row.get('sharp_money_indicator', 'N/A')}")


def main():
    conn = connect_db()

    analyze_odds_data_quality(conn)
    explore_games_date_range(conn)
    explore_odds_joined_with_games(conn)
    explore_hellraiser_schema(conn)
    explore_hellraiser_picks(conn)
    get_sample_complete_odds(conn)

    conn.close()
    logger.info("\n✅ Complete!")


if __name__ == "__main__":
    main()
