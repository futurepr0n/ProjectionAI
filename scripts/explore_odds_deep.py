#!/usr/bin/env python3
"""
Deep dive into odds tracking data
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


def explore_odds_with_game_dates(conn):
    """Get odds that have actual game dates"""
    query = """
    SELECT
        COUNT(*) as total_odds,
        COUNT(game_date) as with_game_date,
        COUNT(*) - COUNT(game_date) as without_game_date
    FROM odds_tracking;
    """

    cursor = conn.cursor()
    cursor.execute(query)
    result = cursor.fetchone()

    logger.info(f"\n📊 Odds Tracking Summary:")
    logger.info(f"   Total odds: {result[0]:,}")
    logger.info(f"   With game date: {result[1]:,}")
    logger.info(f"   Without game date: {result[2]:,}")


def explore_odds_by_prop_type(conn):
    """Get breakdown by prop type"""
    query = """
    SELECT
        prop_type,
        COUNT(*) as count,
        COUNT(DISTINCT player_name) as unique_players,
        COUNT(DISTINCT game_id) as unique_games
    FROM odds_tracking
    GROUP BY prop_type
    ORDER BY count DESC;
    """

    cursor = conn.cursor()
    cursor.execute(query)
    results = cursor.fetchall()

    logger.info(f"\n📋 Odds by Prop Type:")
    logger.info(f"   {'Type':<15} {'Count':>12} {'Players':>12} {'Games':>12}")
    logger.info(f"   {'-'*60}")
    for prop_type, count, players, games in results:
        logger.info(f"   {prop_type:<15} {count:>12,} {players:>12,} {games:>12,}")


def explore_complete_odds(conn):
    """Get odds with complete data (game_date, player_id, line, odds)"""
    query = """
    SELECT
        COUNT(*) as complete_odds,
        COUNT(DISTINCT game_date) as unique_dates,
        MIN(game_date) as earliest_date,
        MAX(game_date) as latest_date
    FROM odds_tracking
    WHERE game_date IS NOT NULL
      AND player_id IS NOT NULL
      AND prop_line IS NOT NULL
      AND (over_odds IS NOT NULL OR under_odds IS NOT NULL);
    """

    cursor = conn.cursor()
    cursor.execute(query)
    result = cursor.fetchone()

    logger.info(f"\n✅ Complete Odds (has game_date, player_id, line, odds):")
    logger.info(f"   Count: {result[0]:,}")
    logger.info(f"   Unique dates: {result[1]:,}")
    logger.info(f"   Date range: {result[2]} to {result[3]}")


def explore_player_stats(conn):
    """Explore player stats tables"""
    tables_to_check = [
        'hitting_stats',
        'pitching_stats',
        'player_season_stats',
        'custom_batter_2025',
        'custom_pitcher_2025',
        'rolling_stats_30_day',
        'rolling_stats_7_day'
    ]

    logger.info(f"\n📊 Player Statistics Tables:")

    for table in tables_to_check:
        try:
            query = f"SELECT COUNT(*) FROM {table};"
            cursor = conn.cursor()
            cursor.execute(query)
            count = cursor.fetchone()[0]

            # Get sample schema
            schema_query = f"""
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_name = '{table}' AND table_schema = 'public'
            LIMIT 15;
            """
            cursor.execute(schema_query)
            columns = cursor.fetchall()

            logger.info(f"\n   📋 {table}: {count:,} rows")
            logger.info(f"      Columns: {', '.join([c[0] for c in columns[:8]])}...")
        except Exception as e:
            logger.error(f"   ❌ Error checking {table}: {e}")


def explore_hellraiser_picks(conn):
    """Explore existing HR predictions"""
    query = """
    SELECT
        COUNT(*) as total_picks,
        COUNT(DISTINCT game_id) as unique_games,
        COUNT(DISTINCT run_date) as unique_dates,
        MIN(run_date) as earliest_date,
        MAX(run_date) as latest_date
    FROM hellraiser_picks;
    """

    try:
        cursor = conn.cursor()
        cursor.execute(query)
        result = cursor.fetchone()

        logger.info(f"\n🎯 Hellraiser Picks (Existing HR predictions):")
        logger.info(f"   Total picks: {result[0]:,}")
        logger.info(f"   Unique games: {result[1]:,}")
        logger.info(f"   Date range: {result[3]} to {result[4]}")
    except Exception as e:
        logger.error(f"   ❌ Error checking hellraiser_picks: {e}")


def sample_odds_data(conn):
    """Get sample odds data with game info"""
    query = """
    SELECT
        ot.game_date,
        ot.player_name,
        ot.prop_type,
        ot.prop_line,
        ot.over_odds,
        ot.under_odds,
        ot.opening_odds,
        g.home_team,
        g.away_team
    FROM odds_tracking ot
    LEFT JOIN games g ON ot.game_id = g.game_id
    WHERE ot.game_date IS NOT NULL
      AND ot.prop_line IS NOT NULL
    ORDER BY ot.game_date DESC
    LIMIT 10;
    """

    try:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute(query)
        results = cursor.fetchall()

        logger.info(f"\n📋 Sample Odds with Game Info:")
        for i, row in enumerate(results, 1):
            logger.info(f"\n   Sample {i}:")
            logger.info(f"     Date: {row['game_date']}")
            logger.info(f"     Player: {row['player_name']}")
            logger.info(f"     Type: {row['prop_type']} @ {row['prop_line']}")
            logger.info(f"     Odds: O{row['over_odds']} / U{row['under_odds']}")
            logger.info(f"     Game: {row['away_team']} @ {row['home_team']}")
    except Exception as e:
        logger.error(f"   ❌ Error sampling odds: {e}")


def explore_line_movement(conn):
    """Explore line movement data"""
    query = """
    SELECT
        COUNT(*) as total_movements,
        COUNT(DISTINCT game_date) as unique_dates,
        COUNT(DISTINCT player_id) as unique_players,
        COUNT(is_steam_move) as steam_moves,
        COUNT(sharp_money_indicator) as sharp_money_moves,
        COUNT(reverse_line_movement) as reverse_moves
    FROM line_movement;
    """

    cursor = conn.cursor()
    cursor.execute(query)
    result = cursor.fetchone()

    logger.info(f"\n📈 Line Movement:")
    logger.info(f"   Total movements: {result[0]:,}")
    logger.info(f"   Unique dates: {result[1]:,}")
    logger.info(f"   Steam moves: {result[3]}")
    logger.info(f"   Sharp money: {result[4]}")
    logger.info(f"   Reverse moves: {result[5]}")


def main():
    conn = connect_db()

    explore_odds_with_game_dates(conn)
    explore_odds_by_prop_type(conn)
    explore_complete_odds(conn)
    explore_player_stats(conn)
    explore_hellraiser_picks(conn)
    explore_line_movement(conn)
    sample_odds_data(conn)

    conn.close()
    logger.info("\n✅ Deep dive complete!")


if __name__ == "__main__":
    main()
