#!/usr/bin/env python3
"""
Explore pitcher data availability
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


def explore_pitcher_tables(conn):
    """Explore all pitcher-related tables"""
    pitcher_tables = [
        'pitcherarsenalstats_2025',
        'pitching_stats',
        'custom_pitcher_2025',
        'pitcher_matchups'
    ]

    for table in pitcher_tables:
        query = f"SELECT COUNT(*) FROM {table};"
        try:
            cursor = conn.cursor()
            cursor.execute(query)
            count = cursor.fetchone()[0]

            # Get schema
            schema_query = f"""
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_name = '{table}' AND table_schema = 'public'
            LIMIT 20;
            """
            cursor.execute(schema_query)
            columns = cursor.fetchall()

            logger.info(f"\n📊 {table}: {count:,} rows")
            logger.info(f"   Columns: {', '.join([c[0] for c in columns[:12]])}...")
        except Exception as e:
            logger.error(f"   ❌ Error: {e}")


def sample_pitcher_stats(conn):
    """Sample pitcher stats table"""
    query = """
    SELECT
        ps.game_id,
        ps.player_name,
        ps.team,
        ps.innings_pitched,
        ps.hits,
        ps.runs,
        ps.earned_runs,
        ps.walks,
        ps.strikeouts,
        g.game_date,
        g.home_team,
        g.away_team
    FROM pitching_stats ps
    JOIN games g ON ps.game_id = g.game_id
    LIMIT 10;
    """

    cursor = conn.cursor(cursor_factory=RealDictCursor)
    cursor.execute(query)
    results = cursor.fetchall()

    logger.info(f"\n📋 Sample Pitching Stats:")
    for i, row in enumerate(results, 1):
        logger.info(f"\n   Sample {i}:")
        logger.info(f"     Game ID: {row['game_id']}")
        logger.info(f"     Date: {row['game_date']}")
        logger.info(f"     Game: {row['away_team']} @ {row['home_team']}")
        logger.info(f"     Pitcher: {row['player_name']} ({row['team']})")
        logger.info(f"     IP: {row['innings_pitched']}, H: {row['hits']}, R: {row['runs']}, ER: {row['earned_runs']}")
        logger.info(f"     BB: {row['walks']}, K: {row['strikeouts']}")


def join_hellraiser_with_pitching_stats(conn):
    """Test joining hellraiser picks with pitching stats via games"""
    query = """
    SELECT
        COUNT(DISTINCT hp.id) as hellraiser_picks,
        COUNT(DISTINCT g.game_id) as games_with_stats,
        COUNT(DISTINCT ps.game_id) as pitcher_stats,
        COUNT(DISTINCT ps.game_id) as matched
    FROM hellraiser_picks hp
    LEFT JOIN games g ON
        hp.game_description = (g.away_team || ' @ ' || g.home_team) OR
        hp.game_description = (g.home_team || ' @ ' || g.away_team)
    LEFT JOIN pitching_stats ps ON g.game_id = ps.game_id;
    """

    cursor = conn.cursor()
    cursor.execute(query)
    result = cursor.fetchone()

    logger.info(f"\n🔗 Hellraiser + Pitching Stats Join:")
    logger.info(f"   Hellraiser picks: {result[0]:,}")
    logger.info(f"   Games found: {result[1]:,}")
    logger.info(f"   Pitcher stats available: {result[2]:,}")
    logger.info(f"   Matched: {result[3]:,}")


def analyze_pitcher_stat_coverage(conn):
    """Analyze how many games have pitcher stats"""
    query = """
    SELECT
        g.game_date,
        COUNT(DISTINCT g.game_id) as total_games,
        COUNT(DISTINCT ps.game_id) as games_with_pitcher_stats,
        ROUND(COUNT(DISTINCT ps.game_id)::NUMERIC / COUNT(DISTINCT g.game_id) * 100, 1) as coverage_percent
    FROM games g
    LEFT JOIN pitching_stats ps ON g.game_id = ps.game_id
    GROUP BY g.game_date
    ORDER BY g.game_date DESC
    LIMIT 20;
    """

    cursor = conn.cursor()
    cursor.execute(query)
    results = cursor.fetchall()

    logger.info(f"\n📊 Pitcher Stats Coverage by Date (last 20):")
    logger.info(f"   {'Date':<12} {'Games':>8} {'With Stats':>12} {'Coverage':>10}")
    logger.info(f"   {'-'*45}")
    for row in results:
        logger.info(f"   {str(row[0]):<12} {row[1]:>8} {row[2]:>12} {row[3]:>10}%")


def check_pitcher_name_matching(conn):
    """Check if pitcher names match between tables"""
    query = """
    SELECT
        COUNT(DISTINCT hp.pitcher_name) as hellraiser_pitchers,
        COUNT(DISTINCT ps.player_name) as pitching_stats_pitchers,
        COUNT(DISTINCT CASE
            WHEN ps.player_name IN (SELECT DISTINCT pitcher_name FROM hellraiser_picks)
            THEN ps.player_name
            ELSE NULL
        END) as matched_names
    FROM hellraiser_picks hp
    CROSS JOIN pitching_stats ps;
    """

    cursor = conn.cursor()
    cursor.execute(query)
    result = cursor.fetchone()

    logger.info(f"\n🏷️  Pitcher Name Matching:")
    logger.info(f"   Hellraiser unique pitchers: {result[0]:,}")
    logger.info(f"   Pitching stats unique pitchers: {result[1]:,}")
    logger.info(f"   Name matches: {result[2]:,}")


def derive_pitcher_metrics_from_pitch_by_pitch(conn):
    """Check if we can derive metrics from play_by_play_pitches"""
    query = """
    SELECT
        COUNT(*) as total_pitches,
        COUNT(DISTINCT pitcher) as unique_pitchers,
        COUNT(DISTINCT result) as unique_results
    FROM play_by_play_plays
    WHERE pitcher IS NOT NULL;
    """

    cursor = conn.cursor()
    cursor.execute(query)
    result = cursor.fetchone()

    logger.info(f"\n🎯 Play-By-Play Pitcher Data:")
    logger.info(f"   Total plays: {result[0]:,}")
    logger.info(f"   Unique pitchers: {result[1]:,}")
    logger.info(f"   Unique results: {result[2]:,}")

    # Get sample results
    results_query = """
    SELECT DISTINCT result
    FROM play_by_play_plays
    WHERE result IS NOT NULL
    ORDER BY result;
    """

    cursor.execute(results_query)
    results = cursor.fetchall()

    logger.info(f"   Result types: {', '.join([r[0] for r in results[:20]])}...")


def main():
    conn = connect_db()

    explore_pitcher_tables(conn)
    sample_pitcher_stats(conn)
    join_hellraiser_with_pitching_stats(conn)
    analyze_pitcher_stat_coverage(conn)
    check_pitcher_name_matching(conn)
    derive_pitcher_metrics_from_pitch_by_pitch(conn)

    conn.close()
    logger.info("\n✅ Complete!")


if __name__ == "__main__":
    main()
