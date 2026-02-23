#!/usr/bin/env python3
"""
ProjectionAI - Database Exploration Script
Explore the remote database schema and available data
"""

import psycopg2
from psycopg2.extras import RealDictCursor
import pandas as pd
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def connect_to_db():
    """Connect to remote database"""
    try:
        conn = psycopg2.connect(
            host='192.168.1.23',
            port=5432,
            database='baseball_migration_test',
            user='postgres',
            password='korn5676'
        )
        logger.info("✅ Connected to remote database")
        return conn
    except Exception as e:
        logger.error(f"❌ Failed to connect: {e}")
        return None


def explore_tables(conn):
    """Get all tables in the database"""
    query = """
    SELECT table_name
    FROM information_schema.tables
    WHERE table_schema = 'public'
    ORDER BY table_name;
    """

    try:
        cursor = conn.cursor()
        cursor.execute(query)
        tables = cursor.fetchall()
        logger.info(f"\n📋 Found {len(tables)} tables:")
        for i, (table,) in enumerate(tables, 1):
            print(f"  {i}. {table}")
        return [t[0] for t in tables]
    except Exception as e:
        logger.error(f"❌ Error exploring tables: {e}")
        return []


def explore_table_schema(conn, table_name):
    """Get schema for a specific table"""
    query = """
    SELECT column_name, data_type, is_nullable, column_default
    FROM information_schema.columns
    WHERE table_name = %s AND table_schema = 'public'
    ORDER BY ordinal_position;
    """

    try:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute(query, (table_name,))
        columns = cursor.fetchall()
        return columns
    except Exception as e:
        logger.error(f"❌ Error exploring schema for {table_name}: {e}")
        return []


def get_table_row_count(conn, table_name):
    """Get row count for a table"""
    query = f"SELECT COUNT(*) FROM {table_name};"

    try:
        cursor = conn.cursor()
        cursor.execute(query)
        count = cursor.fetchone()[0]
        return count
    except Exception as e:
        logger.error(f"❌ Error counting rows in {table_name}: {e}")
        return 0


def explore_play_by_play_pitches(conn):
    """Explore play_by_play_pitches table"""
    table_name = 'play_by_play_pitches'

    logger.info(f"\n{'='*60}")
    logger.info(f"🎯 Exploring: {table_name}")
    logger.info(f"{'='*60}")

    # Get schema
    schema = explore_table_schema(conn, table_name)
    if schema:
        logger.info(f"\n📊 Schema ({len(schema)} columns):")
        for col in schema[:20]:  # Show first 20 columns
            logger.info(f"  - {col['column_name']}: {col['data_type']}")
        if len(schema) > 20:
            logger.info(f"  ... and {len(schema) - 20} more columns")

    # Get row count
    count = get_table_row_count(conn, table_name)
    logger.info(f"\n📈 Total rows: {count:,}")

    # Sample data
    sample_query = f"SELECT * FROM {table_name} LIMIT 3;"
    try:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute(sample_query)
        samples = cursor.fetchall()
        if samples:
            logger.info(f"\n📋 Sample data (3 rows):")
            for i, row in enumerate(samples, 1):
                logger.info(f"\n  Row {i}:")
                for key, value in list(row.items())[:10]:  # Show first 10 columns
                    logger.info(f"    {key}: {value}")
                if len(row) > 10:
                    logger.info(f"    ... and {len(row) - 10} more fields")
    except Exception as e:
        logger.error(f"❌ Error sampling data: {e}")


def explore_play_by_play_plays(conn):
    """Explore play_by_play_plays table"""
    table_name = 'play_by_play_plays'

    logger.info(f"\n{'='*60}")
    logger.info(f"🎯 Exploring: {table_name}")
    logger.info(f"{'='*60}")

    # Get schema
    schema = explore_table_schema(conn, table_name)
    if schema:
        logger.info(f"\n📊 Schema ({len(schema)} columns):")
        for col in schema[:20]:
            logger.info(f"  - {col['column_name']}: {col['data_type']}")
        if len(schema) > 20:
            logger.info(f"  ... and {len(schema) - 20} more columns")

    # Get row count
    count = get_table_row_count(conn, table_name)
    logger.info(f"\n📈 Total rows: {count:,}")


def explore_odds_tracking(conn):
    """Explore odds_tracking table"""
    table_name = 'odds_tracking'

    logger.info(f"\n{'='*60}")
    logger.info(f"🎯 Exploring: {table_name}")
    logger.info(f"{'='*60}")

    # Get schema
    schema = explore_table_schema(conn, table_name)
    if schema:
        logger.info(f"\n📊 Schema ({len(schema)} columns):")
        for col in schema:
            logger.info(f"  - {col['column_name']}: {col['data_type']}")

    # Get row count
    count = get_table_row_count(conn, table_name)
    logger.info(f"\n📈 Total rows: {count:,}")

    # Sample data
    sample_query = f"SELECT * FROM {table_name} LIMIT 5;"
    try:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute(sample_query)
        samples = cursor.fetchall()
        if samples:
            logger.info(f"\n📋 Sample data (5 rows):")
            for i, row in enumerate(samples, 1):
                logger.info(f"\n  Row {i}:")
                for key, value in row.items():
                    logger.info(f"    {key}: {value}")
    except Exception as e:
        logger.error(f"❌ Error sampling data: {e}")


def explore_line_movement(conn):
    """Explore line_movement table"""
    table_name = 'line_movement'

    logger.info(f"\n{'='*60}")
    logger.info(f"🎯 Exploring: {table_name}")
    logger.info(f"{'='*60}")

    # Get schema
    schema = explore_table_schema(conn, table_name)
    if schema:
        logger.info(f"\n📊 Schema ({len(schema)} columns):")
        for col in schema:
            logger.info(f"  - {col['column_name']}: {col['data_type']}")

    # Get row count
    count = get_table_row_count(conn, table_name)
    logger.info(f"\n📈 Total rows: {count:,}")


def explore_game_stats(conn):
    """Look for game statistics tables"""
    # Common table names for game stats
    possible_tables = [
        'game_stats',
        'games',
        'team_stats',
        'player_stats',
        'batting_stats',
        'pitching_stats'
    ]

    query = """
    SELECT table_name
    FROM information_schema.tables
    WHERE table_schema = 'public'
    AND table_name LIKE ANY(%s)
    ORDER BY table_name;
    """

    try:
        cursor = conn.cursor()
        cursor.execute(query, (possible_tables,))
        found_tables = cursor.fetchall()

        if found_tables:
            logger.info(f"\n{'='*60}")
            logger.info(f"🎯 Game Statistics Tables")
            logger.info(f"{'='*60}")

            for (table,) in found_tables:
                logger.info(f"\n📊 Table: {table}")
                count = get_table_row_count(conn, table)
                logger.info(f"   Rows: {count:,}")

                # Get schema
                schema = explore_table_schema(conn, table)
                if schema:
                    logger.info(f"   Columns: {len(schema)}")
                    for col in schema[:10]:
                        logger.info(f"     - {col['column_name']}: {col['data_type']}")
    except Exception as e:
        logger.error(f"❌ Error exploring game stats: {e}")


def analyze_season_coverage(conn):
    """Analyze season date range"""
    # Try different tables for date info
    date_queries = [
        ("play_by_play_pitches", "SELECT MIN(game_date), MAX(game_date), COUNT(DISTINCT game_date) FROM play_by_play_pitches"),
        ("play_by_play_plays", "SELECT MIN(game_date), MAX(game_date), COUNT(DISTINCT game_date) FROM play_by_play_plays"),
        ("odds_tracking", "SELECT MIN(game_date), MAX(game_date), COUNT(DISTINCT game_date) FROM odds_tracking"),
    ]

    logger.info(f"\n{'='*60}")
    logger.info(f"📅 Season Coverage Analysis")
    logger.info(f"{'='*60}")

    for table, query in date_queries:
        try:
            cursor = conn.cursor()
            cursor.execute(query)
            min_date, max_date, unique_dates = cursor.fetchone()

            if min_date and max_date:
                logger.info(f"\n📊 {table}:")
                logger.info(f"   Date range: {min_date} to {max_date}")
                logger.info(f"   Unique dates: {unique_dates}")
        except Exception as e:
            logger.error(f"❌ Error analyzing {table}: {e}")


def main():
    """Main exploration function"""
    logger.info("🚀 Starting database exploration...")
    logger.info("="*60)

    # Connect to database
    conn = connect_to_db()
    if not conn:
        return

    try:
        # Get all tables
        tables = explore_tables(conn)

        # Explore key tables
        explore_play_by_play_pitches(conn)
        explore_play_by_play_plays(conn)
        explore_odds_tracking(conn)
        explore_line_movement(conn)
        explore_game_stats(conn)

        # Analyze season coverage
        analyze_season_coverage(conn)

        # Save summary
        logger.info(f"\n{'='*60}")
        logger.info("✅ Exploration complete!")
        logger.info(f"{'='*60}")

    finally:
        conn.close()
        logger.info("\n🔌 Database connection closed")


if __name__ == "__main__":
    main()
