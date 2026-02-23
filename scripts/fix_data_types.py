#!/usr/bin/env python3
"""
Fix Data Types in Baseball Tables

This script fixes TEXT type issues in custom_pitcher_2025 and hitter_exit_velocity
by creating proper numeric columns and migrating data.

Run this ONCE to migrate the data, then the feature scripts can use numeric columns.
"""

import psycopg2
from psycopg2.extras import RealDictCursor
import pandas as pd
import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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
        logger.info(f"✅ Connected to database")
        return conn
    except Exception as e:
        logger.error(f"❌ Failed to connect: {e}")
        raise


def fix_pitcher_table(conn):
    """Fix TEXT columns to NUMERIC in custom_pitcher_2025"""

    # Only fix the columns we actually need for features
    columns_to_fix = {
        'sl_avg_spin': 'INTEGER',
        'ch_avg_spin': 'INTEGER',
        'cu_avg_spin': 'INTEGER',
        'si_avg_spin': 'INTEGER',
        'fc_avg_spin': 'INTEGER',
        'kn_avg_spin': 'INTEGER',
        'st_avg_spin': 'INTEGER',
        'fo_avg_spin': 'INTEGER',
        'sc_avg_spin': 'INTEGER',
        'fastball_avg_spin': 'INTEGER',
        'breaking_avg_spin': 'INTEGER',
        'offspeed_avg_spin': 'INTEGER',
        'k_percent': 'NUMERIC(5,2)',
        'bb_percent': 'NUMERIC(5,2)',
        'p_era': 'NUMERIC(5,2)',
    }

    cursor = conn.cursor()

    # Check actual column types
    logger.info("📋 Checking actual column types in custom_pitcher_2025...")
    cursor.execute("""
        SELECT column_name, data_type
        FROM information_schema.columns
        WHERE table_name='custom_pitcher_2025'
          AND column_name = ANY(%s)
        ORDER BY ordinal_position;
    """, (list(columns_to_fix.keys()),))
    column_types = {row[0]: row[1] for row in cursor.fetchall()}

    logger.info(f"📋 Found {len(column_types)} columns to check")

    logger.info("🔧 Fixing custom_pitcher_2025 data types...")

    for col, target_type in columns_to_fix.items():
        logger.info(f"   Processing {col}...")

        # Check if column exists and its type
        if col not in column_types:
            logger.info(f"   ⏭️ {col} does not exist, skipping")
            continue

        current_type = column_types[col]
        if current_type in ('integer', 'bigint', 'numeric', 'double precision'):
            logger.info(f"   ⏭️ {col} is already {current_type}, skipping")
            continue

        # Check if numeric column already exists
        check_query = f"""
            SELECT EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_name='custom_pitcher_2025'
                AND column_name='{col}_numeric'
            );
        """
        cursor.execute(check_query)
        exists = cursor.fetchone()[0]

        if not exists:
            # Create numeric column
            alter_query = f"""
                ALTER TABLE custom_pitcher_2025
                ADD COLUMN IF NOT EXISTS {col}_numeric {target_type};
            """
            cursor.execute(alter_query)

            # Migrate data - spin rates are decimals like "2452.0"
            migrate_query = f"""
                UPDATE custom_pitcher_2025
                SET {col}_numeric = NULLIF({col}::numeric, 0)
                WHERE {col}_numeric IS NULL
                  AND {col} IS NOT NULL
                  AND {col} ~ '^[0-9]+(\.[0-9]+)?$';
            """

            cursor.execute(migrate_query)

            logger.info(f"   ✅ {col}: {cursor.rowcount} records migrated")

    conn.commit()
    logger.info("✅ custom_pitcher_2025 data types fixed")


def fix_ev_table(conn):
    """Fix TEXT columns to NUMERIC in hitter_exit_velocity"""

    columns_to_fix = [
        'avg_hit_angle',
        'max_hit_speed',
        'avg_hit_speed',
        'avg_distance',
        'avg_hr_distance',
        'ev95percent',
        'anglesweetspotpercent',
        'brl_percent'
    ]

    cursor = conn.cursor()

    logger.info("🔧 Fixing hitter_exit_velocity data types...")

    for col in columns_to_fix:
        logger.info(f"   Processing {col}...")

        # Check if column exists
        check_exists = f"""
            SELECT EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_name='hitter_exit_velocity'
                AND column_name='{col}'
            );
        """
        cursor.execute(check_exists)
        exists = cursor.fetchone()[0]

        if not exists:
            logger.info(f"   ⏭️ {col} does not exist, skipping")
            continue

        # Check if numeric column already exists
        check_numeric = f"""
            SELECT EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_name='hitter_exit_velocity'
                AND column_name='{col}_numeric'
            );
        """
        cursor.execute(check_numeric)
        has_numeric = cursor.fetchone()[0]

        if not has_numeric:
            # Create numeric column
            alter_query = f"""
                ALTER TABLE hitter_exit_velocity
                ADD COLUMN IF NOT EXISTS {col}_numeric NUMERIC(10,2);
            """
            cursor.execute(alter_query)

            # Migrate data - handle both numeric and percent formats
            migrate_query = f"""
                UPDATE hitter_exit_velocity
                SET {col}_numeric = NULLIF(
                    CASE
                        WHEN {col} ~ '^[0-9]+\.[0-9]+%%$' THEN REPLACE({col}, '%%', '')::numeric
                        WHEN {col} ~ '^[0-9]+%%$' THEN REPLACE({col}, '%%', '')::numeric
                        ELSE {col}::numeric
                    END,
                    0
                )
                WHERE {col}_numeric IS NULL
                  AND {col} IS NOT NULL
                  AND ({col} ~ '^[0-9]+(\.[0-9]+)?%%?$' OR {col} ~ '^[0-9]+\.[0-9]+$' OR {col} ~ '^[0-9]+$');
            """

            cursor.execute(migrate_query)

            logger.info(f"   ✅ {col}: {cursor.rowcount} records migrated")

    conn.commit()
    logger.info("✅ hitter_exit_velocity data types fixed")


def create_migration_summary(conn):
    """Generate summary of migrated data"""
    cursor = conn.cursor()

    logger.info("\n📊 Migration Summary:\n")

    # Pitcher table summary
    cursor.execute("""
        SELECT
            COUNT(*) as total,
            COUNT(sl_avg_spin_numeric) as spin_migrated,
            COUNT(k_percent_numeric) as k_percent_migrated,
            COUNT(p_era_numeric) as era_migrated
        FROM custom_pitcher_2025;
    """)
    pitcher_stats = cursor.fetchone()

    logger.info("custom_pitcher_2025:")
    logger.info(f"  Total records: {pitcher_stats[0]}")
    logger.info(f"  Spin rate columns migrated: {pitcher_stats[1]}")
    logger.info(f"  K% migrated: {pitcher_stats[2]}")
    logger.info(f"  ERA migrated: {pitcher_stats[3]}")

    # EV table summary
    cursor.execute("""
        SELECT
            COUNT(*) as total,
            COUNT(avg_hit_speed_numeric) as speed_migrated,
            COUNT(ev95percent_numeric) as ev95_migrated,
            COUNT(avg_hit_angle_numeric) as angle_migrated
        FROM hitter_exit_velocity;
    """)
    ev_stats = cursor.fetchone()

    logger.info("\nhitter_exit_velocity:")
    logger.info(f"  Total records: {ev_stats[0]}")
    logger.info(f"  Hit speed migrated: {ev_stats[1]}")
    logger.info(f"  EV95+ % migrated: {ev_stats[2]}")
    logger.info(f"  Hit angle migrated: {ev_stats[3]}")

    # Sample spin rate data
    cursor.execute("""
        SELECT
            last_name_first_name,
            sl_avg_spin,
            sl_avg_spin_numeric,
            ch_avg_spin,
            ch_avg_spin_numeric,
            fastball_avg_spin,
            fastball_avg_spin_numeric
        FROM custom_pitcher_2025
        WHERE sl_avg_spin_numeric IS NOT NULL
        LIMIT 5;
    """)

    logger.info("\n📋 Sample Pitcher Spin Rates:")
    for row in cursor.fetchall():
        logger.info(f"  {row[0]}: SL {row[2]} | CH {row[4]} | FF {row[6]}")

    # Sample EV data
    cursor.execute("""
        SELECT
            last_name_first_name,
            year,
            avg_hit_speed,
            avg_hit_speed_numeric,
            ev95percent,
            ev95percent_numeric
        FROM hitter_exit_velocity
        WHERE avg_hit_speed_numeric IS NOT NULL
        LIMIT 5;
    """)

    logger.info("\n📋 Sample Hitter Exit Velocity:")
    for row in cursor.fetchall():
        logger.info(f"  {row[0]} ({row[1]}): Speed {row[3]} | EV95+ {row[5]}%")


def main():
    """Main execution"""
    logger.info("🚀 Starting data type migration...")
    logger.info(f"⏰ Started at: {datetime.now()}")

    conn = connect_db()

    try:
        # Fix pitcher table
        fix_pitcher_table(conn)

        # Fix EV table
        fix_ev_table(conn)

        # Generate summary
        create_migration_summary(conn)

        logger.info(f"\n✅ Migration complete at: {datetime.now()}")

    except Exception as e:
        logger.error(f"❌ Migration failed: {e}")
        conn.rollback()
        raise
    finally:
        conn.close()
        logger.info("✅ Database connection closed")


if __name__ == "__main__":
    main()
