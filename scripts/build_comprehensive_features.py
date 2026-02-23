#!/usr/bin/env python3
"""
Build Comprehensive Features for ProjectionAI

Loads and merges all feature sources into comprehensive_features.csv
"""

import psycopg2
from psycopg2.extras import RealDictCursor
import pandas as pd
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DB_PARAMS = {
    'host': '192.168.1.23',
    'port': 5432,
    'database': 'baseball_migration_test',
    'user': 'postgres',
    'password': 'korn5676'
}


def connect_db():
    conn = psycopg2.connect(**DB_PARAMS)
    logger.info(f"✅ Connected to database")
    return conn


def normalize_pitcher_name(name):
    if not name or pd.isna(name):
        return None
    parts = str(name).strip().split(' ')
    if len(parts) >= 2:
        return f"{parts[-1]}, {' '.join(parts[:-1])}"
    return name


def normalize_player_name(name):
    if not name or pd.isna(name):
        return None
    parts = str(name).strip().split(' ')
    if len(parts) >= 2:
        return f"{parts[-1]}, {' '.join(parts[:-1])}"
    return name


def main():
    logger.info("🚀 Building comprehensive features...")
    conn = connect_db()

    try:
        # Load hellraiser_picks
        query = """
        SELECT
            g.game_id::text as game_id,
            g.game_date,
            g.home_team as game_home_team,
            g.away_team as game_away_team,
            hp.player_name,
            hp.game_description,
            hp.team,
            hp.is_home,
            hp.classification,
            hp.confidence_score,
            hp.odds_decimal,
            hp.pitcher_name as original_pitcher
        FROM hellraiser_picks hp
        LEFT JOIN games g ON
            hp.game_description = (g.away_team || ' @ ' || g.home_team) OR
            hp.game_description = (g.home_team || ' @ ' || g.away_team);
        """
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute(query)
        df = pd.DataFrame(cursor.fetchall())
        logger.info(f"✅ Loaded {len(df)} hellraiser_picks records")

        # Normalize names
        df['player_normalized'] = df['player_name'].apply(normalize_player_name)
        df['pitcher_normalized'] = df['original_pitcher'].apply(normalize_pitcher_name)

        # Load pitcher features
        query = """
        SELECT
            last_name_first_name,
            k_percent_numeric as k_percent,
            p_era_numeric as era,
            bb_percent_numeric as bb_percent,
            sl_avg_spin_numeric as slider_spin,
            ch_avg_spin_numeric as changeup_spin,
            cu_avg_spin_numeric as curveball_spin,
            si_avg_spin_numeric as sinker_spin,
            fc_avg_spin_numeric as cutter_spin,
            kn_avg_spin_numeric as knuckle_spin,
            st_avg_spin_numeric as sweeper_spin,
            fo_avg_spin_numeric as forkball_spin,
            sc_avg_spin_numeric as screwball_spin,
            fastball_avg_spin_numeric as fastball_spin,
            breaking_avg_spin_numeric as breaking_spin,
            offspeed_avg_spin_numeric as offspeed_spin
        FROM custom_pitcher_2025
        WHERE last_name_first_name IS NOT NULL;
        """
        cursor.execute(query)
        pitcher_df = pd.DataFrame(cursor.fetchall())
        logger.info(f"✅ Loaded features for {len(pitcher_df)} pitchers")

        if not pitcher_df.empty:
            pitcher_df['spin_composite'] = (
                pitcher_df['fastball_spin'].fillna(0) * 0.4 +
                pitcher_df['breaking_spin'].fillna(0) * 0.4 +
                pitcher_df['offspeed_spin'].fillna(0) * 0.2
            )

        # Merge pitcher features
        df = df.merge(pitcher_df, left_on='pitcher_normalized', right_on='last_name_first_name', how='left')
        logger.info(f"   Pitcher features merged: {df['k_percent'].notna().sum()}/{len(df)}")

        # Load hitter EV features
        query = """
        WITH latest_ev AS (
            SELECT
                last_name_first_name,
                avg_hit_speed_numeric as avg_ev,
                max_hit_speed_numeric as max_ev,
                ev95percent_numeric as ev95_plus_percent,
                avg_distance_numeric as avg_distance,
                avg_hr_distance_numeric as avg_hr_distance,
                avg_hit_angle_numeric as avg_launch_angle,
                brl_percent_numeric as barrel_rate,
                anglesweetspotpercent_numeric as sweet_spot_rate,
                year,
                ROW_NUMBER() OVER (PARTITION BY last_name_first_name ORDER BY year DESC) as rn
            FROM hitter_exit_velocity
            WHERE last_name_first_name IS NOT NULL
        )
        SELECT
            last_name_first_name,
            avg_ev,
            max_ev,
            ev95_plus_percent,
            avg_distance,
            avg_hr_distance,
            avg_launch_angle,
            barrel_rate,
            sweet_spot_rate
        FROM latest_ev
        WHERE rn = 1;
        """
        cursor.execute(query)
        ev_df = pd.DataFrame(cursor.fetchall())
        logger.info(f"✅ Loaded EV features for {len(ev_df)} hitters")

        # Merge EV features
        df = df.merge(ev_df, left_on='player_normalized', right_on='last_name_first_name', how='left')
        logger.info(f"   Hitter EV features merged: {df['avg_ev'].notna().sum()}/{len(df)}")

        # Load x-stats
        query = """
        SELECT
            last_name_first_name,
            batting_avg as avg,
            on_base_percent as obp,
            slg_percent as slg,
            on_base_plus_slg as ops,
            woba,
            xwoba,
            xba,
            xslg,
            k_percent as k_percent_xstats,
            bb_percent as bb_percent_xstats,
            avg_swing_speed as swing_speed
        FROM custom_batter_2025
        WHERE last_name_first_name IS NOT NULL;
        """
        cursor.execute(query)
        xstats_df = pd.DataFrame(cursor.fetchall())
        logger.info(f"✅ Loaded x-stats for {len(xstats_df)} batters")

        # Merge x-stats
        df = df.merge(xstats_df, left_on='player_normalized', right_on='last_name_first_name', how='left', suffixes=('', '_xstats'))
        logger.info(f"   Batter x-stats merged: {df['xwoba'].notna().sum()}/{len(df)}")

        # Load batted ball trends
        query = """
        SELECT
            player_name as batter_name,
            matchup_type,
            AVG(ground_ball_rate) as avg_gb_rate,
            AVG(fly_ball_rate) as avg_fb_rate,
            AVG(line_drive_rate) as avg_ld_rate,
            AVG(pull_rate) as avg_pull_rate,
            AVG(opposite_rate) as avg_opp_rate,
            SUM(batted_ball_events) as total_events,
            MAX(game_date) as last_date
        FROM daily_batted_ball_tracking
        WHERE player_name IS NOT NULL
          AND batted_ball_events > 5
        GROUP BY player_name, matchup_type;
        """
        cursor.execute(query)
        trends_df = pd.DataFrame(cursor.fetchall())
        logger.info(f"✅ Loaded batted ball trends for {len(trends_df)} batter/matchup combinations")

        if not trends_df.empty:
            trends_wide = trends_df.pivot(
                index='batter_name',
                columns='matchup_type',
                values=['avg_gb_rate', 'avg_fb_rate', 'avg_ld_rate', 'avg_pull_rate', 'avg_opp_rate']
            )
            trends_wide.columns = [f"{stat}_{matchup}" for stat, matchup in trends_wide.columns]
            trends_wide = trends_wide.reset_index()

            df = df.merge(trends_wide, left_on='player_normalized', right_on='batter_name', how='left')
            trend_col = [c for c in df.columns if 'avg_gb_rate' in c][0]
            logger.info(f"   Batted ball trends merged: {df[trend_col].notna().sum()}/{len(df)}")

        # Save
        output_path = '/home/futurepr0n/Development/ProjectionAI/data/comprehensive_features.csv'
        df.to_csv(output_path, index=False)
        logger.info(f"✅ Saved comprehensive features to {output_path}")
        logger.info(f"   Total features: {len(df.columns)}")
        logger.info(f"   Total records: {len(df)}")

        # Summary
        logger.info("\n📊 Feature Completeness:")
        logger.info(f"   Pitcher K%: {df['k_percent'].notna().sum()} ({df['k_percent'].notna().sum()/len(df)*100:.1f}%)")
        logger.info(f"   Hitter EV: {df['avg_ev'].notna().sum()} ({df['avg_ev'].notna().sum()/len(df)*100:.1f}%)")
        logger.info(f"   Batter xwOBA: {df['xwoba'].notna().sum()} ({df['xwoba'].notna().sum()/len(df)*100:.1f}%)")

    finally:
        conn.close()
        logger.info("✅ Database connection closed")


if __name__ == '__main__':
    main()
