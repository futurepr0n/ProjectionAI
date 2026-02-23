#!/usr/bin/env python3
"""
ProjectionAI - Fixed Data Loader
Properly join Hellraiser picks with games and pitcher stats
"""

import psycopg2
from psycopg2.extras import RealDictCursor
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FixedDataLoader:
    """Load data with proper game joins"""

    def __init__(self):
        self.conn = self._connect()

    def _connect(self):
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

    def close(self):
        if self.conn:
            self.conn.close()

    def create_labeled_dataset_v2(self) -> pd.DataFrame:
        """
        Create labeled dataset with proper game joins
        """
        logger.info("🚀 Creating labeled dataset v2 (fixed joins)...")

        # Step 1: Get Hellraiser picks with games (via date matching)
        query = """
        WITH hr_with_games AS (
            SELECT
                hp.id as pick_id,
                hp.analysis_date,
                hp.player_name,
                hp.team,
                hp.pitcher_name,
                hp.venue,
                hp.is_home,
                hp.confidence_score,
                hp.classification,
                hp.barrel_rate,
                hp.exit_velocity_avg,
                hp.hard_hit_percent,
                hp.sweet_spot_percent,
                hp.swing_optimization_score,
                hp.swing_attack_angle,
                hp.swing_bat_speed,
                hp.odds_decimal,
                hp.component_scores,
                g.game_id,
                g.game_date,
                g.home_team,
                g.away_team,
                g.home_score,
                g.away_score
            FROM hellraiser_picks hp
            LEFT JOIN games g ON 
                (hp.team = g.home_team OR hp.team = g.away_team)
                AND hp.analysis_date = g.game_date
            WHERE hp.analysis_date IS NOT NULL
        )
        SELECT * FROM hr_with_games;
        """

        cursor = self.conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute(query)
        data = cursor.fetchall()
        df = pd.DataFrame(data)

        logger.info(f"✅ Loaded {len(df)} Hellraiser picks with game matches")
        logger.info(f"   Games matched: {df['game_id'].notna().sum()}/{len(df)}")

        # Step 2: Get actual HR results
        game_ids = df['game_id'].dropna().unique().tolist()
        hr_results = self._get_hr_results(game_ids)

        # Step 3: Merge with HR results
        df = df.merge(
            hr_results,
            left_on=['game_id', 'player_name'],
            right_on=['game_id', 'batter'],
            how='left'
        )

        # Create label
        df['label'] = (df['hr_count'].fillna(0) > 0).astype(int)

        # Step 4: Get pitcher stats via game_id
        pitcher_stats = self._get_pitcher_stats_bulk(game_ids)
        df = df.merge(pitcher_stats, on='game_id', how='left', suffixes=('', '_pitcher'))

        logger.info(f"\n📊 Dataset Summary:")
        logger.info(f"   Total samples: {len(df):,}")
        logger.info(f"   Positive (HR): {df['label'].sum():,} ({df['label'].mean()*100:.1f}%)")
        logger.info(f"   With game_id: {df['game_id'].notna().sum():,}")
        logger.info(f"   With pitcher stats: {df['pitcher_era'].notna().sum():,}")

        return df

    def _get_hr_results(self, game_ids: List[int]) -> pd.DataFrame:
        """Get HR results from play_by_play"""
        if not game_ids:
            return pd.DataFrame(columns=['game_id', 'batter', 'hr_count'])

        query = """
        SELECT
            pp.game_id,
            pp.batter,
            COUNT(*) as hr_count
        FROM play_by_play_plays pp
        WHERE pp.game_id = ANY(%s)
          AND pp.play_result = 'Home Run'
        GROUP BY pp.game_id, pp.batter
        """

        cursor = self.conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute(query, (game_ids,))
        data = cursor.fetchall()

        df = pd.DataFrame(data)
        logger.info(f"✅ Loaded {len(df)} HR results from play-by-play")
        return df

    def _get_pitcher_stats_bulk(self, game_ids: List[int]) -> pd.DataFrame:
        """Get pitcher stats aggregated by game"""
        if not game_ids:
            return pd.DataFrame()

        query = """
        SELECT
            ps.game_id,
            ps.player_name as pitcher_name,
            ps.team as pitcher_team,
            SUM(ps.innings_pitched) as total_ip,
            SUM(ps.hits) as total_hits,
            SUM(ps.runs) as total_runs,
            SUM(ps.earned_runs) as total_er,
            SUM(ps.walks) as total_bb,
            SUM(ps.strikeouts) as total_so,
            SUM(ps.home_runs) as total_hr_allowed,
            COUNT(*) as pitchers_used,
            -- Derived metrics
            CASE 
                WHEN SUM(ps.innings_pitched) > 0 
                THEN SUM(ps.earned_runs)::NUMERIC / SUM(ps.innings_pitched) * 9
                ELSE NULL 
            END as pitcher_era,
            CASE 
                WHEN SUM(ps.innings_pitched) > 0 
                THEN SUM(ps.home_runs)::NUMERIC / SUM(ps.innings_pitched) * 9
                ELSE NULL 
            END as pitcher_hr_per_9,
            CASE 
                WHEN SUM(ps.innings_pitched) > 0 
                THEN SUM(ps.strikeouts)::NUMERIC / SUM(ps.innings_pitched) * 9
                ELSE NULL 
            END as pitcher_k_per_9,
            CASE 
                WHEN SUM(ps.innings_pitched) > 0 
                THEN (SUM(ps.hits) + SUM(ps.walks))::NUMERIC / SUM(ps.innings_pitched)
                ELSE NULL 
            END as pitcher_whip
        FROM pitching_stats ps
        WHERE ps.game_id = ANY(%s)
        GROUP BY ps.game_id, ps.player_name, ps.team
        """

        cursor = self.conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute(query, (game_ids,))
        data = cursor.fetchall()

        df = pd.DataFrame(data)
        logger.info(f"✅ Loaded pitcher stats for {len(df)} pitcher-games")

        # For each game, get the starting pitcher's stats (first pitcher)
        # Group by game_id and take the first pitcher (most IP usually)
        if not df.empty:
            df = df.sort_values(['game_id', 'total_ip'], ascending=[True, False])
            df = df.groupby('game_id').first().reset_index()

        return df

    def extract_pitcher_features_from_play_by_play(self, game_ids: List[int]) -> pd.DataFrame:
        """
        Extract advanced pitcher features from play-by-play data
        """
        if not game_ids:
            return pd.DataFrame()

        query = """
        WITH pitcher_plays AS (
            SELECT
                pp.game_id,
                pp.pitcher,
                COUNT(*) as total_pitches,
                COUNT(CASE WHEN pp.play_result = 'Home Run' THEN 1 END) as hr_allowed,
                COUNT(CASE WHEN pp.play_result = 'Strikeout' THEN 1 END) as strikeouts,
                COUNT(CASE WHEN pp.play_result = 'Walk' THEN 1 END) as walks,
                COUNT(CASE WHEN pp.play_result LIKE '%%Fly%%' THEN 1 END) as fly_balls,
                COUNT(CASE WHEN pp.play_result LIKE '%%Ground%%' THEN 1 END) as ground_balls,
                COUNT(CASE WHEN pp.play_result LIKE '%%Line%%' THEN 1 END) as line_drives
            FROM play_by_play_plays pp
            WHERE pp.game_id = ANY(%s)
              AND pp.pitcher IS NOT NULL
            GROUP BY pp.game_id, pp.pitcher
        )
        SELECT
            game_id,
            pitcher,
            total_pitches,
            hr_allowed,
            strikeouts,
            walks,
            fly_balls,
            ground_balls,
            line_drives,
            CASE 
                WHEN total_pitches > 0 THEN strikeouts::NUMERIC / total_pitches * 100
                ELSE 0 
            END as k_rate_pct,
            CASE 
                WHEN total_pitches > 0 THEN hr_allowed::NUMERIC / total_pitches * 100
                ELSE 0 
            END as hr_rate_pct,
            CASE 
                WHEN (fly_balls + ground_balls + line_drives) > 0 
                THEN fly_balls::NUMERIC / (fly_balls + ground_balls + line_drives) * 100
                ELSE 0 
            END as fly_ball_pct
        FROM pitcher_plays
        ORDER BY game_id, total_pitches DESC
        """

        cursor = self.conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute(query, (list(game_ids),))
        data = cursor.fetchall()

        df = pd.DataFrame(data)
        logger.info(f"✅ Extracted play-by-play features for {len(df)} pitcher-games")

        return df

    def build_complete_dataset(self) -> pd.DataFrame:
        """
        Build complete dataset with all features
        """
        # Get base labeled dataset
        df = self.create_labeled_dataset_v2()

        if df.empty:
            return df

        # Get game IDs
        game_ids = df['game_id'].dropna().unique().tolist()

        # Get play-by-play pitcher features
        pbp_features = self.extract_pitcher_features_from_play_by_play(game_ids)

        # Merge with main dataset
        if not pbp_features.empty:
            df = df.merge(
                pbp_features,
                left_on=['game_id', 'pitcher_name'],
                right_on=['game_id', 'pitcher'],
                how='left',
                suffixes=('', '_pbp')
            )

        # Log feature availability
        feature_cols = [
            'barrel_rate', 'exit_velocity_avg', 'hard_hit_percent', 'sweet_spot_percent',
            'pitcher_era', 'pitcher_hr_per_9', 'pitcher_k_per_9', 'pitcher_whip',
            'k_rate_pct', 'hr_rate_pct', 'fly_ball_pct',
            'confidence_score'
        ]

        logger.info(f"\n🎯 Feature Availability:")
        for col in feature_cols:
            if col in df.columns:
                available = df[col].notna().sum()
                pct = available / len(df) * 100
                logger.info(f"   {col}: {available:,}/{len(df):,} ({pct:.1f}%)")

        return df

    def save_dataset(self, df: pd.DataFrame, path: str = 'data/complete_dataset.csv'):
        df.to_csv(path, index=False)
        logger.info(f"✅ Saved dataset to {path} ({len(df):,} rows)")


if __name__ == "__main__":
    loader = FixedDataLoader()

    try:
        df = loader.build_complete_dataset()

        if not df.empty:
            loader.save_dataset(df, 'data/complete_dataset.csv')
            logger.info("\n✅ Dataset creation complete!")
    finally:
        loader.close()
