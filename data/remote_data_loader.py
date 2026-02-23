"""
ProjectionAI - Data Loader
Extract training data from remote database
"""

import psycopg2
from psycopg2.extras import RealDictCursor
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from typing import List, Dict, Optional, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RemoteDataLoader:
    """Load data from remote baseball database"""

    def __init__(self):
        self.conn = self._connect()

    def _connect(self):
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

    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            logger.info("🔌 Database connection closed")

    def get_hellraiser_labeled_data(self) -> pd.DataFrame:
        """
        Get Hellraiser picks with labels from actual game results

        Returns:
            DataFrame with features and labels
        """
        query = """
        SELECT
            hp.id as pick_id,
            hp.analysis_date,
            hp.player_name,
            hp.team,
            hp.pitcher_name,
            hp.game_description,
            hp.venue,
            hp.game_time,
            hp.is_home,
            hp.confidence_score,
            hp.classification,
            hp.barrel_rate,
            hp.exit_velocity_avg,
            hp.hard_hit_percent,
            hp.sweet_spot_percent,
            hp.swing_optimization_score,
            hp.swing_attack_angle,
            hp.swing_ideal_rate,
            hp.swing_bat_speed,
            hp.odds_decimal,
            hp.component_scores,
            g.game_id,
            g.home_team,
            g.away_team,
            g.home_score,
            g.away_score
        FROM hellraiser_picks hp
        LEFT JOIN games g ON
            hp.game_description = (g.away_team || ' @ ' || g.home_team) OR
            hp.game_description = (g.home_team || ' @ ' || g.away_team)
        WHERE hp.analysis_date IS NOT NULL
        ORDER BY hp.analysis_date, hp.player_name;
        """

        try:
            cursor = self.conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute(query)
            data = cursor.fetchall()

            df = pd.DataFrame(data)

            logger.info(f"✅ Loaded {len(df)} Hellraiser picks")
            logger.info(f"   Unique dates: {df['analysis_date'].nunique()}")
            logger.info(f"   Unique players: {df['player_name'].nunique()}")

            return df

        except Exception as e:
            logger.error(f"❌ Error loading Hellraiser data: {e}")
            return pd.DataFrame()

    def get_actual_hr_results(self, game_ids: List[int]) -> pd.DataFrame:
        """
        Get actual HR results from play-by-play data

        Args:
            game_ids: List of game IDs to query

        Returns:
            DataFrame with actual HR hitters per game
        """
        if not game_ids:
            return pd.DataFrame()

        query = """
        SELECT
            pp.game_id,
            g.game_date,
            pp.batter,
            COUNT(*) as hr_count
        FROM play_by_play_plays pp
        JOIN games g ON pp.game_id = g.game_id
        WHERE pp.game_id = ANY(%s)
          AND pp.play_result = 'Home Run'
        GROUP BY pp.game_id, g.game_date, pp.batter
        ORDER BY g.game_date, pp.batter;
        """

        try:
            cursor = self.conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute(query, (game_ids,))
            data = cursor.fetchall()

            df = pd.DataFrame(data)

            logger.info(f"✅ Loaded {len(df)} actual HR results")
            logger.info(f"   Total HRs: {df['hr_count'].sum()}")

            return df

        except Exception as e:
            logger.error(f"❌ Error loading HR results: {e}")
            return pd.DataFrame()

    def get_pitcher_stats(self, pitcher_name: str, before_date: str = None) -> Optional[Dict]:
        """
        Get pitcher stats from pitching_stats table

        Args:
            pitcher_name: Pitcher name
            before_date: Optional date filter for historical stats

        Returns:
            Dictionary of pitcher stats
        """
        query = """
        SELECT
            innings_pitched,
            hits,
            runs,
            earned_runs,
            walks,
            strikeouts,
            COUNT(*) as games_pitched
        FROM pitching_stats
        WHERE player_name = %s
        """

        params = [pitcher_name]

        if before_date:
            query += " AND game_id IN (SELECT game_id FROM games WHERE game_date < %s)"
            params.append(before_date)

        query += " GROUP BY innings_pitched, hits, runs, earned_runs, walks, strikeouts;"

        try:
            cursor = self.conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute(query, params)
            result = cursor.fetchone()

            if result:
                # Calculate derived metrics
                if result['innings_pitched'] and result['innings_pitched'] > 0:
                    result['era'] = (result['earned_runs'] / result['innings_pitched']) * 9
                    result['k_per_9'] = (result['strikeouts'] / result['innings_pitched']) * 9
                    result['hr_per_9'] = 0  # Not available in pitching_stats
                else:
                    result['era'] = None
                    result['k_per_9'] = None
                    result['hr_per_9'] = None

                return result

            return None

        except Exception as e:
            logger.error(f"❌ Error getting pitcher stats: {e}")
            return None

    def create_labeled_dataset(self) -> pd.DataFrame:
        """
        Create complete labeled dataset for ML training

        Returns:
            DataFrame with features and HR label
        """
        logger.info("🚀 Creating labeled dataset...")

        # Load Hellraiser picks
        hellraiser_df = self.get_hellraiser_labeled_data()

        if hellraiser_df.empty:
            logger.error("❌ No Hellraiser data available")
            return pd.DataFrame()

        # Get game IDs
        game_ids = hellraiser_df['game_id'].dropna().unique().tolist()

        # Get actual HR results
        hr_results_df = self.get_actual_hr_results(game_ids)

        if hr_results_df.empty:
            logger.warning("⚠️ No HR results found, creating unlabeled dataset")
            hellraiser_df['label'] = None
            return hellraiser_df

        # Merge Hellraiser picks with HR results
        # Match by game_id and player_name (normalized)
        hellraiser_df['player_name_clean'] = hellraiser_df['player_name'].str.strip().str.lower()
        hr_results_df['batter_clean'] = hr_results_df['batter'].str.strip().str.lower()

        labeled_df = hellraiser_df.merge(
            hr_results_df[['game_id', 'batter_clean', 'hr_count', 'game_date']],
            left_on=['game_id', 'player_name_clean'],
            right_on=['game_id', 'batter_clean'],
            how='left'
        )

        # Create label (1 if HR, 0 if not)
        labeled_df['label'] = (labeled_df['hr_count'] > 0).astype(int)

        # Calculate pitcher metrics
        logger.info("📊 Calculating pitcher metrics...")

        pitcher_stats = {}
        for pitcher_name in labeled_df['pitcher_name'].unique():
            if pd.notna(pitcher_name):
                stats = self.get_pitcher_stats(pitcher_name)
                if stats:
                    pitcher_stats[pitcher_name] = stats

        # Add pitcher stats to dataframe
        labeled_df['pitcher_era'] = labeled_df['pitcher_name'].map(lambda x: pitcher_stats.get(x, {}).get('era') if pd.notna(x) else None)
        labeled_df['pitcher_k_per_9'] = labeled_df['pitcher_name'].map(lambda x: pitcher_stats.get(x, {}).get('k_per_9') if pd.notna(x) else None)

        # Log summary
        total_samples = len(labeled_df)
        positive_samples = labeled_df['label'].sum()
        negative_samples = total_samples - positive_samples
        hit_rate = positive_samples / total_samples * 100 if total_samples > 0 else 0

        logger.info(f"\n📊 Dataset Summary:")
        logger.info(f"   Total samples: {total_samples:,}")
        logger.info(f"   Positive (HR): {positive_samples:,} ({hit_rate:.1f}%)")
        logger.info(f"   Negative (no HR): {negative_samples:,} ({100-hit_rate:.1f}%)")
        logger.info(f"   Missing labels: {labeled_df['label'].isna().sum():,}")

        # Feature availability
        logger.info(f"\n🎯 Feature Availability:")
        for col in ['barrel_rate', 'exit_velocity_avg', 'hard_hit_percent', 'sweet_spot_percent',
                    'pitcher_era', 'pitcher_k_per_9', 'confidence_score']:
            available = labeled_df[col].notna().sum()
            percentage = available / total_samples * 100
            logger.info(f"   {col}: {available:,}/{total_samples} ({percentage:.1f}%)")

        return labeled_df

    def get_dataset_statistics(self, df: pd.DataFrame) -> Dict:
        """
        Get dataset statistics for analysis

        Args:
            df: Labeled dataset

        Returns:
            Dictionary of statistics
        """
        stats = {}

        # Overall stats
        stats['total_samples'] = len(df)
        stats['positive_samples'] = df['label'].sum()
        stats['negative_samples'] = len(df) - stats['positive_samples']
        stats['hit_rate'] = stats['positive_samples'] / len(df) * 100

        # Feature stats
        feature_cols = ['barrel_rate', 'exit_velocity_avg', 'hard_hit_percent', 'sweet_spot_percent',
                      'pitcher_era', 'pitcher_k_per_9', 'confidence_score']

        for col in feature_cols:
            if col in df.columns:
                # Convert to float to handle Decimal types
                series = df[col].astype(float)
                stats[f'{col}_mean'] = series.mean()
                stats[f'{col}_std'] = series.std()
                stats[f'{col}_min'] = series.min()
                stats[f'{col}_max'] = series.max()
                stats[f'{col}_null'] = df[col].isna().sum()

        return stats

    def save_dataset(self, df: pd.DataFrame, path: str = 'data/labeled_dataset.csv'):
        """
        Save labeled dataset to CSV

        Args:
            df: Labeled dataset
            path: Output path
        """
        df.to_csv(path, index=False)
        logger.info(f"✅ Dataset saved to {path}")


if __name__ == "__main__":
    # Test data loader
    loader = RemoteDataLoader()

    try:
        # Create labeled dataset
        df = loader.create_labeled_dataset()

        if not df.empty:
            # Get statistics
            stats = loader.get_dataset_statistics(df)

            # Save dataset
            loader.save_dataset(df, 'data/labeled_dataset.csv')

            logger.info("\n✅ Data extraction complete!")
        else:
            logger.error("❌ Failed to create dataset")

    finally:
        loader.close()
