"""
ProjectionAI - Feature Store
Feature engineering pipeline for ML models
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

from database import get_database

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureStore:
    """Engineer features for ML models from raw data"""

    # Elite pitchers list (for HR prediction filtering)
    ELITE_PITCHERS = {
        'skubal', 'cole', 'verlander', 'ohtani', 'glasnow',
        'burnes', 'gausman', 'strider', 'wheeler', 'fried',
        'cease', 'valdez', 'nola', 'urias', 'alcantara'
    }

    # Known power hitters (for HR prediction)
    POWER_HITTERS = {
        'schwarber', 'ohtani', 'harper', 'judge', 'trout',
        'seager', 'betts', 'alvarez', 'alonso', 'ramirez',
        'smith', 'goldschmidt', 'freeman', 'turner', 'soto'
    }

    # Park factors (HR multiplier)
    PARK_FACTORS = {
        'COL': 1.35,  # Coors Field
        'NYY': 1.20,  # Yankee Stadium
        'BOS': 1.15,  # Fenway Park
        'CIN': 1.15,  # Great American Ball Park
        'PHI': 1.10,  # Citizens Bank Park
        'HOU': 1.10,  # Minute Maid Park
        'BAL': 1.05,  # Camden Yards
        'TEX': 1.05,  # Globe Life Field
        'ARI': 1.05,  # Chase Field
        'ATL': 1.00,  # Truist Park
        'CHC': 1.00,  # Wrigley Field
        'LAD': 1.00,  # Dodger Stadium
        'MIL': 1.00,  # American Family Field
        'SDP': 1.00,  # Petco Park
        'WSN': 1.00,  # Nationals Park
        'CLE': 0.95,  # Progressive Field
        'DET': 0.95,  # Comerica Park
        'KCR': 0.95,  # Kauffman Stadium
        'MIN': 0.95,  # Target Field
        'OAK': 0.90,  # Oakland Coliseum
        'MIA': 0.90,  # LoanDepot Park
        'PIT': 0.90,  # PNC Park
        'SEA': 0.90,  # T-Mobile Park
        'SFG': 0.90,  # Oracle Park
        'TBR': 0.90,  # Tropicana Field
    }

    def __init__(self):
        self.db = get_database()

    def get_hitter_features(self, game_id: str, player_id: int, pitcher_id: int,
                          stat_date: str) -> Dict:
        """
        Get features for HR prediction

        Args:
            game_id: Game identifier
            player_id: Batter's MLB ID
            pitcher_id: Pitcher's MLB ID
            stat_date: Date of game

        Returns:
            Feature dictionary
        """
        features = {}

        # Get hitter stats
        hitter_stats = self._get_latest_statcast_stats(player_id, is_pitcher=False, before_date=stat_date)
        if not hitter_stats:
            logger.warning(f"No stats found for hitter {player_id}")
            return {}

        # Get pitcher stats
        pitcher_stats = self._get_latest_statcast_stats(pitcher_id, is_pitcher=True, before_date=stat_date)
        if not pitcher_stats:
            logger.warning(f"No stats found for pitcher {pitcher_id}")
            return {}

        # Get game info
        game_info = self._get_game_info(game_id)
        if not game_info:
            return {}

        # Calculate features
        features.update(self._hitter_power_features(hitter_stats))
        features.update(self._pitcher_quality_features(pitcher_stats))
        features.update(self._matchup_features(hitter_stats, pitcher_stats, game_info))
        features.update(self._contextual_features(game_info))

        return features

    def get_hit_features(self, game_id: str, player_id: int, pitcher_id: int,
                        stat_date: str) -> Dict:
        """Get features for Hit prediction"""
        features = self.get_hitter_features(game_id, player_id, pitcher_id, stat_date)

        # Modify for hit prediction
        if features:
            # Add hit-specific features
            features['contact_quality_score'] = (
                features.get('sweet_spot_percent', 0) * 0.5 +
                (100 - features.get('k_percent', 0)) * 0.3 +
                features.get('avg_hit_speed', 0) / 100 * 0.2
            )

        return features

    def get_strikeout_features(self, game_id: str, player_id: int, pitcher_id: int,
                             stat_date: str) -> Dict:
        """Get features for Strikeout prediction"""
        features = {}

        # Get hitter stats
        hitter_stats = self._get_latest_statcast_stats(player_id, is_pitcher=False, before_date=stat_date)
        if not hitter_stats:
            return {}

        # Get pitcher stats
        pitcher_stats = self._get_latest_statcast_stats(pitcher_id, is_pitcher=True, before_date=stat_date)
        if not pitcher_stats:
            return {}

        # Calculate SO-specific features
        features['pitcher_k_rate'] = pitcher_stats.get('k_percent', 0)
        features['hitter_barrel_rate'] = hitter_stats.get('barrel_rate', 0)
        features['hitter_ev95_plus'] = hitter_stats.get('ev95_plus', 0)
        features['hitter_sweet_spot'] = hitter_stats.get('sweet_spot_percent', 0)

        # Composite feature: How hard is it for this pitcher to strike out this batter?
        features['k_probability_score'] = (
            pitcher_stats.get('k_percent', 0) * 0.7 -
            hitter_stats.get('barrel_rate', 0) * 0.2 -
            hitter_stats.get('ev95_plus', 0) * 0.1
        )

        # Elite pitcher bonus
        pitcher_name = self._get_player_name(pitcher_id)
        if pitcher_name and pitcher_name.lower() in self.ELITE_PITCHERS:
            features['is_elite_pitcher'] = 1
        else:
            features['is_elite_pitcher'] = 0

        return features

    def _get_latest_statcast_stats(self, player_id: int, is_pitcher: bool,
                                 before_date: str) -> Optional[Dict]:
        """Get latest Statcast stats for a player"""
        query = """
        SELECT * FROM statcast_data
        WHERE player_id = %s AND is_pitcher = %s AND stat_date < %s
        ORDER BY stat_date DESC
        LIMIT 1
        """

        result = self.db.execute_query(query, params=(player_id, is_pitcher, before_date), fetch="one")

        return result

    def _get_game_info(self, game_id: str) -> Optional[Dict]:
        """Get game information"""
        query = """
        SELECT * FROM games WHERE game_id = %s
        """

        result = self.db.execute_query(query, params=(game_id,), fetch="one")

        return result

    def _get_player_name(self, player_id: int) -> Optional[str]:
        """Get player name from database"""
        # Try hitters first
        query = """
        SELECT name FROM players WHERE player_id = %s
        """

        result = self.db.execute_query(query, params=(player_id,), fetch="one")
        if result:
            return result['name']

        # Try pitchers
        query = """
        SELECT name FROM pitchers WHERE pitcher_id = %s
        """

        result = self.db.execute_query(query, params=(player_id,), fetch="one")
        if result:
            return result['name']

        return None

    def _hitter_power_features(self, stats: Dict) -> Dict:
        """Calculate hitter power features"""
        features = {}

        # Barrel rate (most important for HR)
        features['barrel_rate'] = stats.get('barrel_rate', 0)

        # EV95+ (exit velocity 95+ mph)
        features['ev95_plus'] = stats.get('ev95_plus', 0)

        # Sweet spot (contact consistency)
        features['sweet_spot_percent'] = stats.get('sweet_spot_percent', 0)

        # Average hit speed
        features['avg_hit_speed'] = stats.get('avg_hit_speed', 0)

        # HR rate
        if stats.get('ab', 0) > 0:
            features['hr_per_ab'] = stats.get('hr', 0) / stats['ab']
        else:
            features['hr_per_ab'] = 0

        # Power score (composite)
        features['power_score'] = (
            stats.get('barrel_rate', 0) * 0.4 +
            stats.get('ev95_plus', 0) * 0.3 +
            stats.get('avg_hit_speed', 0) / 100 * 0.2 +
            features['hr_per_ab'] * 100 * 0.1
        )

        return features

    def _pitcher_quality_features(self, stats: Dict) -> Dict:
        """Calculate pitcher quality features"""
        features = {}

        # HR rate allowed
        features['hr_per_9'] = stats.get('hr_per_9', 0)

        # K rate
        features['pitcher_k_rate'] = stats.get('k_percent', 0)

        # ERA
        features['pitcher_era'] = stats.get('era', 0)

        # Barrel rate allowed
        features['barrel_rate_allowed'] = stats.get('barrel_rate_allowed', 0)

        # Average hit speed allowed
        features['avg_hit_speed_allowed'] = stats.get('avg_hit_speed_allowed', 0)

        # HR vulnerability score
        features['hr_vulnerability'] = (
            stats.get('hr_per_9', 0) * 0.4 +
            stats.get('barrel_rate_allowed', 0) * 0.3 +
            stats.get('avg_hit_speed_allowed', 0) / 100 * 0.2 +
            stats.get('era', 0) * 0.1
        )

        return features

    def _matchup_features(self, hitter_stats: Dict, pitcher_stats: Dict,
                         game_info: Dict) -> Dict:
        """Calculate matchup-specific features"""
        features = {}

        # Handedness advantage
        # (This would need batter/pitcher handedness - placeholder)
        features['handedness_advantage'] = 0.5  # Neutral if unknown

        # Park factor for home team
        home_team = game_info.get('home_team', '')
        features['park_factor'] = self.PARK_FACTORS.get(home_team, 1.0)

        # Recent form (last 7 games) - placeholder
        features['recent_form'] = 1.0  # Neutral

        # Days rest - placeholder
        features['days_rest'] = 0

        return features

    def _contextual_features(self, game_info: Dict) -> Dict:
        """Calculate contextual features"""
        features = {}

        # Temperature (warmer = more HRs)
        temp = game_info.get('weather_temp', 70)
        features['temperature_factor'] = min(max((temp - 50) / 30, 0), 1)

        # Wind (blowing out = more HRs)
        wind = game_info.get('weather_wind', 0)
        if wind > 10:
            features['wind_factor'] = 1.2
        elif wind < -10:
            features['wind_factor'] = 0.8
        else:
            features['wind_factor'] = 1.0

        # Game time (day games vs night games)
        # Placeholder
        features['is_day_game'] = 0

        return features

    def create_training_dataset(self, start_date: str, end_date: str,
                                 prediction_type: str = 'HR') -> pd.DataFrame:
        """
        Create training dataset with features and labels

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            prediction_type: 'HR', 'HIT', or 'SO'

        Returns:
            DataFrame with features and labels
        """
        logger.info(f"📊 Creating {prediction_type} training dataset...")

        # Get all games in date range
        query = """
        SELECT * FROM games
        WHERE game_date >= %s AND game_date <= %s
        ORDER BY game_date
        """

        games = self.db.execute_query(query, params=(start_date, end_date))

        if not games:
            logger.warning("No games found in date range")
            return pd.DataFrame()

        all_features = []
        all_labels = []

        for game in games:
            game_id = game['game_id']

            # Get lineups
            home_lineup = game.get('home_lineup', [])
            away_lineup = game.get('away_lineup', [])
            away_pitcher = game.get('away_pitcher_id')
            home_pitcher = game.get('home_pitcher_id')

            # Process home batters
            for batter_id in home_lineup:
                features = self.get_hitter_features(game_id, batter_id, away_pitcher, game['game_date'])
                if features:
                    features['game_id'] = game_id
                    features['batter_id'] = batter_id
                    features['pitcher_id'] = away_pitcher
                    features['game_date'] = game['game_date']

                    # Get label (did this batter HR?)
                    if prediction_type == 'HR':
                        label = 1 if str(batter_id) in game.get('home_hr_hitters', []) else 0
                    elif prediction_type == 'HIT':
                        label = 1 if str(batter_id) in game.get('home_hit_hitters', []) else 0
                    elif prediction_type == 'SO':
                        # Need SO data - placeholder
                        label = 0

                    all_features.append(features)
                    all_labels.append(label)

            # Process away batters
            for batter_id in away_lineup:
                features = self.get_hitter_features(game_id, batter_id, home_pitcher, game['game_date'])
                if features:
                    features['game_id'] = game_id
                    features['batter_id'] = batter_id
                    features['pitcher_id'] = home_pitcher
                    features['game_date'] = game['game_date']

                    # Get label
                    if prediction_type == 'HR':
                        label = 1 if str(batter_id) in game.get('away_hr_hitters', []) else 0
                    elif prediction_type == 'HIT':
                        label = 1 if str(batter_id) in game.get('away_hit_hitters', []) else 0
                    elif prediction_type == 'SO':
                        label = 0

                    all_features.append(features)
                    all_labels.append(label)

        # Create DataFrame
        df = pd.DataFrame(all_features)
        df['label'] = all_labels

        logger.info(f"✅ Created dataset with {len(df)} samples")
        logger.info(f"   Positive samples: {sum(all_labels)} ({sum(all_labels)/len(all_labels)*100:.1f}%)")

        return df

    def get_feature_importance(self, model, feature_names: List[str]) -> Dict:
        """
        Get feature importance from trained model

        Args:
            model: Trained model with feature_importances_
            feature_names: List of feature names

        Returns:
            Dictionary of feature: importance
        """
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            return dict(zip(feature_names, importance))
        else:
            logger.warning("Model does not have feature_importances_ attribute")
            return {}


if __name__ == "__main__":
    # Test feature store
    store = FeatureStore()

    logger.info("🧪 Testing feature store...")

    # Get sample features (would need actual game_id, player_id, pitcher_id)
    # features = store.get_hitter_features("test_game", 660271, 543037, "2024-08-01")

    logger.info("✅ Feature store initialized")
    logger.info(f"   Elite pitchers: {len(store.ELITE_PITCHERS)}")
    logger.info(f"   Power hitters: {len(store.POWER_HITTERS)}")
    logger.info(f"   Park factors: {len(store.PARK_FACTORS)}")
