"""
ProjectionAI - Statcast Data Loader
Pull historical and live Statcast data from MLB API
"""

import requests
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import time

from database import get_database

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StatcastLoader:
    """Load and process MLB Statcast data"""

    BASE_URL = "https://baseballsavant.mlb.com/statcast_search"

    def __init__(self):
        self.db = get_database()

    def search_statcast(self, player_type: str, player_id: Optional[int] = None,
                       hf: Optional[str] = None, vs: Optional[str] = None,
                       game_type: str = 'R', season: Optional[int] = None,
                       start_date: Optional[str] = None, end_date: Optional[str] = None,
                       team: Optional[str] = None) -> pd.DataFrame:
        """
        Search Statcast using MLB API

        Args:
            player_type: 'batter' or 'pitcher'
            player_id: MLB player ID (optional)
            hf: Hand field ('R', 'L', 'B')
            vs: Versus ('R', 'L', 'B')
            game_type: 'R' (regular season), 'P' (postseason), 'S' (spring training)
            season: Season year
            start_date: YYYY-MM-DD
            end_date: YYYY-MM-DD
            team: Team abbreviation

        Returns:
            DataFrame with Statcast data
        """
        params = {
            'hfPT': player_type,
            'hfPR': 'home',
            'hfAB': 'single',
            'game_type': game_type,
        }

        if player_id:
            params['hfGT'] = 'R|S|P'
            params['playerID'] = player_id

        if hf:
            params['hfHF'] = hf

        if vs:
            params['hfVS'] = vs

        if season:
            params['season'] = season

        if start_date and end_date:
            params['hfInn'] = 'innings'
            params['hfOuts'] = 'outs'
            params['hfBt'] = 'batters'
            params['hfSA'] = '1'
            params['hfBBT'] = 'all'
            params['hfVal': 'home|away'] = 'home|away'
            params['game_date_gt'] = start_date
            params['game_date_lt'] = end_date

        if team:
            params['team'] = team

        try:
            # Use baseball_scraper library if available
            try:
                from baseball_scraper import statcast

                df = statcast(
                    start_dt=start_date or '2023-01-01',
                    end_dt=end_date or datetime.now().strftime('%Y-%m-%d'),
                    player_type=player_type
                )

                logger.info(f"✅ Loaded {len(df)} {player_type} records from baseball_scraper")
                return df

            except ImportError:
                # Fallback to direct API call
                logger.warning("⚠️ baseball_scraper not found, trying direct API...")
                response = requests.get(self.BASE_URL, params=params, timeout=30)

                if response.status_code == 200:
                    data = response.json()
                    if 'search_results' in data:
                        df = pd.DataFrame(data['search_results'])
                        logger.info(f"✅ Loaded {len(df)} {player_type} records from API")
                        return df

            return pd.DataFrame()

        except Exception as e:
            logger.error(f"❌ Statcast API error: {e}")
            return pd.DataFrame()

    def get_hitter_stats(self, player_id: int, season: int, start_date: str, end_date: str) -> Dict:
        """
        Get hitter stats for a player

        Args:
            player_id: MLB player ID
            season: Season year
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            Dictionary of hitter metrics
        """
        df = self.search_statcast(
            player_type='batter',
            player_id=player_id,
            season=season,
            start_date=start_date,
            end_date=end_date
        )

        if df.empty:
            return {}

        stats = self._calculate_hitter_metrics(df)
        stats['player_id'] = player_id
        stats['stat_date'] = end_date
        stats['season'] = season
        stats['is_pitcher'] = False

        return stats

    def get_pitcher_stats(self, pitcher_id: int, season: int, start_date: str, end_date: str) -> Dict:
        """
        Get pitcher stats for a player

        Args:
            pitcher_id: MLB player ID
            season: Season year
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            Dictionary of pitcher metrics
        """
        df = self.search_statcast(
            player_type='pitcher',
            player_id=pitcher_id,
            season=season,
            start_date=start_date,
            end_date=end_date
        )

        if df.empty:
            return {}

        stats = self._calculate_pitcher_metrics(df)
        stats['player_id'] = pitcher_id
        stats['stat_date'] = end_date
        stats['season'] = season
        stats['is_pitcher'] = True

        return stats

    def _calculate_hitter_metrics(self, df: pd.DataFrame) -> Dict:
        """Calculate advanced hitting metrics from Statcast data"""
        stats = {}

        # Basic stats
        stats['pa'] = len(df)
        stats['ab'] = len(df[df['events'].notna()])

        # Outcomes
        stats['hits'] = len(df[df['events'].isin(['single', 'double', 'triple', 'home_run'])])
        stats['hr'] = len(df[df['events'] == 'home_run'])
        stats['so'] = len(df[df['events'] == 'strikeout'])
        stats['bb'] = len(df[df['events'] == 'walk'])

        # Rate stats
        if stats['ab'] > 0:
            stats['avg'] = stats['hits'] / stats['ab']
        else:
            stats['avg'] = 0.0

        if stats['pa'] > 0:
            stats['obp'] = (stats['hits'] + stats['bb']) / stats['pa']
        else:
            stats['obp'] = 0.0

        # Statcast metrics
        if 'launch_speed' in df.columns and 'launch_angle' in df.columns:
            # Hard hit rate (EV95+)
            if 'launch_speed' in df.columns:
                stats['ev95_plus'] = (df['launch_speed'] >= 95).mean() * 100
            else:
                stats['ev95_plus'] = 0.0

            # Sweet spot rate (15-30 degree launch angle)
            if 'launch_angle' in df.columns:
                stats['sweet_spot_percent'] = (
                    (df['launch_angle'] >= 15) & (df['launch_angle'] <= 30)
                ).mean() * 100
            else:
                stats['sweet_spot_percent'] = 0.0

            # Average hit speed
            hit_data = df[df['launch_speed'].notna()]
            if not hit_data.empty:
                stats['avg_hit_speed'] = hit_data['launch_speed'].mean()
                stats['avg_hit_angle'] = hit_data['launch_angle'].mean()
            else:
                stats['avg_hit_speed'] = 0.0
                stats['avg_hit_angle'] = 0.0

            # Barrel rate (Statcast definition: exit velocity >= 98 mph, launch angle 26-30 degrees)
            barrel_condition = (
                (df['launch_speed'] >= 98) &
                (df['launch_angle'] >= 26) &
                (df['launch_angle'] <= 30)
            )
            stats['barrel_rate'] = barrel_condition.mean() * 100 if len(df) > 0 else 0.0

        # Batted ball outcomes
        if 'bb_type' in df.columns:
            bb_type_counts = df['bb_type'].value_counts(normalize=True) * 100
            stats['fb_percent'] = bb_type_counts.get('fly_ball', 0)
            stats['gb_percent'] = bb_type_counts.get('ground_ball', 0)
            stats['ld_percent'] = bb_type_counts.get('line_drive', 0)
            stats['iffb_percent'] = bb_type_counts.get('popup', 0)

        # Pull/Center/Oppo
        if 'p_throws' in df.columns and 'stand' in df.columns:
            df['direction'] = df.apply(
                lambda row: self._get_hit_direction(row['stand'], row['p_throws']),
                axis=1
            )
            direction_counts = df['direction'].value_counts(normalize=True) * 100
            stats['pull_percent'] = direction_counts.get('pull', 0)
            stats['center_percent'] = direction_counts.get('center', 0)
            stats['oppo_percent'] = direction_counts.get('oppo', 0)

        # WRC+ (simplified approximation)
        if stats['pa'] > 0:
            stats['wrc_plus'] = int((stats['obp'] * 1.25 + stats['avg'] * 0.5) * 100)
        else:
            stats['wrc_plus'] = 0

        return stats

    def _calculate_pitcher_metrics(self, df: pd.DataFrame) -> Dict:
        """Calculate advanced pitching metrics from Statcast data"""
        stats = {}

        # Basic stats
        stats['pa'] = len(df)

        # Innings pitched (approximate: PA / 4.5)
        stats['innings_pitched'] = len(df) / 4.5 if len(df) > 0 else 0

        # Outcomes
        stats['hits_allowed'] = len(df[df['events'].isin(['single', 'double', 'triple', 'home_run'])])
        stats['hr_allowed'] = len(df[df['events'] == 'home_run'])
        stats['so'] = len(df[df['events'] == 'strikeout'])
        stats['bb_allowed'] = len(df[df['events'] == 'walk'])

        # Rate stats
        if stats['innings_pitched'] > 0:
            stats['era'] = (stats['hr_allowed'] * 4 + stats['hits_allowed'] * 1) / stats['innings_pitched'] * 9
            stats['fip'] = (13 * stats['hr_allowed'] + 3 * stats['bb_allowed'] - 2 * stats['so']) / stats['innings_pitched']
            stats['hr_per_9'] = stats['hr_allowed'] / stats['innings_pitched'] * 9
            stats['whip'] = (stats['hits_allowed'] + stats['bb_allowed']) / stats['innings_pitched']
        else:
            stats['era'] = 0.0
            stats['fip'] = 0.0
            stats['hr_per_9'] = 0.0
            stats['whip'] = 0.0

        # K rate and BB rate
        if stats['pa'] > 0:
            stats['k_percent'] = stats['so'] / stats['pa'] * 100
            stats['bb_percent'] = stats['bb_allowed'] / stats['pa'] * 100
        else:
            stats['k_percent'] = 0.0
            stats['bb_percent'] = 0.0

        # Allowed contact quality
        if 'launch_speed' in df.columns and 'launch_angle' in df.columns:
            # Average hit speed allowed
            hit_data = df[df['launch_speed'].notna()]
            if not hit_data.empty:
                stats['avg_hit_speed_allowed'] = hit_data['launch_speed'].mean()

                # Barrel rate allowed
                barrel_condition = (
                    (df['launch_speed'] >= 98) &
                    (df['launch_angle'] >= 26) &
                    (df['launch_angle'] <= 30)
                )
                stats['barrel_rate_allowed'] = barrel_condition.mean() * 100
            else:
                stats['avg_hit_speed_allowed'] = 0.0
                stats['barrel_rate_allowed'] = 0.0
        else:
            stats['avg_hit_speed_allowed'] = 0.0
            stats['barrel_rate_allowed'] = 0.0

        return stats

    def _get_hit_direction(self, batter_hand: str, pitcher_hand: str) -> str:
        """Determine hit direction based on batter/pitcher handedness"""
        if not pd.isna(batter_hand) and not pd.isna(pitcher_hand):
            if batter_hand == pitcher_hand:
                return 'pull'
            elif batter_hand != pitcher_hand:
                return 'oppo'
        return 'center'

    def load_season_data(self, season: int = 2024):
        """
        Load full season data for all players

        Args:
            season: Season year to load
        """
        logger.info(f"📊 Loading Statcast data for {season} season...")

        # Get list of teams
        teams = ['BAL', 'BOS', 'NYY', 'TBR', 'TOR',  # AL East
                 'CHW', 'CLE', 'DET', 'KCR', 'MIN',  # AL Central
                 'HOU', 'LAA', 'OAK', 'SEA', 'TEX',  # AL West
                 'ATL', 'MIA', 'NYM', 'PHI', 'WSN',  # NL East
                 'CHC', 'CIN', 'MIL', 'PIT', 'STL',  # NL Central
                 'ARI', 'COL', 'LAD', 'SDP', 'SFG']   # NL West

        total_records = 0

        for team in teams:
            try:
                # Get all batters for this team
                df = self.search_statcast(
                    player_type='batter',
                    team=team,
                    season=season
                )

                if not df.empty and 'player_id' in df.columns:
                    # Get unique player IDs
                    player_ids = df['player_id'].unique()

                    logger.info(f"  {team}: {len(player_ids)} batters")

                    # Load stats for each player
                    for player_id in player_ids:
                        stats = self.get_hitter_stats(player_id, season, f'{season}-04-01', f'{season}-11-01')
                        if stats:
                            self._save_statcast_data(stats)
                            total_records += 1

                # Get all pitchers for this team
                df_pitchers = self.search_statcast(
                    player_type='pitcher',
                    team=team,
                    season=season
                )

                if not df_pitchers.empty and 'player_id' in df_pitchers.columns:
                    pitcher_ids = df_pitchers['player_id'].unique()

                    logger.info(f"  {team}: {len(pitcher_ids)} pitchers")

                    for pitcher_id in pitcher_ids:
                        stats = self.get_pitcher_stats(pitcher_id, season, f'{season}-04-01', f'{season}-11-01')
                        if stats:
                            self._save_statcast_data(stats)
                            total_records += 1

                # Rate limiting
                time.sleep(1)

            except Exception as e:
                logger.error(f"❌ Error loading {team}: {e}")
                continue

        logger.info(f"✅ Loaded {total_records} player records for {season}")

    def _save_statcast_data(self, stats: Dict):
        """Save Statcast data to database"""
        try:
            query = """
            INSERT INTO statcast_data (
                player_id, is_pitcher, stat_date, season,
                pa, ab, hits, hr, so, bb, avg, obp, slg, ops, wrc_plus,
                barrel_rate, ev95_plus, sweet_spot_percent, avg_hit_speed, avg_hit_angle,
                max_hit_speed, hard_hit_percent, launch_angle_avg,
                fb_percent, gb_percent, ld_percent, iffb_percent,
                pull_percent, center_percent, oppo_percent,
                innings_pitched, era, fip, k_percent, bb_percent, hr_per_9,
                avg_hit_speed_allowed, barrel_rate_allowed, whip
            ) VALUES (
                %(player_id)s, %(is_pitcher)s, %(stat_date)s, %(season)s,
                %(pa)s, %(ab)s, %(hits)s, %(hr)s, %(so)s, %(bb)s, %(avg)s, %(obp)s,
                %(slg)s, %(ops)s, %(wrc_plus)s,
                %(barrel_rate)s, %(ev95_plus)s, %(sweet_spot_percent)s,
                %(avg_hit_speed)s, %(avg_hit_angle)s,
                %(max_hit_speed)s, %(hard_hit_percent)s, %(launch_angle_avg)s,
                %(fb_percent)s, %(gb_percent)s, %(ld_percent)s, %(iffb_percent)s,
                %(pull_percent)s, %(center_percent)s, %(oppo_percent)s,
                %(innings_pitched)s, %(era)s, %(fip)s, %(k_percent)s, %(bb_percent)s,
                %(hr_per_9)s, %(avg_hit_speed_allowed)s, %(barrel_rate_allowed)s, %(whip)s
            )
            ON CONFLICT (player_id, is_pitcher, stat_date) DO UPDATE SET
                pa = EXCLUDED.pa, ab = EXCLUDED.ab, hits = EXCLUDED.hits, hr = EXCLUDED.hr,
                so = EXCLUDED.so, bb = EXCLUDED.bb, avg = EXCLUDED.avg, obp = EXCLUDED.obp,
                slg = EXCLUDED.slg, ops = EXCLUDED.ops, wrc_plus = EXCLUDED.wrc_plus,
                barrel_rate = EXCLUDED.barrel_rate, ev95_plus = EXCLUDED.ev95_plus,
                sweet_spot_percent = EXCLUDED.sweet_spot_percent,
                avg_hit_speed = EXCLUDED.avg_hit_speed, avg_hit_angle = EXCLUDED.avg_hit_angle,
                max_hit_speed = EXCLUDED.max_hit_speed, hard_hit_percent = EXCLUDED.hard_hit_percent,
                launch_angle_avg = EXCLUDED.launch_angle_avg,
                fb_percent = EXCLUDED.fb_percent, gb_percent = EXCLUDED.gb_percent,
                ld_percent = EXCLUDED.ld_percent, iffb_percent = EXCLUDED.iffb_percent,
                pull_percent = EXCLUDED.pull_percent, center_percent = EXCLUDED.center_percent,
                oppo_percent = EXCLUDED.oppo_percent,
                innings_pitched = EXCLUDED.innings_pitched, era = EXCLUDED.era, fip = EXCLUDED.fip,
                k_percent = EXCLUDED.k_percent, bb_percent = EXCLUDED.bb_percent,
                hr_per_9 = EXCLUDED.hr_per_9, avg_hit_speed_allowed = EXCLUDED.avg_hit_speed_allowed,
                barrel_rate_allowed = EXCLUDED.barrel_rate_allowed, whip = EXCLUDED.whip
            """

            self.db.execute_query(query, params=stats, fetch="none")

        except Exception as e:
            logger.error(f"❌ Error saving stats for player {stats.get('player_id')}: {e}")


if __name__ == "__main__":
    loader = StatcastLoader()

    # Test loading one player's stats
    logger.info("🧪 Testing Statcast loader...")

    # Load Kyle Schwarber (ID: 660271)
    stats = loader.get_hitter_stats(660271, 2024, '2024-04-01', '2024-11-01')

    if stats:
        print("\n✅ Sample Hitter Stats (Kyle Schwarber):")
        for key, value in stats.items():
            print(f"  {key}: {value}")

    # Load Gerrit Cole (ID: 543037)
    pitcher_stats = loader.get_pitcher_stats(543037, 2024, '2024-04-01', '2024-11-01')

    if pitcher_stats:
        print("\n✅ Sample Pitcher Stats (Gerrit Cole):")
        for key, value in pitcher_stats.items():
            print(f"  {key}: {value}")
