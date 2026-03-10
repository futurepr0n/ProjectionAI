#!/usr/bin/env python3
"""
Build a starter-level strikeout dataset.

Purpose:
- move strikeout modeling off the hitter-row dataset
- create one row per projected/actual starting pitcher per game
- keep both strikeout counts and flexible binary labels (4+/5+/6+)

This script reads existing DB tables only and writes a local CSV.
"""

from __future__ import annotations

import logging
import os
from datetime import timedelta
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor

from name_utils import normalize_name

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OUTPUT_PATH = Path(__file__).parent / 'pitcher_strikeout_dataset.csv'

TEAM_CODE_ALIASES = {
    'AZ': 'ARI',
    'ARI': 'ARI',
    'CHW': 'CWS',
    'CWS': 'CWS',
    'KCR': 'KC',
    'KC': 'KC',
    'SDP': 'SD',
    'SD': 'SD',
    'SFG': 'SF',
    'SF': 'SF',
    'TBR': 'TB',
    'TB': 'TB',
    'WSN': 'WSH',
    'WSH': 'WSH',
    'ATH': 'OAK',
    'OAK': 'OAK',
}


def _to_float(value, default=None):
    if value in (None, ''):
        return default
    try:
        return float(value)
    except Exception:
        return default


def _pitcher_name_key(name: str) -> str:
    normalized = normalize_name(name)
    parts = normalized.split()
    if len(parts) >= 2:
        return f"{parts[0][:1]} {parts[-1]}".strip()
    return normalized


def _normalize_team_code(team_code: str) -> str:
    if not team_code:
        return team_code
    normalized = str(team_code).strip().upper()
    return TEAM_CODE_ALIASES.get(normalized, normalized)


def _name_from_last_first(value: str) -> str:
    if value is None:
        return ''
    text = str(value).strip()
    if ',' in text:
        parts = [part.strip() for part in text.split(',') if part.strip()]
        if len(parts) >= 2:
            return f"{parts[1]} {parts[0]}".strip()
    return text


class PitcherStrikeoutDatasetBuilder:
    def __init__(self):
        self.conn = self._connect()

    def _connect(self):
        return psycopg2.connect(
            host=os.getenv('DB_HOST', '192.168.1.23'),
            port=int(os.getenv('DB_PORT', 5432)),
            database=os.getenv('DB_NAME', 'baseball_migration_test'),
            user=os.getenv('DB_USER', 'postgres'),
            password=os.getenv('DB_PASSWORD', 'korn5676')
        )

    def close(self):
        if self.conn:
            self.conn.close()

    def build(self) -> pd.DataFrame:
        logger.info("Building starter strikeout dataset...")
        starter_df = self._load_starter_spine()
        if starter_df.empty:
            logger.warning("Starter spine is empty")
            return starter_df

        actuals_df = self._load_actual_starter_results()
        starter_df = starter_df.merge(actuals_df, on=['game_id', 'team'], how='left')

        starter_df = self._attach_pitcher_rolling(starter_df)
        starter_df = self._attach_pitcher_season_metrics(starter_df)
        starter_df = self._attach_pitcher_arsenal_metrics(starter_df)
        starter_df = self._attach_opponent_team_form(starter_df)

        starter_df['actual_strikeouts'] = starter_df['actual_strikeouts'].fillna(0).astype(int)
        starter_df['label_so_3_plus'] = (starter_df['actual_strikeouts'] >= 3).astype(int)
        starter_df['label_so_4_plus'] = (starter_df['actual_strikeouts'] >= 4).astype(int)
        starter_df['label_so_5_plus'] = (starter_df['actual_strikeouts'] >= 5).astype(int)
        starter_df['label_so_6_plus'] = (starter_df['actual_strikeouts'] >= 6).astype(int)

        logger.info(
            "Starter strikeout dataset built: %s rows from %s to %s",
            len(starter_df),
            starter_df['game_date'].min(),
            starter_df['game_date'].max(),
        )
        logger.info(
            "Positive rates | 3+: %.2f%% | 4+: %.2f%% | 5+: %.2f%% | 6+: %.2f%%",
            starter_df['label_so_3_plus'].mean() * 100,
            starter_df['label_so_4_plus'].mean() * 100,
            starter_df['label_so_5_plus'].mean() * 100,
            starter_df['label_so_6_plus'].mean() * 100,
        )
        return starter_df

    def save_dataset(self, df: pd.DataFrame, output_path: Path = OUTPUT_PATH):
        df.to_csv(output_path, index=False)
        logger.info("Saved dataset to %s", output_path)

    def _load_starter_spine(self) -> pd.DataFrame:
        cursor = self.conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute(
            """
            WITH lineup_ranked AS (
                SELECT
                    dl.game_date,
                    dl.game_id AS lineup_game_id,
                    dl.home_team,
                    dl.away_team,
                    dl.home_pitcher->>'name' AS home_pitcher_name,
                    dl.away_pitcher->>'name' AS away_pitcher_name,
                    dl.home_lineup,
                    dl.away_lineup,
                    ROW_NUMBER() OVER (
                        PARTITION BY dl.game_date, dl.home_team, dl.away_team
                        ORDER BY dl.game_id
                    ) AS matchup_seq
                FROM daily_lineups dl
                WHERE dl.home_pitcher IS NOT NULL
                  AND dl.away_pitcher IS NOT NULL
            ),
            games_ranked AS (
                SELECT
                    g.game_date,
                    g.game_id AS actual_game_id,
                    g.home_team,
                    g.away_team,
                    ROW_NUMBER() OVER (
                        PARTITION BY g.game_date, g.home_team, g.away_team
                        ORDER BY g.game_id
                    ) AS matchup_seq
                FROM games g
            )
            SELECT
                lr.game_date,
                lr.lineup_game_id,
                gr.actual_game_id,
                lr.home_team,
                lr.away_team,
                lr.home_pitcher_name,
                lr.away_pitcher_name,
                lr.home_lineup,
                lr.away_lineup
            FROM lineup_ranked lr
            LEFT JOIN games_ranked gr
              ON gr.game_date = lr.game_date
             AND gr.home_team = lr.home_team
             AND gr.away_team = lr.away_team
             AND gr.matchup_seq = lr.matchup_seq
            ORDER BY lr.game_date, lr.lineup_game_id
            """
        )
        rows = cursor.fetchall()
        records: List[Dict] = []
        unmatched = 0

        for row in rows:
            if row['actual_game_id'] is None:
                unmatched += 1
                continue
            game_date = row['game_date']
            game_id = int(row['actual_game_id'])
            for side in ('home', 'away'):
                lineup = row[f'{side}_lineup'] or {}
                batting_order = lineup.get('batting_order', []) if isinstance(lineup, dict) else []
                starter_name = row[f'{side}_pitcher_name']
                team = row[f'{side}_team']
                opponent = row['away_team'] if side == 'home' else row['home_team']
                records.append({
                    'game_date': game_date,
                    'game_id': game_id,
                    'lineup_game_id': row['lineup_game_id'],
                    'starter_name': starter_name,
                    'starter_name_normalized': normalize_name(starter_name),
                    'starter_name_key': _pitcher_name_key(starter_name),
                    'team': team,
                    'opponent_team': opponent,
                    'is_home': side == 'home',
                    'lineup_confirmed': bool(lineup.get('confirmed')) if isinstance(lineup, dict) else False,
                    'opponent_lineup_size': len(batting_order),
                })

        df = pd.DataFrame(records)
        logger.info("Loaded starter spine: %s rows (%s unmatched lineup games dropped)", len(df), unmatched)
        return df

    def _load_actual_starter_results(self) -> pd.DataFrame:
        cursor = self.conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute(
            """
            WITH starter_rows AS (
                SELECT
                    ps.game_id,
                    ps.team,
                    ps.player_name AS actual_starter_name,
                    ps.innings_pitched,
                    ps.strikeouts AS actual_strikeouts,
                    ROW_NUMBER() OVER (
                        PARTITION BY ps.game_id, ps.team
                        ORDER BY ps.innings_pitched DESC, ps.strikeouts DESC, ps.id
                    ) AS rn
                FROM pitching_stats ps
            )
            SELECT
                game_id,
                team,
                actual_starter_name,
                innings_pitched AS actual_innings_pitched,
                actual_strikeouts
            FROM starter_rows
            WHERE rn = 1
            """
        )
        df = pd.DataFrame(cursor.fetchall())
        logger.info("Loaded actual starter results: %s rows", len(df))
        return df

    def _load_pitching_history(self) -> pd.DataFrame:
        cursor = self.conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute(
            """
            SELECT
                ps.game_id,
                g.game_date,
                ps.player_name,
                ps.innings_pitched,
                ps.strikeouts,
                ps.earned_runs,
                ps.hits,
                ps.walks
            FROM pitching_stats ps
            JOIN games g ON ps.game_id = g.game_id
            WHERE g.game_date IS NOT NULL
            ORDER BY g.game_date, ps.player_name
            """
        )
        df = pd.DataFrame(cursor.fetchall())
        if df.empty:
            return df
        df['player_name_normalized'] = df['player_name'].apply(normalize_name)
        df['player_name_key'] = df['player_name'].apply(_pitcher_name_key)
        df['game_date'] = pd.to_datetime(df['game_date'])

        cursor.execute(
            """
            SELECT
                pp.game_id,
                pp.pitcher AS player_name,
                COUNT(p.id) AS pitch_count,
                COUNT(DISTINCT pp.id) AS batters_faced
            FROM play_by_play_plays pp
            LEFT JOIN play_by_play_pitches p ON p.play_id = pp.id
            WHERE pp.pitcher IS NOT NULL
            GROUP BY pp.game_id, pp.pitcher
            """
        )
        pitch_count_df = pd.DataFrame(cursor.fetchall())
        if not pitch_count_df.empty:
            pitch_count_df['player_name_key'] = pitch_count_df['player_name'].apply(_pitcher_name_key)
            pitch_count_df = (
                pitch_count_df.groupby(['game_id', 'player_name_key'], as_index=False)[['pitch_count', 'batters_faced']]
                .sum()
            )
            df = df.merge(pitch_count_df, on=['game_id', 'player_name_key'], how='left')
        else:
            df['pitch_count'] = np.nan
            df['batters_faced'] = np.nan
        return df

    def _attach_pitcher_rolling(self, starter_df: pd.DataFrame) -> pd.DataFrame:
        history = self._load_pitching_history()
        if history.empty:
            for col in (
                'starter_avg_ip_30d',
                'starter_avg_so_30d',
                'starter_k_per_9_30d',
                'starter_era_30d',
                'starter_whip_30d',
                'starter_starts_30d',
                'starter_avg_pitches_30d',
                'starter_max_pitches_30d',
                'starter_avg_batters_faced_30d',
            ):
                starter_df[col] = np.nan
            return starter_df

        starter_df = starter_df.copy()
        starter_df['game_date_ts'] = pd.to_datetime(starter_df['game_date'])

        rolling_records = []
        for pitcher_name_key, pitcher_rows in starter_df.groupby('starter_name_key', dropna=False):
            history_rows = history[history['player_name_key'] == pitcher_name_key].copy()
            history_rows = history_rows.sort_values('game_date')

            for idx, row in pitcher_rows.sort_values('game_date_ts').iterrows():
                start_date = row['game_date_ts'] - timedelta(days=30)
                mask = (history_rows['game_date'] < row['game_date_ts']) & (history_rows['game_date'] >= start_date)
                prior = history_rows.loc[mask]

                total_ip = prior['innings_pitched'].astype(float).sum() if not prior.empty else 0.0
                total_so = prior['strikeouts'].astype(float).sum() if not prior.empty else 0.0
                total_er = prior['earned_runs'].astype(float).sum() if not prior.empty else 0.0
                total_hits = prior['hits'].astype(float).sum() if not prior.empty else 0.0
                total_walks = prior['walks'].astype(float).sum() if not prior.empty else 0.0
                prior_pitch_counts = pd.to_numeric(prior.get('pitch_count'), errors='coerce')
                prior_batters_faced = pd.to_numeric(prior.get('batters_faced'), errors='coerce')

                rolling_records.append({
                    'row_index': idx,
                    'starter_starts_30d': int(len(prior)),
                    'starter_avg_ip_30d': float(prior['innings_pitched'].astype(float).mean()) if not prior.empty else np.nan,
                    'starter_avg_so_30d': float(prior['strikeouts'].astype(float).mean()) if not prior.empty else np.nan,
                    'starter_k_per_9_30d': float(total_so / total_ip * 9) if total_ip > 0 else np.nan,
                    'starter_era_30d': float(total_er / total_ip * 9) if total_ip > 0 else np.nan,
                    'starter_whip_30d': float((total_hits + total_walks) / total_ip) if total_ip > 0 else np.nan,
                    'starter_avg_pitches_30d': float(prior_pitch_counts.mean()) if prior_pitch_counts.notna().any() else np.nan,
                    'starter_max_pitches_30d': float(prior_pitch_counts.max()) if prior_pitch_counts.notna().any() else np.nan,
                    'starter_avg_batters_faced_30d': float(prior_batters_faced.mean()) if prior_batters_faced.notna().any() else np.nan,
                })

        rolling_df = pd.DataFrame(rolling_records).set_index('row_index')
        starter_df = starter_df.join(rolling_df, how='left')
        starter_df = starter_df.drop(columns=['game_date_ts'])
        logger.info("Attached rolling pitcher features")
        return starter_df

    def _attach_pitcher_season_metrics(self, starter_df: pd.DataFrame) -> pd.DataFrame:
        cursor = self.conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute(
            """
            SELECT
                last_name_first_name,
                k_percent,
                bb_percent,
                whiff_percent,
                p_total_pitches,
                p_total_swinging_strike
            FROM custom_pitcher_2025
            WHERE last_name_first_name IS NOT NULL
            """
        )
        pitcher_df = pd.DataFrame(cursor.fetchall())
        if pitcher_df.empty:
            for col in ('starter_k_percent_season', 'starter_bb_percent_season', 'starter_whiff_percent_season'):
                starter_df[col] = np.nan
            return starter_df

        pitcher_df['starter_name_normalized'] = pitcher_df['last_name_first_name'].apply(
            lambda value: normalize_name(' '.join(reversed([part.strip() for part in str(value).split(',')[:2]])))
            if ',' in str(value) else normalize_name(str(value))
        )
        pitcher_df = pitcher_df.drop_duplicates('starter_name_normalized')
        season_map = pitcher_df.set_index('starter_name_normalized')

        starter_df = starter_df.copy()
        starter_df['starter_k_percent_season'] = starter_df['starter_name_normalized'].map(
            lambda key: _to_float(season_map.at[key, 'k_percent']) if key in season_map.index else np.nan
        )
        starter_df['starter_bb_percent_season'] = starter_df['starter_name_normalized'].map(
            lambda key: _to_float(season_map.at[key, 'bb_percent']) if key in season_map.index else np.nan
        )
        starter_df['starter_whiff_percent_season'] = starter_df['starter_name_normalized'].map(
            lambda key: _to_float(season_map.at[key, 'whiff_percent']) if key in season_map.index else np.nan
        )
        logger.info("Attached starter season metrics from custom_pitcher_2025")
        return starter_df

    def _attach_pitcher_arsenal_metrics(self, starter_df: pd.DataFrame) -> pd.DataFrame:
        cursor = self.conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute(
            """
            SELECT
                last_name_first_name,
                team_name_alt,
                pitch_type,
                pitch_usage,
                whiff_percent,
                k_percent,
                put_away
            FROM pitcherarsenalstats_2025
            WHERE last_name_first_name IS NOT NULL
              AND pitch_type IS NOT NULL
            """
        )
        arsenal_df = pd.DataFrame(cursor.fetchall())
        if arsenal_df.empty:
            for col in (
                'starter_primary_pitch_usage',
                'starter_primary_pitch_whiff_percent',
                'starter_primary_pitch_k_percent',
                'starter_primary_pitch_put_away',
                'starter_secondary_pitch_usage',
                'starter_secondary_pitch_whiff_percent',
                'starter_secondary_pitch_k_percent',
                'starter_secondary_pitch_put_away',
                'starter_arsenal_whiff_percent',
                'starter_arsenal_k_percent',
                'starter_arsenal_put_away',
                'opp_team_k_vs_primary_pitch',
                'opp_team_whiff_vs_primary_pitch',
                'opp_team_put_away_vs_primary_pitch',
                'opp_team_k_vs_secondary_pitch',
                'opp_team_whiff_vs_secondary_pitch',
                'opp_team_put_away_vs_secondary_pitch',
                'opp_team_k_vs_starter_arsenal',
                'opp_team_whiff_vs_starter_arsenal',
                'opp_team_put_away_vs_starter_arsenal',
            ):
                starter_df[col] = np.nan
            return starter_df

        arsenal_df['starter_name_normalized'] = arsenal_df['last_name_first_name'].apply(
            lambda value: normalize_name(_name_from_last_first(value))
        )
        arsenal_df['team_normalized'] = arsenal_df['team_name_alt'].apply(_normalize_team_code)
        for col in ('pitch_usage', 'whiff_percent', 'k_percent', 'put_away'):
            arsenal_df[col] = pd.to_numeric(arsenal_df[col], errors='coerce')

        starter_profiles: Dict[tuple, Dict[str, float]] = {}
        for (starter_name, team_code), grp in arsenal_df.groupby(['starter_name_normalized', 'team_normalized'], dropna=False):
            grp = grp.dropna(subset=['pitch_usage']).sort_values('pitch_usage', ascending=False).copy()
            if grp.empty:
                continue
            usage_sum = grp['pitch_usage'].sum()
            primary = grp.iloc[0]
            secondary = grp.iloc[1] if len(grp) > 1 else None
            starter_profiles[(starter_name, team_code)] = {
                'primary_pitch_type': primary['pitch_type'],
                'secondary_pitch_type': secondary['pitch_type'] if secondary is not None else None,
                'starter_primary_pitch_usage': _to_float(primary['pitch_usage']),
                'starter_primary_pitch_whiff_percent': _to_float(primary['whiff_percent']),
                'starter_primary_pitch_k_percent': _to_float(primary['k_percent']),
                'starter_primary_pitch_put_away': _to_float(primary['put_away']),
                'starter_secondary_pitch_usage': _to_float(secondary['pitch_usage']) if secondary is not None else np.nan,
                'starter_secondary_pitch_whiff_percent': _to_float(secondary['whiff_percent']) if secondary is not None else np.nan,
                'starter_secondary_pitch_k_percent': _to_float(secondary['k_percent']) if secondary is not None else np.nan,
                'starter_secondary_pitch_put_away': _to_float(secondary['put_away']) if secondary is not None else np.nan,
                'starter_arsenal_whiff_percent': float(np.average(grp['whiff_percent'].fillna(0), weights=grp['pitch_usage'])) if usage_sum > 0 else np.nan,
                'starter_arsenal_k_percent': float(np.average(grp['k_percent'].fillna(0), weights=grp['pitch_usage'])) if usage_sum > 0 else np.nan,
                'starter_arsenal_put_away': float(np.average(grp['put_away'].fillna(0), weights=grp['pitch_usage'])) if usage_sum > 0 else np.nan,
            }

        cursor.execute(
            """
            SELECT
                team_name_alt,
                pitch_type,
                pa,
                whiff_percent,
                k_percent,
                put_away
            FROM hitterpitcharsenalstats_2025
            WHERE team_name_alt IS NOT NULL
              AND pitch_type IS NOT NULL
            """
        )
        hitter_pitch_df = pd.DataFrame(cursor.fetchall())
        team_pitch_profiles: Dict[tuple, Dict[str, float]] = {}
        if not hitter_pitch_df.empty:
            hitter_pitch_df['team_normalized'] = hitter_pitch_df['team_name_alt'].apply(_normalize_team_code)
            hitter_pitch_df['pa_weight'] = pd.to_numeric(hitter_pitch_df['pa'], errors='coerce').fillna(0.0)
            for col in ('whiff_percent', 'k_percent', 'put_away'):
                hitter_pitch_df[col] = pd.to_numeric(hitter_pitch_df[col], errors='coerce')

            for (team_code, pitch_type), grp in hitter_pitch_df.groupby(['team_normalized', 'pitch_type'], dropna=False):
                weight = grp['pa_weight'].clip(lower=0)
                weight_sum = weight.sum()
                team_pitch_profiles[(team_code, pitch_type)] = {
                    'opp_team_whiff_percent': float(np.average(grp['whiff_percent'].fillna(0), weights=weight)) if weight_sum > 0 else np.nan,
                    'opp_team_k_percent': float(np.average(grp['k_percent'].fillna(0), weights=weight)) if weight_sum > 0 else np.nan,
                    'opp_team_put_away': float(np.average(grp['put_away'].fillna(0), weights=weight)) if weight_sum > 0 else np.nan,
                }

        starter_df = starter_df.copy()
        arsenal_records = []
        for idx, row in starter_df.iterrows():
            team_code = _normalize_team_code(row.get('team'))
            opponent_code = _normalize_team_code(row.get('opponent_team'))
            profile = starter_profiles.get((row.get('starter_name_normalized'), team_code))
            if profile is None:
                profile = starter_profiles.get((row.get('starter_name_normalized'), None))

            primary_pitch = profile.get('primary_pitch_type') if profile else None
            secondary_pitch = profile.get('secondary_pitch_type') if profile else None
            opp_primary = team_pitch_profiles.get((opponent_code, primary_pitch), {}) if primary_pitch else {}
            opp_secondary = team_pitch_profiles.get((opponent_code, secondary_pitch), {}) if secondary_pitch else {}

            primary_usage = profile.get('starter_primary_pitch_usage') if profile else np.nan
            secondary_usage = profile.get('starter_secondary_pitch_usage') if profile else np.nan
            usage_weights = np.array([
                primary_usage if primary_usage is not None and not pd.isna(primary_usage) else 0.0,
                secondary_usage if secondary_usage is not None and not pd.isna(secondary_usage) else 0.0,
            ], dtype=float)

            def _weighted_opp(metric: str):
                values = np.array([
                    opp_primary.get(metric, np.nan),
                    opp_secondary.get(metric, np.nan),
                ], dtype=float)
                mask = ~np.isnan(values) & (usage_weights > 0)
                if not mask.any():
                    return np.nan
                return float(np.average(values[mask], weights=usage_weights[mask]))

            arsenal_records.append({
                'row_index': idx,
                'starter_primary_pitch_usage': profile.get('starter_primary_pitch_usage') if profile else np.nan,
                'starter_primary_pitch_whiff_percent': profile.get('starter_primary_pitch_whiff_percent') if profile else np.nan,
                'starter_primary_pitch_k_percent': profile.get('starter_primary_pitch_k_percent') if profile else np.nan,
                'starter_primary_pitch_put_away': profile.get('starter_primary_pitch_put_away') if profile else np.nan,
                'starter_secondary_pitch_usage': profile.get('starter_secondary_pitch_usage') if profile else np.nan,
                'starter_secondary_pitch_whiff_percent': profile.get('starter_secondary_pitch_whiff_percent') if profile else np.nan,
                'starter_secondary_pitch_k_percent': profile.get('starter_secondary_pitch_k_percent') if profile else np.nan,
                'starter_secondary_pitch_put_away': profile.get('starter_secondary_pitch_put_away') if profile else np.nan,
                'starter_arsenal_whiff_percent': profile.get('starter_arsenal_whiff_percent') if profile else np.nan,
                'starter_arsenal_k_percent': profile.get('starter_arsenal_k_percent') if profile else np.nan,
                'starter_arsenal_put_away': profile.get('starter_arsenal_put_away') if profile else np.nan,
                'opp_team_k_vs_primary_pitch': opp_primary.get('opp_team_k_percent', np.nan),
                'opp_team_whiff_vs_primary_pitch': opp_primary.get('opp_team_whiff_percent', np.nan),
                'opp_team_put_away_vs_primary_pitch': opp_primary.get('opp_team_put_away', np.nan),
                'opp_team_k_vs_secondary_pitch': opp_secondary.get('opp_team_k_percent', np.nan),
                'opp_team_whiff_vs_secondary_pitch': opp_secondary.get('opp_team_whiff_percent', np.nan),
                'opp_team_put_away_vs_secondary_pitch': opp_secondary.get('opp_team_put_away', np.nan),
                'opp_team_k_vs_starter_arsenal': _weighted_opp('opp_team_k_percent'),
                'opp_team_whiff_vs_starter_arsenal': _weighted_opp('opp_team_whiff_percent'),
                'opp_team_put_away_vs_starter_arsenal': _weighted_opp('opp_team_put_away'),
            })

        arsenal_join = pd.DataFrame(arsenal_records).set_index('row_index')
        starter_df = starter_df.join(arsenal_join, how='left')
        logger.info("Attached starter arsenal and opponent pitch-type features")
        return starter_df

    def _load_team_batting_history(self) -> pd.DataFrame:
        cursor = self.conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute(
            """
            WITH team_plays AS (
                SELECT
                    g.game_date,
                    CASE
                        WHEN pp.inning_half ILIKE 'Bottom%%' THEN g.home_team
                        ELSE g.away_team
                    END AS batting_team,
                    COUNT(*) AS pa,
                    COUNT(*) FILTER (WHERE pp.play_result = 'Strikeout') AS strikeouts,
                    COUNT(*) FILTER (WHERE pp.play_result IN ('Single','Double','Triple','Home Run')) AS hits
                FROM play_by_play_plays pp
                JOIN games g ON pp.game_id = g.game_id
                WHERE g.game_date IS NOT NULL
                GROUP BY g.game_date, batting_team
            )
            SELECT
                game_date,
                batting_team,
                pa,
                strikeouts,
                hits
            FROM team_plays
            ORDER BY game_date, batting_team
            """
        )
        df = pd.DataFrame(cursor.fetchall())
        if df.empty:
            return df
        df['game_date'] = pd.to_datetime(df['game_date'])
        return df

    def _attach_opponent_team_form(self, starter_df: pd.DataFrame) -> pd.DataFrame:
        team_hist = self._load_team_batting_history()
        if team_hist.empty:
            for col in (
                'opp_team_k_rate_14d',
                'opp_team_hits_per_pa_14d',
                'opp_team_pa_per_game_14d',
                'opp_team_games_14d',
            ):
                starter_df[col] = np.nan
            return starter_df

        starter_df = starter_df.copy()
        starter_df['game_date_ts'] = pd.to_datetime(starter_df['game_date'])

        form_records = []
        for opponent_team, rows in starter_df.groupby('opponent_team'):
            team_rows = team_hist[team_hist['batting_team'] == opponent_team].copy()
            team_rows = team_rows.sort_values('game_date')

            for idx, row in rows.sort_values('game_date_ts').iterrows():
                start_date = row['game_date_ts'] - timedelta(days=14)
                mask = (team_rows['game_date'] < row['game_date_ts']) & (team_rows['game_date'] >= start_date)
                prior = team_rows.loc[mask]

                total_pa = prior['pa'].astype(float).sum() if not prior.empty else 0.0
                total_so = prior['strikeouts'].astype(float).sum() if not prior.empty else 0.0
                total_hits = prior['hits'].astype(float).sum() if not prior.empty else 0.0

                form_records.append({
                    'row_index': idx,
                    'opp_team_games_14d': int(len(prior)),
                    'opp_team_pa_per_game_14d': float(prior['pa'].astype(float).mean()) if not prior.empty else np.nan,
                    'opp_team_k_rate_14d': float(total_so / total_pa) if total_pa > 0 else np.nan,
                    'opp_team_hits_per_pa_14d': float(total_hits / total_pa) if total_pa > 0 else np.nan,
                })

        form_df = pd.DataFrame(form_records).set_index('row_index')
        starter_df = starter_df.join(form_df, how='left')
        starter_df = starter_df.drop(columns=['game_date_ts'])
        logger.info("Attached opponent team recent batting form")
        return starter_df


if __name__ == '__main__':
    builder = PitcherStrikeoutDatasetBuilder()
    try:
        dataset = builder.build()
        if not dataset.empty:
            builder.save_dataset(dataset)
    finally:
        builder.close()
