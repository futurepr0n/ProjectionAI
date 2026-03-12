#!/usr/bin/env python3
"""
Training dataset builder for ProjectionAI.

Row spine: play_by_play_plays aggregated per (game_id, batter).
Full player names from PBP eliminate the abbreviated-name mismatch
that hitting_stats introduces. No dependency on hellraiser_picks for training.

hellraiser_picks is used only in build_for_prediction() at inference time.
"""
import argparse
import psycopg2
from psycopg2.extras import RealDictCursor
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from pathlib import Path
import os
from collections import defaultdict

from name_utils import normalize_name, normalize_to_canonical
from feature_engineering import add_park_factors, add_travel_fatigue

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BATTER_ACTION_RE = (
    r"\s+(hit|struck|walk|fly|bunt|by|grounded|lined|flied|popped|"
    r"singled|doubled|tripled|homered|reached|sacrifice|intentional|"
    r"called|swinging)\b.*$"
)

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

PITCH_TYPE_ALIASES = {
    'four-seam fb': 'FF',
    '4-seam fb': 'FF',
    'four-seam fastball': 'FF',
    'ff': 'FF',
    'sinker': 'SI',
    'si': 'SI',
    'slider': 'SL',
    'sl': 'SL',
    'sweeper': 'ST',
    'st': 'ST',
    'curve': 'CU',
    'curveball': 'CU',
    'cu': 'CU',
    'changeup': 'CH',
    'change-up': 'CH',
    'ch': 'CH',
    'cutter': 'FC',
    'fc': 'FC',
    'splitter': 'FS',
    'fs': 'FS',
    'slurve': 'SV',
    'sv': 'SV',
}

DEFAULT_RECENT_LOOKBACK_GAMES = 20
RECENT_FORM_WINDOWS = (3, 5, 10, 20)
RECENT_FORM_SHRINKAGE_PA = 20.0


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


def _pitcher_name_key(name: str) -> str:
    cleaned = str(name or '').strip().lower()
    parts = cleaned.split()
    if len(parts) >= 2:
        return f"{parts[0][:1]} {parts[-1]}"
    return cleaned


def _normalize_pitch_type(value: str) -> str:
    text = str(value or '').strip().lower()
    return PITCH_TYPE_ALIASES.get(text, str(value or '').strip().upper()[:2])


class DatasetBuilder:
    def __init__(self, recent_lookback_games: int = DEFAULT_RECENT_LOOKBACK_GAMES):
        self.conn = self._connect()
        self.recent_lookback_games = max(1, int(recent_lookback_games or DEFAULT_RECENT_LOOKBACK_GAMES))
        self._canonical_cache = {}
        self._resolution_cache = {}
        self._resolution_stats = defaultdict(lambda: {
            'distinct_names': 0,
            'matched': 0,
            'unmatched': 0,
            'ambiguous': 0,
            'by_type': defaultdict(int),
            'samples': defaultdict(list),
        })
        self._tracked_resolution_keys = set()

    def _connect(self):
        try:
            conn = psycopg2.connect(
                host=os.getenv('DB_HOST', '192.168.1.23'),
                port=int(os.getenv('DB_PORT', 5432)),
                database=os.getenv('DB_NAME', 'baseball_migration_test'),
                user=os.getenv('DB_USER', 'postgres'),
                password=os.getenv('DB_PASSWORD', 'korn5676')
            )
            logger.info("Connected to database")
            return conn
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            return None

    def close(self):
        if self.conn:
            self.conn.close()

    def _track_resolution(self, source: str, name: str, resolution: dict):
        key = (source, name)
        if key in self._tracked_resolution_keys:
            return

        self._tracked_resolution_keys.add(key)
        stats = self._resolution_stats[source]
        stats['distinct_names'] += 1
        stats['by_type'][resolution['match_type']] += 1

        if resolution['matched']:
            stats['matched'] += 1
        else:
            stats['unmatched'] += 1

        if resolution['ambiguous']:
            stats['ambiguous'] += 1

        bucket = 'matched' if resolution['matched'] else 'unmatched'
        samples = stats['samples'][bucket]
        if len(samples) < 5:
            samples.append({
                'raw': resolution['raw_name'],
                'canonical': resolution['canonical_name'],
                'match_type': resolution['match_type']
            })

    def _log_resolution_summary(self):
        if not self._resolution_stats:
            return

        logger.info("Name resolution summary by source:")
        for source, stats in sorted(self._resolution_stats.items()):
            logger.info(
                "  %s | distinct=%s matched=%s unmatched=%s ambiguous=%s match_types=%s",
                source,
                stats['distinct_names'],
                stats['matched'],
                stats['unmatched'],
                stats['ambiguous'],
                dict(stats['by_type'])
            )
            if stats['samples']['unmatched']:
                logger.info("    unmatched samples: %s", stats['samples']['unmatched'])

    def _canonical(self, name: str, source: str = 'unknown') -> str:
        if name not in self._canonical_cache:
            resolution = normalize_to_canonical(name, self.conn, return_metadata=True)
            self._resolution_cache[name] = resolution
            self._canonical_cache[name] = resolution['canonical_name']
        self._track_resolution(source, name, self._resolution_cache[name])
        return self._canonical_cache[name]

    # ------------------------------------------------------------------
    # TRAINING DATA BUILD
    # ------------------------------------------------------------------

    def build(self) -> pd.DataFrame:
        """
        Build training dataset from play-by-play data.
        Spine: one row per (game_id, batter) with outcome labels + enriched features.
        """
        df = self._load_spine()
        if df.empty:
            logger.error("Empty spine — no PBP rows found")
            return df

        game_ids = df['game_id'].unique().tolist()
        logger.info(f"Spine: {len(df)} batter-game rows across {len(game_ids)} games")

        # Canonicalize batter names once
        df['batter_canonical'] = df['player_name'].apply(lambda n: self._canonical(n, 'play_by_play_spine'))

        # Season-level Statcast: EV + xStats
        df = self._attach_hitter_ev(df)
        df = self._attach_xstats(df)
        df = self._attach_handedness_features(df)
        df = self._attach_lineup_context(df)

        # Rolling 14-day rates per batter as of each game_date (no leakage)
        df = self._attach_recent_rates(df)

        # Opposing pitcher rolling 30-day stats per (game_id, opp_team)
        df = self._attach_pitcher_stats(df)
        df = self._attach_pitch_type_matchups(df)
        df = self._attach_batter_vs_pitcher_history(df)

        # Context: park, travel, weather defaults
        df = add_park_factors(df, self.conn)
        df = add_travel_fatigue(df, self.conn)
        df = self._attach_historical_weather(df)
        df = self._attach_hr_weather_interactions(df)
        df = self._attach_recent_context_interactions(df)

        drop_cols = ['batter_canonical', 'opp_team']
        df = df.drop(columns=[c for c in drop_cols if c in df.columns])

        hr_rate  = df['label'].mean() * 100
        hit_rate = df['label_hit'].mean() * 100
        so_rate  = df['label_so'].mean() * 100
        logger.info(
            f"Dataset: {len(df)} rows | HR: {hr_rate:.2f}% | "
            f"Hit: {hit_rate:.2f}% | SO: {so_rate:.2f}%"
        )
        self._log_resolution_summary()
        return df

    def _load_spine(self) -> pd.DataFrame:
        """
        Aggregate PBP per (game, batter):
          - outcome labels (HR / hit / SO)
          - game context (date, home/away teams, is_home, pitcher faced most)
        Top half = away team bats; Bottom half = home team bats.
        """
        self.conn.rollback()
        cursor = self.conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute("""
            WITH cleaned_pbp AS (
                SELECT
                    pp.game_id,
                    g.game_date,
                    g.home_team,
                    g.away_team,
                    regexp_replace(trim(pp.batter), %s, '', 'i')                    AS player_name,
                    pp.inning_half,
                    pp.pitcher,
                    pp.play_result
                FROM play_by_play_plays pp
                JOIN games g ON pp.game_id = g.game_id
                WHERE pp.batter IS NOT NULL
                  AND trim(pp.batter) <> ''
                  AND pp.play_result IS NOT NULL
            ),
            aggregated_pbp AS (
                SELECT
                    cp.game_id,
                    cp.game_date,
                    cp.home_team,
                    cp.away_team,
                    cp.player_name,
                    MODE() WITHIN GROUP (ORDER BY cp.inning_half)                    AS inning_half_mode,
                    MODE() WITHIN GROUP (ORDER BY cp.pitcher)                        AS pitcher_name,
                    COUNT(CASE WHEN cp.play_result = 'Home Run' THEN 1 END)          AS hr_count,
                    COUNT(CASE WHEN cp.play_result IN ('Single','Double','Triple','Home Run') THEN 1 END) AS hit_count,
                    COUNT(CASE WHEN cp.play_result = 'Strikeout' THEN 1 END)         AS so_count,
                    COUNT(CASE WHEN cp.play_result <> 'Other' THEN 1 END)            AS pa_count
                FROM cleaned_pbp cp
                WHERE cp.player_name IS NOT NULL
                  AND cp.player_name <> ''
                  AND lower(cp.player_name) <> 'unknown'
                GROUP BY cp.game_id, cp.player_name, cp.game_date, cp.home_team, cp.away_team
            )
            SELECT
                ap.game_id,
                ap.game_date,
                ap.home_team,
                ap.away_team,
                ap.player_name,
                CASE
                    WHEN ps.team = ap.home_team THEN ap.away_team
                    WHEN ps.team = ap.away_team THEN ap.home_team
                    WHEN ap.inning_half_mode ILIKE 'Bottom%%' THEN ap.home_team
                    ELSE ap.away_team
                END                                                                  AS team,
                CASE
                    WHEN ps.team = ap.away_team THEN true
                    WHEN ps.team = ap.home_team THEN false
                    WHEN ap.inning_half_mode ILIKE 'Bottom%%' THEN true
                    ELSE false
                END                                                                  AS is_home,
                ap.pitcher_name,
                ap.hr_count,
                ap.hit_count,
                ap.so_count,
                ap.pa_count
            FROM aggregated_pbp ap
            LEFT JOIN LATERAL (
                SELECT ps.team
                FROM pitching_stats ps
                WHERE ps.game_id = ap.game_id
                  AND ps.player_name = ap.pitcher_name
                ORDER BY COALESCE(ps.innings_pitched, 0) DESC, ps.id
                LIMIT 1
            ) ps ON true
            WHERE ap.pa_count >= 1
            ORDER BY ap.game_date, ap.player_name
        """, (BATTER_ACTION_RE,))
        data = cursor.fetchall()
        df = pd.DataFrame(data) if data else pd.DataFrame()

        if not df.empty:
            df['label']     = (df['hr_count'].fillna(0)  > 0).astype(int)
            df['label_hit'] = (df['hit_count'].fillna(0) > 0).astype(int)
            df['label_so']  = (df['so_count'].fillna(0)  > 0).astype(int)

        logger.info(f"Loaded PBP spine: {len(df)} batter-game rows")
        return df

    def _attach_hitter_ev(self, df: pd.DataFrame) -> pd.DataFrame:
        """Season-level exit velocity / barrel / sweet-spot."""
        self.conn.rollback()
        cursor = self.conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute("""
            SELECT last_name_first_name          AS raw_name,
                   avg_hit_speed_numeric         AS avg_ev,
                   brl_percent_numeric           AS barrel_rate,
                   anglesweetspotpercent_numeric AS sweet_spot_percent
            FROM hitter_exit_velocity
            WHERE last_name_first_name IS NOT NULL
        """)
        ev = pd.DataFrame(cursor.fetchall())
        logger.info(f"Loaded EV stats for {len(ev)} hitters")

        if ev.empty:
            for col in ('avg_ev', 'barrel_rate', 'sweet_spot_percent'):
                df[col] = np.nan
            return df

        ev['canonical'] = ev['raw_name'].apply(lambda n: self._canonical(n, 'hitter_exit_velocity'))
        ev_map = ev.drop_duplicates('canonical').set_index('canonical')

        df['avg_ev']             = df['batter_canonical'].map(ev_map['avg_ev'])
        df['barrel_rate']        = df['batter_canonical'].map(ev_map['barrel_rate'])
        df['sweet_spot_percent'] = df['batter_canonical'].map(ev_map['sweet_spot_percent'])

        logger.info(f"EV match rate: {df['avg_ev'].notna().mean()*100:.1f}%")
        return df

    def _attach_xstats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Season-level xwOBA / xBA / xSLG."""
        self.conn.rollback()
        cursor = self.conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute("""
            SELECT last_name_first_name AS raw_name, xwoba, xba, xslg
            FROM custom_batter_2025
            WHERE last_name_first_name IS NOT NULL
        """)
        xs = pd.DataFrame(cursor.fetchall())
        logger.info(f"Loaded xstats for {len(xs)} batters")

        if xs.empty:
            for col in ('xwoba', 'xba', 'xslg'):
                df[col] = np.nan
            return df

        xs['canonical'] = xs['raw_name'].apply(lambda n: self._canonical(n, 'custom_batter_2025'))
        xs_map = xs.drop_duplicates('canonical').set_index('canonical')

        df['xwoba'] = df['batter_canonical'].map(xs_map['xwoba'])
        df['xba']   = df['batter_canonical'].map(xs_map['xba'])
        df['xslg']  = df['batter_canonical'].map(xs_map['xslg'])

        logger.info(f"xStats match rate: {df['xwoba'].notna().mean()*100:.1f}%")
        return df

    def _attach_handedness_features(self, df: pd.DataFrame) -> pd.DataFrame:
        feature_cols = [
            'batter_bats_right',
            'batter_bats_left',
            'batter_is_switch',
            'pitcher_throws_right',
            'pitcher_throws_left',
            'same_hand_matchup',
            'platoon_advantage',
        ]
        if df.empty:
            for col in feature_cols:
                df[col] = np.nan
            return df

        self.conn.rollback()
        cursor = self.conn.cursor(cursor_factory=RealDictCursor)

        cursor.execute("""
            SELECT
                COALESCE(full_name, display_name) AS full_name,
                bats,
                throws,
                player_type
            FROM players
            WHERE COALESCE(full_name, display_name) IS NOT NULL
        """)
        players_df = pd.DataFrame(cursor.fetchall())

        batter_lookup = {}
        if not players_df.empty:
            players_df['normalized_name'] = players_df['full_name'].apply(normalize_name)
            players_df['canonical'] = players_df['full_name'].apply(lambda n: self._canonical(n, 'players'))
            players_df['bats'] = players_df['bats'].astype(str).str.upper().str.strip()
            players_df['throws'] = players_df['throws'].astype(str).str.upper().str.strip()
            for _, row in players_df.drop_duplicates('canonical').iterrows():
                batter_lookup[row['canonical']] = row['bats']

        pitcher_lookup = {}
        if not players_df.empty:
            pitcher_rows = players_df[players_df['throws'].isin(['L', 'R'])].copy()
            for _, row in pitcher_rows.drop_duplicates('normalized_name').iterrows():
                pitcher_lookup[row['normalized_name']] = row['throws']

        batter_bats = df['batter_canonical'].map(batter_lookup)
        pitcher_throws = df['pitcher_name'].fillna('').apply(normalize_name).map(pitcher_lookup)

        df['batter_bats_right'] = (batter_bats == 'R').astype(float)
        df['batter_bats_left'] = (batter_bats == 'L').astype(float)
        df['batter_is_switch'] = (batter_bats == 'S').astype(float)
        df['pitcher_throws_right'] = (pitcher_throws == 'R').astype(float)
        df['pitcher_throws_left'] = (pitcher_throws == 'L').astype(float)
        df['same_hand_matchup'] = np.where(
            batter_bats.isin(['L', 'R']) & pitcher_throws.isin(['L', 'R']),
            (batter_bats == pitcher_throws).astype(float),
            np.where(batter_bats == 'S', 0.0, np.nan)
        )
        df['platoon_advantage'] = np.where(
            batter_bats.isin(['L', 'R']) & pitcher_throws.isin(['L', 'R']),
            (batter_bats != pitcher_throws).astype(float),
            np.where(batter_bats == 'S', 1.0, np.nan)
        )

        logger.info(
            "Attached handedness features: batter bats %.1f%% | pitcher throws %.1f%%",
            batter_bats.notna().mean() * 100,
            pitcher_throws.notna().mean() * 100,
        )
        return df

    def _attach_lineup_context(self, df: pd.DataFrame) -> pd.DataFrame:
        feature_cols = [
            'lineup_confirmed',
            'lineup_slot',
            'top_half_lineup',
            'middle_lineup',
            'bottom_half_lineup',
            'projected_pa',
        ]
        if df.empty:
            for col in feature_cols:
                df[col] = np.nan
            return df

        self.conn.rollback()
        cursor = self.conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute("""
            SELECT
                game_date,
                home_team,
                away_team,
                home_lineup,
                away_lineup
            FROM daily_lineups
            WHERE home_lineup IS NOT NULL
               OR away_lineup IS NOT NULL
        """)
        lineups_df = pd.DataFrame(cursor.fetchall())
        if lineups_df.empty:
            for col in feature_cols:
                df[col] = np.nan
            return df

        def _iter_batting_order(lineup):
            if not isinstance(lineup, dict):
                return []
            batting_order = lineup.get('batting_order', []) or []
            results = []
            for idx, entry in enumerate(batting_order, start=1):
                if isinstance(entry, dict):
                    name = (
                        entry.get('name')
                        or entry.get('player_name')
                        or entry.get('fullName')
                        or entry.get('player')
                    )
                else:
                    name = entry
                if not name:
                    continue
                results.append((idx, str(name).strip()))
            return results

        lineup_records = []
        pa_by_slot = {
            1: 4.8,
            2: 4.7,
            3: 4.6,
            4: 4.5,
            5: 4.3,
            6: 4.2,
            7: 4.0,
            8: 3.9,
            9: 3.8,
        }

        for _, row in lineups_df.iterrows():
            for side in ('home', 'away'):
                lineup = row.get(f'{side}_lineup')
                team = row.get(f'{side}_team')
                confirmed = bool(lineup.get('confirmed')) if isinstance(lineup, dict) else False
                for slot, raw_name in _iter_batting_order(lineup):
                    lineup_records.append({
                        'game_date': row['game_date'],
                        'team': team,
                        'batter_canonical': self._canonical(raw_name, 'daily_lineups'),
                        'lineup_confirmed': float(confirmed),
                        'lineup_slot': float(slot),
                        'top_half_lineup': float(slot <= 3),
                        'middle_lineup': float(4 <= slot <= 6),
                        'bottom_half_lineup': float(slot >= 7),
                        'projected_pa': pa_by_slot.get(slot, 4.0),
                    })

        lineup_lookup = pd.DataFrame(lineup_records)
        if lineup_lookup.empty:
            for col in feature_cols:
                df[col] = np.nan
            return df

        lineup_lookup['game_date'] = pd.to_datetime(lineup_lookup['game_date']).dt.date
        lineup_lookup['team'] = lineup_lookup['team'].apply(_normalize_team_code)
        lineup_lookup = lineup_lookup.drop_duplicates(['game_date', 'team', 'batter_canonical'])

        merge_source = df.copy()
        merge_source['team'] = merge_source['team'].apply(_normalize_team_code)

        merged = merge_source.merge(
            lineup_lookup,
            on=['game_date', 'team', 'batter_canonical'],
            how='left'
        )
        logger.info(
            "Attached lineup context: %.1f%% rows matched a lineup slot",
            merged['lineup_slot'].notna().mean() * 100
        )
        return merged

    def _attach_recent_rates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rolling last-N-games HR / hit / SO rates per batter, computed per unique game_date."""
        dates = sorted(df['game_date'].unique().tolist())
        cursor = self.conn.cursor(cursor_factory=RealDictCursor)
        rate_rows = []
        recency_sql = self._recent_rate_select_sql()

        for gd in dates:
            self.conn.rollback()
            cursor.execute(f"""
                WITH batter_games AS (
                    SELECT
                        pp.batter,
                        g.game_date,
                        pp.game_id,
                        MAX(CASE WHEN pp.play_result = 'Home Run' THEN 1 ELSE 0 END)::numeric AS hr_game,
                        MAX(CASE WHEN pp.play_result IN ('Single','Double','Triple','Home Run') THEN 1 ELSE 0 END)::numeric AS hit_game,
                        MAX(CASE WHEN pp.play_result = 'Strikeout' THEN 1 ELSE 0 END)::numeric AS so_game,
                        COUNT(*) FILTER (
                            WHERE pp.play_result <> 'Other'
                              AND pp.play_result IS NOT NULL
                        )::numeric AS pa_game
                    FROM play_by_play_plays pp
                    JOIN games g ON pp.game_id = g.game_id
                    WHERE g.game_date < %s::date
                      AND pp.batter IS NOT NULL
                    GROUP BY pp.batter, g.game_date, pp.game_id
                ),
                ranked_games AS (
                    SELECT
                        batter,
                        game_date,
                        hr_game,
                        hit_game,
                        so_game,
                        pa_game,
                        ROW_NUMBER() OVER (
                            PARTITION BY batter
                            ORDER BY game_date DESC, game_id DESC
                        ) AS rn
                    FROM batter_games
                )
                SELECT
                    batter,
                    MAX(game_date) AS last_game_date,
                    AVG(hr_game) AS season_hr_rate_to_date,
                    AVG(hit_game) AS season_hit_rate_to_date,
                    AVG(so_game) AS season_so_rate_to_date,
                    COUNT(*) AS season_games_prior,
                    SUM(pa_game) AS season_pa_prior,
                    {recency_sql}
                FROM ranked_games
                GROUP BY batter
            """, (gd, *self._recent_rate_query_params()))
            for row in cursor.fetchall():
                row_dict = {
                    'game_date':        gd,
                    'batter_canonical': self._canonical(row['batter'], 'play_by_play_recent_rates'),
                    'recent_hr_rate_14d':  self._optional_float(row.get('recent_hr_rate_14d')),
                    'recent_hit_rate_14d': self._optional_float(row.get('recent_hit_rate_14d')),
                    'recent_so_rate_14d':  self._optional_float(row.get('recent_so_rate_14d')),
                    'recent_form_lookback_games': self.recent_lookback_games,
                    'recent_form_games_used': self._optional_float(row.get('recent_form_games_used')),
                    'recent_form_pa_used': self._optional_float(row.get('recent_form_pa_used')),
                    'days_since_last_game': self._days_since(gd, row.get('last_game_date')),
                    'season_hr_rate_to_date': self._optional_float(row.get('season_hr_rate_to_date')),
                    'season_hit_rate_to_date': self._optional_float(row.get('season_hit_rate_to_date')),
                    'season_so_rate_to_date': self._optional_float(row.get('season_so_rate_to_date')),
                    'season_games_prior': self._optional_float(row.get('season_games_prior')),
                    'season_pa_prior': self._optional_float(row.get('season_pa_prior')),
                }
                for window in RECENT_FORM_WINDOWS:
                    suffix = f"g{window}"
                    row_dict[f'recent_hr_rate_{suffix}'] = self._optional_float(row.get(f'recent_hr_rate_{suffix}'))
                    row_dict[f'recent_hit_rate_{suffix}'] = self._optional_float(row.get(f'recent_hit_rate_{suffix}'))
                    row_dict[f'recent_so_rate_{suffix}'] = self._optional_float(row.get(f'recent_so_rate_{suffix}'))
                    row_dict[f'recent_games_used_{suffix}'] = self._optional_float(row.get(f'recent_games_used_{suffix}'))
                    row_dict[f'recent_pa_used_{suffix}'] = self._optional_float(row.get(f'recent_pa_used_{suffix}'))
                row_dict.update(self._recent_shrinkage_features(row_dict))
                rate_rows.append(row_dict)

        if not rate_rows:
            for col in (
                'recent_hr_rate_14d',
                'recent_hit_rate_14d',
                'recent_so_rate_14d',
                'recent_form_games_used',
                'recent_form_pa_used',
                'days_since_last_game',
                'season_hr_rate_to_date',
                'season_hit_rate_to_date',
                'season_so_rate_to_date',
                'season_games_prior',
                'season_pa_prior',
                'shrunk_recent_hr_rate',
                'shrunk_recent_hit_rate',
                'shrunk_recent_so_rate',
                'recent_vs_season_hr_delta',
                'recent_vs_season_hit_delta',
                'recent_vs_season_so_delta',
            ):
                df[col] = np.nan
            for window in RECENT_FORM_WINDOWS:
                suffix = f"g{window}"
                for col in (
                    f'recent_hr_rate_{suffix}',
                    f'recent_hit_rate_{suffix}',
                    f'recent_so_rate_{suffix}',
                    f'recent_games_used_{suffix}',
                    f'recent_pa_used_{suffix}',
                ):
                    df[col] = np.nan
            return df

        rates = pd.DataFrame(rate_rows).drop_duplicates(subset=['game_date', 'batter_canonical'])
        df = df.merge(rates, on=['game_date', 'batter_canonical'], how='left')
        logger.info(
            "Recent rates match rate: %.1f%% using last %s games",
            df['recent_hr_rate_14d'].notna().mean() * 100,
            self.recent_lookback_games,
        )
        return df

    def _recent_rate_select_sql(self) -> str:
        fields = []
        for window in RECENT_FORM_WINDOWS:
            suffix = f"g{window}"
            fields.extend([
                f"AVG(hr_game) FILTER (WHERE rn <= {window}) AS recent_hr_rate_{suffix}",
                f"AVG(hit_game) FILTER (WHERE rn <= {window}) AS recent_hit_rate_{suffix}",
                f"AVG(so_game) FILTER (WHERE rn <= {window}) AS recent_so_rate_{suffix}",
                f"COUNT(*) FILTER (WHERE rn <= {window}) AS recent_games_used_{suffix}",
                f"SUM(pa_game) FILTER (WHERE rn <= {window}) AS recent_pa_used_{suffix}",
            ])
        fields.extend([
            f"AVG(hr_game) FILTER (WHERE rn <= %s) AS recent_hr_rate_14d",
            f"AVG(hit_game) FILTER (WHERE rn <= %s) AS recent_hit_rate_14d",
            f"AVG(so_game) FILTER (WHERE rn <= %s) AS recent_so_rate_14d",
            f"COUNT(*) FILTER (WHERE rn <= %s) AS recent_form_games_used",
            f"SUM(pa_game) FILTER (WHERE rn <= %s) AS recent_form_pa_used",
        ])
        return ",\n                    ".join(fields)

    def _recent_rate_query_params(self):
        return [self.recent_lookback_games] * 5

    @staticmethod
    def _optional_float(value):
        if value is None or pd.isna(value):
            return None
        return float(value)

    @staticmethod
    def _days_since(target_date, last_game_date):
        if last_game_date is None or pd.isna(last_game_date):
            return None
        return float((pd.Timestamp(target_date) - pd.Timestamp(last_game_date)).days)

    @staticmethod
    def _shrunk_rate(recent_rate, recent_pa, season_rate, shrink_pa=RECENT_FORM_SHRINKAGE_PA):
        if recent_rate is None or season_rate is None:
            return None
        recent_pa = float(recent_pa or 0.0)
        return float(((recent_rate * recent_pa) + (season_rate * shrink_pa)) / (recent_pa + shrink_pa))

    def _recent_shrinkage_features(self, row_dict: dict) -> dict:
        recent_pa = row_dict.get('recent_form_pa_used')
        season_hr = row_dict.get('season_hr_rate_to_date')
        season_hit = row_dict.get('season_hit_rate_to_date')
        season_so = row_dict.get('season_so_rate_to_date')
        recent_hr = row_dict.get('recent_hr_rate_14d')
        recent_hit = row_dict.get('recent_hit_rate_14d')
        recent_so = row_dict.get('recent_so_rate_14d')
        return {
            'shrunk_recent_hr_rate': self._shrunk_rate(recent_hr, recent_pa, season_hr),
            'shrunk_recent_hit_rate': self._shrunk_rate(recent_hit, recent_pa, season_hit),
            'shrunk_recent_so_rate': self._shrunk_rate(recent_so, recent_pa, season_so),
            'recent_vs_season_hr_delta': None if recent_hr is None or season_hr is None else float(recent_hr - season_hr),
            'recent_vs_season_hit_delta': None if recent_hit is None or season_hit is None else float(recent_hit - season_hit),
            'recent_vs_season_so_delta': None if recent_so is None or season_so is None else float(recent_so - season_so),
        }

    def _attach_pitcher_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Rolling 30-day pitching stats for the opposing team as of each game_date.
        Keyed on (game_id, opp_team) to avoid cartesian product on merge.
        """
        df['opp_team'] = df.apply(
            lambda r: r['away_team'] if r['is_home'] else r['home_team'], axis=1
        )

        pairs = df[['game_id', 'game_date', 'opp_team']].drop_duplicates()
        cursor = self.conn.cursor(cursor_factory=RealDictCursor)
        results = []

        for _, row in pairs.iterrows():
            self.conn.rollback()
            cursor.execute("""
                SELECT
                    AVG(ps.earned_runs::numeric / NULLIF(ps.innings_pitched::numeric,0) * 9) AS pitcher_era_30d,
                    AVG(ps.home_runs::numeric  / NULLIF(ps.innings_pitched::numeric,0) * 9) AS pitcher_hr_per_9_30d,
                    AVG(ps.strikeouts::numeric / NULLIF(ps.innings_pitched::numeric,0) * 9) AS pitcher_k_per_9_30d,
                    AVG((ps.hits::numeric + ps.walks::numeric) / NULLIF(ps.innings_pitched::numeric,0)) AS pitcher_whip_30d
                FROM pitching_stats ps
                JOIN games g ON ps.game_id = g.game_id
                WHERE ps.team = %s
                  AND g.game_date >= %s::date - INTERVAL '30 days'
                  AND g.game_date <  %s::date
                  AND ps.innings_pitched > 0
            """, (row['opp_team'], row['game_date'], row['game_date']))
            stat = cursor.fetchone()
            results.append({
                'game_id':              row['game_id'],
                'opp_team':             row['opp_team'],
                'pitcher_era_30d':      float(stat['pitcher_era_30d'])      if stat and stat['pitcher_era_30d']      else None,
                'pitcher_hr_per_9_30d': float(stat['pitcher_hr_per_9_30d']) if stat and stat['pitcher_hr_per_9_30d'] else None,
                'pitcher_k_per_9_30d':  float(stat['pitcher_k_per_9_30d'])  if stat and stat['pitcher_k_per_9_30d']  else None,
                'pitcher_whip_30d':     float(stat['pitcher_whip_30d'])     if stat and stat['pitcher_whip_30d']     else None,
            })

        pitcher_df = pd.DataFrame(results)
        # Merge on (game_id, opp_team) to avoid cartesian product
        df = df.merge(pitcher_df, on=['game_id', 'opp_team'], how='left')
        logger.info(f"Attached pitcher rolling stats for {len(pitcher_df)} game-team pairs")
        return df

    def _attach_recent_context_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df

        updated = df.copy()

        def num(col, default=0.0):
            return pd.to_numeric(updated.get(col), errors='coerce').fillna(default)

        projected_pa = num('projected_pa', 4.2)
        platoon_advantage = num('platoon_advantage', 0.0)
        top_half = num('top_half_lineup', 0.0)
        middle_half = num('middle_lineup', 0.0)
        lineup_confirmed = num('lineup_confirmed', 0.0)

        recent_hr = num('recent_hr_rate_14d', 0.0)
        recent_hit = num('recent_hit_rate_14d', 0.0)
        shrunk_hr = num('shrunk_recent_hr_rate', recent_hr)
        shrunk_hit = num('shrunk_recent_hit_rate', recent_hit)
        hr_delta = num('recent_vs_season_hr_delta', 0.0)
        hit_delta = num('recent_vs_season_hit_delta', 0.0)

        updated['recent_hr_x_projected_pa'] = recent_hr * projected_pa
        updated['recent_hit_x_projected_pa'] = recent_hit * projected_pa
        updated['shrunk_recent_hr_x_projected_pa'] = shrunk_hr * projected_pa
        updated['shrunk_recent_hit_x_projected_pa'] = shrunk_hit * projected_pa
        updated['recent_hr_x_platoon_advantage'] = recent_hr * platoon_advantage
        updated['recent_hit_x_platoon_advantage'] = recent_hit * platoon_advantage
        updated['shrunk_recent_hr_x_platoon_advantage'] = shrunk_hr * platoon_advantage
        updated['shrunk_recent_hit_x_platoon_advantage'] = shrunk_hit * platoon_advantage
        updated['recent_hr_x_top_half_lineup'] = recent_hr * top_half
        updated['recent_hit_x_top_half_lineup'] = recent_hit * top_half
        updated['recent_hr_x_middle_lineup'] = recent_hr * middle_half
        updated['recent_hit_x_middle_lineup'] = recent_hit * middle_half
        updated['hr_delta_x_projected_pa'] = hr_delta * projected_pa
        updated['hit_delta_x_projected_pa'] = hit_delta * projected_pa
        updated['projected_pa_x_lineup_confirmed'] = projected_pa * lineup_confirmed
        return updated

    def _attach_pitch_type_matchups(self, df: pd.DataFrame) -> pd.DataFrame:
        self.conn.rollback()
        cursor = self.conn.cursor(cursor_factory=RealDictCursor)

        cursor.execute("""
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
        """)
        pitcher_arsenal = pd.DataFrame(cursor.fetchall())

        cursor.execute("""
            WITH pitch_events AS (
                SELECT
                    pp.pitcher,
                    CASE
                        WHEN pp.inning_half = 'Top' THEN g.home_team
                        ELSE g.away_team
                    END AS pitcher_team,
                    p.play_id,
                    p.pitch_number,
                    p.pitch_type,
                    p.result,
                    pp.play_result,
                    ROW_NUMBER() OVER (PARTITION BY p.play_id ORDER BY p.pitch_number DESC, p.id DESC) AS reverse_pitch_rank
                FROM play_by_play_pitches p
                JOIN play_by_play_plays pp ON pp.id = p.play_id
                JOIN games g ON g.game_id = pp.game_id
                WHERE pp.pitcher IS NOT NULL
                  AND p.pitch_type IS NOT NULL
            )
            SELECT
                pitcher,
                pitcher_team,
                pitch_type,
                COUNT(*) AS pitch_count,
                AVG(CASE WHEN COALESCE(result, '') ILIKE '%%swing%%strike%%' THEN 1 ELSE 0 END) AS whiff_percent,
                AVG(CASE WHEN reverse_pitch_rank = 1 AND play_result = 'Strikeout' THEN 1 ELSE 0 END) AS k_percent
            FROM pitch_events
            GROUP BY pitcher, pitcher_team, pitch_type
        """)
        fallback_pitcher_arsenal = pd.DataFrame(cursor.fetchall())

        cursor.execute("""
            SELECT
                last_name_first_name,
                team_name_alt,
                pitch_type,
                pa,
                whiff_percent,
                k_percent,
                put_away,
                ba,
                slg,
                woba
            FROM hitterpitcharsenalstats_2025
            WHERE last_name_first_name IS NOT NULL
              AND pitch_type IS NOT NULL
        """)
        hitter_arsenal = pd.DataFrame(cursor.fetchall())

        feature_cols = [
            'pitcher_primary_pitch_usage',
            'pitcher_primary_pitch_whiff_percent',
            'pitcher_primary_pitch_k_percent',
            'pitcher_secondary_pitch_usage',
            'pitcher_secondary_pitch_whiff_percent',
            'pitcher_secondary_pitch_k_percent',
            'pitcher_arsenal_whiff_percent',
            'pitcher_arsenal_k_percent',
            'batter_k_vs_primary_pitch',
            'batter_whiff_vs_primary_pitch',
            'batter_ba_vs_primary_pitch',
            'batter_slg_vs_primary_pitch',
            'batter_woba_vs_primary_pitch',
            'batter_k_vs_secondary_pitch',
            'batter_whiff_vs_secondary_pitch',
            'batter_ba_vs_secondary_pitch',
            'batter_slg_vs_secondary_pitch',
            'batter_woba_vs_secondary_pitch',
            'batter_k_vs_pitcher_arsenal',
            'batter_whiff_vs_pitcher_arsenal',
            'batter_woba_vs_pitcher_arsenal',
        ]
        if pitcher_arsenal.empty or hitter_arsenal.empty:
            for col in feature_cols:
                df[col] = np.nan
            return df

        if not fallback_pitcher_arsenal.empty:
            fallback_pitcher_arsenal['last_name_first_name'] = fallback_pitcher_arsenal['pitcher']
            fallback_pitcher_arsenal['team_name_alt'] = fallback_pitcher_arsenal['pitcher_team']
            fallback_pitcher_arsenal['pitch_type'] = fallback_pitcher_arsenal['pitch_type'].apply(_normalize_pitch_type)
            fallback_pitcher_arsenal['pitch_usage'] = pd.to_numeric(fallback_pitcher_arsenal['pitch_count'], errors='coerce')
            fallback_pitcher_arsenal['whiff_percent'] = pd.to_numeric(fallback_pitcher_arsenal['whiff_percent'], errors='coerce')
            fallback_pitcher_arsenal['k_percent'] = pd.to_numeric(fallback_pitcher_arsenal['k_percent'], errors='coerce')
            fallback_pitcher_arsenal['put_away'] = fallback_pitcher_arsenal['k_percent']
            fallback_pitcher_arsenal = fallback_pitcher_arsenal[[
                'last_name_first_name', 'team_name_alt', 'pitch_type', 'pitch_usage', 'whiff_percent', 'k_percent', 'put_away'
            ]]
            pitcher_arsenal = pd.concat([pitcher_arsenal, fallback_pitcher_arsenal], ignore_index=True)

        pitcher_arsenal['pitcher_name_normalized'] = pitcher_arsenal['last_name_first_name'].apply(
            lambda value: _name_from_last_first(value).strip().lower()
        )
        pitcher_arsenal['pitcher_canonical'] = pitcher_arsenal['last_name_first_name'].apply(
            lambda value: self._canonical(_name_from_last_first(value), 'pitcher_arsenal')
        )
        pitcher_arsenal['team_normalized'] = pitcher_arsenal['team_name_alt'].apply(_normalize_team_code)
        for col in ('pitch_usage', 'whiff_percent', 'k_percent', 'put_away'):
            pitcher_arsenal[col] = pd.to_numeric(pitcher_arsenal[col], errors='coerce')

        pitcher_profiles = {}
        pitcher_profiles_by_name = {}
        for (pitcher_canonical, team_code), grp in pitcher_arsenal.groupby(['pitcher_canonical', 'team_normalized'], dropna=False):
            grp = grp.dropna(subset=['pitch_usage']).sort_values('pitch_usage', ascending=False).copy()
            if grp.empty:
                continue
            usage_sum = grp['pitch_usage'].sum()
            primary = grp.iloc[0]
            secondary = grp.iloc[1] if len(grp) > 1 else None
            profile = {
                'primary_pitch_type': primary['pitch_type'],
                'secondary_pitch_type': secondary['pitch_type'] if secondary is not None else None,
                'pitcher_primary_pitch_usage': float(primary['pitch_usage']),
                'pitcher_primary_pitch_whiff_percent': float(primary['whiff_percent']) if pd.notna(primary['whiff_percent']) else np.nan,
                'pitcher_primary_pitch_k_percent': float(primary['k_percent']) if pd.notna(primary['k_percent']) else np.nan,
                'pitcher_secondary_pitch_usage': float(secondary['pitch_usage']) if secondary is not None and pd.notna(secondary['pitch_usage']) else np.nan,
                'pitcher_secondary_pitch_whiff_percent': float(secondary['whiff_percent']) if secondary is not None and pd.notna(secondary['whiff_percent']) else np.nan,
                'pitcher_secondary_pitch_k_percent': float(secondary['k_percent']) if secondary is not None and pd.notna(secondary['k_percent']) else np.nan,
                'pitcher_arsenal_whiff_percent': float(np.average(grp['whiff_percent'].fillna(0), weights=grp['pitch_usage'])) if usage_sum > 0 else np.nan,
                'pitcher_arsenal_k_percent': float(np.average(grp['k_percent'].fillna(0), weights=grp['pitch_usage'])) if usage_sum > 0 else np.nan,
            }
            pitcher_profiles[(pitcher_canonical, team_code)] = profile
            pitcher_profiles_by_name.setdefault(pitcher_canonical, profile)

        hitter_arsenal['batter_canonical'] = hitter_arsenal['last_name_first_name'].apply(
            lambda n: self._canonical(n, 'hitter_pitch_arsenal')
        )
        for col in ('pa', 'whiff_percent', 'k_percent', 'put_away', 'ba', 'slg', 'woba'):
            hitter_arsenal[col] = pd.to_numeric(hitter_arsenal[col], errors='coerce')

        batter_profiles = {}
        for (batter_canonical, pitch_type), grp in hitter_arsenal.groupby(['batter_canonical', 'pitch_type'], dropna=False):
            weights = grp['pa'].fillna(0.0).clip(lower=0)
            weight_sum = weights.sum()
            batter_profiles[(batter_canonical, pitch_type)] = {
                'batter_whiff_percent': float(np.average(grp['whiff_percent'].fillna(0), weights=weights)) if weight_sum > 0 else np.nan,
                'batter_k_percent': float(np.average(grp['k_percent'].fillna(0), weights=weights)) if weight_sum > 0 else np.nan,
                'batter_ba': float(np.average(grp['ba'].fillna(0), weights=weights)) if weight_sum > 0 else np.nan,
                'batter_slg': float(np.average(grp['slg'].fillna(0), weights=weights)) if weight_sum > 0 else np.nan,
                'batter_woba': float(np.average(grp['woba'].fillna(0), weights=weights)) if weight_sum > 0 else np.nan,
            }

        df = df.copy()
        df['pitcher_canonical'] = df['pitcher_name'].apply(lambda value: self._canonical(value, 'pitcher_name_matchup'))
        df['opp_team_normalized'] = df['opp_team'].apply(_normalize_team_code)
        rows = []
        for idx, row in df.iterrows():
            profile = pitcher_profiles.get((row['pitcher_canonical'], row['opp_team_normalized']))
            if profile is None:
                profile = pitcher_profiles_by_name.get(row['pitcher_canonical'])
            primary = profile.get('primary_pitch_type') if profile else None
            secondary = profile.get('secondary_pitch_type') if profile else None
            batter_primary = batter_profiles.get((row['batter_canonical'], primary), {}) if primary else {}
            batter_secondary = batter_profiles.get((row['batter_canonical'], secondary), {}) if secondary else {}

            primary_usage = profile.get('pitcher_primary_pitch_usage') if profile else np.nan
            secondary_usage = profile.get('pitcher_secondary_pitch_usage') if profile else np.nan
            usage_weights = np.array([
                primary_usage if primary_usage is not None and not pd.isna(primary_usage) else 0.0,
                secondary_usage if secondary_usage is not None and not pd.isna(secondary_usage) else 0.0,
            ], dtype=float)

            def _weighted(metric: str):
                values = np.array([
                    batter_primary.get(metric, np.nan),
                    batter_secondary.get(metric, np.nan),
                ], dtype=float)
                mask = ~np.isnan(values) & (usage_weights > 0)
                if not mask.any():
                    return np.nan
                return float(np.average(values[mask], weights=usage_weights[mask]))

            rows.append({
                'row_index': idx,
                'primary_pitch_type': primary,
                'secondary_pitch_type': secondary,
                'pitcher_primary_pitch_usage': profile.get('pitcher_primary_pitch_usage') if profile else np.nan,
                'pitcher_primary_pitch_whiff_percent': profile.get('pitcher_primary_pitch_whiff_percent') if profile else np.nan,
                'pitcher_primary_pitch_k_percent': profile.get('pitcher_primary_pitch_k_percent') if profile else np.nan,
                'pitcher_secondary_pitch_usage': profile.get('pitcher_secondary_pitch_usage') if profile else np.nan,
                'pitcher_secondary_pitch_whiff_percent': profile.get('pitcher_secondary_pitch_whiff_percent') if profile else np.nan,
                'pitcher_secondary_pitch_k_percent': profile.get('pitcher_secondary_pitch_k_percent') if profile else np.nan,
                'pitcher_arsenal_whiff_percent': profile.get('pitcher_arsenal_whiff_percent') if profile else np.nan,
                'pitcher_arsenal_k_percent': profile.get('pitcher_arsenal_k_percent') if profile else np.nan,
                'batter_k_vs_primary_pitch': batter_primary.get('batter_k_percent', np.nan),
                'batter_whiff_vs_primary_pitch': batter_primary.get('batter_whiff_percent', np.nan),
                'batter_ba_vs_primary_pitch': batter_primary.get('batter_ba', np.nan),
                'batter_slg_vs_primary_pitch': batter_primary.get('batter_slg', np.nan),
                'batter_woba_vs_primary_pitch': batter_primary.get('batter_woba', np.nan),
                'batter_k_vs_secondary_pitch': batter_secondary.get('batter_k_percent', np.nan),
                'batter_whiff_vs_secondary_pitch': batter_secondary.get('batter_whiff_percent', np.nan),
                'batter_ba_vs_secondary_pitch': batter_secondary.get('batter_ba', np.nan),
                'batter_slg_vs_secondary_pitch': batter_secondary.get('batter_slg', np.nan),
                'batter_woba_vs_secondary_pitch': batter_secondary.get('batter_woba', np.nan),
                'batter_k_vs_pitcher_arsenal': _weighted('batter_k_percent'),
                'batter_whiff_vs_pitcher_arsenal': _weighted('batter_whiff_percent'),
                'batter_woba_vs_pitcher_arsenal': _weighted('batter_woba'),
            })

        join_df = pd.DataFrame(rows).set_index('row_index')
        df = df.join(join_df, how='left')
        df = df.drop(columns=['pitcher_canonical', 'opp_team_normalized'])
        logger.info("Attached hitter pitch-type matchup features")
        return df

    def _attach_batter_vs_pitcher_history(self, df: pd.DataFrame) -> pd.DataFrame:
        feature_cols = [
            'prior_games_vs_pitcher',
            'prior_pa_vs_pitcher',
            'prior_hits_vs_pitcher',
            'prior_hr_vs_pitcher',
            'prior_so_vs_pitcher',
            'prior_hit_rate_vs_pitcher',
            'prior_hr_rate_vs_pitcher',
            'prior_so_rate_vs_pitcher',
            'prior_avg_pa_vs_pitcher',
            'days_since_last_vs_pitcher',
            'last_hits_vs_pitcher',
            'last_hr_vs_pitcher',
            'last_so_vs_pitcher',
        ]
        if df.empty:
            for col in feature_cols:
                df[col] = np.nan
            return df

        history = df.copy()
        history['game_date'] = pd.to_datetime(history['game_date'], errors='coerce')
        history = history.sort_values(['batter_canonical', 'pitcher_name', 'game_date', 'game_id']).copy()
        grouped = history.groupby(['batter_canonical', 'pitcher_name'], dropna=False)

        history['prior_games_vs_pitcher'] = grouped.cumcount()
        history['prior_pa_vs_pitcher'] = grouped['pa_count'].cumsum()
        history['prior_hits_vs_pitcher'] = grouped['hit_count'].cumsum()
        history['prior_hr_vs_pitcher'] = grouped['hr_count'].cumsum()
        history['prior_so_vs_pitcher'] = grouped['so_count'].cumsum()
        for col in ('prior_pa_vs_pitcher', 'prior_hits_vs_pitcher', 'prior_hr_vs_pitcher', 'prior_so_vs_pitcher'):
            history[col] = history.groupby(['batter_canonical', 'pitcher_name'], dropna=False)[col].shift(fill_value=0)
        history['last_hits_vs_pitcher'] = grouped['hit_count'].shift(1)
        history['last_hr_vs_pitcher'] = grouped['hr_count'].shift(1)
        history['last_so_vs_pitcher'] = grouped['so_count'].shift(1)
        history['days_since_last_vs_pitcher'] = (
            history['game_date'] - grouped['game_date'].shift(1)
        ).dt.days

        prior_pa = history['prior_pa_vs_pitcher'].replace(0, np.nan)
        prior_games = history['prior_games_vs_pitcher'].replace(0, np.nan)
        history['prior_hit_rate_vs_pitcher'] = history['prior_hits_vs_pitcher'] / prior_pa
        history['prior_hr_rate_vs_pitcher'] = history['prior_hr_vs_pitcher'] / prior_pa
        history['prior_so_rate_vs_pitcher'] = history['prior_so_vs_pitcher'] / prior_pa
        history['prior_avg_pa_vs_pitcher'] = history['prior_pa_vs_pitcher'] / prior_games

        logger.info(
            "Attached batter-vs-pitcher history: %.1f%% rows have prior matchup games",
            history['prior_games_vs_pitcher'].gt(0).mean() * 100
        )
        return history

    def _attach_historical_weather(self, df: pd.DataFrame) -> pd.DataFrame:
        feature_cols = [
            'wind_speed_mph',
            'temp_f',
            'precip_prob',
            'wind_out_factor',
            'dew_point_f',
            'air_carry_factor',
            'wind_out_to_center_mph',
            'wind_out_to_left_field_mph',
            'wind_out_to_right_field_mph',
            'wind_in_from_center_mph',
            'crosswind_mph',
            'roof_closed_estimated',
            'roof_status_confidence',
            'weather_data_available',
        ]
        if df.empty:
            for col in feature_cols:
                df[col] = np.nan
            return df

        self.conn.rollback()
        cursor = self.conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute("""
            SELECT
                game_id,
                wind_speed_mph,
                temp_f,
                precipitation_mm,
                wind_out_factor,
                dew_point_f,
                air_carry_factor,
                wind_out_to_center_mph,
                wind_out_to_left_field_mph,
                wind_out_to_right_field_mph,
                wind_in_from_center_mph,
                crosswind_mph,
                roof_status_estimated,
                roof_status_confidence,
                weather_available
            FROM historical_game_weather
            WHERE game_id = ANY(%s)
        """, (df['game_id'].dropna().unique().tolist(),))
        weather_df = pd.DataFrame(cursor.fetchall())
        if weather_df.empty:
            df['wind_speed_mph'] = 0.0
            df['temp_f'] = 70.0
            df['precip_prob'] = 0.0
            df['wind_out_factor'] = 1.0
            df['dew_point_f'] = 55.0
            df['air_carry_factor'] = 1.0
            df['wind_out_to_center_mph'] = 0.0
            df['wind_out_to_left_field_mph'] = 0.0
            df['wind_out_to_right_field_mph'] = 0.0
            df['wind_in_from_center_mph'] = 0.0
            df['crosswind_mph'] = 0.0
            df['roof_closed_estimated'] = 0.0
            df['roof_status_confidence'] = 0.0
            df['weather_data_available'] = 0.0
            logger.warning("No historical weather rows found; using placeholder defaults")
            return df

        weather_df = weather_df.rename(columns={'precipitation_mm': 'precip_prob'})
        merged = df.merge(weather_df, on='game_id', how='left')
        merged['wind_speed_mph'] = pd.to_numeric(merged['wind_speed_mph'], errors='coerce').fillna(0.0)
        merged['temp_f'] = pd.to_numeric(merged['temp_f'], errors='coerce').fillna(70.0)
        merged['precip_prob'] = pd.to_numeric(merged['precip_prob'], errors='coerce').fillna(0.0)
        merged['wind_out_factor'] = pd.to_numeric(merged['wind_out_factor'], errors='coerce').fillna(1.0)
        merged['dew_point_f'] = pd.to_numeric(merged['dew_point_f'], errors='coerce').fillna(55.0)
        merged['air_carry_factor'] = pd.to_numeric(merged['air_carry_factor'], errors='coerce').fillna(1.0)
        merged['wind_out_to_center_mph'] = pd.to_numeric(merged['wind_out_to_center_mph'], errors='coerce').fillna(0.0)
        merged['wind_out_to_left_field_mph'] = pd.to_numeric(merged['wind_out_to_left_field_mph'], errors='coerce').fillna(0.0)
        merged['wind_out_to_right_field_mph'] = pd.to_numeric(merged['wind_out_to_right_field_mph'], errors='coerce').fillna(0.0)
        merged['wind_in_from_center_mph'] = pd.to_numeric(merged['wind_in_from_center_mph'], errors='coerce').fillna(0.0)
        merged['crosswind_mph'] = pd.to_numeric(merged['crosswind_mph'], errors='coerce').fillna(0.0)
        merged['roof_closed_estimated'] = (merged['roof_status_estimated'].fillna('').astype(str).str.lower() == 'closed').astype(float)
        merged['roof_status_confidence'] = pd.to_numeric(merged['roof_status_confidence'], errors='coerce').fillna(0.0)
        merged['weather_data_available'] = merged['weather_available'].fillna(False).astype(float)
        logger.info(
            "Attached historical weather: %.1f%% rows with weather_available",
            merged['weather_available'].fillna(False).astype(bool).mean() * 100
        )
        return merged.drop(columns=['weather_available', 'roof_status_estimated'])

    def _attach_hr_weather_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        feature_cols = [
            'batter_pull_wind_mph',
            'batter_oppo_wind_mph',
            'batter_pull_air_carry',
            'batter_pull_weather_boost',
        ]
        if df.empty:
            for col in feature_cols:
                df[col] = np.nan
            return df

        left_field = pd.to_numeric(df.get('wind_out_to_left_field_mph'), errors='coerce').fillna(0.0)
        right_field = pd.to_numeric(df.get('wind_out_to_right_field_mph'), errors='coerce').fillna(0.0)
        carry = pd.to_numeric(df.get('air_carry_factor'), errors='coerce').fillna(1.0)
        roof_closed = pd.to_numeric(df.get('roof_closed_estimated'), errors='coerce').fillna(0.0)
        roof_conf = pd.to_numeric(df.get('roof_status_confidence'), errors='coerce').fillna(0.0)
        weather_avail = pd.to_numeric(df.get('weather_data_available'), errors='coerce').fillna(0.0)
        bats_left = pd.to_numeric(df.get('batter_bats_left'), errors='coerce').fillna(0.0)
        bats_right = pd.to_numeric(df.get('batter_bats_right'), errors='coerce').fillna(0.0)
        switch = pd.to_numeric(df.get('batter_is_switch'), errors='coerce').fillna(0.0)

        pull_wind = np.where(
            bats_left > 0,
            right_field,
            np.where(bats_right > 0, left_field, (left_field + right_field) / 2.0)
        )
        oppo_wind = np.where(
            bats_left > 0,
            left_field,
            np.where(bats_right > 0, right_field, (left_field + right_field) / 2.0)
        )
        switch_adjust = np.where(switch > 0, 0.9, 1.0)
        exposure = weather_avail * (1.0 - (roof_closed * roof_conf))

        df['batter_pull_wind_mph'] = np.round(pull_wind * switch_adjust, 3)
        df['batter_oppo_wind_mph'] = np.round(oppo_wind * switch_adjust, 3)
        df['batter_pull_air_carry'] = np.round(df['batter_pull_wind_mph'] * carry, 3)
        df['batter_pull_weather_boost'] = np.round(df['batter_pull_air_carry'] * exposure, 3)
        return df

    def save_dataset(self, df: pd.DataFrame, path: str = None):
        if path is None:
            path = str(Path(__file__).parent / 'complete_dataset.csv')
        df.to_csv(path, index=False)
        logger.info(f"Saved dataset to {path}")

    # ------------------------------------------------------------------
    # INFERENCE: enrich today's picks for model scoring
    # (picks sourced externally — hellraiser or any other system)
    # ------------------------------------------------------------------

    def load_todays_picks(self, target_date=None) -> pd.DataFrame:
        if target_date is None:
            target_date = datetime.now().date()

        self.conn.rollback()
        cursor = self.conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute("""
            SELECT id, analysis_date, player_name, team, pitcher_name, is_home,
                   confidence_score, odds_decimal,
                   barrel_rate, exit_velocity_avg, hard_hit_percent, sweet_spot_percent,
                   swing_optimization_score, swing_attack_angle, swing_bat_speed
            FROM hellraiser_picks
            WHERE analysis_date = %s
        """, (target_date,))
        df = pd.DataFrame(cursor.fetchall())
        logger.info(f"Loaded {len(df)} picks for {target_date}")
        return df

    def build_for_prediction(self, picks_df: pd.DataFrame, as_of_date) -> pd.DataFrame:
        """Enrich picks with model features for inference (no outcome labels)."""
        if picks_df.empty:
            return pd.DataFrame()

        picks_df = picks_df.copy()
        picks_df['batter_canonical'] = picks_df['player_name'].apply(lambda n: self._canonical(n, 'hellraiser_picks'))

        ev_map = self._load_ev_lookup()
        xs_map = self._load_xstats_lookup()
        pitcher_rolling = self._get_pitcher_rolling_for_date(as_of_date)
        recent = self._get_recent_rates_for_date(as_of_date)

        picks_df['avg_ev']             = picks_df['batter_canonical'].map(ev_map['avg_ev'])
        picks_df['barrel_rate_ev']     = picks_df['batter_canonical'].map(ev_map['barrel_rate'])
        picks_df['sweet_spot_percent'] = picks_df['batter_canonical'].map(ev_map['sweet_spot_percent'])
        picks_df['xwoba'] = picks_df['batter_canonical'].map(xs_map['xwoba'])
        picks_df['xba']   = picks_df['batter_canonical'].map(xs_map['xba'])
        picks_df['xslg']  = picks_df['batter_canonical'].map(xs_map['xslg'])

        if not pitcher_rolling.empty:
            picks_df = picks_df.merge(pitcher_rolling, on='pitcher_name', how='left')
        if not recent.empty:
            picks_df = picks_df.merge(recent, on='batter_canonical', how='left')
        picks_df['opp_team'] = np.nan
        picks_df = self._attach_pitch_type_matchups(picks_df)
        picks_df = self._attach_handedness_features(picks_df)
        picks_df['game_date'] = pd.to_datetime(as_of_date).date()
        picks_df = self._attach_lineup_context(picks_df)
        bvp_history = self._get_batter_vs_pitcher_history_for_date(as_of_date)
        if not bvp_history.empty:
            picks_df = picks_df.merge(bvp_history, on=['batter_canonical', 'pitcher_name'], how='left')
        picks_df = picks_df.drop(columns=[c for c in ('opp_team',) if c in picks_df.columns])

        picks_df = add_park_factors(picks_df, self.conn)
        picks_df['wind_speed_mph'] = 0.0
        picks_df['temp_f'] = 70.0
        picks_df['precip_prob'] = 0.0
        picks_df['wind_out_factor'] = 1.0
        picks_df['dew_point_f'] = 55.0
        picks_df['air_carry_factor'] = 1.0
        picks_df['wind_out_to_center_mph'] = 0.0
        picks_df['wind_out_to_left_field_mph'] = 0.0
        picks_df['wind_out_to_right_field_mph'] = 0.0
        picks_df['wind_in_from_center_mph'] = 0.0
        picks_df['crosswind_mph'] = 0.0
        picks_df['roof_closed_estimated'] = 0.0
        picks_df['roof_status_confidence'] = 0.0
        picks_df['weather_data_available'] = 0.0
        picks_df = self._attach_hr_weather_interactions(picks_df)

        picks_df = picks_df.drop(
            columns=[c for c in ('confidence_score', 'odds_decimal', 'batter_canonical') if c in picks_df.columns]
        )
        return picks_df

    # helpers for build_for_prediction --------------------------------

    def _get_pitcher_rolling_for_date(self, as_of_date) -> pd.DataFrame:
        self.conn.rollback()
        cursor = self.conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute("""
            SELECT ps.player_name AS pitcher_name,
                   AVG(ps.earned_runs::numeric / NULLIF(ps.innings_pitched::numeric,0) * 9) AS pitcher_era_30d,
                   AVG(ps.home_runs::numeric  / NULLIF(ps.innings_pitched::numeric,0) * 9) AS pitcher_hr_per_9_30d,
                   AVG(ps.strikeouts::numeric / NULLIF(ps.innings_pitched::numeric,0) * 9) AS pitcher_k_per_9_30d,
                   AVG((ps.hits::numeric + ps.walks::numeric) / NULLIF(ps.innings_pitched::numeric,0)) AS pitcher_whip_30d
            FROM pitching_stats ps
            JOIN games g ON ps.game_id = g.game_id
            WHERE g.game_date >= %s::date - INTERVAL '30 days'
              AND g.game_date <  %s::date
              AND ps.innings_pitched > 0
            GROUP BY ps.player_name
        """, (as_of_date, as_of_date))
        return pd.DataFrame(cursor.fetchall())

    def _load_ev_lookup(self) -> dict:
        self.conn.rollback()
        cursor = self.conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute("""
            SELECT last_name_first_name AS raw_name,
                   avg_hit_speed_numeric AS avg_ev,
                   brl_percent_numeric   AS barrel_rate,
                   anglesweetspotpercent_numeric AS sweet_spot_percent
            FROM hitter_exit_velocity WHERE last_name_first_name IS NOT NULL
        """)
        result = {'avg_ev': {}, 'barrel_rate': {}, 'sweet_spot_percent': {}}
        for r in cursor.fetchall():
            c = self._canonical(r['raw_name'], 'hitter_exit_velocity_lookup')
            result['avg_ev'][c]             = r['avg_ev']
            result['barrel_rate'][c]        = r['barrel_rate']
            result['sweet_spot_percent'][c] = r['sweet_spot_percent']
        return result

    def _load_xstats_lookup(self) -> dict:
        self.conn.rollback()
        cursor = self.conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute("""
            SELECT last_name_first_name AS raw_name, xwoba, xba, xslg
            FROM custom_batter_2025 WHERE last_name_first_name IS NOT NULL
        """)
        result = {'xwoba': {}, 'xba': {}, 'xslg': {}}
        for r in cursor.fetchall():
            c = self._canonical(r['raw_name'], 'custom_batter_2025_lookup')
            result['xwoba'][c] = r['xwoba']
            result['xba'][c]   = r['xba']
            result['xslg'][c]  = r['xslg']
        return result

    def _get_recent_rates_for_date(self, as_of_date) -> pd.DataFrame:
        self.conn.rollback()
        cursor = self.conn.cursor(cursor_factory=RealDictCursor)
        recency_sql = self._recent_rate_select_sql()
        cursor.execute(f"""
            WITH batter_games AS (
                SELECT
                    pp.batter,
                    g.game_date,
                    pp.game_id,
                    MAX(CASE WHEN pp.play_result = 'Home Run' THEN 1 ELSE 0 END)::numeric AS hr_game,
                    MAX(CASE WHEN pp.play_result IN ('Single','Double','Triple','Home Run') THEN 1 ELSE 0 END)::numeric AS hit_game,
                    MAX(CASE WHEN pp.play_result = 'Strikeout' THEN 1 ELSE 0 END)::numeric AS so_game,
                    COUNT(*) FILTER (
                        WHERE pp.play_result <> 'Other'
                          AND pp.play_result IS NOT NULL
                    )::numeric AS pa_game
                FROM play_by_play_plays pp
                JOIN games g ON pp.game_id = g.game_id
                WHERE g.game_date < %s::date
                  AND pp.batter IS NOT NULL
                GROUP BY pp.batter, g.game_date, pp.game_id
            ),
            ranked_games AS (
                SELECT
                    batter,
                    game_date,
                    hr_game,
                    hit_game,
                    so_game,
                    pa_game,
                    ROW_NUMBER() OVER (
                        PARTITION BY batter
                        ORDER BY game_date DESC, game_id DESC
                    ) AS rn
                FROM batter_games
            )
            SELECT
                batter,
                MAX(game_date) AS last_game_date,
                AVG(hr_game) AS season_hr_rate_to_date,
                AVG(hit_game) AS season_hit_rate_to_date,
                AVG(so_game) AS season_so_rate_to_date,
                COUNT(*) AS season_games_prior,
                SUM(pa_game) AS season_pa_prior,
                {recency_sql}
            FROM ranked_games
            GROUP BY batter
        """, (as_of_date, *self._recent_rate_query_params()))
        df = pd.DataFrame(cursor.fetchall())
        if df.empty:
            return df
        df['batter_canonical'] = df['batter'].apply(lambda n: self._canonical(n, 'play_by_play_recent_lookup'))
        df['recent_form_lookback_games'] = self.recent_lookback_games
        df['days_since_last_game'] = df['last_game_date'].apply(lambda d: self._days_since(as_of_date, d))
        shrink_rows = []
        for record in df.to_dict('records'):
            shrink_rows.append(self._recent_shrinkage_features(record))
        if shrink_rows:
            shrink_df = pd.DataFrame(shrink_rows)
            df = pd.concat([df.reset_index(drop=True), shrink_df.reset_index(drop=True)], axis=1)
        keep_cols = [
            'batter_canonical',
            'recent_hr_rate_14d',
            'recent_hit_rate_14d',
            'recent_so_rate_14d',
            'recent_form_lookback_games',
            'recent_form_games_used',
            'recent_form_pa_used',
            'days_since_last_game',
            'season_hr_rate_to_date',
            'season_hit_rate_to_date',
            'season_so_rate_to_date',
            'season_games_prior',
            'season_pa_prior',
            'shrunk_recent_hr_rate',
            'shrunk_recent_hit_rate',
            'shrunk_recent_so_rate',
            'recent_vs_season_hr_delta',
            'recent_vs_season_hit_delta',
            'recent_vs_season_so_delta',
        ]
        for window in RECENT_FORM_WINDOWS:
            suffix = f"g{window}"
            keep_cols.extend([
                f'recent_hr_rate_{suffix}',
                f'recent_hit_rate_{suffix}',
                f'recent_so_rate_{suffix}',
                f'recent_games_used_{suffix}',
                f'recent_pa_used_{suffix}',
            ])
        return df[keep_cols]

    def _get_batter_vs_pitcher_history_for_date(self, as_of_date) -> pd.DataFrame:
        self.conn.rollback()
        cursor = self.conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute("""
            SELECT
                g.game_date,
                pp.batter,
                pp.pitcher,
                COUNT(CASE WHEN pp.play_result <> 'Other' THEN 1 END) AS pa_count,
                COUNT(CASE WHEN pp.play_result IN ('Single','Double','Triple','Home Run') THEN 1 END) AS hit_count,
                COUNT(CASE WHEN pp.play_result = 'Home Run' THEN 1 END) AS hr_count,
                COUNT(CASE WHEN pp.play_result = 'Strikeout' THEN 1 END) AS so_count
            FROM play_by_play_plays pp
            JOIN games g ON pp.game_id = g.game_id
            WHERE g.game_date < %s::date
              AND pp.batter IS NOT NULL
              AND trim(pp.batter) <> ''
              AND lower(trim(pp.batter)) <> 'unknown'
              AND pp.pitcher IS NOT NULL
            GROUP BY g.game_date, pp.batter, pp.pitcher
            ORDER BY pp.batter, pp.pitcher, g.game_date
        """, (as_of_date,))
        history = pd.DataFrame(cursor.fetchall())
        if history.empty:
            return history

        history['batter_canonical'] = history['batter'].apply(lambda n: self._canonical(n, 'play_by_play_bvp_lookup'))
        history['game_date'] = pd.to_datetime(history['game_date'], errors='coerce')
        history = history.sort_values(['batter_canonical', 'pitcher', 'game_date']).copy()
        grouped = history.groupby(['batter_canonical', 'pitcher'], dropna=False)
        summary = grouped.agg(
            prior_games_vs_pitcher=('game_date', 'count'),
            prior_pa_vs_pitcher=('pa_count', 'sum'),
            prior_hits_vs_pitcher=('hit_count', 'sum'),
            prior_hr_vs_pitcher=('hr_count', 'sum'),
            prior_so_vs_pitcher=('so_count', 'sum'),
            last_game_date=('game_date', 'max'),
            last_hits_vs_pitcher=('hit_count', 'last'),
            last_hr_vs_pitcher=('hr_count', 'last'),
            last_so_vs_pitcher=('so_count', 'last'),
        ).reset_index()

        prior_pa = summary['prior_pa_vs_pitcher'].replace(0, np.nan)
        prior_games = summary['prior_games_vs_pitcher'].replace(0, np.nan)
        summary['prior_hit_rate_vs_pitcher'] = summary['prior_hits_vs_pitcher'] / prior_pa
        summary['prior_hr_rate_vs_pitcher'] = summary['prior_hr_vs_pitcher'] / prior_pa
        summary['prior_so_rate_vs_pitcher'] = summary['prior_so_vs_pitcher'] / prior_pa
        summary['prior_avg_pa_vs_pitcher'] = summary['prior_pa_vs_pitcher'] / prior_games
        summary['days_since_last_vs_pitcher'] = (
            pd.Timestamp(as_of_date) - summary['last_game_date']
        ).dt.days
        summary = summary.rename(columns={'pitcher': 'pitcher_name'})
        return summary[[
            'batter_canonical',
            'pitcher_name',
            'prior_games_vs_pitcher',
            'prior_pa_vs_pitcher',
            'prior_hits_vs_pitcher',
            'prior_hr_vs_pitcher',
            'prior_so_vs_pitcher',
            'prior_hit_rate_vs_pitcher',
            'prior_hr_rate_vs_pitcher',
            'prior_so_rate_vs_pitcher',
            'prior_avg_pa_vs_pitcher',
            'days_since_last_vs_pitcher',
            'last_hits_vs_pitcher',
            'last_hr_vs_pitcher',
            'last_so_vs_pitcher',
        ]]

def parse_args():
    parser = argparse.ArgumentParser(description="Build hitter training dataset.")
    parser.add_argument(
        "--recent-lookback-games",
        type=int,
        default=DEFAULT_RECENT_LOOKBACK_GAMES,
        help="Number of prior games to use for hitter recent-form rates.",
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    builder = DatasetBuilder(recent_lookback_games=args.recent_lookback_games)
    try:
        df = builder.build()
        if not df.empty:
            builder.save_dataset(df)
            logger.info("Dataset creation complete")
    finally:
        builder.close()
