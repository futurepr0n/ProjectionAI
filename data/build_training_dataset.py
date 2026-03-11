#!/usr/bin/env python3
"""
Training dataset builder for ProjectionAI.

Row spine: play_by_play_plays aggregated per (game_id, batter).
Full player names from PBP eliminate the abbreviated-name mismatch
that hitting_stats introduces. No dependency on hellraiser_picks for training.

hellraiser_picks is used only in build_for_prediction() at inference time.
"""
import psycopg2
from psycopg2.extras import RealDictCursor
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from pathlib import Path
import os
from collections import defaultdict

from name_utils import normalize_to_canonical
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


class DatasetBuilder:
    def __init__(self):
        self.conn = self._connect()
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

        # Rolling 14-day rates per batter as of each game_date (no leakage)
        df = self._attach_recent_rates(df)

        # Opposing pitcher rolling 30-day stats per (game_id, opp_team)
        df = self._attach_pitcher_stats(df)
        df = self._attach_pitch_type_matchups(df)
        df = self._attach_batter_vs_pitcher_history(df)

        # Context: park, travel, weather defaults
        df = add_park_factors(df, self.conn)
        df = add_travel_fatigue(df, self.conn)
        df['wind_speed_mph'] = 0.0
        df['temp_f'] = 70.0
        df['precip_prob'] = 0.0
        df['wind_out_factor'] = 1.0

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

    def _attach_recent_rates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rolling 14-day HR / hit / SO rates per batter, computed per unique game_date."""
        dates = sorted(df['game_date'].unique().tolist())
        cursor = self.conn.cursor(cursor_factory=RealDictCursor)
        rate_rows = []

        for gd in dates:
            self.conn.rollback()
            cursor.execute("""
                SELECT
                    pp.batter,
                    COUNT(CASE WHEN pp.play_result = 'Home Run' THEN 1 END)::numeric /
                        NULLIF(COUNT(*), 0)                                          AS recent_hr_rate_14d,
                    COUNT(CASE WHEN pp.play_result IN ('Single','Double','Triple','Home Run') THEN 1 END)::numeric /
                        NULLIF(COUNT(*), 0)                                          AS recent_hit_rate_14d,
                    COUNT(CASE WHEN pp.play_result = 'Strikeout' THEN 1 END)::numeric /
                        NULLIF(COUNT(*), 0)                                          AS recent_so_rate_14d
                FROM play_by_play_plays pp
                JOIN games g ON pp.game_id = g.game_id
                WHERE g.game_date >= %s::date - INTERVAL '14 days'
                  AND g.game_date <  %s::date
                  AND pp.batter IS NOT NULL
                GROUP BY pp.batter
            """, (gd, gd))
            for row in cursor.fetchall():
                rate_rows.append({
                    'game_date':        gd,
                    'batter_canonical': self._canonical(row['batter'], 'play_by_play_recent_rates'),
                    'recent_hr_rate_14d':  float(row['recent_hr_rate_14d'])  if row['recent_hr_rate_14d']  else None,
                    'recent_hit_rate_14d': float(row['recent_hit_rate_14d']) if row['recent_hit_rate_14d'] else None,
                    'recent_so_rate_14d':  float(row['recent_so_rate_14d'])  if row['recent_so_rate_14d']  else None,
                })

        if not rate_rows:
            for col in ('recent_hr_rate_14d', 'recent_hit_rate_14d', 'recent_so_rate_14d'):
                df[col] = np.nan
            return df

        rates = pd.DataFrame(rate_rows).drop_duplicates(subset=['game_date', 'batter_canonical'])
        df = df.merge(rates, on=['game_date', 'batter_canonical'], how='left')
        logger.info(f"Recent rates match rate: {df['recent_hr_rate_14d'].notna().mean()*100:.1f}%")
        return df

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

        pitcher_arsenal['pitcher_name_normalized'] = pitcher_arsenal['last_name_first_name'].apply(
            lambda value: _name_from_last_first(value).strip().lower()
        )
        pitcher_arsenal['pitcher_name_key'] = pitcher_arsenal['pitcher_name_normalized'].apply(_pitcher_name_key)
        pitcher_arsenal['team_normalized'] = pitcher_arsenal['team_name_alt'].apply(_normalize_team_code)
        for col in ('pitch_usage', 'whiff_percent', 'k_percent', 'put_away'):
            pitcher_arsenal[col] = pd.to_numeric(pitcher_arsenal[col], errors='coerce')

        pitcher_profiles = {}
        pitcher_profiles_by_name = {}
        for (name_key, team_code), grp in pitcher_arsenal.groupby(['pitcher_name_key', 'team_normalized'], dropna=False):
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
            pitcher_profiles[(name_key, team_code)] = profile
            pitcher_profiles_by_name.setdefault(name_key, profile)

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
        df['pitcher_name_key'] = df['pitcher_name'].apply(_pitcher_name_key)
        df['opp_team_normalized'] = df['opp_team'].apply(_normalize_team_code)
        rows = []
        for idx, row in df.iterrows():
            profile = pitcher_profiles.get((row['pitcher_name_key'], row['opp_team_normalized']))
            if profile is None:
                profile = pitcher_profiles_by_name.get(row['pitcher_name_key'])
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
        df = df.drop(columns=['pitcher_name_key', 'opp_team_normalized'])
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
        bvp_history = self._get_batter_vs_pitcher_history_for_date(as_of_date)
        if not bvp_history.empty:
            picks_df = picks_df.merge(bvp_history, on=['batter_canonical', 'pitcher_name'], how='left')
        picks_df = picks_df.drop(columns=[c for c in ('opp_team',) if c in picks_df.columns])

        picks_df = add_park_factors(picks_df, self.conn)
        picks_df['wind_speed_mph']  = 0.0
        picks_df['temp_f']          = 70.0
        picks_df['precip_prob']     = 0.0
        picks_df['wind_out_factor'] = 1.0

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
        cursor.execute("""
            SELECT pp.batter,
                   COUNT(CASE WHEN pp.play_result = 'Home Run' THEN 1 END)::numeric /
                       NULLIF(COUNT(*), 0) AS recent_hr_rate_14d,
                   COUNT(CASE WHEN pp.play_result IN ('Single','Double','Triple','Home Run') THEN 1 END)::numeric /
                       NULLIF(COUNT(*), 0) AS recent_hit_rate_14d,
                   COUNT(CASE WHEN pp.play_result = 'Strikeout' THEN 1 END)::numeric /
                       NULLIF(COUNT(*), 0) AS recent_so_rate_14d
            FROM play_by_play_plays pp
            JOIN games g ON pp.game_id = g.game_id
            WHERE g.game_date >= %s::date - INTERVAL '14 days'
              AND g.game_date <  %s::date
              AND pp.batter IS NOT NULL
            GROUP BY pp.batter
        """, (as_of_date, as_of_date))
        df = pd.DataFrame(cursor.fetchall())
        if df.empty:
            return df
        df['batter_canonical'] = df['batter'].apply(lambda n: self._canonical(n, 'play_by_play_recent_lookup'))
        return df[['batter_canonical', 'recent_hr_rate_14d', 'recent_hit_rate_14d', 'recent_so_rate_14d']]

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


if __name__ == '__main__':
    builder = DatasetBuilder()
    try:
        df = builder.build()
        if not df.empty:
            builder.save_dataset(df)
            logger.info("Dataset creation complete")
    finally:
        builder.close()
