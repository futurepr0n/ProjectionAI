#!/usr/bin/env python3
"""
ProjectionAI - Updated Dashboard with Date Listing and Results
"""

from flask import Flask, render_template, jsonify, request
import pandas as pd
import numpy as np
import re
import xgboost as xgb
import lightgbm as lgb
import joblib
import json
import psycopg2
from psycopg2.extras import RealDictCursor
import logging
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple
import os
import subprocess
import math

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _sanitize_for_json(obj):
    """Recursively sanitize NaN/Inf values for JSON serialization"""
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    return obj

app = Flask(__name__)

# Signal thresholds per target type
SIGNAL_THRESHOLDS = {
    'hr':  {'STRONG_BUY': 0.67, 'BUY': 0.60, 'MODERATE': 0.52, 'AVOID': 0.43},
    'hit': {'STRONG_BUY': 0.87, 'BUY': 0.83, 'MODERATE': 0.78, 'AVOID': 0.72},
    'so':  {'STRONG_BUY': 0.47, 'BUY': 0.43, 'MODERATE': 0.39, 'AVOID': 0.36},
}

MODEL_ARTIFACT_PREFIX = {
    'hr': 'hr',
    'hit': 'hit',
    'so': 'pitcher_so_3_plus',
}

SO_THRESHOLD_PREFIX = {
    3: 'pitcher_so_3_plus',
    4: 'pitcher_so_4_plus',
    5: 'pitcher_so_5_plus',
    6: 'pitcher_so_6_plus',
}

TEAM_CODE_ALIASES = {
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


class PredictionEngine:
    """Engine for generating live predictions"""

    def __init__(self):
        self.model = None
        self.xgb_model = None
        self.lgb_model = None
        self.meta_model = None
        self.feature_names = None
        self.train_medians = None
        self.models = {}
        self.so_models = {}
        self.db_conn = None
        self.dataset = None
        self.pitcher_so_dataset = None
        self.load_model()
        self.connect_db()
        self.dataset = self._load_hitter_dataset()
        self.pitcher_so_dataset = self._load_pitcher_so_dataset()
        self._compute_thresholds()

    def reload_models(self):
        """Reload all model artifacts from disk (call after training completes)"""
        logger.info("🔄 Reloading models from disk...")
        self.load_model()
        loaded = [t for t in ['hr', 'hit', 'so'] if self.models.get(t)]
        logger.info(f"✅ Reload complete. Loaded: {loaded}")

    def load_model(self, model_path=None):
        """Load current artifacts for HR, Hit, and starter strikeout targets."""
        if model_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            artifacts_dir = os.path.join(base_dir, 'models', 'artifacts')
        else:
            artifacts_dir = os.path.dirname(model_path)

        self.models = {'hr': None, 'hit': None, 'so': None}
        self.so_models = {}

        for target in ['hr', 'hit']:
            try:
                artifact_prefix = MODEL_ARTIFACT_PREFIX[target]
                xgb_path = os.path.join(artifacts_dir, f'{artifact_prefix}_xgb.json')
                lgb_path = os.path.join(artifacts_dir, f'{artifact_prefix}_lgb.txt')
                meta_path = os.path.join(artifacts_dir, f'{artifact_prefix}_meta.pkl')

                if not os.path.exists(xgb_path) or not os.path.exists(lgb_path) or not os.path.exists(meta_path):
                    logger.warning(f"⚠️ {target.upper()} ensemble artifacts missing at {artifacts_dir}")
                    self.models[target] = None
                    continue

                xgb_model = xgb.XGBClassifier()
                xgb_model.load_model(xgb_path)

                lgb_model = lgb.Booster(model_file=lgb_path)

                meta_artifact = joblib.load(meta_path)
                meta_model = meta_artifact.get('meta')
                feature_names = meta_artifact.get('features', [])
                train_medians = meta_artifact.get('train_medians', {})

                self.models[target] = {
                    'xgb': xgb_model,
                    'lgb': lgb_model,
                    'meta': meta_model,
                    'features': feature_names,
                    'train_medians': train_medians,
                    'artifact_prefix': artifact_prefix,
                }
                logger.info(f"✅ {target.upper()} Ensemble loaded: {len(feature_names)} features")

            except Exception as e:
                logger.error(f"❌ Failed to load {target.upper()} ensemble: {e}")
                self.models[target] = None

        for threshold, artifact_prefix in SO_THRESHOLD_PREFIX.items():
            try:
                xgb_path = os.path.join(artifacts_dir, f'{artifact_prefix}_xgb.json')
                lgb_path = os.path.join(artifacts_dir, f'{artifact_prefix}_lgb.txt')
                meta_path = os.path.join(artifacts_dir, f'{artifact_prefix}_meta.pkl')

                if not os.path.exists(xgb_path) or not os.path.exists(lgb_path) or not os.path.exists(meta_path):
                    logger.warning("⚠️ SO %s+ ensemble artifacts missing at %s", threshold, artifacts_dir)
                    self.so_models[threshold] = None
                    continue

                xgb_model = xgb.XGBClassifier()
                xgb_model.load_model(xgb_path)
                lgb_model = lgb.Booster(model_file=lgb_path)
                meta_artifact = joblib.load(meta_path)
                feature_names = meta_artifact.get('features', [])
                train_medians = meta_artifact.get('train_medians', {})

                self.so_models[threshold] = {
                    'xgb': xgb_model,
                    'lgb': lgb_model,
                    'meta': meta_artifact.get('meta'),
                    'features': feature_names,
                    'train_medians': train_medians,
                    'artifact_prefix': artifact_prefix,
                    'threshold': threshold,
                }
                logger.info("✅ SO %s+ Ensemble loaded: %s features", threshold, len(feature_names))
            except Exception as e:
                logger.error("❌ Failed to load SO %s+ ensemble: %s", threshold, e)
                self.so_models[threshold] = None

        self.models['so'] = self.so_models.get(3)

        # Set legacy attributes for backward compatibility (HR model)
        if self.models['hr']:
            self.model = 'ensemble'
            self.xgb_model = self.models['hr']['xgb']
            self.lgb_model = self.models['hr']['lgb']
            self.meta_model = self.models['hr']['meta']
            self.feature_names = self.models['hr']['features']
            self.train_medians = self.models['hr']['train_medians']
        else:
            self.model = None
            self.xgb_model = None
            self.lgb_model = None
            self.meta_model = None
            self.feature_names = []
            self.train_medians = {}

    def connect_db(self):
        """Connect to remote database"""
        try:
            self.db_conn = psycopg2.connect(
                host='192.168.1.23',
                port=5432,
                database='baseball_migration_test',
                user='postgres',
                password='korn5676'
            )
            logger.info("✅ Database connected")
        except Exception as e:
            logger.error(f"❌ Database connection failed: {e}")

    def _load_hitter_dataset(self):
        """Load hitter-row CSV dataset at engine startup."""
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        csv_path = os.path.join(base_dir, 'data', 'complete_dataset.csv')
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            df['game_date'] = pd.to_datetime(df['game_date']).dt.date
            _ACTION_RE = re.compile(
                r'\s+(hit|struck|walk|fly|bunt|by|grounded|lined|flied|popped|'
                r'singled|doubled|tripled|homered|reached|sacrifice|intentional|'
                r'called|swinging)\b.*$',
                re.IGNORECASE
            )
            df['player_name'] = df['player_name'].str.replace(_ACTION_RE, '', regex=True).str.strip()
            logger.info(f"✅ Dataset loaded: {len(df)} rows, {df['game_date'].nunique()} dates")
            return df
        logger.warning("⚠️ complete_dataset.csv not found")
        return pd.DataFrame()

    def _load_pitcher_so_dataset(self):
        """Load starter strikeout dataset used by the SO target."""
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        csv_path = os.path.join(base_dir, 'data', 'pitcher_strikeout_dataset.csv')
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            df['game_date'] = pd.to_datetime(df['game_date']).dt.date
            logger.info(f"✅ Pitcher SO dataset loaded: {len(df)} rows, {df['game_date'].nunique()} dates")
            return df
        logger.warning("⚠️ pitcher_strikeout_dataset.csv not found")
        return pd.DataFrame()

    def _dataset_for_target(self, target: str) -> pd.DataFrame:
        return self.pitcher_so_dataset if target == 'so' else self.dataset

    def _so_threshold(self, value: Optional[str]) -> int:
        try:
            threshold = int(value) if value is not None else 3
        except Exception:
            threshold = 3
        return threshold if threshold in SO_THRESHOLD_PREFIX else 3

    def _model_data_for_target(self, target: str, so_threshold: Optional[int] = None):
        if target == 'so':
            return self.so_models.get(so_threshold or 3)
        return self.models.get(target)

    def _compute_thresholds(self):
        """Compute signal thresholds from model output percentiles on training data."""
        for target in ['hr', 'hit', 'so']:
            model_data = self._model_data_for_target(target, so_threshold=3)
            dataset = self._dataset_for_target(target)
            if not model_data or dataset is None or dataset.empty:
                continue
            try:
                X = dataset.reindex(columns=model_data['features'])
                for feat in model_data['features']:
                    X[feat] = pd.to_numeric(X[feat], errors='coerce').fillna(
                        model_data['train_medians'].get(feat, 0)
                    )
                proba = model_data['xgb'].predict_proba(X)[:, 1]
                SIGNAL_THRESHOLDS[target] = {
                    'STRONG_BUY': float(np.percentile(proba, 90)),
                    'BUY':        float(np.percentile(proba, 75)),
                    'MODERATE':   float(np.percentile(proba, 50)),
                    'AVOID':      float(np.percentile(proba, 25)),
                }
                logger.info(f"Thresholds [{target}]: {SIGNAL_THRESHOLDS[target]}")
            except Exception as e:
                logger.error(f"❌ Failed to compute thresholds for {target}: {e}")

    def _get_park_factor(self, team_or_venue: str) -> float:
        """Get park factor for a team (HR impact multiplier)"""
        team_or_venue = self._normalize_team_code(team_or_venue)
        if not team_or_venue:
            return 1.0

        park_factors = {
            'COL': 1.35, 'NYY': 1.20, 'BOS': 1.15, 'CIN': 1.15, 'PHI': 1.10,
            'HOU': 1.10, 'BAL': 1.05, 'TEX': 1.05, 'ARI': 1.05, 'ATL': 1.00,
            'CHC': 1.00, 'LAD': 1.00, 'MIL': 1.00, 'SDP': 1.00, 'WSN': 1.00,
            'CLE': 0.95, 'DET': 0.95, 'KCR': 0.95, 'MIN': 0.95, 'OAK': 0.90,
            'MIA': 0.90, 'PIT': 0.90, 'SEA': 0.90, 'SFG': 0.90, 'TBR': 0.90,
        }
        return park_factors.get(team_or_venue, 1.0)

    def _normalize_team_code(self, team_code: Optional[str]) -> Optional[str]:
        """Collapse known team-code aliases to one dashboard-facing code."""
        if not team_code:
            return team_code
        return TEAM_CODE_ALIASES.get(str(team_code).strip().upper(), str(team_code).strip().upper())

    def _get_batter_xstats(self, player_name: str) -> dict:
        """Get expected stats (xWOBA, xBA, xSLG) for a batter"""
        if not self.db_conn or not player_name:
            return {'xwoba': 0.320, 'xba': 0.250, 'xslg': 0.420}

        try:
            cursor = self.db_conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute("""
            SELECT xwoba, xba, xslg FROM custom_batter_2025
            WHERE last_name_first_name = %s OR last_name_first_name ILIKE %s
            LIMIT 1
            """, (player_name, f"%{player_name.split()[-1]}%"))

            result = cursor.fetchone()
            if result:
                return {
                    'xwoba': float(result['xwoba']) if result['xwoba'] else 0.320,
                    'xba': float(result['xba']) if result['xba'] else 0.250,
                    'xslg': float(result['xslg']) if result['xslg'] else 0.420
                }
        except Exception as e:
            logger.warning(f"Error fetching xstats for {player_name}: {e}")

        return {'xwoba': 0.320, 'xba': 0.250, 'xslg': 0.420}

    def _get_pitcher_rolling_stats(self, pitcher_name: str, game_date) -> dict:
        """Get 30-day rolling pitcher stats"""
        if not self.db_conn or not pitcher_name:
            return {
                'pitcher_era': 4.5,
                'pitcher_hr_per_9': 1.2,
                'pitcher_k_per_9': 20.0,
                'pitcher_whip': 1.3
            }

        try:
            cursor = self.db_conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute("""
            SELECT
                AVG(CAST(earned_runs AS NUMERIC) / NULLIF(CAST(innings_pitched AS NUMERIC), 0) * 9) as era,
                AVG(CAST(home_runs AS NUMERIC) / NULLIF(CAST(innings_pitched AS NUMERIC), 0) * 9) as hr_per_9,
                AVG(CAST(strikeouts AS NUMERIC) / NULLIF(CAST(innings_pitched AS NUMERIC), 0) * 9) as k_per_9,
                AVG((CAST(hits AS NUMERIC) + CAST(walks AS NUMERIC)) / NULLIF(CAST(innings_pitched AS NUMERIC), 0)) as whip
            FROM pitching_stats ps
            JOIN games g ON ps.game_id = g.game_id
            WHERE g.game_date BETWEEN %s - INTERVAL '30 days' AND %s - INTERVAL '1 day'
              AND ps.player_name = %s
              AND CAST(ps.innings_pitched AS NUMERIC) > 0
            """, (game_date, game_date, pitcher_name))

            result = cursor.fetchone()
            if result:
                return {
                    'pitcher_era': float(result['era']) if result['era'] else 4.5,
                    'pitcher_hr_per_9': float(result['hr_per_9']) if result['hr_per_9'] else 1.2,
                    'pitcher_k_per_9': float(result['k_per_9']) if result['k_per_9'] else 20.0,
                    'pitcher_whip': float(result['whip']) if result['whip'] else 1.3
                }
        except Exception as e:
            logger.warning(f"Error fetching pitcher stats for {pitcher_name}: {e}")

        return {
            'pitcher_era': 4.5,
            'pitcher_hr_per_9': 1.2,
            'pitcher_k_per_9': 20.0,
            'pitcher_whip': 1.3
        }

    def _get_pitcher_predictions_for_date(self, target_date: date) -> pd.DataFrame:
        dataset = self.pitcher_so_dataset
        if dataset is None or dataset.empty:
            return pd.DataFrame()
        day_df = dataset[dataset['game_date'] == target_date].copy()
        if not day_df.empty:
            logger.info("✅ Loaded %s starter strikeout rows for %s", len(day_df), target_date)
        return day_df

    def _get_players_for_date(self, target_date: date) -> pd.DataFrame:
        """Query hitting_stats joined with games for a specific date from database"""
        if not self.db_conn:
            return pd.DataFrame()

        try:
            cursor = self.db_conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute("""
            SELECT
                hs.game_id,
                hs.player_name,
                hs.team,
                g.home_team,
                g.away_team,
                g.game_date,
                hs.home_runs,
                hs.hits,
                hs.strikeouts,
                hs.at_bats,
                hs.runs,
                hs.rbi,
                hs.walks,
                hs.avg,
                hs.obp,
                hs.slg,
                -- xstats from custom_batter_2025 (bulk join, avoids per-player queries)
                cb.xwoba,
                cb.xba,
                cb.xslg,
                cb.sweet_spot_percent,
                cb.barrel_batted_rate AS barrel_rate,
                cb.pa AS pa_count
            FROM hitting_stats hs
            JOIN games g ON hs.game_id = g.game_id
            LEFT JOIN LATERAL (
                SELECT xwoba, xba, xslg, sweet_spot_percent, barrel_batted_rate, pa
                FROM custom_batter_2025 cb2
                WHERE cb2.last_name_first_name ILIKE '%%' || split_part(hs.player_name, ' ', 2) || '%%'
                ORDER BY cb2.data_date DESC
                LIMIT 1
            ) cb ON true
            WHERE g.game_date = %s
            ORDER BY hs.player_name
            """, (target_date,))

            results = cursor.fetchall()
            if not results:
                return pd.DataFrame()

            # Convert to DataFrame
            df = pd.DataFrame([dict(r) for r in results])
            before = len(df)
            df = df.drop_duplicates(subset=['game_id', 'player_name', 'team']).copy()
            removed = before - len(df)
            if removed > 0:
                logger.info("Deduped %s hitter rows for %s", removed, target_date)
            logger.info(f"✅ Loaded {len(df)} players from DB for {target_date}")
            return df

        except Exception as e:
            logger.warning(f"Error querying DB for {target_date}: {e}")
            return pd.DataFrame()

    def get_available_dates(self) -> List[Dict]:
        """Get all dates with picks from DB or CSV dataset"""
        dates_list = []

        # Try DB first
        if self.db_conn:
            try:
                cursor = self.db_conn.cursor(cursor_factory=RealDictCursor)
                cursor.execute("""
                SELECT DISTINCT
                    g.game_date,
                    COUNT(DISTINCT hs.player_name) as total_picks
                FROM games g
                LEFT JOIN hitting_stats hs ON g.game_id = hs.game_id
                WHERE g.game_date IS NOT NULL
                GROUP BY g.game_date
                ORDER BY g.game_date DESC
                LIMIT 365
                """)

                results = cursor.fetchall()
                if results:
                    dates_list = [
                        {
                            'date': str(r['game_date']),
                            'analysis_date': str(r['game_date']),
                            'total_picks': int(r['total_picks'] or 0)
                        }
                        for r in results
                    ]
                    logger.info(f"✅ Loaded {len(dates_list)} dates from DB")
                    return dates_list
            except Exception as e:
                logger.warning(f"DB query failed, falling back to CSV: {e}")

        # Fallback to CSV
        if self.dataset is None or self.dataset.empty:
            return []
        counts = (
            self.dataset.groupby('game_date')
            .agg(total_picks=('player_name', 'count'))
            .reset_index()
            .sort_values('game_date', ascending=False)
        )
        return [
            {
                'date': row['game_date'].isoformat(),
                'analysis_date': row['game_date'].isoformat(),
                'total_picks': int(row['total_picks'])
            }
            for _, row in counts.iterrows()
        ]

    def get_results_for_date(self, target_date: date, target: str = 'hr') -> Dict:
        """Get actual results for a date from play-by-play by target type"""
        if not self.db_conn:
            return {}

        target = target.lower() if target else 'hr'

        # Map target to play_result values and count column
        result_mapping = {
            'hr': ('Home Run', 'hr_count'),
            'hit': (('Single', 'Double', 'Triple', 'Home Run'), 'hit_count'),
            'so': ('Strikeout', 'so_count')
        }

        if target not in result_mapping:
            logger.warning(f"Unknown target type: {target}")
            return {}

        play_results, count_col = result_mapping[target]

        try:
            cursor = self.db_conn.cursor(cursor_factory=RealDictCursor)

            if isinstance(play_results, tuple):
                query = f"""
                    SELECT
                        pp.batter,
                        m.game_id,
                        g.home_team,
                        g.away_team,
                        COUNT(*) as {count_col}
                    FROM play_by_play_plays pp
                    JOIN play_by_play_metadata m ON pp.metadata_id = m.id
                    JOIN games g ON m.game_id = g.game_id
                    WHERE g.game_date = %s
                      AND pp.play_result IN ({','.join(['%s'] * len(play_results))})
                    GROUP BY pp.batter, m.game_id, g.home_team, g.away_team
                """
                cursor.execute(query, (target_date, *play_results))
            else:
                query = f"""
                    SELECT
                        pp.batter,
                        m.game_id,
                        g.home_team,
                        g.away_team,
                        COUNT(*) as {count_col}
                    FROM play_by_play_plays pp
                    JOIN play_by_play_metadata m ON pp.metadata_id = m.id
                    JOIN games g ON m.game_id = g.game_id
                    WHERE g.game_date = %s
                      AND pp.play_result = %s
                    GROUP BY pp.batter, m.game_id, g.home_team, g.away_team
                """
                cursor.execute(query, (target_date, play_results))

            results = cursor.fetchall()

            results_by_lastname = {}
            for r in results:
                lastname = r['batter'].strip().lower().split()[-1] if r['batter'] else ''
                if lastname:
                    results_by_lastname[lastname] = {
                        count_col: r[count_col],
                        'game_id': r['game_id'],
                        'team': r['home_team'] if r['home_team'] else r['away_team']
                    }

            return results_by_lastname
        except Exception as e:
            logger.error(f"❌ Error getting {target.upper()} results: {e}")
            return {}

    def get_hr_results_for_date(self, target_date: date) -> Dict:
        """Get actual HR results for a date from play-by-play (backward compatibility)"""
        return self.get_results_for_date(target_date, 'hr')

    def get_hellraiser_picks(self, analysis_date: date) -> List[Dict]:
        """Get Hellraiser picks for a date"""
        if not self.db_conn:
            return []

        try:
            cursor = self.db_conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute("""
                SELECT * FROM hellraiser_picks
                WHERE analysis_date = %s
                ORDER BY confidence_score DESC
            """, (analysis_date,))
            picks = cursor.fetchall()
            return [dict(p) for p in picks]
        except Exception as e:
            logger.error(f"❌ Error getting picks: {e}")
            return []

    def predict(self, features: Dict, target: str = 'hr', model_data_override=None) -> Dict:
        """Generate prediction for a player using v4 ensemble"""
        target = target.lower() if target else 'hr'

        model_data = model_data_override or self.models.get(target)

        if model_data is None:
            logger.warning(f"Model for target '{target}' not available, returning neutral prediction")
            return {
                'probability': 0.5,
                'signal': 'UNAVAILABLE',
                'implied_probability': 0.0,
                'edge': 0.0,
                'edge_pct': 0.0
            }

        try:
            # Prepare feature dataframe
            X = pd.DataFrame([features])

            # Ensure all features exist and impute missing values
            for feat in model_data['features']:
                if feat not in X.columns:
                    X[feat] = model_data['train_medians'].get(feat, 0)
                elif X[feat].isna().any():
                    X[feat] = X[feat].fillna(model_data['train_medians'].get(feat, 0))

            X = X[model_data['features']]

            # Coerce all columns to numeric (DB may return strings for some fields)
            for col in X.columns:
                X[col] = pd.to_numeric(X[col], errors='coerce')
            X = X.fillna(model_data['train_medians'])

            # Generate base predictions
            xgb_proba = model_data['xgb'].predict_proba(X)[:, 1]
            lgb_proba = model_data['lgb'].predict(X)

            # Use XGB probability as primary signal (LGB is a consistency check).
            # The meta-stacker's large negative intercept (fitted to a very low base rate)
            # suppresses all outputs to ~2%, making signals meaningless. XGB alone gives
            # meaningful differentiation: elite batters ~17%, median ~13%.
            prob = float(xgb_proba[0])

            # Get thresholds for this target type
            thresholds = SIGNAL_THRESHOLDS.get(target, SIGNAL_THRESHOLDS['hr'])

            # Apply target-specific thresholds
            if prob >= thresholds['STRONG_BUY']:
                signal = 'STRONG_BUY'
            elif prob >= thresholds['BUY']:
                signal = 'BUY'
            elif prob >= thresholds['MODERATE']:
                signal = 'MODERATE'
            elif prob >= thresholds['AVOID']:
                signal = 'AVOID'
            else:
                signal = 'STRONG_SELL'

            odds = features.get('odds_decimal', 0)
            if odds and float(odds) > 1:
                implied_prob = 1 / float(odds)
                edge = prob - implied_prob
            else:
                implied_prob = 0
                edge = 0

            return {
                'probability': float(prob),
                'signal': signal,
                'implied_probability': float(implied_prob),
                'edge': float(edge),
                'edge_pct': float(edge * 100)
            }

        except Exception as e:
            logger.error(f"Prediction error for target '{target}': {e}")
            return {
                'probability': 0.5,
                'signal': 'ERROR',
                'implied_probability': 0.0,
                'edge': 0.0,
                'edge_pct': 0.0
            }

    def compute_composite_score(self, row: dict, all_rows: list) -> float:
        """
        Composite pick quality score (0-100), normalized across today's prediction pool.
        Distinct from model probability — measures evidence strength across key dimensions.
        """
        def safe_float(v, default=0.0):
            try: return float(v) if v is not None else default
            except: return default

        def normalize(val, vals, invert=False):
            lo, hi = min(vals), max(vals)
            if hi == lo: return 0.5
            n = (val - lo) / (hi - lo)
            return 1.0 - n if invert else n

        # Extract this row's values
        xwoba       = safe_float(row.get('xwoba', 0.320))
        barrel_rate = safe_float(row.get('barrel_rate', 0.06))
        avg_ev      = safe_float(row.get('avg_ev', 88.0))
        p_hr9       = safe_float(row.get('pitcher_hr_per_9_30d', 1.2))
        p_era       = safe_float(row.get('pitcher_era_30d', 4.0))
        hr_rate_14d = safe_float(row.get('recent_hr_rate_14d', 0.03))
        park_factor = safe_float(row.get('park_factor', 1.0))
        wind_out    = safe_float(row.get('wind_out_factor', 1.0))
        temp_f      = safe_float(row.get('temp_f', 72.0))
        fatigue     = safe_float(row.get('travel_fatigue_score', 0.0))

        # Build population arrays for normalization
        def pool(key, default): return [safe_float(r.get(key, default)) for r in all_rows]

        # Component 1: Power (higher = better)
        xwoba_n  = normalize(xwoba,       pool('xwoba', 0.320))
        barrel_n = normalize(barrel_rate, pool('barrel_rate', 0.06))
        ev_n     = normalize(avg_ev,      pool('avg_ev', 88.0))
        power = (xwoba_n * 0.45 + barrel_n * 0.35 + ev_n * 0.20)

        # Component 2: Pitcher matchup (higher hr9/ERA = more favorable for batter)
        p_hr9_n = normalize(p_hr9, pool('pitcher_hr_per_9_30d', 1.2))
        p_era_n = normalize(p_era, pool('pitcher_era_30d', 4.0))
        matchup = (p_hr9_n * 0.6 + p_era_n * 0.4)

        # Component 3: Recent form
        form = normalize(hr_rate_14d, pool('recent_hr_rate_14d', 0.03))

        # Component 4: Park + environment (park_factor + wind + temp boost)
        pf_n   = normalize(park_factor, pool('park_factor', 1.0))
        wind_n = normalize(wind_out,    pool('wind_out_factor', 1.0))
        temp_n = normalize(max(temp_f, 40.0), [max(safe_float(r.get('temp_f', 72.0)), 40.0) for r in all_rows])
        env = (pf_n * 0.5 + wind_n * 0.3 + temp_n * 0.2)

        # Component 5: Freshness (low fatigue = good, so invert)
        freshness = normalize(fatigue, pool('travel_fatigue_score', 0.0), invert=True)

        # Weighted composite
        raw = (power * 0.30 + matchup * 0.25 + form * 0.20 + env * 0.15 + freshness * 0.10)
        return round(raw * 100, 1)

    def compute_pitcher_strikeout_score(self, row: dict, all_rows: list) -> float:
        """Starter strikeout score (0-100), separate from model probability."""
        def safe_float(v, default=0.0):
            try:
                return float(v) if v is not None and not pd.isna(v) else default
            except Exception:
                return default

        def normalize(val, vals):
            vals = [safe_float(v) for v in vals]
            lo, hi = min(vals), max(vals)
            if hi == lo:
                return 0.5
            return (safe_float(val) - lo) / (hi - lo)

        def pool(key, default):
            return [safe_float(r.get(key, default), default) for r in all_rows]

        k9 = safe_float(row.get('starter_k_per_9_30d'), 8.0)
        avg_so = safe_float(row.get('starter_avg_so_30d'), 4.5)
        season_k = safe_float(row.get('starter_k_percent_season'), 22.0)
        whiff = safe_float(row.get('starter_whiff_percent_season'), 24.0)
        opp_k = safe_float(row.get('opp_team_k_rate_14d'), 0.22)
        opp_pa = safe_float(row.get('opp_team_pa_per_game_14d'), 38.0)
        avg_ip = safe_float(row.get('starter_avg_ip_30d'), 5.0)

        components = (
            normalize(k9, pool('starter_k_per_9_30d', 8.0)) * 0.26 +
            normalize(avg_so, pool('starter_avg_so_30d', 4.5)) * 0.24 +
            normalize(season_k, pool('starter_k_percent_season', 22.0)) * 0.16 +
            normalize(whiff, pool('starter_whiff_percent_season', 24.0)) * 0.14 +
            normalize(opp_k, pool('opp_team_k_rate_14d', 0.22)) * 0.12 +
            normalize(opp_pa, pool('opp_team_pa_per_game_14d', 38.0)) * 0.04 +
            normalize(avg_ip, pool('starter_avg_ip_30d', 5.0)) * 0.04
        )
        return round(components * 100, 1)

    def generate_daily_predictions_with_results(self, target_date: date, target: str = 'hr', so_threshold: int = 3) -> Dict:
        """Generate predictions for a date with actual results from DB or CSV dataset"""
        target = target.lower() if target else 'hr'
        if target == 'so':
            return self.generate_pitcher_strikeout_predictions(target_date, so_threshold=so_threshold)

        label_col = 'label' if target == 'hr' else f'label_{target}'
        model_data = self._model_data_for_target(target)

        if model_data is None:
            return {'date': target_date.isoformat(), 'target': target, 'predictions': [], 'stats': {}}

        # Try DB first
        day_df = None
        if self.db_conn:
            day_df = self._get_players_for_date(target_date)

        # Fallback to CSV
        if day_df is None or day_df.empty:
            if self.dataset is None or self.dataset.empty:
                return {'date': target_date.isoformat(), 'target': target, 'predictions': [], 'stats': {}}
            day_df = self.dataset[self.dataset['game_date'] == target_date].copy()

        if day_df.empty:
            return {'date': target_date.isoformat(), 'target': target, 'predictions': [], 'stats': {}}

        # Collect all rows for normalization in compute_composite_score
        all_rows_list = [row.to_dict() for _, row in day_df.iterrows()]

        predictions = []
        for _, row in day_df.iterrows():
            # Build feature_row first so defaults are available for model features too
            feature_row = dict(row)
            feature_row.setdefault('pitcher_era_30d', 4.5)
            feature_row.setdefault('pitcher_hr_per_9_30d', 1.2)
            feature_row.setdefault('pitcher_k_per_9_30d', 20.0)
            feature_row.setdefault('pitcher_whip_30d', 1.3)
            feature_row.setdefault('park_factor', self._get_park_factor(row.get('home_team', '')))
            feature_row.setdefault('travel_distance_miles', 0.0)
            feature_row.setdefault('timezone_changes', 0)
            feature_row.setdefault('travel_fatigue_score', 72.0)
            feature_row.setdefault('wind_speed_mph', 0.0)
            feature_row.setdefault('temp_f', 72.0)
            feature_row.setdefault('precip_prob', 0.0)
            feature_row.setdefault('wind_out_factor', 1.0)
            feature_row.setdefault('recent_hr_rate_14d', 0.03)
            feature_row.setdefault('recent_hit_rate_14d', 0.25)
            feature_row.setdefault('recent_so_rate_14d', 0.20)

            features = {
                feat: feature_row.get(feat, model_data['train_medians'].get(feat, 0))
                for feat in model_data['features']
            }
            pred = self.predict(features, target=target)

            # Skip STRONG_SELL predictions
            if pred.get('signal') == 'STRONG_SELL':
                continue

            # Fix: derive did_occur from actual stat columns when label col missing (DB rows)
            if label_col in row and row.get(label_col) is not None:
                did_occur = bool(row.get(label_col, False))
            elif target == 'hr':
                did_occur = int(row.get('home_runs', 0) or 0) > 0
            elif target == 'hit':
                did_occur = int(row.get('hits', 0) or 0) > 0
            elif target == 'so':
                did_occur = int(row.get('strikeouts', 0) or 0) > 0
            else:
                did_occur = False

            # Fix: derive is_home from team vs home_team when column missing (DB rows)
            is_home = row.get('is_home')
            if is_home is None:
                is_home = (row.get('team') == row.get('home_team'))
            team_code = self._normalize_team_code(row.get('team'))
            home_team = self._normalize_team_code(row.get('home_team'))
            away_team = self._normalize_team_code(row.get('away_team'))
            opponent = away_team if is_home else home_team

            # Use home_runs, hits, strikeouts from DB (or label columns from CSV)
            actual_count = 0
            if target == 'hr':
                actual_count = int(row.get('home_runs', row.get('hr_count', 0)) or 0)
            elif target == 'hit':
                actual_count = int(row.get('hits', row.get('hit_count', 0)) or 0)
            elif target == 'so':
                actual_count = int(row.get('strikeouts', row.get('so_count', 0)) or 0)

            prediction = {
                'player_name': row.get('player_name', ''),
                'team': team_code,
                'opponent': opponent,
                'pitcher_name': row.get('pitcher_name'),
                'is_home': is_home,
                **pred,
                'signal_label': pred.get('signal'),
                'score': self.compute_composite_score(feature_row, all_rows_list),
                'odds_decimal': None,
                f'actual_{target}': did_occur,
                f'actual_{target}_count': actual_count,
            }
            predictions.append(prediction)

        deduped_predictions = {}
        for pred in predictions:
            dedupe_key = (
                pred.get('player_name'),
                pred.get('team'),
                pred.get('opponent'),
            )
            existing = deduped_predictions.get(dedupe_key)
            if existing is None or pred.get('probability', 0) > existing.get('probability', 0):
                deduped_predictions[dedupe_key] = pred

        predictions = list(deduped_predictions.values())

        predictions.sort(key=lambda x: x['probability'], reverse=True)

        # Reassign signals by rank within this slate (percentile-based per date)
        # Top 10% → STRONG_BUY, next 15% → BUY, next 25% → MODERATE,
        # next 25% → AVOID, bottom 25% → STRONG_SELL
        n = len(predictions)
        if n > 0:
            cutoffs = [
                (0.10, 'STRONG_BUY'),
                (0.25, 'BUY'),
                (0.50, 'MODERATE'),
                (0.75, 'AVOID'),
                (1.00, 'STRONG_SELL'),
            ]
            for i, pred in enumerate(predictions):
                rank_pct = i / n
                for threshold, label in cutoffs:
                    if rank_pct < threshold:
                        pred['signal_label'] = label
                        pred['signal'] = label
                        break

        total = len(predictions)
        hits = sum(1 for p in predictions if p[f'actual_{target}'])
        stats = {
            'total_picks': total,
            f'total_{target}s': hits,
            'hit_rate': round(hits / total * 100, 1) if total > 0 else 0,
        }

        available_teams = sorted({
            pred['team'] for pred in predictions
            if pred.get('team')
        })
        available_matchups = sorted({
            ' vs '.join(sorted([pred['team'], pred['opponent']]))
            for pred in predictions
            if pred.get('team') and pred.get('opponent')
        })

        result = {
            'date': target_date.isoformat(),
            'target': target,
            'predictions': predictions,
            'stats': stats,
            'available_teams': available_teams,
            'available_matchups': available_matchups,
        }

        # Sanitize NaN/Inf values before returning
        return _sanitize_for_json(result)

    def generate_pitcher_strikeout_predictions(self, target_date: date, so_threshold: int = 3) -> Dict:
        """Generate starter strikeout predictions using the selected threshold model."""
        target = 'so'
        so_threshold = self._so_threshold(str(so_threshold))
        label_col = f'label_so_{so_threshold}_plus'
        model_data = self._model_data_for_target(target, so_threshold=so_threshold)

        if model_data is None:
            return {'date': target_date.isoformat(), 'target': target, 'predictions': [], 'stats': {}}

        day_df = self._get_pitcher_predictions_for_date(target_date)
        if day_df.empty:
            return {'date': target_date.isoformat(), 'target': target, 'predictions': [], 'stats': {}}

        all_rows_list = [row.to_dict() for _, row in day_df.iterrows()]
        predictions = []

        for _, row in day_df.iterrows():
            feature_row = dict(row)
            features = {
                feat: feature_row.get(feat, model_data['train_medians'].get(feat, 0))
                for feat in model_data['features']
            }
            pred = self.predict(features, target=target, model_data_override=model_data)

            team_code = self._normalize_team_code(row.get('team'))
            opponent = self._normalize_team_code(row.get('opponent_team'))
            actual_count = int(row.get('actual_strikeouts', 0) or 0)
            did_occur = bool(row.get(label_col, 0)) if label_col in row else actual_count >= so_threshold

            prediction = {
                'player_name': row.get('starter_name', ''),
                'team': team_code,
                'opponent': opponent,
                'pitcher_name': row.get('starter_name', ''),
                'is_home': bool(row.get('is_home', False)),
                **pred,
                'signal_label': pred.get('signal'),
                'score': self.compute_pitcher_strikeout_score(feature_row, all_rows_list),
                'odds_decimal': None,
                'prop_target': f'{so_threshold}+ K',
                'so_threshold': so_threshold,
                'actual_so': did_occur,
                'actual_so_count': actual_count,
            }
            predictions.append(prediction)

        predictions.sort(key=lambda x: x['probability'], reverse=True)

        n = len(predictions)
        if n > 0:
            cutoffs = [
                (0.10, 'STRONG_BUY'),
                (0.25, 'BUY'),
                (0.50, 'MODERATE'),
                (0.75, 'AVOID'),
                (1.00, 'STRONG_SELL'),
            ]
            for i, pred in enumerate(predictions):
                rank_pct = i / n
                for threshold, label in cutoffs:
                    if rank_pct < threshold:
                        pred['signal_label'] = label
                        pred['signal'] = label
                        break

        total = len(predictions)
        hits = sum(1 for p in predictions if p['actual_so'])
        result = {
            'date': target_date.isoformat(),
            'target': target,
            'predictions': predictions,
            'stats': {
                'total_picks': total,
                'total_sos': hits,
                'hit_rate': round(hits / total * 100, 1) if total > 0 else 0,
                'prop_target': f'{so_threshold}+ K',
                'so_threshold': so_threshold,
            },
            'available_teams': sorted({pred['team'] for pred in predictions if pred.get('team')}),
            'available_matchups': sorted({
                ' vs '.join(sorted([pred['team'], pred['opponent']]))
                for pred in predictions if pred.get('team') and pred.get('opponent')
            }),
        }
        return _sanitize_for_json(result)


# Initialize prediction engine
engine = PredictionEngine()


@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('dashboard.html')


@app.route('/api/dates')
def get_available_dates():
    """Get all available dates with picks"""
    dates = engine.get_available_dates()
    return jsonify({
        'total_dates': len(dates),
        'date_range': {
            'earliest': dates[-1]['analysis_date'] if dates else None,
            'latest': dates[0]['analysis_date'] if dates else None
        },
        'dates': dates
    })


@app.route('/api/predictions/<date_str>')
def get_predictions_for_date(date_str):
    """Get predictions for a specific date with results"""
    try:
        target_date = datetime.strptime(date_str, '%Y-%m-%d').date()
        target = request.args.get('target', 'hr').lower()
        so_threshold = engine._so_threshold(request.args.get('so_threshold'))
        data = engine.generate_daily_predictions_with_results(target_date, target=target, so_threshold=so_threshold)

        return jsonify(_sanitize_for_json(data))
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/predictions/today')
def get_today_predictions():
    """Get today's predictions (will be empty if no Hellraiser picks)"""
    today = date.today()
    so_threshold = engine._so_threshold(request.args.get('so_threshold'))
    data = engine.generate_daily_predictions_with_results(today, so_threshold=so_threshold)
    return jsonify(_sanitize_for_json(data))


@app.route('/api/model/stats')
def get_model_stats():
    """Get model statistics and artifact status for all targets"""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    artifacts_dir = os.path.join(base_dir, 'models', 'artifacts')

    models_status = {}
    any_artifacts_ready = False
    needs_reload = False
    for target in ['hr', 'hit', 'so']:
        if target == 'so':
            artifacts_exist = all(
                os.path.exists(os.path.join(artifacts_dir, f'{prefix}_xgb.json')) and
                os.path.exists(os.path.join(artifacts_dir, f'{prefix}_lgb.txt')) and
                os.path.exists(os.path.join(artifacts_dir, f'{prefix}_meta.pkl'))
                for prefix in SO_THRESHOLD_PREFIX.values()
            )
            artifact_prefix = 'multi-threshold starter K'
        else:
            artifact_prefix = MODEL_ARTIFACT_PREFIX[target]
            artifacts_exist = (
                os.path.exists(os.path.join(artifacts_dir, f'{artifact_prefix}_xgb.json')) and
                os.path.exists(os.path.join(artifacts_dir, f'{artifact_prefix}_lgb.txt')) and
                os.path.exists(os.path.join(artifacts_dir, f'{artifact_prefix}_meta.pkl'))
            )
        if artifacts_exist:
            any_artifacts_ready = True
        if artifacts_exist and engine.models.get(target) is None:
            needs_reload = True
        models_status[target] = {
            'artifacts_ready': artifacts_exist,
            'loaded': engine.models.get(target) is not None,
            'features': len(engine.models[target]['features']) if engine.models.get(target) else 0,
            'artifact_prefix': artifact_prefix,
        }

    if needs_reload:
        engine.reload_models()
        for target in ['hr', 'hit', 'so']:
            models_status[target]['loaded'] = engine.models.get(target) is not None
            models_status[target]['features'] = len(engine.models[target]['features']) if engine.models.get(target) else 0

    model_ready = engine.model is not None
    return jsonify({
        'model_type': 'XGBoost + LightGBM + LogisticRegression (v4 Ensemble)',
        'models': models_status,
        'primary_model_status': 'ready' if model_ready else 'unavailable',
        'model_status': 'ready' if model_ready else 'unavailable',
        'artifacts_ready': any_artifacts_ready,
        'hr_stats': {
            'roc_auc': 0.9458,
            'strong_buy_hit_rate': 84.4,
            'buy_hit_rate': 49.4,
        },
        'training_samples': 11462,
        'data_range': '2025-06-23 to 2025-09-01',
        'available_dates': 62
    })


@app.route('/api/model/train', methods=['POST'])
def train_model():
    """Start async model training"""
    try:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        # Start training in background
        subprocess.Popen(
            ['python', '-m', 'models.train_models_v4'],
            cwd=base_dir,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

        logger.info("Training started in background")
        return jsonify({'status': 'training_started'}), 202
    except Exception as e:
        logger.error(f"Failed to start training: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/model/reload', methods=['POST'])
def reload_model():
    """Manually reload model artifacts from disk"""
    try:
        engine.reload_models()
        loaded = [t for t in ['hr', 'hit', 'so'] if engine.models.get(t)]
        return jsonify({'status': 'reloaded', 'loaded_models': loaded})
    except Exception as e:
        logger.error(f"Failed to reload models: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/analysis/summary')
def get_analysis_summary():
    """Get aggregated analysis summary for date range"""
    from_date_str = request.args.get('from')
    to_date_str = request.args.get('to')

    if not from_date_str or not to_date_str:
        return jsonify({'error': 'Missing from/to date parameters'}), 400

    try:
        from_date = datetime.strptime(from_date_str, '%Y-%m-%d').date()
        to_date = datetime.strptime(to_date_str, '%Y-%m-%d').date()

        if not engine.db_conn:
            return jsonify({'error': 'Database connection unavailable'}), 500

        cursor = engine.db_conn.cursor(cursor_factory=RealDictCursor)

        # Get all predictions in date range
        cursor.execute("""
            SELECT * FROM hellraiser_picks
            WHERE analysis_date BETWEEN %s AND %s
            ORDER BY analysis_date DESC
        """, (from_date, to_date))

        picks = cursor.fetchall()
        if not picks:
            return jsonify({'error': 'No picks found in date range'}), 404

        picks_df = pd.DataFrame(picks)

        # Get HR results
        all_preds = []
        for _, pick in picks_df.iterrows():
            game_date = pick.get('analysis_date')
            hr_results = engine.get_hr_results_for_date(game_date)

            player_name = pick.get('player_name', '')
            lastname = player_name.strip().lower().split()[-1] if player_name else ''
            hr_info = hr_results.get(lastname)

            all_preds.append({
                'player_name': player_name,
                'classification': pick.get('classification'),
                'barrel_rate': pick.get('barrel_rate'),
                'wind_speed': pick.get('wind_speed'),
                'travel_distance': pick.get('travel_distance'),
                'park_factor': pick.get('park_factor'),
                'actual_hr': hr_info is not None,
                'odds_decimal': pick.get('odds_decimal')
            })

        preds_df = pd.DataFrame(all_preds)

        # Feature coverage
        feature_coverage = {
            'barrel_rate': (preds_df['barrel_rate'].notna().sum() / len(preds_df) * 100) if len(preds_df) > 0 else 0,
            'weather_data': (preds_df['wind_speed'].notna().sum() / len(preds_df) * 100) if len(preds_df) > 0 else 0,
            'travel_data': (preds_df['travel_distance'].notna().sum() / len(preds_df) * 100) if len(preds_df) > 0 else 0,
            'park_factor': (preds_df['park_factor'].notna().sum() / len(preds_df) * 100) if len(preds_df) > 0 else 0,
        }

        # Aggregated stats
        hits = preds_df['actual_hr'].sum()
        total = len(preds_df)
        total_profit = 0

        for _, pred in preds_df.iterrows():
            if pred['actual_hr']:
                odds = pred.get('odds_decimal', 1)
                if odds and odds > 1:
                    total_profit += (odds - 1)
            else:
                total_profit -= 1

        aggregated_stats = {
            'total_picks': total,
            'total_hits': int(hits),
            'overall_hit_rate': (hits / total * 100) if total > 0 else 0,
            'roi': (total_profit / total * 100) if total > 0 else 0,
        }

        # Hit rate by tier
        hit_rate_by_tier = {}
        for classification in preds_df['classification'].unique():
            tier_preds = preds_df[preds_df['classification'] == classification]
            tier_hits = tier_preds['actual_hr'].sum()
            hit_rate_by_tier[str(classification)] = {
                'hit_rate': (tier_hits / len(tier_preds) * 100) if len(tier_preds) > 0 else 0,
                'count': len(tier_preds)
            }

        # AUC trend (simplified - daily hit rate)
        auc_trend = []
        for analysis_date in sorted(preds_df.index.unique() if hasattr(preds_df.index, 'unique') else []):
            daily_preds = preds_df.iloc[[i for i, p in enumerate(picks_df['analysis_date']) if p == analysis_date]]
            if len(daily_preds) > 0:
                daily_hits = daily_preds['actual_hr'].sum()
                daily_rate = daily_hits / len(daily_preds)
                auc_trend.append({
                    'date': str(analysis_date),
                    'auc': min(0.99, max(0.5, daily_rate + 0.5))  # Simple approximation
                })

        # Feature importance (mock data - would come from trained model)
        feature_importance = [
            {'feature': 'barrel_rate', 'importance': 0.185},
            {'feature': 'exit_velocity_avg', 'importance': 0.142},
            {'feature': 'pitcher_hr_per_9', 'importance': 0.138},
            {'feature': 'sweet_spot_percent', 'importance': 0.125},
            {'feature': 'park_factor', 'importance': 0.098},
            {'feature': 'travel_fatigue_score', 'importance': 0.087},
            {'feature': 'wind_speed_mph', 'importance': 0.065},
            {'feature': 'hard_hit_percent', 'importance': 0.063},
            {'feature': 'pitcher_whip', 'importance': 0.052},
            {'feature': 'temp_f', 'importance': 0.045},
        ]

        return jsonify({
            'feature_coverage': feature_coverage,
            'aggregated_stats': aggregated_stats,
            'hit_rate_by_tier': hit_rate_by_tier,
            'auc_trend': auc_trend,
            'feature_importance': feature_importance
        })

    except Exception as e:
        logger.error(f"Error in analysis summary: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/results/<date_str>')
def get_results_for_date(date_str):
    """Get actual HR results for a date"""
    try:
        target_date = datetime.strptime(date_str, '%Y-%m-%d').date()
        results = engine.get_hr_results_for_date(target_date)

        return jsonify({
            'date': date_str,
            'total_hrs': len(results),
            'hr_hitters': results
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/analysis')
def analysis():
    """Analysis dashboard page"""
    return render_template('analysis.html')


@app.route('/master-list')
def master_list():
    """Master list page with all predictions and filters"""
    return render_template('master_list.html')


@app.route('/api/stats/all-targets')
def get_all_target_stats():
    """Historical pick accuracy — counts top-25% picks per date (STRONG_BUY + BUY tier).
    Signals are rank-based within each date's slate, so picks here mirror what the
    predictions endpoint shows."""
    so_threshold = engine._so_threshold(request.args.get('so_threshold'))
    label_map = {'hr': 'label', 'hit': 'label_hit', 'so': f'label_so_{so_threshold}_plus'}
    result = {}

    for target, label_col in label_map.items():
        target_so_threshold = so_threshold if target == 'so' else 3
        model_data = engine._model_data_for_target(target, so_threshold=target_so_threshold)
        dataset = engine._dataset_for_target(target)
        if dataset is None or dataset.empty:
            result[target] = {'picks': 0, 'hit_rate': 0}
            continue
        if not model_data or label_col not in dataset.columns:
            result[target] = {'picks': 0, 'hit_rate': 0}
            continue

        df = dataset.dropna(subset=[label_col]).copy()
        X = df.reindex(columns=model_data['features'])
        for feat in model_data['features']:
            X[feat] = pd.to_numeric(X[feat], errors='coerce').fillna(
                model_data['train_medians'].get(feat, 0)
            )

        proba = model_data['xgb'].predict_proba(X)[:, 1]
        df = df.copy()
        df['_proba'] = proba
        actuals = df[label_col].astype(bool).values

        # Rank within each date — top 25% per date = STRONG_BUY + BUY = "picks"
        if 'game_date' in df.columns:
            mask = np.zeros(len(df), dtype=bool)
            for _, grp in df.groupby('game_date'):
                idx = grp.index
                n = len(idx)
                sorted_idx = grp['_proba'].sort_values(ascending=False).index
                top25 = sorted_idx[:max(1, int(n * 0.25))]
                mask[df.index.get_indexer(top25)] = True
        else:
            # Fallback: global top 25%
            cutoff = np.percentile(proba, 75)
            mask = proba >= cutoff

        picks = int(mask.sum())
        hits = int(actuals[mask].sum()) if picks > 0 else 0
        hit_rate = round(hits / picks * 100, 1) if picks > 0 else 0

        result[target] = {'picks': picks, 'hit_rate': hit_rate}

    return jsonify(result)


@app.route('/api/predictions/all')
def get_all_predictions():
    """Get all predictions across all dates with results"""
    target = request.args.get('target', 'hr').lower()
    so_threshold = engine._so_threshold(request.args.get('so_threshold'))
    actual_field = f'actual_{target}'
    dates = engine.get_available_dates()
    all_predictions = []

    for d in dates:
        try:
            date_str = d.get('analysis_date') or d.get('date', '')
            if not date_str:
                continue

            if isinstance(date_str, str):
                if 'T' in date_str:
                    date_str = date_str.split('T')[0]
                target_date = datetime.strptime(date_str, '%Y-%m-%d').date()
            else:
                target_date = date_str

            data = engine.generate_daily_predictions_with_results(target_date, target=target, so_threshold=so_threshold)

            for pred in data.get('predictions', []):
                pred['analysis_date'] = target_date.isoformat()
                all_predictions.append(pred)
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            continue

    total_profit = 0
    total_bets = 0
    hits = 0

    for pred in all_predictions:
        if pred.get(actual_field) is not None:
            total_bets += 1
            odds = pred.get('odds_decimal')
            if pred.get(actual_field) == True:
                hits += 1
                if odds and odds > 1:
                    total_profit += (odds - 1)
            else:
                total_profit -= 1

    hit_rate = (hits / total_bets * 100) if total_bets > 0 else 0
    roi = (total_profit / total_bets * 100) if total_bets > 0 else 0

    result = {
        'total_predictions': len(all_predictions),
        'predictions': all_predictions,
        'overall_stats': {
            'hit_rate': round(hit_rate, 1),
            'roi': round(roi, 1)
        }
    }

    return jsonify(_sanitize_for_json(result))


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002, debug=True)
