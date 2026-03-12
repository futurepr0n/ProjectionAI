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

from data.name_utils import normalize_name, resolve_name_match

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


def _safe_float(value, default=0.0):
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return default
        return float(value)
    except Exception:
        return default


def _has_value(value) -> bool:
    try:
        return value is not None and not pd.isna(value)
    except Exception:
        return value is not None


def _sigmoid(value):
    try:
        return 1.0 / (1.0 + math.exp(-float(value)))
    except Exception:
        return 0.5


def _normalize_name_key(value: str) -> str:
    text = re.sub(r'[^a-z0-9 ]+', '', str(value or '').lower()).strip()
    return re.sub(r'\s+', ' ', text)


def _surname_key(value: str) -> str:
    text = _normalize_name_key(value)
    if not text:
        return ''
    parts = [part for part in text.split(' ') if part]
    suffixes = {'jr', 'sr', 'ii', 'iii', 'iv'}
    parts = [part for part in parts if part not in suffixes]
    return parts[-1] if parts else ''

app = Flask(__name__)

DEFAULT_HITTER_RECENT_LOOKBACK_GAMES = 20
RECENT_FORM_WINDOWS = (3, 5, 10, 20)
RECENT_FORM_SHRINKAGE_PA = 20.0

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


def _name_from_last_first(value: str) -> str:
    if value is None:
        return ''
    text = str(value).strip()
    if ',' in text:
        parts = [part.strip() for part in text.split(',') if part.strip()]
        if len(parts) >= 2:
            return f"{parts[1]} {parts[0]}".strip()
    return text


def _normalize_pitch_type(value: str) -> str:
    text = str(value or '').strip().lower()
    return PITCH_TYPE_ALIASES.get(text, str(value or '').strip().upper()[:2])


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
        self.training_results = {}
        self._canonical_name_cache = {}
        self._player_lookup_cache = None
        self._pitch_matchup_cache = None
        self.load_model()
        self.connect_db()
        self.dataset = self._load_hitter_dataset()
        self.pitcher_so_dataset = self._load_pitcher_so_dataset()
        self._load_training_results()
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
                serving_mode = self._determine_serving_mode(xgb_model, feature_names)

                self.models[target] = {
                    'xgb': xgb_model,
                    'lgb': lgb_model,
                    'meta': meta_model,
                    'features': feature_names,
                    'train_medians': train_medians,
                    'artifact_prefix': artifact_prefix,
                    'serving_mode': serving_mode,
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
                serving_mode = self._determine_serving_mode(xgb_model, feature_names)

                self.so_models[threshold] = {
                    'xgb': xgb_model,
                    'lgb': lgb_model,
                    'meta': meta_artifact.get('meta'),
                    'features': feature_names,
                    'train_medians': train_medians,
                    'artifact_prefix': artifact_prefix,
                    'threshold': threshold,
                    'serving_mode': serving_mode,
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

    def _determine_serving_mode(self, xgb_model, feature_names: List[str]) -> str:
        try:
            booster_features = list(xgb_model.get_booster().feature_names or [])
        except Exception:
            booster_features = []
        if booster_features and booster_features != list(feature_names):
            logger.warning("XGBoost artifact feature mismatch detected. Falling back to LightGBM primary serving.")
            return 'lgb_primary'
        return 'xgb_primary'

    def _load_training_results(self):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        artifacts_dir = os.path.join(base_dir, 'models', 'artifacts')
        result_files = {
            'hr_hit': os.path.join(artifacts_dir, 'training_results.json'),
            'pitcher_so': os.path.join(artifacts_dir, 'pitcher_strikeout_training_results.json'),
        }
        loaded = {}
        for key, path in result_files.items():
            try:
                if os.path.exists(path):
                    with open(path, 'r', encoding='utf-8') as fh:
                        loaded[key] = json.load(fh)
                else:
                    loaded[key] = {}
            except Exception as exc:
                logger.warning("Could not load training results from %s: %s", path, exc)
                loaded[key] = {}
        self.training_results = loaded

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
        """Load hitter feature snapshot, preferring derived DB table over CSV."""
        _ACTION_RE = re.compile(
            r'\s+(hit|struck|walk|fly|bunt|by|grounded|lined|flied|popped|'
            r'singled|doubled|tripled|homered|reached|sacrifice|intentional|'
            r'called|swinging)\b.*$',
            re.IGNORECASE
        )

        if self.db_conn:
            try:
                df = pd.read_sql_query("SELECT * FROM derived_hitter_features", self.db_conn)
                if not df.empty:
                    df['game_date'] = pd.to_datetime(df['game_date']).dt.date
                    df['player_name'] = df['player_name'].str.replace(_ACTION_RE, '', regex=True).str.strip()
                    logger.info(f"✅ Hitter dataset loaded from DB: {len(df)} rows, {df['game_date'].nunique()} dates")
                    return df
            except Exception as exc:
                logger.warning("⚠️ Could not load derived_hitter_features from DB: %s", exc)

        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        csv_path = os.path.join(base_dir, 'data', 'complete_dataset.csv')
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            df['game_date'] = pd.to_datetime(df['game_date']).dt.date
            df['player_name'] = df['player_name'].str.replace(_ACTION_RE, '', regex=True).str.strip()
            logger.info(f"✅ Hitter dataset loaded from CSV: {len(df)} rows, {df['game_date'].nunique()} dates")
            return df
        logger.warning("⚠️ complete_dataset.csv not found")
        return pd.DataFrame()

    def _load_pitcher_so_dataset(self):
        """Load starter strikeout feature snapshot, preferring derived DB table over CSV."""
        if self.db_conn:
            try:
                df = pd.read_sql_query("SELECT * FROM derived_pitcher_so_features", self.db_conn)
                if not df.empty:
                    df['game_date'] = pd.to_datetime(df['game_date']).dt.date
                    logger.info(f"✅ Pitcher SO dataset loaded from DB: {len(df)} rows, {df['game_date'].nunique()} dates")
                    return df
            except Exception as exc:
                logger.warning("⚠️ Could not load derived_pitcher_so_features from DB: %s", exc)

        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        csv_path = os.path.join(base_dir, 'data', 'pitcher_strikeout_dataset.csv')
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            df['game_date'] = pd.to_datetime(df['game_date']).dt.date
            logger.info(f"✅ Pitcher SO dataset loaded from CSV: {len(df)} rows, {df['game_date'].nunique()} dates")
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

    def _training_summary_for_target(self, target: str, so_threshold: Optional[int] = None) -> Dict:
        if target == 'so':
            key = f'pitcher_so_{self._so_threshold(str(so_threshold or 3))}_plus'
            return self.training_results.get('pitcher_so', {}).get(key, {}) or {}
        return self.training_results.get('hr_hit', {}).get(target, {}) or {}

    def _artifact_version_for_model(self, model_data: Optional[Dict]) -> Dict:
        if not model_data:
            return {}
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        artifacts_dir = os.path.join(base_dir, 'models', 'artifacts')
        prefix = model_data.get('artifact_prefix')
        if not prefix:
            return {}
        paths = [
            os.path.join(artifacts_dir, f'{prefix}_xgb.json'),
            os.path.join(artifacts_dir, f'{prefix}_lgb.txt'),
            os.path.join(artifacts_dir, f'{prefix}_meta.pkl'),
        ]
        existing = [p for p in paths if os.path.exists(p)]
        if not existing:
            return {'artifact_prefix': prefix}
        latest_mtime = max(os.path.getmtime(p) for p in existing)
        return {
            'artifact_prefix': prefix,
            'updated_at': datetime.fromtimestamp(latest_mtime).isoformat(),
            'feature_count': len(model_data.get('features', [])),
            'serving_mode': model_data.get('serving_mode', 'xgb_primary'),
        }

    def _expected_calibration_error(self, y_true: np.ndarray, y_prob: np.ndarray, bins: int = 10) -> float:
        if len(y_true) == 0:
            return float('nan')
        y_true = np.asarray(y_true, dtype=float)
        y_prob = np.asarray(y_prob, dtype=float)
        edges = np.linspace(0.0, 1.0, bins + 1)
        total = len(y_true)
        ece = 0.0
        for idx in range(bins):
            if idx == bins - 1:
                mask = (y_prob >= edges[idx]) & (y_prob <= edges[idx + 1])
            else:
                mask = (y_prob >= edges[idx]) & (y_prob < edges[idx + 1])
            if not np.any(mask):
                continue
            avg_prob = float(np.mean(y_prob[mask]))
            avg_true = float(np.mean(y_true[mask]))
            ece += abs(avg_prob - avg_true) * (np.sum(mask) / total)
        return float(ece)

    def get_target_calibration_summary(self, target: str, so_threshold: Optional[int] = None, classification: Optional[str] = None) -> Dict:
        target = target.lower()
        model_data = self._model_data_for_target(target, so_threshold=so_threshold)
        dataset = self._dataset_for_target(target)
        label_col = 'label' if target == 'hr' else ('label_hit' if target == 'hit' else f'label_so_{self._so_threshold(str(so_threshold or 3))}_plus')
        if model_data is None or dataset is None or dataset.empty or label_col not in dataset.columns:
            return {}

        df = dataset.dropna(subset=[label_col]).copy()
        if df.empty:
            return {}

        X = df.reindex(columns=model_data['features'])
        for feat in model_data['features']:
            X[feat] = pd.to_numeric(X[feat], errors='coerce').fillna(model_data['train_medians'].get(feat, 0))

        proba = self._predict_primary_proba(X, model_data)
        df['_proba'] = proba
        df['_actual'] = df[label_col].astype(int)

        classification = (classification or '').strip().upper()
        if classification:
            if 'game_date' in df.columns:
                signals = pd.Series(index=df.index, dtype=object)
                cutoffs = [
                    (0.10, 'STRONG_BUY'),
                    (0.25, 'BUY'),
                    (0.50, 'MODERATE'),
                    (0.75, 'AVOID'),
                    (1.00, 'STRONG_SELL'),
                ]
                for _, grp in df.groupby('game_date'):
                    sorted_idx = grp['_proba'].sort_values(ascending=False).index.tolist()
                    n = len(sorted_idx)
                    for rank, idx in enumerate(sorted_idx):
                        rank_pct = rank / n if n else 1.0
                        label = 'STRONG_SELL'
                        for threshold, candidate in cutoffs:
                            if rank_pct < threshold:
                                label = candidate
                                break
                        signals.at[idx] = label
                df = df.assign(_signal_label=signals)
                df = df[df['_signal_label'] == classification].copy()

        if df.empty:
            return {}

        y_true = df['_actual'].to_numpy(dtype=float)
        y_prob = df['_proba'].to_numpy(dtype=float)
        return {
            'rows': int(len(df)),
            'ece': round(self._expected_calibration_error(y_true, y_prob) * 100, 2),
            'avg_probability_pct': round(float(np.mean(y_prob)) * 100, 1),
            'actual_rate_pct': round(float(np.mean(y_true)) * 100, 1),
        }

    def get_target_model_info(self, target: str, so_threshold: Optional[int] = None, classification: Optional[str] = None) -> Dict:
        target = target.lower()
        model_data = self._model_data_for_target(target, so_threshold=so_threshold)
        training_summary = self._training_summary_for_target(target, so_threshold=so_threshold)
        holdout_metrics = training_summary.get('holdout_metrics', {})
        calibration = self.get_target_calibration_summary(target, so_threshold=so_threshold, classification=classification)

        info = {
            'target': target,
            'serving_mode': (model_data or {}).get('serving_mode', 'unavailable'),
            'artifact': self._artifact_version_for_model(model_data),
            'training': {
                'generated_at': training_summary.get('generated_at') or self.training_results.get('hr_hit', {}).get('generated_at') or self.training_results.get('pitcher_so', {}).get('generated_at'),
                'train_rows': training_summary.get('train_rows'),
                'holdout_rows': training_summary.get('holdout_rows'),
                'train_start_date': training_summary.get('train_start_date'),
                'train_end_date': training_summary.get('train_end_date'),
                'holdout_start_date': training_summary.get('holdout_start_date'),
                'holdout_end_date': training_summary.get('holdout_end_date'),
                'feature_count': len((model_data or {}).get('features', [])),
            },
            'holdout_metrics': {
                'xgb_roc_auc': _safe_float((holdout_metrics.get('xgb') or {}).get('roc_auc'), None),
                'lgb_roc_auc': _safe_float((holdout_metrics.get('lgb') or {}).get('roc_auc'), None),
                'meta_roc_auc': _safe_float((holdout_metrics.get('meta') or {}).get('roc_auc'), None),
                'xgb_brier': _safe_float((holdout_metrics.get('xgb') or {}).get('brier_score'), None),
                'lgb_brier': _safe_float((holdout_metrics.get('lgb') or {}).get('brier_score'), None),
                'meta_brier': _safe_float((holdout_metrics.get('meta') or {}).get('brier_score'), None),
            },
            'runtime_calibration': calibration,
        }
        return _sanitize_for_json(info)

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
                proba = self._predict_primary_proba(X, model_data)
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

    def _prediction_team_context(self, player_names: List[str]) -> Dict[str, Dict]:
        if not self.db_conn or not player_names:
            return {}

        unique_names = sorted({name for name in player_names if name})
        if not unique_names:
            return {}

        current_team_map = {}
        canonical_map = {}
        normalized_targets = {name: re.sub(r'[^a-z0-9 ]+', '', str(name).lower()).strip() for name in unique_names}

        cursor = self.db_conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute(
            """
            SELECT
                COALESCE(full_name, display_name) AS player_name,
                display_name,
                team_code
            FROM players
            WHERE COALESCE(full_name, display_name) IS NOT NULL
            """
        )
        for row in cursor.fetchall():
            candidate_names = [row.get('player_name'), row.get('display_name')]
            normalized_candidates = {
                re.sub(r'[^a-z0-9 ]+', '', str(candidate).lower()).strip()
                for candidate in candidate_names if candidate
            }
            matched_player_names = [
                requested_name for requested_name, normalized_name in normalized_targets.items()
                if normalized_name in normalized_candidates
            ]
            for requested_name in matched_player_names:
                canonical_name = normalized_targets[requested_name]
                current_team_map[requested_name] = self._normalize_team_code(row.get('team_code'))
                canonical_map[requested_name] = canonical_name

        if not canonical_map:
            return {}

        cursor.execute(
            """
            SELECT canonical_name, STRING_AGG(DISTINCT team_code, '|' ORDER BY team_code) AS roster_teams
            FROM official_team_roster_snapshots
            WHERE canonical_name = ANY(%s)
            GROUP BY canonical_name
            """,
            (list(set(canonical_map.values())),)
        )
        roster_history = {row['canonical_name']: row['roster_teams'] for row in cursor.fetchall()}

        cursor.execute(
            """
            SELECT team_code, transaction_text
            FROM official_mlb_transactions
            WHERE transaction_date BETWEEN DATE '2025-03-01' AND DATE '2025-11-30'
              AND team_code IS NOT NULL
              AND transaction_text IS NOT NULL
            """
        )
        transaction_history = {canonical_name: set() for canonical_name in set(canonical_map.values())}
        transaction_rows = cursor.fetchall()
        for row in transaction_rows:
            text = str(row['transaction_text'] or '').lower()
            for canonical_name in transaction_history:
                if canonical_name and canonical_name in text:
                    transaction_history[canonical_name].add(self._normalize_team_code(row['team_code']))

        context = {}
        for player_name in unique_names:
            canonical_name = canonical_map.get(player_name)
            roster_teams = roster_history.get(canonical_name, '')
            transaction_teams = '|'.join(sorted(transaction_history.get(canonical_name, set())))
            context[player_name] = {
                'current_team': current_team_map.get(player_name),
                'roster_teams': roster_teams,
                'transaction_teams': transaction_teams,
                'canonical_name': canonical_name,
            }
        return context

    def _annotate_prediction_team_validation(self, predictions: List[Dict]) -> List[Dict]:
        if not predictions:
            return predictions

        context_map = self._prediction_team_context([pred.get('player_name') for pred in predictions])
        for pred in predictions:
            context = context_map.get(pred.get('player_name')) or {}
            prediction_team = self._normalize_team_code(pred.get('team'))
            current_team = self._normalize_team_code(context.get('current_team'))
            roster_teams = [self._normalize_team_code(team) for team in (context.get('roster_teams') or '').split('|') if team]
            transaction_teams = [self._normalize_team_code(team) for team in (context.get('transaction_teams') or '').split('|') if team]

            validation = 'unknown'
            warning = False
            if prediction_team and current_team:
                if prediction_team == current_team:
                    validation = 'current_team_match'
                elif prediction_team in roster_teams or prediction_team in transaction_teams:
                    validation = 'historical_team_match'
                else:
                    validation = 'possible_team_mismatch'
                    warning = True
            elif prediction_team and (prediction_team in roster_teams or prediction_team in transaction_teams):
                validation = 'historical_team_match'

            pred['team_validation'] = {
                'status': validation,
                'warning': warning,
                'current_team': current_team,
                'official_roster_teams': roster_teams,
                'official_transaction_teams': transaction_teams,
            }
        return predictions

    def _normalize_team_code(self, team_code: Optional[str]) -> Optional[str]:
        """Collapse known team-code aliases to one dashboard-facing code."""
        if not team_code:
            return team_code
        return TEAM_CODE_ALIASES.get(str(team_code).strip().upper(), str(team_code).strip().upper())

    def _canonical_name(self, name: str) -> str:
        raw = str(name or '').strip()
        if not raw:
            return ''
        if raw not in self._canonical_name_cache:
            try:
                resolution = resolve_name_match(raw, self.db_conn)
                self._canonical_name_cache[raw] = resolution.get('canonical_name') or normalize_name(raw)
            except Exception:
                self._canonical_name_cache[raw] = normalize_name(raw)
        return self._canonical_name_cache[raw]

    def _load_player_lookup(self) -> Dict[str, Dict]:
        if self._player_lookup_cache is not None:
            return self._player_lookup_cache

        lookup = {'bats': {}, 'throws': {}}
        if not self.db_conn:
            self._player_lookup_cache = lookup
            return lookup

        try:
            cursor = self.db_conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute("""
                SELECT COALESCE(full_name, display_name) AS player_name, bats, throws
                FROM players
                WHERE COALESCE(full_name, display_name) IS NOT NULL
            """)
            rows = cursor.fetchall()
            for row in rows:
                player_name = str(row['player_name']).strip()
                canonical = self._canonical_name(player_name)
                normalized = normalize_name(player_name)
                bats = str(row.get('bats') or '').upper().strip()
                throws = str(row.get('throws') or '').upper().strip()
                for key in {canonical, normalized}:
                    if key:
                        if bats in {'L', 'R', 'S'}:
                            lookup['bats'][key] = bats
                        if throws in {'L', 'R'}:
                            lookup['throws'][key] = throws
        except Exception as exc:
            logger.warning("Could not load player lookup cache: %s", exc)

        self._player_lookup_cache = lookup
        return lookup

    def _ensure_pitch_matchup_cache(self) -> Dict[str, Dict]:
        if self._pitch_matchup_cache is not None:
            return self._pitch_matchup_cache

        cache = {'pitcher_profiles': {}, 'pitcher_profiles_by_name': {}, 'batter_profiles': {}}
        if not self.db_conn:
            self._pitch_matchup_cache = cache
            return cache

        try:
            cursor = self.db_conn.cursor(cursor_factory=RealDictCursor)
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
        except Exception as exc:
            logger.warning("Could not build pitch matchup cache: %s", exc)
            self._pitch_matchup_cache = cache
            return cache

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

        if not pitcher_arsenal.empty:
            pitcher_arsenal['pitcher_canonical'] = pitcher_arsenal['last_name_first_name'].apply(
                lambda value: self._canonical_name(_name_from_last_first(value))
            )
            pitcher_arsenal['team_normalized'] = pitcher_arsenal['team_name_alt'].apply(self._normalize_team_code)
            for col in ('pitch_usage', 'whiff_percent', 'k_percent', 'put_away'):
                pitcher_arsenal[col] = pd.to_numeric(pitcher_arsenal[col], errors='coerce')

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
                cache['pitcher_profiles'][(pitcher_canonical, team_code)] = profile
                cache['pitcher_profiles_by_name'].setdefault(pitcher_canonical, profile)

        if not hitter_arsenal.empty:
            hitter_arsenal['batter_canonical'] = hitter_arsenal['last_name_first_name'].apply(self._canonical_name)
            for col in ('pa', 'whiff_percent', 'k_percent', 'put_away', 'ba', 'slg', 'woba'):
                hitter_arsenal[col] = pd.to_numeric(hitter_arsenal[col], errors='coerce')
            for (batter_canonical, pitch_type), grp in hitter_arsenal.groupby(['batter_canonical', 'pitch_type'], dropna=False):
                weights = grp['pa'].fillna(0.0).clip(lower=0)
                weight_sum = weights.sum()
                cache['batter_profiles'][(batter_canonical, pitch_type)] = {
                    'batter_whiff_percent': float(np.average(grp['whiff_percent'].fillna(0), weights=weights)) if weight_sum > 0 else np.nan,
                    'batter_k_percent': float(np.average(grp['k_percent'].fillna(0), weights=weights)) if weight_sum > 0 else np.nan,
                    'batter_ba': float(np.average(grp['ba'].fillna(0), weights=weights)) if weight_sum > 0 else np.nan,
                    'batter_slg': float(np.average(grp['slg'].fillna(0), weights=weights)) if weight_sum > 0 else np.nan,
                    'batter_woba': float(np.average(grp['woba'].fillna(0), weights=weights)) if weight_sum > 0 else np.nan,
                }

        self._pitch_matchup_cache = cache
        return cache

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

    def _get_starter_options_for_date(self, target_date: date) -> List[Dict]:
        if not self.db_conn:
            return []

        try:
            cursor = self.db_conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute("""
                SELECT
                    dl.game_date,
                    dl.home_team,
                    dl.away_team,
                    dl.home_pitcher->>'name' AS home_pitcher_name,
                    dl.away_pitcher->>'name' AS away_pitcher_name
                FROM daily_lineups dl
                WHERE dl.game_date = %s
                ORDER BY dl.home_team, dl.away_team
            """, (target_date,))
            lineup_rows = cursor.fetchall()
            if not lineup_rows:
                return []

            cursor.execute("""
                SELECT
                    team_code,
                    COALESCE(display_name, full_name) AS pitcher_name
                FROM players
                WHERE player_type = 'pitcher'
                  AND active = true
                  AND COALESCE(display_name, full_name) IS NOT NULL
            """)
            players_df = pd.DataFrame(cursor.fetchall())
            candidate_map: Dict[str, List[str]] = {}
            if not players_df.empty:
                players_df['team_code'] = players_df['team_code'].apply(self._normalize_team_code)
                for team_code, group in players_df.groupby('team_code'):
                    names = sorted({str(name).strip() for name in group['pitcher_name'] if str(name).strip()})
                    candidate_map[team_code] = names

            options: List[Dict] = []
            for row in lineup_rows:
                home_team = self._normalize_team_code(row['home_team'])
                away_team = self._normalize_team_code(row['away_team'])
                home_pitcher = (row.get('home_pitcher_name') or '').strip()
                away_pitcher = (row.get('away_pitcher_name') or '').strip()

                def build_team_entry(team_code: str, current_pitcher: str) -> Dict:
                    candidates = candidate_map.get(team_code, []).copy()
                    if current_pitcher and current_pitcher not in candidates:
                        candidates.insert(0, current_pitcher)
                    return {
                        'team': team_code,
                        'current_pitcher': current_pitcher or None,
                        'candidates': candidates,
                    }

                options.append({
                    'matchup': f"{away_team} @ {home_team}",
                    'home_team': build_team_entry(home_team, home_pitcher),
                    'away_team': build_team_entry(away_team, away_pitcher),
                })
            return options
        except Exception as exc:
            logger.warning("Could not load starter options for %s: %s", target_date, exc)
            return []

    def _apply_starter_overrides(self, df: pd.DataFrame, target_date: date, starter_overrides: Optional[Dict[str, str]]) -> pd.DataFrame:
        if df.empty or not starter_overrides:
            return df

        normalized_overrides = {
            self._normalize_team_code(team): (pitcher or '').strip()
            for team, pitcher in starter_overrides.items()
            if team and pitcher and str(pitcher).strip()
        }
        if not normalized_overrides:
            return df

        stats_cache: Dict[Tuple[str, date], Dict] = {}
        updated = df.copy()
        team_code = updated.get('team', pd.Series(index=updated.index, dtype=object)).apply(self._normalize_team_code)
        home_team = updated.get('home_team', pd.Series(index=updated.index, dtype=object)).apply(self._normalize_team_code)
        away_team = updated.get('away_team', pd.Series(index=updated.index, dtype=object)).apply(self._normalize_team_code)
        opponent = np.where(team_code == home_team, away_team, home_team)
        updated['override_opponent_team'] = opponent
        updated['starter_override_applied'] = False

        for opp_team, pitcher_name in normalized_overrides.items():
            mask = updated['override_opponent_team'] == opp_team
            if not mask.any():
                continue

            cache_key = (pitcher_name, target_date)
            if cache_key not in stats_cache:
                stats_cache[cache_key] = self._get_pitcher_rolling_stats(pitcher_name, target_date)
            rolling = stats_cache[cache_key]

            updated.loc[mask, 'pitcher_name'] = pitcher_name
            updated.loc[mask, 'pitcher_era_30d'] = rolling.get('pitcher_era', 4.5)
            updated.loc[mask, 'pitcher_hr_per_9_30d'] = rolling.get('pitcher_hr_per_9', 1.2)
            updated.loc[mask, 'pitcher_k_per_9_30d'] = rolling.get('pitcher_k_per_9', 20.0)
            updated.loc[mask, 'pitcher_whip_30d'] = rolling.get('pitcher_whip', 1.3)
            updated.loc[mask, 'starter_override_applied'] = True

        updated = updated.drop(columns=['override_opponent_team'], errors='ignore')
        updated = self._attach_live_handedness_and_pitch_matchups(updated)
        updated = self._attach_live_bvp_history(updated, target_date)
        return updated

    def _get_pitcher_predictions_for_date(self, target_date: date) -> pd.DataFrame:
        dataset = self.pitcher_so_dataset
        if dataset is None or dataset.empty:
            return pd.DataFrame()
        day_df = dataset[dataset['game_date'] == target_date].copy()
        if not day_df.empty:
            logger.info("✅ Loaded %s starter strikeout rows for %s", len(day_df), target_date)
        return day_df

    def _attach_live_pitcher_rollups(self, df: pd.DataFrame, target_date: date) -> pd.DataFrame:
        if df.empty:
            return df

        unique_pitchers = sorted({str(name).strip() for name in df.get('pitcher_name', []) if str(name).strip()})
        if not unique_pitchers:
            return df

        stats_map = {
            pitcher_name: self._get_pitcher_rolling_stats(pitcher_name, target_date)
            for pitcher_name in unique_pitchers
        }
        updated = df.copy()
        updated['pitcher_era_30d'] = updated['pitcher_name'].map(lambda n: stats_map.get(str(n).strip(), {}).get('pitcher_era', 4.5))
        updated['pitcher_hr_per_9_30d'] = updated['pitcher_name'].map(lambda n: stats_map.get(str(n).strip(), {}).get('pitcher_hr_per_9', 1.2))
        updated['pitcher_k_per_9_30d'] = updated['pitcher_name'].map(lambda n: stats_map.get(str(n).strip(), {}).get('pitcher_k_per_9', 20.0))
        updated['pitcher_whip_30d'] = updated['pitcher_name'].map(lambda n: stats_map.get(str(n).strip(), {}).get('pitcher_whip', 1.3))
        return updated

    def _attach_live_handedness_and_pitch_matchups(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df

        player_lookup = self._load_player_lookup()
        pitch_cache = self._ensure_pitch_matchup_cache()
        updated = df.copy()

        updated['batter_canonical_live'] = updated['player_name'].apply(self._canonical_name)
        updated['pitcher_canonical_live'] = updated['pitcher_name'].apply(self._canonical_name)
        updated['team_normalized_live'] = updated['team'].apply(self._normalize_team_code)
        updated['home_team_normalized_live'] = updated['home_team'].apply(self._normalize_team_code)
        updated['away_team_normalized_live'] = updated['away_team'].apply(self._normalize_team_code)
        updated['opp_team_normalized_live'] = np.where(
            updated['team_normalized_live'] == updated['home_team_normalized_live'],
            updated['away_team_normalized_live'],
            updated['home_team_normalized_live'],
        )

        batter_bats = updated['batter_canonical_live'].map(player_lookup['bats'])
        pitcher_throws = updated['pitcher_canonical_live'].map(player_lookup['throws'])
        updated['batter_bats_right'] = (batter_bats == 'R').astype(float)
        updated['batter_bats_left'] = (batter_bats == 'L').astype(float)
        updated['batter_is_switch'] = (batter_bats == 'S').astype(float)
        updated['pitcher_throws_right'] = (pitcher_throws == 'R').astype(float)
        updated['pitcher_throws_left'] = (pitcher_throws == 'L').astype(float)
        updated['same_handed_matchup'] = np.where(
            batter_bats.isin(['L', 'R']) & pitcher_throws.isin(['L', 'R']),
            (batter_bats == pitcher_throws).astype(float),
            np.where(batter_bats == 'S', 0.0, np.nan)
        )
        updated['platoon_advantage'] = np.where(
            batter_bats.isin(['L', 'R']) & pitcher_throws.isin(['L', 'R']),
            (batter_bats != pitcher_throws).astype(float),
            np.where(batter_bats == 'S', 1.0, np.nan)
        )

        rows = []
        for idx, row in updated.iterrows():
            profile = pitch_cache['pitcher_profiles'].get((row['pitcher_canonical_live'], row['opp_team_normalized_live']))
            if profile is None:
                profile = pitch_cache['pitcher_profiles_by_name'].get(row['pitcher_canonical_live'])
            primary = profile.get('primary_pitch_type') if profile else None
            secondary = profile.get('secondary_pitch_type') if profile else None
            batter_primary = pitch_cache['batter_profiles'].get((row['batter_canonical_live'], primary), {}) if primary else {}
            batter_secondary = pitch_cache['batter_profiles'].get((row['batter_canonical_live'], secondary), {}) if secondary else {}

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

        matchup_df = pd.DataFrame(rows).set_index('row_index') if rows else pd.DataFrame()
        if not matchup_df.empty:
            updated = updated.join(matchup_df, how='left', rsuffix='_runtime')
            for col in matchup_df.columns:
                runtime_col = f"{col}_runtime"
                if runtime_col in updated.columns:
                    updated[col] = updated[runtime_col].combine_first(updated.get(col))
                    updated = updated.drop(columns=[runtime_col])

        return updated.drop(
            columns=[
                'batter_canonical_live',
                'pitcher_canonical_live',
                'team_normalized_live',
                'home_team_normalized_live',
                'away_team_normalized_live',
                'opp_team_normalized_live',
            ],
            errors='ignore'
        )

    def _attach_live_bvp_history(self, df: pd.DataFrame, target_date: date) -> pd.DataFrame:
        if df.empty or not self.db_conn:
            return df

        updated = df.copy()
        updated['batter_canonical_live'] = updated['player_name'].apply(self._canonical_name)
        pairs = updated[['batter_canonical_live', 'pitcher_name']].dropna().drop_duplicates()
        if pairs.empty:
            return updated.drop(columns=['batter_canonical_live'], errors='ignore')

        batter_names = tuple({str(name).strip() for name in updated['player_name'] if str(name).strip()})
        pitcher_names = tuple({str(name).strip() for name in updated['pitcher_name'] if str(name).strip()})
        if not batter_names or not pitcher_names:
            return updated.drop(columns=['batter_canonical_live'], errors='ignore')

        try:
            cursor = self.db_conn.cursor(cursor_factory=RealDictCursor)
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
                  AND pp.batter = ANY(%s)
                  AND pp.pitcher = ANY(%s)
                GROUP BY g.game_date, pp.batter, pp.pitcher
                ORDER BY pp.batter, pp.pitcher, g.game_date
            """, (target_date, list(batter_names), list(pitcher_names)))
            history = pd.DataFrame(cursor.fetchall())
        except Exception as exc:
            logger.warning("Could not attach live BvP history: %s", exc)
            return updated.drop(columns=['batter_canonical_live'], errors='ignore')

        if history.empty:
            zero_cols = [
                'prior_games_vs_pitcher', 'prior_pa_vs_pitcher', 'prior_hits_vs_pitcher',
                'prior_hr_vs_pitcher', 'prior_so_vs_pitcher', 'prior_hit_rate_vs_pitcher',
                'prior_hr_rate_vs_pitcher', 'prior_so_rate_vs_pitcher', 'prior_avg_pa_vs_pitcher',
                'days_since_last_vs_pitcher', 'last_hits_vs_pitcher', 'last_hr_vs_pitcher', 'last_so_vs_pitcher',
            ]
            for col in zero_cols:
                if col not in updated.columns:
                    updated[col] = np.nan
            return updated.drop(columns=['batter_canonical_live'], errors='ignore')

        history['batter_canonical_live'] = history['batter'].apply(self._canonical_name)
        history['game_date'] = pd.to_datetime(history['game_date'], errors='coerce')
        history = history.sort_values(['batter_canonical_live', 'pitcher', 'game_date']).copy()
        grouped = history.groupby(['batter_canonical_live', 'pitcher'], dropna=False)
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
        ).reset_index().rename(columns={'pitcher': 'pitcher_name'})
        prior_pa = summary['prior_pa_vs_pitcher'].replace(0, np.nan)
        prior_games = summary['prior_games_vs_pitcher'].replace(0, np.nan)
        summary['prior_hit_rate_vs_pitcher'] = summary['prior_hits_vs_pitcher'] / prior_pa
        summary['prior_hr_rate_vs_pitcher'] = summary['prior_hr_vs_pitcher'] / prior_pa
        summary['prior_so_rate_vs_pitcher'] = summary['prior_so_vs_pitcher'] / prior_pa
        summary['prior_avg_pa_vs_pitcher'] = summary['prior_pa_vs_pitcher'] / prior_games
        summary['days_since_last_vs_pitcher'] = (pd.Timestamp(target_date) - summary['last_game_date']).dt.days

        updated = updated.merge(summary, on=['batter_canonical_live', 'pitcher_name'], how='left')
        return updated.drop(columns=['batter_canonical_live'], errors='ignore')

    def _attach_live_recent_hitter_rates_by_games(
        self,
        df: pd.DataFrame,
        target_date: date,
        lookback_games: int = DEFAULT_HITTER_RECENT_LOOKBACK_GAMES,
    ) -> pd.DataFrame:
        if df.empty or not self.db_conn:
            return df

        updated = df.copy()
        player_names = sorted({str(name).strip() for name in updated['player_name'] if str(name).strip()})
        if not player_names:
            return updated
        lookback_games = max(1, int(lookback_games or DEFAULT_HITTER_RECENT_LOOKBACK_GAMES))
        recency_sql = self._live_recent_rate_select_sql()

        try:
            cursor = self.db_conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute(f"""
                WITH ranked_games AS (
                    SELECT
                        hs.player_name,
                        g.game_date,
                        hs.game_id,
                        (COALESCE(hs.at_bats, 0) + COALESCE(hs.walks, 0))::numeric AS pa_game,
                        CASE WHEN COALESCE(hs.at_bats, 0) > 0 AND COALESCE(hs.home_runs, 0) > 0 THEN 1.0 ELSE 0.0 END AS hr_game,
                        CASE WHEN COALESCE(hs.at_bats, 0) > 0 AND COALESCE(hs.hits, 0) > 0 THEN 1.0 ELSE 0.0 END AS hit_game,
                        CASE WHEN COALESCE(hs.at_bats, 0) > 0 AND COALESCE(hs.strikeouts, 0) > 0 THEN 1.0 ELSE 0.0 END AS so_game,
                        ROW_NUMBER() OVER (
                            PARTITION BY hs.player_name
                            ORDER BY g.game_date DESC, hs.game_id DESC
                        ) AS rn
                    FROM hitting_stats hs
                    JOIN games g ON hs.game_id = g.game_id
                    WHERE g.game_date < %s::date
                      AND hs.player_name = ANY(%s)
                )
                SELECT
                    player_name,
                    MAX(game_date) AS last_game_date,
                    AVG(hr_game) AS season_hr_rate_to_date,
                    AVG(hit_game) AS season_hit_rate_to_date,
                    AVG(so_game) AS season_so_rate_to_date,
                    COUNT(*) AS season_games_prior,
                    SUM(pa_game) AS season_pa_prior,
                    {recency_sql}
                FROM ranked_games
                GROUP BY player_name
            """, (target_date, player_names, lookback_games, lookback_games, lookback_games, lookback_games, lookback_games))
            rates = pd.DataFrame(cursor.fetchall())
        except Exception as exc:
            logger.warning("Could not attach live recent hitter rates: %s", exc)
            return updated

        if rates.empty:
            return updated

        rates['player_key'] = rates['player_name'].apply(_normalize_name_key)
        rate_map = rates.drop_duplicates('player_key').set_index('player_key')
        player_key = updated['player_name'].apply(_normalize_name_key)
        for col in ['recent_hr_rate_14d', 'recent_hit_rate_14d', 'recent_so_rate_14d']:
            live_values = player_key.map(rate_map[col])
            if col in updated.columns:
                updated[col] = live_values.combine_first(updated[col])
            else:
                updated[col] = live_values
        updated['recent_form_games_used'] = player_key.map(rate_map['recent_form_games_used'])
        updated['recent_form_pa_used'] = player_key.map(rate_map['recent_form_pa_used'])
        last_game_dates = pd.to_datetime(player_key.map(rate_map['last_game_date']), errors='coerce')
        updated['days_since_last_game'] = (pd.Timestamp(target_date) - last_game_dates).dt.days
        updated['season_hr_rate_to_date'] = player_key.map(rate_map['season_hr_rate_to_date'])
        updated['season_hit_rate_to_date'] = player_key.map(rate_map['season_hit_rate_to_date'])
        updated['season_so_rate_to_date'] = player_key.map(rate_map['season_so_rate_to_date'])
        updated['season_games_prior'] = player_key.map(rate_map['season_games_prior'])
        updated['season_pa_prior'] = player_key.map(rate_map['season_pa_prior'])
        for window in RECENT_FORM_WINDOWS:
            suffix = f"g{window}"
            for col in [
                f'recent_hr_rate_{suffix}',
                f'recent_hit_rate_{suffix}',
                f'recent_so_rate_{suffix}',
                f'recent_games_used_{suffix}',
                f'recent_pa_used_{suffix}',
            ]:
                updated[col] = player_key.map(rate_map[col])
        updated['shrunk_recent_hr_rate'] = self._shrunk_recent_rate(
            updated.get('recent_hr_rate_14d'),
            updated.get('recent_form_pa_used'),
            updated.get('season_hr_rate_to_date'),
        )
        updated['shrunk_recent_hit_rate'] = self._shrunk_recent_rate(
            updated.get('recent_hit_rate_14d'),
            updated.get('recent_form_pa_used'),
            updated.get('season_hit_rate_to_date'),
        )
        updated['shrunk_recent_so_rate'] = self._shrunk_recent_rate(
            updated.get('recent_so_rate_14d'),
            updated.get('recent_form_pa_used'),
            updated.get('season_so_rate_to_date'),
        )
        updated['recent_vs_season_hr_delta'] = updated['recent_hr_rate_14d'] - updated['season_hr_rate_to_date']
        updated['recent_vs_season_hit_delta'] = updated['recent_hit_rate_14d'] - updated['season_hit_rate_to_date']
        updated['recent_vs_season_so_delta'] = updated['recent_so_rate_14d'] - updated['season_so_rate_to_date']
        updated['recent_form_lookback_games'] = lookback_games
        updated['recent_form_mode'] = 'games'
        return updated

    @staticmethod
    def _live_recent_rate_select_sql() -> str:
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
            "AVG(hr_game) FILTER (WHERE rn <= %s) AS recent_hr_rate_14d",
            "AVG(hit_game) FILTER (WHERE rn <= %s) AS recent_hit_rate_14d",
            "AVG(so_game) FILTER (WHERE rn <= %s) AS recent_so_rate_14d",
            "COUNT(*) FILTER (WHERE rn <= %s) AS recent_form_games_used",
            "SUM(pa_game) FILTER (WHERE rn <= %s) AS recent_form_pa_used",
        ])
        return ",\n                    ".join(fields)

    @staticmethod
    def _shrunk_recent_rate(recent_series, recent_pa_series, season_series, shrink_pa=RECENT_FORM_SHRINKAGE_PA):
        recent = pd.to_numeric(recent_series, errors='coerce')
        recent_pa = pd.to_numeric(recent_pa_series, errors='coerce').fillna(0.0)
        season = pd.to_numeric(season_series, errors='coerce')
        denominator = recent_pa + shrink_pa
        values = ((recent * recent_pa) + (season * shrink_pa)) / denominator
        return values.where(recent.notna() & season.notna())

    def _attach_live_lineup_context(self, df: pd.DataFrame, target_date: date) -> pd.DataFrame:
        if df.empty or not self.db_conn:
            return df

        feature_cols = [
            'lineup_confirmed',
            'lineup_slot',
            'top_half_lineup',
            'middle_lineup',
            'bottom_half_lineup',
            'projected_pa',
        ]
        updated = df.copy()
        batting_order = pd.to_numeric(updated.get('batting_order'), errors='coerce')
        updated['lineup_slot'] = updated.get('lineup_slot').combine_first(batting_order) if 'lineup_slot' in updated.columns else batting_order
        updated['lineup_confirmed'] = updated.get('lineup_confirmed').combine_first(batting_order.notna().astype(float)) if 'lineup_confirmed' in updated.columns else batting_order.notna().astype(float)
        updated['top_half_lineup'] = updated.get('top_half_lineup').combine_first((batting_order <= 3).astype(float).where(batting_order.notna())) if 'top_half_lineup' in updated.columns else (batting_order <= 3).astype(float).where(batting_order.notna())
        updated['middle_lineup'] = updated.get('middle_lineup').combine_first(((batting_order >= 4) & (batting_order <= 6)).astype(float).where(batting_order.notna())) if 'middle_lineup' in updated.columns else ((batting_order >= 4) & (batting_order <= 6)).astype(float).where(batting_order.notna())
        updated['bottom_half_lineup'] = updated.get('bottom_half_lineup').combine_first((batting_order >= 7).astype(float).where(batting_order.notna())) if 'bottom_half_lineup' in updated.columns else (batting_order >= 7).astype(float).where(batting_order.notna())
        projected_pa_map = {1: 4.8, 2: 4.7, 3: 4.6, 4: 4.5, 5: 4.3, 6: 4.2, 7: 4.0, 8: 3.9, 9: 3.8}
        projected_from_order = batting_order.map(projected_pa_map)
        updated['projected_pa'] = updated.get('projected_pa').combine_first(projected_from_order) if 'projected_pa' in updated.columns else projected_from_order

        try:
            cursor = self.db_conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute("""
                SELECT
                    game_date,
                    home_team,
                    away_team,
                    home_lineup,
                    away_lineup
                FROM daily_lineups
                WHERE game_date = %s
                  AND (home_lineup IS NOT NULL OR away_lineup IS NOT NULL)
            """, (target_date,))
            lineups_df = pd.DataFrame(cursor.fetchall())
        except Exception as exc:
            logger.warning("Could not attach live lineup context: %s", exc)
            return updated

        if lineups_df.empty:
            return updated

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
                if name:
                    results.append((idx, str(name).strip()))
            return results

        pa_by_slot = {1: 4.8, 2: 4.7, 3: 4.6, 4: 4.5, 5: 4.3, 6: 4.2, 7: 4.0, 8: 3.9, 9: 3.8}
        lineup_records = []
        for _, row in lineups_df.iterrows():
            for side in ('home', 'away'):
                lineup = row.get(f'{side}_lineup')
                team = self._normalize_team_code(row.get(f'{side}_team'))
                confirmed = bool(lineup.get('confirmed')) if isinstance(lineup, dict) else False
                for slot, raw_name in _iter_batting_order(lineup):
                    canonical = self._canonical_name(raw_name)
                    lineup_records.append({
                        'team': team,
                        'player_key': _normalize_name_key(raw_name),
                        'canonical_key': _normalize_name_key(canonical),
                        'surname_key': _surname_key(raw_name),
                        'lineup_confirmed_live': float(confirmed),
                        'lineup_slot_live': float(slot),
                        'top_half_lineup_live': float(slot <= 3),
                        'middle_lineup_live': float(4 <= slot <= 6),
                        'bottom_half_lineup_live': float(slot >= 7),
                        'projected_pa_live': pa_by_slot.get(slot, 4.0),
                    })

        lineup_lookup = pd.DataFrame(lineup_records)
        if lineup_lookup.empty:
            return updated

        exact_lookup = lineup_lookup.drop(columns=['canonical_key', 'surname_key'], errors='ignore')
        exact_lookup = exact_lookup.drop_duplicates(['team', 'player_key'])
        merge_source = updated.copy()
        merge_source['team'] = merge_source['team'].apply(self._normalize_team_code)
        merge_source['player_key'] = merge_source['player_name'].apply(_normalize_name_key)
        merge_source['canonical_key'] = merge_source['player_name'].apply(self._canonical_name).apply(_normalize_name_key)
        merge_source['surname_key'] = merge_source['player_name'].apply(_surname_key)

        merged = merge_source.merge(exact_lookup, on=['team', 'player_key'], how='left')

        canonical_lookup = lineup_lookup.drop_duplicates(['team', 'canonical_key'])
        merged = merged.merge(
            canonical_lookup.drop(columns=['player_key', 'surname_key'], errors='ignore'),
            on=['team', 'canonical_key'],
            how='left',
            suffixes=('', '_canonical')
        )

        surname_counts = (
            lineup_lookup.groupby(['team', 'surname_key']).size().rename('surname_count').reset_index()
        )
        surname_unique = lineup_lookup.merge(
            surname_counts[surname_counts['surname_count'] == 1][['team', 'surname_key']],
            on=['team', 'surname_key'],
            how='inner'
        ).drop_duplicates(['team', 'surname_key'])
        merged = merged.merge(
            surname_unique.drop(columns=['player_key', 'canonical_key'], errors='ignore'),
            on=['team', 'surname_key'],
            how='left',
            suffixes=('', '_surname')
        )

        for col in feature_cols:
            live_col = f"{col}_live"
            if live_col in merged.columns:
                if col in merged.columns:
                    merged[col] = merged[col].combine_first(merged[live_col])
                else:
                    merged[col] = merged[live_col]
            canonical_col = f"{col}_live_canonical"
            if canonical_col in merged.columns:
                merged[col] = merged[col].combine_first(merged[canonical_col])
            surname_col = f"{col}_live_surname"
            if surname_col in merged.columns:
                merged[col] = merged[col].combine_first(merged[surname_col])

        matched_share = pd.to_numeric(merged.get('lineup_slot'), errors='coerce').notna().mean() * 100
        logger.info("Live lineup context matched %.1f%% of hitter rows for %s", matched_share, target_date)

        drop_cols = ['player_key', 'canonical_key', 'surname_key']
        drop_cols.extend([f"{c}_live" for c in feature_cols])
        drop_cols.extend([f"{c}_live_canonical" for c in feature_cols])
        drop_cols.extend([f"{c}_live_surname" for c in feature_cols])
        return merged.drop(columns=drop_cols, errors='ignore')

    def _attach_live_recent_context_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df

        updated = df.copy()

        def num(col, default=0.0):
            return pd.to_numeric(updated.get(col), errors='coerce').fillna(default)

        projected_pa = num('projected_pa', 4.2)
        platoon_advantage = num('platoon_advantage', 0.0)
        top_half = num('top_half_lineup', 0.0)
        middle_lineup = num('middle_lineup', 0.0)
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
        updated['recent_hr_x_middle_lineup'] = recent_hr * middle_lineup
        updated['recent_hit_x_middle_lineup'] = recent_hit * middle_lineup
        updated['hr_delta_x_projected_pa'] = hr_delta * projected_pa
        updated['hit_delta_x_projected_pa'] = hit_delta * projected_pa
        updated['projected_pa_x_lineup_confirmed'] = projected_pa * lineup_confirmed
        return updated

    def _enrich_live_hitter_rows_from_dataset(self, df: pd.DataFrame, target_date: date) -> pd.DataFrame:
        if df.empty or self.dataset is None or self.dataset.empty:
            return df

        dataset_day = self.dataset[self.dataset['game_date'] == target_date].copy()
        if dataset_day.empty:
            return df

        enrich_cols = [
            'game_id',
            'player_name',
            'team',
            'pitcher_name',
            'recent_hr_rate_14d',
            'recent_hit_rate_14d',
            'recent_so_rate_14d',
            'pitcher_era_30d',
            'pitcher_hr_per_9_30d',
            'pitcher_k_per_9_30d',
            'pitcher_whip_30d',
            'primary_pitch_type',
            'secondary_pitch_type',
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
            'prior_games_vs_pitcher',
            'prior_hit_rate_vs_pitcher',
            'prior_hr_rate_vs_pitcher',
            'lineup_slot',
            'projected_pa',
        ]
        available_cols = [col for col in enrich_cols if col in dataset_day.columns]
        dataset_day = dataset_day[available_cols].copy()

        dataset_day['player_key'] = dataset_day['player_name'].apply(_normalize_name_key)
        dataset_day['surname_key'] = dataset_day['player_name'].apply(_surname_key)
        dataset_day['team_key'] = dataset_day['team'].apply(self._normalize_team_code)
        df['player_key'] = df['player_name'].apply(_normalize_name_key)
        df['surname_key'] = df['player_name'].apply(_surname_key)
        df['team_key'] = df['team'].apply(self._normalize_team_code)

        exact_day = dataset_day.drop_duplicates(subset=['game_id', 'player_key', 'team_key'])
        merged = df.merge(
            exact_day,
            on=['game_id', 'player_key', 'team_key'],
            how='left',
            suffixes=('', '_dataset')
        )

        surname_counts = (
            dataset_day.groupby(['game_id', 'team_key', 'surname_key']).size().rename('surname_count').reset_index()
        )
        surname_unique = dataset_day.merge(
            surname_counts[surname_counts['surname_count'] == 1][['game_id', 'team_key', 'surname_key']],
            on=['game_id', 'team_key', 'surname_key'],
            how='inner'
        ).drop_duplicates(subset=['game_id', 'team_key', 'surname_key'])
        surname_cols = [col for col in surname_unique.columns if col not in {'player_key'}]
        surname_unique = surname_unique[surname_cols]
        merged = merged.merge(
            surname_unique,
            on=['game_id', 'team_key', 'surname_key'],
            how='left',
            suffixes=('', '_surname')
        )

        for col in available_cols:
            if col in {'game_id', 'player_name', 'team'}:
                continue
            dataset_col = f'{col}_dataset'
            if dataset_col in merged.columns:
                merged[col] = merged[col].where(merged[col].notna(), merged[dataset_col])
                merged = merged.drop(columns=[dataset_col])
            surname_col = f'{col}_surname'
            if surname_col in merged.columns:
                merged[col] = merged[col].where(merged[col].notna(), merged[surname_col])
                merged = merged.drop(columns=[surname_col])

        return merged.drop(columns=['player_key', 'team_key', 'surname_key'])

    def _get_players_for_date(
        self,
        target_date: date,
        recent_lookback_games: int = DEFAULT_HITTER_RECENT_LOOKBACK_GAMES,
    ) -> pd.DataFrame:
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
                CASE
                    WHEN hs.team = g.home_team THEN dl.away_pitcher_name
                    WHEN hs.team = g.away_team THEN dl.home_pitcher_name
                    ELSE NULL
                END AS pitcher_name,
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
                hs.batting_order,
                -- xstats from custom_batter_2025 (bulk join, avoids per-player queries)
                cb.xwoba,
                cb.xba,
                cb.xslg,
                cb.sweet_spot_percent,
                cb.barrel_batted_rate AS barrel_rate,
                cb.pa AS pa_count,
                hgw.wind_speed_mph,
                hgw.temp_f,
                hgw.precipitation_mm AS precip_prob,
                hgw.wind_out_factor,
                hgw.dew_point_f,
                hgw.air_carry_factor,
                hgw.wind_out_to_center_mph,
                hgw.wind_out_to_left_field_mph,
                hgw.wind_out_to_right_field_mph,
                hgw.wind_in_from_center_mph,
                hgw.crosswind_mph,
                hgw.roof_status_estimated,
                hgw.roof_status_confidence,
                CASE WHEN hgw.weather_available THEN 1.0 ELSE 0.0 END AS weather_data_available
            FROM hitting_stats hs
            JOIN games g ON hs.game_id = g.game_id
            LEFT JOIN (
                SELECT
                    game_date,
                    home_team,
                    away_team,
                    home_pitcher->>'name' AS home_pitcher_name,
                    away_pitcher->>'name' AS away_pitcher_name
                FROM daily_lineups
            ) dl
              ON dl.game_date = g.game_date
             AND dl.home_team = g.home_team
             AND dl.away_team = g.away_team
            LEFT JOIN LATERAL (
                SELECT xwoba, xba, xslg, sweet_spot_percent, barrel_batted_rate, pa
                FROM custom_batter_2025 cb2
                WHERE cb2.last_name_first_name ILIKE '%%' || split_part(hs.player_name, ' ', 2) || '%%'
                ORDER BY cb2.data_date DESC
                LIMIT 1
            ) cb ON true
            LEFT JOIN historical_game_weather hgw ON hgw.game_id = g.game_id
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
            df = self._attach_live_recent_hitter_rates_by_games(df, target_date, recent_lookback_games)
            df = self._attach_live_lineup_context(df, target_date)
            df = self._attach_live_pitcher_rollups(df, target_date)
            df = self._enrich_live_hitter_rows_from_dataset(df, target_date)
            df = self._attach_live_handedness_and_pitch_matchups(df)
            df = self._attach_live_bvp_history(df, target_date)
            df = self._attach_live_batter_weather_context(df)
            df = self._attach_live_recent_context_interactions(df)
            removed = before - len(df)
            if removed > 0:
                logger.info("Deduped %s hitter rows for %s", removed, target_date)
            logger.info(f"✅ Loaded {len(df)} players from DB for {target_date}")
            return df

        except Exception as e:
            logger.warning(f"Error querying DB for {target_date}: {e}")
            return pd.DataFrame()

    def _attach_live_batter_weather_context(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty or not self.db_conn:
            return df

        try:
            cursor = self.db_conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute("""
                SELECT COALESCE(full_name, display_name) AS player_name, bats
                FROM players
                WHERE COALESCE(full_name, display_name) IS NOT NULL
            """)
            players_df = pd.DataFrame(cursor.fetchall())
            if players_df.empty:
                return df

            players_df['name_key'] = players_df['player_name'].apply(_normalize_name_key)
            bats_lookup = (
                players_df.drop_duplicates('name_key')
                .set_index('name_key')['bats']
                .astype(str)
                .str.upper()
                .to_dict()
            )

            player_key = df['player_name'].apply(_normalize_name_key)
            bats = player_key.map(bats_lookup)
            df['batter_bats_left'] = (bats == 'L').astype(float)
            df['batter_bats_right'] = (bats == 'R').astype(float)
            df['batter_is_switch'] = (bats == 'S').astype(float)

            left_field = pd.to_numeric(df.get('wind_out_to_left_field_mph'), errors='coerce').fillna(0.0)
            right_field = pd.to_numeric(df.get('wind_out_to_right_field_mph'), errors='coerce').fillna(0.0)
            carry = pd.to_numeric(df.get('air_carry_factor'), errors='coerce').fillna(1.0)
            roof_conf = pd.to_numeric(df.get('roof_status_confidence'), errors='coerce').fillna(0.0)
            roof_closed = (df.get('roof_status_estimated', '').fillna('').astype(str).str.lower() == 'closed').astype(float)
            weather_avail = pd.to_numeric(df.get('weather_data_available'), errors='coerce').fillna(0.0)

            pull_wind = np.where(
                df['batter_bats_left'] > 0,
                right_field,
                np.where(df['batter_bats_right'] > 0, left_field, (left_field + right_field) / 2.0)
            )
            oppo_wind = np.where(
                df['batter_bats_left'] > 0,
                left_field,
                np.where(df['batter_bats_right'] > 0, right_field, (left_field + right_field) / 2.0)
            )
            switch_adjust = np.where(df['batter_is_switch'] > 0, 0.9, 1.0)
            exposure = weather_avail * (1.0 - (roof_closed * roof_conf))

            df['batter_pull_wind_mph'] = np.round(pull_wind * switch_adjust, 3)
            df['batter_oppo_wind_mph'] = np.round(oppo_wind * switch_adjust, 3)
            df['batter_pull_air_carry'] = np.round(df['batter_pull_wind_mph'] * carry, 3)
            df['batter_pull_weather_boost'] = np.round(df['batter_pull_air_carry'] * exposure, 3)
        except Exception as exc:
            logger.warning("Could not attach live batter weather context: %s", exc)

        return df

    def get_available_dates(self, target: str = 'hr') -> List[Dict]:
        """Get available dates scoped to the selected target."""
        target = (target or 'hr').lower()
        if target != 'so' and self.db_conn:
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
                    return [
                        {
                            'date': str(r['game_date']),
                            'analysis_date': str(r['game_date']),
                            'total_picks': int(r['total_picks'] or 0)
                        }
                        for r in results
                    ]
            except Exception as e:
                logger.warning(f"DB query failed for dates, falling back to dataset scope: {e}")

        if target == 'so':
            if self.pitcher_so_dataset is None or self.pitcher_so_dataset.empty:
                return []
            counts = (
                self.pitcher_so_dataset.groupby('game_date')
                .agg(total_picks=('starter_name', 'count'))
                .reset_index()
                .sort_values('game_date', ascending=False)
            )
        else:
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
            X = self._prepare_feature_frame(features, model_data)

            prob = float(self._predict_primary_proba(X, model_data)[0])

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

    def _prepare_feature_frame(self, features: Dict, model_data: Dict) -> pd.DataFrame:
        X = pd.DataFrame([features])

        for feat in model_data['features']:
            if feat not in X.columns:
                X[feat] = model_data['train_medians'].get(feat, 0)
            elif X[feat].isna().any():
                X[feat] = X[feat].fillna(model_data['train_medians'].get(feat, 0))

        X = X[model_data['features']]

        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce')
        X = X.fillna(model_data['train_medians'])
        return X

    def _predict_primary_proba(self, X: pd.DataFrame, model_data: Dict) -> np.ndarray:
        serving_mode = model_data.get('serving_mode', 'xgb_primary')
        if serving_mode == 'lgb_primary':
            return np.asarray(model_data['lgb'].predict(X), dtype=float)
        return model_data['xgb'].predict_proba(X)[:, 1]

    def _feature_display_label(self, name: str) -> str:
        overrides = {
            'xwoba': 'xwOBA',
            'avg_ev': 'Average EV',
            'barrel_rate': 'Barrel rate',
            'xba': 'xBA',
            'xslg': 'xSLG',
            'ops': 'OPS',
            'iso': 'ISO',
            'k_percent': 'K%',
            'bb_percent': 'BB%',
            'starter_k_per_9_30d': 'Starter K/9 (30d)',
            'starter_avg_so_30d': 'Starter avg K (30d)',
            'starter_k_percent_season': 'Starter K% season',
            'starter_whiff_percent_season': 'Starter whiff% season',
            'opp_team_k_rate_14d': 'Opponent K rate (14d)',
            'starter_avg_ip_30d': 'Starter avg IP (30d)',
            'starter_prior_k_per_9_vs_opp': 'Prior K/9 vs opp',
            'prior_hit_rate_vs_pitcher': 'BvP hit rate',
            'prior_hr_rate_vs_pitcher': 'BvP HR rate',
            'prior_games_vs_pitcher': 'BvP games',
            'pitcher_hr_per_9_30d': 'Pitcher HR/9 (30d)',
            'pitcher_era_30d': 'Pitcher ERA (30d)',
            'recent_hr_rate_14d': 'Recent HR rate (lookback games)',
            'recent_hit_rate_14d': 'Recent hit rate (lookback games)',
            'park_factor': 'Park factor',
            'travel_fatigue_score': 'Travel fatigue',
            'wind_out_factor': 'Wind out factor',
            'temp_f': 'Temperature',
        }
        if name in overrides:
            return overrides[name]
        words = name.replace('_', ' ').split()
        return ' '.join(word.upper() if len(word) <= 3 else word.capitalize() for word in words)

    def _feature_display_value(self, name: str, value) -> str:
        value = _safe_float(value, None)
        if value is None:
            return '—'

        if any(token in name for token in ['rate', 'percent']) and abs(value) <= 1.0:
            return f"{value * 100:.1f}%"
        if name.endswith('_pct') and abs(value) <= 1.0:
            return f"{value * 100:.1f}%"
        if name in {'xwoba', 'xba', 'xslg', 'ops', 'iso', 'park_factor'}:
            return f"{value:.3f}" if value < 10 else f"{value:.1f}"
        if name in {'avg_ev'}:
            return f"{value:.1f} mph"
        if name in {'temp_f'}:
            return f"{value:.0f} F"
        if abs(value) >= 100 or value.is_integer():
            return f"{value:.0f}"
        return f"{value:.2f}"

    def get_model_feature_explanation(self, features: Dict, target: str = 'hr', model_data_override=None, top_n: int = 3) -> Dict:
        target = target.lower() if target else 'hr'
        model_data = model_data_override or self.models.get(target)
        if model_data is None or model_data.get('xgb') is None or model_data.get('serving_mode') == 'lgb_primary':
            return {'available': False, 'summary': 'Model contribution details unavailable.'}

        try:
            X = self._prepare_feature_frame(features, model_data)
            dmatrix = xgb.DMatrix(X, feature_names=list(X.columns))
            contribs = model_data['xgb'].get_booster().predict(dmatrix, pred_contribs=True)[0]
            feature_names = list(X.columns)
            base_margin = float(contribs[-1])
            base_probability = _sigmoid(base_margin)

            rows = []
            for idx, feature_name in enumerate(feature_names):
                impact = float(contribs[idx])
                if abs(impact) < 1e-6:
                    continue
                rows.append({
                    'feature': feature_name,
                    'label': self._feature_display_label(feature_name),
                    'raw_value': self._feature_display_value(feature_name, X.iloc[0, idx]),
                    'impact': impact,
                    'direction': 'positive' if impact > 0 else 'negative',
                })

            top_positive = sorted([r for r in rows if r['impact'] > 0], key=lambda r: r['impact'], reverse=True)[:top_n]
            top_negative = sorted([r for r in rows if r['impact'] < 0], key=lambda r: r['impact'])[:top_n]

            for row in top_positive + top_negative:
                before = _sigmoid(base_margin)
                after = _sigmoid(base_margin + row['impact'])
                row['probability_delta_pct'] = round((after - before) * 100, 1)

            strongest = sorted(rows, key=lambda r: abs(r['impact']), reverse=True)[:2]
            if strongest:
                summary = 'Primary model drivers: ' + ', '.join(
                    f"{item['label']} ({'up' if item['impact'] > 0 else 'down'})" for item in strongest
                ) + '.'
            else:
                summary = 'Model contribution details were minimal for this row.'

            return {
                'available': True,
                'summary': summary,
                'base_probability_pct': round(base_probability * 100, 1),
                'top_positive': top_positive,
                'top_negative': top_negative,
            }
        except Exception as exc:
            logger.warning("Could not compute model feature explanation for %s: %s", target, exc)
            return {'available': False, 'summary': 'Model contribution details unavailable.'}

    def _build_team_hitter_outlook(self, target_date: date) -> Dict[str, List[Dict]]:
        if self.dataset is None or self.dataset.empty:
            return {}

        day_df = None
        if self.db_conn:
            day_df = self._get_players_for_date(target_date)

        if day_df is None or day_df.empty:
            if self.dataset is None or self.dataset.empty:
                return {}
            day_df = self.dataset[self.dataset['game_date'] == target_date].copy()

        if day_df.empty:
            return {}

        hr_model = self._model_data_for_target('hr')
        hit_model = self._model_data_for_target('hit')
        if hr_model is None or hit_model is None:
            return {}

        profiles: Dict[str, List[Dict]] = {}
        for _, row in day_df.iterrows():
            feature_row = dict(row)
            hr_features = {
                feat: feature_row.get(feat, hr_model['train_medians'].get(feat, 0))
                for feat in hr_model['features']
            }
            hit_features = {
                feat: feature_row.get(feat, hit_model['train_medians'].get(feat, 0))
                for feat in hit_model['features']
            }
            hr_prob = _safe_float(self.predict(hr_features, target='hr', model_data_override=hr_model).get('probability'))
            hit_prob = _safe_float(self.predict(hit_features, target='hit', model_data_override=hit_model).get('probability'))
            recent_so = _safe_float(row.get('recent_so_rate_14d'), 0.22)
            k_vs_arsenal = _safe_float(row.get('batter_k_vs_pitcher_arsenal'), 0.24)
            whiff_vs_arsenal = _safe_float(row.get('batter_whiff_vs_pitcher_arsenal'), 0.25)
            damage_score = (hit_prob * 0.55) + (hr_prob * 0.30) + (_safe_float(row.get('xwoba'), 0.320) * 0.15)
            k_risk_score = (recent_so * 0.30) + (k_vs_arsenal * 0.45) + (whiff_vs_arsenal * 0.25)

            team_code = self._normalize_team_code(row.get('team'))
            profiles.setdefault(team_code, []).append({
                'player_name': row.get('player_name', ''),
                'lineup_slot': int(_safe_float(row.get('lineup_slot'), 0)),
                'hr_probability': hr_prob,
                'hit_probability': hit_prob,
                'recent_so_rate_14d': recent_so,
                'batter_k_vs_pitcher_arsenal': k_vs_arsenal,
                'batter_whiff_vs_pitcher_arsenal': whiff_vs_arsenal,
                'damage_score': damage_score,
                'k_risk_score': k_risk_score,
            })

        for team_code, entries in profiles.items():
            profiles[team_code] = sorted(
                entries,
                key=lambda item: (
                    item.get('lineup_slot', 99) if item.get('lineup_slot', 0) > 0 else 99,
                    -item.get('hit_probability', 0.0),
                )
            )
        return profiles

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

    def build_hitter_explanation(self, row: dict, all_rows: list, prediction: dict, target: str, actual_count: int, did_occur: bool) -> Dict:
        def optional_number(value, digits=1, suffix=''):
            if not _has_value(value):
                return 'Unavailable'
            try:
                return f"{round(float(value), digits)}{suffix}"
            except Exception:
                return str(value)

        def optional_percent(value, scale=100.0, digits=1):
            if not _has_value(value):
                return 'Unavailable'
            try:
                return f"{round(float(value) * scale, digits)}%"
            except Exception:
                return 'Unavailable'

        def optional_pct_points(value, digits=1):
            if not _has_value(value):
                return 'Unavailable'
            try:
                return f"{round(float(value), digits)}%"
            except Exception:
                return 'Unavailable'

        def normalize(val, vals, invert=False):
            lo, hi = min(vals), max(vals)
            if hi == lo:
                return 0.5
            n = (val - lo) / (hi - lo)
            return 1.0 - n if invert else n

        def pool(key, default):
            return [_safe_float(r.get(key, default), default) for r in all_rows]

        xwoba = _safe_float(row.get('xwoba'), 0.320)
        barrel_rate = _safe_float(row.get('barrel_rate'), 6.0)
        avg_ev = _safe_float(row.get('avg_ev'), 88.0)
        p_hr9 = _safe_float(row.get('pitcher_hr_per_9_30d'), 1.2)
        p_era = _safe_float(row.get('pitcher_era_30d'), 4.0)
        recent_hr = _safe_float(row.get('recent_hr_rate_14d'), 0.03)
        recent_hit = _safe_float(row.get('recent_hit_rate_14d'), 0.25)
        park_factor = _safe_float(row.get('park_factor'), 1.0)
        wind_out = _safe_float(row.get('wind_out_factor'), 1.0)
        temp_f = _safe_float(row.get('temp_f'), 72.0)
        dew_point_f = _safe_float(row.get('dew_point_f'), 55.0)
        air_carry_factor = _safe_float(row.get('air_carry_factor'), 1.0)
        pull_wind = _safe_float(row.get('batter_pull_wind_mph'), 0.0)
        pull_weather_boost = _safe_float(row.get('batter_pull_weather_boost'), 0.0)
        roof_closed = _safe_float(row.get('roof_closed_estimated'), 0.0)
        weather_available = _safe_float(row.get('weather_data_available'), 0.0)
        fatigue = _safe_float(row.get('travel_fatigue_score'), 0.0)

        power = (
            normalize(xwoba, pool('xwoba', 0.320)) * 0.45 +
            normalize(barrel_rate, pool('barrel_rate', 6.0)) * 0.35 +
            normalize(avg_ev, pool('avg_ev', 88.0)) * 0.20
        )
        matchup = (
            normalize(p_hr9, pool('pitcher_hr_per_9_30d', 1.2)) * 0.6 +
            normalize(p_era, pool('pitcher_era_30d', 4.0)) * 0.4
        )
        form_key = 'recent_hr_rate_14d' if target == 'hr' else 'recent_hit_rate_14d'
        form_val = recent_hr if target == 'hr' else recent_hit
        form = normalize(form_val, pool(form_key, 0.03 if target == 'hr' else 0.25))
        env = (
            normalize(park_factor, pool('park_factor', 1.0)) * 0.5 +
            normalize(wind_out, pool('wind_out_factor', 1.0)) * 0.3 +
            normalize(max(temp_f, 40.0), [max(_safe_float(r.get('temp_f'), 72.0), 40.0) for r in all_rows]) * 0.2
        )
        freshness = normalize(fatigue, pool('travel_fatigue_score', 0.0), invert=True)

        components = [
            {'label': 'Power quality', 'score': round(power * 100, 1), 'weight': 0.30},
            {'label': 'Pitcher matchup', 'score': round(matchup * 100, 1), 'weight': 0.25},
            {'label': 'Recent form', 'score': round(form * 100, 1), 'weight': 0.20},
            {'label': 'Park and environment', 'score': round(env * 100, 1), 'weight': 0.15},
            {'label': 'Freshness', 'score': round(freshness * 100, 1), 'weight': 0.10},
        ]
        components = sorted(components, key=lambda item: item['score'], reverse=True)
        model_drivers = self.get_model_feature_explanation(row, target=target)
        matchup_cards = [
            {
                'title': 'Pitcher Matchup',
                'rows': [
                    {'label': 'Starting pitcher', 'value': row.get('pitcher_name') or 'Unknown'},
                    {'label': 'Primary pitch', 'value': row.get('primary_pitch_type') or 'Unavailable'},
                    {'label': 'Secondary pitch', 'value': row.get('secondary_pitch_type') or 'Unavailable'},
                    {'label': 'Pitcher arsenal whiff%', 'value': optional_pct_points(row.get('pitcher_arsenal_whiff_percent'), digits=1)},
                    {'label': 'Pitcher arsenal K%', 'value': optional_pct_points(row.get('pitcher_arsenal_k_percent'), digits=1)},
                    {'label': 'Batter wOBA vs arsenal', 'value': optional_number(row.get('batter_woba_vs_pitcher_arsenal'), digits=3)},
                    {'label': 'Batter K% vs arsenal', 'value': optional_pct_points(row.get('batter_k_vs_pitcher_arsenal'), digits=1)},
                    {'label': 'Batter whiff% vs arsenal', 'value': optional_pct_points(row.get('batter_whiff_vs_pitcher_arsenal'), digits=1)},
                ],
            }
        ]

        return {
            'prediction': {
                'signal': prediction.get('signal_label'),
                'score': prediction.get('score'),
                'probability_pct': round(_safe_float(prediction.get('probability')) * 100, 1),
                'summary': f"{prediction.get('signal_label', 'Rating')} driven by {components[0]['label'].lower()} and {components[1]['label'].lower()}.",
                'components': components,
                'model_drivers': model_drivers,
                'extra_cards': matchup_cards,
                'details': [
                    {'label': 'xwOBA', 'value': round(xwoba, 3)},
                    {'label': 'Barrel rate', 'value': round(barrel_rate, 1), 'suffix': '%'},
                    {'label': 'Avg EV', 'value': round(avg_ev, 1), 'suffix': ' mph'},
                    {'label': 'Pitcher HR/9 (30d)', 'value': round(p_hr9, 2)},
                    {'label': 'Pitcher ERA (30d)', 'value': round(p_era, 2)},
                    {'label': 'Starter override', 'value': 'Applied' if row.get('starter_override_applied') else 'No'},
                    {'label': f"Recent HR rate ({int(_safe_float(row.get('recent_form_lookback_games'), DEFAULT_HITTER_RECENT_LOOKBACK_GAMES))}g)", 'value': round(recent_hr * 100, 1), 'suffix': '%'},
                    {'label': f"Recent Hit rate ({int(_safe_float(row.get('recent_form_lookback_games'), DEFAULT_HITTER_RECENT_LOOKBACK_GAMES))}g)", 'value': round(recent_hit * 100, 1), 'suffix': '%'},
                    {'label': 'Shrunk recent HR rate', 'value': round(_safe_float(row.get('shrunk_recent_hr_rate'), recent_hr) * 100, 1), 'suffix': '%'},
                    {'label': 'Shrunk recent Hit rate', 'value': round(_safe_float(row.get('shrunk_recent_hit_rate'), recent_hit) * 100, 1), 'suffix': '%'},
                    {'label': 'Recent games used', 'value': int(_safe_float(row.get('recent_form_games_used'), 0))},
                    {'label': 'Recent PA used', 'value': round(_safe_float(row.get('recent_form_pa_used'), 0), 1)},
                    {'label': 'Days since last game', 'value': int(_safe_float(row.get('days_since_last_game'), 0)) if _has_value(row.get('days_since_last_game')) else 'Unavailable'},
                    {'label': 'Park factor', 'value': round(park_factor, 2)},
                    {'label': 'Weather data available', 'value': 'Yes' if weather_available >= 1.0 else 'Estimated fallback'},
                    {'label': 'Temperature', 'value': round(temp_f, 1), 'suffix': ' F'},
                    {'label': 'Dew point', 'value': round(dew_point_f, 1), 'suffix': ' F'},
                    {'label': 'Air carry factor', 'value': round(air_carry_factor, 2)},
                    {'label': 'Pull-side wind', 'value': round(pull_wind, 1), 'suffix': ' mph'},
                    {'label': 'Pull-side weather boost', 'value': round(pull_weather_boost, 2)},
                    {'label': 'Roof estimate', 'value': 'Closed' if roof_closed >= 1.0 else 'Open / outdoor'},
                    {'label': 'Travel fatigue', 'value': round(fatigue, 1)},
                    {'label': 'BvP games', 'value': int(_safe_float(row.get('prior_games_vs_pitcher'), 0))},
                    {'label': 'BvP hit rate', 'value': round(_safe_float(row.get('prior_hit_rate_vs_pitcher'), 0) * 100, 1), 'suffix': '%'},
                    {'label': 'BvP HR rate', 'value': round(_safe_float(row.get('prior_hr_rate_vs_pitcher'), 0) * 100, 1), 'suffix': '%'},
                ],
            },
            'results': {
                'available': True,
                'did_occur': bool(did_occur),
                'actual_count': int(actual_count),
                'summary': (
                    f"Outcome matched the prediction with {actual_count} recorded {target.upper()} event(s)."
                    if did_occur else
                    f"Outcome missed. Recorded {actual_count} {target.upper()} event(s)."
                ),
            }
        }

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

    def build_pitcher_explanation(self, row: dict, all_rows: list, prediction: dict, actual_count: int, did_occur: bool, so_threshold: int, opponent_lineup_outlook: Optional[List[Dict]] = None) -> Dict:
        def optional_text(value):
            if value is None or pd.isna(value) or str(value).strip() == '':
                return 'Unavailable'
            return str(value)

        def optional_percent_points(value, digits: int = 1):
            if value is None or pd.isna(value):
                return 'Unavailable'
            return f"{round(float(value), digits)}%"

        def normalize(val, vals):
            vals = [_safe_float(v) for v in vals]
            lo, hi = min(vals), max(vals)
            if hi == lo:
                return 0.5
            return (_safe_float(val) - lo) / (hi - lo)

        def pool(key, default):
            return [_safe_float(r.get(key, default), default) for r in all_rows]

        k9 = _safe_float(row.get('starter_k_per_9_30d'), 8.0)
        avg_so = _safe_float(row.get('starter_avg_so_30d'), 4.5)
        season_k = _safe_float(row.get('starter_k_percent_season'), 22.0)
        whiff = _safe_float(row.get('starter_whiff_percent_season'), 24.0)
        opp_k = _safe_float(row.get('opp_team_k_rate_14d'), 0.22)
        avg_ip = _safe_float(row.get('starter_avg_ip_30d'), 5.0)

        components = [
            {'label': 'Recent strikeout rate', 'score': round(normalize(k9, pool('starter_k_per_9_30d', 8.0)) * 100, 1), 'weight': 0.26},
            {'label': 'Recent strikeout volume', 'score': round(normalize(avg_so, pool('starter_avg_so_30d', 4.5)) * 100, 1), 'weight': 0.24},
            {'label': 'Season strikeout skill', 'score': round(normalize(season_k, pool('starter_k_percent_season', 22.0)) * 100, 1), 'weight': 0.16},
            {'label': 'Swing-and-miss profile', 'score': round(normalize(whiff, pool('starter_whiff_percent_season', 24.0)) * 100, 1), 'weight': 0.14},
            {'label': 'Opponent strikeout tendency', 'score': round(normalize(opp_k, pool('opp_team_k_rate_14d', 0.22)) * 100, 1), 'weight': 0.12},
            {'label': 'Workload', 'score': round(normalize(avg_ip, pool('starter_avg_ip_30d', 5.0)) * 100, 1), 'weight': 0.08},
        ]
        components = sorted(components, key=lambda item: item['score'], reverse=True)
        model_drivers = self.get_model_feature_explanation(
            row,
            target='so',
            model_data_override=self._model_data_for_target('so', so_threshold=so_threshold)
        )
        opponent_lineup_outlook = opponent_lineup_outlook or []
        expected_lineup_rows = []
        k_risk_rows = []
        damage_rows = []
        for entry in opponent_lineup_outlook[:9]:
            slot = entry.get('lineup_slot')
            slot_text = f"#{slot}" if slot else 'Proj'
            expected_lineup_rows.append({
                'label': entry.get('player_name', 'Unknown'),
                'value': f"{slot_text} • Hit {entry.get('hit_probability', 0.0) * 100:.1f}% • HR {entry.get('hr_probability', 0.0) * 100:.1f}%"
            })
        for entry in sorted(opponent_lineup_outlook, key=lambda item: item.get('k_risk_score', 0), reverse=True)[:3]:
            k_risk_rows.append({
                'label': entry.get('player_name', 'Unknown'),
                'value': f"K risk {entry.get('k_risk_score', 0.0) * 100:.1f} • Arsenal K {entry.get('batter_k_vs_pitcher_arsenal', 0.0) * 100:.1f}%"
            })
        for entry in sorted(opponent_lineup_outlook, key=lambda item: item.get('damage_score', 0), reverse=True)[:3]:
            damage_rows.append({
                'label': entry.get('player_name', 'Unknown'),
                'value': f"Damage {entry.get('damage_score', 0.0) * 100:.1f} • Hit {entry.get('hit_probability', 0.0) * 100:.1f}% • HR {entry.get('hr_probability', 0.0) * 100:.1f}%"
            })
        extra_cards = [
            {
                'title': 'Pitch Mix Matchup',
                'rows': [
                    {'label': 'Primary pitch', 'value': optional_text(row.get('primary_pitch_type'))},
                    {'label': 'Secondary pitch', 'value': optional_text(row.get('secondary_pitch_type'))},
                    {'label': 'Starter arsenal whiff%', 'value': optional_percent_points(row.get('starter_arsenal_whiff_percent'))},
                    {'label': 'Starter arsenal K%', 'value': optional_percent_points(row.get('starter_arsenal_k_percent'))},
                    {'label': 'Starter put-away', 'value': optional_percent_points(row.get('starter_arsenal_put_away'))},
                    {'label': 'Opponent K vs arsenal', 'value': optional_percent_points(row.get('opp_team_k_vs_starter_arsenal'))},
                    {'label': 'Opponent whiff vs arsenal', 'value': optional_percent_points(row.get('opp_team_whiff_vs_starter_arsenal'))},
                ],
            },
        ]
        if expected_lineup_rows:
            extra_cards.append({'title': 'Expected Opponent Lineup', 'rows': expected_lineup_rows})
        else:
            extra_cards.append({
                'title': 'Expected Opponent Lineup',
                'rows': [{'label': 'Coverage', 'value': 'Unavailable for this date from current hitter matchup sources.'}],
            })
        if k_risk_rows:
            extra_cards.append({'title': 'Most K-Susceptible Batters', 'rows': k_risk_rows})
        if damage_rows:
            extra_cards.append({'title': 'Batters To Respect', 'rows': damage_rows})

        return {
            'prediction': {
                'signal': prediction.get('signal_label'),
                'score': prediction.get('score'),
                'probability_pct': round(_safe_float(prediction.get('probability')) * 100, 1),
                'summary': f"{prediction.get('signal_label', 'Rating')} driven by {components[0]['label'].lower()} and {components[1]['label'].lower()}.",
                'components': components,
                'model_drivers': model_drivers,
                'extra_cards': extra_cards,
                'details': [
                    {'label': 'Target', 'value': f'{so_threshold}+ K'},
                    {'label': 'Starter K/9 (30d)', 'value': round(k9, 2)},
                    {'label': 'Starter avg K (30d)', 'value': round(avg_so, 2)},
                    {'label': 'Starter K% season', 'value': round(season_k, 1), 'suffix': '%'},
                    {'label': 'Starter whiff% season', 'value': round(whiff, 1), 'suffix': '%'},
                    {'label': 'Opponent K rate (14d)', 'value': round(opp_k * 100, 1), 'suffix': '%'},
                    {'label': 'Starter avg IP (30d)', 'value': round(avg_ip, 2)},
                    {'label': 'Prior starts vs opp', 'value': int(_safe_float(row.get('starter_prior_starts_vs_opp'), 0))},
                    {'label': 'Prior K/9 vs opp', 'value': round(_safe_float(row.get('starter_prior_k_per_9_vs_opp'), 0), 2)},
                ],
            },
            'results': {
                'available': True,
                'did_occur': bool(did_occur),
                'actual_count': int(actual_count),
                'summary': (
                    f"Starter cleared the {so_threshold}+ K target with {actual_count} strikeouts."
                    if did_occur else
                    f"Starter finished below the {so_threshold}+ K target with {actual_count} strikeouts."
                ),
            }
        }

    def generate_daily_predictions_with_results(
        self,
        target_date: date,
        target: str = 'hr',
        so_threshold: int = 3,
        starter_overrides: Optional[Dict[str, str]] = None,
        recent_lookback_games: int = DEFAULT_HITTER_RECENT_LOOKBACK_GAMES,
    ) -> Dict:
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
            day_df = self._get_players_for_date(target_date, recent_lookback_games=recent_lookback_games)

        # Fallback to CSV
        if day_df is None or day_df.empty:
            if self.dataset is None or self.dataset.empty:
                return {'date': target_date.isoformat(), 'target': target, 'predictions': [], 'stats': {}}
            day_df = self.dataset[self.dataset['game_date'] == target_date].copy()

        if day_df.empty:
            return {'date': target_date.isoformat(), 'target': target, 'predictions': [], 'stats': {}}

        day_df = self._apply_starter_overrides(day_df, target_date, starter_overrides)

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
            prediction['explanation'] = self.build_hitter_explanation(
                feature_row,
                all_rows_list,
                prediction,
                target,
                actual_count,
                did_occur
            )
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
        predictions = self._annotate_prediction_team_validation(predictions)

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
            'recent_form_mode': 'games',
            'recent_lookback_games': recent_lookback_games,
        }
        model_info = self.get_target_model_info(target)

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
            'model_info': model_info,
            'available_teams': available_teams,
            'available_matchups': available_matchups,
            'starter_options': self._get_starter_options_for_date(target_date),
            'starter_overrides': starter_overrides or {},
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
        team_hitter_outlook = self._build_team_hitter_outlook(target_date)
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
            prediction['explanation'] = self.build_pitcher_explanation(
                feature_row,
                all_rows_list,
                prediction,
                actual_count,
                did_occur,
                so_threshold,
                opponent_lineup_outlook=team_hitter_outlook.get(opponent, []),
            )
            predictions.append(prediction)

        predictions.sort(key=lambda x: x['probability'], reverse=True)
        predictions = self._annotate_prediction_team_validation(predictions)

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
            'model_info': self.get_target_model_info(target, so_threshold=so_threshold),
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
    target = request.args.get('target', 'hr')
    dates = engine.get_available_dates(target=target)
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
        starter_overrides_raw = request.args.get('starter_overrides', '')
        recent_lookback_games = max(
            1,
            int(request.args.get('recent_lookback_games', DEFAULT_HITTER_RECENT_LOOKBACK_GAMES) or DEFAULT_HITTER_RECENT_LOOKBACK_GAMES),
        )
        starter_overrides = {}
        if starter_overrides_raw:
            try:
                starter_overrides = json.loads(starter_overrides_raw)
            except json.JSONDecodeError:
                starter_overrides = {}
        data = engine.generate_daily_predictions_with_results(
            target_date,
            target=target,
            so_threshold=so_threshold,
            starter_overrides=starter_overrides,
            recent_lookback_games=recent_lookback_games,
        )

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
    target_summaries = {
        'hr': engine.get_target_model_info('hr'),
        'hit': engine.get_target_model_info('hit'),
        'so': engine.get_target_model_info('so', so_threshold=3),
    }
    return jsonify({
        'model_type': 'XGBoost + LightGBM + LogisticRegression artifacts',
        'serving_strategy': 'xgb_primary',
        'models': models_status,
        'primary_model_status': 'ready' if model_ready else 'unavailable',
        'model_status': 'ready' if model_ready else 'unavailable',
        'artifacts_ready': any_artifacts_ready,
        'targets': target_summaries,
        'training_samples': len(engine.dataset) if engine.dataset is not None else 0,
        'data_range': {
            'hitter': {
                'start': min(engine.dataset['game_date']).isoformat() if engine.dataset is not None and not engine.dataset.empty else None,
                'end': max(engine.dataset['game_date']).isoformat() if engine.dataset is not None and not engine.dataset.empty else None,
            },
            'starter_so': {
                'start': min(engine.pitcher_so_dataset['game_date']).isoformat() if engine.pitcher_so_dataset is not None and not engine.pitcher_so_dataset.empty else None,
                'end': max(engine.pitcher_so_dataset['game_date']).isoformat() if engine.pitcher_so_dataset is not None and not engine.pitcher_so_dataset.empty else None,
            }
        },
        'available_dates': len(engine.get_available_dates())
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
    """Historical header stats across the full dataset for each target."""
    so_threshold = engine._so_threshold(request.args.get('so_threshold'))
    classification = (request.args.get('classification') or '').strip().upper()
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
        df['_actual'] = df[label_col].astype(bool)

        if classification:
            if 'game_date' in df.columns:
                signals = pd.Series(index=df.index, dtype=object)
                cutoffs = [
                    (0.10, 'STRONG_BUY'),
                    (0.25, 'BUY'),
                    (0.50, 'MODERATE'),
                    (0.75, 'AVOID'),
                    (1.00, 'STRONG_SELL'),
                ]
                for _, grp in df.groupby('game_date'):
                    sorted_idx = grp['_proba'].sort_values(ascending=False).index.tolist()
                    n = len(sorted_idx)
                    for rank, idx in enumerate(sorted_idx):
                        rank_pct = rank / n if n else 1.0
                        label = 'STRONG_SELL'
                        for threshold, candidate in cutoffs:
                            if rank_pct < threshold:
                                label = candidate
                                break
                        signals.at[idx] = label
                df['_signal_label'] = signals
            else:
                df['_signal_label'] = 'ALL'

            df = df[df['_signal_label'] == classification].copy()

        rows = int(len(df))
        hits = int(df['_actual'].sum()) if rows > 0 else 0
        hit_rate = round(hits / rows * 100, 1) if rows > 0 else 0

        result[target] = {
            'picks': rows,
            'hit_rate': hit_rate,
            'scope': 'classification' if classification else 'full_dataset',
            'dataset_rows': rows,
            'classification': classification or 'ALL_TYPES',
            'model_info': engine.get_target_model_info(target, so_threshold=target_so_threshold, classification=classification),
        }

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
