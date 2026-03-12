#!/usr/bin/env python3
from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from train_models_v4 import ARTIFACTS_DIR, ModelPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_PATH = Path(__file__).parent.parent / 'data' / 'pitcher_strikeout_dataset.csv'
RESULTS_PATH = ARTIFACTS_DIR / 'pitcher_strikeout_training_results.json'

TARGETS = {
    'pitcher_so_3_plus': 'label_so_3_plus',
    'pitcher_so_4_plus': 'label_so_4_plus',
    'pitcher_so_5_plus': 'label_so_5_plus',
    'pitcher_so_6_plus': 'label_so_6_plus',
}

COMMON_FEATURES = [
    'is_home',
    'lineup_confirmed',
    'opponent_lineup_size',
    'starter_starts_30d',
    'starter_avg_ip_30d',
    'starter_avg_so_30d',
    'starter_k_per_9_30d',
    'starter_era_30d',
    'starter_whip_30d',
    'starter_avg_pitches_30d',
    'starter_max_pitches_30d',
    'starter_avg_batters_faced_30d',
    'starter_k_percent_season',
    'starter_bb_percent_season',
    'starter_whiff_percent_season',
    'opp_team_games_14d',
    'opp_team_pa_per_game_14d',
    'opp_team_k_rate_14d',
    'opp_team_hits_per_pa_14d',
]

THRESHOLD_FEATURES = {
    'pitcher_so_3_plus': COMMON_FEATURES + [
        'starter_prior_starts_vs_opp',
        'starter_prior_avg_pitches_vs_opp',
        'starter_prior_avg_batters_faced_vs_opp',
        'starter_days_since_last_vs_opp',
        'starter_last_so_vs_opp',
    ],
    'pitcher_so_4_plus': COMMON_FEATURES + [
        'starter_prior_starts_vs_opp',
        'starter_prior_ip_vs_opp',
        'starter_prior_so_vs_opp',
        'starter_prior_k_per_9_vs_opp',
        'starter_days_since_last_vs_opp',
        'starter_last_so_vs_opp',
        'starter_primary_pitch_usage',
        'starter_primary_pitch_whiff_percent',
        'starter_primary_pitch_k_percent',
        'starter_secondary_pitch_usage',
        'starter_secondary_pitch_whiff_percent',
        'starter_secondary_pitch_k_percent',
        'opp_team_k_vs_primary_pitch',
        'opp_team_whiff_vs_primary_pitch',
        'opp_team_k_vs_secondary_pitch',
        'opp_team_whiff_vs_secondary_pitch',
    ],
    'pitcher_so_5_plus': COMMON_FEATURES + [
        'starter_prior_starts_vs_opp',
        'starter_prior_ip_vs_opp',
        'starter_prior_so_vs_opp',
        'starter_prior_k_per_9_vs_opp',
        'starter_prior_avg_pitches_vs_opp',
        'starter_prior_avg_batters_faced_vs_opp',
        'starter_days_since_last_vs_opp',
        'starter_last_so_vs_opp',
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
    ],
    'pitcher_so_6_plus': COMMON_FEATURES + [
        'starter_prior_starts_vs_opp',
        'starter_prior_ip_vs_opp',
        'starter_prior_so_vs_opp',
        'starter_prior_k_per_9_vs_opp',
        'starter_prior_avg_pitches_vs_opp',
        'starter_prior_avg_batters_faced_vs_opp',
        'starter_last_so_vs_opp',
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
    ],
}


def main() -> int:
    if not DATA_PATH.exists():
        logger.error("Pitcher strikeout dataset not found at %s", DATA_PATH)
        return 1

    df = pd.read_csv(DATA_PATH)
    df['game_date'] = pd.to_datetime(df['game_date'], errors='coerce')

    pipeline = ModelPipeline()
    results = {'generated_at': pd.Timestamp.now(tz='UTC').isoformat()}

    for artifact_name, label_col in TARGETS.items():
        if label_col not in df.columns:
            logger.warning("Missing %s, skipping %s", label_col, artifact_name)
            continue
        if df[label_col].sum() < 25:
            logger.warning("Not enough positives in %s, skipping %s", label_col, artifact_name)
            continue

        logger.info("Training %s using %s", artifact_name, label_col)
        feature_cols = [col for col in THRESHOLD_FEATURES[artifact_name] if col in df.columns]
        logger.info("%s feature count: %s", artifact_name, len(feature_cols))
        results[artifact_name] = pipeline._train_pipeline(df, label_col, artifact_name, feature_cols=feature_cols)
        logger.info(
            "%s holdout META AUC=%.4f",
            artifact_name,
            results[artifact_name]['auc'],
        )

    with RESULTS_PATH.open('w') as handle:
        json.dump(results, handle, indent=2, default=str)

    logger.info("Saved pitcher strikeout training summary to %s", RESULTS_PATH)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
