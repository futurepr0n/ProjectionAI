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
        results[artifact_name] = pipeline._train_pipeline(df, label_col, artifact_name)
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
