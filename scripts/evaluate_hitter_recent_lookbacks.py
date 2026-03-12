#!/usr/bin/env python3
"""
Evaluate hitter recent-form lookback windows by rebuilding the hitter dataset
and training HR/Hit models without overwriting active artifacts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
DATA_DIR = BASE_DIR / "data"
if str(DATA_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_DIR))

from data.build_training_dataset import DatasetBuilder
from models.train_models_v4 import ModelPipeline


OUTPUT_PATH = BASE_DIR / "output" / "hitter_recent_lookback_sweep.json"


def parse_args():
    parser = argparse.ArgumentParser(description="Sweep hitter recent-form lookback windows.")
    parser.add_argument(
        "--lookbacks",
        nargs="+",
        type=int,
        default=[5, 10, 15, 20],
        help="Lookback windows in games to evaluate.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    pipeline = ModelPipeline()
    results = {"lookbacks": []}

    for lookback in args.lookbacks:
        builder = DatasetBuilder(recent_lookback_games=lookback)
        try:
            df = builder.build()
        finally:
            builder.close()

        if df.empty:
            results["lookbacks"].append({"lookback_games": lookback, "error": "empty_dataset"})
            continue

        df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
        hr = pipeline.train_hr_model(df, save_artifacts=False)
        hit = pipeline.train_hit_model(df, save_artifacts=False)

        results["lookbacks"].append(
            {
                "lookback_games": lookback,
                "rows": int(len(df)),
                "hr_auc": hr["auc"],
                "hit_auc": hit["auc"],
                "hr_holdout": hr["holdout_metrics"],
                "hit_holdout": hit["holdout_metrics"],
            }
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(results, indent=2, default=str))
    print(OUTPUT_PATH)
    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
