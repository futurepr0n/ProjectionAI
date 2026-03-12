#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models import train_models_v4 as tm


OUTPUT_PATH = Path("output/hitter_recency_pruning.json")

RAW_WINDOW_FEATURES = {
    "recent_hr_rate_g3",
    "recent_hit_rate_g3",
    "recent_so_rate_g3",
    "recent_games_used_g3",
    "recent_pa_used_g3",
    "recent_hr_rate_g5",
    "recent_hit_rate_g5",
    "recent_so_rate_g5",
    "recent_games_used_g5",
    "recent_pa_used_g5",
    "recent_hr_rate_g10",
    "recent_hit_rate_g10",
    "recent_so_rate_g10",
    "recent_games_used_g10",
    "recent_pa_used_g10",
    "recent_hr_rate_g20",
    "recent_hit_rate_g20",
    "recent_so_rate_g20",
    "recent_games_used_g20",
    "recent_pa_used_g20",
}

RAW_SELECTED_FEATURES = {
    "recent_hr_rate_14d",
    "recent_hit_rate_14d",
    "recent_so_rate_14d",
    "recent_form_games_used",
    "recent_form_pa_used",
    "days_since_last_game",
}

SHRINK_FEATURES = {
    "season_hr_rate_to_date",
    "season_hit_rate_to_date",
    "season_so_rate_to_date",
    "season_games_prior",
    "season_pa_prior",
    "shrunk_recent_hr_rate",
    "shrunk_recent_hit_rate",
    "shrunk_recent_so_rate",
    "recent_vs_season_hr_delta",
    "recent_vs_season_hit_delta",
    "recent_vs_season_so_delta",
}

INTERACTION_FEATURES = {
    "recent_hr_x_projected_pa",
    "recent_hit_x_projected_pa",
    "shrunk_recent_hr_x_projected_pa",
    "shrunk_recent_hit_x_projected_pa",
    "recent_hr_x_platoon_advantage",
    "recent_hit_x_platoon_advantage",
    "shrunk_recent_hr_x_platoon_advantage",
    "shrunk_recent_hit_x_platoon_advantage",
    "recent_hr_x_top_half_lineup",
    "recent_hit_x_top_half_lineup",
    "recent_hr_x_middle_lineup",
    "recent_hit_x_middle_lineup",
    "hr_delta_x_projected_pa",
    "hit_delta_x_projected_pa",
    "projected_pa_x_lineup_confirmed",
}

SCENARIOS = {
    "baseline_interactions": {"hr": set(), "hit": set()},
    "drop_all_raw_windows": {"hr": RAW_WINDOW_FEATURES, "hit": RAW_WINDOW_FEATURES},
    "drop_raw_selected": {"hr": RAW_SELECTED_FEATURES, "hit": RAW_SELECTED_FEATURES},
    "keep_shrunk_and_interactions_only": {
        "hr": RAW_WINDOW_FEATURES | RAW_SELECTED_FEATURES,
        "hit": RAW_WINDOW_FEATURES | RAW_SELECTED_FEATURES,
    },
    "drop_interactions_keep_shrunk": {"hr": INTERACTION_FEATURES, "hit": INTERACTION_FEATURES},
    "drop_shrunk_keep_interactions": {"hr": SHRINK_FEATURES, "hit": SHRINK_FEATURES},
}


def run() -> dict:
    df = pd.read_csv("data/complete_dataset.csv", low_memory=False)
    df["game_date"] = pd.to_datetime(df["game_date"])

    original_exclusions = {k: set(v) for k, v in tm.TARGET_EXCLUDED_FEATURES.items()}
    results = {"scenarios": []}

    for name, extra in SCENARIOS.items():
        tm.TARGET_EXCLUDED_FEATURES.clear()
        for target in ("hr", "hit"):
            tm.TARGET_EXCLUDED_FEATURES[target] = set(original_exclusions.get(target, set())) | set(extra.get(target, set()))

        pipe = tm.ModelPipeline()
        hr = pipe.train_hr_model(df, save_artifacts=False)
        hit = pipe.train_hit_model(df, save_artifacts=False)
        hr_holdout = hr.get("holdout", {})
        hit_holdout = hit.get("holdout", {})
        results["scenarios"].append(
            {
                "scenario": name,
                "hr_auc": hr_holdout.get("meta", {}).get("roc_auc"),
                "hit_auc": hit_holdout.get("meta", {}).get("roc_auc"),
                "hr_ap": hr_holdout.get("meta", {}).get("average_precision"),
                "hit_ap": hit_holdout.get("meta", {}).get("average_precision"),
                "hr_features": len(hr["features"]),
                "hit_features": len(hit["features"]),
            }
        )

    tm.TARGET_EXCLUDED_FEATURES.clear()
    tm.TARGET_EXCLUDED_FEATURES.update(original_exclusions)
    return results


def main() -> None:
    results = run()
    OUTPUT_PATH.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
