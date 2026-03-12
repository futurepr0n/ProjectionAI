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


OUTPUT_PATH = Path("output/hitter_feature_ablation.json")

AblationGroups = {
    "bvp": {
        "prior_games_vs_pitcher",
        "prior_pa_vs_pitcher",
        "prior_hits_vs_pitcher",
        "prior_hr_vs_pitcher",
        "prior_so_vs_pitcher",
        "last_hits_vs_pitcher",
        "last_hr_vs_pitcher",
        "last_so_vs_pitcher",
        "days_since_last_vs_pitcher",
        "prior_hit_rate_vs_pitcher",
        "prior_hr_rate_vs_pitcher",
        "prior_so_rate_vs_pitcher",
        "prior_avg_pa_vs_pitcher",
    },
    "travel": {
        "travel_distance_miles",
        "timezone_changes",
        "travel_fatigue_score",
    },
    "pitch_matchup": {
        "pitcher_primary_pitch_usage",
        "pitcher_primary_pitch_whiff_percent",
        "pitcher_primary_pitch_k_percent",
        "pitcher_secondary_pitch_usage",
        "pitcher_secondary_pitch_whiff_percent",
        "pitcher_secondary_pitch_k_percent",
        "pitcher_arsenal_whiff_percent",
        "pitcher_arsenal_k_percent",
        "batter_k_vs_primary_pitch",
        "batter_whiff_vs_primary_pitch",
        "batter_ba_vs_primary_pitch",
        "batter_slg_vs_primary_pitch",
        "batter_woba_vs_primary_pitch",
        "batter_k_vs_secondary_pitch",
        "batter_whiff_vs_secondary_pitch",
        "batter_ba_vs_secondary_pitch",
        "batter_slg_vs_secondary_pitch",
        "batter_woba_vs_secondary_pitch",
        "batter_k_vs_pitcher_arsenal",
        "batter_whiff_vs_pitcher_arsenal",
        "batter_woba_vs_pitcher_arsenal",
    },
    "weather_simple": {
        "wind_speed_mph",
        "temp_f",
        "wind_out_factor",
        "roof_closed_estimated",
        "weather_data_available",
    },
    "recency_raw_windows": {
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
    },
}

SCENARIOS = {
    "baseline_current_gating": {"hr": set(), "hit": set()},
    "drop_bvp": {"hr": AblationGroups["bvp"], "hit": AblationGroups["bvp"]},
    "drop_travel": {"hr": AblationGroups["travel"], "hit": AblationGroups["travel"]},
    "drop_pitch_matchup": {"hr": AblationGroups["pitch_matchup"], "hit": AblationGroups["pitch_matchup"]},
    "drop_raw_recency_windows": {"hr": AblationGroups["recency_raw_windows"], "hit": AblationGroups["recency_raw_windows"]},
    "drop_bvp_and_travel": {"hr": AblationGroups["bvp"] | AblationGroups["travel"], "hit": AblationGroups["bvp"] | AblationGroups["travel"]},
    "drop_hit_weather_simple": {"hr": set(), "hit": AblationGroups["weather_simple"]},
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
