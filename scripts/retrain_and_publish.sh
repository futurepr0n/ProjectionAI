#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

LOOKBACK_GAMES=20

while [[ $# -gt 0 ]]; do
  case "$1" in
    --recent-lookback-games)
      LOOKBACK_GAMES="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

echo "Rebuilding hitter dataset with lookback=$LOOKBACK_GAMES"
venv/bin/python data/build_training_dataset.py --recent-lookback-games "$LOOKBACK_GAMES"

echo "Rebuilding starter strikeout dataset"
venv/bin/python data/build_pitcher_strikeout_dataset.py

echo "Training hitter models"
venv/bin/python models/train_models_v4.py

echo "Training starter strikeout models"
venv/bin/python models/train_pitcher_strikeout_models.py

echo "Syncing derived feature snapshots to DB"
venv/bin/python data/sync_feature_snapshots_to_db.py

echo "Retrain and publish workflow complete"
echo "Next: review metrics and update models/artifacts/serving_manifest.json if promotion is approved."
