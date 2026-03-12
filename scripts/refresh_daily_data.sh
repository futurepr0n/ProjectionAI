#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DATE_ARG=""
GENERATE_EXPORT=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --date)
      DATE_ARG="$2"
      shift 2
      ;;
    --skip-export)
      GENERATE_EXPORT=0
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if [[ -z "$DATE_ARG" ]]; then
  DATE_ARG="$(date +%F)"
fi

echo "Refreshing daily data for $DATE_ARG"
venv/bin/python data/fetch_daily_lineups.py --date "$DATE_ARG"
venv/bin/python data/backfill_historical_weather.py --start-date "$DATE_ARG" --end-date "$DATE_ARG"
venv/bin/python data/sync_feature_snapshots_to_db.py

if [[ "$GENERATE_EXPORT" -eq 1 ]]; then
  venv/bin/python scripts/generate_daily_predictions.py "$DATE_ARG"
fi

echo "Daily refresh complete for $DATE_ARG"
