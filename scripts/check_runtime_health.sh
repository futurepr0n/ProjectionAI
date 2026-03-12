#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

URL="${1:-http://127.0.0.1:5002/api/model/stats}"

echo "Checking runtime health: $URL"
BODY="$(curl -fsS "$URL")"
echo "$BODY"

if ! grep -q '"serving_mode"' <<<"$BODY"; then
  echo "Health check failed: serving mode not present" >&2
  exit 1
fi

echo "Runtime health check passed"
