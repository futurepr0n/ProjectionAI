#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))

import dashboards.app as dashboard_app

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).parent.parent / 'output'
OUTPUT_DIR.mkdir(exist_ok=True)

TARGET_METADATA = {
    'hr': {
        'display_name': 'Home Run',
        'prediction_scope': 'hitter_vs_pitcher',
        'notes': [],
    },
    'hit': {
        'display_name': 'Any Hit',
        'prediction_scope': 'hitter_vs_pitcher',
        'notes': [],
    },
    'so': {
        'display_name': 'Starter Strikeouts',
        'prediction_scope': 'starting_pitcher_vs_opponent_team',
        'notes': [
            'SO predictions are one row per projected starter against the opposing lineup context.',
            'The strikeout threshold is configurable at export time.',
        ],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Generate batch prediction exports using the same engine as the dashboard.'
    )
    parser.add_argument(
        'target_date',
        nargs='?',
        default=date.today().isoformat(),
        help='Date to score in YYYY-MM-DD format. Defaults to today.'
    )
    parser.add_argument(
        '--targets',
        default='hr,hit,so',
        help='Comma-separated targets to score. Defaults to hr,hit,so.'
    )
    parser.add_argument(
        '--output-path',
        help='Optional explicit output path. If omitted, writes output/predictions_<date>.json.'
    )
    parser.add_argument(
        '--pretty',
        action='store_true',
        help='Pretty-print JSON output.'
    )
    parser.add_argument(
        '--so-threshold',
        type=int,
        default=3,
        help='Starter strikeout threshold for the SO target. Supported values: 3, 4, 5, 6.'
    )
    return parser.parse_args()


def normalize_targets(raw_targets: str) -> List[str]:
    targets = []
    for target in raw_targets.split(','):
        cleaned = target.strip().lower()
        if not cleaned:
            continue
        if cleaned not in TARGET_METADATA:
            raise ValueError(f"Unsupported target: {cleaned}")
        if cleaned not in targets:
            targets.append(cleaned)
    if not targets:
        raise ValueError('At least one valid target is required')
    return targets


def collect_model_versions(engine: dashboard_app.PredictionEngine) -> Dict[str, Dict]:
    versions = {}
    for target, model_data in engine.models.items():
        if not model_data:
            versions[target] = {'loaded': False}
            continue

        meta = getattr(model_data.get('meta'), '__class__', None)
        versions[target] = {
            'loaded': True,
            'feature_count': len(model_data.get('features', [])),
            'meta_model_class': meta.__name__ if meta else None,
        }
    return versions


def generate_predictions_for_date(target_date: date, targets: List[str], so_threshold: int = 3) -> Dict:
    logger.info("Generating predictions for %s across targets=%s", target_date, targets)

    engine = dashboard_app.engine
    result = {
        'generated_at': datetime.now(UTC).isoformat(),
        'date': target_date.isoformat(),
        'targets': {},
        'model_versions': collect_model_versions(engine),
    }

    for target in targets:
        payload = engine.generate_daily_predictions_with_results(
            target_date,
            target=target,
            so_threshold=so_threshold,
        )
        payload['metadata'] = TARGET_METADATA[target]
        if target == 'so':
            payload['metadata'] = {
                **TARGET_METADATA[target],
                'display_name': f'Starter Strikeouts {so_threshold}+',
                'threshold': so_threshold,
            }
        payload['stats']['available_team_count'] = len(payload.get('available_teams', []))
        payload['stats']['available_matchup_count'] = len(payload.get('available_matchups', []))
        result['targets'][target] = payload
        logger.info(
            "Scored %s: %s predictions across %s teams",
            target,
            len(payload.get('predictions', [])),
            len(payload.get('available_teams', [])),
        )

    return dashboard_app._sanitize_for_json(result)


def main() -> int:
    args = parse_args()
    target_date = datetime.strptime(args.target_date, '%Y-%m-%d').date()
    targets = normalize_targets(args.targets)
    so_threshold = args.so_threshold if args.so_threshold in {3, 4, 5, 6} else 3

    output_path = Path(args.output_path) if args.output_path else OUTPUT_DIR / f'predictions_{target_date}.json'
    payload = generate_predictions_for_date(target_date, targets, so_threshold=so_threshold)

    with output_path.open('w') as handle:
        json.dump(payload, handle, indent=2 if args.pretty else None)

    logger.info("Saved predictions to %s", output_path)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
