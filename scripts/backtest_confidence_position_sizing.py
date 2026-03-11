#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.remote_data_loader import RemoteDataLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).parent.parent / 'output'
OUTPUT_DIR.mkdir(exist_ok=True)


@dataclass
class StrategyResult:
    name: str
    bets: int
    stake: float
    profit: float
    roi_pct: float
    hit_rate_pct: float
    avg_stake: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Backtest flat vs confidence-weighted HR position sizing on historical hellraiser picks.'
    )
    parser.add_argument(
        '--start-date',
        help='Optional start date filter in YYYY-MM-DD.'
    )
    parser.add_argument(
        '--end-date',
        help='Optional end date filter in YYYY-MM-DD.'
    )
    parser.add_argument(
        '--min-confidence',
        type=float,
        default=None,
        help='Optional minimum confidence_score filter.'
    )
    parser.add_argument(
        '--output-path',
        help='Optional explicit JSON output path.'
    )
    return parser.parse_args()


def load_historical_hr_picks() -> pd.DataFrame:
    loader = RemoteDataLoader()
    try:
        df = loader.create_labeled_dataset()
    finally:
        loader.close()

    if df.empty:
        return df

    df = df.copy()
    df = df[df['odds_decimal'].notna()].copy()
    df['analysis_date'] = pd.to_datetime(df['analysis_date']).dt.date
    df['confidence_score'] = pd.to_numeric(df['confidence_score'], errors='coerce')
    df['odds_decimal'] = pd.to_numeric(df['odds_decimal'], errors='coerce')
    df['label'] = pd.to_numeric(df['label'], errors='coerce').fillna(0).astype(int)
    df = df[df['odds_decimal'] > 1].copy()
    df = df[df['confidence_score'].notna()].copy()
    return df


def apply_filters(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    filtered = df.copy()
    if args.start_date:
        filtered = filtered[filtered['analysis_date'] >= pd.to_datetime(args.start_date).date()]
    if args.end_date:
        filtered = filtered[filtered['analysis_date'] <= pd.to_datetime(args.end_date).date()]
    if args.min_confidence is not None:
        filtered = filtered[filtered['confidence_score'] >= args.min_confidence]
    return filtered


def _settle(df: pd.DataFrame, stake_col: str, result_col: str = 'label') -> StrategyResult:
    stake = df[stake_col].sum()
    profit = np.where(
        df[result_col].astype(int) == 1,
        df[stake_col] * (df['odds_decimal'] - 1.0),
        -df[stake_col]
    ).sum()
    bets = len(df)
    hits = int(df[result_col].sum())
    return StrategyResult(
        name=stake_col,
        bets=bets,
        stake=float(stake),
        profit=float(profit),
        roi_pct=float((profit / stake) * 100) if stake > 0 else 0.0,
        hit_rate_pct=float((hits / bets) * 100) if bets > 0 else 0.0,
        avg_stake=float(stake / bets) if bets > 0 else 0.0,
    )


def build_stakes(df: pd.DataFrame) -> pd.DataFrame:
    scored = df.copy()
    confidence = scored['confidence_score'].astype(float)
    scored['implied_probability'] = 1.0 / scored['odds_decimal'].astype(float)
    scored['confidence_probability'] = confidence / 100.0
    scored['confidence_edge'] = scored['confidence_probability'] - scored['implied_probability']

    # Keep weighted strategies comparable to flat staking by normalizing mean stake to 1.0.
    linear = 0.5 + 1.5 * ((confidence - confidence.min()) / max(confidence.max() - confidence.min(), 1e-9))
    linear = linear / linear.mean()

    percentile = confidence.rank(pct=True, method='average')
    tiered = np.select(
        [percentile >= 0.9, percentile >= 0.75, percentile >= 0.5],
        [1.75, 1.25, 1.0],
        default=0.5
    )
    tiered = tiered / tiered.mean()

    scored['flat_unit'] = 1.0
    scored['confidence_linear'] = linear
    scored['confidence_tiered'] = tiered
    return scored


def summarize_by_date(df: pd.DataFrame, stake_col: str) -> Dict:
    rows: List[Dict] = []
    for analysis_date, group in df.groupby('analysis_date'):
        settled = _settle(group, stake_col)
        rows.append({
            'date': analysis_date.isoformat(),
            'bets': settled.bets,
            'stake': round(settled.stake, 4),
            'profit': round(settled.profit, 4),
            'roi_pct': round(settled.roi_pct, 2),
            'hit_rate_pct': round(settled.hit_rate_pct, 2),
        })
    return {
        'days': rows,
        'profitable_days': sum(1 for row in rows if row['profit'] > 0),
        'total_days': len(rows),
    }


def _top_n_per_day(df: pd.DataFrame, n: int) -> pd.DataFrame:
    return (
        df.sort_values(['analysis_date', 'confidence_score'], ascending=[True, False])
        .groupby('analysis_date', group_keys=False)
        .head(n)
        .copy()
    )


def build_scenarios(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    scenarios = {
        'all_picks': df.copy(),
        'confidence_70_plus': df[df['confidence_score'] >= 70].copy(),
        'confidence_80_plus': df[df['confidence_score'] >= 80].copy(),
        'edge_5_plus': df[df['confidence_edge'] >= 0.05].copy(),
        'edge_10_plus': df[df['confidence_edge'] >= 0.10].copy(),
        'top_5_per_day': _top_n_per_day(df, 5),
        'top_10_per_day': _top_n_per_day(df, 10),
    }

    conf70 = scenarios['confidence_70_plus']
    if not conf70.empty:
        scenarios['top_5_per_day_conf70'] = _top_n_per_day(conf70, 5)
        scenarios['top_10_per_day_conf70'] = _top_n_per_day(conf70, 10)

    edge5 = scenarios['edge_5_plus']
    if not edge5.empty:
        scenarios['top_5_per_day_edge5'] = _top_n_per_day(edge5, 5)
        scenarios['top_10_per_day_edge5'] = _top_n_per_day(edge5, 10)

    return {name: frame for name, frame in scenarios.items() if not frame.empty}


def main() -> int:
    args = parse_args()
    df = load_historical_hr_picks()
    if df.empty:
        logger.error('No historical HR picks with odds/confidence available.')
        return 1

    df = apply_filters(df, args)
    if df.empty:
        logger.error('No rows remain after filters.')
        return 1

    strategy_names = ['flat_unit', 'confidence_linear', 'confidence_tiered']
    scenario_payload = []
    for scenario_name, scenario_df in build_scenarios(df).items():
        scored = build_stakes(scenario_df)
        results = [_settle(scored, strategy) for strategy in strategy_names]
        scenario_payload.append({
            'name': scenario_name,
            'sample': {
                'bets': int(len(scored)),
                'date_min': scored['analysis_date'].min().isoformat(),
                'date_max': scored['analysis_date'].max().isoformat(),
                'avg_confidence': round(float(scored['confidence_score'].mean()), 4),
                'avg_odds_decimal': round(float(scored['odds_decimal'].mean()), 4),
                'hr_hit_rate_pct': round(float(scored['label'].mean() * 100), 2),
            },
            'strategies': [
                {
                    'name': result.name,
                    'bets': result.bets,
                    'stake': round(result.stake, 4),
                    'profit': round(result.profit, 4),
                    'roi_pct': round(result.roi_pct, 2),
                    'hit_rate_pct': round(result.hit_rate_pct, 2),
                    'avg_stake': round(result.avg_stake, 4),
                }
                for result in results
            ],
            'daily_breakdown': {
                strategy: summarize_by_date(scored, strategy)
                for strategy in strategy_names
            }
        })

    baseline = next((scenario for scenario in scenario_payload if scenario['name'] == 'all_picks'), None)
    if baseline is None:
        logger.error('No baseline scenario generated.')
        return 1

    payload = {
        'filters': {
            'start_date': args.start_date,
            'end_date': args.end_date,
            'min_confidence': args.min_confidence,
        },
        'sample': baseline['sample'],
        'strategies': baseline['strategies'],
        'daily_breakdown': baseline['daily_breakdown'],
        'scenarios': scenario_payload,
    }

    output_path = Path(args.output_path) if args.output_path else OUTPUT_DIR / 'confidence_position_sizing_backtest.json'
    output_path.write_text(json.dumps(payload, indent=2))

    logger.info('Backtest sample: %s bets from %s to %s', payload['sample']['bets'], payload['sample']['date_min'], payload['sample']['date_max'])
    for scenario in payload['scenarios']:
        logger.info('Scenario: %s | bets=%s | hit_rate=%s%%', scenario['name'], scenario['sample']['bets'], scenario['sample']['hr_hit_rate_pct'])
        for result in scenario['strategies']:
            logger.info(
                '  %s | ROI=%s%% profit=%s stake=%s avg_stake=%s',
                result['name'],
                result['roi_pct'],
                result['profit'],
                result['stake'],
                result['avg_stake'],
            )
    logger.info('Saved backtest output to %s', output_path)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
