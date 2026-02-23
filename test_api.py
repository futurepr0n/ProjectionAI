#!/usr/bin/env python3
"""
Simple test of the dashboard API
"""
import sys
sys.path.insert(0, '/home/futurepr0n/Development/ProjectionAI')

from datetime import datetime
from dashboards.app import engine

# Test with 2025-08-29
target_date = datetime.strptime('2025-08-29', '%Y-%m-%d').date()
print(f"Testing predictions for {target_date}")

data = engine.generate_daily_predictions_with_results(target_date)

print(f"Total picks: {data['stats']['total_picks']}")
print(f"Total HRs: {data['stats']['total_hrs']}")
print(f"STRONG_BUY: {data['stats']['strong_buy_count']} picks, {data['stats']['strong_buy_hits']} hits")
print(f"BUY: {data['stats']['buy_count']} picks, {data['stats']['buy_hits']} hits")

print("\nTop 10 predictions:")
for p in data['predictions'][:10]:
    hr = '✅ HR' if p.get('actual_hr') else '❌'
    print(f"  {hr} {p['player_name']:<20} ({p['team']:<3}) Conf: {p.get('hellraiser_confidence')} Signal: {p['signal']}")
