#!/usr/bin/env python3
"""
Researcher Agent: Profitable Markers Analysis
Analyzes ProjectionAI data to identify profitable betting patterns and missing features.
"""

import pandas as pd
import numpy as np
import json
from collections import defaultdict

def load_data():
    """Load the complete labeled dataset."""
    print("Loading labeled_dataset.csv...")
    df = pd.read_csv('data/labeled_dataset.csv', low_memory=False)

    # Calculate profit/loss
    df['profit'] = np.where(df['label'] == 1, df['odds_decimal'] - 1, -1)

    print(f"Loaded {len(df)} records")
    return df

def audit_missing_features(df):
    """Audit what features are currently tracked vs missing."""
    print("\n" + "="*80)
    print("FEATURE AUDIT: Current Tracking vs Missing Features")
    print("="*80)

    current_features = {
        "Batter Swing Metrics": ['swing_optimization_score', 'swing_attack_angle',
                                 'swing_ideal_rate', 'swing_bat_speed'],
        "Batter Quality Metrics": ['barrel_rate', 'exit_velocity_avg',
                                   'hard_hit_percent', 'sweet_spot_percent'],
        "Pitcher Metrics": ['pitcher_era', 'pitcher_k_per_9'],
        "Game Context": ['is_home', 'confidence_score', 'odds_decimal', 'venue'],
        "Game Info": ['game_time', 'game_description', 'home_team', 'away_team']
    }

    missing_features = {
        "Weather (Wind/Temp)": "NOT TRACKED",
        "Travel (Days Rest/Distance)": "NOT TRACKED",
        "Stadium Dimensions": "NOT TRACKED",
        "Umpire Bias": "NOT TRACKED",
        "Stadium Factors": "NOT TRACKED",
        "Time of Day/Day of Week": "PARTIAL (game_time exists, but not parsed)",
        "Temperature/Wind Data": "NOT TRACKED",
        "Previous Day's Games": "NOT TRACKED",
        "Team Rest Days": "NOT TRACKED",
        "Travel Distance": "NOT TRACKED"
    }

    # Report current feature completeness
    print("\nCURRENT FEATURES TRACKED:")
    print("-" * 80)
    for category, features in current_features.items():
        print(f"\n{category}:")
        for feat in features:
            non_null = df[feat].notna().sum()
            pct = (non_null / len(df)) * 100
            status = "✓" if pct > 50 else "⚠" if pct > 10 else "✗"
            print(f"  {status} {feat}: {pct:.1f}% complete ({non_null:,}/{len(df):,})")

    # Report missing features
    print("\n\nMISSING FEATURES NOT CURRENTLY TRACKED:")
    print("-" * 80)
    for feature, status in missing_features.items():
        print(f"  ✗ {feature}: {status}")

    return missing_features

def analyze_roi_patterns(df):
    """Analyze which feature combinations result in positive ROI."""
    print("\n" + "="*80)
    print("ROI ANALYSIS: Finding Profitable Patterns")
    print("="*80)

    # Overall stats
    total_bets = len(df)
    total_profit = df['profit'].sum()
    overall_roi = (total_profit / total_bets) * 100
    win_rate = (df['label'] == 1).sum() / total_bets * 100

    print(f"\nOVERALL PERFORMANCE:")
    print(f"  Total Bets: {total_bets:,}")
    print(f"  Win Rate: {win_rate:.2f}%")
    print(f"  Total Profit/Loss: ${total_profit:,.2f}")
    print(f"  Overall ROI: {overall_roi:.2f}%")

    # Filter to records with decent feature coverage
    has_batter_metrics = df['barrel_rate'].notna()
    has_swing_metrics = df['swing_optimization_score'].notna()

    print(f"\nDATA COVERAGE:")
    print(f"  Records with batter metrics: {has_batter_metrics.sum():,} ({has_batter_metrics.sum()/len(df)*100:.1f}%)")
    print(f"  Records with swing metrics: {has_swing_metrics.sum():,} ({has_swing_metrics.sum()/len(df)*100:.1f}%)")

    # Analyze winning patterns
    winners = df[df['label'] == 1].copy()
    losers = df[df['label'] == 0].copy()

    print(f"\nWINNER CHARACTERISTICS:")
    print(f"  Total winners: {len(winners):,}")

    # Find patterns in winners
    profitable_segments = []

    # Pattern 1: High confidence + specific odds range
    if has_swing_metrics.any():
        high_conf = df[df['confidence_score'] >= 70]
        if len(high_conf) > 100:
            seg_roi = (high_conf['profit'].sum() / len(high_conf)) * 100
            seg_win = (high_conf['label'] == 1).sum() / len(high_conf) * 100
            profitable_segments.append({
                'name': 'High Confidence (≥70%)',
                'count': len(high_conf),
                'win_rate': seg_win,
                'roi': seg_roi
            })

    # Pattern 2: Home games
    home_games = df[df['is_home'] == True]
    if len(home_games) > 100:
        seg_roi = (home_games['profit'].sum() / len(home_games)) * 100
        seg_win = (home_games['label'] == 1).sum() / len(home_games) * 100
        profitable_segments.append({
            'name': 'Home Games',
            'count': len(home_games),
            'win_rate': seg_win,
            'roi': seg_roi
        })

    # Pattern 3: Specific odds ranges
    for odds_min, odds_max in [(2.0, 4.0), (4.0, 6.0), (6.0, 8.0), (8.0, 12.0)]:
        segment = df[(df['odds_decimal'] >= odds_min) & (df['odds_decimal'] < odds_max)]
        if len(segment) > 100:
            seg_roi = (segment['profit'].sum() / len(segment)) * 100
            seg_win = (segment['label'] == 1).sum() / len(segment) * 100
            profitable_segments.append({
                'name': f'Odds {odds_min}-{odds_max}',
                'count': len(segment),
                'win_rate': seg_win,
                'roi': seg_roi
            })

    # Pattern 4: Combining swing metrics + confidence
    if has_swing_metrics.any():
        good_swing = df[(df['swing_optimization_score'].notna()) &
                       (df['swing_optimization_score'] >= df['swing_optimization_score'].median())]
        if len(good_swing) > 100:
            seg_roi = (good_swing['profit'].sum() / len(good_swing)) * 100
            seg_win = (good_swing['label'] == 1).sum() / len(good_swing) * 100
            profitable_segments.append({
                'name': 'Above-Avg Swing Score',
                'count': len(good_swing),
                'win_rate': seg_win,
                'roi': seg_roi
            })

    # Print segments
    print("\nSEGMENT ANALYSIS:")
    print("-" * 80)
    for seg in profitable_segments:
        indicator = "✓" if seg['roi'] > 0 else "✗"
        print(f"  {indicator} {seg['name']}:")
        print(f"      Bets: {seg['count']:,} | Win Rate: {seg['win_rate']:.1f}% | ROI: {seg['roi']:.2f}%")

    # Find the best segments
    print("\n\nBEST PERFORMING SEGMENTS (ROI > 0):")
    print("-" * 80)
    positive_segments = [s for s in profitable_segments if s['roi'] > 0]
    if positive_segments:
        for seg in sorted(positive_segments, key=lambda x: x['roi'], reverse=True)[:5]:
            print(f"  • {seg['name']}: ROI {seg['roi']:.2f}%, Win Rate {seg['win_rate']:.1f}%, {seg['count']:,} bets")
    else:
        print("  ⚠ No segments with positive ROI found!")

    return profitable_segments

def analyze_venue_performance(df):
    """Analyze performance by venue to identify stadium effects."""
    print("\n" + "="*80)
    print("VENUE ANALYSIS: Stadium-Specific Performance")
    print("="*80)

    venue_stats = []
    for venue in df['venue'].unique():
        if pd.notna(venue):
            venue_df = df[df['venue'] == venue]
            if len(venue_df) >= 50:  # Only analyze venues with sufficient data
                roi = (venue_df['profit'].sum() / len(venue_df)) * 100
                win_rate = (venue_df['label'] == 1).sum() / len(venue_df) * 100
                venue_stats.append({
                    'venue': venue,
                    'bets': len(venue_df),
                    'win_rate': win_rate,
                    'roi': roi
                })

    print("\nTOP 10 VENUES BY BET VOLUME:")
    print("-" * 80)
    for v in sorted(venue_stats, key=lambda x: x['bets'], reverse=True)[:10]:
        print(f"  {v['venue'][:30]:30s} | Bets: {v['bets']:4d} | Win%: {v['win_rate']:5.1f}% | ROI: {v['roi']:6.2f}%")

    print("\n\nTOP 10 VENUES BY ROI (minimum 100 bets):")
    print("-" * 80)
    high_volume = [v for v in venue_stats if v['bets'] >= 100]
    for v in sorted(high_volume, key=lambda x: x['roi'], reverse=True)[:10]:
        print(f"  {v['venue'][:30]:30s} | Bets: {v['bets']:4d} | Win%: {v['win_rate']:5.1f}% | ROI: {v['roi']:6.2f}%")

    return venue_stats

def identify_high_confidence_markers(df):
    """Identify markers that correlate with wins."""
    print("\n" + "="*80)
    print("HIGH CONFIDENCE MARKERS: Patterns to Require Next Season")
    print("="*80)

    winners = df[df['label'] == 1]
    losers = df[df['label'] == 0]

    markers = []

    # Marker 1: High confidence threshold
    for conf_thresh in [50, 60, 70, 80, 90]:
        conf_winners = winners[winners['confidence_score'] >= conf_thresh]
        conf_losers = losers[losers['confidence_score'] >= conf_thresh]
        if len(conf_winners) > 0:
            win_rate_at_thresh = len(conf_winners) / (len(conf_winners) + len(conf_losers)) * 100 if (len(conf_winners) + len(conf_losers)) > 0 else 0
            markers.append({
                'marker': f"Confidence Score ≥ {conf_thresh}%",
                'winners': len(conf_winners),
                'win_rate': win_rate_at_thresh
            })

    # Marker 2: Odds range
    for odds_range in [(2.0, 4.0), (4.0, 6.0), (6.0, 8.0), (8.0, 10.0)]:
        range_winners = winners[(winners['odds_decimal'] >= odds_range[0]) &
                               (winners['odds_decimal'] < odds_range[1])]
        range_losers = losers[(losers['odds_decimal'] >= odds_range[0]) &
                            (losers['odds_decimal'] < odds_range[1])]
        if len(range_winners) > 10:
            win_rate = len(range_winners) / (len(range_winners) + len(range_losers)) * 100 if (len(range_winners) + len(range_losers)) > 0 else 0
            markers.append({
                'marker': f"Odds {odds_range[0]}-{odds_range[1]}",
                'winners': len(range_winners),
                'win_rate': win_rate
            })

    # Marker 3: Home/Away
    home_winners = winners[winners['is_home'] == True]
    home_losers = losers[losers['is_home'] == True]
    away_winners = winners[winners['is_home'] == False]
    away_losers = losers[losers['is_home'] == False]

    if len(home_winners) > 0:
        home_win_rate = len(home_winners) / (len(home_winners) + len(home_losers)) * 100 if (len(home_winners) + len(home_losers)) > 0 else 0
        markers.append({
            'marker': 'Home Game',
            'winners': len(home_winners),
            'win_rate': home_win_rate
        })

    if len(away_winners) > 0:
        away_win_rate = len(away_winners) / (len(away_winners) + len(away_losers)) * 100 if (len(away_winners) + len(away_losers)) > 0 else 0
        markers.append({
            'marker': 'Away Game',
            'winners': len(away_winners),
            'win_rate': away_win_rate
        })

    # Marker 4: Swing metrics (if available)
    if df['swing_optimization_score'].notna().sum() > 1000:
        median_swing = df['swing_optimization_score'].median()
        high_swing_winners = winners[winners['swing_optimization_score'] >= median_swing]
        high_swing_losers = losers[losers['swing_optimization_score'] >= median_swing]
        if len(high_swing_winners) > 10:
            swing_win_rate = len(high_swing_winners) / (len(high_swing_winners) + len(high_swing_losers)) * 100 if (len(high_swing_winners) + len(high_swing_losers)) > 0 else 0
            markers.append({
                'marker': f'Swing Score ≥ {median_swing:.1f}',
                'winners': len(high_swing_winners),
                'win_rate': swing_win_rate
            })

    print("\nMARKER PERFORMANCE:")
    print("-" * 80)
    for m in sorted(markers, key=lambda x: x['win_rate'], reverse=True):
        print(f"  {m['marker']:30s} | Winners: {m['winners']:4d} | Win Rate: {m['win_rate']:5.2f}%")

    return markers

def generate_recommendations():
    """Generate final recommendations for next season."""
    print("\n" + "="*80)
    print("FINAL RECOMMENDATIONS")
    print("="*80)

    recommendations = {
        "Critical Missing Features": [
            "Weather data (wind direction/speed, temperature) - critical for home run potential",
            "Stadium dimensions (fence distances) - essential for context",
            "Travel factors (days rest, travel distance) - player fatigue indicators",
            "Umpire bias data (zone metrics, strikeout rates) - impact on pitch selection"
        ],
        "Data Quality Issues": [
            "Pitcher stats (ERA, K/9) only 0.2% complete - must be prioritized",
            "Advanced batter stats (barrel_rate, exit_velocity) only 42% complete",
            "Inconsistent feature coverage across records"
        ],
        "Strategy Recommendations": [
            "Set minimum confidence score threshold for all picks",
            "Focus on specific odds ranges with better historical performance",
            "Track and analyze venue-specific patterns",
            "Implement stricter data quality filters before model training"
        ],
        "Priority Actions": [
            "Integrate weather API into data pipeline",
            "Add stadium dimension database",
            "Calculate travel metrics from schedule data",
            "Fix pitcher stats data collection to achieve >90% coverage",
            "Improve batter stats coverage to >80%"
        ]
    }

    for category, items in recommendations.items():
        print(f"\n{category}:")
        for i, item in enumerate(items, 1):
            print(f"  {i}. {item}")

    return recommendations

def main():
    """Main analysis function."""
    print("="*80)
    print("PROJECTIONAI: PROFITABLE MARKERS RESEARCH")
    print("="*80)

    # Load data
    df = load_data()

    # Run analyses
    missing_features = audit_missing_features(df)
    profitable_segments = analyze_roi_patterns(df)
    venue_stats = analyze_venue_performance(df)
    markers = identify_high_confidence_markers(df)
    recommendations = generate_recommendations()

    # Save results
    results = {
        'overall_stats': {
            'total_bets': int(len(df)),
            'win_rate': float((df['label'] == 1).sum() / len(df) * 100),
            'total_profit': float(df['profit'].sum()),
            'roi': float((df['profit'].sum() / len(df)) * 100)
        },
        'missing_features': missing_features,
        'profitable_segments': profitable_segments,
        'high_confidence_markers': markers,
        'recommendations': recommendations
    }

    with open('research_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print("\n" + "="*80)
    print("✓ Analysis complete! Results saved to 'research_results.json'")
    print("="*80)

if __name__ == '__main__':
    main()
