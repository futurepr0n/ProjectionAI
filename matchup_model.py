#!/usr/bin/env python3
"""
Matchup Analysis Model - HR Prediction
Phase 1: Build XGBoost model using comprehensive features
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support
import xgboost as xgb
import json
from datetime import datetime

print("=" * 80)
print("PHASE 1: Matchup Prediction Model")
print("=" * 80)

# Load data
print("\n[1] Loading data...")
comprehensive = pd.read_csv('data/comprehensive_features.csv')
labeled = pd.read_csv('data/labeled_dataset.csv')

print(f"  comprehensive_features.csv: {len(comprehensive)} rows")
print(f"  labeled_dataset.csv: {len(labeled)} rows")

# Check columns
print("\n[2] Checking columns...")
print(f"  Comprehensive columns: {comprehensive.shape[1]}")
print(f"  Labeled columns: {labeled.shape[1]}")

# Merge on game_id and player_normalized
print("\n[3] Merging datasets...")
# Normalize player names for merging
comprehensive['merge_key'] = comprehensive['player_normalized'].fillna('').str.lower()
labeled['merge_key'] = labeled['player_name_clean'].fillna('').str.lower()

merged = pd.merge(
    comprehensive,
    labeled[['game_id', 'merge_key', 'label', 'odds_decimal', 'game_date']],
    on=['game_id', 'merge_key'],
    how='inner'
)

print(f"  Merged: {len(merged)} rows")
print(f"  HR outcomes: {merged['label'].sum()}/{len(merged)} ({merged['label'].mean()*100:.1f}%)")

# Select features
print("\n[4] Selecting features...")

# Hitter features (from exit velocity and batter stats)
hitter_features = [
    'avg_ev', 'max_ev', 'ev95_plus_percent', 'avg_distance', 'avg_hr_distance',
    'avg_launch_angle', 'barrel_rate', 'sweet_spot_rate',
    'avg', 'obp', 'slg', 'ops', 'woba', 'xwoba', 'xba', 'xslg',
    'k_percent_xstats', 'bb_percent_xstats', 'swing_speed'
]

# Pitcher features
pitcher_features = [
    'k_percent', 'era', 'bb_percent',
    'fastball_spin', 'breaking_spin', 'offspeed_spin', 'spin_composite'
]

# Game context
context_features = ['is_home']

# All features
all_features = hitter_features + pitcher_features + context_features

# Check which features exist and have data
available_features = []
for feat in all_features:
    if feat in merged.columns:
        non_null_pct = merged[feat].notna().mean() * 100
        if non_null_pct > 10:  # At least 10% coverage
            available_features.append(feat)
            print(f"  ✓ {feat}: {non_null_pct:.1f}% coverage")
        else:
            print(f"  ✗ {feat}: {non_null_pct:.1f}% coverage (insufficient)")
    else:
        print(f"  ✗ {feat}: NOT FOUND")

print(f"\n  Total available features: {len(available_features)}")

# Prepare training data
print("\n[5] Preparing training data...")

# Drop rows with missing label
train_data = merged.dropna(subset=['label']).copy()

# Fill missing feature values
for feat in available_features:
    if train_data[feat].dtype in ['float64', 'int64']:
        train_data[feat] = train_data[feat].fillna(train_data[feat].median())
    else:
        train_data[feat] = train_data[feat].fillna(0)

X = train_data[available_features]
y = train_data['label']
odds = train_data['odds_decimal']

# Train/test split by date (time-based split)
if 'game_date_x' in train_data.columns:
    train_data['game_date_x'] = pd.to_datetime(train_data['game_date_x'])
    split_date = train_data['game_date_x'].quantile(0.8)  # 80% train, 20% test
    train_idx = train_data['game_date_x'] <= split_date
    test_idx = train_data['game_date_x'] > split_date

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    odds_train, odds_test = odds[train_idx], odds[test_idx]

    print(f"  Train: {len(X_train)} rows ({y_train.mean()*100:.1f}% HR rate)")
    print(f"  Test: {len(X_test)} rows ({y_test.mean()*100:.1f}% HR rate)")
else:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    odds_train, odds_test = odds[train_test_split(X, y, test_size=0.2, random_state=42)[1]], odds[train_test_split(X, y, test_size=0.2, random_state=42)[1]]

# Train XGBoost model
print("\n[6] Training XGBoost model...")

# Calculate scale_pos_weight for imbalanced data
scale_pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()
print(f"  Class imbalance: 1:{scale_pos_weight:.1f}")
print(f"  Using scale_pos_weight: {scale_pos_weight:.2f}")

model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    eval_metric='auc',
    use_label_encoder=False
)

model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False
)

# Evaluate
print("\n[7] Evaluating model...")

# Predictions
y_pred_proba = model.predict_proba(X_test)[:, 1]
y_pred = (y_pred_proba > 0.5).astype(int)

# Metrics
auc = roc_auc_score(y_test, y_pred_proba)
accuracy = accuracy_score(y_test, y_pred)
precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')

print(f"  ROC AUC: {auc:.4f}")
print(f"  Accuracy: {accuracy:.4f}")
print(f"  Precision: {precision:.4f}")
print(f"  Recall: {recall:.4f}")
print(f"  F1 Score: {f1:.4f}")

# Feature importance
print("\n[8] Feature importance...")
importance = model.feature_importances_
feature_importance = sorted(zip(available_features, importance), key=lambda x: x[1], reverse=True)

for feat, imp in feature_importance[:10]:
    print(f"  {feat:30s}: {imp:.4f}")

# Save results
results = {
    'model_type': 'XGBoost HR Prediction',
    'timestamp': datetime.now().isoformat(),
    'data': {
        'total_rows': len(train_data),
        'train_rows': len(X_train),
        'test_rows': len(X_test),
        'hr_rate': float(y.mean()),
        'features_used': available_features
    },
    'metrics': {
        'auc': float(auc),
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1)
    },
    'feature_importance': [
        {'feature': feat, 'importance': float(imp)}
        for feat, imp in feature_importance
    ]
}

with open('data/matchup_model_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n  Results saved to data/matchup_model_results.json")

# Simulate betting strategy
print("\n[9] Simulating betting strategy...")

# Kelly criterion simulation
def simulate_kelly(y_true, y_pred_proba, odds, kelly_fraction=0.25, min_edge=0.05):
    """Simulate Kelly criterion betting"""
    bankroll = 1000
    bets = []

    for i in range(len(y_true)):
        # Calculate implied probability
        implied_prob = 1 / odds.iloc[i]

        # Calculate edge
        edge = y_pred_proba[i] - implied_prob

        # Only bet if we have sufficient edge
        if edge > min_edge:
            # Kelly fraction (capped at 25%)
            kelly_pct = min(kelly_fraction * edge / (odds.iloc[i] - 1), 0.25)
            wager = bankroll * kelly_pct

            if wager > 0:
                if y_true.iloc[i] == 1:
                    profit = wager * (odds.iloc[i] - 1)
                    bankroll += profit
                else:
                    bankroll -= wager

                bets.append({
                    'edge': edge,
                    'wager': wager,
                    'won': y_true.iloc[i] == 1,
                    'bankroll': bankroll
                })

    return bankroll, len(bets), len([b for b in bets if b['won']]) / len(bets) if bets else 0

# Simulate different thresholds
print("\n  Testing different edge thresholds...")
thresholds = [0.02, 0.05, 0.08, 0.10, 0.15]

for min_edge in thresholds:
    final_bankroll, num_bets, win_rate = simulate_kelly(y_test, y_pred_proba, odds_test, min_edge=min_edge)
    roi = (final_bankroll - 1000) / 1000 * 100
    print(f"  Edge > {min_edge*100:4.1f}%: {num_bets:4d} bets, ${final_bankroll:7.2f}, ROI: {roi:+6.1f}%, Win rate: {win_rate:.1%}")

print("\n" + "=" * 80)
print("PHASE 1 COMPLETE")
print("=" * 80)
print("\nNext Steps:")
print("1. Analyze which features are most predictive")
print("2. Add missing critical features (weather, travel, stadium)")
print("3. Backtest on full 2025 season data")
print("4. Compare with current -67% ROI baseline")
