#!/usr/bin/env python3
"""
Matchup Analysis Model - HR Prediction (Using complete_dataset.csv)
Phase 1: Build XGBoost model using available features
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support, confusion_matrix
import xgboost as xgb
import json
from datetime import datetime

print("=" * 80)
print("PHASE 1: Matchup Prediction Model (complete_dataset.csv)")
print("=" * 80)

# Load data
print("\n[1] Loading data...")
df = pd.read_csv('data/complete_dataset.csv', low_memory=False)

print(f"  complete_dataset.csv: {len(df)} rows")

# Check label distribution
print("\n[2] Analyzing labels...")
print(f"  Total rows: {len(df)}")
print(f"  HRs (label=1): {df['label'].sum()}")
print(f"  No HR (label=0): {(df['label'] == 0).sum()}")
print(f"  HR rate: {df['label'].mean()*100:.2f}%")
print(f"  Missing labels: {df['label'].isna().sum()}")

# Select features
print("\n[3] Selecting features...")

# Hitter features
hitter_features = [
    'barrel_rate', 'exit_velocity_avg', 'hard_hit_percent', 'sweet_spot_percent',
    'swing_optimization_score', 'swing_attack_angle', 'swing_bat_speed'
]

# Pitcher features
pitcher_features = [
    'pitcher_era', 'pitcher_k_per_9', 'pitcher_whip', 'pitcher_hr_per_9',
    'k_rate_pct', 'hr_rate_pct', 'fly_ball_pct'
]

# Game context
context_features = ['is_home', 'confidence_score', 'odds_decimal']

# All features
all_features = hitter_features + pitcher_features + context_features

# Check which features exist and have data
available_features = []
feature_coverage = {}

for feat in all_features:
    if feat in df.columns:
        non_null_pct = df[feat].notna().mean() * 100
        feature_coverage[feat] = non_null_pct
        if non_null_pct > 50:  # At least 50% coverage
            available_features.append(feat)
            print(f"  ✓ {feat:30s}: {non_null_pct:5.1f}% coverage")
        else:
            print(f"  ✗ {feat:30s}: {non_null_pct:5.1f}% coverage (insufficient)")
    else:
        print(f"  ✗ {feat:30s}: NOT FOUND")
        feature_coverage[feat] = 0

print(f"\n  Total available features: {len(available_features)}")

# Prepare training data
print("\n[4] Preparing training data...")

# Drop rows with missing label
train_data = df.dropna(subset=['label']).copy()
print(f"  After dropping missing labels: {len(train_data)} rows")

# Fill missing feature values
for feat in available_features:
    if train_data[feat].dtype in ['float64', 'int64']:
        median_val = train_data[feat].median()
        train_data[feat] = train_data[feat].fillna(median_val)
    else:
        train_data[feat] = train_data[feat].fillna(0)

X = train_data[available_features]
y = train_data['label']

# Save odds for betting simulation
if 'odds_decimal' in train_data.columns:
    odds = train_data['odds_decimal']
else:
    odds = None

# Train/test split (time-based if game_date available)
if 'game_date' in train_data.columns and train_data['game_date'].notna().any():
    train_data['game_date'] = pd.to_datetime(train_data['game_date'], errors='coerce')
    valid_dates = train_data['game_date'].notna()

    if valid_dates.sum() > len(train_data) * 0.5:
        split_date = train_data['game_date'].quantile(0.8)
        train_idx = train_data['game_date'] <= split_date
        test_idx = train_data['game_date'] > split_date

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        if odds is not None:
            odds_train, odds_test = odds[train_idx], odds[test_idx]

        print(f"  Train: {len(X_train)} rows ({y_train.mean()*100:.2f}% HR rate)")
        print(f"  Test: {len(X_test)} rows ({y_test.mean()*100:.2f}% HR rate)")
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        if odds is not None:
            odds_train, odds_test = odds[y_test.index], odds[y_test.index]
        print(f"  Using random split (insufficient date coverage)")
else:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    if odds is not None:
        odds_train, odds_test = odds[y_test.index], odds[y_test.index]
    print(f"  Using random split (no game_date)")

# Train XGBoost model
print("\n[5] Training XGBoost model...")

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
    use_label_encoder=False,
    n_jobs=-1
)

model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False
)

# Evaluate
print("\n[6] Evaluating model...")

# Predictions
y_pred_proba = model.predict_proba(X_test)[:, 1]
y_pred = (y_pred_proba > 0.5).astype(int)

# Metrics
auc = roc_auc_score(y_test, y_pred_proba)
accuracy = accuracy_score(y_test, y_pred)
precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
cm = confusion_matrix(y_test, y_pred)

print(f"  ROC AUC: {auc:.4f}")
print(f"  Accuracy: {accuracy:.4f}")
print(f"  Precision: {precision:.4f}")
print(f"  Recall: {recall:.4f}")
print(f"  F1 Score: {f1:.4f}")
print(f"\n  Confusion Matrix:")
print(f"    Predicted 0: {cm[0,0]:6d}  (actual 0)")
print(f"    Predicted 1: {cm[1,0]:6d}  (actual 0 - false positives)")
print(f"    Predicted 0: {cm[0,1]:6d}  (actual 1 - false negatives)")
print(f"    Predicted 1: {cm[1,1]:6d}  (actual 1)")

# Feature importance
print("\n[7] Feature importance...")
importance = model.feature_importances_
feature_importance = sorted(zip(available_features, importance), key=lambda x: x[1], reverse=True)

for feat, imp in feature_importance[:15]:
    cov = feature_coverage.get(feat, 0)
    print(f"  {feat:30s}: {imp:.4f} (coverage: {cov:.1f}%)")

# Betting simulation if odds available
if odds is not None and len(odds_test) > 0:
    print("\n[8] Simulating betting strategies...")

    def simulate_strategy(y_true, y_pred_proba, odds, strategy_name, threshold_func, description=""):
        """Simulate betting strategy"""
        bankroll = 1000
        bets = []

        for i in range(len(y_true)):
            threshold = threshold_func(y_pred_proba[i], odds.iloc[i])

            if threshold:
                implied_prob = 1 / odds.iloc[i]
                edge = y_pred_proba[i] - implied_prob

                if edge > 0:
                    kelly_pct = min(0.25 * edge / (odds.iloc[i] - 1), 0.25)
                    wager = bankroll * kelly_pct

                    if wager > 0:
                        if y_true.iloc[i] == 1:
                            profit = wager * (odds.iloc[i] - 1)
                            bankroll += profit
                        else:
                            bankroll -= wager

                        bets.append({'edge': edge, 'wager': wager, 'won': y_true.iloc[i] == 1})

        roi = (bankroll - 1000) / 1000 * 100
        win_rate = len([b for b in bets if b['won']]) / len(bets) if bets else 0
        print(f"  {strategy_name:30s}: ROI: {roi:+6.1f}%, {len(bets):4d} bets, Win: {win_rate:.1%} - {description}")

    # Strategy 1: All bets (baseline)
    simulate_strategy(y_test, y_pred_proba, odds_test, "All Bets",
                     lambda p, o: True, "Place bet on every prediction")

    # Strategy 2: Minimum edge 5%
    simulate_strategy(y_test, y_pred_proba, odds_test, "Min Edge 5%",
                     lambda p, o: p - 1/o > 0.05, "Only bet when model edge > 5%")

    # Strategy 3: Minimum edge 8%
    simulate_strategy(y_test, y_pred_proba, odds_test, "Min Edge 8%",
                     lambda p, o: p - 1/o > 0.08, "Only bet when model edge > 8%")

    # Strategy 4: High confidence only (p > 0.15)
    simulate_strategy(y_test, y_pred_proba, odds_test, "High Confidence (>15%)",
                     lambda p, o: p > 0.15, "Only bet when predicted HR prob > 15%")

    # Strategy 5: Conservative (edge > 10%)
    simulate_strategy(y_test, y_pred_proba, odds_test, "Conservative (>10% edge)",
                     lambda p, o: p - 1/o > 0.10, "Only bet when model edge > 10%")

else:
    print("\n[8] Skipping betting simulation (no odds data)")

# Save results
results = {
    'model_type': 'XGBoost HR Prediction',
    'timestamp': datetime.now().isoformat(),
    'data': {
        'total_rows': len(df),
        'labeled_rows': len(train_data),
        'train_rows': len(X_train),
        'test_rows': len(X_test),
        'hr_rate': float(y.mean()),
        'features_used': available_features,
        'feature_coverage': feature_coverage
    },
    'metrics': {
        'auc': float(auc),
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'confusion_matrix': cm.tolist()
    },
    'feature_importance': [
        {'feature': feat, 'importance': float(imp)}
        for feat, imp in feature_importance
    ]
}

with open('data/matchup_model_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n  Results saved to data/matchup_model_results.json")

print("\n" + "=" * 80)
print("PHASE 1 COMPLETE")
print("=" * 80)
print("\nKey Findings:")
print(f"  • Model AUC: {auc:.4f} (higher is better, 0.5 is random)")
print(f"  • Current HR rate in data: {y.mean()*100:.2f}%")
print(f"  • Model can predict HRs with {auc:.1%} accuracy")
print("\nNext Steps:")
print("  1. Analyze which features are most predictive")
print("  2. Add missing critical features (weather, travel, stadium dimensions)")
print("  3. Backtest on full season with time-based split")
print("  4. Compare with current -67% ROI baseline")
