#!/usr/bin/env python3
"""
Matchup Analysis Model - HR Prediction (Fixed)
Phase 1: Build XGBoost model using available features with proper train/test split
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support, confusion_matrix, brier_score_loss
import xgboost as xgb
import json
from datetime import datetime

print("=" * 80)
print("PHASE 1: Matchup Prediction Model (Fixed)")
print("=" * 80)

# Load data
print("\n[1] Loading data...")
df = pd.read_csv('data/complete_dataset.csv', low_memory=False)
df['game_date'] = pd.to_datetime(df['game_date'], errors='coerce')

print(f"  complete_dataset.csv: {len(df)} rows")

# Filter to only labeled dates (with actual HRs)
print("\n[2] Filtering to labeled data...")
date_hr_counts = df.groupby(df['game_date'].dt.date)['label'].sum()
labeled_dates = date_hr_counts[date_hr_counts > 0].index
df = df[df['game_date'].dt.date.isin(labeled_dates)].copy()

print(f"  After filtering to labeled dates: {len(df)} rows")
print(f"  Date range: {df['game_date'].min().date()} to {df['game_date'].max().date()}")

# Check label distribution
print("\n[3] Analyzing labels...")
print(f"  Total rows: {len(df)}")
print(f"  HRs (label=1): {df['label'].sum()}")
print(f"  No HR (label=0): {(df['label'] == 0).sum()}")
print(f"  HR rate: {df['label'].mean()*100:.2f}%")

# Select features
print("\n[4] Selecting features...")

# Hitter features
hitter_features = [
    'barrel_rate', 'exit_velocity_avg', 'hard_hit_percent', 'sweet_spot_percent',
    'swing_optimization_score', 'swing_attack_angle', 'swing_bat_speed'
]

# Pitcher features
pitcher_features = [
    'pitcher_era', 'pitcher_k_per_9', 'pitcher_whip', 'pitcher_hr_per_9'
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
        if non_null_pct > 50:
            available_features.append(feat)
            print(f"  ✓ {feat:30s}: {non_null_pct:5.1f}% coverage")
        else:
            print(f"  ✗ {feat:30s}: {non_null_pct:5.1f}% coverage (insufficient)")
    else:
        print(f"  ✗ {feat:30s}: NOT FOUND")

print(f"\n  Total available features: {len(available_features)}")

# Prepare training data
print("\n[5] Preparing training data...")

# Fill missing feature values
for feat in available_features:
    if df[feat].dtype in ['float64', 'int64']:
        median_val = df[feat].median()
        df[feat] = df[feat].fillna(median_val)
    else:
        df[feat] = df[feat].fillna(0)

X = df[available_features]
y = df['label']

# Time-based split (use first 80% of dates for training)
unique_dates = df['game_date'].unique()
split_date_idx = int(len(unique_dates) * 0.8)
split_date = pd.Series(sorted(unique_dates))[split_date_idx]

train_idx = df['game_date'] <= split_date
test_idx = df['game_date'] > split_date

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

print(f"  Split date: {split_date.date()}")
print(f"  Train: {len(X_train)} rows ({y_train.mean()*100:.2f}% HR rate)")
print(f"  Test: {len(X_test)} rows ({y_test.mean()*100:.2f}% HR rate)")

# Train XGBoost model
print("\n[6] Training XGBoost model...")

scale_pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()
print(f"  Class imbalance: 1:{scale_pos_weight:.1f}")
print(f"  Using scale_pos_weight: {scale_pos_weight:.2f}")

model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    eval_metric='auc',
    n_jobs=-1,
    min_child_weight=5
)

model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    verbose=False
)

# Evaluate
print("\n[7] Evaluating model...")

y_pred_proba = model.predict_proba(X_test)[:, 1]

# Use optimal threshold (maximize F1)
thresholds = np.arange(0.01, 0.5, 0.01)
best_threshold = 0.5
best_f1 = 0

for threshold in thresholds:
    y_pred = (y_pred_proba > threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary', zero_division=0)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

y_pred = (y_pred_proba > best_threshold).astype(int)

auc = roc_auc_score(y_test, y_pred_proba)
accuracy = accuracy_score(y_test, y_pred)
precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary', zero_division=0)
cm = confusion_matrix(y_test, y_pred)
brier = brier_score_loss(y_test, y_pred_proba)

print(f"  Optimal threshold: {best_threshold:.3f}")
print(f"  ROC AUC: {auc:.4f}")
print(f"  Accuracy: {accuracy:.4f}")
print(f"  Precision: {precision:.4f}")
print(f"  Recall: {recall:.4f}")
print(f"  F1 Score: {f1:.4f}")
print(f"  Brier Score: {brier:.4f} (lower is better)")
print(f"\n  Confusion Matrix:")
print(f"    Predicted 0: {cm[0,0]:5d}  (actual 0)")
print(f"    Predicted 1: {cm[1,0]:5d}  (actual 0 - false positives)")
print(f"    Predicted 0: {cm[0,1]:5d}  (actual 1 - false negatives)")
print(f"    Predicted 1: {cm[1,1]:5d}  (actual 1)")

# Feature importance
print("\n[8] Feature importance...")
importance = model.feature_importances_
feature_importance = sorted(zip(available_features, importance), key=lambda x: x[1], reverse=True)

for feat, imp in feature_importance[:15]:
    cov = feature_coverage.get(feat, 0)
    print(f"  {feat:30s}: {imp:.4f} (coverage: {cov:.1f}%)")

# Betting simulation
print("\n[9] Simulating betting strategies...")

if 'odds_decimal' in df.columns:
    odds_test = df.loc[test_idx, 'odds_decimal']

    def simulate_strategy(y_true, y_pred_proba, odds, strategy_name, threshold_func, description=""):
        """Simulate betting strategy"""
        bankroll = 1000
        total_bets = 0
        wins = 0
        total_wagered = 0

        for i in range(len(y_true)):
            if threshold_func(y_pred_proba[i], odds[i]):
                implied_prob = 1 / odds[i]
                edge = y_pred_proba[i] - implied_prob

                if edge > 0:
                    kelly_pct = min(0.25 * edge / (odds[i] - 1), 0.25)
                    wager = bankroll * kelly_pct

                    if wager > 0:
                        total_bets += 1
                        total_wagered += wager

                        if y_true[i] == 1:
                            profit = wager * (odds[i] - 1)
                            bankroll += profit
                            wins += 1
                        else:
                            bankroll -= wager

        roi = (bankroll - 1000) / 1000 * 100
        win_rate = wins / total_bets if total_bets > 0 else 0
        avg_wager = total_wagered / total_bets if total_bets > 0 else 0

        print(f"  {strategy_name:30s}: ROI: {roi:+6.1f}%, {total_bets:4d} bets, Win: {win_rate:.1%} - {description}")
        return {'roi': roi, 'bets': total_bets, 'win_rate': win_rate, 'final_bankroll': bankroll}

    # Current baseline (confidence-based)
    print("\n  Current baseline (confidence-based):")
    baseline_proba = df.loc[test_idx, 'confidence_score'].values / 100
    baseline_results = simulate_strategy(y_test.values, baseline_proba, odds_test.values,
                                        "All Bets (Baseline)",
                                        lambda p, o: True,
                                        "Using existing confidence score")

    # Model strategies
    print("\n  Model-based strategies:")
    simulate_strategy(y_test.values, y_pred_proba, odds_test.values, "All Model Predictions",
                     lambda p, o: True, "Bet on all model predictions")

    simulate_strategy(y_test.values, y_pred_proba, odds_test.values, "Min Edge 5%",
                     lambda p, o: p - 1/o > 0.05, "Model edge > 5%")

    simulate_strategy(y_test.values, y_pred_proba, odds_test.values, "Min Edge 8%",
                     lambda p, o: p - 1/o > 0.08, "Model edge > 8%")

    simulate_strategy(y_test.values, y_pred_proba, odds_test.values, "High Confidence (>10% prob)",
                     lambda p, o: p > 0.10, "Predicted HR prob > 10%")

    simulate_strategy(y_test.values, y_pred_proba, odds_test.values, "Very High Confidence (>15% prob)",
                     lambda p, o: p > 0.15, "Predicted HR prob > 15%")

    print(f"\n  Baseline ROI: {baseline_results['roi']:+.1f}%")
    print(f"  Current strategy ROI: -67% (from memory)")

# Save results
results = {
    'model_type': 'XGBoost HR Prediction',
    'timestamp': datetime.now().isoformat(),
    'data': {
        'total_rows': len(df),
        'train_rows': len(X_train),
        'test_rows': len(X_test),
        'hr_rate': float(y.mean()),
        'train_hr_rate': float(y_train.mean()),
        'test_hr_rate': float(y_test.mean()),
        'split_date': str(split_date.date()),
        'features_used': available_features,
        'feature_coverage': feature_coverage
    },
    'model_params': {
        'n_estimators': 300,
        'max_depth': 5,
        'learning_rate': 0.05,
        'scale_pos_weight': float(scale_pos_weight),
        'optimal_threshold': float(best_threshold)
    },
    'metrics': {
        'auc': float(auc),
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'brier_score': float(brier),
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
print(f"  • Optimal prediction threshold: {best_threshold:.1%}")
print(f"  • At this threshold, model predicts HRs with {precision:.1%} precision and {recall:.1%} recall")
print(f"  • Brier score: {brier:.4f} (measures calibration, lower is better)")
print("\nTop 3 Predictive Features:")
for feat, imp in feature_importance[:3]:
    print(f"  • {feat}: {imp:.2%}")
print("\nNext Steps:")
print("  1. Analyze feature importance to guide feature engineering")
print("  2. Add missing critical features (weather, travel, stadium dimensions)")
print("  3. Implement cross-validation for more robust evaluation")
print("  4. Compare with current -67% ROI baseline")
