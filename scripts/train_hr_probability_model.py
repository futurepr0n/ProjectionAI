#!/usr/bin/env python3
"""
Train HR Probability Model for ProjectionAI

Builds XGBoost model to predict home run probability for hitter-pitcher matchups.
Replaces gut-feel confidence system with data-driven predictions.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score
import xgboost as xgb
import xgboost as xgb
import logging
from pathlib import Path
from datetime import datetime

# Make matplotlib optional
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None
    logging.warning("matplotlib not available, plotting disabled")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

FEATURES_PATH = '/home/futurepr0n/Development/ProjectionAI/data/comprehensive_features.csv'
OUTPUT_DIR = Path('/home/futurepr0n/Development/ProjectionAI/models')
OUTPUT_DIR.mkdir(exist_ok=True)

# Feature groups for analysis
NUMERIC_FEATURES = [
    'avg_ev', 'max_ev', 'ev95_plus_percent', 'barrel_rate', 'sweet_spot_rate',
    'avg_distance', 'avg_hr_distance', 'avg_launch_angle',
    'k_percent', 'era', 'bb_percent',
    'spin_composite', 'fastball_spin', 'breaking_spin', 'offspeed_spin',
    'xwoba', 'xba', 'xslg', 'xso', 'ops', 'swing_speed'
]

CATEGORICAL_FEATURES = ['is_home', 'team']


def load_data():
    """Load and preprocess features"""
    logger.info(f"Loading features from {FEATURES_PATH}")
    df = pd.read_csv(FEATURES_PATH, low_memory=False)
    logger.info(f"Loaded {len(df)} records")

    # Filter to only picks with outcome data
    # Note: For now, we'll use all records and focus on feature importance
    # In production, you'd filter to games where we have outcomes

    # Fill numeric features with median
    for col in NUMERIC_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)

    # Log feature completeness
    logger.info("\nFeature Completeness:")
    for group, features in [
        ('Hitter EV', ['avg_ev', 'max_ev', 'barrel_rate', 'avg_distance']),
        ('Hitter x-Stats', ['xwoba', 'xBA', 'xSLG', 'swing_speed']),
        ('Pitcher Stats', ['k_percent', 'era', 'bb_percent', 'spin_composite']),
    ]:
        complete = df[available].notna().sum()
        logger.info(f"  {group}: {complete}/{len(df):,} ({complete/len(df)*100:.1f}%)")

    return df


def train_model(df):
    """Train XGBoost model for HR probability"""
    logger.info("Training XGBoost model...")

    # Select available features
    available_features = [f for f in NUMERIC_FEATURES if f in df.columns]

    # Prepare training data
    # Note: We don't have actual HR outcomes in current data
    # For now, we'll use the features to demonstrate the modeling pipeline
    # In production, you'd need to join with actual results

    X = df[available_features].fillna(0)
    y = (df['confidence_score'] >= 70).astype(int)  # Using high confidence as proxy for HR

    # Encode categorical features
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            if df[col].dtype == 'object':
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].fillna('Unknown'))

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train XGBoost
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss'
    )

    model.fit(X_train, y_train)

    # Predict probabilities
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Calculate metrics
    logloss = log_loss(y_test, y_pred_proba)
    brier = brier_score_loss(y_test, y_pred_proba)

    logger.info(f"\nModel Performance:")
    logger.info(f"  Log Loss: {logloss:.4f}")
    logger.info(f"  Brier Score: {brier:.4f}")
    logger.info(f"  ROC AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")

    # Feature importance
    importance = pd.DataFrame({
        'feature': available_features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    logger.info("\nTop 10 Most Important Features:")
    for idx, row in importance.head(10).iterrows():
        logger.info(f"   {row['feature']}: {row['importance']:.4f}")

    return model, importance, X_test.columns.tolist()


def feature_importance_analysis(df, importance):
    """Analyze feature distributions by outcome"""
    logger.info("\n=== FEATURE DISTRIBUTION ANALYSIS ===")

    # Group by proxy outcome (confidence >= 70)
    df['proxy_outcome'] = (df['confidence_score'] >= 70).astype(int)

    for feature in importance.head(5)['feature'].tolist():
        if feature in df.columns:
            logger.info(f"\n{feature}:")
            logger.info(f"  All: {df[feature].mean():.2f} (median: {df[feature].median():.2f})")
            logger.info(f"  High Confidence: {df[df['proxy_outcome'] == 1][feature].mean():.2f}")
            logger.info(f"  Low Confidence: {df[df['proxy_outcome'] == 0][feature].mean():.2f}")


def backtest_simulation(df, model):
    """Simulate betting strategy"""
    logger.info("\n=== BACKTESTING SIMULATION ===")

    # Split data by date for time-series validation
    if 'game_date' in df.columns:
        df['game_date'] = pd.to_datetime(df['game_date'])
        df = df.sort_values('game_date')
        split_point = int(len(df) * 0.8)
        train_df = df.iloc[:split_point]
        test_df = df.iloc[split_point:]

        logger.info(f"Train on: {train_df['game_date'].min()} to {train_df['game_date'].max()}")
        logger.info(f"Test on:  {test_df['game_date'].min()} to {test_df['game_date'].max()}")

        # Simulate betting on test set
        # High confidence (>=70) = bet
        test_df['bet'] = (test_df['confidence_score'] >= 70).astype(int)

        # Actual HR proxy (confidence >= 70)
        test_df['actual_hr'] = (test_df['confidence_score'] >= 70).astype(int)

        # Calculate results
        total_bets = len(test_df)
        winners = test_df['actual_hr'].sum()
        roi = (test_df['odds_decimal'] * test_df['actual_hr']).sum() - total_bets

        logger.info(f"\nBacktest Results:")
        logger.info(f"  Total Bets: {total_bets}")
        logger.info(f"  Win Rate: {winners/total_bets*100:.1f}%")
        logger.info(f"  ROI: {roi:.2f}")


def main():
    """Main execution"""
    logger.info("=" * 60)
    logger.info("PROJECTIONAI: HR Probability Model Training")
    logger.info("=" * 60)
    logger.info(f"Started: {datetime.now()}")

    # Load data
    df = load_data()

    if len(df) == 0:
        logger.error("No data loaded!")
        return

    # Train model
    model, importance, feature_names = train_model(df)

    # Feature importance analysis
    feature_importance_analysis(df, importance)

    # Backtest simulation
    backtest_simulation(df, model)

    # Save model
    model_path = OUTPUT_DIR / 'hr_probability_model.json'
    model.save_model(model_path, format='json')
    logger.info(f"\nModel saved to {model_path}")

    # Save feature importance
    importance_path = OUTPUT_DIR / 'feature_importance.csv'
    importance.to_csv(importance_path, index=False)
    logger.info(f"Feature importance saved to {importance_path}")

    # Create analysis summary
    summary = f"""
=== MODEL TRAINING SUMMARY ===
Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}
Records Used: {len(df):,}

FEATURE IMPORTANCE (Top 10):
{importance.head(10).to_string(index=False)}

NOTES:
- Model trained using confidence_score >= 70 as proxy for HR outcomes
- Actual HR outcomes needed for proper evaluation
- Weather, travel, and stadium features would significantly improve accuracy
- Current model provides baseline for feature engineering pipeline
"""

    summary_path = OUTPUT_DIR / 'model_training_summary.txt'
    with open(summary_path, 'w') as f:
        f.write(summary)
    logger.info(f"Summary saved to {summary_path}")

    logger.info("\n" + "=" * 60)
    logger.info("Training complete")
    logger.info(f"Completed: {datetime.now()}")


if __name__ == '__main__':
    main()
