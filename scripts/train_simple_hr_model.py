#!/usr/bin/env python3
"""
Simple HR Probability Model for ProjectionAI

Training XGBoost model to predict home run probability.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import logging
from pathlib import Path
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

FEATURES_PATH = '/home/futurepr0n/Development/ProjectionAI/data/comprehensive_features.csv'
OUTPUT_DIR = Path('/home/futurepr0n/Development/ProjectionAI/models')
OUTPUT_DIR.mkdir(exist_ok=True)


def load_data():
    """Load and preprocess features"""
    logger.info(f"Loading features from {FEATURES_PATH}")
    df = pd.read_csv(FEATURES_PATH, low_memory=False)
    logger.info(f"Loaded {len(df)} records")

    # Select numeric features only
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Remove features with too many NaNs
    for col in numeric_cols:
        null_pct = df[col].isna().sum() / len(df)
        if null_pct > 0.3:
            df = df.drop(columns=[col])

    # Define target (proxy for HR outcome)
    df['target'] = (df['confidence_score'] >= 70).astype(int)

    logger.info(f"After filtering: {len(df)} records, {len(numeric_cols)} features")

    return df, numeric_cols


def train_model(df, feature_names):
    """Train XGBoost model"""
    logger.info("Training XGBoost model...")

    X = df[feature_names].fillna(0)
    y = df['target']

    # Train/test split
    split_point = int(len(df) * 0.8)
    X_train = df.iloc[:split_point][feature_names].fillna(0)
    y_train = df.iloc[:split_point]['target']

    X_test = df.iloc[split_point:][feature_names].fillna(0)
    y_test = df.iloc[split_point:]['target']

    logger.info(f"Train set: {len(X_train)}, Test set: {len(X_test)}")

    # Train XGBoost
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )

    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Calculate accuracy
    accuracy = (y_pred == y_test).mean()

    logger.info(f"\nModel Performance:")
    logger.info(f"  Accuracy: {accuracy:.4f}")
    logger.info(f"  Win Rate: {accuracy * 100:.1f}%")

    # Feature importance
    importance = model.feature_importances_
    sorted_idx = np.argsort(importance)[::-1]

    logger.info("\nTop 10 Most Important Features:")
    for i in range(min(10, len(sorted_idx))):
        logger.info(f"  {feature_names[sorted_idx[i]]}: {importance[sorted_idx[i]]:.4f}")

    return model, feature_names, importance


def main():
    """Main execution"""
    logger.info("=" * 60)
    logger.info("PROJECTIONAI: HR Probability Model Training")
    logger.info("=" * 60)
    logger.info(f"Started: {datetime.now()}")

    # Load data
    df, feature_names = load_data()

    if len(df) == 0:
        logger.error("No data loaded!")
        return

    # Train model
    model, feature_names, importance = train_model(df, feature_names)

    # Save model
    model_path = OUTPUT_DIR / 'hr_probability_model.json'
    model.save_model(model_path, format='json')
    logger.info(f"\nModel saved to {model_path}")

    # Create analysis summary
    summary = f"""
=== MODEL TRAINING SUMMARY ===
Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}
Records Used: {len(df)}

FEATURE IMPORTANCE (Top 10):
"""

    sorted_idx = np.argsort(importance)[::-1]
    for i in range(min(10, len(sorted_idx))):
        summary += f"{feature_names[sorted_idx[i]]}: {importance[sorted_idx[i]]:.4f}\n"

    summary += f"""
NOTES:
- Using confidence_score >= 70 as proxy for HR outcome
- Model trained with {len(feature_names)} features
- Features with >30% NaNs were filtered out
- In production, you need actual HR outcomes for proper evaluation
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
