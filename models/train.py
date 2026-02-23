"""
ProjectionAI - Model Training
Train XGBoost models for HR, Hit, and Strikeout predictions
"""

import xgboost as xgb
import pandas as pd
import numpy as np
import logging
from typing import Dict, Tuple, Optional
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.calibration import calibration_curve
import joblib
import json
from datetime import datetime

from data.feature_store import FeatureStore
from data.database import get_database

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """Train and evaluate XGBoost models for sports betting"""

    def __init__(self):
        self.db = get_database()
        self.feature_store = FeatureStore()

    def train_hr_model(self, start_date: str = '2023-04-01', end_date: str = '2024-11-01',
                       model_path: str = 'models/hr_model.json') -> xgb.XGBClassifier:
        """
        Train home run prediction model

        Args:
            start_date: Training start date
            end_date: Training end date
            model_path: Path to save model

        Returns:
            Trained XGBoost model
        """
        logger.info(f"🏠 Training HR model ({start_date} to {end_date})...")

        # Create training dataset
        df = self.feature_store.create_training_dataset(start_date, end_date, prediction_type='HR')

        if df.empty:
            logger.error("❌ No training data available")
            return None

        # Prepare features
        feature_cols = [col for col in df.columns if col not in ['game_id', 'batter_id', 'pitcher_id', 'game_date', 'label']]
        X = df[feature_cols]
        y = df['label']

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Calculate class weights (HR is rare, ~5%)
        scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)
        logger.info(f"   Class weight: {scale_pos_weight:.2f}")

        # Initialize model with parameters optimized for HR prediction
        params = {
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 200,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 1,  # Minimum loss reduction
            'min_child_weight': 5,
            'scale_pos_weight': scale_pos_weight,
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'random_state': 42,
            'n_jobs': -1
        }

        model = xgb.XGBClassifier(**params)

        # Train with early stopping
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )

        # Evaluate
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        metrics = self._calculate_metrics(y_test, y_pred, y_prob, "HR")

        # Feature importance
        importance = self.feature_store.get_feature_importance(model, feature_cols)
        self._log_feature_importance(importance)

        # Save model
        model.save_model(model_path)
        logger.info(f"✅ Model saved to {model_path}")

        # Save feature names
        with open(model_path.replace('.json', '_features.json'), 'w') as f:
            json.dump(feature_cols, f, indent=2)

        return model

    def train_hit_model(self, start_date: str = '2023-04-01', end_date: str = '2024-11-01',
                       model_path: str = 'models/hit_model.json') -> xgb.XGBClassifier:
        """
        Train hit prediction model

        Args:
            start_date: Training start date
            end_date: Training end date
            model_path: Path to save model

        Returns:
            Trained XGBoost model
        """
        logger.info(f"🎳 Training HIT model ({start_date} to {end_date})...")

        # Create training dataset
        df = self.feature_store.create_training_dataset(start_date, end_date, prediction_type='HIT')

        if df.empty:
            logger.error("❌ No training data available")
            return None

        # Prepare features
        feature_cols = [col for col in df.columns if col not in ['game_id', 'batter_id', 'pitcher_id', 'game_date', 'label']]
        X = df[feature_cols]
        y = df['label']

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Calculate class weights (hit rate ~40-50%)
        scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)
        logger.info(f"   Class weight: {scale_pos_weight:.2f}")

        # Initialize model with parameters optimized for hit prediction
        params = {
            'max_depth': 5,
            'learning_rate': 0.1,
            'n_estimators': 150,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 0.5,
            'min_child_weight': 3,
            'scale_pos_weight': scale_pos_weight,
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'random_state': 42,
            'n_jobs': -1
        }

        model = xgb.XGBClassifier(**params)

        # Train
        model.fit(X_train, y_train, verbose=False)

        # Evaluate
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        metrics = self._calculate_metrics(y_test, y_pred, y_prob, "HIT")

        # Feature importance
        importance = self.feature_store.get_feature_importance(model, feature_cols)
        self._log_feature_importance(importance)

        # Save model
        model.save_model(model_path)
        logger.info(f"✅ Model saved to {model_path}")

        # Save feature names
        with open(model_path.replace('.json', '_features.json'), 'w') as f:
            json.dump(feature_cols, f, indent=2)

        return model

    def train_so_model(self, start_date: str = '2023-04-01', end_date: str = '2024-11-01',
                      model_path: str = 'models/so_model.json') -> xgb.XGBClassifier:
        """
        Train strikeout prediction model

        Args:
            start_date: Training start date
            end_date: Training end date
            model_path: Path to save model

        Returns:
            Trained XGBoost model
        """
        logger.info(f"👟 Training SO model ({start_date} to {end_date})...")

        # Create training dataset
        df = self.feature_store.create_training_dataset(start_date, end_date, prediction_type='SO')

        if df.empty:
            logger.error("❌ No training data available")
            return None

        # Prepare features
        feature_cols = [col for col in df.columns if col not in ['game_id', 'batter_id', 'pitcher_id', 'game_date', 'label']]
        X = df[feature_cols]
        y = df['label']

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Calculate class weights
        scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)
        logger.info(f"   Class weight: {scale_pos_weight:.2f}")

        # Initialize model with parameters optimized for SO prediction
        params = {
            'max_depth': 5,
            'learning_rate': 0.1,
            'n_estimators': 150,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 0.3,
            'min_child_weight': 3,
            'scale_pos_weight': scale_pos_weight,
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'random_state': 42,
            'n_jobs': -1
        }

        model = xgb.XGBClassifier(**params)

        # Train
        model.fit(X_train, y_train, verbose=False)

        # Evaluate
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        metrics = self._calculate_metrics(y_test, y_pred, y_prob, "SO")

        # Feature importance
        importance = self.feature_store.get_feature_importance(model, feature_cols)
        self._log_feature_importance(importance)

        # Save model
        model.save_model(model_path)
        logger.info(f"✅ Model saved to {model_path}")

        # Save feature names
        with open(model_path.replace('.json', '_features.json'), 'w') as f:
            json.dump(feature_cols, f, indent=2)

        return model

    def _calculate_metrics(self, y_true, y_pred, y_prob, model_name: str) -> Dict:
        """Calculate and log model metrics"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0
        }

        logger.info(f"\n📊 {model_name} Model Metrics:")
        logger.info(f"   Accuracy:  {metrics['accuracy']:.4f}")
        logger.info(f"   Precision: {metrics['precision']:.4f}")
        logger.info(f"   Recall:    {metrics['recall']:.4f}")
        logger.info(f"   F1 Score:  {metrics['f1']:.4f}")
        logger.info(f"   ROC AUC:   {metrics['roc_auc']:.4f}")
        logger.info(f"   Positive samples: {sum(y_true)}/{len(y_true)} ({sum(y_true)/len(y_true)*100:.1f}%)")

        return metrics

    def _log_feature_importance(self, importance: Dict):
        """Log top feature importance"""
        sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]

        logger.info(f"\n🎯 Top 10 Feature Importance:")
        for feature, score in sorted_features:
            logger.info(f"   {feature}: {score:.4f}")

    def calibrate_model(self, model: xgb.XGBClassifier, X_test: pd.DataFrame,
                       y_test: pd.Series, n_bins: int = 10):
        """
        Calibrate model probabilities to match actual hit rates

        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
            n_bins: Number of calibration bins

        Returns:
            Calibration curve data
        """
        y_prob = model.predict_proba(X_test)[:, 1]

        # Get calibration curve
        prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=n_bins, strategy='quantile')

        logger.info(f"\n📈 Calibration Analysis (HR Model):")
        logger.info(f"   Predicted Prob | Actual Hit Rate | Sample Size")
        logger.info(f"   {'-'*50}")

        for pred, actual in zip(prob_pred, prob_true):
            logger.info(f"   {pred*100:.1f}%           | {actual*100:.1f}%         | -")

        # Calculate calibration error
        calibration_error = np.mean(np.abs(prob_true - prob_pred))
        logger.info(f"\n   Mean Calibration Error: {calibration_error:.4f}")

        return prob_true, prob_pred

    def train_all_models(self, start_date: str = '2023-04-01', end_date: str = '2024-11-01'):
        """
        Train all models (HR, Hit, SO)

        Args:
            start_date: Training start date
            end_date: Training end date
        """
        logger.info(f"🚀 Training all models ({start_date} to {end_date})...")
        logger.info("="*60)

        # Train HR model
        hr_model = self.train_hr_model(start_date, end_date)

        # Train Hit model
        hit_model = self.train_hit_model(start_date, end_date)

        # Train SO model
        so_model = self.train_so_model(start_date, end_date)

        logger.info("="*60)
        logger.info("✅ All models trained successfully!")

        return {
            'hr_model': hr_model,
            'hit_model': hit_model,
            'so_model': so_model
        }


if __name__ == "__main__":
    # Train all models
    trainer = ModelTrainer()

    # Note: This requires database with historical data
    # Uncomment when data is available:
    # models = trainer.train_all_models()

    logger.info("🎓 Model trainer initialized")
    logger.info("💡 To train models, run: python models/train.py")
    logger.info("   (Requires database with historical Statcast data)")
