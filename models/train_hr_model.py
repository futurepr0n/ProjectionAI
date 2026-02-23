#!/usr/bin/env python3
"""
ProjectionAI - XGBoost Training
Train HR prediction model with proper features
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import joblib
import json
import logging
from datetime import datetime
from typing import Dict, List, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HRModelTrainer:
    """Train XGBoost model for HR prediction"""

    def __init__(self, data_path: str = 'data/complete_dataset.csv'):
        self.data_path = data_path
        self.model = None
        self.scaler = None
        self.imputer = None
        self.feature_names = None

    def load_data(self) -> pd.DataFrame:
        """Load the complete dataset"""
        df = pd.read_csv(self.data_path)
        logger.info(f"✅ Loaded {len(df):,} samples from {self.data_path}")
        return df

    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features for training
        
        Features:
        - Hitter metrics: barrel_rate, exit_velocity_avg, hard_hit_percent, sweet_spot_percent
        - Pitcher metrics: pitcher_era, pitcher_hr_per_9, pitcher_k_per_9, pitcher_whip
        - Game context: is_home, confidence_score
        """
        # Define features
        feature_cols = [
            # Hitter Statcast metrics
            'barrel_rate',
            'exit_velocity_avg',
            'hard_hit_percent',
            'sweet_spot_percent',
            'swing_optimization_score',
            'swing_attack_angle',
            'swing_bat_speed',
            
            # Pitcher quality metrics
            'pitcher_era',
            'pitcher_hr_per_9',
            'pitcher_k_per_9',
            'pitcher_whip',
            
            # Play-by-play pitcher features
            'k_rate_pct',
            'hr_rate_pct',
            'fly_ball_pct',
            
            # Context
            'is_home',
            'confidence_score',
            'odds_decimal'
        ]

        # Filter to available columns
        available_cols = [col for col in feature_cols if col in df.columns]
        logger.info(f"\n🎯 Using {len(available_cols)} features:")
        for col in available_cols:
            available = df[col].notna().sum()
            logger.info(f"   {col}: {available:,}/{len(df):,} ({available/len(df)*100:.1f}%)")

        X = df[available_cols].copy()
        y = df['label'].copy()

        # Store feature names
        self.feature_names = available_cols

        return X, y

    def train_model(self, X: pd.DataFrame, y: pd.Series, 
                    test_size: float = 0.2) -> Dict:
        """
        Train XGBoost model with cross-validation
        """
        logger.info(f"\n🚀 Training XGBoost model...")

        # Handle missing values
        self.imputer = SimpleImputer(strategy='median')
        X_imputed = pd.DataFrame(
            self.imputer.fit_transform(X),
            columns=X.columns
        )

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_imputed, y, 
            test_size=test_size, 
            random_state=42,
            stratify=y
        )

        logger.info(f"   Training samples: {len(X_train):,}")
        logger.info(f"   Test samples: {len(X_test):,}")
        logger.info(f"   Positive class: {y_train.sum():,} ({y_train.mean()*100:.1f}%)")

        # Calculate class weight (HR is rare)
        scale_pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()
        logger.info(f"   Class weight: {scale_pos_weight:.2f}")

        # Initialize XGBoost
        self.model = xgb.XGBClassifier(
            max_depth=6,
            learning_rate=0.1,
            n_estimators=200,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=1,
            min_child_weight=5,
            scale_pos_weight=scale_pos_weight,
            objective='binary:logistic',
            eval_metric='logloss',
            random_state=42,
            n_jobs=-1,
            use_label_encoder=False
        )

        # Train
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )

        # Evaluate
        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)[:, 1]

        metrics = self._calculate_metrics(y_test, y_pred, y_prob)

        # Cross-validation
        cv_scores = cross_val_score(
            self.model, X_imputed, y, 
            cv=5, 
            scoring='roc_auc'
        )
        metrics['cv_roc_auc_mean'] = cv_scores.mean()
        metrics['cv_roc_auc_std'] = cv_scores.std()

        logger.info(f"\n📊 Cross-validation ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

        # Feature importance
        self._log_feature_importance()

        return metrics

    def _calculate_metrics(self, y_true: pd.Series, y_pred: np.ndarray, 
                          y_prob: np.ndarray) -> Dict:
        """Calculate and log metrics"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_true, y_prob)
        }

        logger.info(f"\n📊 Model Performance:")
        logger.info(f"   Accuracy:  {metrics['accuracy']:.4f}")
        logger.info(f"   Precision: {metrics['precision']:.4f}")
        logger.info(f"   Recall:    {metrics['recall']:.4f}")
        logger.info(f"   F1 Score:  {metrics['f1']:.4f}")
        logger.info(f"   ROC-AUC:   {metrics['roc_auc']:.4f}")

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        logger.info(f"\n📈 Confusion Matrix:")
        logger.info(f"   TN: {cm[0,0]:,}  FP: {cm[0,1]:,}")
        logger.info(f"   FN: {cm[1,0]:,}  TP: {cm[1,1]:,}")

        # Classification report
        logger.info(f"\n📋 Classification Report:")
        print(classification_report(y_true, y_pred, target_names=['No HR', 'HR']))

        return metrics

    def _log_feature_importance(self):
        """Log feature importance"""
        importance = self.model.feature_importances_
        feature_imp = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)

        logger.info(f"\n🎯 Top 10 Feature Importance:")
        for i, row in feature_imp.head(10).iterrows():
            logger.info(f"   {row['feature']}: {row['importance']:.4f}")

        return feature_imp

    def calibrate_predictions(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """
        Analyze calibration of predictions
        """
        logger.info(f"\n📊 Calibration Analysis:")

        X_imputed = pd.DataFrame(
            self.imputer.transform(X),
            columns=X.columns
        )

        y_prob = self.model.predict_proba(X_imputed)[:, 1]

        # Bin predictions
        bins = np.linspace(0, 1, 11)
        bin_indices = np.digitize(y_prob, bins)

        calibration_data = []
        for i in range(1, 11):
            mask = bin_indices == i
            if mask.sum() > 0:
                predicted_prob = y_prob[mask].mean()
                actual_rate = y[mask].mean()
                count = mask.sum()
                calibration_data.append({
                    'bin': i,
                    'predicted_prob': predicted_prob,
                    'actual_rate': actual_rate,
                    'count': count
                })

        logger.info(f"   {'Bin':<5} {'Predicted':>12} {'Actual':>12} {'Count':>8}")
        logger.info(f"   {'-'*45}")
        for data in calibration_data:
            logger.info(f"   {data['bin']:<5} {data['predicted_prob']*100:>11.1f}% {data['actual_rate']*100:>11.1f}% {data['count']:>8,}")

        return calibration_data

    def save_model(self, model_path: str = 'models/hr_model.json'):
        """Save trained model"""
        self.model.save_model(model_path)
        logger.info(f"✅ Model saved to {model_path}")

        # Save feature names
        with open(model_path.replace('.json', '_features.json'), 'w') as f:
            json.dump(self.feature_names, f, indent=2)

        # Save imputer
        joblib.dump(self.imputer, model_path.replace('.json', '_imputer.pkl'))

        logger.info(f"✅ Feature names and imputer saved")

    def run_full_training(self) -> Dict:
        """
        Run complete training pipeline
        """
        # Load data
        df = self.load_data()

        # Prepare features
        X, y = self.prepare_features(df)

        # Train model
        metrics = self.train_model(X, y)

        # Calibrate
        calibration = self.calibrate_predictions(X, y)

        # Save model
        self.save_model()

        return {
            'metrics': metrics,
            'calibration': calibration,
            'feature_count': len(self.feature_names),
            'sample_count': len(df)
        }


if __name__ == "__main__":
    trainer = HRModelTrainer()

    results = trainer.run_full_training()

    logger.info(f"\n✅ Training complete!")
    logger.info(f"   ROC-AUC: {results['metrics']['roc_auc']:.4f}")
    logger.info(f"   Features: {results['feature_count']}")
    logger.info(f"   Samples: {results['sample_count']:,}")
