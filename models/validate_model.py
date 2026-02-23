#!/usr/bin/env python3
"""
ProjectionAI - Validation Framework
Compare model predictions to Hellraiser baseline
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import json
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report
)
import logging
from typing import Dict, List, Tuple
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelValidator:
    """Validate and compare models"""

    def __init__(self):
        self.model = None
        self.imputer = None
        self.feature_names = None

    def load_model(self, model_path: str = 'models/hr_model.json'):
        """Load trained model"""
        self.model = xgb.XGBClassifier()
        self.model.load_model(model_path)

        # Load feature names
        with open(model_path.replace('.json', '_features.json'), 'r') as f:
            self.feature_names = json.load(f)

        # Load imputer
        self.imputer = joblib.load(model_path.replace('.json', '_imputer.pkl'))

        logger.info(f"✅ Loaded model from {model_path}")
        logger.info(f"   Features: {len(self.feature_names)}")

    def validate_on_dataset(self, data_path: str = 'data/complete_dataset.csv') -> Dict:
        """
        Validate model on full dataset
        """
        # Load data
        df = pd.read_csv(data_path)
        logger.info(f"\n📊 Validating on {len(df):,} samples")

        # Prepare features
        X = df[self.feature_names].copy()
        y = df['label'].copy()

        # Impute missing values
        X_imputed = pd.DataFrame(
            self.imputer.transform(X),
            columns=X.columns
        )

        # Get predictions
        y_prob = self.model.predict_proba(X_imputed)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, zero_division=0),
            'recall': recall_score(y, y_pred, zero_division=0),
            'f1': f1_score(y, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y, y_prob)
        }

        logger.info(f"\n📈 Model Performance on Full Dataset:")
        logger.info(f"   Accuracy:  {metrics['accuracy']:.4f}")
        logger.info(f"   Precision: {metrics['precision']:.4f}")
        logger.info(f"   Recall:    {metrics['recall']:.4f}")
        logger.info(f"   F1 Score:  {metrics['f1']:.4f}")
        logger.info(f"   ROC-AUC:   {metrics['roc_auc']:.4f}")

        return metrics

    def compare_to_hellraiser(self, data_path: str = 'data/complete_dataset.csv') -> Dict:
        """
        Compare XGBoost model to Hellraiser baseline
        """
        df = pd.read_csv(data_path)

        # Prepare features
        X = df[self.feature_names].copy()
        y = df['label'].copy()

        # Impute
        X_imputed = pd.DataFrame(
            self.imputer.transform(X),
            columns=X.columns
        )

        # XGBoost predictions
        xgb_prob = self.model.predict_proba(X_imputed)[:, 1]
        df['xgb_prob'] = xgb_prob

        # Hellraiser predictions (use confidence_score as proxy)
        # Normalize confidence to 0-1 range
        if 'confidence_score' in df.columns:
            df['hellraiser_prob'] = df['confidence_score'] / 100.0
        else:
            df['hellraiser_prob'] = 0.5

        # Compare at different thresholds
        results = {}

        for threshold in [0.5, 0.6, 0.7, 0.8, 0.85, 0.9]:
            xgb_pred = (xgb_prob >= threshold).astype(int)
            hr_pred = (df['hellraiser_prob'] >= threshold).astype(int)

            xgb_positive = xgb_pred.sum()
            hr_positive = hr_pred.sum()

            # Calculate hit rates
            if xgb_positive > 0:
                xgb_hit_rate = y[xgb_pred == 1].mean() * 100
                xgb_precision = precision_score(y, xgb_pred, zero_division=0)
            else:
                xgb_hit_rate = 0
                xgb_precision = 0

            if hr_positive > 0:
                hr_hit_rate = y[hr_pred == 1].mean() * 100
                hr_precision = precision_score(y, hr_pred, zero_division=0)
            else:
                hr_hit_rate = 0
                hr_precision = 0

            results[threshold] = {
                'xgb_predictions': xgb_positive,
                'xgb_hit_rate': xgb_hit_rate,
                'xgb_precision': xgb_precision,
                'hr_predictions': hr_positive,
                'hr_hit_rate': hr_hit_rate,
                'hr_precision': hr_precision,
                'improvement': xgb_hit_rate - hr_hit_rate
            }

        logger.info(f"\n🏆 Model Comparison (XGBoost vs Hellraiser):")
        logger.info(f"   {'Threshold':<12} {'XGB Preds':>12} {'XGB Hit%':>12} {'HR Preds':>12} {'HR Hit%':>12} {'Improvement':>12}")
        logger.info(f"   {'-'*80}")

        for threshold, data in sorted(results.items()):
            logger.info(f"   {threshold:<12.2f} {data['xgb_predictions']:>12,} {data['xgb_hit_rate']:>11.1f}% {data['hr_predictions']:>12,} {data['hr_hit_rate']:>11.1f}% {data['improvement']:>+11.1f}%")

        return results

    def analyze_by_confidence_tier(self, data_path: str = 'data/complete_dataset.csv') -> Dict:
        """
        Analyze predictions by confidence tier
        """
        df = pd.read_csv(data_path)

        X = df[self.feature_names].copy()
        X_imputed = pd.DataFrame(
            self.imputer.transform(X),
            columns=X.columns
        )

        y_prob = self.model.predict_proba(X_imputed)[:, 1]
        df['xgb_prob'] = y_prob

        # Define tiers
        tiers = [
            ('STRONG_BUY', 0.85, 1.0),
            ('BUY', 0.70, 0.85),
            ('MODERATE', 0.55, 0.70),
            ('AVOID', 0.40, 0.55),
            ('STRONG_SELL', 0.0, 0.40)
        ]

        logger.info(f"\n📊 Analysis by Confidence Tier:")
        logger.info(f"   {'Tier':<15} {'Count':>8} {'Actual HR':>12} {'Hit Rate':>12} {'Avg Prob':>12}")
        logger.info(f"   {'-'*65}")

        tier_results = {}
        for tier_name, low, high in tiers:
            mask = (y_prob >= low) & (y_prob < high)
            count = mask.sum()

            if count > 0:
                actual_hr = df.loc[mask, 'label'].sum()
                hit_rate = actual_hr / count * 100
                avg_prob = y_prob[mask].mean()
            else:
                actual_hr = 0
                hit_rate = 0
                avg_prob = 0

            tier_results[tier_name] = {
                'count': count,
                'actual_hr': actual_hr,
                'hit_rate': hit_rate,
                'avg_prob': avg_prob
            }

            logger.info(f"   {tier_name:<15} {count:>8,} {actual_hr:>12,} {hit_rate:>11.1f}% {avg_prob:>11.1%}")

        return tier_results

    def generate_betting_recommendations(self, data_path: str = 'data/complete_dataset.csv',
                                         min_edge: float = 0.05) -> pd.DataFrame:
        """
        Generate betting recommendations with edge calculation
        """
        df = pd.read_csv(data_path)

        X = df[self.feature_names].copy()
        X_imputed = pd.DataFrame(
            self.imputer.transform(X),
            columns=X.columns
        )

        y_prob = self.model.predict_proba(X_imputed)[:, 1]
        df['xgb_prob'] = y_prob

        # Calculate implied probability from odds
        if 'odds_decimal' in df.columns:
            # Handle missing odds
            df['implied_prob'] = 1 / df['odds_decimal'].fillna(10)
            df['edge'] = df['xgb_prob'] - df['implied_prob']
        else:
            df['implied_prob'] = 0.5
            df['edge'] = 0

        # Filter for positive edge
        recommendations = df[df['edge'] >= min_edge].copy()
        recommendations = recommendations.sort_values('edge', ascending=False)

        logger.info(f"\n💰 Betting Recommendations (min edge: {min_edge*100:.0f}%):")
        logger.info(f"   Total recommendations: {len(recommendations):,}")
        logger.info(f"   Average edge: {recommendations['edge'].mean()*100:.1f}%")

        # Show top 10
        logger.info(f"\n   Top 10 Picks:")
        for i, (_, row) in enumerate(recommendations.head(10).iterrows(), 1):
            logger.info(f"   {i}. {row.get('player_name', 'Unknown')} - "
                       f"Prob: {row['xgb_prob']:.1%}, "
                       f"Edge: {row['edge']:+.1%}, "
                       f"Odds: {row.get('odds_decimal', 0):.2f}")

        return recommendations

    def run_full_validation(self) -> Dict:
        """Run complete validation pipeline"""
        logger.info("🚀 Running full model validation...")

        # Load model
        self.load_model()

        # Validate
        metrics = self.validate_on_dataset()

        # Compare to Hellraiser
        comparison = self.compare_to_hellraiser()

        # Analyze by tier
        tiers = self.analyze_by_confidence_tier()

        # Generate recommendations
        recommendations = self.generate_betting_recommendations()

        return {
            'metrics': metrics,
            'comparison': comparison,
            'tiers': tiers,
            'recommendations_count': len(recommendations)
        }


if __name__ == "__main__":
    validator = ModelValidator()
    results = validator.run_full_validation()

    logger.info(f"\n✅ Validation complete!")
