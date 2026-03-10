#!/usr/bin/env python3
from pathlib import Path
import json
import logging
from typing import Dict, List, Tuple

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ARTIFACTS_DIR = Path(__file__).parent / 'artifacts'
ARTIFACTS_DIR.mkdir(exist_ok=True)


def _safe_roc_auc(y_true: pd.Series, proba: np.ndarray) -> float:
    return float(roc_auc_score(y_true, proba)) if y_true.nunique() > 1 else float('nan')


def _safe_average_precision(y_true: pd.Series, proba: np.ndarray) -> float:
    return float(average_precision_score(y_true, proba)) if y_true.nunique() > 1 else float('nan')


def _safe_brier(y_true: pd.Series, proba: np.ndarray) -> float:
    return float(brier_score_loss(y_true, proba)) if y_true.nunique() > 1 else float('nan')


def _class_weight(y: pd.Series) -> float:
    pos = float(y.sum())
    neg = float(len(y) - pos)
    return (neg / pos) if pos > 0 else 1.0


class ModelPipeline:

    def train_hr_model(self, df: pd.DataFrame) -> Dict:
        return self._train_pipeline(df, 'label', 'hr')

    def train_hit_model(self, df: pd.DataFrame) -> Dict:
        return self._train_pipeline(df, 'label_hit', 'hit')

    def train_so_model(self, df: pd.DataFrame) -> Dict:
        return self._train_pipeline(df, 'label_so', 'so')

    def _feature_columns(self, df: pd.DataFrame, label_col: str) -> List[str]:
        exclude_cols = {
            label_col, 'label', 'label_hit', 'label_so', 'game_date',
            'player_name', 'game_id', 'confidence_score', 'odds_decimal', 'pick_id',
            'hr_count', 'hit_count', 'batter_so', 'so_count', 'batter_hit',
            'avg_ev', 'barrel_rate.1', 'sweet_spot_percent.1', 'player_name.1', 'player_name.2',
            'actual_strikeouts', 'actual_innings_pitched', 'lineup_game_id',
            'starter_name', 'starter_name_normalized', 'starter_name_key', 'actual_starter_name',
        }
        return [
            col for col in df.columns
            if col not in exclude_cols
            and not col.startswith('label_')
            and pd.api.types.is_numeric_dtype(df[col])
        ]

    def _temporal_holdout_split(
        self, X: pd.DataFrame, y: pd.Series, dates: pd.Series, holdout_frac: float = 0.2
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
        unique_dates = sorted(pd.Series(dates).dropna().unique())
        if len(unique_dates) < 5:
            split_idx = max(int(len(X) * (1 - holdout_frac)), 1)
            return (
                X.iloc[:split_idx].copy(), X.iloc[split_idx:].copy(),
                y.iloc[:split_idx].copy(), y.iloc[split_idx:].copy(),
                dates.iloc[:split_idx].copy(), dates.iloc[split_idx:].copy(),
            )

        holdout_date_count = max(int(np.ceil(len(unique_dates) * holdout_frac)), 1)
        holdout_start = unique_dates[-holdout_date_count]
        train_mask = dates < holdout_start
        holdout_mask = dates >= holdout_start

        if train_mask.sum() == 0 or holdout_mask.sum() == 0:
            split_idx = max(int(len(X) * (1 - holdout_frac)), 1)
            return (
                X.iloc[:split_idx].copy(), X.iloc[split_idx:].copy(),
                y.iloc[:split_idx].copy(), y.iloc[split_idx:].copy(),
                dates.iloc[:split_idx].copy(), dates.iloc[split_idx:].copy(),
            )

        return (
            X.loc[train_mask].copy(), X.loc[holdout_mask].copy(),
            y.loc[train_mask].copy(), y.loc[holdout_mask].copy(),
            dates.loc[train_mask].copy(), dates.loc[holdout_mask].copy(),
        )

    def _train_eval_tail_split(
        self, X: pd.DataFrame, y: pd.Series, dates: pd.Series, eval_frac: float = 0.15
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        unique_dates = sorted(pd.Series(dates).dropna().unique())
        if len(unique_dates) < 4:
            split_idx = max(int(len(X) * (1 - eval_frac)), 1)
            return X.iloc[:split_idx].copy(), X.iloc[split_idx:].copy(), y.iloc[:split_idx].copy(), y.iloc[split_idx:].copy()

        eval_date_count = max(int(np.ceil(len(unique_dates) * eval_frac)), 1)
        eval_start = unique_dates[-eval_date_count]
        train_mask = dates < eval_start
        eval_mask = dates >= eval_start

        if train_mask.sum() == 0 or eval_mask.sum() == 0:
            split_idx = max(int(len(X) * (1 - eval_frac)), 1)
            return X.iloc[:split_idx].copy(), X.iloc[split_idx:].copy(), y.iloc[:split_idx].copy(), y.iloc[split_idx:].copy()

        return X.loc[train_mask].copy(), X.loc[eval_mask].copy(), y.loc[train_mask].copy(), y.loc[eval_mask].copy()

    def _fit_xgb(
        self, X_train: pd.DataFrame, y_train: pd.Series, X_eval: pd.DataFrame, y_eval: pd.Series
    ) -> XGBClassifier:
        model = XGBClassifier(
            max_depth=4,
            learning_rate=0.03,
            n_estimators=1000,
            min_child_weight=10,
            reg_lambda=1.5,
            scale_pos_weight=_class_weight(y_train),
            eval_metric='auc',
            early_stopping_rounds=50,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train, eval_set=[(X_eval, y_eval)], verbose=False)
        return model

    def _fit_lgb(
        self, X_train: pd.DataFrame, y_train: pd.Series, X_eval: pd.DataFrame, y_eval: pd.Series
    ):
        model = lgb.LGBMClassifier(
            max_depth=4,
            learning_rate=0.03,
            n_estimators=1000,
            min_child_samples=20,
            scale_pos_weight=_class_weight(y_train),
            random_state=42
        )
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_eval, y_eval)],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )
        return model

    def _metrics(self, y_true: pd.Series, proba: np.ndarray) -> Dict[str, float]:
        return {
            'roc_auc': _safe_roc_auc(y_true, proba),
            'average_precision': _safe_average_precision(y_true, proba),
            'brier_score': _safe_brier(y_true, proba),
        }

    def _train_pipeline(self, df: pd.DataFrame, label_col: str, name: str) -> Dict:
        logger.info(f"Training {name.upper()} model...")

        df = df.sort_values('game_date').reset_index(drop=True)
        feature_cols = self._feature_columns(df, label_col)

        mask = df[label_col].notna()
        X = df.loc[mask, feature_cols].copy()
        y = df.loc[mask, label_col].astype(int).copy()
        dates = pd.to_datetime(df.loc[mask, 'game_date'], errors='coerce')

        logger.info(f"Training rows after dropping null labels: {len(X)} / {len(df)}")

        X_train, X_holdout, y_train, y_holdout, train_dates, holdout_dates = self._temporal_holdout_split(X, y, dates)
        logger.info(
            "Temporal split (%s): train=%s rows, holdout=%s rows, train_dates=%s..%s, holdout_dates=%s..%s",
            name,
            len(X_train),
            len(X_holdout),
            train_dates.min().date() if not train_dates.empty else None,
            train_dates.max().date() if not train_dates.empty else None,
            holdout_dates.min().date() if not holdout_dates.empty else None,
            holdout_dates.max().date() if not holdout_dates.empty else None,
        )

        tscv = TimeSeriesSplit(n_splits=5)
        oof_xgb = np.full(len(X_train), np.nan)
        oof_lgb = np.full(len(X_train), np.nan)
        fold_summaries = []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train), start=1):
            X_tr = X_train.iloc[train_idx].copy()
            X_val = X_train.iloc[val_idx].copy()
            y_tr = y_train.iloc[train_idx].copy()
            y_val = y_train.iloc[val_idx].copy()

            medians = X_tr.median()
            X_tr_imp = X_tr.fillna(medians)
            X_val_imp = X_val.fillna(medians)

            xgb_model = self._fit_xgb(X_tr_imp, y_tr, X_val_imp, y_val)
            lgb_model = self._fit_lgb(X_tr_imp, y_tr, X_val_imp, y_val)

            xgb_proba = xgb_model.predict_proba(X_val_imp)[:, 1]
            lgb_proba = lgb_model.predict_proba(X_val_imp)[:, 1]

            oof_xgb[val_idx] = xgb_proba
            oof_lgb[val_idx] = lgb_proba

            fold_summary = {
                'fold': fold,
                'xgb': self._metrics(y_val, xgb_proba),
                'lgb': self._metrics(y_val, lgb_proba),
            }
            fold_summaries.append(fold_summary)
            logger.info(
                "Fold %s %s AUCs | XGB=%.4f LGB=%.4f",
                fold,
                name,
                fold_summary['xgb']['roc_auc'],
                fold_summary['lgb']['roc_auc'],
            )

        valid_oof = ~np.isnan(oof_xgb) & ~np.isnan(oof_lgb)
        meta_train_X = pd.DataFrame({'xgb': oof_xgb[valid_oof], 'lgb': oof_lgb[valid_oof]})
        meta_train_y = y_train.iloc[np.where(valid_oof)[0]].copy()

        if meta_train_y.nunique() < 2:
            logger.warning("Meta-training set for %s has one class; using DummyClassifier", name)
            meta_model = DummyClassifier(strategy='prior')
            meta_model.fit(meta_train_X, meta_train_y)
            cv_meta_metrics = {'roc_auc': float('nan'), 'average_precision': float('nan'), 'brier_score': float('nan')}
        else:
            meta_model = LogisticRegression(max_iter=1000)
            meta_model.fit(meta_train_X, meta_train_y)
            cv_meta_metrics = self._metrics(meta_train_y, meta_model.predict_proba(meta_train_X)[:, 1])

        X_base_train, X_base_eval, y_base_train, y_base_eval = self._train_eval_tail_split(X_train, y_train, train_dates)
        train_medians = X_base_train.median()
        X_base_train_imp = X_base_train.fillna(train_medians)
        X_base_eval_imp = X_base_eval.fillna(train_medians)
        X_holdout_imp = X_holdout.fillna(train_medians)

        final_xgb = self._fit_xgb(X_base_train_imp, y_base_train, X_base_eval_imp, y_base_eval)
        final_lgb = self._fit_lgb(X_base_train_imp, y_base_train, X_base_eval_imp, y_base_eval)

        holdout_xgb = final_xgb.predict_proba(X_holdout_imp)[:, 1]
        holdout_lgb = final_lgb.predict_proba(X_holdout_imp)[:, 1]
        holdout_meta_X = pd.DataFrame({'xgb': holdout_xgb, 'lgb': holdout_lgb})
        holdout_meta = meta_model.predict_proba(holdout_meta_X)[:, 1]

        holdout_metrics = {
            'xgb': self._metrics(y_holdout, holdout_xgb),
            'lgb': self._metrics(y_holdout, holdout_lgb),
            'meta': self._metrics(y_holdout, holdout_meta),
        }

        logger.info(
            "Holdout %s AUCs | XGB=%.4f LGB=%.4f META=%.4f",
            name,
            holdout_metrics['xgb']['roc_auc'],
            holdout_metrics['lgb']['roc_auc'],
            holdout_metrics['meta']['roc_auc'],
        )

        final_xgb.save_model(str(ARTIFACTS_DIR / f'{name}_xgb.json'))
        final_lgb.booster_.save_model(str(ARTIFACTS_DIR / f'{name}_lgb.txt'))
        joblib.dump(
            {
                'meta': meta_model,
                'features': feature_cols,
                'train_medians': train_medians.to_dict(),
                'training_metadata': {
                    'target': name,
                    'label_column': label_col,
                    'train_rows': int(len(X_train)),
                    'holdout_rows': int(len(X_holdout)),
                    'train_start_date': str(train_dates.min().date()) if not train_dates.empty else None,
                    'train_end_date': str(train_dates.max().date()) if not train_dates.empty else None,
                    'holdout_start_date': str(holdout_dates.min().date()) if not holdout_dates.empty else None,
                    'holdout_end_date': str(holdout_dates.max().date()) if not holdout_dates.empty else None,
                    'cv_folds': fold_summaries,
                    'cv_meta': cv_meta_metrics,
                    'holdout_metrics': holdout_metrics,
                }
            },
            str(ARTIFACTS_DIR / f'{name}_meta.pkl')
        )

        logger.info(f"Saved artifacts for {name}")

        return {
            'features': feature_cols,
            'train_rows': int(len(X_train)),
            'holdout_rows': int(len(X_holdout)),
            'train_positive_rate': round(float(y_train.mean()), 6),
            'holdout_positive_rate': round(float(y_holdout.mean()), 6),
            'train_start_date': str(train_dates.min().date()) if not train_dates.empty else None,
            'train_end_date': str(train_dates.max().date()) if not train_dates.empty else None,
            'holdout_start_date': str(holdout_dates.min().date()) if not holdout_dates.empty else None,
            'holdout_end_date': str(holdout_dates.max().date()) if not holdout_dates.empty else None,
            'cv_folds': fold_summaries,
            'cv_meta': cv_meta_metrics,
            'holdout_metrics': holdout_metrics,
            'auc': holdout_metrics['meta']['roc_auc'],
        }


if __name__ == '__main__':
    data_path = Path(__file__).parent.parent / 'data' / 'complete_dataset.csv'
    if not data_path.exists():
        logger.error(f"Dataset not found at {data_path}")
        raise SystemExit(1)

    df = pd.read_csv(data_path)
    df['game_date'] = pd.to_datetime(df['game_date'], errors='coerce')

    if 'label' not in df.columns:
        logger.error("No 'label' column in dataset")
        raise SystemExit(1)

    pipeline = ModelPipeline()
    results = {'generated_at': pd.Timestamp.utcnow().isoformat()}

    results['hr'] = pipeline.train_hr_model(df)
    logger.info(f"HR model trained: holdout META AUC={results['hr']['auc']:.4f}")

    if 'label_hit' in df.columns:
        results['hit'] = pipeline.train_hit_model(df)
        logger.info(f"Hit model trained: holdout META AUC={results['hit']['auc']:.4f}")
    else:
        logger.warning("label_hit column not found, skipping hit model training")

    if 'label_so' in df.columns and df['label_so'].sum() > 10:
        results['so'] = pipeline.train_so_model(df)
        logger.info(f"SO model trained: holdout META AUC={results['so']['auc']:.4f}")
    else:
        logger.warning("label_so column not found or insufficient data, skipping SO model training")

    with open(ARTIFACTS_DIR / 'training_results.json', 'w') as file_handle:
        json.dump(results, file_handle, indent=2, default=str)

    logger.info("Training complete")
