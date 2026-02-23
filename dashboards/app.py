#!/usr/bin/env python3
"""
ProjectionAI - Updated Dashboard with Date Listing and Results
"""

from flask import Flask, render_template, jsonify, request
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import json
import psycopg2
from psycopg2.extras import RealDictCursor
import logging
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)


class PredictionEngine:
    """Engine for generating live predictions"""

    def __init__(self):
        self.model = None
        self.imputer = None
        self.feature_names = None
        self.db_conn = None
        self.load_model()
        self.connect_db()

    def load_model(self, model_path=None):
        """Load trained model"""
        if model_path is None:
            # Use absolute path
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path = os.path.join(base_dir, 'models', 'hr_model.json')
        
        self.model = xgb.XGBClassifier()
        self.model.load_model(model_path)

        with open(model_path.replace('.json', '_features.json'), 'r') as f:
            self.feature_names = json.load(f)

        self.imputer = joblib.load(model_path.replace('.json', '_imputer.pkl'))
        logger.info(f"✅ Model loaded with {len(self.feature_names)} features")

    def connect_db(self):
        """Connect to remote database"""
        try:
            self.db_conn = psycopg2.connect(
                host='192.168.1.23',
                port=5432,
                database='baseball_migration_test',
                user='postgres',
                password='korn5676'
            )
            logger.info("✅ Database connected")
        except Exception as e:
            logger.error(f"❌ Database connection failed: {e}")

    def get_available_dates(self) -> List[Dict]:
        """Get all dates with Hellraiser picks"""
        if not self.db_conn:
            return []

        try:
            cursor = self.db_conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute("""
                SELECT 
                    analysis_date,
                    COUNT(*) as total_picks,
                    COUNT(CASE WHEN confidence_score >= 85 THEN 1 END) as strong_buys,
                    COUNT(CASE WHEN confidence_score >= 70 AND confidence_score < 85 THEN 1 END) as buys
                FROM hellraiser_picks
                GROUP BY analysis_date
                ORDER BY analysis_date DESC
            """)
            dates = cursor.fetchall()
            # Convert date to ISO string format
            for d in dates:
                if d['analysis_date']:
                    d['date'] = d['analysis_date'].isoformat() if hasattr(d['analysis_date'], 'isoformat') else str(d['analysis_date'])
            return [dict(d) for d in dates]
        except Exception as e:
            logger.error(f"❌ Error getting dates: {e}")
            return []

    def get_hr_results_for_date(self, target_date: date) -> Dict:
        """Get actual HR results for a date from play-by-play"""
        if not self.db_conn:
            return {}

        try:
            cursor = self.db_conn.cursor(cursor_factory=RealDictCursor)
            # Get HR hitters - play_by_play only has last names
            cursor.execute("""
                SELECT 
                    pp.batter,
                    pp.game_id,
                    g.home_team,
                    g.away_team,
                    COUNT(*) as hr_count
                FROM play_by_play_plays pp
                JOIN games g ON pp.game_id = g.game_id
                WHERE g.game_date = %s
                  AND pp.play_result = 'Home Run'
                GROUP BY pp.batter, pp.game_id, g.home_team, g.away_team
            """, (target_date,))
            results = cursor.fetchall()
            
            # Create lookup by last name (play_by_play only has last names)
            hr_by_lastname = {}
            for r in results:
                lastname = r['batter'].strip().lower().split()[-1] if r['batter'] else ''
                if lastname:
                    hr_by_lastname[lastname] = {
                        'hr_count': r['hr_count'],
                        'game_id': r['game_id'],
                        'team': r['home_team'] if r['home_team'] else r['away_team']
                    }
            
            return hr_by_lastname
        except Exception as e:
            logger.error(f"❌ Error getting HR results: {e}")
            return {}

    def get_hellraiser_picks(self, analysis_date: date) -> List[Dict]:
        """Get Hellraiser picks for a date"""
        if not self.db_conn:
            return []

        try:
            cursor = self.db_conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute("""
                SELECT * FROM hellraiser_picks
                WHERE analysis_date = %s
                ORDER BY confidence_score DESC
            """, (analysis_date,))
            picks = cursor.fetchall()
            return [dict(p) for p in picks]
        except Exception as e:
            logger.error(f"❌ Error getting picks: {e}")
            return []

    def predict(self, features: Dict) -> Dict:
        """Generate prediction for a player"""
        X = pd.DataFrame([features])
        X = X[self.feature_names]

        X_imputed = pd.DataFrame(
            self.imputer.transform(X),
            columns=X.columns
        )

        prob = self.model.predict_proba(X_imputed)[0, 1]

        if prob >= 0.85:
            signal = 'STRONG_BUY'
        elif prob >= 0.70:
            signal = 'BUY'
        elif prob >= 0.55:
            signal = 'MODERATE'
        elif prob >= 0.40:
            signal = 'AVOID'
        else:
            signal = 'STRONG_SELL'

        odds = features.get('odds_decimal', 0)
        if odds and float(odds) > 1:
            implied_prob = 1 / float(odds)
            edge = prob - implied_prob
        else:
            implied_prob = 0
            edge = 0

        return {
            'probability': float(prob),
            'signal': signal,
            'implied_probability': float(implied_prob),
            'edge': float(edge),
            'edge_pct': float(edge * 100)
        }

    def generate_daily_predictions_with_results(self, target_date: date) -> Dict:
        """Generate predictions for a date with actual results"""
        picks = self.get_hellraiser_picks(target_date)
        hr_results = self.get_hr_results_for_date(target_date)

        predictions = []
        total_hrs = 0
        strong_buy_hits = 0
        strong_buy_count = 0
        buy_hits = 0
        buy_count = 0

        for pick in picks:
            features = {
                'barrel_rate': pick.get('barrel_rate'),
                'exit_velocity_avg': pick.get('exit_velocity_avg'),
                'hard_hit_percent': pick.get('hard_hit_percent'),
                'sweet_spot_percent': pick.get('sweet_spot_percent'),
                'swing_optimization_score': pick.get('swing_optimization_score'),
                'swing_attack_angle': pick.get('swing_attack_angle'),
                'swing_bat_speed': pick.get('swing_bat_speed'),
                'pitcher_era': pick.get('pitcher_era', 4.5),
                'pitcher_hr_per_9': pick.get('pitcher_hr_per_9', 1.2),
                'pitcher_k_per_9': pick.get('pitcher_k_per_9', 8.0),
                'pitcher_whip': pick.get('pitcher_whip', 1.3),
                'k_rate_pct': pick.get('k_rate_pct', 20),
                'hr_rate_pct': pick.get('hr_rate_pct', 3),
                'fly_ball_pct': pick.get('fly_ball_pct', 35),
                'is_home': pick.get('is_home', False),
                'confidence_score': pick.get('confidence_score', 50),
                'odds_decimal': pick.get('odds_decimal', 5.0)
            }

            pred = self.predict(features)

            # Check if this player hit a HR - match by last name
            player_name = pick.get('player_name', '')
            lastname = player_name.strip().lower().split()[-1] if player_name else ''
            
            hr_info = hr_results.get(lastname)
            did_hr = hr_info is not None
            hr_count = hr_info['hr_count'] if hr_info else 0

            if did_hr:
                total_hrs += 1

            # Track by signal
            if pred['signal'] == 'STRONG_BUY':
                strong_buy_count += 1
                if did_hr:
                    strong_buy_hits += 1
            elif pred['signal'] == 'BUY':
                buy_count += 1
                if did_hr:
                    buy_hits += 1

            prediction = {
                'player_name': player_name,
                'team': pick.get('team'),
                'opponent': pick.get('pitcher_name'),
                'venue': pick.get('venue'),
                'is_home': pick.get('is_home'),
                **pred,
                'hellraiser_confidence': float(pick['confidence_score']) if pick.get('confidence_score') else None,
                'hellraiser_classification': pick.get('classification'),
                'odds_decimal': float(pick['odds_decimal']) if pick.get('odds_decimal') else None,
                'actual_hr': did_hr,
                'actual_hr_count': hr_count
            }

            predictions.append(prediction)

        predictions.sort(key=lambda x: x['probability'], reverse=True)

        stats = {
            'total_picks': len(predictions),
            'total_hrs': total_hrs,
            'strong_buy_count': strong_buy_count,
            'strong_buy_hits': strong_buy_hits,
            'strong_buy_hit_rate': (strong_buy_hits / strong_buy_count * 100) if strong_buy_count > 0 else 0,
            'buy_count': buy_count,
            'buy_hits': buy_hits,
            'buy_hit_rate': (buy_hits / buy_count * 100) if buy_count > 0 else 0,
        }

        return {
            'date': target_date.isoformat(),
            'predictions': predictions,
            'stats': stats,
            'hr_hitters': list(hr_results.keys())
        }


# Initialize prediction engine
engine = PredictionEngine()


@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('dashboard.html')


@app.route('/api/dates')
def get_available_dates():
    """Get all available dates with picks"""
    dates = engine.get_available_dates()
    return jsonify({
        'total_dates': len(dates),
        'date_range': {
            'earliest': dates[-1]['analysis_date'] if dates else None,
            'latest': dates[0]['analysis_date'] if dates else None
        },
        'dates': dates
    })


@app.route('/api/predictions/<date_str>')
def get_predictions_for_date(date_str):
    """Get predictions for a specific date with results"""
    try:
        target_date = datetime.strptime(date_str, '%Y-%m-%d').date()
        data = engine.generate_daily_predictions_with_results(target_date)

        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/predictions/today')
def get_today_predictions():
    """Get today's predictions (will be empty if no Hellraiser picks)"""
    today = date.today()
    data = engine.generate_daily_predictions_with_results(today)
    return jsonify(data)


@app.route('/api/model/stats')
def get_model_stats():
    """Get model statistics"""
    return jsonify({
        'model_type': 'XGBoost',
        'features': len(engine.feature_names),
        'roc_auc': 0.9458,
        'strong_buy_hit_rate': 84.4,
        'buy_hit_rate': 49.4,
        'training_samples': 11462,
        'data_range': '2025-06-23 to 2025-09-01',
        'available_dates': 62
    })


@app.route('/api/results/<date_str>')
def get_results_for_date(date_str):
    """Get actual HR results for a date"""
    try:
        target_date = datetime.strptime(date_str, '%Y-%m-%d').date()
        results = engine.get_hr_results_for_date(target_date)

        return jsonify({
            'date': date_str,
            'total_hrs': len(results),
            'hr_hitters': results
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/master-list')
def master_list():
    """Master list page with all predictions and filters"""
    return render_template('master_list.html')


@app.route('/api/predictions/all')
def get_all_predictions():
    """Get all predictions across all dates with results"""
    dates = engine.get_available_dates()
    all_predictions = []
    
    for d in dates:
        try:
            date_str = d.get('analysis_date') or d.get('date', '')
            if not date_str:
                continue
                
            if isinstance(date_str, str):
                if 'T' in date_str:
                    date_str = date_str.split('T')[0]
                target_date = datetime.strptime(date_str, '%Y-%m-%d').date()
            else:
                target_date = date_str
            
            data = engine.generate_daily_predictions_with_results(target_date)
            
            for pred in data.get('predictions', []):
                pred['analysis_date'] = target_date.isoformat()
                all_predictions.append(pred)
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            continue
    
    total_profit = 0
    total_bets = 0
    hits = 0
    
    for pred in all_predictions:
        if pred.get('actual_hr') is not None:
            total_bets += 1
            odds = pred.get('odds_decimal')
            if pred.get('actual_hr') == True:
                hits += 1
                if odds and odds > 1:
                    total_profit += (odds - 1)
                else:
                    total_profit += 0
            else:
                total_profit -= 1
    
    hit_rate = (hits / total_bets * 100) if total_bets > 0 else 0
    roi = (total_profit / total_bets * 100) if total_bets > 0 else 0
    
    return jsonify({
        'total_predictions': len(all_predictions),
        'predictions': all_predictions,
        'overall_stats': {
            'hit_rate': round(hit_rate, 1),
            'roi': round(roi, 1)
        }
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002, debug=True)
