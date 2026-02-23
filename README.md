# ProjectionAI - MLB Betting Projection System

## Overview

ProjectionAI is a machine learning-based betting projection system for MLB home run predictions. The system uses XGBoost models to predict home run probability based on player matchup data, pitcher statistics, and game context.

## Current Status

- **Phase 1 Complete**: Initial XGBoost model trained
- **Model Performance**: ROC AUC 0.63, F1 Score 0.26
- **Best Strategy**: Min Edge 8% (-71.2% ROI, +28.6% over baseline)
- **Next Steps**: Add missing critical features (weather, travel, stadium dimensions)

## Project Structure

```
ProjectionAI/
├── data/                          # Training data and results
│   ├── complete_dataset.csv         # Labeled training data
│   ├── comprehensive_features.csv  # Feature engineering output
│   └── matchup_model_results.json # Model performance metrics
├── matchup_model_v3.py            # Phase 1 ML model
├── generate_report.py              # Results dashboard generator
├── start_server.py                # Local dashboard server
├── README.md                      # This file
└── .gitignore                     # Git ignore patterns
```

## Key Findings (Phase 1)

### Model Performance
| Metric | Value |
|--------|-------|
| ROC AUC | 0.6305 |
| F1 Score | 0.2607 |
| Precision | 16.9% |
| Recall | 57.0% |
| Best ROI | -71.2% (Min Edge 8%) |

### Top Predictive Features
1. **pitcher_hr_per_9** (15.1%) - Home runs allowed per 9 innings
2. **odds_decimal** (14.7%) - Bookmaker odds (data leakage warning)
3. **confidence_score** (11.1%) - Existing confidence prediction
4. **swing_optimization_score** (10.2%) - Swing mechanics
5. **swing_attack_angle** (10.1%) - Launch angle

### Missing Critical Features
- Weather data (temperature, wind, humidity)
- Travel distance and rest days
- Stadium dimensions and park factors
- Umpire strike zone analysis

## Getting Started

### Prerequisites
- Python 3.8+
- pandas, numpy, scikit-learn, xgboost
- PostgreSQL (for data storage)

### Installation

```bash
cd ~/Development/ProjectionAI
pip install -r requirements.txt  # If you create one
```

### Running the Model

```bash
# Train the matchup prediction model
python3 matchup_model_v3.py

# Generate results report
python3 generate_report.py

# Start local dashboard
python3 start_server.py
# Open http://localhost:8000/results_report.html
```

## Database Schema

Tables on `192.168.1.23:5432/baseball_migration_test`:
- `hitter_exit_velocity` - Exit velocity metrics (1,017 records)
- `custom_pitcher_2025` - Pitcher spin rates and stats (757 records)
- `custom_batter_2025` - Hitter x-stats and swing speed (609 records)
- `daily_batted_ball_tracking` - Daily batted ball trends (32,126 records)

## Roadmap

### Phase 1 ✅ Complete
- [x] Data collection and feature engineering
- [x] Baseline XGBoost model
- [x] Betting strategy simulation
- [x] Results dashboard

### Phase 2: Feature Enhancement
- [ ] Add historical weather data
- [ ] Calculate travel distance and rest days
- [ ] Add stadium dimension reference
- [ ] Implement umpire zone analysis

### Phase 3: Model Improvement
- [ ] Remove data leakage features
- [ ] Implement cross-validation
- [ ] Hyperparameter tuning
- [ ] Ensemble methods

### Phase 4: Production
- [ ] Real-time prediction pipeline
- [ ] Automated data ingestion
- [ ] Live betting dashboard
- [ ] Bankroll tracking

## Results Dashboard

The project includes an interactive HTML dashboard showing:
- Model performance metrics
- Betting strategy comparison
- Feature importance visualization
- Confusion matrix analysis

Access at: `http://localhost:8000/results_report.html`

## License

Proprietary - Not for commercial use.

## Contact

For questions or updates, contact Mark.

---

**Note**: This is an experimental system. Do not make real betting decisions based on these predictions without thorough validation and understanding of the risks involved.
