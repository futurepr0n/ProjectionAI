# ProjectionAI - Project Plan

**Next-Generation Sports Betting Projection System**

---

## 🎯 Mission

Build a profitable, data-driven sports betting projection dashboard that combines machine learning, ensemble methods, and disciplined bankroll management to achieve consistent ROI.

---

## 📊 Current State Analysis

### What Works (From Existing System)
- ✅ Elite power hitter identification (Schwarber, Ohtani, Harper = 100% hit rate)
- ✅ Dashboard infrastructure (live at localhost:5001)
- ✅ Full MLB coverage (30 teams, 9 batters each)
- ✅ Three prediction types (HR, Hit, Strikeout)
- ✅ Statcast metrics integration (barrel rate, EV95+, sweet spot)
- ✅ Historical validation framework

### What Needs Improvement
- ❌ HR hit rate 21% (below random baseline of 33%)
- ❌ Too many predictions (433 in 2 days, should be ~150)
- ❌ Rule-based algorithm (no ML model)
- ❌ No odds integration (can't calculate actual ROI)
- ❌ No bankroll management (Kelly criterion not implemented)
- ❌ No real-time updates (manual data entry)
- ❌ Confidence thresholds not calibrated

---

## 🚀 Vision: ProjectionAI 1.0

### Core Features

1. **Machine Learning Prediction Engine**
   - Replace rule-based algorithm with gradient boosting model
   - Train on historical Statcast data + odds + results
   - Feature engineering: barrel rate, EV95+, sweet spot, pitcher quality, park factors
   - Confidence calibration: predicted probability = actual hit rate

2. **Ensemble Prediction System**
   - HR Model: XGBoost classifier (target: 25-30% hit rate at 85+ confidence)
   - Hit Model: XGBoost classifier (target: 50-55% hit rate at 75+ confidence)
   - Strikeout Model: XGBoost classifier (target: 60-65% hit rate at 80+ confidence)
   - Meta-learner: Combines all three signals into edge calculation

3. **Real-Time Odds Integration**
   - Scrape opening/closing lines from major books
   - Calculate implied probability
   - Compare to our predicted edge
   - Only bet when edge > threshold (e.g., 5%)

4. **Bankroll Management (Kelly Criterion)**
   - Calculate optimal bet size based on edge and odds
   - Fractional Kelly (0.25-0.5) for risk management
   - Track bankroll growth over time
   - Position sizing limits (max 2% per bet)

5. **Live Dashboard**
   - Today's predictions with confidence and odds
   - Top STRONG_BUY plays (highest edge)
   - Bankroll tracking and ROI visualization
   - Historical performance by bet type
   - Live game status and results

6. **Backtesting Engine**
   - Test strategies on historical data (2023-2025)
   - Calculate CLV (Closing Line Value)
   - Track performance by signal tier
   - Validate model calibration

---

## 🏗️ Architecture

### Data Pipeline
```
Statcast API → Database → Feature Store → ML Models → Predictions → Odds API → Edge Calculation → Dashboard → Betting
```

### Components

1. **Data Layer** (`/data`)
   - `statcast_loader.py` - Pull Statcast data from MLB API
   - `odds_scraper.py` - Scrape odds from DraftKings/FanDuel
   - `database.py` - PostgreSQL database schema
   - `feature_store.py` - Feature engineering pipeline

2. **Model Layer** (`/models`)
   - `hr_model.py` - XGBoost HR prediction model
   - `hit_model.py` - XGBoost Hit prediction model
   - `so_model.py` - XGBoost Strikeout prediction model
   - `ensemble.py` - Meta-learner for edge calculation
   - `train.py` - Model training script
   - `evaluate.py` - Model evaluation and calibration

3. **Dashboard Layer** (`/dashboards`)
   - `app.py` - Flask/Sidekiq dashboard
   - `templates/index.html` - Main dashboard UI
   - `api.py` - REST API for predictions
   - `websocket.py` - Real-time updates

4. **Notebooks** (`/notebooks`)
   - `01_exploratory_analysis.ipynb` - Data exploration
   - `02_feature_engineering.ipynb` - Feature development
   - `03_model_training.ipynb` - Model development
   - `04_backtesting.ipynb` - Strategy validation

5. **Documentation** (`/docs`)
   - `FEATURES.md` - Feature engineering guide
   - `MODEL_GUIDE.md` - Model documentation
   - `API.md` - API reference
   - `DEPLOYMENT.md` - Deployment guide

---

## 📈 Success Metrics

### Model Performance Targets
| Prediction Type | Hit Rate Target | Confidence Threshold | Expected Bets/Season |
|----------------|-----------------|----------------------|---------------------|
| Home Run (HR) | 28-30% | 85+ | ~150 |
| Hit | 52-55% | 75+ | ~250 |
| Strikeout (SO) | 60-65% | 80+ | ~200 |

### ROI Targets
| Strategy | Expected ROI | Bankroll Growth |
|----------|-------------|-----------------|
| Conservative | +5-8% | 1.5x / year |
| Moderate | +10-15% | 2.0x / year |
| Aggressive | +8-12% | 1.8x / year |

### Key Metrics
- **Edge > 5%** - Only bet when predicted probability > implied odds + 5%
- **Kelly Fraction** - 0.25-0.5 (quarter to half Kelly for risk management)
- **Max Position Size** - 2% of bankroll per bet
- **CLV Positive** - 60%+ of bets beat closing line
- **Sharpe Ratio** - Target > 1.5

---

## 🔬 Research Questions

1. **Feature Importance**
   - Which Statcast metrics are most predictive? (barrel rate, EV95+)
   - How much do pitcher quality factors matter?
   - Are park factors significant for HR predictions?

2. **Model Selection**
   - XGBoost vs LightGBM vs Random Forest?
   - Neural network for time-series?
   - Ensemble of multiple models?

3. **Calibration**
   - Do predicted probabilities match actual hit rates?
   - How to calibrate confidence scores?
   - Platt scaling vs isotonic regression?

4. **Odds Integration**
   - Which books offer best MLB props?
   - How to handle line movement?
   - When to bet (early line vs closer to game)?

5. **Bankroll Management**
   - Full Kelly vs fractional Kelly?
   - Kelly fraction optimization (0.25, 0.5, 0.75)?
   - Position sizing by confidence tier?

---

## 🗓️ Implementation Timeline

### Phase 1: Data Foundation (Week 1)
- [ ] Set up PostgreSQL database
- [ ] Pull historical Statcast data (2023-2025)
- [ ] Scrape historical odds (if available)
- [ ] Build feature engineering pipeline
- [ ] Create feature store

### Phase 2: Model Development (Week 2-3)
- [ ] Train HR prediction model (XGBoost)
- [ ] Train Hit prediction model (XGBoost)
- [ ] Train Strikeout prediction model (XGBoost)
- [ ] Build ensemble meta-learner
- [ ] Calibrate confidence scores

### Phase 3: Backtesting (Week 4)
- [ ] Test models on historical data
- [ ] Calculate ROI by strategy
- [ ] Validate CLV performance
- [ ] Optimize confidence thresholds
- [ ] Tune Kelly fraction

### Phase 4: Dashboard (Week 5-6)
- [ ] Build Flask dashboard
- [ ] Integrate real-time odds
- [ ] Add bankroll tracking
- [ ] Implement Kelly criterion calculator
- [ ] Add performance visualizations

### Phase 5: Production (Week 7-8)
- [ ] Deploy to production server
- [ ] Set up automated predictions (cron jobs)
- [ ] Configure alerts (email/SMS)
- [ ] Monitor performance
- [ ] Iterate and improve

---

## 🛠️ Technology Stack

**Backend**
- Python 3.11+
- XGBoost/LightGBM for ML models
- PostgreSQL for database
- Flask/Sidekiq for dashboard
- Redis for caching

**Data**
- MLB Statcast API
- Odds API (DraftKings/FanDuel/Bookmaker)
- Pandas/NumPy for data processing
- Scikit-learn for ML utilities

**Frontend**
- HTML5/CSS3/JavaScript
- Chart.js for visualizations
- Bootstrap for responsive design
- WebSocket for real-time updates

**Infrastructure**
- AWS/GCP for hosting
- Docker for containerization
- Nginx for reverse proxy
- PostgreSQL RDS for database

---

## 🎯 MVP Definition (Minimum Viable Product)

**What We'll Ship First:**
1. ✅ XGBoost HR prediction model trained on 2023-2025 data
2. ✅ Flask dashboard with today's predictions
3. ✅ Odds integration (implied probability)
4. ✅ Edge calculation (predicted - implied)
5. ✅ Kelly criterion bet sizing
6. ✅ Basic bankroll tracking

**What Comes Later:**
- ⏳ Hit and Strikeout models
- ⏳ Ensemble meta-learner
- ⏳ Real-time line movement tracking
- ⏳ Advanced analytics (CLV, regression analysis)
- ⏳ Mobile app
- ⏳ Automated betting integration

---

## 💡 Innovation Points

1. **Confidence Calibration** - Unlike existing systems, we'll ensure predicted probability = actual hit rate
2. **Edge-Based Betting** - Only bet when predicted edge > threshold, not just high confidence
3. **Kelly Criterion** - Mathematically optimal bet sizing based on edge and bankroll
4. **Meta-Learning Ensemble** - Combine HR/Hit/SO signals for better overall predictions
5. **CLV Tracking** - Closing Line Value to prove long-term edge
6. **Real-Time Dashboard** - Live predictions with odds and edge

---

## 🤔 Questions for Mark

Before I proceed, I need to clarify a few things:

1. **Odds Data Source**
   - Do you have historical odds data (2023-2025)?
   - Which books do you use for MLB props?
   - Can we scrape odds APIs, or do you have data access?

2. **Bankroll Size**
   - What's your starting bankroll for betting?
   - What's your risk tolerance (conservative/moderate/aggressive)?
   - Kelly fraction preference (0.25, 0.5, 0.75)?

3. **Deployment Target**
   - Should this run on your local machine or a cloud server?
   - Do you need mobile access?
   - Should it integrate with the existing Hellraiser system?

4. **Betting Integration**
   - Do you want automated betting, or manual bet placement?
   - Which books do you use (DraftKings, FanDuel, etc.)?
   - Need API integration with betting platforms?

5. **Focus Priority**
   - Should we start with HR-only model, or build all three (HR/Hit/SO)?
   - ML-first approach, or improve existing rule-based system first?
   - Dashboard-first, or model-first?

---

## 📞 Communication

**How I'll Reach You:**
- Discord bot in #general channel for major updates
- Terminal messages here for daily progress
- Email/SMS alerts for STRONG_BUY plays (if configured)

**How to Reach Me:**
- Terminal: Just type a message here
- Discord: Mention @Clawd or DM
- Anytime you have questions or want adjustments

---

**Project Start Date:** February 16, 2026
**Estimated Completion:** April 2026 (8 weeks)
**Team:** Riff (AI) + Mark (Human)

---

Let's build something profitable. 🚀
