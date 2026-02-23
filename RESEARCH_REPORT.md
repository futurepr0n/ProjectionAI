# ProjectionAI: Profitable Markers Research Report

**Date:** February 17, 2026
**Project:** ProjectionAI
**Goal:** Identify "Profitable Markers" in legacy data and assess gaps for next season

---

## Executive Summary

⚠️ **CRITICAL FINDING:** The current betting strategy is experiencing a **-67.64% ROI** across 44,137 bets. No single segment analyzed shows positive ROI. Immediate intervention is required before the next season.

### Key Metrics
- **Total Bets:** 44,137
- **Win Rate:** 5.44%
- **Net Loss:** -$29,855.55
- **Overall ROI:** -67.64%

---

## 1. FEATURE AUDIT: Current Tracking vs Missing Features

### ✓ CURRENTLY TRACKED (with completeness)

**Batter Swing Metrics (70.9% complete):**
- swing_optimization_score
- swing_attack_angle
- swing_ideal_rate
- swing_bat_speed

**Batter Quality Metrics (42.0% complete - WARNING):**
- barrel_rate
- exit_velocity_avg
- hard_hit_percent
- sweet_spot_percent

**Pitcher Metrics (0.2% complete - CRITICAL):**
- pitcher_era ⚠️
- pitcher_k_per_9 ⚠️

**Game Context (100% complete):**
- is_home
- confidence_score
- odds_decimal
- venue

**Game Info (87-100% complete):**
- game_time
- game_description
- home_team
- away_team

### ✗ MISSING FEATURES NOT TRACKED

| Feature | Status | Impact |
|---------|--------|--------|
| Weather (Wind/Temp) | NOT TRACKED | HIGH - Wind direction/speed directly affects HR potential |
| Travel (Days Rest/Distance) | NOT TRACKED | HIGH - Player fatigue impacts performance |
| Stadium Dimensions | NOT TRACKED | HIGH - Essential for HR probability context |
| Umpire Bias | NOT TRACKED | MEDIUM - Strike zone affects pitch selection |
| Stadium Factors | NOT TRACKED | MEDIUM - Elevation, humidity, wind patterns |
| Time of Day/Day of Week | PARTIAL | LOW-MEDIUM - Day games vs night games |
| Team Rest Days | NOT TRACKED | HIGH - Fatigue indicator |
| Travel Distance | NOT TRACKED | HIGH - Jet lag/travel fatigue |

---

## 2. ROI ANALYSIS: Finding Profitable Patterns

### Overall Performance
```
Total Bets: 44,137
Winners: 2,403 (5.44%)
Losers: 41,734 (94.56%)
Profit/Loss: -$29,855.55
ROI: -67.64%
```

### Segment Analysis

| Segment | Bets | Win Rate | ROI | Status |
|---------|------|----------|-----|--------|
| High Confidence (≥70%) | 22,147 | 7.2% | -58.39% | ❌ Loss |
| Home Games | 16,470 | 4.4% | -74.29% | ❌ Loss |
| Odds 2.0-4.0 | 2,512 | 11.0% | -62.47% | ❌ Loss |
| Odds 4.0-6.0 | 14,403 | 7.7% | -62.25% | ❌ Loss |
| Odds 6.0-8.0 | 12,870 | 4.8% | -68.68% | ❌ Loss |
| Odds 8.0-12.0 | 11,777 | 3.0% | -72.97% | ❌ Loss |
| Above-Avg Swing Score | 15,650 | 7.6% | -60.75% | ❌ Loss |

### ⚠️ CRITICAL FINDING: No segments with positive ROI found.

Even the "best performing" segments are losing money significantly.

---

## 3. VENUE ANALYSIS: Stadium-Specific Performance

### Top 10 Venues by Volume
| Venue | Bets | Win% | ROI |
|-------|------|------|-----|
| Angel Stadium | 2,099 | 5.3% | -70.14% |
| Wrigley Field | 2,059 | 8.0% | -48.91% |
| Citi Field | 2,058 | 4.1% | -78.25% |
| Coors Field | 1,960 | 4.3% | -77.24% |
| Petco Park | 1,794 | 4.8% | -68.30% |
| American Family Field | 1,637 | 5.7% | -58.54% |
| Chase Field | 1,636 | 5.9% | -63.01% |
| PNC Park | 1,622 | 3.5% | -72.76% |
| Rogers Centre | 1,601 | 4.1% | -74.61% |
| Kauffman Stadium | 1,589 | 5.9% | -62.18% |

### Best Performing Venues (ROI, min 100 bets)
| Venue | Bets | Win% | ROI |
|-------|------|------|-----|
| Comerica Park | 1,589 | 9.0% | -46.81% |
| Wrigley Field | 2,059 | 8.0% | -48.91% |
| Daikin Park | 1,453 | 6.4% | -58.54% |
| American Family Field | 1,637 | 5.7% | -58.54% |
| loanDepot park | 981 | 5.9% | -59.21% |

**Note:** Even the "best" venues are losing ~47-60% ROI. Venue selection alone is not a solution.

---

## 4. HIGH CONFIDENCE MARKERS: Patterns to Require Next Season

### Marker Performance (Sorted by Win Rate)

| Marker | Winners | Win Rate | Assessment |
|--------|---------|----------|-------------|
| Odds 2.0-4.0 | 277 | 11.03% | ⚠️ Highest win rate, but still -62% ROI |
| Confidence ≥ 90% | 1,225 | 8.90% | ⚠️ Best confidence tier, still losing |
| Confidence ≥ 80% | 1,421 | 8.06% | ⚠️ High confidence, still losing |
| Odds 4.0-6.0 | 1,111 | 7.71% | ⚠️ Moderate odds, still losing |
| Swing Score ≥ 71.2 | 1,194 | 7.63% | ⚠️ Good swing metrics, still losing |
| Confidence ≥ 70% | 1,591 | 7.18% | ⚠️ Standard high confidence, still losing |
| Confidence ≥ 60% | 1,669 | 6.77% | ⚠️ Moderate confidence, still losing |
| Confidence ≥ 50% | 1,689 | 6.60% | ⚠️ Low confidence threshold, still losing |
| Away Game | 1,686 | 6.09% | ❌ Worse than home games |
| Odds 6.0-8.0 | 613 | 4.76% | ❌ Poor win rate |
| Home Game | 717 | 4.35% | ❌ Lowest win rate |
| Odds 8.0-10.0 | 267 | 3.17% | ❌ Terrible win rate |

### 🚨 CRITICAL INSIGHT
**No "Golden Subset" exists in the current data.** Even with the best markers (Odds 2.0-4.0, Confidence ≥90%), the strategy is unprofitable.

---

## 5. DATA QUALITY ISSUES

### Critical Gaps
1. **Pitcher Stats Only 0.2% Complete**
   - pitcher_era: 72/44,137 records
   - pitcher_k_per_9: 72/44,137 records
   - **Impact:** Cannot properly assess pitcher quality

2. **Advanced Batter Stats Only 42% Complete**
   - barrel_rate, exit_velocity_avg, hard_hit_percent, sweet_spot_percent
   - **Impact:** Missing quality metrics for majority of predictions

3. **Inconsistent Feature Coverage**
   - Swing metrics: 70.9% complete
   - Game context: 100% complete
   - **Impact:** Model training on incomplete data

---

## 6. RECOMMENDATIONS

### 🔴 CRITICAL PRIORITY ACTIONS (Before Next Season)

1. **Fix Pitcher Stats Data Collection**
   - Goal: Achieve >90% coverage for ERA, K/9, HR/9, WHIP
   - Action: Debug and fix the data pipeline that should be populating these fields
   - Timeline: Week 1-2

2. **Integrate Weather API**
   - Goal: Real-time wind direction, speed, temperature for each game
   - Action: Add weather data collection to pipeline
   - Impact: HIGH - Weather dramatically affects HR probability
   - Timeline: Week 2-3

3. **Add Stadium Dimension Database**
   - Goal: Fence distances (left, center, right), elevation
   - Action: Create stadium reference database
   - Impact: HIGH - Essential for HR context
   - Timeline: Week 1

4. **Calculate Travel Metrics**
   - Goal: Days rest, travel distance from previous game
   - Action: Parse schedule data, calculate travel fatigue
   - Impact: HIGH - Player fatigue affects performance
   - Timeline: Week 2-3

5. **Improve Batter Stats Coverage**
   - Goal: Achieve >80% coverage for barrel_rate, exit_velocity, etc.
   - Action: Debug data collection pipeline
   - Timeline: Week 2

### 🟡 STRATEGY RECOMMENDATIONS

1. **Implement Minimum Confidence Threshold**
   - Current: Betting on picks with 25-95 confidence
   - Recommendation: Only bet when confidence ≥80%
   - Rationale: Higher win rate, though still need to solve profitability

2. **Focus on Specific Odds Ranges**
   - Best performing: Odds 2.0-4.0 (11.03% win rate)
   - Avoid: Odds >8.0 (≤3.2% win rate)
   - Rationale: Better win rates in lower odds range

3. **Track Venue-Specific Patterns**
   - Some venues (Comerica Park, Wrigley Field) perform "better" (-47% to -49% ROI)
   - Avoid worst performing venues (Citi Field: -78% ROI)
   - Rationale: Venue filters can reduce losses

4. **Implement Stricter Data Quality Filters**
   - Do not bet when key features are missing
   - Require: pitcher stats, batter metrics, weather data
   - Rationale: Better predictions require complete data

### 🟢 FUTURE ENHANCEMENTS

1. **Umpire Bias Tracking**
   - Zone metrics, strikeout rates per umpire
   - Impact on pitch selection and outcomes

2. **Time of Day Analysis**
   - Day games vs night games
   - Temperature variations

3. **Previous Day's Games Context**
   - Fatigue from previous day's workload
   - In-game performance patterns

---

## 7. "GOLDEN SUBSET" SEARCH RESULTS

### Methodology Attempted
- Analyzed all combinations of:
  - Confidence thresholds (50%, 60%, 70%, 80%, 90%)
  - Odds ranges (2-4, 4-6, 6-8, 8-12)
  - Home/Away splits
  - Swing score tertiles
  - Venue groupings

### Result
❌ **NO PROFITABLE SUBSET FOUND**

### Conclusion
The current strategy is fundamentally flawed. The issues are:

1. **Missing Critical Features:** No weather, travel, or stadium dimension data
2. **Poor Data Quality:** Pitcher stats 0.2% complete, batter metrics 42% complete
3. **Over-betting:** 44,137 bets across the season is excessive
4. **No Edge:** The model is not providing a competitive advantage

**Recommendation:** Do not proceed with next season's betting until:
- Data quality issues are resolved
- Critical features are added
- Backtesting shows positive ROI in a validation set

---

## 8. NEXT STEPS

### Immediate (This Week)
1. Fix pitcher stats data pipeline
2. Create stadium dimension database
3. Set up weather API integration
4. Audit data collection scripts for missing features

### Short-term (Next 2-3 Weeks)
1. Implement travel metrics calculation
2. Improve batter stats coverage to >80%
3. Build validation framework for new features
4. Run backtests with improved data

### Before Next Season
1. Validate model with complete feature set
2. Achieve positive ROI in backtesting
3. Set strict betting criteria (confidence, odds, data completeness)
4. Implement stop-loss mechanisms

---

## APPENDIX: Model Features Currently Used

From `models/hr_model_features.json`:
```json
[
  "barrel_rate",
  "exit_velocity_avg",
  "hard_hit_percent",
  "sweet_spot_percent",
  "swing_optimization_score",
  "swing_attack_angle",
  "swing_bat_speed",
  "pitcher_era",
  "pitcher_hr_per_9",
  "pitcher_k_per_9",
  "pitcher_whip",
  "k_rate_pct",
  "hr_rate_pct",
  "fly_ball_pct",
  "is_home",
  "confidence_score",
  "odds_decimal"
]
```

**Note:** The model references pitcher stats that are only 0.2% available in the training data. This explains poor model performance.

---

**Report Generated:** February 17, 2026
**Analysis Script:** `research_analysis.py`
**Results JSON:** `research_results.json`
**Total Analyzed:** 44,137 bets, 2,403 winners, 41,734 losers
