# ProjectionAI Implementation Backlog

Purpose: track high-value improvements and check them off as they are implemented and verified.

## Current Findings

- The active pipeline is [data/build_training_dataset.py](/Users/futurepr0n/Development/Capping.Pro/Github/ProjectionAI/data/build_training_dataset.py) -> [models/train_models_v4.py](/Users/futurepr0n/Development/Capping.Pro/Github/ProjectionAI/models/train_models_v4.py) -> [dashboards/app.py](/Users/futurepr0n/Development/Capping.Pro/Github/ProjectionAI/dashboards/app.py).
- Training previously mixed time-series CV with a random final split. That made reported performance optimistic.
- The name registry exists, but matching has relied too heavily on exact alias hits and last-name-only fallback. That creates both silent feature loss and silent wrong matches.
- Current source cardinalities observed during investigation:
  - `play_by_play_plays` distinct batters: `2453`
  - `hitter_exit_velocity` distinct names: `443`
  - `custom_batter_2025` distinct names: `609`
  - `player_name_map` rows: `1504`
  - `player_name_map` rows with aliases: `895`

## Training / Evaluation

- [x] Replace random final train/test split with temporal holdout evaluation.
- [x] Add out-of-fold base-model predictions for meta-learner training.
- [x] Save richer holdout metrics and split metadata into model artifacts.
- [ ] Reconcile serving behavior in [dashboards/app.py](/Users/futurepr0n/Development/Capping.Pro/Github/ProjectionAI/dashboards/app.py) so it either uses the full ensemble or explicitly becomes an XGBoost-only path.
- [ ] Add precision-at-k / lift metrics per target for betting use, not just ROC-oriented metrics.
- [ ] Add target-specific calibration checks and reliability plots.
- [ ] Add a reproducible backtest script that evaluates decisions by date, threshold, and bankroll rules.
- [ ] Investigate the unusually high `so` performance and confirm labels/features are not inflating the metric.

## Name Matching / Feature Coverage

- [x] Remove arbitrary last-name-only canonicalization from the active resolver when multiple candidates exist.
- [x] Add explicit name-resolution metadata (`match_type`, `ambiguous`, candidates) instead of silent fallback.
- [x] Log source-by-source resolution stats in [data/build_training_dataset.py](/Users/futurepr0n/Development/Capping.Pro/Github/ProjectionAI/data/build_training_dataset.py) so feature loss is visible.
- [x] Build a persistent alias-audit layer using new DB tables (`player_name_resolution_audit`, `player_name_alias_audit`).
- [x] Add a review/apply workflow so only reviewed alias proposals are promoted into `player_name_map`.
- [x] Add bulk review updates keyed on `alias_to_apply` so repeated play-by-play variants can be approved/rejected together.
- [ ] Extend the audit tables/flow with columns such as:
  - `source_table`
  - `raw_name`
  - `normalized_name`
  - `resolved_canonical_name`
  - `match_type`
  - `match_confidence`
  - `review_status`
- [ ] Add a review queue for unresolved or ambiguous names instead of silently accepting them.
- [ ] Prefer MLBAM/Statcast IDs wherever the source provides them; use names as fallback only.
- [ ] Add source-specific parsers for known formats (`Lastname, Firstname`, `F. Lastname`, suffixes, accents).
- [ ] Track per-source join coverage in every dataset build:
  - EV coverage
  - xStats coverage
  - recent-rates coverage
  - pitcher-feature coverage
- [ ] Fail the build or emit a strong warning when coverage drops below agreed thresholds.
- [ ] Add a recurring audit job that reports:
  - newly unseen names
  - ambiguous surnames
  - alias collisions
  - rows that changed canonical mapping
- [x] Add date-aware team history validation from official transaction / roster sources so team checks use game-date context, not just a static current `team_code`.
- [ ] Surface a "possible team mismatch" badge in the app with a tooltip when a reviewed/allowed association still has unresolved team context.

## Data / Features

- [ ] Replace placeholder weather defaults with real historical and game-time weather features.
- [ ] Replace `pa_count` with a true pregame projection or lineup-context feature if it is being used for inference.
- [ ] Add handedness splits for batter vs pitcher.
- [ ] Add projected lineup spot and expected plate appearances.
- [ ] Add park/environment features that are directionally meaningful for HRs:
  - wind direction
  - humidity
  - roof open/closed state where applicable
- [ ] Add bullpen quality / opener-following context for pitcher matchup features.
- [ ] Add stronger missingness handling:
  - presence indicators
  - source freshness
  - fallback provenance
- [x] Build a starter-level strikeout dataset from `daily_lineups`, `games`, `pitching_stats`, and team batting history instead of using hitter rows.
- [x] Add starter rolling-form features and opponent team recent strikeout-form features for the starter strikeout dataset.
- [x] Generate flexible starter strikeout labels (`3+`, `4+`, `5+`, `6+`) so the final betting threshold can be chosen explicitly.
- [x] Add pitch-type matchup features:
  - starter primary pitch mix
  - opponent team results versus similar pitch types
  - opponent team swing-and-miss tendencies by pitch type
- [ ] Add prior batter-vs-pitcher / team-vs-starter matchup features with careful leakage controls.
- [x] Add probable-pitch-count / leash features so strikeout probabilities reflect expected workload, not just skill rate.

## Product / Serving

- [x] Confirm the Flask app boots cleanly and serves locally.
- [x] Add a per-day team selector in the dashboard so predictions can be narrowed to one club quickly.
- [x] Add an "include opponent" dashboard filter so one team selection can also show the day’s opposing-side matchup context.
- [x] Expand batch prediction in [scripts/generate_daily_predictions.py](/Users/futurepr0n/Development/Capping.Pro/Github/ProjectionAI/scripts/generate_daily_predictions.py) so it matches the app and scores all three supported probabilities:
  - hitter hit probability
  - hitter home run probability
  - starting pitcher strikeout probability versus the opposing team
- [x] Train baseline starter strikeout models for `3+`, `4+`, `5+`, and `6+` thresholds and save artifacts for review.
- [x] Replace the current SO export/app path with a starter-based strikeout model instead of hitter-row strikeout proxies.
- [x] Add strikeout-threshold selection so the dashboard/API/batch export can score `3+`, `4+`, `5+`, or `6+` starter strikeouts from the same dated starter dataset.
- [ ] Decide whether the default strikeout view should stay `3+ K` or become sportsbook-line-aware per game/date.
- [ ] Add a visible "possible team mismatch" badge/tooltip to prediction rows when alias resolution is allowed but historical team validation remains unresolved.
- [ ] Unify signal logic so probabilities, thresholds, and rank-based labels are not competing systems.
- [ ] Add model/artifact versioning to responses and generated output.
- [ ] Add endpoint-level health checks for DB, dataset, and artifacts.

## Suggested Order

1. Stabilize training evaluation and ensemble-serving behavior.
2. Build the alias audit/review flow so name problems stop being silent.
3. Add coverage gates and better feature-missingness reporting.
4. Replace placeholder weather and improve pregame-only feature design.
5. Expand backtesting and betting-oriented evaluation.
