## Hitter Recency Optimization Plan

Purpose: improve the newer game-based hitter recency path for `HR` and `Hit` without reverting to calendar-day windows.

### Problem Statement

The product logic moved in the right direction:

- live serving now uses last-N-games recent form instead of last-N-days
- the dashboard lets users inspect multiple lookback windows
- this better matches how hitters and pitchers actually appear in games

But the first training pass on game-based recency underperformed the older hitter artifacts.

Observed holdout results:

- older pre-change hitter artifacts were around:
  - `HR`: `~0.685`
  - `Hit`: `~0.693`
- current game-based hitter artifacts are around:
  - `HR`: `0.6699`
  - `Hit`: `0.6787`

This means the direction is correct, but the first implementation of the recency block is not yet tuned well enough for training.

### Current Experiment Notes

Completed so far:

- moved live and offline hitter recency to last-N-games
- tested single-window defaults (`5`, `10`, `15`, `20`)
- promoted `20` games as the best single-window compromise of the tested game-based windows
- fixed a real data issue where `0.0` recent rates were being treated like missing values in recency attachment
- added multi-window recency fields plus sample-quality fields:
  - recent rates for `3`, `5`, `10`, `20` games
  - games used
  - PA used
  - days since last game
- added first-pass shrinkage features against season-to-date baseline
- ran a first target-specific recency gating pass:
  - `HR` kept HR-focused recency and dropped hit/SO-specific recency extras
  - `Hit` dropped HR-burst recency extras

What happened:

- coverage improved materially after the `0.0` handling fix
- the broader recency feature set still did not beat the earlier hitter artifacts on holdout
- first shrinkage pass improved over the raw multi-window pass slightly for `HR`, but still did not recover the older baseline
- first target-specific gating pass was essentially neutral to slightly negative:
  - `HR` around `0.6599`
  - `Hit` around `0.6770`
- first interaction pass was the best post-recency result so far:
  - added:
    - recent-form x projected PA
    - recent-form x platoon advantage
    - recent-form x top/middle lineup opportunity
    - recent-vs-season delta x projected PA
  - evaluation-only result:
    - `HR` around `0.6676`
    - `Hit` around `0.6822`

Implication:

- the remaining problem is not "missing recency fields"
- the remaining problem is feature weighting / interaction design around recency, not just raw inclusion or exclusion
- interaction terms are currently the strongest recency optimization direction tested

### Likely Causes

The regression is probably not caused by "games instead of days" by itself. More likely causes are:

- the old models were tuned around the old feature distribution, and the new recency fields changed that distribution materially
- one single lookback is unlikely to fit both `HR` and `Hit`
- raw rate features may be too noisy for small sample windows
- current game-based recency uses only one feature family and does not encode sample quality strongly enough
- missingness and sparse recent-game counts may now be interacting badly with the rest of the feature set
- the live serving path and offline training path are now more aligned conceptually, but the hitter training feature block still needs target-specific tuning

### Principles

- keep game-based recency as the primary direction
- do not revert to day-based recency as the main feature family
- treat day-based spans as optional secondary context only if they prove additive
- evaluate changes with temporal holdout only
- prefer smaller, evidence-driven recency changes over large untested bundles
- keep live serving and offline training conceptually aligned

### Immediate Goals

1. Recover or exceed prior hitter holdout quality while staying on a game-based recency design.
2. Make the recency block more stable across sparse players, bench bats, injuries, and lineup changes.
3. Separate what helps `HR` from what helps `Hit` instead of forcing one recency design across both targets.

### Workstreams

### 1. Expand the recency feature family

Current game-based recency is still too narrow. Add:

- multi-window recent form features:
  - last `3` games
  - last `5` games
  - last `10` games
  - last `20` games
- rolling counts, not just rates:
  - recent PA
  - recent hits
  - recent HR
  - recent SO
- recency sample-quality features:
  - games used
  - PA used
  - days since last game
  - days since last start for the opposing pitcher
- weighted recency variants:
  - exponentially weighted recent performance
  - heavier weight on most recent `3-5` games

Why:

- `HR` is sparse and likely needs longer or weighted windows
- `Hit` may benefit from shorter windows plus PA/sample context
- sample-quality indicators should help the model know when a recent rate is weak evidence

### 2. Make recency target-specific

Do not force `HR` and `Hit` to consume identical recent-form features.

Test separately:

- `HR`
  - longer windows
  - power-focused recent stats
  - recent barrel/EV style trend fields if source coverage allows
- `Hit`
  - contact/PA consistency
  - shorter windows
  - on-base / batting average style trend fields

Expected direction:

- `HR` probably wants longer memory and stronger shrinkage
- `Hit` probably wants more immediate contact-form information

### 3. Add shrinkage and baseline blending

Raw recent rates are noisy. Blend them with season or player baseline.

Candidate derived features:

- shrunk recent HR rate
- shrunk recent hit rate
- shrunk recent strikeout rate
- recent-minus-season delta
- recent / season ratio

Example pattern:

- `shrunk_recent_hit_rate = (recent_hits + k * season_hit_rate) / (recent_pa + k)`

Why:

- protects against overreacting to tiny recent samples
- keeps game-based recency but makes it more stable

### 4. Add availability and missingness intelligence

Game-based recency becomes more useful when the model can tell whether a number is trustworthy.

Add explicit features:

- `recent_form_available`
- `recent_form_games_used`
- `recent_form_pa_used`
- `low_recent_sample_flag`
- `returning_from_gap_flag`
- `days_since_last_game`

Why:

- a player coming off injury or rest should not be treated the same as a player with a clean recent sample

### 5. Revisit interaction terms around lineup context

The new lineup matching is much better, but recency should interact with expected opportunity.

Test interaction-style features:

- recent hit rate x projected PA
- recent HR rate x lineup slot
- recent SO rate x pitcher K tendency
- recent form x handedness/platoon advantage

Why:

- recent form without opportunity context is weaker
- a hot hitter batting ninth is different from a hot hitter batting second

### 6. Evaluate feature gating, not just feature addition

Some of the older hitter features may now be redundant or noisy once game-based recency is present.

Run ablation tests on:

- weather subsets
- batter-vs-pitcher history
- pitch-type matchup block
- lineup-slot / projected-PA block
- travel-fatigue block
- old recent-form fields versus new multi-window recency

Why:

- the right answer may be "better recency plus less of something else", not just more total features

### 7. Keep live serving and training aligned

The dashboard now lets users choose the recent-form window. The trained artifacts still need a disciplined default.

Plan:

- train with a fixed default design for production
- keep the dashboard lookback control for exploratory what-if use
- clearly mark when a user-selected live window differs from the training default

Future enhancement:

- optionally expose "training default" vs "custom live recency" in the modal

### Experiment Matrix

Run these in order.

#### Phase A: Baseline recovery

- [ ] Freeze a clean baseline from the current game-based implementation.
- [ ] Train `HR` and `Hit` with only one recent window at a time:
  - `5`
  - `10`
  - `15`
  - `20`
- [ ] Confirm the best single-window baseline per target.

Success criteria:

- recover the best currently observed game-based baseline cleanly and reproducibly

#### Phase B: Multi-window recency

- [ ] Add simultaneous windows: `3`, `5`, `10`, `20`
- [ ] Add sample-quality features alongside each window
- [ ] Test `HR` and `Hit` separately with temporal holdout

Success criteria:

- improve over the single-window game-based baseline for at least one target

#### Phase C: Shrinkage / baseline blending

- [ ] Add shrunk recent rates using season priors
- [ ] Add recent-minus-season delta features
- [ ] Compare raw-only vs shrunk-only vs combined

Success criteria:

- improved holdout AUC or better precision-at-k stability, especially on `HR`

#### Phase D: Interaction and gating

- [ ] Add lineup/opportunity interactions
- [ ] Ablate older feature families against the stronger recency block
- [ ] Keep only what helps each target

Success criteria:

- final target-specific hitter feature sets that outperform the current game-based build

### Evaluation Standard

Do not judge these changes on ROC AUC alone.

Track:

- temporal holdout ROC AUC
- temporal holdout average precision
- Brier score
- precision-at-k by date
- top-N hit rate by date
- class-specific performance for:
  - `Strong Buy`
  - `Buy`
  - `Moderate`
- stability of metrics across months, not just one pooled holdout

### Product / UX Follow-Through

Once the recency block is stable:

- [ ] show the training-default lookback in the dashboard
- [ ] show whether a live prediction is using default or user-overridden recency
- [ ] include recent-form sample quality in the modal:
  - games used
  - PA used
  - days since last game

### Data Dependencies

This plan depends on:

- `hitting_stats` staying complete enough to build last-N-games history
- `daily_lineups` continuing to improve lineup slot coverage
- better name matching for later-date lineup joins

It does not require weather changes to proceed.

### Recommended Execution Order

1. Freeze the current game-based baseline.
2. Add multi-window recent-form features and sample-quality indicators.
3. Add shrinkage/blending against season baseline.
4. Run target-specific ablation and gating.
5. Re-train final `HR` and `Hit` artifacts.
6. Update dashboard copy so live recency controls reflect the production default clearly.

### Current Recommendation

The best next implementation step is:

- build multi-window hitter recent-form features with explicit sample-quality indicators

That is the most likely way to improve the game-based recency block without abandoning the correct product logic.
