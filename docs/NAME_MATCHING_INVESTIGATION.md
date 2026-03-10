# Name Matching Investigation

## What Was Checked

- Reviewed the active resolver in [data/name_utils.py](/Users/futurepr0n/Development/Capping.Pro/Github/ProjectionAI/data/name_utils.py)
- Reviewed the registry seeding logic in [data/migrate_player_names.py](/Users/futurepr0n/Development/Capping.Pro/Github/ProjectionAI/data/migrate_player_names.py)
- Audited source-table cardinalities and registry shape against the live database

## Current Coverage Snapshot

- `play_by_play_plays.batter`: `2453` distinct names
- `hitter_exit_velocity.last_name_first_name`: `443` distinct names
- `custom_batter_2025.last_name_first_name`: `609` distinct names
- `player_name_map`: `1504` rows
- `player_name_map` rows with aliases: `895`

With the updated resolver, current source names resolve as:

- `play_by_play_plays.batter`: `2453 / 2453`
- `hitter_exit_velocity.last_name_first_name`: `443 / 443`
- `custom_batter_2025.last_name_first_name`: `609 / 609`

Match types observed:

- play-by-play: `1900 exact_alias`, `553 exact_canonical`
- EV: `71 exact_alias`, `372 exact_canonical`
- xStats: `118 exact_alias`, `491 exact_canonical`

## Important Caveat

High resolution coverage does not prove high mapping quality.

The current migration logic in [data/migrate_player_names.py](/Users/futurepr0n/Development/Capping.Pro/Github/ProjectionAI/data/migrate_player_names.py) seeds aliases by looking up:

- `WHERE last_name = %s LIMIT 1`

That means many aliases may already have been attached to a canonical player through a surname-only decision. Once that happens, later lookups appear "successful" because the alias is now present in `player_name_map`, even if it was attached to the wrong player.

In other words:

- low unresolved-count does not mean low error-rate
- the risk is silent false-positive joins, not just silent feature loss

## Registry Ambiguity

Observed ambiguous surnames in `player_name_map`: `210`

Sample ambiguous surnames:

- `smith` (`9`)
- `garcia` (`6`)
- `cruz` (`5`)
- `johnson` (`5`)
- `lee` (`5`)
- `lopez` (`5`)
- `miller` (`5`)
- `rodriguez` (`5`)
- `williams` (`5`)

One additional warning sign: `jr.` appears as a surname bucket in the registry, which suggests suffix parsing is still dirty in some records.

## What Was Changed

Updated [data/name_utils.py](/Users/futurepr0n/Development/Capping.Pro/Github/ProjectionAI/data/name_utils.py):

- canonical resolution now returns metadata
- arbitrary surname-only fallback is blocked when multiple candidates exist
- same-surname disambiguation now prefers:
  - exact canonical match
  - exact alias match
  - exact first-name + last-name
  - first-initial + last-name
  - conservative same-surname fuzzy match
- registry lookups now use an in-memory cache built from `player_name_map`

Updated [data/build_training_dataset.py](/Users/futurepr0n/Development/Capping.Pro/Github/ProjectionAI/data/build_training_dataset.py):

- dataset builds now log source-by-source resolution stats
- unresolved and ambiguous names are no longer silent during feature assembly

## Best Next Improvement

Add a persistent alias audit/review layer instead of letting registry seeding mutate the truth table invisibly.

Recommended table shape:

- `source_table`
- `raw_name`
- `normalized_name`
- `resolved_canonical_name`
- `match_type`
- `match_confidence`
- `resolved_by`
- `review_status`
- `created_at`
- `updated_at`

Recommended policy:

- exact canonical / exact alias matches can auto-accept
- ambiguous surname matches should require review
- any fuzzy match should be reviewable and traceable
- source rows with IDs should store those IDs and prefer them over names

## Bottom Line

The main problem is no longer simple "missing names". The deeper issue is that alias truth can be polluted during migration, which makes downstream coverage look healthy even when some joins may be wrong.
