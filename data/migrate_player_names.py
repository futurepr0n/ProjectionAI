#!/usr/bin/env python3
"""
Player Name Registry Migration / Audit

This script supports two modes:

1. Registry mutation
   - create and seed `player_name_map`
   - update alias arrays in `player_name_map`

2. Non-destructive audit
   - create new audit tables
   - record proposed source-name resolutions without mutating source tables
   - leave existing shared data unchanged
"""

import argparse
import logging
import os
import re
from typing import Tuple
from pathlib import Path

import pandas as pd
import psycopg2
from psycopg2.extras import Json, RealDictCursor

from name_utils import normalize_name, resolve_name_match

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


SOURCE_CONFIG = {
    'players': ('players', 'full_name'),
    'hitter_exit_velocity': ('hitter_exit_velocity', 'last_name_first_name'),
    'custom_batter_2025': ('custom_batter_2025', 'last_name_first_name'),
    'play_by_play_plays': ('play_by_play_plays', 'batter'),
}
PLAY_BY_PLAY_ACTION_RE = re.compile(
    r'\s+(hit|struck|walk|fly|bunt|by|grounded|lined|flied|popped|'
    r'singled|doubled|tripled|homered|reached|sacrifice|intentional|'
    r'called|swinging|fouled|safe|out)\b.*$',
    re.IGNORECASE
)


class PlayerNameMigration:
    def __init__(self):
        self.conn = self._connect()

    def _connect(self):
        try:
            conn = psycopg2.connect(
                host=os.getenv('DB_HOST', '192.168.1.23'),
                port=int(os.getenv('DB_PORT', 5432)),
                database=os.getenv('DB_NAME', 'baseball_migration_test'),
                user=os.getenv('DB_USER', 'postgres'),
                password=os.getenv('DB_PASSWORD', 'korn5676')
            )
            logger.info("Connected to database")
            return conn
        except Exception as exc:
            logger.error(f"Database connection failed: {exc}")
            return None

    def close(self):
        if self.conn:
            self.conn.close()

    def create_table(self):
        """Create the player_name_map table."""
        if not self.conn:
            logger.error("Cannot create table: no database connection")
            return False

        try:
            cursor = self.conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS player_name_map (
                    id SERIAL PRIMARY KEY,
                    canonical_name VARCHAR(100) NOT NULL,
                    mlb_id INTEGER,
                    aliases TEXT[] NOT NULL DEFAULT '{}',
                    last_name VARCHAR(60),
                    first_name VARCHAR(60),
                    first_initial CHAR(1),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(canonical_name)
                );
                """
            )
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_pnm_mlb_id ON player_name_map(mlb_id);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_pnm_last_name ON player_name_map(last_name);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_pnm_first_initial ON player_name_map(first_initial);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_pnm_aliases ON player_name_map USING GIN(aliases);")
            self.conn.commit()
            logger.info("✅ player_name_map table created with indexes")
            return True
        except Exception as exc:
            logger.error(f"❌ Failed to create table: {exc}")
            self.conn.rollback()
            return False

    def create_audit_tables(self):
        """Create non-destructive audit tables."""
        if not self.conn:
            logger.error("Cannot create audit tables: no database connection")
            return False

        try:
            cursor = self.conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS player_name_resolution_audit (
                    id SERIAL PRIMARY KEY,
                    source_table VARCHAR(100) NOT NULL,
                    source_column VARCHAR(100) NOT NULL,
                    raw_name TEXT NOT NULL,
                    normalized_name TEXT,
                    proposed_canonical_name TEXT,
                    matched BOOLEAN NOT NULL DEFAULT false,
                    ambiguous BOOLEAN NOT NULL DEFAULT false,
                    match_type VARCHAR(50) NOT NULL,
                    candidate_count INTEGER NOT NULL DEFAULT 0,
                    candidates JSONB NOT NULL DEFAULT '[]'::jsonb,
                    review_status VARCHAR(30) NOT NULL DEFAULT 'pending',
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(source_table, source_column, raw_name)
                );
                """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_pnra_source_table
                ON player_name_resolution_audit(source_table);
                """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_pnra_review_status
                ON player_name_resolution_audit(review_status);
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS player_name_alias_audit (
                    id SERIAL PRIMARY KEY,
                    source_table VARCHAR(100) NOT NULL,
                    raw_name TEXT NOT NULL,
                    alias_to_apply TEXT,
                    normalized_name TEXT,
                    proposed_canonical_name TEXT,
                    proposed_match_type VARCHAR(50) NOT NULL,
                    observed_teams TEXT,
                    team_validation VARCHAR(30),
                    official_roster_teams TEXT,
                    official_transaction_teams TEXT,
                    historical_team_validation VARCHAR(40),
                    apply_action VARCHAR(30) NOT NULL,
                    review_status VARCHAR(30) NOT NULL DEFAULT 'pending',
                    notes TEXT,
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(source_table, raw_name, proposed_canonical_name)
                );
                """
            )
            cursor.execute(
                """
                ALTER TABLE player_name_alias_audit
                ADD COLUMN IF NOT EXISTS alias_to_apply TEXT
                """
            )
            cursor.execute(
                """
                ALTER TABLE player_name_alias_audit
                ADD COLUMN IF NOT EXISTS observed_teams TEXT
                """
            )
            cursor.execute(
                """
                ALTER TABLE player_name_alias_audit
                ADD COLUMN IF NOT EXISTS team_validation VARCHAR(30)
                """
            )
            cursor.execute(
                """
                ALTER TABLE player_name_alias_audit
                ADD COLUMN IF NOT EXISTS official_roster_teams TEXT
                """
            )
            cursor.execute(
                """
                ALTER TABLE player_name_alias_audit
                ADD COLUMN IF NOT EXISTS official_transaction_teams TEXT
                """
            )
            cursor.execute(
                """
                ALTER TABLE player_name_alias_audit
                ADD COLUMN IF NOT EXISTS historical_team_validation VARCHAR(40)
                """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_pnaa_review_status
                ON player_name_alias_audit(review_status);
                """
            )
            self.conn.commit()
            logger.info("✅ Audit tables created")
            return True
        except Exception as exc:
            logger.error(f"❌ Failed to create audit tables: {exc}")
            self.conn.rollback()
            return False

    def seed_from_players_table(self):
        """Seed canonical names from players table (full_name column)."""
        if not self.conn:
            logger.error("Cannot seed: no database connection")
            return False

        try:
            cursor = self.conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute(
                """
                SELECT player_id, full_name
                FROM players
                WHERE full_name IS NOT NULL AND full_name != ''
                """
            )
            players = cursor.fetchall()
            logger.info(f"Processing {len(players)} players from players table")

            inserted = 0
            for player in players:
                name = player['full_name'].strip()
                player_id = player['player_id']
                parts = name.split()
                last_name = parts[-1] if parts else ''
                first_name = parts[0] if parts else ''
                first_initial = first_name[:1] if first_name else ''

                cursor.execute(
                    """
                    INSERT INTO player_name_map
                    (canonical_name, mlb_id, last_name, first_name, first_initial)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (canonical_name) DO NOTHING
                    """,
                    (name, player_id, last_name, first_name, first_initial)
                )
                if cursor.rowcount > 0:
                    inserted += 1

            self.conn.commit()
            logger.info(f"✅ Seeded {inserted} canonical names from players table")
            return True
        except Exception as exc:
            logger.error(f"❌ Failed to seed from players table: {exc}")
            self.conn.rollback()
            return False

    def _normalize_for_registry(self, name: str) -> Tuple[str, str, str]:
        canonical = normalize_name(name).title()
        parts = canonical.split()
        first_name = parts[0] if parts else ''
        last_name = parts[-1] if parts else ''
        first_initial = first_name[:1] if first_name else ''
        return canonical, first_name, last_name

    def _source_names(self, source_table: str, source_column: str):
        cursor = self.conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute(
            f"""
            SELECT DISTINCT {source_column} AS raw_name
            FROM {source_table}
            WHERE {source_column} IS NOT NULL AND {source_column} != ''
            ORDER BY {source_column}
            """
        )
        return cursor.fetchall()

    def _clean_source_name(self, source_table: str, raw_name: str) -> str:
        name = (raw_name or '').strip()
        if not name:
            return ''

        if source_table == 'play_by_play_plays':
            if ' pitches to ' in name:
                name = name.split(' pitches to ', 1)[1].strip()
            name = re.sub(r'\s*\([^)]*\)\s*', ' ', name)
            name = PLAY_BY_PLAY_ACTION_RE.sub('', name).strip(' -,:;')

        return name.strip()

    def _load_play_by_play_team_context(self):
        cursor = self.conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute(
            """
            WITH pbp_team_context AS (
                SELECT
                    pp.batter AS raw_name,
                    CASE
                        WHEN pp.inning_half ILIKE 'Bottom%%' THEN g.home_team
                        ELSE g.away_team
                    END AS batting_team
                FROM play_by_play_plays pp
                JOIN games g ON pp.game_id = g.game_id
                WHERE pp.batter IS NOT NULL AND pp.batter != ''
            )
            SELECT
                raw_name,
                STRING_AGG(DISTINCT batting_team, '|' ORDER BY batting_team) AS observed_teams
            FROM pbp_team_context
            GROUP BY raw_name
            """
        )
        return {row['raw_name']: row['observed_teams'] for row in cursor.fetchall()}

    def _load_official_roster_history(self):
        cursor = self.conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute(
            """
            SELECT canonical_name, STRING_AGG(DISTINCT team_code, '|' ORDER BY team_code) AS roster_teams
            FROM official_team_roster_snapshots
            WHERE canonical_name IS NOT NULL AND canonical_name != ''
            GROUP BY canonical_name
            """
        )
        return {row['canonical_name']: row['roster_teams'] for row in cursor.fetchall()}

    def _load_official_transaction_history(self, canonical_names):
        normalized_targets = {
            canonical_name: normalize_name(canonical_name)
            for canonical_name in canonical_names
            if canonical_name
        }
        if not normalized_targets:
            return {}

        cursor = self.conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute(
            """
            SELECT team_code, transaction_text
            FROM official_mlb_transactions
            WHERE transaction_date BETWEEN DATE '2025-03-01' AND DATE '2025-11-30'
              AND team_code IS NOT NULL
              AND transaction_text IS NOT NULL
            """
        )

        history = {}
        for row in cursor.fetchall():
            transaction_text = normalize_name(row['transaction_text'])
            if not transaction_text:
                continue
            for canonical_name, normalized_canonical in normalized_targets.items():
                if normalized_canonical and normalized_canonical in transaction_text:
                    history.setdefault(canonical_name, set()).add(row['team_code'])

        return {
            canonical_name: '|'.join(sorted(team_codes))
            for canonical_name, team_codes in history.items()
        }

    def _review_status_for_resolution(self, source_table: str, raw_name: str, cleaned_name: str, resolution: dict) -> str:
        if not resolution['matched']:
            return 'pending'

        if resolution.get('team_validation') == 'team_mismatch':
            return 'pending_review'

        if resolution['match_type'] in {'fuzzy_same_last_name', 'first_initial_last', 'unique_last_name'}:
            return 'pending_review'

        if len(cleaned_name.split()) == 1:
            return 'pending_review'

        if source_table == 'play_by_play_plays' and cleaned_name != raw_name:
            return 'pending_review'

        return 'auto_matched'

    def _upsert_alias(self, raw_name: str, resolution: dict, source_table: str):
        cursor = self.conn.cursor(cursor_factory=RealDictCursor)
        canonical_name = resolution['canonical_name']
        normalized_name = resolution['normalized_name']
        alias_to_apply = resolution.get('alias_to_apply') or raw_name

        cursor.execute(
            """
            SELECT id, canonical_name, aliases
            FROM player_name_map
            WHERE canonical_name = %s
            """,
            (canonical_name,)
        )
        target = cursor.fetchone()

        if target:
            aliases = list(target['aliases']) if target['aliases'] else []
            if alias_to_apply not in aliases and normalized_name not in [normalize_name(alias) for alias in aliases]:
                aliases.append(alias_to_apply)
                cursor.execute(
                    """
                    UPDATE player_name_map
                    SET aliases = %s, updated_at = CURRENT_TIMESTAMP
                    WHERE id = %s
                    """,
                    (aliases, target['id'])
                )
                return 'alias_updated'
            return 'alias_exists'

        canonical, first_name, last_name = self._normalize_for_registry(alias_to_apply)
        cursor.execute(
            """
            INSERT INTO player_name_map
            (canonical_name, aliases, last_name, first_name, first_initial)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (canonical_name) DO NOTHING
            """,
            (canonical, [alias_to_apply], last_name, first_name, first_name[:1] if first_name else '')
        )
        return 'canonical_inserted' if cursor.rowcount > 0 else 'canonical_exists'

    def add_aliases_from_source(self, source_key: str):
        """Mutating path: update player_name_map from a configured source."""
        if not self.conn:
            logger.error("Cannot add aliases: no database connection")
            return False
        if source_key not in SOURCE_CONFIG:
            logger.error(f"Unknown source key: {source_key}")
            return False

        source_table, source_column = SOURCE_CONFIG[source_key]
        try:
            names = self._source_names(source_table, source_column)
            logger.info(f"Processing {len(names)} names from {source_table}.{source_column}")
            changed = 0

            for row in names:
                raw_name = row['raw_name'].strip()
                cleaned_name = self._clean_source_name(source_table, raw_name)
                if not cleaned_name:
                    continue
                resolution = resolve_name_match(cleaned_name, conn=self.conn)
                resolution['alias_to_apply'] = cleaned_name

                if resolution['match_type'] in {'ambiguous_last_name', 'unresolved', 'empty', 'lookup_error', 'no_db'}:
                    continue

                action = self._upsert_alias(raw_name, resolution, source_table)
                if action in {'alias_updated', 'canonical_inserted'}:
                    changed += 1

            self.conn.commit()
            logger.info(f"✅ Applied {changed} alias updates from {source_table}")
            return True
        except Exception as exc:
            logger.error(f"❌ Failed to add aliases from {source_table}: {exc}")
            self.conn.rollback()
            return False

    def audit_source(self, source_key: str):
        """Non-destructive path: write proposed resolutions to audit tables only."""
        if not self.conn:
            logger.error("Cannot audit source: no database connection")
            return False
        if source_key not in SOURCE_CONFIG:
            logger.error(f"Unknown source key: {source_key}")
            return False

        source_table, source_column = SOURCE_CONFIG[source_key]

        try:
            players_cursor = self.conn.cursor(cursor_factory=RealDictCursor)
            players_cursor.execute(
                """
                SELECT full_name, team_code
                FROM players
                WHERE full_name IS NOT NULL AND full_name != ''
                """
            )
            player_team_map = {row['full_name']: row['team_code'] for row in players_cursor.fetchall()}
            pbp_team_map = self._load_play_by_play_team_context() if source_table == 'play_by_play_plays' else {}
            official_roster_map = self._load_official_roster_history()
            source_names = self._source_names(source_table, source_column)
            canonical_candidates = set()
            for row in source_names:
                cleaned_name = self._clean_source_name(source_table, row['raw_name'].strip())
                if not cleaned_name:
                    continue
                canonical_candidates.add(resolve_name_match(cleaned_name, conn=self.conn)['canonical_name'])
            official_transaction_map = self._load_official_transaction_history(canonical_candidates)

            cursor = self.conn.cursor()
            names = source_names
            logger.info(f"Auditing {len(names)} names from {source_table}.{source_column}")

            cursor.execute("DELETE FROM player_name_alias_audit WHERE source_table = %s", (source_table,))
            cursor.execute("DELETE FROM player_name_resolution_audit WHERE source_table = %s", (source_table,))

            summary = {}
            alias_rows = 0

            for row in names:
                raw_name = row['raw_name'].strip()
                cleaned_name = self._clean_source_name(source_table, raw_name)
                if not cleaned_name:
                    continue
                resolution = resolve_name_match(cleaned_name, conn=self.conn)
                resolution['alias_to_apply'] = cleaned_name
                observed_teams = pbp_team_map.get(raw_name) if source_table == 'play_by_play_plays' else None
                resolution['observed_teams'] = observed_teams
                resolution['team_validation'] = self._team_validation_label(
                    source_table,
                    player_team_map.get(resolution['canonical_name']),
                    observed_teams
                )
                resolution['official_roster_teams'] = official_roster_map.get(resolution['canonical_name'])
                resolution['official_transaction_teams'] = official_transaction_map.get(resolution['canonical_name'])
                resolution['historical_team_validation'] = self._historical_team_validation_label(
                    observed_teams,
                    resolution['official_roster_teams'],
                    resolution['official_transaction_teams']
                )
                match_type = resolution['match_type']
                review_status = self._review_status_for_resolution(source_table, raw_name, cleaned_name, resolution)
                summary[match_type] = summary.get(match_type, 0) + 1

                cursor.execute(
                    """
                    INSERT INTO player_name_resolution_audit
                    (
                        source_table, source_column, raw_name, normalized_name,
                        proposed_canonical_name, matched, ambiguous, match_type,
                        candidate_count, candidates, review_status, updated_at
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
                    ON CONFLICT (source_table, source_column, raw_name) DO UPDATE SET
                        normalized_name = EXCLUDED.normalized_name,
                        proposed_canonical_name = EXCLUDED.proposed_canonical_name,
                        matched = EXCLUDED.matched,
                        ambiguous = EXCLUDED.ambiguous,
                        match_type = EXCLUDED.match_type,
                        candidate_count = EXCLUDED.candidate_count,
                        candidates = EXCLUDED.candidates,
                        updated_at = CURRENT_TIMESTAMP
                    """,
                        (
                            source_table,
                            source_column,
                            raw_name,
                        resolution['normalized_name'],
                        resolution['canonical_name'],
                        resolution['matched'],
                        resolution['ambiguous'],
                        match_type,
                        resolution['candidate_count'],
                        Json(resolution['candidates']),
                        'pending' if resolution['ambiguous'] or not resolution['matched'] else review_status,
                    )
                )

                if resolution['matched']:
                    cursor.execute(
                        """
                        INSERT INTO player_name_alias_audit
                        (
                            source_table, raw_name, alias_to_apply, normalized_name, proposed_canonical_name,
                            proposed_match_type, observed_teams, team_validation, official_roster_teams,
                            official_transaction_teams,
                            historical_team_validation, apply_action, review_status, updated_at
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
                        ON CONFLICT (source_table, raw_name, proposed_canonical_name) DO UPDATE SET
                            alias_to_apply = EXCLUDED.alias_to_apply,
                            proposed_match_type = EXCLUDED.proposed_match_type,
                            observed_teams = EXCLUDED.observed_teams,
                            team_validation = EXCLUDED.team_validation,
                            official_roster_teams = EXCLUDED.official_roster_teams,
                            official_transaction_teams = EXCLUDED.official_transaction_teams,
                            historical_team_validation = EXCLUDED.historical_team_validation,
                            apply_action = EXCLUDED.apply_action,
                            review_status = EXCLUDED.review_status,
                            updated_at = CURRENT_TIMESTAMP
                        """,
                        (
                            source_table,
                            raw_name,
                            cleaned_name,
                            resolution['normalized_name'],
                            resolution['canonical_name'],
                            match_type,
                            observed_teams,
                            resolution['team_validation'],
                            resolution['official_roster_teams'],
                            resolution['official_transaction_teams'],
                            resolution['historical_team_validation'],
                            'attach_alias',
                            review_status,
                        )
                    )
                    alias_rows += 1

            self.conn.commit()
            logger.info(f"✅ Audit complete for {source_table}: {summary}")
            logger.info(f"   Proposed alias rows recorded: {alias_rows}")
            return True
        except Exception as exc:
            logger.error(f"❌ Failed to audit {source_table}: {exc}")
            self.conn.rollback()
            return False

    def audit_all_sources(self):
        ok = True
        for source_key in ('hitter_exit_velocity', 'custom_batter_2025', 'play_by_play_plays'):
            ok = self.audit_source(source_key) and ok
        return ok

    def get_alias_review_summary(self, sample_limit: int = 10):
        if not self.conn:
            logger.error("Cannot get alias review summary: no database connection")
            return

        cursor = self.conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute(
            """
            SELECT source_table, review_status, proposed_match_type, COUNT(*) AS row_count
            FROM player_name_alias_audit
            GROUP BY source_table, review_status, proposed_match_type
            ORDER BY source_table, review_status, row_count DESC, proposed_match_type
            """
        )
        rows = cursor.fetchall()
        if not rows:
            logger.info("No alias audit rows found")
            return

        logger.info("\n📊 Alias Review Summary:")
        current_source = None
        for row in rows:
            if row['source_table'] != current_source:
                current_source = row['source_table']
                logger.info(f"   {current_source}:")
            logger.info(
                "     status=%s match_type=%s count=%s",
                row['review_status'],
                row['proposed_match_type'],
                row['row_count']
            )

        cursor.execute(
            """
            SELECT source_table, raw_name, alias_to_apply, proposed_canonical_name, proposed_match_type, review_status
            FROM player_name_alias_audit
            WHERE review_status IN ('pending_review', 'pending')
            ORDER BY source_table, raw_name
            LIMIT %s
            """,
            (sample_limit,)
        )
        samples = cursor.fetchall()
        if samples:
            logger.info("\n   Pending review samples:")
            for row in samples:
                logger.info(
                    "     %s | %s => %s -> %s (%s, %s)",
                    row['source_table'],
                    row['raw_name'],
                    row['alias_to_apply'] or row['raw_name'],
                    row['proposed_canonical_name'],
                    row['proposed_match_type'],
                    row['review_status']
                )

    def set_alias_review_status(self, source_table: str, raw_name: str, review_status: str, proposed_canonical_name: str = None):
        if not self.conn:
            logger.error("Cannot set review status: no database connection")
            return False

        if review_status not in {'pending', 'pending_review', 'approved', 'rejected', 'auto_matched'}:
            logger.error(f"Unsupported review status: {review_status}")
            return False

        try:
            cursor = self.conn.cursor()
            if proposed_canonical_name:
                cursor.execute(
                    """
                    UPDATE player_name_alias_audit
                    SET review_status = %s, updated_at = CURRENT_TIMESTAMP
                    WHERE source_table = %s AND raw_name = %s AND proposed_canonical_name = %s
                    """,
                    (review_status, source_table, raw_name, proposed_canonical_name)
                )
            else:
                cursor.execute(
                    """
                    UPDATE player_name_alias_audit
                    SET review_status = %s, updated_at = CURRENT_TIMESTAMP
                    WHERE source_table = %s AND raw_name = %s
                    """,
                    (review_status, source_table, raw_name)
                )
            self.conn.commit()
            logger.info("✅ Updated %s alias audit row(s)", cursor.rowcount)
            return True
        except Exception as exc:
            logger.error(f"❌ Failed to set review status: {exc}")
            self.conn.rollback()
            return False

    def set_alias_review_status_bulk(
        self,
        source_table: str,
        alias_to_apply: str,
        review_status: str,
        proposed_canonical_name: str = None,
        dry_run: bool = False,
    ):
        if not self.conn:
            logger.error("Cannot set bulk review status: no database connection")
            return False

        if review_status not in {'pending', 'pending_review', 'approved', 'rejected', 'auto_matched'}:
            logger.error(f"Unsupported review status: {review_status}")
            return False

        try:
            cursor = self.conn.cursor(cursor_factory=RealDictCursor)
            if proposed_canonical_name:
                cursor.execute(
                    """
                    SELECT source_table, raw_name, alias_to_apply, proposed_canonical_name, proposed_match_type, review_status
                    FROM player_name_alias_audit
                    WHERE source_table = %s
                      AND alias_to_apply = %s
                      AND proposed_canonical_name = %s
                    ORDER BY raw_name
                    """,
                    (source_table, alias_to_apply, proposed_canonical_name)
                )
            else:
                cursor.execute(
                    """
                    SELECT source_table, raw_name, alias_to_apply, proposed_canonical_name, proposed_match_type, review_status
                    FROM player_name_alias_audit
                    WHERE source_table = %s
                      AND alias_to_apply = %s
                    ORDER BY proposed_canonical_name, raw_name
                    """,
                    (source_table, alias_to_apply)
                )

            rows = cursor.fetchall()
            if not rows:
                logger.info("No alias audit rows matched source_table=%s alias_to_apply=%s", source_table, alias_to_apply)
                return True

            logger.info(
                "Matched %s row(s) for bulk review update: source_table=%s alias_to_apply=%s",
                len(rows),
                source_table,
                alias_to_apply
            )
            for row in rows[:10]:
                logger.info(
                    "  %s => %s (%s, current=%s)",
                    row['raw_name'],
                    row['proposed_canonical_name'],
                    row['proposed_match_type'],
                    row['review_status']
                )
            if len(rows) > 10:
                logger.info("  ... %s more row(s)", len(rows) - 10)

            if dry_run:
                logger.info("✅ Dry run only; no review statuses updated")
                return True

            if proposed_canonical_name:
                cursor.execute(
                    """
                    UPDATE player_name_alias_audit
                    SET review_status = %s, updated_at = CURRENT_TIMESTAMP
                    WHERE source_table = %s
                      AND alias_to_apply = %s
                      AND proposed_canonical_name = %s
                    """,
                    (review_status, source_table, alias_to_apply, proposed_canonical_name)
                )
            else:
                cursor.execute(
                    """
                    UPDATE player_name_alias_audit
                    SET review_status = %s, updated_at = CURRENT_TIMESTAMP
                    WHERE source_table = %s
                      AND alias_to_apply = %s
                    """,
                    (review_status, source_table, alias_to_apply)
                )

            self.conn.commit()
            logger.info("✅ Bulk updated %s alias audit row(s)", cursor.rowcount)
            return True
        except Exception as exc:
            logger.error(f"❌ Failed to set bulk review status: {exc}")
            self.conn.rollback()
            return False

    def apply_alias_audit(self, review_statuses=None, dry_run: bool = False):
        """
        Apply reviewed alias proposals into player_name_map.

        This mutates only player_name_map, never the source tables.
        """
        if not self.conn:
            logger.error("Cannot apply alias audit: no database connection")
            return False

        review_statuses = review_statuses or ['approved']

        try:
            cursor = self.conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute(
                """
                SELECT source_table, raw_name, alias_to_apply, normalized_name, proposed_canonical_name, proposed_match_type, team_validation, review_status
                FROM player_name_alias_audit
                WHERE review_status = ANY(%s)
                ORDER BY source_table, raw_name
                """,
                (review_statuses,)
            )
            rows = cursor.fetchall()
            logger.info("Applying %s reviewed alias row(s) with statuses=%s", len(rows), review_statuses)

            applied = 0
            skipped = 0

            for row in rows:
                resolution = {
                    'canonical_name': row['proposed_canonical_name'],
                    'normalized_name': row['normalized_name'],
                    'alias_to_apply': row['alias_to_apply'] or row['raw_name'],
                }
                if dry_run:
                    logger.info(
                        "DRY RUN | %s | %s => %s -> %s (%s, team=%s)",
                        row['source_table'],
                        row['raw_name'],
                        row['alias_to_apply'] or row['raw_name'],
                        row['proposed_canonical_name'],
                        row['proposed_match_type'],
                        row['team_validation']
                    )
                    applied += 1
                    continue

                if row['review_status'] == 'auto_matched' and row['team_validation'] == 'team_mismatch':
                    skipped += 1
                    continue

                action = self._upsert_alias(row['raw_name'], resolution, row['source_table'])
                if action in {'alias_updated', 'canonical_inserted'}:
                    applied += 1
                else:
                    skipped += 1

            if dry_run:
                logger.info("✅ Dry run complete: %s rows would be applied", applied)
                return True

            self.conn.commit()
            logger.info("✅ Applied alias audit rows: updated=%s skipped=%s", applied, skipped)
            return True
        except Exception as exc:
            logger.error(f"❌ Failed to apply alias audit: {exc}")
            self.conn.rollback()
            return False

    def export_pending_reviews(self, output_path: str, source_table: str = None):
        """Export pending review rows to CSV for manual triage."""
        if not self.conn:
            logger.error("Cannot export pending reviews: no database connection")
            return False

        try:
            query = """
                SELECT
                    aa.source_table,
                    aa.raw_name,
                    aa.alias_to_apply,
                    aa.proposed_canonical_name,
                    aa.proposed_match_type,
                    aa.review_status,
                    ra.normalized_name,
                    ra.candidate_count,
                    ra.candidates,
                    ra.ambiguous
                FROM player_name_alias_audit aa
                LEFT JOIN player_name_resolution_audit ra
                  ON ra.source_table = aa.source_table
                 AND ra.raw_name = aa.raw_name
                WHERE aa.review_status IN ('pending', 'pending_review')
            """
            params = []
            if source_table:
                query += " AND aa.source_table = %s"
                params.append(source_table)
            query += " ORDER BY aa.source_table, aa.alias_to_apply, aa.raw_name"

            df = pd.read_sql(query, self.conn, params=params if params else None)
            output = Path(output_path)
            output.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output, index=False)
            logger.info("✅ Exported %s pending review row(s) to %s", len(df), output)
            return True
        except Exception as exc:
            logger.error(f"❌ Failed to export pending reviews: {exc}")
            return False

    def export_grouped_pending_reviews(self, output_path: str, source_table: str = None):
        """
        Export grouped pending review rows for faster approval workflows.

        For play-by-play rows, include observed matchup teams derived from inning half
        and compare them to the proposed player's current team_code when available.
        """
        if not self.conn:
            logger.error("Cannot export grouped pending reviews: no database connection")
            return False

        try:
            query = """
                SELECT
                    aa.source_table,
                    aa.alias_to_apply,
                    aa.proposed_canonical_name,
                    aa.proposed_match_type,
                    aa.review_status,
                    MAX(aa.observed_teams) AS observed_teams,
                    MAX(aa.team_validation) AS team_validation,
                    MAX(aa.official_roster_teams) AS official_roster_teams,
                    MAX(aa.official_transaction_teams) AS official_transaction_teams,
                    MAX(aa.historical_team_validation) AS historical_team_validation,
                    COUNT(*) AS raw_variant_count,
                    STRING_AGG(DISTINCT aa.raw_name, ' || ' ORDER BY aa.raw_name) AS raw_name_samples,
                    MAX(ra.candidate_count) AS candidate_count,
                    MAX(ra.candidates::text) AS candidates,
                    MAX(ra.ambiguous::int) AS ambiguous_flag
                FROM player_name_alias_audit aa
                LEFT JOIN player_name_resolution_audit ra
                  ON ra.source_table = aa.source_table
                 AND ra.raw_name = aa.raw_name
                WHERE aa.review_status IN ('pending', 'pending_review')
            """
            params = []
            if source_table:
                query += " AND aa.source_table = %s"
                params.append(source_table)
            query += """
                GROUP BY
                    aa.source_table,
                    aa.alias_to_apply,
                    aa.proposed_canonical_name,
                    aa.proposed_match_type,
                    aa.review_status
                ORDER BY aa.source_table, raw_variant_count DESC, aa.alias_to_apply
            """

            grouped = pd.read_sql(query, self.conn, params=params if params else None)
            if grouped.empty:
                output = Path(output_path)
                output.parent.mkdir(parents=True, exist_ok=True)
                grouped.to_csv(output, index=False)
                logger.info("✅ Exported empty grouped review file to %s", output)
                return True

            players = pd.read_sql(
                """
                SELECT full_name, team_code, position, player_type, active
                FROM players
                WHERE full_name IS NOT NULL AND full_name != ''
                """,
                self.conn
            )
            players = players.drop_duplicates(subset=['full_name'])
            grouped = grouped.merge(
                players,
                left_on='proposed_canonical_name',
                right_on='full_name',
                how='left'
            ).drop(columns=['full_name'])

            output = Path(output_path)
            output.parent.mkdir(parents=True, exist_ok=True)
            grouped.to_csv(output, index=False)
            logger.info("✅ Exported %s grouped pending review row(s) to %s", len(grouped), output)
            return True
        except Exception as exc:
            logger.error(f"❌ Failed to export grouped pending reviews: {exc}")
            return False

    def export_team_resolution_gaps(self, output_path: str):
        """
        Export rows where team validation needs attention.

        This prioritizes:
        - team_mismatch
        - no_player_team
        - no_observed_team
        """
        if not self.conn:
            logger.error("Cannot export team resolution gaps: no database connection")
            return False

        try:
            query = """
                SELECT
                    source_table,
                    alias_to_apply,
                    proposed_canonical_name,
                    proposed_match_type,
                    review_status,
                    observed_teams,
                    team_validation,
                    official_roster_teams,
                    MAX(official_transaction_teams) AS official_transaction_teams,
                    historical_team_validation,
                    COUNT(*) AS raw_variant_count,
                    STRING_AGG(DISTINCT raw_name, ' || ' ORDER BY raw_name) AS raw_name_samples
                FROM player_name_alias_audit
                WHERE source_table = 'play_by_play_plays'
                  AND team_validation IN ('team_mismatch', 'no_player_team', 'no_observed_team')
                GROUP BY
                    source_table,
                    alias_to_apply,
                    proposed_canonical_name,
                    proposed_match_type,
                    review_status,
                    observed_teams,
                    team_validation,
                    official_roster_teams,
                    historical_team_validation
                ORDER BY
                    CASE team_validation
                        WHEN 'team_mismatch' THEN 1
                        WHEN 'no_player_team' THEN 2
                        WHEN 'no_observed_team' THEN 3
                        ELSE 4
                    END,
                    raw_variant_count DESC,
                    alias_to_apply
            """
            gaps = pd.read_sql(query, self.conn)

            players = pd.read_sql(
                """
                SELECT full_name, team_code, position, player_type, active
                FROM players
                WHERE full_name IS NOT NULL AND full_name != ''
                """,
                self.conn
            ).drop_duplicates(subset=['full_name'])

            gaps = gaps.merge(
                players,
                left_on='proposed_canonical_name',
                right_on='full_name',
                how='left'
            ).drop(columns=['full_name'])

            output = Path(output_path)
            output.parent.mkdir(parents=True, exist_ok=True)
            gaps.to_csv(output, index=False)
            logger.info("✅ Exported %s team-resolution gap row(s) to %s", len(gaps), output)
            return True
        except Exception as exc:
            logger.error(f"❌ Failed to export team resolution gaps: {exc}")
            return False

    def _team_validation_label(self, source_table: str, player_team_code: str, observed_teams: str):
        if source_table != 'play_by_play_plays':
            return 'not_available'
        if observed_teams is None or (isinstance(observed_teams, float) and pd.isna(observed_teams)) or str(observed_teams).strip() == '':
            return 'no_observed_team'
        if player_team_code is None or (isinstance(player_team_code, float) and pd.isna(player_team_code)) or str(player_team_code).strip() == '':
            return 'no_player_team'

        observed = {team.strip() for team in str(observed_teams).split('|') if team and team.strip()}
        return 'team_match' if str(player_team_code).strip() in observed else 'team_mismatch'

    def _historical_team_validation_label(self, observed_teams: str, official_roster_teams: str, official_transaction_teams: str):
        if observed_teams is None or (isinstance(observed_teams, float) and pd.isna(observed_teams)) or str(observed_teams).strip() == '':
            return 'no_observed_team'

        observed = {team.strip() for team in str(observed_teams).split('|') if team and team.strip()}
        roster = {team.strip() for team in str(official_roster_teams).split('|') if official_roster_teams and team and team.strip()}
        transactions = {team.strip() for team in str(official_transaction_teams).split('|') if official_transaction_teams and team and team.strip()}

        if roster and observed & roster:
            return 'official_roster_match'
        if transactions and observed & transactions:
            return 'official_transaction_match'
        if roster or transactions:
            return 'official_history_mismatch'
        return 'no_official_history'

    def get_stats(self):
        """Get statistics on the player_name_map table."""
        if not self.conn:
            logger.error("Cannot get stats: no database connection")
            return

        try:
            cursor = self.conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute("SELECT COUNT(*) AS total_entries FROM player_name_map")
            total = cursor.fetchone()['total_entries']
            cursor.execute("SELECT COUNT(*) AS with_mlb_id FROM player_name_map WHERE mlb_id IS NOT NULL")
            with_ids = cursor.fetchone()['with_mlb_id']
            cursor.execute("SELECT COUNT(*) AS with_aliases FROM player_name_map WHERE array_length(aliases, 1) > 0")
            with_aliases = cursor.fetchone()['with_aliases']

            logger.info("\n📊 Player Name Map Statistics:")
            logger.info(f"   Total entries: {total}")
            logger.info(f"   With MLB IDs: {with_ids}")
            logger.info(f"   With aliases: {with_aliases}")
        except Exception as exc:
            logger.error(f"Error getting stats: {exc}")

    def get_audit_stats(self):
        if not self.conn:
            logger.error("Cannot get audit stats: no database connection")
            return

        try:
            cursor = self.conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute(
                """
                SELECT source_table, match_type, COUNT(*) AS row_count
                FROM player_name_resolution_audit
                GROUP BY source_table, match_type
                ORDER BY source_table, row_count DESC, match_type
                """
            )
            rows = cursor.fetchall()
            if not rows:
                logger.info("No audit rows found")
                return

            logger.info("\n📊 Player Name Audit Statistics:")
            current_source = None
            for row in rows:
                if row['source_table'] != current_source:
                    current_source = row['source_table']
                    logger.info(f"   {current_source}:")
                logger.info(f"     {row['match_type']}: {row['row_count']}")
        except Exception as exc:
            logger.error(f"Error getting audit stats: {exc}")


def main():
    parser = argparse.ArgumentParser(description='Player Name Registry Migration / Audit')
    parser.add_argument('--create-table', action='store_true', help='Create player_name_map table')
    parser.add_argument('--seed', action='store_true', help='Seed player_name_map from all sources (mutates player_name_map)')
    parser.add_argument('--all', action='store_true', help='Create player_name_map table and seed it (mutates player_name_map)')
    parser.add_argument('--stats', action='store_true', help='Show player_name_map statistics')
    parser.add_argument('--create-audit-tables', action='store_true', help='Create non-destructive audit tables')
    parser.add_argument('--audit-source', choices=list(SOURCE_CONFIG.keys()), help='Audit a single configured source without mutating player_name_map')
    parser.add_argument('--audit-all', action='store_true', help='Audit configured sources without mutating player_name_map')
    parser.add_argument('--audit-stats', action='store_true', help='Show audit table statistics')
    parser.add_argument('--alias-review-summary', action='store_true', help='Show alias audit review summary and pending samples')
    parser.add_argument('--set-review-status', action='store_true', help='Set review status for alias audit rows')
    parser.add_argument('--source-table', help='Source table for review-status updates')
    parser.add_argument('--raw-name', help='Raw name for review-status updates')
    parser.add_argument('--proposed-canonical-name', help='Optional canonical name for review-status updates')
    parser.add_argument('--review-status', help='Review status to set or apply, e.g. approved/rejected')
    parser.add_argument('--apply-approved', action='store_true', help='Apply approved alias audit rows into player_name_map')
    parser.add_argument('--apply-auto-matched', action='store_true', help='Apply auto_matched alias audit rows into player_name_map')
    parser.add_argument('--dry-run', action='store_true', help='Dry run for apply commands')
    parser.add_argument('--export-pending-reviews', action='store_true', help='Export pending review rows to CSV')
    parser.add_argument('--output-path', help='Output file path for export commands')
    parser.add_argument('--set-review-status-bulk', action='store_true', help='Set review status for all rows sharing a cleaned alias')
    parser.add_argument('--alias-to-apply', help='Cleaned alias value for bulk review updates')
    parser.add_argument('--export-grouped-pending-reviews', action='store_true', help='Export grouped pending review rows with team context to CSV')
    parser.add_argument('--export-team-resolution-gaps', action='store_true', help='Export team mismatch / missing-team review rows to CSV')

    args = parser.parse_args()
    migration = PlayerNameMigration()

    try:
        if args.all or args.create_table:
            logger.info("Step 1: Creating player_name_map table...")
            migration.create_table()

        if args.create_audit_tables:
            logger.info("Step 2: Creating audit tables...")
            migration.create_audit_tables()

        if args.all or args.seed:
            logger.info("Step 3: Seeding from players table...")
            migration.seed_from_players_table()

            logger.info("Step 4: Adding aliases from hitter_exit_velocity...")
            migration.add_aliases_from_source('hitter_exit_velocity')

            logger.info("Step 5: Adding aliases from custom_batter_2025...")
            migration.add_aliases_from_source('custom_batter_2025')

            logger.info("Step 6: Adding aliases from play_by_play_plays...")
            migration.add_aliases_from_source('play_by_play_plays')

        if args.audit_source:
            logger.info(f"Auditing source: {args.audit_source}")
            migration.audit_source(args.audit_source)

        if args.audit_all:
            logger.info("Auditing all configured sources...")
            migration.audit_all_sources()

        if args.stats:
            migration.get_stats()

        if args.audit_stats:
            migration.get_audit_stats()

        if args.alias_review_summary:
            migration.get_alias_review_summary()

        if args.set_review_status:
            if not args.source_table or not args.raw_name or not args.review_status:
                parser.error('--set-review-status requires --source-table, --raw-name, and --review-status')
            migration.set_alias_review_status(
                args.source_table,
                args.raw_name,
                args.review_status,
                proposed_canonical_name=args.proposed_canonical_name
            )

        if args.set_review_status_bulk:
            if not args.source_table or not args.alias_to_apply or not args.review_status:
                parser.error('--set-review-status-bulk requires --source-table, --alias-to-apply, and --review-status')
            migration.set_alias_review_status_bulk(
                args.source_table,
                args.alias_to_apply,
                args.review_status,
                proposed_canonical_name=args.proposed_canonical_name,
                dry_run=args.dry_run
            )

        if args.apply_approved or args.apply_auto_matched:
            review_statuses = []
            if args.apply_approved:
                review_statuses.append('approved')
            if args.apply_auto_matched:
                review_statuses.append('auto_matched')
            migration.apply_alias_audit(review_statuses=review_statuses, dry_run=args.dry_run)

        if args.export_pending_reviews:
            output_path = args.output_path or 'output/pending_name_reviews.csv'
            migration.export_pending_reviews(output_path, source_table=args.source_table)

        if args.export_grouped_pending_reviews:
            output_path = args.output_path or 'output/grouped_pending_name_reviews.csv'
            migration.export_grouped_pending_reviews(output_path, source_table=args.source_table)

        if args.export_team_resolution_gaps:
            output_path = args.output_path or 'output/team_resolution_gaps.csv'
            migration.export_team_resolution_gaps(output_path)

        if not any([
            args.all, args.create_table, args.seed, args.stats,
            args.create_audit_tables, args.audit_source, args.audit_all, args.audit_stats,
            args.alias_review_summary, args.set_review_status, args.apply_approved, args.apply_auto_matched,
            args.export_pending_reviews, args.set_review_status_bulk, args.export_grouped_pending_reviews,
            args.export_team_resolution_gaps
        ]):
            logger.info("No action specified. Use --help for options.")
    finally:
        migration.close()


if __name__ == '__main__':
    main()
