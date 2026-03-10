#!/usr/bin/env python3
"""
Official MLB team-history ingestion.

Phase 1:
- store raw official MLB transaction rows by date
- store official team roster snapshots by team/date from MLB press releases

This creates new tables only. It does not modify existing source tables.
"""

import argparse
import logging
import os
import re
from datetime import datetime, timedelta
from typing import Iterable, List, Tuple

import pandas as pd
import psycopg2
import requests
from psycopg2.extras import RealDictCursor

from name_utils import normalize_to_canonical, normalize_name

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

HEADERS = {
    'User-Agent': 'ProjectionAI/1.0 (+local team history ingestion)'
}

TEAM_NAME_TO_CODE = {
    'Arizona Diamondbacks': 'ARI',
    'Atlanta Braves': 'ATL',
    'Athletics': 'ATH',
    'Baltimore Orioles': 'BAL',
    'Boston Red Sox': 'BOS',
    'Chicago Cubs': 'CHC',
    'Chicago White Sox': 'CWS',
    'Cincinnati Reds': 'CIN',
    'Cleveland Guardians': 'CLE',
    'Colorado Rockies': 'COL',
    'Detroit Tigers': 'DET',
    'Houston Astros': 'HOU',
    'Kansas City Royals': 'KC',
    'Los Angeles Angels': 'LAA',
    'Los Angeles Dodgers': 'LAD',
    'Miami Marlins': 'MIA',
    'Milwaukee Brewers': 'MIL',
    'Minnesota Twins': 'MIN',
    'New York Mets': 'NYM',
    'New York Yankees': 'NYY',
    'Philadelphia Phillies': 'PHI',
    'Pittsburgh Pirates': 'PIT',
    'San Diego Padres': 'SD',
    'San Francisco Giants': 'SF',
    'Seattle Mariners': 'SEA',
    'St. Louis Cardinals': 'STL',
    'Tampa Bay Rays': 'TB',
    'Texas Rangers': 'TEX',
    'Toronto Blue Jays': 'TOR',
    'Washington Nationals': 'WSH',
}
OPENING_DAY_2025_ROSTER_URLS = {
    'KC': 'https://www.mlb.com/press-release/press-release-royals-announce-2025-opening-day-roster',
    'COL': 'https://www.mlb.com/press-release/press-release-colorado-rockies-announce-2025-opening-day-roster',
    'BOS': 'https://www.mlb.com/news/press-release-red-sox-set-2025-opening-day-roster',
    'STL': 'https://www.mlb.com/press-release/press-release-cardinals-announce-2025-opening-day-roster',
    'ATH': 'https://www.mlb.com/press-release/press-release-athletics-announce-opening-day-2025-roster',
    'TB': 'https://www.mlb.com/press-release/press-release-rays-announce-2025-opening-day-roster',
    'LAD': 'https://www.mlb.com/press-release/press-release-dodgers-announce-opening-day-26-man-roster-x9036',
    'TOR': 'https://www.mlb.com/bluejays/press-release/press-release-blue-jays-announce-opening-day-roster-x8327',
    'SD': 'https://www.mlb.com/padres/press-release/press-release-padres-announce-2025-opening-day-roster',
}

ROSTER_SECTION_RE = re.compile(
    r'^(Right-handed Pitchers|Left-handed Pitchers|Pitchers|Catchers|Infielders|Outfielders|'
    r'Infielder/Outfielders|Infielder/Outfielder|Utility Players|RHP|LHP|C|INF|OF)\s*(?:\(\d+\))?:\s*(.*)$',
    re.IGNORECASE
)


class PlayerTeamHistoryBuilder:
    def __init__(self):
        self.conn = self._connect()

    def _connect(self):
        return psycopg2.connect(
            host=os.getenv('DB_HOST', '192.168.1.23'),
            port=int(os.getenv('DB_PORT', 5432)),
            database=os.getenv('DB_NAME', 'baseball_migration_test'),
            user=os.getenv('DB_USER', 'postgres'),
            password=os.getenv('DB_PASSWORD', 'korn5676')
        )

    def close(self):
        if self.conn:
            self.conn.close()

    def create_tables(self):
        cursor = self.conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS official_mlb_transactions (
                id SERIAL PRIMARY KEY,
                transaction_date DATE NOT NULL,
                page_number INTEGER NOT NULL DEFAULT 1,
                team_name TEXT,
                team_code VARCHAR(5),
                transaction_text TEXT NOT NULL,
                source_url TEXT NOT NULL,
                source_type VARCHAR(30) NOT NULL DEFAULT 'transactions_page',
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(transaction_date, page_number, team_name, transaction_text, source_url)
            );
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS official_team_roster_snapshots (
                id SERIAL PRIMARY KEY,
                snapshot_date DATE NOT NULL,
                team_code VARCHAR(5) NOT NULL,
                team_name TEXT,
                raw_name TEXT NOT NULL,
                normalized_name TEXT,
                canonical_name TEXT,
                source_url TEXT NOT NULL,
                source_type VARCHAR(30) NOT NULL DEFAULT 'opening_day_roster',
                notes TEXT,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(snapshot_date, team_code, raw_name, source_url)
            );
            """
        )
        self.conn.commit()
        logger.info("✅ official team-history tables are ready")

    def _infer_team_from_transaction(self, transaction_text: str) -> Tuple[str, str]:
        if not transaction_text:
            return None, None

        cleaned = re.sub(r'^\d{2}/\d{2}/\d{2}\s+', '', transaction_text).strip()
        for team_name, team_code in TEAM_NAME_TO_CODE.items():
            if cleaned.startswith(team_name):
                return team_name, team_code
            if f" by {team_name}" in cleaned:
                return team_name, team_code
        return None, None

    def ingest_transactions_for_date(self, target_date: str, max_pages: int = 3):
        date_obj = datetime.strptime(target_date, '%Y-%m-%d').date()
        cleanup_cursor = self.conn.cursor()
        cleanup_cursor.execute(
            "DELETE FROM official_mlb_transactions WHERE transaction_date = %s",
            (date_obj,)
        )
        self.conn.commit()
        inserted = 0
        for page_number in range(1, max_pages + 1):
            url = f"https://www.mlb.com/transactions/{date_obj:%Y/%m/%d}/p-{page_number}"
            try:
                tables = pd.read_html(url)
            except ValueError:
                logger.info("No transaction table found at %s", url)
                break

            if not tables:
                break

            table = tables[0]
            if table.empty or 'Transaction' not in table.columns:
                break

            rows = []
            for _, row in table.iterrows():
                team_name = str(row.get('Team', '')).strip() or None
                if team_name == 'nan':
                    team_name = None
                transaction_text = str(row.get('Transaction', '')).strip()
                if not transaction_text:
                    continue
                inferred_team_name, inferred_team_code = self._infer_team_from_transaction(transaction_text)
                rows.append((
                    team_name or inferred_team_name,
                    TEAM_NAME_TO_CODE.get(team_name) if team_name else inferred_team_code,
                    transaction_text
                ))

            if not rows:
                break

            cursor = self.conn.cursor()
            for team_name, team_code, transaction_text in rows:
                cursor.execute(
                    """
                    INSERT INTO official_mlb_transactions
                    (transaction_date, page_number, team_name, team_code, transaction_text, source_url)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (transaction_date, page_number, team_name, transaction_text, source_url)
                    DO UPDATE SET
                        team_code = EXCLUDED.team_code
                    """,
                    (date_obj, page_number, team_name, team_code, transaction_text, url)
                )
                inserted += cursor.rowcount

            self.conn.commit()
            logger.info("Loaded %s transaction rows from %s", len(rows), url)

        logger.info("✅ Inserted %s official transaction rows for %s", inserted, target_date)

    def ingest_transactions_date_range(self, start_date: str, end_date: str, max_pages: int = 3):
        start = datetime.strptime(start_date, '%Y-%m-%d').date()
        end = datetime.strptime(end_date, '%Y-%m-%d').date()
        current = start
        while current <= end:
            self.ingest_transactions_for_date(current.isoformat(), max_pages=max_pages)
            current += timedelta(days=1)

    def _fetch_article_text(self, url: str) -> str:
        try:
            from bs4 import BeautifulSoup
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("bs4 is required for roster article ingestion") from exc

        response = requests.get(url, headers=HEADERS, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        text = soup.get_text('\n')
        text = re.sub(r'\n+', '\n', text)
        return text

    def _extract_team_name_from_title(self, text: str) -> str:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        title = lines[0] if lines else ''
        for team_name in TEAM_NAME_TO_CODE:
            if team_name in title:
                return team_name
        for line in lines[:10]:
            for team_name in TEAM_NAME_TO_CODE:
                if team_name in line:
                    return team_name
        return ''

    def _parse_roster_names(self, article_text: str) -> List[str]:
        names = []
        lines = [line.strip().lstrip('*').strip() for line in article_text.splitlines() if line.strip()]
        for idx, line in enumerate(lines):
            match = ROSTER_SECTION_RE.match(line)
            if not match:
                continue

            raw_players = match.group(2).strip()
            if not raw_players and idx + 1 < len(lines):
                raw_players = lines[idx + 1].strip()

            raw_players = re.sub(r'\*\*', '', raw_players)
            parts = [part.strip(' .') for part in raw_players.split(',') if part.strip()]
            names.extend(parts)
        deduped = []
        seen = set()
        for name in names:
            normalized = normalize_name(name)
            if normalized and normalized not in seen:
                deduped.append(name)
                seen.add(normalized)
        return deduped

    def ingest_roster_snapshot(self, team_code: str, snapshot_date: str, source_url: str, source_type: str = 'opening_day_roster'):
        article_text = self._fetch_article_text(source_url)
        snapshot_dt = datetime.strptime(snapshot_date, '%Y-%m-%d').date()
        team_name = next((name for name, code in TEAM_NAME_TO_CODE.items() if code == team_code), '') or self._extract_team_name_from_title(article_text)
        names = self._parse_roster_names(article_text)

        if not names:
            raise ValueError(f'No roster names parsed from {source_url}')

        cursor = self.conn.cursor()
        inserted = 0
        for raw_name in names:
            canonical_name = normalize_to_canonical(raw_name, self.conn)
            cursor.execute(
                """
                INSERT INTO official_team_roster_snapshots
                (snapshot_date, team_code, team_name, raw_name, normalized_name, canonical_name, source_url, source_type)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT DO NOTHING
                """,
                (
                    snapshot_dt,
                    team_code,
                    team_name,
                    raw_name,
                    normalize_name(raw_name),
                    canonical_name,
                    source_url,
                    source_type,
                )
            )
            inserted += cursor.rowcount

        self.conn.commit()
        logger.info("✅ Inserted %s roster snapshot rows for %s on %s", inserted, team_code, snapshot_date)

    def ingest_known_opening_day_rosters_2025(self, team_codes: List[str] = None):
        codes = team_codes or list(OPENING_DAY_2025_ROSTER_URLS.keys())
        for team_code in codes:
            if team_code not in OPENING_DAY_2025_ROSTER_URLS:
                logger.warning("Skipping unknown team code for known roster batch: %s", team_code)
                continue
            url = OPENING_DAY_2025_ROSTER_URLS[team_code]
            try:
                self.ingest_roster_snapshot(team_code, '2025-03-27', url, source_type='opening_day_roster')
            except Exception as exc:
                logger.error("Failed to ingest known opening day roster for %s: %s", team_code, exc)


def main():
    parser = argparse.ArgumentParser(description='Build official MLB player team history support tables')
    parser.add_argument('--create-tables', action='store_true', help='Create official team-history tables')
    parser.add_argument('--transactions-date', help='Ingest official MLB transactions for one YYYY-MM-DD date')
    parser.add_argument('--transactions-start', help='Start date for transaction ingestion')
    parser.add_argument('--transactions-end', help='End date for transaction ingestion')
    parser.add_argument('--max-pages', type=int, default=3, help='Max transaction pages to fetch per date')
    parser.add_argument('--roster-url', help='Official MLB roster article URL')
    parser.add_argument('--team-code', help='Team code for roster snapshot ingestion')
    parser.add_argument('--snapshot-date', help='Snapshot date YYYY-MM-DD for roster ingestion')
    parser.add_argument('--known-opening-day-2025', action='store_true', help='Load a batch of known official 2025 Opening Day roster snapshots')
    parser.add_argument('--team-codes', help='Comma-separated team codes for known opening-day batch')

    args = parser.parse_args()
    builder = PlayerTeamHistoryBuilder()

    try:
        if args.create_tables:
            builder.create_tables()

        if args.transactions_date:
            builder.ingest_transactions_for_date(args.transactions_date, max_pages=args.max_pages)

        if args.transactions_start and args.transactions_end:
            builder.ingest_transactions_date_range(args.transactions_start, args.transactions_end, max_pages=args.max_pages)

        if args.roster_url:
            if not args.team_code or not args.snapshot_date:
                parser.error('--roster-url requires --team-code and --snapshot-date')
            builder.ingest_roster_snapshot(args.team_code, args.snapshot_date, args.roster_url)

        if args.known_opening_day_2025:
            team_codes = [code.strip().upper() for code in args.team_codes.split(',')] if args.team_codes else None
            builder.ingest_known_opening_day_rosters_2025(team_codes=team_codes)

        if not any([
            args.create_tables,
            args.transactions_date,
            args.transactions_start and args.transactions_end,
            args.roster_url,
            args.known_opening_day_2025,
        ]):
            logger.info('No action specified. Use --help for options.')
    finally:
        builder.close()


if __name__ == '__main__':
    main()
