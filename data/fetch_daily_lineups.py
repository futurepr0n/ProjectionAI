#!/usr/bin/env python3
"""
Fetch MLB starting lineups from MLB.com and upsert them into daily_lineups.

This replaces the old server-local JSON updater workflow with a ProjectionAI-
local ingester that writes directly to the shared Postgres table used by the app.
"""

from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

import psycopg2
from bs4 import BeautifulSoup
from psycopg2.extras import Json, RealDictCursor
import requests


BASE_URL = "https://www.mlb.com/starting-lineups"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    )
}


TEAM_CODE_ALIASES = {
    "AZ": "ARI",
    "ARI": "ARI",
    "ATH": "OAK",
    "CHW": "CWS",
    "CWS": "CWS",
    "KC": "KC",
    "KCR": "KC",
    "OAK": "OAK",
    "SD": "SD",
    "SDP": "SD",
    "SF": "SF",
    "SFG": "SF",
    "TB": "TB",
    "TBR": "TB",
    "WSH": "WSH",
    "WSN": "WSH",
}


def normalize_team_code(team_code: str) -> str:
    code = str(team_code or "").strip().upper()
    return TEAM_CODE_ALIASES.get(code, code)


def connect_db():
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "192.168.1.23"),
        port=int(os.getenv("DB_PORT", 5432)),
        database=os.getenv("DB_NAME", "baseball_migration_test"),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD", "korn5676"),
    )


def parse_player_id(href: Optional[str]) -> Optional[int]:
    if not href:
        return None
    match = re.search(r"/player/[^/]+-(\d+)", href)
    if match:
        return int(match.group(1))
    return None


def parse_pitcher_stats(text: str) -> Dict[str, Any]:
    summary = {"record": None, "era": None, "strikeouts": None}
    raw = " ".join(str(text or "").split())
    if not raw:
        return summary

    record_match = re.search(r"(\d+\s*-\s*\d+)", raw)
    era_match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*ERA", raw, flags=re.IGNORECASE)
    so_match = re.search(r"(\d+)\s*SO", raw, flags=re.IGNORECASE)
    if record_match:
        summary["record"] = record_match.group(1).replace(" ", "")
    if era_match:
        summary["era"] = float(era_match.group(1))
    if so_match:
        summary["strikeouts"] = int(so_match.group(1))
    return summary


def parse_lineup_player(li_tag, order: int) -> Optional[Dict[str, Any]]:
    link = li_tag.select_one("a.starting-lineups__player--link")
    if link is None:
        return None

    name = " ".join(link.get_text(" ", strip=True).split())
    if not name:
        return None

    detail_text = " ".join(li_tag.get_text(" ", strip=True).split())
    bats_match = re.search(r"\(([SLR])\)", detail_text)
    position_match = re.search(r"\([SLR]\)\s*([A-Z0-9]+)$", detail_text)

    return {
        "order": order,
        "name": name,
        "player_id": parse_player_id(link.get("href")),
        "bats": bats_match.group(1) if bats_match else None,
        "position": position_match.group(1) if position_match else None,
    }


def parse_pitcher_summary(summary_tag) -> Optional[Dict[str, Any]]:
    if summary_tag is None:
        return None
    link = summary_tag.select_one(".starting-lineups__pitcher-name a.starting-lineups__pitcher--link")
    if link is None:
        return None

    name = " ".join(link.get_text(" ", strip=True).split())
    if not name:
        return None

    hand = None
    hand_tag = summary_tag.select_one(".starting-lineups__pitcher-pitch-hand")
    if hand_tag:
        hand_text = hand_tag.get_text(" ", strip=True).upper()
        if "LHP" in hand_text:
            hand = "L"
        elif "RHP" in hand_text:
            hand = "R"

    stats_tag = summary_tag.select_one(".starting-lineups__pitcher-stats-summary")
    stats = parse_pitcher_stats(stats_tag.get_text(" ", strip=True) if stats_tag else "")

    return {
        "name": name,
        "player_id": parse_player_id(link.get("href")),
        "throws": hand,
        **stats,
    }


@dataclass
class MatchupRecord:
    game_date: date
    season: int
    game_id: Optional[str]
    home_team: str
    away_team: str
    home_pitcher: Optional[Dict[str, Any]]
    away_pitcher: Optional[Dict[str, Any]]
    home_lineup: Optional[Dict[str, Any]]
    away_lineup: Optional[Dict[str, Any]]
    game_time: Optional[str]
    venue: Optional[str]
    game_status: Optional[str]
    data_source: str = "mlb_html"


class DailyLineupScraper:
    def __init__(self) -> None:
        self.session = requests.Session()
        self.session.headers.update(HEADERS)

    def fetch_page(self, target_date: date) -> str:
        url = f"{BASE_URL}/{target_date.isoformat()}"
        response = self.session.get(url, timeout=30)
        response.raise_for_status()
        return response.text

    def parse_page(self, target_date: date, html: str) -> List[MatchupRecord]:
        soup = BeautifulSoup(html, "lxml")
        matchup_tags = soup.select("div.starting-lineups__matchup")
        records: List[MatchupRecord] = []

        for matchup in matchup_tags:
            team_links = matchup.select(".starting-lineups__team-name--link")
            if len(team_links) < 2:
                continue

            away_team = normalize_team_code(team_links[0].get("data-tri-code") or team_links[0].get_text(strip=True))
            home_team = normalize_team_code(team_links[1].get("data-tri-code") or team_links[1].get_text(strip=True))

            game_time = None
            time_tag = matchup.select_one(".starting-lineups__game-date-time time")
            if time_tag and time_tag.get("datetime"):
                try:
                    game_time = datetime.fromisoformat(
                        time_tag.get("datetime").replace("Z", "+00:00")
                    ).strftime("%H:%M")
                except ValueError:
                    game_time = time_tag.get("datetime")[:10]

            venue_tag = matchup.select_one(".starting-lineups__game-location")
            venue = " ".join(venue_tag.get_text(" ", strip=True).split()) if venue_tag else None

            status_tag = matchup.select_one(".starting-lineups__game-state")
            game_status = " ".join(status_tag.get_text(" ", strip=True).split()) if status_tag else None

            pitcher_summaries = matchup.select(".starting-lineups__pitcher-summary")
            away_pitcher = parse_pitcher_summary(pitcher_summaries[0]) if len(pitcher_summaries) >= 1 else None
            home_pitcher = parse_pitcher_summary(pitcher_summaries[-1]) if len(pitcher_summaries) >= 2 else None

            lineup_groups = matchup.select(".starting-lineups__teams--sm.starting-lineups__teams--xl ol.starting-lineups__team")
            if len(lineup_groups) < 2:
                lineup_groups = matchup.select(".starting-lineups__teams ol.starting-lineups__team")[:2]

            away_lineup = None
            home_lineup = None
            if len(lineup_groups) >= 2:
                away_players = [
                    player
                    for idx, li_tag in enumerate(lineup_groups[0].select("li.starting-lineups__player"), start=1)
                    if (player := parse_lineup_player(li_tag, idx))
                ]
                home_players = [
                    player
                    for idx, li_tag in enumerate(lineup_groups[1].select("li.starting-lineups__player"), start=1)
                    if (player := parse_lineup_player(li_tag, idx))
                ]
                away_lineup = {"confirmed": len(away_players) >= 9, "batting_order": away_players}
                home_lineup = {"confirmed": len(home_players) >= 9, "batting_order": home_players}

            records.append(
                MatchupRecord(
                    game_date=target_date,
                    season=target_date.year,
                    game_id=matchup.get("data-gamepk"),
                    home_team=home_team,
                    away_team=away_team,
                    home_pitcher=home_pitcher,
                    away_pitcher=away_pitcher,
                    home_lineup=home_lineup,
                    away_lineup=away_lineup,
                    game_time=game_time,
                    venue=venue,
                    game_status=game_status,
                )
            )

        return records

    def upsert_records(self, conn, records: List[MatchupRecord]) -> int:
        if not records:
            return 0

        updated = 0
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            for record in records:
                cursor.execute(
                    """
                    UPDATE daily_lineups
                    SET season = %s,
                        game_id = %s,
                        home_pitcher = %s,
                        away_pitcher = %s,
                        home_lineup = %s,
                        away_lineup = %s,
                        game_time = %s,
                        venue = %s,
                        game_status = %s,
                        data_source = %s,
                        last_updated = NOW()
                    WHERE game_date = %s
                      AND home_team = %s
                      AND away_team = %s
                    RETURNING id
                    """,
                    (
                        record.season,
                        record.game_id,
                        Json(record.home_pitcher) if record.home_pitcher is not None else None,
                        Json(record.away_pitcher) if record.away_pitcher is not None else None,
                        Json(record.home_lineup) if record.home_lineup is not None else None,
                        Json(record.away_lineup) if record.away_lineup is not None else None,
                        record.game_time,
                        record.venue,
                        record.game_status,
                        record.data_source,
                        record.game_date,
                        record.home_team,
                        record.away_team,
                    ),
                )
                existing = cursor.fetchone()
                if existing:
                    updated += 1
                    continue

                cursor.execute(
                    """
                    INSERT INTO daily_lineups (
                        game_date,
                        season,
                        game_id,
                        home_team,
                        away_team,
                        home_pitcher,
                        away_pitcher,
                        home_lineup,
                        away_lineup,
                        game_time,
                        venue,
                        game_status,
                        data_source,
                        last_updated
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
                    """,
                    (
                        record.game_date,
                        record.season,
                        record.game_id,
                        record.home_team,
                        record.away_team,
                        Json(record.home_pitcher) if record.home_pitcher is not None else None,
                        Json(record.away_pitcher) if record.away_pitcher is not None else None,
                        Json(record.home_lineup) if record.home_lineup is not None else None,
                        Json(record.away_lineup) if record.away_lineup is not None else None,
                        record.game_time,
                        record.venue,
                        record.game_status,
                        record.data_source,
                    ),
                )
                updated += 1
        conn.commit()
        return updated


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch MLB starting lineups into daily_lineups.")
    parser.add_argument("--date", help="Single date in YYYY-MM-DD format.")
    parser.add_argument("--start-date", help="Backfill start date in YYYY-MM-DD format.")
    parser.add_argument("--end-date", help="Backfill end date in YYYY-MM-DD format.")
    return parser.parse_args()


def iter_dates(args: argparse.Namespace) -> List[date]:
    if args.date:
        return [datetime.strptime(args.date, "%Y-%m-%d").date()]
    if args.start_date and args.end_date:
        start = datetime.strptime(args.start_date, "%Y-%m-%d").date()
        end = datetime.strptime(args.end_date, "%Y-%m-%d").date()
        days = (end - start).days
        return [start + timedelta(days=offset) for offset in range(days + 1)]
    return [date.today()]


def main() -> None:
    args = parse_args()
    scraper = DailyLineupScraper()
    conn = connect_db()
    try:
        total = 0
        for target_date in iter_dates(args):
            html = scraper.fetch_page(target_date)
            records = scraper.parse_page(target_date, html)
            count = scraper.upsert_records(conn, records)
            total += count
            print(f"{target_date}: parsed={len(records)} upserted={count}")
        print(f"total_upserted={total}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
