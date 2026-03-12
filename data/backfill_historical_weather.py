#!/usr/bin/env python3
"""
Backfill historical MLB game weather into a dedicated database table.

Source:
- Open-Meteo historical archive API

This script creates new table(s) only and does not modify existing game/stadium rows.
"""

from __future__ import annotations

import argparse
import logging
import os
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Tuple
from zoneinfo import ZoneInfo

import psycopg2
import requests
from psycopg2.extras import RealDictCursor

from feature_engineering import DOME_STADIUMS, STADIUM_LOCATIONS, TEAM_CODE_ALIASES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

PARK_ORIENTATION_DEGREES = {
    'ARI': 18.0,
    'ATL': 32.0,
    'BAL': 42.0,
    'BOS': 50.0,
    'CHC': 42.0,
    'CWS': 40.0,
    'CIN': 32.0,
    'CLE': 20.0,
    'COL': 15.0,
    'DET': 25.0,
    'HOU': 18.0,
    'KC': 30.0,
    'LAA': 45.0,
    'LAD': 42.0,
    'MIA': 35.0,
    'MIL': 40.0,
    'MIN': 28.0,
    'NYM': 50.0,
    'NYY': 50.0,
    'OAK': 50.0,
    'PHI': 20.0,
    'PIT': 25.0,
    'SD': 30.0,
    'SEA': 35.0,
    'SF': 55.0,
    'STL': 25.0,
    'TB': 35.0,
    'TEX': 32.0,
    'TOR': 32.0,
    'WSH': 35.0,
}


def _normalize_team_code(team_code: str) -> str:
    if not team_code:
        return team_code
    normalized = str(team_code).strip().upper()
    return TEAM_CODE_ALIASES.get(normalized, normalized)


class HistoricalWeatherBackfiller:
    def __init__(self):
        self.conn = psycopg2.connect(
            host=os.getenv('DB_HOST', '192.168.1.23'),
            port=int(os.getenv('DB_PORT', 5432)),
            database=os.getenv('DB_NAME', 'baseball_migration_test'),
            user=os.getenv('DB_USER', 'postgres'),
            password=os.getenv('DB_PASSWORD', 'korn5676')
        )
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'ProjectionAI/1.0 historical weather backfill'
        })
        self._api_cache: Dict[Tuple[str, str], Dict] = {}

    def close(self):
        self.conn.close()
        self.session.close()

    def create_tables(self):
        cursor = self.conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS historical_game_weather (
                id SERIAL PRIMARY KEY,
                game_id INTEGER NOT NULL UNIQUE,
                game_date DATE NOT NULL,
                home_team VARCHAR(5) NOT NULL,
                away_team VARCHAR(5) NOT NULL,
                venue TEXT,
                roof_type TEXT,
                weather_source VARCHAR(30) NOT NULL DEFAULT 'open_meteo_archive',
                source_quality VARCHAR(40) NOT NULL DEFAULT 'park_lookup',
                latitude DOUBLE PRECISION,
                longitude DOUBLE PRECISION,
                local_game_time TIMESTAMP,
                observation_time TIMESTAMP,
                temp_f DOUBLE PRECISION,
                relative_humidity DOUBLE PRECISION,
                wind_speed_mph DOUBLE PRECISION,
                wind_direction_deg DOUBLE PRECISION,
                precipitation_mm DOUBLE PRECISION,
                pressure_msl_hpa DOUBLE PRECISION,
                weather_code INTEGER,
                weather_available BOOLEAN NOT NULL DEFAULT FALSE,
                is_dome BOOLEAN NOT NULL DEFAULT FALSE,
                roof_status_estimated VARCHAR(20),
                roof_status_confidence DOUBLE PRECISION,
                dew_point_f DOUBLE PRECISION,
                air_carry_factor DOUBLE PRECISION,
                wind_out_to_center_mph DOUBLE PRECISION,
                wind_out_to_left_field_mph DOUBLE PRECISION,
                wind_out_to_right_field_mph DOUBLE PRECISION,
                wind_in_from_center_mph DOUBLE PRECISION,
                crosswind_mph DOUBLE PRECISION,
                wind_out_factor DOUBLE PRECISION,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            """
        )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_historical_game_weather_game_date
            ON historical_game_weather (game_date);
            """
        )
        cursor.execute(
            """
            ALTER TABLE historical_game_weather
            ALTER COLUMN source_quality TYPE VARCHAR(40);
            """
        )
        cursor.execute(
            """
            ALTER TABLE historical_game_weather
            ALTER COLUMN source_quality SET DEFAULT 'park_lookup';
            """
        )
        for statement in [
            "ALTER TABLE historical_game_weather ADD COLUMN IF NOT EXISTS roof_status_estimated VARCHAR(20);",
            "ALTER TABLE historical_game_weather ADD COLUMN IF NOT EXISTS roof_status_confidence DOUBLE PRECISION;",
            "ALTER TABLE historical_game_weather ADD COLUMN IF NOT EXISTS dew_point_f DOUBLE PRECISION;",
            "ALTER TABLE historical_game_weather ADD COLUMN IF NOT EXISTS air_carry_factor DOUBLE PRECISION;",
            "ALTER TABLE historical_game_weather ADD COLUMN IF NOT EXISTS wind_out_to_center_mph DOUBLE PRECISION;",
            "ALTER TABLE historical_game_weather ADD COLUMN IF NOT EXISTS wind_out_to_left_field_mph DOUBLE PRECISION;",
            "ALTER TABLE historical_game_weather ADD COLUMN IF NOT EXISTS wind_out_to_right_field_mph DOUBLE PRECISION;",
            "ALTER TABLE historical_game_weather ADD COLUMN IF NOT EXISTS wind_in_from_center_mph DOUBLE PRECISION;",
            "ALTER TABLE historical_game_weather ADD COLUMN IF NOT EXISTS crosswind_mph DOUBLE PRECISION;",
        ]:
            cursor.execute(statement)
        self.conn.commit()
        logger.info("historical_game_weather table ready")

    def _load_games(self, start_date: str | None = None, end_date: str | None = None) -> List[Dict]:
        clauses = []
        params: List = []
        if start_date:
            clauses.append("g.game_date >= %s")
            params.append(start_date)
        if end_date:
            clauses.append("g.game_date <= %s")
            params.append(end_date)
        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""

        cursor = self.conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute(
            f"""
            SELECT
                g.game_id,
                g.game_date,
                g.home_team,
                g.away_team,
                g.venue,
                g.date_time,
                s.roof_type,
                s.stadium_name
            FROM games g
            LEFT JOIN stadiums s
              ON s.home_team = g.home_team
            {where_sql}
            ORDER BY g.game_date, g.game_id
            """,
            params
        )
        return cursor.fetchall()

    def _game_location(self, game_row: Dict) -> Dict:
        home_team = _normalize_team_code(game_row['home_team'])
        location = STADIUM_LOCATIONS.get(home_team)
        if not location:
            return {}
        return {
            'team_code': home_team,
            'lat': location['lat'],
            'lon': location['lon'],
            'tz': location['tz'],
        }

    def _is_dome_game(self, game_row: Dict) -> bool:
        roof_type = str(game_row.get('roof_type') or '').lower()
        venue = str(game_row.get('venue') or game_row.get('stadium_name') or '')
        if roof_type in {'fixed', 'retractable'} and 'open' not in roof_type:
            return True
        return venue in DOME_STADIUMS

    def _local_game_time(self, game_row: Dict, tz_name: str) -> datetime:
        game_dt = game_row.get('date_time')
        if game_dt is None:
            local_zone = ZoneInfo(tz_name)
            return datetime.combine(game_row['game_date'], datetime.min.time().replace(hour=19)).replace(tzinfo=local_zone)
        if game_dt.tzinfo is None:
            game_dt = game_dt.replace(tzinfo=timezone.utc)
        return game_dt.astimezone(ZoneInfo(tz_name))

    def _fetch_day_weather(self, lat: float, lon: float, game_date: str, tz_name: str) -> Dict:
        cache_key = (f"{lat:.3f},{lon:.3f}", game_date)
        if cache_key in self._api_cache:
            return self._api_cache[cache_key]

        params = {
            'latitude': lat,
            'longitude': lon,
            'start_date': game_date,
            'end_date': game_date,
            'timezone': tz_name,
            'temperature_unit': 'fahrenheit',
            'wind_speed_unit': 'mph',
            'hourly': ','.join([
                'temperature_2m',
                'relative_humidity_2m',
                'precipitation',
                'wind_speed_10m',
                'wind_direction_10m',
                'pressure_msl',
                'weather_code',
            ]),
        }
        response = self.session.get(OPEN_METEO_ARCHIVE_URL, params=params, timeout=30)
        response.raise_for_status()
        payload = response.json()
        self._api_cache[cache_key] = payload
        return payload

    def _pick_nearest_observation(self, payload: Dict, local_game_time: datetime) -> Dict:
        hourly = payload.get('hourly', {})
        times = hourly.get('time', [])
        if not times:
            return {}

        target = local_game_time.replace(minute=0, second=0, microsecond=0)
        nearest_index = None
        nearest_diff = None
        for idx, value in enumerate(times):
            observed_time = datetime.fromisoformat(value).replace(tzinfo=local_game_time.tzinfo)
            diff = abs((observed_time - target).total_seconds())
            if nearest_diff is None or diff < nearest_diff:
                nearest_index = idx
                nearest_diff = diff

        if nearest_index is None:
            return {}

        def _hourly_value(key, default=None):
            values = hourly.get(key, [])
            if nearest_index >= len(values):
                return default
            return values[nearest_index]

        observed_time = datetime.fromisoformat(times[nearest_index]).replace(tzinfo=local_game_time.tzinfo)
        wind_speed = _hourly_value('wind_speed_10m')
        wind_dir = _hourly_value('wind_direction_10m')
        return {
            'observation_time': observed_time,
            'temp_f': _hourly_value('temperature_2m'),
            'relative_humidity': _hourly_value('relative_humidity_2m'),
            'wind_speed_mph': wind_speed,
            'wind_direction_deg': wind_dir,
            'precipitation_mm': _hourly_value('precipitation'),
            'pressure_msl_hpa': _hourly_value('pressure_msl'),
            'weather_code': _hourly_value('weather_code'),
            'wind_out_factor': self._wind_out_factor(wind_speed, wind_dir),
            'weather_available': True,
        }

    def _dew_point_f(self, temp_f, relative_humidity):
        try:
            temp_c = (float(temp_f) - 32.0) * 5.0 / 9.0
            rh = max(1e-6, min(float(relative_humidity), 100.0))
            a = 17.27
            b = 237.7
            alpha = ((a * temp_c) / (b + temp_c)) + __import__('math').log(rh / 100.0)
            dew_c = (b * alpha) / (a - alpha)
            return dew_c * 9.0 / 5.0 + 32.0
        except Exception:
            return None

    def _estimate_roof_status(self, roof_type: str, precip_mm, temp_f, wind_speed_mph) -> Tuple[str, float]:
        roof_text = str(roof_type or '').lower()
        precip_mm = float(precip_mm or 0.0)
        temp_f = float(temp_f or 72.0)
        wind_speed_mph = float(wind_speed_mph or 0.0)

        if 'fixed' in roof_text or 'dome' in roof_text:
            return 'closed', 1.0
        if 'retractable' in roof_text:
            if precip_mm > 0.2 or temp_f < 58.0 or temp_f > 94.0 or wind_speed_mph > 18.0:
                return 'closed', 0.75
            return 'open', 0.65
        return 'open_air', 0.95

    def _wind_components(self, home_team: str, wind_speed_mph, wind_direction_deg) -> Tuple[float, float, float, float, float]:
        try:
            wind_speed = float(wind_speed_mph or 0.0)
            wind_direction = float(wind_direction_deg or 0.0)
        except Exception:
            return 0.0, 0.0, 0.0, 0.0, 0.0

        center_field_bearing = PARK_ORIENTATION_DEGREES.get(_normalize_team_code(home_team))
        if center_field_bearing is None:
            return 0.0, 0.0, 0.0, 0.0, 0.0

        wind_to_direction = (wind_direction + 180.0) % 360.0
        math_mod = __import__('math')

        def _component(field_bearing: float) -> float:
            delta = ((wind_to_direction - field_bearing + 180.0) % 360.0) - 180.0
            return wind_speed * math_mod.cos(math_mod.radians(delta))

        center_component = _component(center_field_bearing)
        left_field_component = _component((center_field_bearing - 45.0) % 360.0)
        right_field_component = _component((center_field_bearing + 45.0) % 360.0)
        delta = ((wind_to_direction - center_field_bearing + 180.0) % 360.0) - 180.0
        cross_component = abs(wind_speed * math_mod.sin(math_mod.radians(delta)))
        return (
            round(max(0.0, center_component), 3),
            round(max(0.0, left_field_component), 3),
            round(max(0.0, right_field_component), 3),
            round(max(0.0, -center_component), 3),
            round(cross_component, 3),
        )

    def _air_carry_factor(self, temp_f, dew_point_f, roof_status_estimated: str) -> float:
        temp_f = float(temp_f or 72.0)
        dew_point_f = float(dew_point_f or 55.0)
        boost = 1.0 + ((temp_f - 70.0) * 0.0025) + ((dew_point_f - 55.0) * 0.0015)
        if roof_status_estimated == 'closed':
            boost = (boost + 1.0) / 2.0
        return round(min(max(boost, 0.92), 1.10), 3)

    def _wind_out_factor(self, wind_speed, wind_direction) -> float:
        try:
            wind_speed = float(wind_speed or 0.0)
            wind_direction = float(wind_direction or 0.0)
        except Exception:
            return 1.0

        # Very coarse approximation: southerly / south-westerly winds are treated
        # as more favorable for carry for this first pass.
        if 135 <= wind_direction <= 225:
            return round(1.0 + min(wind_speed, 20.0) * 0.01, 3)
        if 315 <= wind_direction or wind_direction <= 45:
            return round(max(0.85, 1.0 - min(wind_speed, 20.0) * 0.01), 3)
        return 1.0

    def _upsert_weather_row(self, game_row: Dict, location: Dict, local_game_time: datetime, weather: Dict, is_dome: bool):
        cursor = self.conn.cursor()
        roof_status_estimated, roof_status_confidence = self._estimate_roof_status(
            game_row.get('roof_type'),
            weather.get('precipitation_mm'),
            weather.get('temp_f'),
            weather.get('wind_speed_mph'),
        )
        dew_point_f = self._dew_point_f(weather.get('temp_f'), weather.get('relative_humidity'))
        (
            wind_out_to_center_mph,
            wind_out_to_left_field_mph,
            wind_out_to_right_field_mph,
            wind_in_from_center_mph,
            crosswind_mph,
        ) = self._wind_components(
            game_row.get('home_team'),
            weather.get('wind_speed_mph'),
            weather.get('wind_direction_deg'),
        )
        air_carry_factor = self._air_carry_factor(weather.get('temp_f'), dew_point_f, roof_status_estimated)
        cursor.execute(
            """
            INSERT INTO historical_game_weather (
                game_id, game_date, home_team, away_team, venue, roof_type,
                latitude, longitude, local_game_time, observation_time,
                temp_f, relative_humidity, wind_speed_mph, wind_direction_deg,
                precipitation_mm, pressure_msl_hpa, weather_code, weather_available,
                is_dome, roof_status_estimated, roof_status_confidence, dew_point_f,
                air_carry_factor, wind_out_to_center_mph, wind_out_to_left_field_mph,
                wind_out_to_right_field_mph, wind_in_from_center_mph,
                crosswind_mph, wind_out_factor
            ) VALUES (
                %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s,
                %s, %s, %s, %s,
                %s, %s, %s, %s,
                %s, %s, %s, %s,
                %s, %s, %s, %s, %s,
                %s, %s
            )
            ON CONFLICT (game_id) DO UPDATE SET
                game_date = EXCLUDED.game_date,
                home_team = EXCLUDED.home_team,
                away_team = EXCLUDED.away_team,
                venue = EXCLUDED.venue,
                roof_type = EXCLUDED.roof_type,
                latitude = EXCLUDED.latitude,
                longitude = EXCLUDED.longitude,
                local_game_time = EXCLUDED.local_game_time,
                observation_time = EXCLUDED.observation_time,
                temp_f = EXCLUDED.temp_f,
                relative_humidity = EXCLUDED.relative_humidity,
                wind_speed_mph = EXCLUDED.wind_speed_mph,
                wind_direction_deg = EXCLUDED.wind_direction_deg,
                precipitation_mm = EXCLUDED.precipitation_mm,
                pressure_msl_hpa = EXCLUDED.pressure_msl_hpa,
                weather_code = EXCLUDED.weather_code,
                weather_available = EXCLUDED.weather_available,
                is_dome = EXCLUDED.is_dome,
                roof_status_estimated = EXCLUDED.roof_status_estimated,
                roof_status_confidence = EXCLUDED.roof_status_confidence,
                dew_point_f = EXCLUDED.dew_point_f,
                air_carry_factor = EXCLUDED.air_carry_factor,
                wind_out_to_center_mph = EXCLUDED.wind_out_to_center_mph,
                wind_out_to_left_field_mph = EXCLUDED.wind_out_to_left_field_mph,
                wind_out_to_right_field_mph = EXCLUDED.wind_out_to_right_field_mph,
                wind_in_from_center_mph = EXCLUDED.wind_in_from_center_mph,
                crosswind_mph = EXCLUDED.crosswind_mph,
                wind_out_factor = EXCLUDED.wind_out_factor,
                updated_at = CURRENT_TIMESTAMP
            """,
            (
                game_row['game_id'],
                game_row['game_date'],
                _normalize_team_code(game_row['home_team']),
                _normalize_team_code(game_row['away_team']),
                game_row.get('venue'),
                game_row.get('roof_type'),
                location.get('lat'),
                location.get('lon'),
                local_game_time,
                weather.get('observation_time'),
                weather.get('temp_f'),
                weather.get('relative_humidity'),
                weather.get('wind_speed_mph'),
                weather.get('wind_direction_deg'),
                weather.get('precipitation_mm'),
                weather.get('pressure_msl_hpa'),
                weather.get('weather_code'),
                bool(weather.get('weather_available', False)),
                is_dome,
                roof_status_estimated,
                roof_status_confidence,
                dew_point_f,
                air_carry_factor,
                wind_out_to_center_mph,
                wind_out_to_left_field_mph,
                wind_out_to_right_field_mph,
                wind_in_from_center_mph,
                crosswind_mph,
                weather.get('wind_out_factor'),
            )
        )

    def backfill(self, start_date: str | None = None, end_date: str | None = None):
        games = self._load_games(start_date=start_date, end_date=end_date)
        logger.info("Loaded %s games for weather backfill", len(games))
        inserted = 0
        for idx, game_row in enumerate(games, start=1):
            location = self._game_location(game_row)
            if not location:
                continue

            is_dome = self._is_dome_game(game_row)
            local_game_time = self._local_game_time(game_row, location['tz'])
            weather_payload = {}
            if is_dome:
                weather_payload = {
                    'observation_time': local_game_time,
                    'temp_f': 72.0,
                    'relative_humidity': None,
                    'wind_speed_mph': 0.0,
                    'wind_direction_deg': None,
                    'precipitation_mm': 0.0,
                    'pressure_msl_hpa': None,
                    'weather_code': None,
                    'weather_available': True,
                    'wind_out_factor': 1.0,
                }
            else:
                try:
                    payload = self._fetch_day_weather(location['lat'], location['lon'], game_row['game_date'].isoformat(), location['tz'])
                    weather_payload = self._pick_nearest_observation(payload, local_game_time)
                except Exception as exc:
                    logger.warning("Weather fetch failed for game %s: %s", game_row['game_id'], exc)
                    weather_payload = {'weather_available': False, 'wind_out_factor': 1.0}

            self._upsert_weather_row(game_row, location, local_game_time, weather_payload, is_dome)
            inserted += 1
            if idx % 100 == 0:
                self.conn.commit()
                logger.info("Processed %s / %s games", idx, len(games))

        self.conn.commit()
        logger.info("Backfilled weather for %s games", inserted)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--create-table', action='store_true')
    parser.add_argument('--start-date')
    parser.add_argument('--end-date')
    parser.add_argument('--backfill-all', action='store_true')
    args = parser.parse_args()

    backfiller = HistoricalWeatherBackfiller()
    try:
        if args.create_table:
            backfiller.create_tables()

        if args.backfill_all or args.start_date or args.end_date:
            backfiller.backfill(start_date=args.start_date, end_date=args.end_date)
    finally:
        backfiller.close()


if __name__ == '__main__':
    main()
