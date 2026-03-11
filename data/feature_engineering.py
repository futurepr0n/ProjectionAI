import pandas as pd
import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor
import logging
import os
import json
from datetime import datetime, timedelta
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TEAM_CODE_ALIASES = {
    'ATH': 'OAK',
    'AZ': 'ARI',
    'CHW': 'CWS',
    'CWS': 'CWS',
    'KCR': 'KC',
    'SD': 'SDP',
    'SDP': 'SDP',
    'SF': 'SFG',
    'SFG': 'SFG',
    'TB': 'TB',
    'TBR': 'TB',
    'WSH': 'WSN',
    'WSN': 'WSN',
}

PARK_FACTORS = {
    'COL': 1.35, 'NYY': 1.20, 'BOS': 1.15, 'CIN': 1.15, 'PHI': 1.10,
    'HOU': 1.10, 'BAL': 1.05, 'TEX': 1.05, 'ARI': 1.05, 'ATL': 1.00,
    'CHC': 1.00, 'LAD': 1.00, 'MIL': 1.00, 'SDP': 1.00, 'WSN': 1.00,
    'CLE': 0.95, 'DET': 0.95, 'KCR': 0.95, 'MIN': 0.95, 'OAK': 0.90,
    'MIA': 0.90, 'PIT': 0.90, 'SEA': 0.90, 'SFG': 0.90, 'TBR': 0.90,
}

# Stadium coordinates
STADIUM_LOCATIONS = {
    'ARI': {'lat': 33.445, 'lon': -112.066, 'city': 'Phoenix', 'tz': 'America/Phoenix'},
    'ATL': {'lat': 33.890, 'lon': -84.467, 'city': 'Atlanta', 'tz': 'America/New_York'},
    'BAL': {'lat': 39.284, 'lon': -76.621, 'city': 'Baltimore', 'tz': 'America/New_York'},
    'BOS': {'lat': 42.346, 'lon': -71.097, 'city': 'Boston', 'tz': 'America/New_York'},
    'CHC': {'lat': 41.948, 'lon': -87.655, 'city': 'Chicago', 'tz': 'America/Chicago'},
    'CHW': {'lat': 41.830, 'lon': -87.633, 'city': 'Chicago', 'tz': 'America/Chicago'},
    'CIN': {'lat': 39.097, 'lon': -84.506, 'city': 'Cincinnati', 'tz': 'America/New_York'},
    'CLE': {'lat': 41.496, 'lon': -81.685, 'city': 'Cleveland', 'tz': 'America/New_York'},
    'COL': {'lat': 39.756, 'lon': -104.994, 'city': 'Denver', 'tz': 'America/Denver'},
    'DET': {'lat': 42.339, 'lon': -83.048, 'city': 'Detroit', 'tz': 'America/New_York'},
    'HOU': {'lat': 29.757, 'lon': -95.355, 'city': 'Houston', 'tz': 'America/Chicago'},
    'KC': {'lat': 39.051, 'lon': -94.480, 'city': 'Kansas City', 'tz': 'America/Chicago'},
    'LAA': {'lat': 33.800, 'lon': -117.882, 'city': 'Anaheim', 'tz': 'America/Los_Angeles'},
    'LAD': {'lat': 34.073, 'lon': -118.240, 'city': 'Los Angeles', 'tz': 'America/Los_Angeles'},
    'MIA': {'lat': 25.778, 'lon': -80.219, 'city': 'Miami', 'tz': 'America/New_York'},
    'MIL': {'lat': 43.028, 'lon': -87.971, 'city': 'Milwaukee', 'tz': 'America/Chicago'},
    'MIN': {'lat': 44.981, 'lon': -93.277, 'city': 'Minneapolis', 'tz': 'America/Chicago'},
    'NYM': {'lat': 40.757, 'lon': -73.845, 'city': 'New York', 'tz': 'America/New_York'},
    'NYY': {'lat': 40.829, 'lon': -73.926, 'city': 'New York', 'tz': 'America/New_York'},
    'OAK': {'lat': 37.751, 'lon': -122.200, 'city': 'Oakland', 'tz': 'America/Los_Angeles'},
    'PHI': {'lat': 39.906, 'lon': -75.166, 'city': 'Philadelphia', 'tz': 'America/New_York'},
    'PIT': {'lat': 40.446, 'lon': -80.005, 'city': 'Pittsburgh', 'tz': 'America/New_York'},
    'SDP': {'lat': 32.707, 'lon': -117.156, 'city': 'San Diego', 'tz': 'America/Los_Angeles'},
    'SEA': {'lat': 47.591, 'lon': -122.332, 'city': 'Seattle', 'tz': 'America/Los_Angeles'},
    'SFG': {'lat': 37.778, 'lon': -122.389, 'city': 'San Francisco', 'tz': 'America/Los_Angeles'},
    'STL': {'lat': 38.622, 'lon': -90.192, 'city': 'St. Louis', 'tz': 'America/Chicago'},
    'TB': {'lat': 27.768, 'lon': -82.653, 'city': 'St. Petersburg', 'tz': 'America/New_York'},
    'TEX': {'lat': 32.747, 'lon': -97.082, 'city': 'Arlington', 'tz': 'America/Chicago'},
    'TOR': {'lat': 43.641, 'lon': -79.389, 'city': 'Toronto', 'tz': 'America/Toronto'},
    'WSN': {'lat': 38.873, 'lon': -77.007, 'city': 'Washington', 'tz': 'America/New_York'},
}

# Dome stadiums (no weather impact)
DOME_STADIUMS = [
    'Tropicana Field',
    'Rogers Centre',
    'Chase Field',
    'Minute Maid Park',
    'American Family Field',
    'Globe Life Field',
    'T-Mobile Park',
    'loanDepot Park'
]


def _connect_db():
    try:
        conn = psycopg2.connect(
            host=os.getenv('DB_HOST', '192.168.1.23'),
            port=int(os.getenv('DB_PORT', 5432)),
            database=os.getenv('DB_NAME', 'baseball_migration_test'),
            user=os.getenv('DB_USER', 'postgres'),
            password=os.getenv('DB_PASSWORD', 'korn5676')
        )
        return conn
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        return None


def get_park_factors_from_db(engine=None) -> dict:
    """Query park factors from database, fallback to hardcoded"""
    if engine is None:
        return PARK_FACTORS

    try:
        conn = engine
        conn.rollback()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute("""
            SELECT home_team as team_code, park_factor_hr / 100.0 as park_hr_factor
            FROM stadiums
            WHERE home_team IS NOT NULL AND park_factor_hr IS NOT NULL
        """)
        data = cursor.fetchall()

        if len(data) >= 20:
            factors = {row['team_code']: row['park_hr_factor'] for row in data}
            logger.info(f"Loaded {len(factors)} park factors from database")
            return factors
        else:
            logger.warning(f"Only {len(data)} park factors in DB, using hardcoded")
            return PARK_FACTORS
    except Exception as e:
        logger.warning(f"Failed to load park factors from DB: {e}, using hardcoded")
        return PARK_FACTORS


def add_park_factors(df: pd.DataFrame, engine=None, team_col: str = 'away_team') -> pd.DataFrame:
    factors = get_park_factors_from_db(engine)
    df['park_factor'] = df[team_col].map(factors).fillna(1.0)
    return df


def _normalize_team_code(team_code: str) -> str:
    if not team_code:
        return team_code
    normalized = str(team_code).strip().upper()
    return TEAM_CODE_ALIASES.get(normalized, normalized)


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance in miles between two coordinates"""
    R = 3959  # Earth radius in miles
    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    delta_lat = np.radians(lat2 - lat1)
    delta_lon = np.radians(lon2 - lon1)

    a = np.sin(delta_lat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(delta_lon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c


def get_timezone_offset(tz_name: str) -> int:
    """Get UTC offset in hours for timezone"""
    tz_map = {
        'America/Los_Angeles': -8,
        'America/Denver': -7,
        'America/Chicago': -6,
        'America/New_York': -5,
        'America/Phoenix': -7,
        'America/Toronto': -5,
    }
    return tz_map.get(tz_name, -5)


def add_travel_fatigue(df: pd.DataFrame, conn=None) -> pd.DataFrame:
    """Add travel distance, timezone changes, and fatigue score"""
    if df.empty or 'team' not in df.columns or 'game_date' not in df.columns:
        logger.warning("Cannot add travel fatigue: missing team or game_date column")
        return df

    if conn is None:
        conn = _connect_db()

    if conn is None:
        logger.warning("Cannot add travel fatigue: no database connection")
        df['travel_distance_miles'] = 0.0
        df['timezone_changes'] = 0
        df['travel_fatigue_score'] = 0.0
        return df

    try:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        working = df.copy()
        working['game_date'] = pd.to_datetime(working['game_date'], errors='coerce').dt.date
        working['team_norm'] = working['team'].apply(_normalize_team_code)

        unique_team_dates = (
            working[['team_norm', 'game_date']]
            .dropna()
            .drop_duplicates()
        )

        if unique_team_dates.empty:
            working['travel_distance_miles'] = 0.0
            working['timezone_changes'] = 0
            working['travel_fatigue_score'] = 0.0
            return working.drop(columns=['team_norm'])

        teams = unique_team_dates['team_norm'].dropna().unique().tolist()
        max_game_date = unique_team_dates['game_date'].max()

        cursor.execute("""
            SELECT game_date, home_team, away_team
            FROM games
            WHERE game_date <= %s
              AND (home_team = ANY(%s) OR away_team = ANY(%s))
            ORDER BY game_date, game_id
        """, (max_game_date, teams, teams))
        games = pd.DataFrame(cursor.fetchall())

        if games.empty:
            working['travel_distance_miles'] = 0.0
            working['timezone_changes'] = 0
            working['travel_fatigue_score'] = 0.0
            return working.drop(columns=['team_norm'])

        games['game_date'] = pd.to_datetime(games['game_date'], errors='coerce').dt.date
        home_games = games[['game_date', 'home_team']].rename(columns={'home_team': 'team_norm'})
        home_games['location_team'] = home_games['team_norm']
        away_games = games[['game_date', 'away_team', 'home_team']].rename(columns={'away_team': 'team_norm', 'home_team': 'location_team'})
        appearances = pd.concat([
            home_games[['team_norm', 'game_date', 'location_team']],
            away_games[['team_norm', 'game_date', 'location_team']],
        ], ignore_index=True)

        appearances['team_norm'] = appearances['team_norm'].apply(_normalize_team_code)
        appearances['location_team'] = appearances['location_team'].apply(_normalize_team_code)
        appearances = appearances.sort_values(['team_norm', 'game_date', 'location_team'])
        appearances['prev_location_team'] = appearances.groupby('team_norm')['location_team'].shift(1)
        appearances['prev_location_team'] = appearances['prev_location_team'].fillna(appearances['team_norm'])

        travel_lookup = (
            appearances[['team_norm', 'game_date', 'location_team', 'prev_location_team']]
            .drop_duplicates(subset=['team_norm', 'game_date'], keep='last')
        )

        def compute_travel(row):
            curr_loc = STADIUM_LOCATIONS.get(row['location_team'], {})
            prev_loc = STADIUM_LOCATIONS.get(row['prev_location_team'], {})
            if not curr_loc or not prev_loc:
                return pd.Series({'travel_distance_miles': 0.0, 'timezone_changes': 0})
            distance = haversine_distance(
                prev_loc['lat'], prev_loc['lon'],
                curr_loc['lat'], curr_loc['lon']
            )
            tz_change = abs(get_timezone_offset(curr_loc['tz']) - get_timezone_offset(prev_loc['tz']))
            return pd.Series({'travel_distance_miles': distance, 'timezone_changes': tz_change})

        travel_metrics = travel_lookup.apply(compute_travel, axis=1)
        travel_lookup = pd.concat([travel_lookup, travel_metrics], axis=1)
        working = working.merge(
            travel_lookup[['team_norm', 'game_date', 'travel_distance_miles', 'timezone_changes']],
            on=['team_norm', 'game_date'],
            how='left'
        )
        working['travel_distance_miles'] = working['travel_distance_miles'].fillna(0.0)
        working['timezone_changes'] = working['timezone_changes'].fillna(0).astype(int)

    except Exception as e:
        logger.warning(f"Error calculating travel fatigue: {e}")
        working = df.copy()
        working['travel_distance_miles'] = 0.0
        working['timezone_changes'] = 0

    working['travel_fatigue_score'] = (
        (working['travel_distance_miles'] / 3000 * 25) +
        (working['timezone_changes'] * 10)
    ).clip(0, 100)

    logger.info("Added travel fatigue features")
    return working.drop(columns=['team_norm'], errors='ignore')


def add_pitcher_rolling_stats(df: pd.DataFrame, conn=None) -> pd.DataFrame:
    if conn is None:
        conn = _connect_db()
    if conn is None:
        logger.warning("Cannot add pitcher rolling stats: no database connection")
        return df

    cursor = conn.cursor(cursor_factory=RealDictCursor)
    pitcher_names = df['pitcher_name'].unique().tolist()

    if not pitcher_names:
        return df

    query = """
    SELECT
        ps.player_name as pitcher_name,
        AVG(ps.home_runs::numeric / NULLIF(ps.innings_pitched::numeric, 0) * 9) as pitcher_hr_per_9_30d,
        AVG(ps.earned_runs::numeric / NULLIF(ps.innings_pitched::numeric, 0) * 9) as pitcher_era_30d,
        AVG(ps.strikeouts::numeric / NULLIF(ps.innings_pitched::numeric, 0) * 9) as pitcher_k_per_9_30d,
        AVG((ps.hits::numeric + ps.walks::numeric) / NULLIF(ps.innings_pitched::numeric, 0)) as pitcher_whip_30d
    FROM pitching_stats ps
    WHERE ps.player_name = ANY(%s)
      AND ps.innings_pitched > 0
    GROUP BY ps.player_name
    """

    cursor.execute(query, (pitcher_names,))
    data = cursor.fetchall()
    pitcher_df = pd.DataFrame(data) if data else pd.DataFrame()

    df = df.merge(pitcher_df, on='pitcher_name', how='left')
    logger.info(f"Added pitcher rolling stats to {pitcher_df['pitcher_name'].nunique()} pitchers")
    return df


def add_hitter_ev_stats(df: pd.DataFrame, conn=None) -> pd.DataFrame:
    if conn is None:
        conn = _connect_db()
    if conn is None:
        logger.warning("Cannot add hitter EV stats: no database connection")
        return df

    cursor = conn.cursor(cursor_factory=RealDictCursor)

    query = """
    SELECT
        last_name_first_name as player_name,
        avg_hit_speed_numeric as avg_ev,
        brl_percent_numeric as barrel_rate_ev,
        anglesweetspotpercent_numeric as sweet_spot_rate
    FROM hitter_exit_velocity
    WHERE last_name_first_name IS NOT NULL
    """

    cursor.execute(query)
    data = cursor.fetchall()
    ev_df = pd.DataFrame(data) if data else pd.DataFrame()

    if not ev_df.empty:
        df = df.merge(ev_df, on='player_name', how='left')
        logger.info(f"Added EV stats for {len(ev_df)} hitters")

    return df


def load_weather_cache() -> dict:
    """Load cached weather data"""
    cache_path = os.path.join(os.path.dirname(__file__), 'weather_cache.json')
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load weather cache: {e}")
    return {}


def save_weather_cache(cache: dict):
    """Save weather cache to disk"""
    cache_path = os.path.join(os.path.dirname(__file__), 'weather_cache.json')
    try:
        with open(cache_path, 'w') as f:
            json.dump(cache, f, indent=2)
    except Exception as e:
        logger.warning(f"Failed to save weather cache: {e}")


def fetch_historical_weather(lat: float, lon: float, date: str) -> dict:
    """Fetch historical weather from Open-Meteo API"""
    cache = load_weather_cache()
    cache_key = f"{lat:.2f}_{lon:.2f}_{date}"

    if cache_key in cache:
        return cache[cache_key]

    try:
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            'latitude': lat,
            'longitude': lon,
            'start_date': date,
            'end_date': date,
            'hourly': 'windspeed_10m,winddirection_10m,temperature_2m,precipitation_probability',
            'windspeed_unit': 'mph',
            'temperature_unit': 'fahrenheit',
            'timezone': 'auto'
        }

        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            hourly = data.get('hourly', {})

            if hourly:
                wind_speeds = hourly.get('windspeed_10m', [])
                temps = hourly.get('temperature_2m', [])
                precip_probs = hourly.get('precipitation_probability', [])

                try:
                    ws = [x for x in wind_speeds if x is not None] if wind_speeds else []
                    ts = [x for x in temps if x is not None] if temps else []
                    ps = [x for x in precip_probs if x is not None] if precip_probs else []

                    result = {
                        'wind_speed_mph': np.mean(ws) if ws else 0,
                        'temp_f': np.mean(ts) if ts else 70,
                        'precip_prob': np.mean(ps) if ps else 0,
                        'wind_out_factor': 1.0 if (np.mean(ws) if ws else 0) < 10 else 0.95
                    }
                except Exception as e:
                    logger.warning(f"Error processing weather data for {lat},{lon}: {e}")
                    return {'wind_speed_mph': 0, 'temp_f': 70, 'precip_prob': 0, 'wind_out_factor': 1.0}

                cache[cache_key] = result
                save_weather_cache(cache)
                return result
    except Exception as e:
        logger.warning(f"Failed to fetch weather for {lat},{lon} on {date}: {e}")

    return {'wind_speed_mph': 0, 'temp_f': 70, 'precip_prob': 0, 'wind_out_factor': 1.0}


def add_weather_context(df: pd.DataFrame) -> pd.DataFrame:
    """Add weather features from Open-Meteo historical API"""
    if df.empty or 'team' not in df.columns or 'game_date' not in df.columns:
        logger.warning("Cannot add weather context: missing team or game_date column")
        return df

    wind_speeds = []
    temps = []
    precip_probs = []
    wind_out_factors = []

    for idx, row in df.iterrows():
        team = row.get('team')
        game_date = row.get('game_date')

        if not team or not game_date:
            wind_speeds.append(0)
            temps.append(70)
            precip_probs.append(0)
            wind_out_factors.append(1.0)
            continue

        # Get location
        loc = STADIUM_LOCATIONS.get(team, {})
        if not loc or 'lat' not in loc or 'lon' not in loc:
            wind_speeds.append(0)
            temps.append(70)
            precip_probs.append(0)
            wind_out_factors.append(1.0)
            continue

        # Skip dome stadiums (check team code against DOME_STADIUMS list by team)
        if team in DOME_STADIUMS:
            wind_speeds.append(0)
            temps.append(70)
            precip_probs.append(0)
            wind_out_factors.append(1.0)
            continue

        # Fetch weather
        date_str = str(game_date).split()[0] if isinstance(game_date, str) else game_date.isoformat()
        weather = fetch_historical_weather(loc['lat'], loc['lon'], date_str)

        wind_speeds.append(weather['wind_speed_mph'])
        temps.append(weather['temp_f'])
        precip_probs.append(weather['precip_prob'])
        wind_out_factors.append(weather['wind_out_factor'])

    df['wind_speed_mph'] = wind_speeds
    df['temp_f'] = temps
    df['precip_prob'] = precip_probs
    df['wind_out_factor'] = wind_out_factors

    logger.info("Added weather context features")
    return df


def add_xstats(df: pd.DataFrame, conn=None) -> pd.DataFrame:
    if conn is None:
        conn = _connect_db()
    if conn is None:
        logger.warning("Cannot add xstats: no database connection")
        return df

    cursor = conn.cursor(cursor_factory=RealDictCursor)

    query = """
    SELECT
        last_name_first_name as player_name,
        xwoba, xba, xslg
    FROM custom_batter_2025
    WHERE last_name_first_name IS NOT NULL
    """

    cursor.execute(query)
    data = cursor.fetchall()
    xstats_df = pd.DataFrame(data) if data else pd.DataFrame()

    if not xstats_df.empty:
        df = df.merge(xstats_df, on='player_name', how='left')
        logger.info(f"Added xstats for {len(xstats_df)} batters")

    return df


def add_recent_hr_rate(df: pd.DataFrame, conn=None) -> pd.DataFrame:
    if conn is None:
        conn = _connect_db()
    if conn is None:
        logger.warning("Cannot add recent HR rate: no database connection")
        return df

    if 'game_date' not in df.columns:
        logger.warning("Cannot add recent HR rate: no game_date column")
        return df

    cursor = conn.cursor(cursor_factory=RealDictCursor)

    for idx, row in df.iterrows():
        if pd.isna(row.get('player_name')) or pd.isna(row.get('game_date')):
            continue

        player_name = row['player_name']
        game_date = row['game_date']

        query = """
        SELECT
            COUNT(CASE WHEN pp.play_result = 'Home Run' THEN 1 END)::NUMERIC /
            NULLIF(COUNT(*), 0) as recent_hr_rate_14d
        FROM play_by_play_plays pp
        JOIN games g ON pp.game_id = g.game_id
        WHERE g.game_date BETWEEN %s - INTERVAL '14 days' AND %s - INTERVAL '1 day'
          AND pp.batter = %s
        """

        cursor.execute(query, (game_date, game_date, player_name))
        result = cursor.fetchone()

        if result and result.get('recent_hr_rate_14d'):
            df.at[idx, 'recent_hr_rate_14d'] = result['recent_hr_rate_14d']

    logger.info("Added recent HR rates")
    return df


def add_composite_features(df: pd.DataFrame) -> pd.DataFrame:
    if 'xslg' in df.columns and 'park_factor' in df.columns:
        df['adjusted_power'] = df['xslg'].fillna(0) * df['park_factor'].fillna(1.0)
    else:
        df['adjusted_power'] = 0

    pitcher_vuln_cols = [c for c in ['pitcher_hr_per_9_30d', 'pitcher_era_30d'] if c in df.columns]
    if pitcher_vuln_cols:
        df['pitcher_hr_vulnerability'] = df[pitcher_vuln_cols].mean(axis=1)
    else:
        df['pitcher_hr_vulnerability'] = 0

    logger.info("Added composite features")
    return df


def engineer_features(df: pd.DataFrame, conn=None) -> pd.DataFrame:
    df = add_park_factors(df, conn)
    df = add_pitcher_rolling_stats(df, conn)
    df = add_hitter_ev_stats(df, conn)
    df = add_xstats(df, conn)
    df = add_recent_hr_rate(df, conn)
    df = add_travel_fatigue(df, conn)
    df = add_weather_context(df)
    df = add_composite_features(df)

    logger.info(f"Feature engineering complete: {len(df.columns)} features")
    return df
