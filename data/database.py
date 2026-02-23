"""
ProjectionAI - Database Layer
PostgreSQL database connection and schema management
"""

import psycopg2
from psycopg2 import pool
from psycopg2.extras import execute_values
import logging
from datetime import datetime
from typing import List, Dict, Optional, Any
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Database:
    """PostgreSQL database connection pool and utilities"""

    def __init__(self, database="projectionai", host="localhost", port=5432,
                 user="futurepr0n", password=None, pool_size=10):
        """
        Initialize database connection pool

        Args:
            database: Database name
            host: Database host
            port: Database port
            user: Database user
            password: Database password (None for trust auth)
            pool_size: Connection pool size
        """
        self.database = database
        self.host = host
        self.port = port
        self.user = user
        self.password = password or os.getenv('DB_PASSWORD')
        self.pool = None

    def connect(self):
        """Create connection pool"""
        try:
            self.pool = psycopg2.pool.ThreadedConnectionPool(
                minconn=2,
                maxconn=20,
                database=self.database,
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password
            )
            logger.info(f"✅ Connected to database: {self.database}")
            return True
        except psycopg2.OperationalError as e:
            logger.error(f"❌ Failed to connect to database: {e}")
            logger.info("💡 Try running: createdb projectionai")
            return False

    def close(self):
        """Close all connections in pool"""
        if self.pool:
            self.pool.closeall()
            logger.info("✅ Database connection pool closed")

    def get_connection(self):
        """Get a connection from the pool"""
        if not self.pool:
            self.connect()
        return self.pool.getconn()

    def return_connection(self, conn):
        """Return connection to pool"""
        self.pool.putconn(conn)

    def execute_query(self, query: str, params: tuple = None, fetch: str = "all"):
        """
        Execute a SQL query

        Args:
            query: SQL query string
            params: Query parameters
            fetch: "all", "one", or "none" (default: "all")

        Returns:
            Query results or None
        """
        conn = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            cursor.execute(query, params or ())

            if fetch == "all":
                results = cursor.fetchall()
                columns = [desc[0] for desc in cursor.description]
                return [dict(zip(columns, row)) for row in results]
            elif fetch == "one":
                row = cursor.fetchone()
                if row:
                    columns = [desc[0] for desc in cursor.description]
                    return dict(zip(columns, row))
                return None
            else:
                conn.commit()
                return None

        except Exception as e:
            logger.error(f"❌ Query execution error: {e}")
            if conn:
                conn.rollback()
            return None
        finally:
            if conn:
                self.return_connection(conn)

    def execute_batch(self, query: str, data: List[tuple]):
        """
        Execute batch insert with psycopg2.execute_values

        Args:
            query: SQL query with placeholders
            data: List of tuples containing row data
        """
        conn = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            execute_values(cursor, query, data)
            conn.commit()

            logger.info(f"✅ Batch insert: {len(data)} rows")

        except Exception as e:
            logger.error(f"❌ Batch insert error: {e}")
            if conn:
                conn.rollback()
        finally:
            if conn:
                self.return_connection(conn)

    def create_schema(self):
        """Create all database tables"""
        try:
            self._create_games_table()
            self._create_players_table()
            self._create_pitchers_table()
            self._create_statcast_data_table()
            self._create_predictions_table()
            self._create_odds_table()
            self._create_bets_table()
            self._create_bankroll_table()
            self._create_indexes()
            logger.info("✅ All tables created successfully")
        except Exception as e:
            logger.error(f"❌ Schema creation error: {e}")

    def _create_games_table(self):
        """Create games table"""
        query = """
        CREATE TABLE IF NOT EXISTS games (
            id SERIAL PRIMARY KEY,
            game_id VARCHAR(20) UNIQUE,
            home_team VARCHAR(3) NOT NULL,
            away_team VARCHAR(3) NOT NULL,
            home_pitcher VARCHAR(50),
            away_pitcher VARCHAR(50),
            home_pitcher_id INTEGER,
            away_pitcher_id INTEGER,
            game_date DATE NOT NULL,
            game_time TIME,
            venue VARCHAR(50),
            weather_temp FLOAT,
            weather_wind FLOAT,
            weather_conditions VARCHAR(20),
            home_score INTEGER,
            away_score INTEGER,
            status VARCHAR(20) DEFAULT 'scheduled',
            home_lineup JSONB,
            away_lineup JSONB,
            home_hr_hitters TEXT[],
            away_hr_hitters TEXT[],
            home_hit_hitters TEXT[],
            away_hit_hitters TEXT[],
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        self.execute_query(query, fetch="none")
        logger.info("✅ games table created")

    def _create_players_table(self):
        """Create players (batters) table"""
        query = """
        CREATE TABLE IF NOT EXISTS players (
            id SERIAL PRIMARY KEY,
            player_id INTEGER UNIQUE,
            name VARCHAR(50) NOT NULL,
            team VARCHAR(3),
            position VARCHAR(10),
            bats VARCHAR(1),
            throws VARCHAR(1),
            height FLOAT,
            weight INTEGER,
            debut_date DATE,
            active BOOLEAN DEFAULT TRUE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        self.execute_query(query, fetch="none")
        logger.info("✅ players table created")

    def _create_pitchers_table(self):
        """Create pitchers table"""
        query = """
        CREATE TABLE IF NOT EXISTS pitchers (
            id SERIAL PRIMARY KEY,
            pitcher_id INTEGER UNIQUE,
            name VARCHAR(50) NOT NULL,
            team VARCHAR(3),
            throws VARCHAR(1),
            height FLOAT,
            weight INTEGER,
            debut_date DATE,
            active BOOLEAN DEFAULT TRUE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        self.execute_query(query, fetch="none")
        logger.info("✅ pitchers table created")

    def _create_statcast_data_table(self):
        """Create Statcast data table"""
        query = """
        CREATE TABLE IF NOT EXISTS statcast_data (
            id SERIAL PRIMARY KEY,
            game_id VARCHAR(20),
            player_id INTEGER NOT NULL,
            is_pitcher BOOLEAN DEFAULT FALSE,
            stat_date DATE NOT NULL,
            season INTEGER NOT NULL,

            -- Hitter metrics
            pa INTEGER,
            ab INTEGER,
            hits INTEGER,
            hr INTEGER,
            so INTEGER,
            bb INTEGER,
            avg FLOAT,
            obp FLOAT,
            slg FLOAT,
            ops FLOAT,
            wrc_plus INTEGER,

            -- Statcast advanced metrics
            barrel_rate FLOAT,
            ev95_plus FLOAT,
            sweet_spot_percent FLOAT,
            avg_hit_speed FLOAT,
            avg_hit_angle FLOAT,
            max_hit_speed FLOAT,
            hard_hit_percent FLOAT,
            launch_angle_avg FLOAT,

            -- Batted ball outcomes
            fb_percent FLOAT,
            gb_percent FLOAT,
            ld_percent FLOAT,
            iffb_percent FLOAT,
            pull_percent FLOAT,
            center_percent FLOAT,
            oppo_percent FLOAT,

            -- Pitcher metrics (if is_pitcher = TRUE)
            innings_pitched FLOAT,
            era FLOAT,
            fip FLOAT,
            k_percent FLOAT,
            bb_percent FLOAT,
            hr_per_9 FLOAT,
            avg_hit_speed_allowed FLOAT,
            barrel_rate_allowed FLOAT,
            whip FLOAT,

            -- Season-to-date vs career split
            is_sts BOOLEAN DEFAULT FALSE,

            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

            UNIQUE(game_id, player_id, is_pitcher, stat_date)
        );
        """
        self.execute_query(query, fetch="none")
        logger.info("✅ statcast_data table created")

    def _create_predictions_table(self):
        """Create predictions table"""
        query = """
        CREATE TABLE IF NOT EXISTS predictions (
            id SERIAL PRIMARY KEY,
            game_id VARCHAR(20),
            player_id INTEGER,
            player_name VARCHAR(50),
            team VARCHAR(3),
            opponent VARCHAR(3),
            pitcher_id INTEGER,
            pitcher_name VARCHAR(50),
            prediction_date DATE NOT NULL,
            prediction_type VARCHAR(10) NOT NULL, -- 'HR', 'HIT', 'SO'

            -- Model predictions
            hr_probability FLOAT,
            hit_probability FLOAT,
            so_probability FLOAT,

            -- Signal classification
            signal VARCHAR(15), -- 'STRONG_BUY', 'BUY', 'AVOID', 'STRONG_SELL'
            confidence INTEGER,

            -- Reasoning
            reasoning TEXT,

            -- Edge calculation
            book_odds FLOAT,
            implied_probability FLOAT,
            edge FLOAT,

            -- Result
            is_correct BOOLEAN,
            actual_outcome VARCHAR(10),
            result_confirmed_at TIMESTAMP,

            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

            CONSTRAINT fk_game FOREIGN KEY (game_id) REFERENCES games(game_id),
            CONSTRAINT fk_player FOREIGN KEY (player_id) REFERENCES players(player_id)
        );
        """
        self.execute_query(query, fetch="none")
        logger.info("✅ predictions table created")

    def _create_odds_table(self):
        """Create odds table"""
        query = """
        CREATE TABLE IF NOT EXISTS odds (
            id SERIAL PRIMARY KEY,
            game_id VARCHAR(20),
            player_id INTEGER,
            player_name VARCHAR(50),
            book VARCHAR(20) NOT NULL,
            bet_type VARCHAR(10) NOT NULL, -- 'HR', 'HIT', 'SO'
            line_type VARCHAR(10) NOT NULL, -- 'OVER', 'UNDER'
            line_number FLOAT NOT NULL, -- e.g., 0.5 for HR O/U 0.5
            american_odds INTEGER, -- e.g., +150, -110
            decimal_odds FLOAT,
            implied_probability FLOAT,
            is_opening BOOLEAN DEFAULT FALSE,
            is_closing BOOLEAN DEFAULT FALSE,
            scraped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

            CONSTRAINT fk_odds_game FOREIGN KEY (game_id) REFERENCES games(game_id)
        );
        """
        self.execute_query(query, fetch="none")
        logger.info("✅ odds table created")

    def _create_bets_table(self):
        """Create bets table"""
        query = """
        CREATE TABLE IF NOT EXISTS bets (
            id SERIAL PRIMARY KEY,
            prediction_id INTEGER NOT NULL,
            game_id VARCHAR(20),
            player_id INTEGER,
            player_name VARCHAR(50),
            bet_type VARCHAR(10) NOT NULL, -- 'HR', 'HIT', 'SO'
            line_type VARCHAR(10) NOT NULL, -- 'OVER', 'UNDER'
            line_number FLOAT NOT NULL,
            book VARCHAR(20) NOT NULL,

            -- Kelly criterion
            bankroll_before FLOAT NOT NULL,
            kelly_fraction FLOAT NOT NULL,
            bet_amount FLOAT NOT NULL,
            american_odds INTEGER,
            decimal_odds FLOAT NOT NULL,
            implied_probability FLOAT NOT NULL,

            -- Result
            won BOOLEAN,
            profit_loss FLOAT,
            result_confirmed_at TIMESTAMP,

            -- Tracking
            placed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

            CONSTRAINT fk_bet_prediction FOREIGN KEY (prediction_id) REFERENCES predictions(id)
        );
        """
        self.execute_query(query, fetch="none")
        logger.info("✅ bets table created")

    def _create_bankroll_table(self):
        """Create bankroll tracking table"""
        query = """
        CREATE TABLE IF NOT EXISTS bankroll (
            id SERIAL PRIMARY KEY,
            transaction_date DATE NOT NULL,
            starting_balance FLOAT NOT NULL,
            bets_placed INTEGER DEFAULT 0,
            bets_won INTEGER DEFAULT 0,
            bets_lost INTEGER DEFAULT 0,
            profit_loss FLOAT DEFAULT 0,
            roi_percent FLOAT DEFAULT 0,
            current_balance FLOAT NOT NULL,
            notes TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        self.execute_query(query, fetch="none")
        logger.info("✅ bankroll table created")

    def _create_indexes(self):
        """Create indexes for performance"""
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_games_date ON games(game_date);",
            "CREATE INDEX IF NOT EXISTS idx_games_status ON games(status);",
            "CREATE INDEX IF NOT EXISTS idx_statcast_player_date ON statcast_data(player_id, stat_date);",
            "CREATE INDEX IF NOT EXISTS idx_statcast_pitcher_date ON statcast_data(player_id, stat_date) WHERE is_pitcher = TRUE;",
            "CREATE INDEX IF NOT EXISTS idx_predictions_date_type ON predictions(prediction_date, prediction_type);",
            "CREATE INDEX IF NOT EXISTS idx_predictions_game ON predictions(game_id);",
            "CREATE INDEX IF NOT EXISTS idx_odds_game_player ON odds(game_id, player_id);",
            "CREATE INDEX IF NOT EXISTS idx_bets_date ON bets(placed_at);",
            "CREATE INDEX IF NOT EXISTS idx_bankroll_date ON bankroll(transaction_date);",
        ]

        for idx_query in indexes:
            self.execute_query(idx_query, fetch="none")

        logger.info("✅ All indexes created")


# Singleton database instance
_db = None


def get_database():
    """Get or create database singleton"""
    global _db
    if _db is None:
        _db = Database()
        _db.connect()
    return _db


if __name__ == "__main__":
    # Test database connection and create schema
    db = Database()
    if db.connect():
        db.create_schema()
        db.close()
    else:
        print("❌ Could not connect to database. Make sure PostgreSQL is running.")
