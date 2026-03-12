#!/usr/bin/env python3
"""
Load feature snapshot CSVs into derived PostgreSQL tables so the app can
serve from DB-backed snapshots instead of direct CSV reads.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

DB_HOST = os.getenv("DB_HOST", "192.168.1.23")
DB_PORT = int(os.getenv("DB_PORT", 5432))
DB_NAME = os.getenv("DB_NAME", "baseball_migration_test")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "korn5676")


def db_url() -> str:
    return f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    if "game_date" in df.columns:
        df["game_date"] = pd.to_datetime(df["game_date"]).dt.date
    return df


def sync_table(engine, table_name: str, df: pd.DataFrame, index_sql: list[str]) -> None:
    df.to_sql(table_name, engine, if_exists="replace", index=False, method="multi", chunksize=1000)
    with engine.begin() as conn:
        for stmt in index_sql:
            conn.execute(text(stmt))


def main() -> None:
    engine = create_engine(db_url())

    hitter_df = load_csv(DATA_DIR / "complete_dataset.csv")
    pitcher_df = load_csv(DATA_DIR / "pitcher_strikeout_dataset.csv")

    sync_table(
        engine,
        "derived_hitter_features",
        hitter_df,
        [
            "CREATE INDEX IF NOT EXISTS idx_derived_hitter_features_game_date ON derived_hitter_features (game_date)",
            "CREATE INDEX IF NOT EXISTS idx_derived_hitter_features_game_player_team ON derived_hitter_features (game_id, player_name, team)",
        ],
    )
    sync_table(
        engine,
        "derived_pitcher_so_features",
        pitcher_df,
        [
            "CREATE INDEX IF NOT EXISTS idx_derived_pitcher_so_features_game_date ON derived_pitcher_so_features (game_date)",
            "CREATE INDEX IF NOT EXISTS idx_derived_pitcher_so_features_game_starter_team ON derived_pitcher_so_features (game_id, starter_name, team)",
        ],
    )

    print(
        f"synced hitter={len(hitter_df)} rows ({hitter_df['game_date'].min()} to {hitter_df['game_date'].max()}) "
        f"pitcher={len(pitcher_df)} rows ({pitcher_df['game_date'].min()} to {pitcher_df['game_date'].max()})"
    )


if __name__ == "__main__":
    main()
