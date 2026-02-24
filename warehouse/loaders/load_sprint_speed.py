"""
load_sprint_speed.py
Load Statcast sprint speed data for all available seasons.

Usage:
    python load_sprint_speed.py
"""

from pybaseball import statcast_sprint_speed
from sqlalchemy import create_engine, text
import pandas as pd
import time

DB_URL = "postgresql://julianlopez@localhost:5432/baseball"
engine = create_engine(DB_URL)

# Create table
with engine.begin() as conn:
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS statcast.sprint_speed (
            player_id INTEGER,
            season INTEGER,
            sprint_speed NUMERIC,
            hp_to_1b NUMERIC,
            bolts INTEGER,
            competitive_runs INTEGER,
            PRIMARY KEY (player_id, season)
        )
    """))

for year in range(2015, 2026):
    print(f"Loading {year}...", end=" ", flush=True)
    try:
        df = statcast_sprint_speed(year)
        records = df[['player_id', 'sprint_speed', 'hp_to_1b', 'bolts', 'competitive_runs']].copy()
        records['season'] = year
        records['player_id'] = records['player_id'].astype(int)

        with engine.begin() as conn:
            conn.execute(text("DELETE FROM statcast.sprint_speed WHERE season = :yr"), {"yr": year})

        records.to_sql('sprint_speed', engine, schema='statcast',
                       if_exists='append', index=False, method='multi', chunksize=500)
        print(f"{len(records)} players")
    except Exception as e:
        print(f"ERROR: {e}")
    time.sleep(2)

print("\nDone!")
