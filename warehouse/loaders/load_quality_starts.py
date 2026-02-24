"""
load_quality_starts.py
Calculate quality starts from MLB API game logs.
QS = game started with 6+ IP and <= 3 ER.

Usage:
    python load_quality_starts.py
"""

import requests
import time
from sqlalchemy import create_engine, text

DB_URL = "postgresql://julianlopez@localhost:5432/baseball"
engine = create_engine(DB_URL)

# Get all pitcher-season combos with starts
with engine.connect() as conn:
    pitcher_seasons = conn.execute(text("""
            SELECT DISTINCT pitcher AS player_id, season
            FROM statcast.pitching_stats
            WHERE games_started >= 1
            ORDER BY season, player_id
        """)).fetchall()
    
print(f"Processing {len(pitcher_seasons)} pitcher-seasons...")

batch = []
errors = 0

for i, (player_id, season) in enumerate(pitcher_seasons):
    try:
        url = f"https://statsapi.mlb.com/api/v1/people/{player_id}/stats?stats=gameLog&group=pitching&season={season}"
        r = requests.get(url, timeout=15)
        data = r.json()

        splits = data['stats'][0]['splits']

        gs = 0
        qs = 0
        for g in splits:
            stat = g['stat']
            if stat.get('gamesStarted', 0) >= 1:
                gs += 1
                ip_str = stat.get('inningsPitched', '0')
                ip = float(ip_str)
                er = stat.get('earnedRuns', 99)
                if ip >= 6.0 and er <= 3:
                    qs += 1

        batch.append({
            'player_id': player_id,
            'season': season,
            'games_started': gs,
            'quality_starts': qs
        })

        # Write in batches of 100
        if len(batch) >= 100:
            with engine.begin() as conn:
                for row in batch:
                    conn.execute(text("""
                        INSERT INTO mlb_api.quality_starts
                        VALUES (:player_id, :season, :games_started, :quality_starts)
                        ON CONFLICT (player_id, season) DO UPDATE
                        SET games_started = EXCLUDED.games_started,
                            quality_starts = EXCLUDED.quality_starts
                    """), row)
            batch = []

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(pitcher_seasons)} done")

        time.sleep(0.3)

    except Exception as e:
        errors += 1
        if errors <= 5:
            print(f"  Error for {player_id}/{season}: {e}")

# Write remaining
if batch:
    with engine.begin() as conn:
        for row in batch:
            conn.execute(text("""
                INSERT INTO mlb_api.quality_starts
                VALUES (:player_id, :season, :games_started, :quality_starts)
                ON CONFLICT (player_id, season) DO UPDATE
                SET games_started = EXCLUDED.games_started,
                    quality_starts = EXCLUDED.quality_starts
            """), row)

print(f"\nDone! Errors: {errors}")
