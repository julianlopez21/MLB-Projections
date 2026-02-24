"""
load_mlb_batting.py
Pull official batting stats from MLB Stats API
for every batter-season in our Statcast data.

Usage:
  python load_mlb_batting.py
"""

import time
import requests
import pandas as pd
from sqlalchemy import create_engine, text

DB_URL = "postgresql://julianlopez@localhost:5432/baseball"
engine = create_engine(DB_URL)

with engine.begin() as conn:
    batter_seasons = conn.execute(text("""
        SELECT DISTINCT p.batter, EXTRACT(year FROM g.game_date)::int AS season
        FROM statcast.pitches p
        JOIN statcast.games g ON g.game_pk = p.game_pk
        WHERE p.events IS NOT NULL
          AND p.events NOT IN ('truncated_pa','ejection','game_advisory')
          AND g.game_type = 'R'
        ORDER BY season, batter
    """)).fetchall()

print(f"Total batter-seasons: {len(batter_seasons)}")

with engine.begin() as conn:
    existing = set(conn.execute(text(
        "SELECT player_id, season FROM mlb_api.batting_stats"
    )).fetchall())

to_pull = [(pid, szn) for pid, szn in batter_seasons if (pid, szn) not in existing]
print(f"Already have: {len(existing)}, need to pull: {len(to_pull)}")

if not to_pull:
    print("Nothing to pull. Done!")
    exit()

rows = []
errors = 0

for i, (player_id, season) in enumerate(to_pull, 1):
    try:
        r = requests.get(
            f"https://statsapi.mlb.com/api/v1/people/{player_id}/stats",
            params={"stats": "season", "season": season, "group": "hitting"},
            timeout=10
        )
        if r.status_code == 200:
            data = r.json()
            splits = data.get("stats", [{}])[0].get("splits", [])
            if splits:
                s = splits[0]["stat"]
                team = splits[0].get("team", {})
                rows.append({
                    "player_id": player_id,
                    "season": season,
                    "team_id": team.get("id"),
                    "team_name": team.get("name"),
                    "age": s.get("age"),
                    "games_played": s.get("gamesPlayed"),
                    "plate_appearances": s.get("plateAppearances"),
                    "at_bats": s.get("atBats"),
                    "runs": s.get("runs"),
                    "hits": s.get("hits"),
                    "doubles": s.get("doubles"),
                    "triples": s.get("triples"),
                    "home_runs": s.get("homeRuns"),
                    "rbi": s.get("rbi"),
                    "strike_outs": s.get("strikeOuts"),
                    "base_on_balls": s.get("baseOnBalls"),
                    "intentional_walks": s.get("intentionalWalks"),
                    "hit_by_pitch": s.get("hitByPitch"),
                    "stolen_bases": s.get("stolenBases"),
                    "caught_stealing": s.get("caughtStealing"),
                    "ground_into_dp": s.get("groundIntoDoublePlay"),
                    "sac_bunts": s.get("sacBunts"),
                    "sac_flies": s.get("sacFlies"),
                    "total_bases": s.get("totalBases"),
                    "left_on_base": s.get("leftOnBase"),
                    "ground_outs": s.get("groundOuts"),
                    "air_outs": s.get("airOuts"),
                    "number_of_pitches": s.get("numberOfPitches"),
                    "catchers_interference": s.get("catchersInterference"),
                    "avg": s.get("avg"),
                    "obp": s.get("obp"),
                    "slg": s.get("slg"),
                    "ops": s.get("ops"),
                    "babip": s.get("babip"),
                    "stolen_base_pct": s.get("stolenBasePercentage"),
                    "caught_stealing_pct": s.get("caughtStealingPercentage"),
                    "go_ao_ratio": s.get("groundOutsToAirouts"),
                    "ab_per_hr": s.get("atBatsPerHomeRun"),
                })
            else:
                errors += 1
        else:
            errors += 1
    except Exception:
        errors += 1

    if i % 100 == 0:
        print(f"  {i}/{len(to_pull)} pulled ({len(rows)} success, {errors} errors)")

    if len(rows) >= 500:
        df = pd.DataFrame(rows)
        df.to_sql("batting_stats", engine, schema="mlb_api",
                  if_exists="append", index=False, method="multi", chunksize=500)
        print(f"  Inserted {len(rows)} rows")
        rows = []

    time.sleep(0.05)

if rows:
    df = pd.DataFrame(rows)
    df.to_sql("batting_stats", engine, schema="mlb_api",
              if_exists="append", index=False, method="multi", chunksize=500)
    print(f"  Inserted {len(rows)} rows")

with engine.begin() as conn:
    count = conn.execute(text("SELECT COUNT(*) FROM mlb_api.batting_stats")).scalar()

print(f"\nDone! Total rows: {count}, Errors: {errors}")
