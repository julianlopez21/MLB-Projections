"""
load_player_bio.py
Pull biographical data from MLB Stats API for all players in our database.

Usage:
  python load_player_bio.py
"""

import time
import requests
import pandas as pd
from sqlalchemy import create_engine, text

DB_URL = "postgresql://julianlopez@localhost:5432/baseball"
engine = create_engine(DB_URL)

with engine.begin() as conn:
    all_players = [r[0] for r in conn.execute(text(
        "SELECT DISTINCT player_id FROM statcast.players"
    )).fetchall()]

    existing = set(r[0] for r in conn.execute(text(
        "SELECT player_id FROM mlb_api.player_bio"
    )).fetchall())

to_pull = [pid for pid in all_players if pid not in existing]
print(f"Total players: {len(all_players)}, already have: {len(existing)}, need to pull: {len(to_pull)}")

if not to_pull:
    print("Nothing to pull. Done!")
    exit()

rows = []
errors = 0

for i, player_id in enumerate(to_pull, 1):
    try:
        r = requests.get(
            f"https://statsapi.mlb.com/api/v1/people/{player_id}",
            timeout=10
        )
        if r.status_code == 200:
            data = r.json()
            if data.get("people"):
                p = data["people"][0]
                pos = p.get("primaryPosition", {})
                rows.append({
                    "player_id": player_id,
                    "full_name": p.get("fullName"),
                    "first_name": p.get("firstName"),
                    "last_name": p.get("lastName"),
                    "birth_date": p.get("birthDate"),
                    "birth_city": p.get("birthCity"),
                    "birth_state": p.get("birthStateProvince"),
                    "birth_country": p.get("birthCountry"),
                    "height": p.get("height"),
                    "weight": p.get("weight"),
                    "primary_position": pos.get("abbreviation"),
                    "position_type": pos.get("type"),
                    "bat_side": p.get("batSide", {}).get("code"),
                    "pitch_hand": p.get("pitchHand", {}).get("code"),
                    "draft_year": p.get("draftYear"),
                    "mlb_debut_date": p.get("mlbDebutDate"),
                    "active": p.get("active"),
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
        df.to_sql("player_bio", engine, schema="mlb_api",
                  if_exists="append", index=False, method="multi", chunksize=500)
        print(f"  Inserted {len(rows)} rows")
        rows = []

    time.sleep(0.05)

if rows:
    df = pd.DataFrame(rows)
    df.to_sql("player_bio", engine, schema="mlb_api",
              if_exists="append", index=False, method="multi", chunksize=500)
    print(f"  Inserted {len(rows)} rows")

with engine.begin() as conn:
    count = conn.execute(text("SELECT COUNT(*) FROM mlb_api.player_bio")).scalar()

print(f"\nDone! Total rows: {count}, Errors: {errors}")
