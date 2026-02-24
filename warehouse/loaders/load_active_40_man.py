"""
load_rosters.py
Pull current 40-man rosters from MLB API for all 30 teams.
Saves to mlb_api.active_40_man with player_id, team, and season.

Usage:
    python load_rosters.py
"""

import requests
import time
from sqlalchemy import create_engine, text

from datetime import datetime

DB_URL = "postgresql://julianlopez@localhost:5432/baseball"
engine = create_engine(DB_URL)

SEASON = datetime.now().year

# ── Create table ─────────────────────────────────────────
with engine.begin() as conn:
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS mlb_api.active_40_man (
            player_id INTEGER,
            season INTEGER,
            team TEXT,
            team_name TEXT,
            full_name TEXT,
            position TEXT,
            status TEXT,
            PRIMARY KEY (player_id, season)
        )
    """))

# ── Get all MLB teams ────────────────────────────────────
print("Fetching teams...")
r = requests.get(f"https://statsapi.mlb.com/api/v1/teams?sportId=1&season={SEASON}", timeout=10)
teams = r.json()['teams']
print(f"  {len(teams)} teams")

# ── Pull 40-man rosters ──────────────────────────────────
print(f"\nPulling {SEASON} 40-man rosters...")
rows = []

for team in teams:
    team_id = team['id']
    team_abbr = team['abbreviation']
    team_name = team['name']

    try:
        r = requests.get(
            f"https://statsapi.mlb.com/api/v1/teams/{team_id}/roster",
            params={"rosterType": "40Man", "season": SEASON},
            timeout=10
        )
        roster = r.json().get('roster', [])

        for p in roster:
            rows.append({
                'player_id': p['person']['id'],
                'season': SEASON,
                'team': team_abbr,
                'team_name': team_name,
                'full_name': p['person']['fullName'],
                'position': p.get('position', {}).get('abbreviation'),
                'status': p.get('status', {}).get('description'),
            })

        print(f"  {team_abbr}: {len(roster)} players")
    except Exception as e:
        print(f"  {team_abbr}: ERROR - {e}")

    time.sleep(0.2)

# ── Save to database ─────────────────────────────────────
print(f"\nSaving {len(rows)} roster entries...")

with engine.begin() as conn:
    conn.execute(text(f"DELETE FROM mlb_api.active_40_man WHERE season = {SEASON}"))

    for row in rows:
        conn.execute(text("""
            INSERT INTO mlb_api.active_40_man
            VALUES (:player_id, :season, :team, :team_name, :full_name, :position, :status)
            ON CONFLICT (player_id, season) DO UPDATE
            SET team = EXCLUDED.team, team_name = EXCLUDED.team_name,
                full_name = EXCLUDED.full_name, position = EXCLUDED.position,
                status = EXCLUDED.status
        """), row)

# ── Summary ──────────────────────────────────────────────
with engine.begin() as conn:
    count = conn.execute(text(f"SELECT COUNT(*) FROM mlb_api.active_40_man WHERE season = {SEASON}")).scalar()

print(f"\nDone! {count} players on 40-man rosters for {SEASON}")
