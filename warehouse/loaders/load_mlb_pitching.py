"""
load_mlb_pitching.py
Pull official pitching stats from MLB Stats API
for every pitcher-season in our Statcast data.

Usage:
  python load_mlb_pitching.py
"""

import time
import requests
import pandas as pd
from sqlalchemy import create_engine, text

DB_URL = "postgresql://julianlopez@localhost:5432/baseball"
engine = create_engine(DB_URL)

# ── get all pitcher-season combos we need ────────────────
with engine.begin() as conn:
    pitcher_seasons = conn.execute(text("""
        SELECT DISTINCT pitcher, season
        FROM statcast.pitching_stats
        ORDER BY season, pitcher
    """)).fetchall()

print(f"Need to pull {len(pitcher_seasons)} pitcher-seasons")

# ── check what we already have ───────────────────────────
with engine.begin() as conn:
    existing = set(conn.execute(text("""
        SELECT player_id, season FROM mlb_api.pitching_stats
    """)).fetchall())

to_pull = [(pid, szn) for pid, szn in pitcher_seasons if (pid, szn) not in existing]
print(f"Already have {len(existing)}, need to pull {len(to_pull)}")

if not to_pull:
    print("Nothing to pull. Done!")
    exit()

# ── pull from MLB API ────────────────────────────────────
rows = []
errors = 0

for i, (player_id, season) in enumerate(to_pull, 1):
    try:
        r = requests.get(
            f"https://statsapi.mlb.com/api/v1/people/{player_id}/stats",
            params={"stats": "season", "season": season, "group": "pitching"},
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
                    "games_started": s.get("gamesStarted"),
                    "games_finished": s.get("gamesFinished"),
                    "complete_games": s.get("completeGames"),
                    "shutouts": s.get("shutouts"),
                    "wins": s.get("wins"),
                    "losses": s.get("losses"),
                    "saves": s.get("saves"),
                    "save_opps": s.get("saveOpportunities"),
                    "holds": s.get("holds"),
                    "blown_saves": s.get("blownSaves"),
                    "innings_pitched": s.get("inningsPitched"),
                    "outs": s.get("outs"),
                    "hits": s.get("hits"),
                    "runs": s.get("runs"),
                    "earned_runs": s.get("earnedRuns"),
                    "home_runs": s.get("homeRuns"),
                    "strike_outs": s.get("strikeOuts"),
                    "base_on_balls": s.get("baseOnBalls"),
                    "intentional_walks": s.get("intentionalWalks"),
                    "hit_by_pitch": s.get("hitByPitch"),
                    "balks": s.get("balks"),
                    "wild_pitches": s.get("wildPitches"),
                    "pickoffs": s.get("pickoffs"),
                    "ground_outs": s.get("groundOuts"),
                    "air_outs": s.get("airOuts"),
                    "doubles": s.get("doubles"),
                    "triples": s.get("triples"),
                    "at_bats": s.get("atBats"),
                    "total_bases": s.get("totalBases"),
                    "batters_faced": s.get("battersFaced"),
                    "number_of_pitches": s.get("numberOfPitches"),
                    "strikes": s.get("strikes"),
                    "stolen_bases": s.get("stolenBases"),
                    "caught_stealing": s.get("caughtStealing"),
                    "ground_into_dp": s.get("groundIntoDoublePlay"),
                    "sac_bunts": s.get("sacBunts"),
                    "sac_flies": s.get("sacFlies"),
                    "catchers_interference": s.get("catchersInterference"),
                    "inherited_runners": s.get("inheritedRunners"),
                    "inherited_runners_scored": s.get("inheritedRunnersScored"),
                    "era": s.get("era"),
                    "whip": s.get("whip"),
                    "avg": s.get("avg"),
                    "obp": s.get("obp"),
                    "slg": s.get("slg"),
                    "ops": s.get("ops"),
                    "strike_percentage": s.get("strikePercentage"),
                    "win_percentage": s.get("winPercentage"),
                    "stolen_base_pct": s.get("stolenBasePercentage"),
                    "caught_stealing_pct": s.get("caughtStealingPercentage"),
                    "go_ao_ratio": s.get("groundOutsToAirouts"),
                    "k_walk_ratio": s.get("strikeoutWalkRatio"),
                    "k_per_9": s.get("strikeoutsPer9Inn"),
                    "bb_per_9": s.get("walksPer9Inn"),
                    "h_per_9": s.get("hitsPer9Inn"),
                    "r_per_9": s.get("runsScoredPer9"),
                    "hr_per_9": s.get("homeRunsPer9"),
                    "pitches_per_inning": s.get("pitchesPerInning"),
                })
        else:
            errors += 1
    except Exception as ex:
        errors += 1

    # progress update every 100
    if i % 100 == 0:
        print(f"  {i}/{len(to_pull)} pulled ({len(rows)} success, {errors} errors)")

    # batch insert every 500
    if len(rows) >= 500:
        df = pd.DataFrame(rows)
        df.to_sql("pitching_stats", engine, schema="mlb_api",
                  if_exists="append", index=False, method="multi", chunksize=500)
        print(f"  Inserted {len(rows)} rows")
        rows = []

    time.sleep(0.05)  # small delay to be respectful

# ── insert remaining rows ────────────────────────────────
if rows:
    df = pd.DataFrame(rows)
    df.to_sql("pitching_stats", engine, schema="mlb_api",
              if_exists="append", index=False, method="multi", chunksize=500)
    print(f"  Inserted {len(rows)} rows")

# ── summary ──────────────────────────────────────────────
with engine.begin() as conn:
    count = conn.execute(text("SELECT COUNT(*) FROM mlb_api.pitching_stats")).scalar()

print(f"""
{'=' * 50}
DONE
{'=' * 50}
  Total rows in mlb_api.pitching_stats: {count:,}
  Errors: {errors}
""")
