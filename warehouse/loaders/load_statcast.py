"""
backfill_statcast.py
Load a range of dates into the baseball warehouse.
Idempotent — delete + re-insert per game_pk.

Usage (PyCharm Run Config or terminal):
  python backfill_statcast.py 2024-03-28 2024-04-03
  python backfill_statcast.py 2024-06-15              (single day)
"""

import sys
import time
import requests
import pandas as pd
from datetime import datetime, timedelta
from pybaseball import statcast
from sqlalchemy import create_engine, text
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="pybaseball")


DB_URL = "postgresql://julianlopez@localhost:5432/baseball"
engine = create_engine(DB_URL)

# ── date range from command line ─────────────────────────
if len(sys.argv) == 3:
    start = sys.argv[1]
    end   = sys.argv[2]
elif len(sys.argv) == 2:
    start = end = sys.argv[1]
else:
    print("Usage: python backfill_statcast.py START_DATE [END_DATE]")
    print("  e.g. python backfill_statcast.py 2024-03-28 2024-04-07")
    sys.exit(1)

print(f"Backfilling: {start} → {end}")

# ── pull in weekly chunks (Savant can timeout on big ranges) ─
def date_chunks(start_str, end_str, chunk_days=7):
    s = datetime.strptime(start_str, "%Y-%m-%d")
    e = datetime.strptime(end_str, "%Y-%m-%d")
    while s <= e:
        chunk_end = min(s + timedelta(days=chunk_days - 1), e)
        yield s.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")
        s = chunk_end + timedelta(days=1)

all_dfs = []
for chunk_start, chunk_end in date_chunks(start, end):
    print(f"  Pulling {chunk_start} → {chunk_end} ...", end=" ", flush=True)
    try:
        df = statcast(start_dt=chunk_start, end_dt=chunk_end)
        if df is not None and len(df) > 0:
            all_dfs.append(df)
            print(f"{len(df):,} pitches")
        else:
            print("no data")
    except Exception as ex:
        print(f"ERROR: {ex}")
    time.sleep(2)

if not all_dfs:
    print("No data pulled. Exiting.")
    sys.exit(0)

df = pd.concat(all_dfs, ignore_index=True)
print(f"\nTotal pitches pulled: {len(df):,}")

# ── upsert games ─────────────────────────────────────────
games = df[['game_pk', 'game_date', 'game_type', 'home_team', 'away_team']].drop_duplicates()

with engine.begin() as conn:
    for _, g in games.iterrows():
        conn.execute(text("""
            INSERT INTO statcast.games (game_pk, game_date, game_type, home_team, away_team)
            VALUES (:pk, :dt, :gt, :home, :away)
            ON CONFLICT (game_pk) DO NOTHING
        """), {"pk": int(g.game_pk), "dt": g.game_date,
               "gt": g.game_type, "home": g.home_team, "away": g.away_team})

print(f"Upserted {len(games)} games")

# ── upsert players ───────────────────────────────────────
pitcher_names = df[['pitcher', 'player_name']].drop_duplicates()
name_map = dict(zip(pitcher_names['pitcher'].astype(int), pitcher_names['player_name']))

all_ids = set(df['pitcher'].dropna().astype(int)) | set(df['batter'].dropna().astype(int))
need_lookup = all_ids - set(name_map.keys())

if need_lookup:
    with engine.begin() as conn:
        existing = conn.execute(
            text("SELECT player_id, player_name FROM statcast.players WHERE player_id = ANY(:ids)"),
            {"ids": list(need_lookup)}
        ).fetchall()
        for pid, pname in existing:
            if pname:
                name_map[pid] = pname
                need_lookup.discard(pid)

print(f"Have {len(name_map)} names, need to look up {len(need_lookup)} from MLB API")

if need_lookup:
    found = 0
    for i, pid in enumerate(need_lookup, 1):
        try:
            r = requests.get(f"https://statsapi.mlb.com/api/v1/people/{pid}", timeout=10)
            if r.status_code == 200:
                p = r.json()['people'][0]
                name_map[pid] = f"{p['lastName']}, {p['firstName']}"
                found += 1
        except Exception:
            pass
        if i % 100 == 0:
            print(f"  {i}/{len(need_lookup)} looked up...")
            time.sleep(1)
    print(f"Found {found}/{len(need_lookup)} names from API")

with engine.begin() as conn:
    for pid, pname in name_map.items():
        conn.execute(text("""
            INSERT INTO statcast.players (player_id, player_name)
            VALUES (:id, :name)
            ON CONFLICT (player_id) DO UPDATE
            SET player_name = COALESCE(statcast.players.player_name, :name)
        """), {"id": int(pid), "name": pname})

print(f"Upserted {len(name_map)} players")

# ── load pitches ─────────────────────────────────────────
pitch_cols = [
    'game_pk', 'at_bat_number', 'pitch_number', 'pitcher', 'batter',
    'pitch_type', 'pitch_name', 'release_speed', 'release_pos_x', 'release_pos_z',
    'release_spin_rate', 'spin_axis', 'pfx_x', 'pfx_z', 'plate_x', 'plate_z',
    'zone', 'p_throws', 'stand', 'balls', 'strikes', 'outs_when_up', 'inning',
    'inning_topbot', 'events', 'description', 'type', 'launch_speed',
    'launch_angle', 'hit_distance_sc', 'hc_x', 'hc_y',
    'estimated_ba_using_speedangle', 'estimated_woba_using_speedangle',
    'bb_type', 'on_1b', 'on_2b', 'on_3b', 'if_fielding_alignment',
    'of_fielding_alignment', 'sz_top', 'sz_bot'
]

# Get actual DB columns so we never insert something that doesn't exist
with engine.begin() as conn:
    db_cols = {row[0] for row in conn.execute(text(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_schema = 'statcast' AND table_name = 'pitches'"
    )).fetchall()}

skip = {'pitch_id'}  # auto-generated, never insert
available = [c for c in pitch_cols if c in df.columns and c in db_cols and c not in skip]

pitches = df[available].copy()

int_cols = ['game_pk', 'at_bat_number', 'pitch_number', 'pitcher', 'batter',
            'zone', 'balls', 'strikes', 'outs_when_up', 'inning']
for c in int_cols:
    if c in pitches.columns:
        pitches[c] = pd.to_numeric(pitches[c], errors='coerce').astype('Int64')

# delete existing pitches for these games, then re-insert
game_pks = pitches['game_pk'].dropna().unique().tolist()

with engine.begin() as conn:
    conn.execute(
        text("DELETE FROM statcast.pitches WHERE game_pk = ANY(:pks)"),
        {"pks": game_pks}
    )

pitches.to_sql('pitches', engine, schema='statcast',
               if_exists='append', index=False,
               method='multi', chunksize=500)

print(f"Loaded {len(pitches):,} pitches for {len(game_pks)} games")

# ── summary ──────────────────────────────────────────────
with engine.begin() as conn:
    counts = conn.execute(text("""
        SELECT
            (SELECT COUNT(*) FROM statcast.games)   AS games,
            (SELECT COUNT(*) FROM statcast.players)  AS players,
            (SELECT COUNT(*) FROM statcast.pitches)  AS pitches
    """)).fetchone()

print(f"""
{'=' * 50}
DATABASE TOTALS
{'=' * 50}
  Games:     {counts[0]:,}
  Players:   {counts[1]:,}
  Pitches:   {counts[2]:,}
""")
