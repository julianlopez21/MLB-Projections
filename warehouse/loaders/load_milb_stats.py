"""
load_milb_stats.py
Load minor league batting and pitching stats from MLB API.
sportIds: 11=AAA, 12=AA, 13=High-A, 14=Single-A

Usage:
    python load_milb_stats.py
"""

import requests
import time
from sqlalchemy import create_engine, text

DB_URL = "postgresql://julianlopez@localhost:5432/baseball"
engine = create_engine(DB_URL)

# ── Create tables ────────────────────────────────────────
with engine.begin() as conn:
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS mlb_api.milb_batting_stats (
            player_id INTEGER,
            season INTEGER,
            level TEXT,
            team_name TEXT,
            age INTEGER,
            games INTEGER,
            pa INTEGER,
            ab INTEGER,
            hits INTEGER,
            doubles INTEGER,
            triples INTEGER,
            hr INTEGER,
            runs INTEGER,
            rbi INTEGER,
            sb INTEGER,
            cs INTEGER,
            bb INTEGER,
            so INTEGER,
            hbp INTEGER,
            avg NUMERIC,
            obp NUMERIC,
            slg NUMERIC,
            ops NUMERIC,
            babip NUMERIC,
            tb INTEGER,
            PRIMARY KEY (player_id, season, level, team_name)
        )
    """))

    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS mlb_api.milb_pitching_stats (
            player_id INTEGER,
            season INTEGER,
            level TEXT,
            team_name TEXT,
            age INTEGER,
            games INTEGER,
            games_started INTEGER,
            wins INTEGER,
            losses INTEGER,
            saves INTEGER,
            ip NUMERIC,
            hits INTEGER,
            hr INTEGER,
            bb INTEGER,
            so INTEGER,
            hbp INTEGER,
            era NUMERIC,
            whip NUMERIC,
            k_9 NUMERIC,
            bb_9 NUMERIC,
            hr_9 NUMERIC,
            k_bb NUMERIC,
            avg_against NUMERIC,
            obp_against NUMERIC,
            fip NUMERIC,
            PRIMARY KEY (player_id, season, level, team_name)
        )
    """))

SPORT_IDS = {'11': 'AAA', '12': 'AA', '13': 'High-A', '14': 'Single-A'}


def safe_float(val):
    """Convert API stat to float, returning None for missing/invalid values like '.---'"""
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


# ── Load batting stats ───────────────────────────────────
print("=" * 50)
print("Loading MiLB BATTING stats")
print("=" * 50)

for year in range(2015, 2026):
    print(f"  {year}...", end=" ", flush=True)
    total = 0

    for sport_id, level in SPORT_IDS.items():
        offset = 0
        while True:
            url = (f"https://statsapi.mlb.com/api/v1/stats"
                   f"?stats=season&group=hitting&season={year}"
                   f"&sportIds={sport_id}&limit=250&offset={offset}"
                   f"&playerPool=ALL")
            r = requests.get(url, timeout=30)
            data = r.json()

            splits = data['stats'][0]['splits']
            if not splits:
                break

            rows = []
            for s in splits:
                stat = s['stat']
                rows.append({
                    'player_id': s['player']['id'],
                    'season': year,
                    'level': level,
                    'team_name': s['team']['name'],
                    'age': stat.get('age'),
                    'games': stat.get('gamesPlayed', 0),
                    'pa': stat.get('plateAppearances', 0),
                    'ab': stat.get('atBats', 0),
                    'hits': stat.get('hits', 0),
                    'doubles': stat.get('doubles', 0),
                    'triples': stat.get('triples', 0),
                    'hr': stat.get('homeRuns', 0),
                    'runs': stat.get('runs', 0),
                    'rbi': stat.get('rbi', 0),
                    'sb': stat.get('stolenBases', 0),
                    'cs': stat.get('caughtStealing', 0),
                    'bb': stat.get('baseOnBalls', 0),
                    'so': stat.get('strikeOuts', 0),
                    'hbp': stat.get('hitByPitch', 0),
                    'avg': safe_float(stat.get('avg')),
                    'obp': safe_float(stat.get('obp')),
                    'slg': safe_float(stat.get('slg')),
                    'ops': safe_float(stat.get('ops')),
                    'babip': safe_float(stat.get('babip')),
                    'tb': stat.get('totalBases', 0),
                })

            with engine.begin() as conn:
                for row in rows:
                    conn.execute(text("""
                        INSERT INTO mlb_api.milb_batting_stats
                        VALUES (:player_id, :season, :level, :team_name, :age,
                                :games, :pa, :ab, :hits, :doubles, :triples, :hr,
                                :runs, :rbi, :sb, :cs, :bb, :so, :hbp,
                                :avg, :obp, :slg, :ops, :babip, :tb)
                        ON CONFLICT (player_id, season, level, team_name) DO UPDATE
                        SET games = EXCLUDED.games, pa = EXCLUDED.pa, ab = EXCLUDED.ab,
                            hits = EXCLUDED.hits, hr = EXCLUDED.hr, avg = EXCLUDED.avg,
                            obp = EXCLUDED.obp, slg = EXCLUDED.slg, ops = EXCLUDED.ops,
                            sb = EXCLUDED.sb, bb = EXCLUDED.bb, so = EXCLUDED.so,
                            tb = EXCLUDED.tb
                    """), row)

            total += len(rows)
            offset += 250
            if len(splits) < 250:
                break
            time.sleep(0.5)

    print(f"{total} player-seasons")
    time.sleep(1)


# ── Load FIP constants ───────────────────────────────────
fip_constants = {}
with engine.begin() as conn:
    rows = conn.execute(text("SELECT season, cfip FROM statcast.league_constants")).fetchall()
    for row in rows:
        fip_constants[row[0]] = float(row[1])

# ── Load pitching stats ──────────────────────────────────
print("\n" + "=" * 50)
print("Loading MiLB PITCHING stats")
print("=" * 50)

for year in range(2015, 2026):
    print(f"  {year}...", end=" ", flush=True)
    total = 0

    for sport_id, level in SPORT_IDS.items():
        offset = 0
        while True:
            url = (f"https://statsapi.mlb.com/api/v1/stats"
                   f"?stats=season&group=pitching&season={year}"
                   f"&sportIds={sport_id}&limit=250&offset={offset}"
                   f"&playerPool=ALL")
            r = requests.get(url, timeout=30)
            data = r.json()

            splits = data['stats'][0]['splits']
            if not splits:
                break

            rows = []
            for s in splits:
                stat = s['stat']

                ip = safe_float(stat.get('inningsPitched'))
                era = safe_float(stat.get('era'))
                whip = safe_float(stat.get('whip'))
                avg_against = safe_float(stat.get('avg'))
                obp_against = safe_float(stat.get('obp'))

                so = stat.get('strikeOuts', 0)
                bb = stat.get('baseOnBalls', 0)
                hr = stat.get('homeRuns', 0)

                # Compute per-9 rates and FIP from raw counts
                ip_safe = ip if ip and ip > 0 else None
                k_9 = round(so * 9 / ip_safe, 2) if ip_safe else None
                bb_9 = round(bb * 9 / ip_safe, 2) if ip_safe else None
                hr_9 = round(hr * 9 / ip_safe, 2) if ip_safe else None
                k_bb = round(k_9 - bb_9, 2) if k_9 is not None and bb_9 is not None else None
                fip_constant = fip_constants.get(year, 3.10)
                fip = round(((13 * hr + 3 * bb - 2 * so) / ip_safe) + fip_constant, 2) if ip_safe else None

                rows.append({
                    'player_id': s['player']['id'],
                    'season': year,
                    'level': level,
                    'team_name': s['team']['name'],
                    'age': stat.get('age'),
                    'games': stat.get('gamesPlayed', 0),
                    'games_started': stat.get('gamesStarted', 0),
                    'wins': stat.get('wins', 0),
                    'losses': stat.get('losses', 0),
                    'saves': stat.get('saves', 0),
                    'ip': ip,
                    'hits': stat.get('hits', 0),
                    'hr': hr,
                    'bb': bb,
                    'so': so,
                    'hbp': stat.get('hitByPitch', 0),
                    'era': era,
                    'whip': whip,
                    'k_9': k_9,
                    'bb_9': bb_9,
                    'hr_9': hr_9,
                    'k_bb': k_bb,
                    'avg_against': avg_against,
                    'obp_against': obp_against,
                    'fip': fip,
                })

            with engine.begin() as conn:
                for row in rows:
                    conn.execute(text("""
                        INSERT INTO mlb_api.milb_pitching_stats
                        VALUES (:player_id, :season, :level, :team_name, :age,
                                :games, :games_started, :wins, :losses, :saves,
                                :ip, :hits, :hr, :bb, :so, :hbp,
                                :era, :whip, :k_9, :bb_9, :hr_9, :k_bb,
                                :avg_against, :obp_against, :fip)
                        ON CONFLICT (player_id, season, level, team_name) DO UPDATE
                        SET games = EXCLUDED.games, games_started = EXCLUDED.games_started,
                            wins = EXCLUDED.wins, losses = EXCLUDED.losses, saves = EXCLUDED.saves,
                            ip = EXCLUDED.ip, hits = EXCLUDED.hits, hr = EXCLUDED.hr,
                            bb = EXCLUDED.bb, so = EXCLUDED.so, hbp = EXCLUDED.hbp,
                            era = EXCLUDED.era, whip = EXCLUDED.whip,
                            k_9 = EXCLUDED.k_9, bb_9 = EXCLUDED.bb_9, hr_9 = EXCLUDED.hr_9,
                            k_bb = EXCLUDED.k_bb, avg_against = EXCLUDED.avg_against,
                            obp_against = EXCLUDED.obp_against, fip = EXCLUDED.fip
                    """), row)

            total += len(rows)
            offset += 250
            if len(splits) < 250:
                break
            time.sleep(0.5)

    print(f"{total} player-seasons")
    time.sleep(1)

print("\nDone!")
