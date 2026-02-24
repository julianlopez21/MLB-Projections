from sqlalchemy import create_engine, text

DB_URL = 'postgresql://julianlopez@localhost:5432/baseball'
engine = create_engine(DB_URL)

with engine.connect() as conn:
    batting = conn.execute(text("SELECT MIN(season), MAX(season) FROM mlb_api.batting_stats")).fetchone()
    pitching = conn.execute(text("SELECT MIN(season), MAX(season) FROM mlb_api.pitching_stats")).fetchone()
    statcast = conn.execute(text("SELECT MIN(EXTRACT(YEAR FROM game_date))::INT, MAX(EXTRACT(YEAR FROM game_date))::INT FROM statcast.games")).fetchone()
    constants = conn.execute(text("SELECT MIN(season), MAX(season) FROM statcast.league_constants")).fetchone()
    parks = conn.execute(text("SELECT MIN(season), MAX(season) FROM statcast.park_factors")).fetchone()

FIRST_SEASON = max(batting[0], pitching[0], statcast[0], constants[0], parks[0])
LATEST_SEASON = min(batting[1], pitching[1], statcast[1], constants[1], parks[1])
PROJECTION_SEASON = LATEST_SEASON + 1

print(f"Data available: {FIRST_SEASON} - {LATEST_SEASON}")
print(f"Projecting: {PROJECTION_SEASON}")