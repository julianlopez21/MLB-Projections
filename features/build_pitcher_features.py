import sys
sys.path.append('..')
from config import engine, FIRST_SEASON, LATEST_SEASON
from sqlalchemy import text
import pandas as pd
import numpy as np

MIN_IP = 10

# ── Step 1: Load base data ───────────────────────────────
print("Step 1: Loading pitching stats...")

trad_statcast = pd.read_sql(text(f"""
    SELECT * from statcast.pitching_stats
    WHERE ip >= {MIN_IP}
    AND season BETWEEN {FIRST_SEASON} AND {LATEST_SEASON}
    AND split_type = 'total'"""), engine)

trad_mlb = pd.read_sql(text(f"""
    SELECT        
        player_id, season,
        holds, blown_saves, save_opps,
        inherited_runners, inherited_runners_scored,
        wild_pitches, complete_games, shutouts
        from mlb_api.pitching_stats
    WHERE season BETWEEN {FIRST_SEASON} AND {LATEST_SEASON}"""), engine)

# Get per-team splits for IP-weighted park factors and team strength
# (total splits have pitcher_team = 'TOT', so we use per-team splits)
team_splits = pd.read_sql(text(f"""
    SELECT pitcher AS player_id, season, pitcher_team AS team, ip
    FROM statcast.pitching_stats
    WHERE split_type != 'total'
      AND season BETWEEN {FIRST_SEASON} AND {LATEST_SEASON}
"""), engine)

# Build display team string (e.g. "CHC/NYY" for traded players)
team_lookup = (team_splits.sort_values('ip', ascending=False)
               .groupby(['player_id', 'season'])['team']
               .apply(lambda x: '/'.join(x))
               .reset_index()
               .rename(columns={'team': 'team'}))

advanced = pd.read_sql(text(f"""
    SELECT * from statcast.pitching_advanced
    WHERE season BETWEEN {FIRST_SEASON} AND {LATEST_SEASON}"""), engine)

quality_starts = pd.read_sql(text(f"""
    SELECT 
        player_id, season, quality_starts
        from mlb_api.quality_starts
    WHERE season BETWEEN {FIRST_SEASON} AND {LATEST_SEASON}"""), engine)

bio = pd.read_sql(text("""
    SELECT player_id, bat_side, primary_position, birth_date, mlb_debut_date
    FROM mlb_api.player_bio"""), engine)

# ── Step 2: Merge into single dataframe ──────────────────
print("\nStep 2: Merging datasets...")

trad_statcast = trad_statcast.rename(columns={'pitcher': 'player_id'})
trad_statcast = trad_statcast.drop(columns=['pitcher_team'])

features = trad_statcast.merge(trad_mlb, on=['player_id', 'season'], how='left')
features = features.merge(team_lookup, on=['player_id', 'season'], how='left')
features = features.merge(advanced, on=['player_id', 'season'], how='left')
features = features.merge(quality_starts, on=['player_id', 'season'], how='left')
features = features.merge(bio, on=['player_id'], how='left')

features['birth_date'] = pd.to_datetime(features['birth_date'])
features['mlb_debut_date'] = pd.to_datetime(features['mlb_debut_date'])
features['years_experience'] = features['season'] - features['mlb_debut_date'].dt.year
features['years_experience'] = features['years_experience'].clip(lower=0)

print(f"  Merged: {len(features)} rows")

# ── Step 3: Data Validation ──────────────────
n_before = len(trad_statcast)
n_after = len(features)
if n_before != n_after:
    print(f"ERROR: Row count changed: {n_before} → {n_after} (duplicates from merge!)")
else:
    print(f"  ✓ Row count stable: {n_after}")

dupes = features.groupby(['player_id', 'season']).size()
dupes = dupes[dupes > 1]
if len(dupes) > 0:
    print(f"ERROR: {len(dupes)} duplicate player-seasons detected!")
    print(f"Examples: {dupes.head().index.tolist()}")
else:
    print(f"No duplicate player-seasons detected!")

# ── Step 4: Engineer derived features ────────────────────
print("\nStep 4: Engineering derived features...")

features['sv_hld'] = features['saves'].fillna(0) + features['holds'].fillna(0)
features['starter_pct'] = features['games_started'] / features['games'].replace(0, np.nan)
features['is_starter'] = (features['starter_pct'] >= 0.5).astype(int)
features['ip_per_start'] = features['ip'] / features['games_started']
features['fb_brk_velo_diff'] = features['fb_velo'] - features['brk_velo']
features['fb_offspeed_velo_diff'] = features['fb_velo'] - features['offspeed_velo']
features['gb_fb_ratio'] = features['gb_rate'] / features['fb_rate'].replace(0, np.nan)
features['ir_scored_pct'] = features['inherited_runners_scored'] / features['inherited_runners'].replace(0, np.nan)
features['babip_minus_xwoba'] = features['babip'] - features['xwoba_against']

print(f"  Total columns: {len(features.columns)}")

# ── Step 5: Regress small samples toward league average ──
# Players with fewer IP have noisy stats. Blend toward league
# average proportional to sample size.
# Formula: blended = (IP * stat + k * lg_avg) / (IP + k)
# k = IP needed for 50/50 weight with league average
print("\nStep 5: Regressing small samples to mean...")

league_avgs = features.groupby('season').agg({
    'era': 'mean', 'whip': 'mean', 'k_pct': 'mean', 'bb_pct': 'mean',
    'babip': 'mean', 'hr_9': 'mean', 'fip': 'mean', 'xwoba_against': 'mean',
    'hr_fb': 'mean', 'ba_against': 'mean',
}).reset_index()
league_avgs = league_avgs.rename(columns={c: f'lg_{c}' for c in league_avgs.columns if c != 'season'})
features = features.merge(league_avgs, on='season', how='left')

reliability = {
    'era': 80, 'whip': 70, 'k_pct': 40, 'bb_pct': 50,
    'babip': 120, 'hr_9': 100, 'fip': 60, 'xwoba_against': 60,
    'hr_fb': 100, 'ba_against': 80,
}
for stat, k in reliability.items():
    features[f'{stat}_regressed'] = (
        features['ip'] * features[stat] + k * features[f'lg_{stat}']
    ) / (features['ip'] + k)

features = features.drop(columns=[c for c in features.columns if c.startswith('lg_')])

print(f"  Added {len(reliability)} regressed features. Total columns: {len(features.columns)}")

# ── Step 6: Rolling multi-year weighted averages ─────────
print("\nStep 6: Building rolling averages...")

rolling_cols = [
    # Statcast advanced (skill-based, benefit most from smoothing)
    'fb_velo', 'ff_velo', 'brk_velo', 'offspeed_velo',
    'extension', 'whiff_rate', 'chase_rate', 'zone_rate', 'csw_rate',
    'avg_ev_against', 'avg_la_against', 'xwoba_against',
    'barrel_rate_against', 'hard_hit_rate_against',
    'gb_rate', 'fb_rate',
    # Traditional rates
    'k_pct', 'bb_pct', 'hr_fb', 'babip',
    # Derived
    'fb_brk_velo_diff', 'fb_offspeed_velo_diff', 'gb_fb_ratio',
]

features = features.sort_values(['player_id', 'season'])

for col in rolling_cols:
    features[f'{col}_2yr'] = (
        features.groupby('player_id')[col]
        .apply(lambda x: x.rolling(2, min_periods=1)
               .apply(lambda w: np.average(w, weights=range(1, len(w)+1))))
        .reset_index(level=0, drop=True)
    )
    features[f'{col}_3yr'] = (
        features.groupby('player_id')[col]
        .apply(lambda x: x.rolling(3, min_periods=1)
               .apply(lambda w: np.average(w, weights=range(1, len(w)+1))))
        .reset_index(level=0, drop=True)
    )

print(f"  Added {len(rolling_cols) * 2} rolling features. Total columns: {len(features.columns)}")

# ── Step 7: Platoon splits ───────────────────────────────
# Performance vs LHB and RHB separately. Some pitchers have
# massive platoon gaps — e.g. lefty specialists who dominate
# LHB but get crushed by RHB.
print("\nStep 7: Building platoon split features...")

platoon = pd.read_sql(text(f"""
    SELECT
        p.pitcher AS player_id,
        EXTRACT(YEAR FROM g.game_date)::INT AS season,
        p.stand AS bat_side,
        COUNT(*) AS pitches_thrown,
        ROUND(AVG(CASE WHEN p.launch_speed IS NOT NULL THEN p.launch_speed END)::NUMERIC, 1) AS exit_velo,
        ROUND(AVG(CASE WHEN p.launch_speed IS NOT NULL THEN
            p.estimated_woba_using_speedangle END)::NUMERIC, 3) AS xwoba,
        ROUND(SUM(CASE WHEN p.description IN (
            'swinging_strike', 'swinging_strike_blocked'
        ) THEN 1 ELSE 0 END)::NUMERIC / NULLIF(SUM(CASE WHEN p.description IN (
            'swinging_strike', 'swinging_strike_blocked',
            'foul', 'foul_tip', 'foul_bunt',
            'hit_into_play', 'hit_into_play_no_out', 'hit_into_play_score'
        ) THEN 1 ELSE 0 END), 0), 3) AS whiff_rate
    FROM statcast.pitches p
    JOIN statcast.games g ON p.game_pk = g.game_pk
    WHERE g.game_type = 'R'
      AND p.stand IN ('L', 'R')
      AND EXTRACT(YEAR FROM g.game_date) BETWEEN {FIRST_SEASON} AND {LATEST_SEASON}
    GROUP BY p.pitcher, EXTRACT(YEAR FROM g.game_date), p.stand
"""), engine)

for split in ['L', 'R']:
    split_df = platoon[platoon['bat_side'] == split].copy()
    split_df = split_df.rename(columns={
        'pitches_thrown': f'vs{split}_pitches',
        'exit_velo': f'vs{split}_exit_velo',
        'xwoba': f'vs{split}_xwoba',
        'whiff_rate': f'vs{split}_whiff_rate',
    })
    split_df = split_df.drop(columns=['bat_side'])
    features = features.merge(split_df, on=['player_id', 'season'], how='left')

# Platoon gap features
features['platoon_xwoba_gap'] = features['vsR_xwoba'] - features['vsL_xwoba']
features['platoon_whiff_gap'] = features['vsR_whiff_rate'] - features['vsL_whiff_rate']
features['platoon_ev_gap'] = features['vsR_exit_velo'] - features['vsL_exit_velo']

print(f"  Platoon data matched: {features['vsR_xwoba'].notna().sum()} / {len(features)}")
print(f"  Total columns: {len(features.columns)}")

# ── Step 8: First half / second half splits ──────────────
# Pitchers who fade in the second half may be fatiguing or
# tipping pitches. Trends in velo, whiff rate, and xwoba
# from H1 to H2 are predictive of next-season performance.
print("\nStep 8: Building first/second half split features...")

halves = pd.read_sql(text(f"""
    SELECT
        p.pitcher AS player_id,
        EXTRACT(YEAR FROM g.game_date)::INT AS season,
        CASE WHEN EXTRACT(MONTH FROM g.game_date) <= 6 THEN 'H1' ELSE 'H2' END AS half,
        COUNT(*) AS pitches,
        ROUND(AVG(p.release_speed)::NUMERIC, 1) AS avg_velo,
        ROUND(AVG(CASE WHEN p.launch_speed IS NOT NULL THEN p.launch_speed END)::NUMERIC, 1) AS exit_velo,
        ROUND(AVG(CASE WHEN p.launch_speed IS NOT NULL THEN
            p.estimated_woba_using_speedangle END)::NUMERIC, 3) AS xwoba,
        ROUND(SUM(CASE WHEN p.description IN (
            'swinging_strike', 'swinging_strike_blocked'
        ) THEN 1 ELSE 0 END)::NUMERIC / NULLIF(SUM(CASE WHEN p.description IN (
            'swinging_strike', 'swinging_strike_blocked',
            'foul', 'foul_tip', 'foul_bunt',
            'hit_into_play', 'hit_into_play_no_out', 'hit_into_play_score'
        ) THEN 1 ELSE 0 END), 0), 3) AS whiff_rate
    FROM statcast.pitches p
    JOIN statcast.games g ON p.game_pk = g.game_pk
    WHERE g.game_type = 'R'
      AND EXTRACT(YEAR FROM g.game_date) BETWEEN {FIRST_SEASON} AND {LATEST_SEASON}
    GROUP BY p.pitcher, EXTRACT(YEAR FROM g.game_date),
             CASE WHEN EXTRACT(MONTH FROM g.game_date) <= 6 THEN 'H1' ELSE 'H2' END
"""), engine)

for half in ['H1', 'H2']:
    half_df = halves[halves['half'] == half].copy()
    half_df = half_df.rename(columns={
        'avg_velo': f'velo_{half.lower()}',
        'exit_velo': f'ev_{half.lower()}',
        'xwoba': f'xwoba_{half.lower()}',
        'whiff_rate': f'whiff_{half.lower()}',
    })
    half_df = half_df.drop(columns=['half', 'pitches'])
    features = features.merge(half_df, on=['player_id', 'season'], how='left')

# Trend features: H2 minus H1 (positive = got worse for most stats)
features['velo_trend'] = features['velo_h2'] - features['velo_h1']
features['ev_trend'] = features['ev_h2'] - features['ev_h1']
features['xwoba_trend'] = features['xwoba_h2'] - features['xwoba_h1']
features['whiff_trend'] = features['whiff_h2'] - features['whiff_h1']

print(f"  Total columns: {len(features.columns)}")

# ── Step 9: Age curves and role-based aging ──────────────
# Pitchers peak around 27-29, then decline. Starters age more
# gracefully than relievers. Velocity decline accelerates
# after 30, especially for power arms.
print("\nStep 9: Adding age and role features...")

# Precise age as of July 1 (midseason)
features['age'] = (
    pd.to_datetime(features['season'].astype(str) + '-07-01') - pd.to_datetime(features['birth_date'])
).dt.days / 365.25

# Age curve shape
features['age_from_peak'] = features['age'] - 28
features['age_from_peak_sq'] = features['age_from_peak'] ** 2
features['post_peak'] = (features['age'] > 29).astype(int)

# Role-specific aging interactions
features['age_starter'] = features['age_from_peak'] * features['is_starter']
features['age_reliever'] = features['age_from_peak'] * (1 - features['is_starter'])

# Velocity aging — older pitchers lose velo faster
features['age_velo'] = features['age_from_peak'] * features['fb_velo']

print(f"  Total columns: {len(features.columns)}")

# ── Step 10: Park factors ────────────────────────────────
# For traded players, compute IP-weighted park factor across
# all teams they pitched for that season.
print("\nStep 10: Adding park factors...")

park_factors = pd.read_sql(text("""
    SELECT season, team, basic AS park_factor
    FROM statcast.park_factors
"""), engine)

splits_pf = team_splits.merge(park_factors, on=['team', 'season'], how='left')
weighted_pf = (splits_pf.groupby(['player_id', 'season'])
               .apply(lambda g: pd.Series({
                   'park_factor': np.average(g['park_factor'].dropna(),
                                             weights=g.loc[g['park_factor'].notna(), 'ip'])
                   if g['park_factor'].notna().any() and g.loc[g['park_factor'].notna(), 'ip'].sum() > 0
                   else np.nan,
               }), include_groups=False)
               .reset_index())

features = features.merge(weighted_pf, on=['player_id', 'season'], how='left')
features['era_park_adj'] = features['era'] * (100 / features['park_factor'].replace(0, 100))
features['fip_park_adj'] = features['fip'] * (100 / features['park_factor'].replace(0, 100))

print(f"  Park factors matched: {features['park_factor'].notna().sum()} / {len(features)}")
print(f"  Total columns: {len(features.columns)}")

# ── Step 11: Team strength (defense) ─────────────────────
# IP-weighted team strength for traded players.
print("\nStep 11: Adding team strength...")

team_str = pd.read_sql(text("""
    SELECT team, season, team_runs, team_ops, team_hr
    FROM statcast.team_strength
"""), engine)

splits_ts = team_splits.merge(team_str, on=['team', 'season'], how='left')
weighted_ts = (splits_ts.groupby(['player_id', 'season'])
               .apply(lambda g: pd.Series({
                   col: np.average(g[col].dropna(),
                                   weights=g.loc[g[col].notna(), 'ip'])
                   if g[col].notna().any() and g.loc[g[col].notna(), 'ip'].sum() > 0
                   else np.nan
                   for col in ['team_runs', 'team_ops', 'team_hr']
               }), include_groups=False)
               .reset_index())

features = features.merge(weighted_ts, on=['player_id', 'season'], how='left')

print(f"  Team strength matched: {features['team_runs'].notna().sum()} / {len(features)}")
print(f"  Total columns: {len(features.columns)}")

# ── Step 12: MiLB stats ─────────────────────────────────
# For young pitchers with limited MLB data, minor league stats
# help fill in the picture. A rookie with 30 MLB IP but
# 120 AAA IP has more signal than MLB stats alone suggest.
print("\nStep 12: Adding MiLB stats...")

milb = pd.read_sql(text("""
    SELECT player_id, season, level, ip AS milb_ip,
           era AS milb_era, whip AS milb_whip,
           k_9 AS milb_k_9, bb_9 AS milb_bb_9,
           hr_9 AS milb_hr_9, fip AS milb_fip
    FROM mlb_api.milb_pitching_stats
    WHERE ip >= 10
"""), engine)

# Rank levels: AAA is most relevant, Single-A least
level_rank = {'AAA': 1, 'AA': 2, 'High-A': 3, 'Single-A': 4}
milb['level_rank'] = milb['level'].map(level_rank)

# For each player-season, keep only the highest level played
milb = milb.sort_values(['player_id', 'season', 'level_rank'])
milb = milb.drop_duplicates(subset=['player_id', 'season'], keep='first')
milb = milb.drop(columns=['level_rank', 'level'])

# Derived MiLB rates
milb['milb_k_bb'] = milb['milb_k_9'] - milb['milb_bb_9']

# Join MiLB data from the SAME year or PRIOR year
milb_same = milb.copy()
features = features.merge(milb_same, on=['player_id', 'season'], how='left')

# Also get prior year MiLB stats for players without same-year data
milb_prior = milb.copy()
milb_prior['season'] = milb_prior['season'] + 1
milb_prior = milb_prior.rename(columns={c: f'{c}_prior' for c in milb_prior.columns
                                         if c not in ['player_id', 'season']})
features = features.merge(milb_prior, on=['player_id', 'season'], how='left')

# Fill same-year MiLB gaps with prior-year data
for col in ['milb_ip', 'milb_era', 'milb_whip', 'milb_k_9', 'milb_bb_9',
            'milb_hr_9', 'milb_fip', 'milb_k_bb']:
    features[col] = features[col].fillna(features.get(f'{col}_prior'))

# Drop the prior columns — they've been merged in
prior_cols = [c for c in features.columns if c.endswith('_prior')]
features = features.drop(columns=prior_cols)

print(f"  MiLB data matched: {features['milb_ip'].notna().sum()} / {len(features)}")
print(f"  Total columns: {len(features.columns)}")

# ── Step 13: Save to database ────────────────────────────
print("\nStep 13: Saving to database...")

with engine.begin() as conn:
    conn.execute(text("CREATE SCHEMA IF NOT EXISTS projections"))
    conn.execute(text("DROP TABLE IF EXISTS projections.pitcher_features"))

features.to_sql(
    'pitcher_features', engine, schema='projections',
    if_exists='replace', index=False, method='multi', chunksize=500
)

print(f"  Saved {len(features)} rows to projections.pitcher_features")
print(f"  Columns: {len(features.columns)}")

# Preview train/test split
train = features[features['season'] <= LATEST_SEASON - 1]
test = features[features['season'] == LATEST_SEASON]
print(f"\n  Train seasons: {train['season'].min()} - {train['season'].max()} ({len(train)} rows)")
print(f"  Test season: {test['season'].min()} ({len(test)} rows)")