"""
build_hitter_features.py
Build season-level feature table for hitter projections.

Steps:
    1. Load base data (statcast views + MLB API extras)
    2. Merge into single dataframe
    3. Add player bio data
    4. Engineer derived rate features
    5. Regress small samples toward league average
    6. Build rolling multi-year weighted averages
    7. Add platoon split features (vs LHP / vs RHP)
    8. Add first half / second half splits and trends
    9. Add age curves and position-based aging
   10. Add park factors
   11. Save to database

Usage:
    cd ~/Judy/Python/machine-learning/mlb-projections
    python features/build_hitter_features.py
"""

import sys
sys.path.append("..")
from config import engine, FIRST_SEASON, LATEST_SEASON
from sqlalchemy import text
import pandas as pd
import numpy as np

MIN_PA = 50

# ── Step 1: Load base data ───────────────────────────────
print("Step 1: Loading batting stats...")

traditional = pd.read_sql(text(f"""
    SELECT * FROM statcast.batting_stats
    WHERE pa >= {MIN_PA}
      AND split_type = 'total'
      AND season BETWEEN {FIRST_SEASON} AND {LATEST_SEASON}
"""), engine)
print(f"  {len(traditional)} player-seasons loaded")

# Per-team splits for PA-weighted park factors and team strength
team_splits = pd.read_sql(text(f"""
    SELECT batter AS player_id, season, batter_team AS team, pa
    FROM statcast.batting_stats
    WHERE split_type = 'team'
      AND season BETWEEN {FIRST_SEASON} AND {LATEST_SEASON}
"""), engine)

# Display team string (e.g. "CHC/NYY" for traded players)
team_lookup = (team_splits.sort_values('pa', ascending=False)
               .groupby(['player_id', 'season'])['team']
               .apply(lambda x: '/'.join(x))
               .reset_index()
               .rename(columns={'team': 'team'}))

advanced = pd.read_sql(text(f"""
    SELECT * FROM statcast.batting_advanced
    WHERE season BETWEEN {FIRST_SEASON} AND {LATEST_SEASON}
"""), engine)
print(f"  {len(advanced)} player-seasons with advanced metrics")

# SB, CS, R, RBI, TB are not in the statcast views
mlb_extras = pd.read_sql(text(f"""
    SELECT
        player_id, season,
        stolen_bases AS sb, caught_stealing AS cs,
        runs AS r, rbi, total_bases AS tb
    FROM mlb_api.batting_stats
    WHERE plate_appearances >= {MIN_PA}
      AND season BETWEEN {FIRST_SEASON} AND {LATEST_SEASON}
"""), engine)
print(f"  {len(mlb_extras)} player-seasons with SB/CS/R/RBI/TB")

# ── Step 2: Merge into single dataframe ──────────────────
print("\nStep 2: Merging datasets...")

traditional = traditional.rename(columns={'batter': 'player_id'})
traditional = traditional.drop(columns=['batter_team', 'split_type', 'park_factor', 'wrc_plus'], errors='ignore')

features = traditional.merge(team_lookup, on=['player_id', 'season'], how='left')
features = features.merge(advanced, on=['player_id', 'season'], how='left')
features = features.merge(mlb_extras, on=['player_id', 'season'], how='left')

print(f"  Merged: {len(features)} rows")
print(f"  Rows with advanced data: {features['avg_exit_velo'].notna().sum()}")

# ── Step 3: Add player bio data ──────────────────────────
# Age, experience, handedness, and position all affect projections.
print("\nStep 3: Adding player bio data...")

bio = pd.read_sql(text("""
    SELECT player_id, bat_side, primary_position, birth_date, mlb_debut_date
    FROM mlb_api.player_bio
"""), engine)

features = features.merge(bio, on='player_id', how='left')

features['birth_date'] = pd.to_datetime(features['birth_date'])
features['mlb_debut_date'] = pd.to_datetime(features['mlb_debut_date'])
features['years_experience'] = features['season'] - features['mlb_debut_date'].dt.year
features['years_experience'] = features['years_experience'].clip(lower=0)

print(f"  Bio matched: {features['bat_side'].notna().sum()} / {len(features)}")

# ── Step 4: Engineer derived features ────────────────────
# Combine raw stats into ratios that are more predictive
# than any single stat alone.
print("\nStep 4: Engineering derived features...")

features['bb_rate'] = features['bb'] / features['pa']
features['k_rate'] = features['k'] / features['pa']
features['iso'] = features['slg'] - features['avg']
features['sb_rate'] = features['sb'] / (features['sb'] + features['cs']).replace(0, np.nan)
features['speed_proxy'] = (features['sb'] + features['3b']) / features['pa']
features['hr_per_batted_ball'] = features['hr'] / features['batted_balls'].replace(0, np.nan)
features['xwoba_minus_woba'] = features['xwoba'] - features['woba']
features['avg_minus_xba'] = features['avg'] - features['xba']

print(f"  Total columns: {len(features.columns)}")

# ── Step 5: Regress small samples toward league average ──
# Players with few PA have noisy stats. Blend toward league
# average proportional to sample size.
# Formula: blended = (PA * stat + k * lg_avg) / (PA + k)
# k = PA needed for 50/50 weight with league average
print("\nStep 5: Regressing small samples to mean...")

league_avgs = features.groupby('season').agg({
    'avg': 'mean', 'obp': 'mean', 'slg': 'mean', 'ops': 'mean',
    'woba': 'mean', 'iso': 'mean', 'bb_rate': 'mean', 'k_rate': 'mean',
}).reset_index()
league_avgs = league_avgs.rename(columns={c: f'lg_{c}' for c in league_avgs.columns if c != 'season'})
features = features.merge(league_avgs, on='season', how='left')

reliability = {
    'avg': 500, 'obp': 400, 'slg': 300, 'ops': 350,
    'woba': 350, 'iso': 200, 'bb_rate': 120, 'k_rate': 60
}
for stat, k in reliability.items():
    features[f'{stat}_regressed'] = (
        features['pa'] * features[stat] + k * features[f'lg_{stat}']
    ) / (features['pa'] + k)

features = features.drop(columns=[c for c in features.columns if c.startswith('lg_')])
print(f"  Added {len(reliability)} regressed features. Total columns: {len(features.columns)}")

# ── Step 6: Rolling multi-year weighted averages ─────────
# Smooth out single-season noise with weighted rolling averages.
# Weights: current year 3x, prior year 2x, two years ago 1x.
print("\nStep 6: Building rolling averages...")

rolling_cols = [
    'avg_exit_velo', 'avg_launch_angle', 'xba', 'xwoba',
    'barrel_rate', 'hard_hit_rate', 'swing_rate', 'whiff_rate',
    'chase_rate', 'zone_swing_rate', 'zone_contact_rate',
    'bb_rate', 'k_rate', 'iso', 'speed_proxy', 'hr_per_batted_ball'
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
# Performance vs LHP and RHP separately. Some hitters have
# massive platoon gaps that affect their projections.
print("\nStep 7: Building platoon split features...")

platoon = pd.read_sql(text(f"""
    SELECT
        p.batter AS player_id,
        EXTRACT(YEAR FROM g.game_date)::INT AS season,
        p.p_throws,
        COUNT(*) AS pitches_seen,
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
      AND p.p_throws IN ('L', 'R')
      AND EXTRACT(YEAR FROM g.game_date) BETWEEN {FIRST_SEASON} AND {LATEST_SEASON}
    GROUP BY p.batter, EXTRACT(YEAR FROM g.game_date), p.p_throws
"""), engine)

for split in ['L', 'R']:
    split_df = platoon[platoon['p_throws'] == split].copy()
    split_df = split_df.rename(columns={
        'exit_velo': f'ev_vs_{split}', 'xwoba': f'xwoba_vs_{split}',
        'whiff_rate': f'whiff_vs_{split}', 'pitches_seen': f'pitches_vs_{split}'
    }).drop(columns=['p_throws'])
    features = features.merge(split_df, on=['player_id', 'season'], how='left')

features['xwoba_platoon_gap'] = features['xwoba_vs_L'] - features['xwoba_vs_R']
features['whiff_platoon_gap'] = features['whiff_vs_L'] - features['whiff_vs_R']

print(f"  Total columns: {len(features.columns)}")

# ── Step 8: First half / second half splits ──────────────
# Second half performance is more predictive of next season.
# Trend features capture mid-season adjustments.
print("\nStep 8: Building half splits...")

half_splits = pd.read_sql(text(f"""
    SELECT
        p.batter AS player_id,
        EXTRACT(YEAR FROM g.game_date)::INT AS season,
        CASE WHEN EXTRACT(MONTH FROM g.game_date) >= 7 THEN 'H2' ELSE 'H1' END AS half,
        ROUND(AVG(CASE WHEN p.launch_speed IS NOT NULL THEN p.launch_speed END)::NUMERIC, 1) AS exit_velo,
        ROUND(AVG(CASE WHEN p.launch_speed IS NOT NULL THEN
            p.estimated_woba_using_speedangle END)::NUMERIC, 3) AS xwoba,
        ROUND(AVG(CASE WHEN p.launch_speed IS NOT NULL THEN
            CASE WHEN p.launch_speed >= 98 AND p.launch_angle BETWEEN 26 AND 30 THEN 1
                 WHEN p.launch_speed >= 99 AND p.launch_angle BETWEEN 25 AND 31 THEN 1
                 WHEN p.launch_speed >= 100 AND p.launch_angle BETWEEN 24 AND 33 THEN 1
                 WHEN p.launch_speed >= 101 AND p.launch_angle BETWEEN 23 AND 34 THEN 1
                 WHEN p.launch_speed >= 102 AND p.launch_angle BETWEEN 22 AND 35 THEN 1
                 WHEN p.launch_speed >= 103 AND p.launch_angle BETWEEN 21 AND 36 THEN 1
                 WHEN p.launch_speed >= 104 AND p.launch_angle BETWEEN 20 AND 37 THEN 1
                 WHEN p.launch_speed >= 105 AND p.launch_angle BETWEEN 19 AND 38 THEN 1
                 WHEN p.launch_speed >= 106 AND p.launch_angle BETWEEN 18 AND 39 THEN 1
                 WHEN p.launch_speed >= 107 AND p.launch_angle BETWEEN 17 AND 40 THEN 1
                 WHEN p.launch_speed >= 108 AND p.launch_angle BETWEEN 16 AND 41 THEN 1
                 WHEN p.launch_speed >= 109 AND p.launch_angle BETWEEN 15 AND 42 THEN 1
                 WHEN p.launch_speed >= 110 AND p.launch_angle BETWEEN 14 AND 43 THEN 1
                 ELSE 0 END
            END)::NUMERIC, 3) AS barrel_rate,
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
    GROUP BY p.batter, EXTRACT(YEAR FROM g.game_date),
             CASE WHEN EXTRACT(MONTH FROM g.game_date) >= 7 THEN 'H2' ELSE 'H1' END
"""), engine)

for half in ['H1', 'H2']:
    half_df = half_splits[half_splits['half'] == half].copy()
    half_df = half_df.rename(columns={
        'exit_velo': f'ev_{half.lower()}', 'xwoba': f'xwoba_{half.lower()}',
        'barrel_rate': f'barrel_{half.lower()}', 'whiff_rate': f'whiff_{half.lower()}'
    }).drop(columns=['half'])
    features = features.merge(half_df, on=['player_id', 'season'], how='left')

features['ev_trend'] = features['ev_h2'] - features['ev_h1']
features['xwoba_trend'] = features['xwoba_h2'] - features['xwoba_h1']
features['barrel_trend'] = features['barrel_h2'] - features['barrel_h1']
features['whiff_trend'] = features['whiff_h2'] - features['whiff_h1']

print(f"  Total columns: {len(features.columns)}")

# ── Step 9: Age curves and position-based aging ──────────
# Players peak around 27-28, then decline. The rate of decline
# differs by position — catchers age fastest, corner bats slowest.
print("\nStep 9: Adding age and position features...")

# Precise age as of July 1 (midseason)
features['age'] = (
    pd.to_datetime(features['season'].astype(str) + '-07-01') - pd.to_datetime(features['birth_date'])
).dt.days / 365.25

# Age curve shape
features['age_from_peak'] = features['age'] - 27
features['age_from_peak_sq'] = features['age_from_peak'] ** 2
features['post_peak'] = (features['age'] > 28).astype(int)
features['age_speed'] = features['age_from_peak'] * features['speed_proxy']

# Position groups that age similarly
pos_groups = {
    'C': 'catcher',
    'SS': 'speed_pos', 'CF': 'speed_pos', '2B': 'speed_pos',
    'RF': 'corner', 'LF': 'corner', '3B': 'corner',
    '1B': 'power', 'DH': 'power',
    'OF': 'speed_pos'
}
features['pos_group'] = features['primary_position'].map(pos_groups).fillna('other')

# Position-specific aging interactions
features['is_catcher'] = (features['primary_position'] == 'C').astype(int)
features['age_catcher'] = features['age_from_peak'] * features['is_catcher']
features['is_speed_pos'] = (features['pos_group'] == 'speed_pos').astype(int)
features['age_speed_pos'] = features['age_from_peak'] * features['is_speed_pos']

print(f"  Total columns: {len(features.columns)}")

# ── Step 10: Park factors ────────────────────────────────
# For traded players, compute PA-weighted park factor across
# all teams they played for that season.
print("\nStep 10: Adding park factors...")

park_factors = pd.read_sql(text("""
    SELECT season, team, basic AS park_factor
    FROM statcast.park_factors
"""), engine)

# Drop the NULL park_factor that came from the totals view
features = features.drop(columns=['park_factor'], errors='ignore')

splits_pf = team_splits.merge(park_factors, on=['team', 'season'], how='left')
weighted_pf = (splits_pf.groupby(['player_id', 'season'])
               .apply(lambda g: pd.Series({
                   'park_factor': np.average(g['park_factor'].dropna(),
                                             weights=g.loc[g['park_factor'].notna(), 'pa'])
                   if g['park_factor'].notna().any() and g.loc[g['park_factor'].notna(), 'pa'].sum() > 0
                   else np.nan,
               }), include_groups=False)
               .reset_index())

features = features.merge(weighted_pf, on=['player_id', 'season'], how='left')
features['iso_park_adj'] = features['iso'] * (100 / features['park_factor'].replace(0, 100))

print(f"  Park factors matched: {features['park_factor'].notna().sum()} / {len(features)}")
print(f"  Total columns: {len(features.columns)}")

# ── Step 11: Sprint speed ────────────────────────────────
# Statcast-measured speed in feet/second. Directly predicts
# SB, triples, infield hits, and runs scored. Much better
# than our speed_proxy approximation from SB + triples.
print("\nStep 11: Adding sprint speed...")

sprint = pd.read_sql(text("""
    SELECT player_id, season, sprint_speed, hp_to_1b, bolts, competitive_runs
    FROM statcast.sprint_speed
"""), engine)

features = features.merge(sprint, on=['player_id', 'season'], how='left')

print(f"  Sprint speed matched: {features['sprint_speed'].notna().sum()} / {len(features)}")

# ── Step 12: Lineup position ─────────────────────────────
# Where a hitter bats in the order affects R and RBI.
# A 3-hole hitter gets more RBI opportunities than a 9-hole guy.
# Leadoff hitters score more runs. This is context the model
# needs to project counting stats accurately.
print("\nStep 12: Adding lineup position...")

lineup = pd.read_sql(text("""
    SELECT player_id, season, avg_lineup_pos,
           games AS lineup_games
    FROM statcast.lineup_position
"""), engine)

features = features.merge(lineup, on=['player_id', 'season'], how='left')

# Create indicators for key lineup roles
features['is_leadoff'] = (features['avg_lineup_pos'] <= 1.5).astype(int)
features['is_heart'] = ((features['avg_lineup_pos'] >= 2.5) & (features['avg_lineup_pos'] <= 5.0)).astype(int)
features['is_bottom'] = (features['avg_lineup_pos'] >= 7.0).astype(int)

print(f"  Lineup position matched: {features['avg_lineup_pos'].notna().sum()} / {len(features)}")

# ── Step 13: Team strength ───────────────────────────────
# PA-weighted team strength for traded players.
print("\nStep 13: Adding team strength...")

team_str = pd.read_sql(text("""
    SELECT team, season, team_runs, team_ops, team_hr
    FROM statcast.team_strength
"""), engine)

splits_ts = team_splits.merge(team_str, on=['team', 'season'], how='left')
weighted_ts = (splits_ts.groupby(['player_id', 'season'])
               .apply(lambda g: pd.Series({
                   col: np.average(g[col].dropna(),
                                   weights=g.loc[g[col].notna(), 'pa'])
                   if g[col].notna().any() and g.loc[g[col].notna(), 'pa'].sum() > 0
                   else np.nan
                   for col in ['team_runs', 'team_ops', 'team_hr']
               }), include_groups=False)
               .reset_index())

features = features.merge(weighted_ts, on=['player_id', 'season'], how='left')

print(f"  Team strength matched: {features['team_runs'].notna().sum()} / {len(features)}")

# ── Step 14: MiLB stats ─────────────────────────────────
# For young players with limited MLB data, minor league stats
# help fill in the picture. A rookie with 100 MLB PA but
# 500 AAA PA has more signal than the MLB stats alone suggest.
#
# We aggregate each player's most recent MiLB season into
# summary features, prioritizing the highest level played.
print("\nStep 14: Adding MiLB stats...")

milb = pd.read_sql(text("""
    SELECT player_id, season, level, pa AS milb_pa, ab AS milb_ab,
           avg AS milb_avg, obp AS milb_obp, slg AS milb_slg,
           ops AS milb_ops, hr AS milb_hr, sb AS milb_sb,
           bb AS milb_bb, so AS milb_so
    FROM mlb_api.milb_batting_stats
    WHERE pa >= 50
"""), engine)

# Rank levels: AAA is most relevant, Single-A least
level_rank = {'AAA': 1, 'AA': 2, 'High-A': 3, 'Single-A': 4}
milb['level_rank'] = milb['level'].map(level_rank)

# For each player-season, keep only the highest level played
milb = milb.sort_values(['player_id', 'season', 'level_rank'])
milb = milb.drop_duplicates(subset=['player_id', 'season'], keep='first')
milb = milb.drop(columns=['level_rank'])

# MiLB rates
milb['milb_iso'] = milb['milb_slg'] - milb['milb_avg']
milb['milb_bb_rate'] = milb['milb_bb'] / milb['milb_pa'].replace(0, np.nan)
milb['milb_k_rate'] = milb['milb_so'] / milb['milb_pa'].replace(0, np.nan)
milb['milb_hr_rate'] = milb['milb_hr'] / milb['milb_pa'].replace(0, np.nan)
milb['milb_sb_rate'] = milb['milb_sb'] / milb['milb_pa'].replace(0, np.nan)

# Drop counting stats — rates are more comparable across levels
milb = milb.drop(columns=['milb_ab', 'milb_hr', 'milb_sb', 'milb_bb', 'milb_so', 'level'])

# Join MiLB data from the SAME year or PRIOR year
# (a player might play in minors in April then get called up)
milb_same = milb.copy()
features = features.merge(milb_same, on=['player_id', 'season'], how='left')

# Also get prior year MiLB stats for players without same-year data
milb_prior = milb.copy()
milb_prior['season'] = milb_prior['season'] + 1
milb_prior = milb_prior.rename(columns={c: f'{c}_prior' for c in milb_prior.columns
                                         if c not in ['player_id', 'season']})
features = features.merge(milb_prior, on=['player_id', 'season'], how='left')

# Fill same-year MiLB gaps with prior-year data
for col in ['milb_pa', 'milb_avg', 'milb_obp', 'milb_slg', 'milb_ops',
            'milb_iso', 'milb_bb_rate', 'milb_k_rate', 'milb_hr_rate', 'milb_sb_rate']:
    features[col] = features[col].fillna(features.get(f'{col}_prior'))

# Drop the prior columns — they've been merged in
prior_cols = [c for c in features.columns if c.endswith('_prior')]
features = features.drop(columns=prior_cols)

print(f"  MiLB data matched: {features['milb_pa'].notna().sum()} / {len(features)}")
print(f"  Total columns: {len(features.columns)}")

# At the end of the file, AFTER Step 14 (MiLB stats), replace everything from the preview onward:

# ── Step 15: Save to database ────────────────────────────
print("\nStep 15: Saving to database...")

with engine.begin() as conn:
    conn.execute(text("CREATE SCHEMA IF NOT EXISTS projections"))
    conn.execute(text("DROP TABLE IF EXISTS projections.hitter_features"))

features.to_sql(
    'hitter_features', engine, schema='projections',
    if_exists='replace', index=False, method='multi', chunksize=500
)

print(f"  Saved {len(features)} rows to projections.hitter_features")
print(f"  Columns: {len(features.columns)}")

# Preview train/test split
train = features[features['season'] <= LATEST_SEASON - 1]
test = features[features['season'] == LATEST_SEASON]
print(f"\n  Train seasons: {train['season'].min()} - {train['season'].max()} ({len(train)} rows)")
print(f"  Test season: {test['season'].min()} ({len(test)} rows)")
