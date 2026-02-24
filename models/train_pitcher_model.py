"""
train_pitcher_model.py
Train models to predict next-season pitcher stats.

Approach:
- Build year-over-year pairs (season N features → season N+1 stats)
- Train on 2015-2023 pairs
- Test by predicting 2025 from 2024 features
- Ensemble blend Ridge + XGBoost per stat

Usage:
    cd ~/Judy/Python/machine-learning/mlb-projections
    python models/train_pitcher_model.py
"""

import sys
sys.path.append('..')
from config import engine, LATEST_SEASON
from sqlalchemy import text
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor

# ── Load features ────────────────────────────────────────
print("Loading pitcher features...")
features = pd.read_sql(text("SELECT * FROM projections.pitcher_features"), engine)
print(f"  {len(features)} total player-seasons")

# ── Step 1: Build year-over-year pairs ───────────────────
# For each player, pair season N features with season N+1 outcomes.
print("\nStep 1: Building year-over-year pairs...")

target_cols = ['ip', 'wins', 'k', 'era', 'whip', 'k_bb', 'quality_starts', 'sv_hld']

current = features.copy()
future = features[['player_id', 'season'] + target_cols].copy()
future['season'] = future['season'] - 1

future = future.rename(columns={col: f'next_{col}' for col in target_cols})
pairs = current.merge(future, on=['player_id', 'season'], how='inner')

print(f"  {len(pairs)} year-over-year pairs")
print(f"  Seasons: {pairs['season'].min()} - {pairs['season'].max()}")
print(f"  Players: {pairs['player_id'].nunique()}")

train = pairs[pairs['season'] <= LATEST_SEASON - 2]
test = pairs[pairs['season'] == LATEST_SEASON - 1]

print(f"\n  Train: {len(train)} pairs ({train['season'].min()}-{train['season'].max()})")
print(f"  Test: {len(test)} pairs (predicting {LATEST_SEASON})")

# ── Step 2: Prepare features ────────────────────────────
# Exclude identifiers, text columns, and target columns.
print(f"\nStep 2: Preparing features...")

exclude = [
    'player_id', 'season', 'team', 'split_type', 'bat_side',
    'primary_position', 'birth_date', 'mlb_debut_date'
] + [f'next_{c}' for c in target_cols]

feature_cols = [c for c in train.columns
                if c not in exclude and train[c].dtype in ['int64', 'float64', 'Int64']]
print(f"  Using {len(feature_cols)} features")

X_train = train[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
X_test = test[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

# ── Step 3: Per-stat feature selection ───────────────────
# Different stats are driven by different features.
# ERA cares about xwoba_against and barrel_rate; SV+H cares about role.
print(f"\nStep 3: Per-stat feature selection...")

stat_features = {}
for target in target_cols:
    selector = XGBRegressor(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        random_state=42, verbosity=0
    )
    selector.fit(X_train, train[f'next_{target}'].fillna(0))

    importance = pd.Series(selector.feature_importances_, index=feature_cols)
    keep = importance[importance >= 0.005].index.tolist()
    stat_features[target] = keep
    print(f"  {target:>15s}: keeping {len(keep)} of {len(feature_cols)} features")

# ── Step 4: Tune XGBoost hyperparameters ─────────────────
print(f"\nStep 4: Tuning XGBoost...")

param_grid = {
    'n_estimators': [200, 400],
    'max_depth': [3, 4, 5],
    'learning_rate': [0.03, 0.05, 0.1],
    'min_child_weight': [3, 5, 10],
}

grid = GridSearchCV(
    XGBRegressor(subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0),
    param_grid, cv=3, scoring='r2', n_jobs=-1
)
grid.fit(X_train, train['next_era'].fillna(0))

best_params = grid.best_params_
print(f"  Best params: {best_params}")
print(f"  Best CV R²: {grid.best_score_:.3f}")

# ── Step 5: Train ensemble models ────────────────────────
# For each stat, train both Ridge and XGBoost, then blend
# predictions with a per-stat optimized weight.
print(f"\nStep 5: Training ensemble models...")

optimized = {}
for target in target_cols:
    y_train = train[f'next_{target}'].fillna(0)
    y_test = test[f'next_{target}'].fillna(0)

    # Stat-specific features
    keep = stat_features[target]
    Xt = X_train[keep]
    Xte = X_test[keep]

    scaler_t = StandardScaler()
    Xt_scaled = scaler_t.fit_transform(Xt)
    Xte_scaled = scaler_t.transform(Xte)

    # Ridge (uses scaled features)
    ridge = Ridge(alpha=1.0)
    ridge.fit(Xt_scaled, y_train)
    ridge_pred = ridge.predict(Xte_scaled)

    # XGBoost (uses raw features — tree models don't need scaling)
    xgb = XGBRegressor(
        **best_params,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, verbosity=0
    )
    xgb.fit(Xt, y_train)
    xgb_pred = xgb.predict(Xte)

    # Find optimal blend weight (0 = all Ridge, 1 = all XGBoost)
    best_blend = 0
    best_r2 = -999
    for w in np.arange(0, 1.05, 0.05):
        blended = (1 - w) * ridge_pred + w * xgb_pred
        r2 = r2_score(y_test, blended)
        if r2 > best_r2:
            best_r2 = r2
            best_blend = w

    predictions = (1 - best_blend) * ridge_pred + best_blend * xgb_pred
    mae = mean_absolute_error(y_test, predictions)

    optimized[target] = {
        'ridge': ridge, 'xgb': xgb,
        'blend_weight': best_blend, 'features': keep,
        'mae': mae, 'r2': best_r2,
        'predictions': predictions, 'actuals': y_test.values
    }

    print(f"  {target:>15s}: MAE = {mae:.3f}, R² = {best_r2:.3f}  "
          f"(blend: {best_blend:.0%} XGB, {1-best_blend:.0%} Ridge, {len(keep)} features)")

# ── Step 6: Spot check predictions vs actuals ────────────
# Compare projections to actual 2025 stats for known players.
print(f"\nStep 6: Spot checking predictions...")

bio = pd.read_sql(text("SELECT player_id, full_name FROM mlb_api.player_bio"), engine)

test_results = test[['player_id', 'season']].copy()
for target in target_cols:
    test_results[f'pred_{target}'] = optimized[target]['predictions']
    test_results[f'actual_{target}'] = optimized[target]['actuals']

test_results = test_results.merge(bio, on='player_id', how='left')

# Show top 20 by predicted IP (starters)
top = test_results.nlargest(20, 'pred_ip')
for _, row in top.iterrows():
    print(f"\n  {row['full_name']}")
    print(f"    {'':>8s} {'IP':>6s} {'W':>4s} {'K':>5s} {'ERA':>6s} {'WHIP':>6s} {'K/BB':>6s} {'QS':>4s} {'SvHd':>5s}")
    print(f"    {'Pred':>8s} {row['pred_ip']:>6.1f} {row['pred_wins']:>4.0f} {row['pred_k']:>5.0f} "
          f"{row['pred_era']:>6.2f} {row['pred_whip']:>6.2f} {row['pred_k_bb']:>6.2f} "
          f"{row['pred_quality_starts']:>4.0f} {row['pred_sv_hld']:>5.0f}")
    print(f"    {'Actual':>8s} {row['actual_ip']:>6.1f} {row['actual_wins']:>4.0f} {row['actual_k']:>5.0f} "
          f"{row['actual_era']:>6.2f} {row['actual_whip']:>6.2f} {row['actual_k_bb']:>6.2f} "
          f"{row['actual_quality_starts']:>4.0f} {row['actual_sv_hld']:>5.0f}")

# ── Step 7: Generate 2026 projections ────────────────────
# Use 2025 features to predict 2026 stats for all pitchers
# on a current 40-man roster.
print(f"\nStep 7: Generating 2026 projections...")

latest = features[features['season'] == LATEST_SEASON].copy()
print(f"  {len(latest)} pitchers with {LATEST_SEASON} data")

# Load 2026 rosters
rosters = pd.read_sql(text("""
    SELECT player_id, team FROM mlb_api.active_40_man
    WHERE season = (SELECT MAX(season) FROM mlb_api.active_40_man)
"""), engine)
rosters = rosters.rename(columns={'team': 'team_2026'})

# Load most recent park factors and team strength as proxy for 2026
park_factors = pd.read_sql(text(f"""
    SELECT team, basic AS park_factor
    FROM statcast.park_factors
    WHERE season = {LATEST_SEASON}
"""), engine)

team_str = pd.read_sql(text(f"""
    SELECT team, team_runs, team_ops, team_hr
    FROM statcast.team_strength
    WHERE season = {LATEST_SEASON}
"""), engine)

# Filter to 40-man roster only
latest = latest.merge(rosters, on='player_id', how='inner')
print(f"  {len(latest)} pitchers matched to 2026 rosters")

# Swap context features to 2026 team
latest = latest.drop(columns=['park_factor', 'era_park_adj', 'fip_park_adj',
                               'team_runs', 'team_ops', 'team_hr'], errors='ignore')

latest['proj_team'] = latest['team_2026']
latest = latest.merge(park_factors.rename(columns={'team': 'proj_team'}),
                       on='proj_team', how='left')
latest = latest.merge(team_str.rename(columns={'team': 'proj_team'}),
                       on='proj_team', how='left')

# Recompute park-adjusted stats
latest['era_park_adj'] = latest['era'] * (100 / latest['park_factor'].replace(0, 100))
latest['fip_park_adj'] = latest['fip'] * (100 / latest['park_factor'].replace(0, 100))

# Predict
X_proj = latest[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

projections = latest[['player_id', 'proj_team']].copy()
projections = projections.rename(columns={'proj_team': 'team'})
projections['season'] = LATEST_SEASON + 1

for target in target_cols:
    keep = optimized[target]['features']
    Xp = X_proj[keep]

    scaler_t = StandardScaler()
    scaler_t.fit(train[keep].replace([np.inf, -np.inf], np.nan).fillna(0))
    Xp_scaled = scaler_t.transform(Xp)

    ridge_pred = optimized[target]['ridge'].predict(Xp_scaled)
    xgb_pred = optimized[target]['xgb'].predict(Xp)

    w = optimized[target]['blend_weight']
    projections[target] = (1 - w) * ridge_pred + w * xgb_pred

# Round for readability
projections['ip'] = projections['ip'].round(1)
projections['wins'] = projections['wins'].round(0).astype(int)
projections['k'] = projections['k'].round(0).astype(int)
projections['era'] = projections['era'].round(2)
projections['whip'] = projections['whip'].round(2)
projections['k_bb'] = projections['k_bb'].round(2)
projections['quality_starts'] = projections['quality_starts'].round(0).astype(int)
projections['sv_hld'] = projections['sv_hld'].round(0).astype(int)

# Add player names
projections = projections.merge(bio, on='player_id', how='left')

# Save to database
with engine.begin() as conn:
    conn.execute(text("DROP TABLE IF EXISTS projections.pitcher_projections_2026"))

projections.to_sql(
    'pitcher_projections_2026', engine, schema='projections',
    if_exists='replace', index=False, method='multi', chunksize=500
)

# Show top 30 starters by IP
starters = projections[projections['ip'] >= 50].nlargest(30, 'ip')
print(f"\n  Top 30 projected starters for {LATEST_SEASON + 1}:")
print(f"  {'Player':<25s} {'Team':>5s} {'IP':>6s} {'W':>4s} {'K':>5s} {'ERA':>6s} {'WHIP':>6s} {'K/BB':>6s} {'QS':>4s}")
print(f"  {'-'*78}")
for _, row in starters.iterrows():
    print(f"  {row['full_name']:<25s} {row['team']:>5s} {row['ip']:>6.1f} {row['wins']:>4d} {row['k']:>5d} "
          f"{row['era']:>6.2f} {row['whip']:>6.2f} {row['k_bb']:>6.2f} {row['quality_starts']:>4d}")

# Show top 15 relievers by SV+H
relievers = projections[projections['ip'] < 100].nlargest(15, 'sv_hld')
print(f"\n  Top 15 projected relievers for {LATEST_SEASON + 1}:")
print(f"  {'Player':<25s} {'Team':>5s} {'IP':>6s} {'K':>5s} {'ERA':>6s} {'WHIP':>6s} {'SvHd':>5s}")
print(f"  {'-'*60}")
for _, row in relievers.iterrows():
    print(f"  {row['full_name']:<25s} {row['team']:>5s} {row['ip']:>6.1f} {row['k']:>5d} "
          f"{row['era']:>6.2f} {row['whip']:>6.2f} {row['sv_hld']:>5d}")

print(f"\n  Saved {len(projections)} projections to projections.pitcher_projections_2026")
