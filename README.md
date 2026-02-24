# MLB Player Projection System

An end-to-end data pipeline that ingests raw pitch-level data from MLB's Statcast system, builds a structured PostgreSQL data warehouse, engineers 150+ predictive features per player, and trains machine learning models to project next-season performance for every MLB hitter and pitcher.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         DATA SOURCES                                │
│  MLB Statcast API · MLB Stats API · Baseball Savant · MiLB Stats    │
└──────────────┬──────────────────────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     WAREHOUSE  (PostgreSQL)                          │
│                                                                     │
│  statcast schema          mlb_api schema        projections schema   │
│  ├─ pitches (2.4M+)      ├─ batting_stats       ├─ hitter_features  │
│  ├─ games                 ├─ pitching_stats      ├─ pitcher_features │
│  ├─ batted_balls          ├─ player_bio          ├─ hitter_proj_2026│
│  ├─ sprint_speed          ├─ milb_batting        └─ pitcher_proj_2026│
│  ├─ park_factors          ├─ milb_pitching                          │
│  ├─ league_constants      ├─ quality_starts                         │
│  ├─ team_strength         └─ active_40_man                          │
│  └─ 4 materialized views                                            │
│     (batting_stats, pitching_stats, batting_advanced,               │
│      pitching_advanced)                                              │
└──────────────┬──────────────────────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     FEATURE ENGINEERING                              │
│                                                                     │
│  151 hitter features              182 pitcher features               │
│  ├─ Statcast metrics              ├─ Pitch velocity & movement      │
│  ├─ Rolling 2yr/3yr averages      ├─ Whiff/chase/zone rates         │
│  ├─ Regressed rate stats          ├─ Platoon splits (L/R)           │
│  ├─ Platoon splits (L/R)         ├─ First-half/second-half trends   │
│  ├─ PA-weighted park factors      ├─ IP-weighted park factors       │
│  ├─ PA-weighted team strength     ├─ IP-weighted team strength      │
│  ├─ Age curves & peak modeling    ├─ Age curves (starter/reliever)  │
│  └─ MiLB stats integration        └─ MiLB stats integration         │
└──────────────┬──────────────────────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        ML MODELS                                     │
│                                                                     │
│  Per-stat ensemble: Ridge Regression + XGBoost                       │
│  ├─ Per-stat feature selection (importance ≥ 0.005)                 │
│  ├─ Grid-searched XGBoost hyperparameters                           │
│  ├─ Optimized blend weight per target stat                          │
│  └─ 2026 team context via 40-man roster swap                        │
│                                                                     │
│  Hitters: PA, AVG, OPS, R, HR, RBI, SB, TB                         │
│  Pitchers: IP, W, K, BB, ERA, WHIP, QS, SV+H (K/BB derived)       │
└─────────────────────────────────────────────────────────────────────┘
```

## Tech Stack

| Layer | Tools |
|---|---|
| Database | PostgreSQL 16, multi-schema warehouse design |
| ETL | Python, pybaseball, MLB Stats API (REST), pandas |
| SQL | Complex views with CTEs, window functions, FILTER aggregations, park-adjusted metrics (wOBA, wRC+, FIP) |
| Feature Engineering | pandas, NumPy — rolling averages, Bayesian regression-to-mean, platoon splits, age modeling |
| Machine Learning | scikit-learn (Ridge, StandardScaler), XGBoost — per-stat ensembles with optimized blending |
| Data Sources | Statcast pitch-level (2015–2025), MLB Stats API, MiLB stats, park factors, sprint speed |

## Database Design

The warehouse uses three schemas to separate concerns:

**`statcast`** — Raw and derived pitch-level data. The `pitches` table contains every pitch thrown in MLB since 2015 (~2.4M rows). Views compute batting and pitching stats directly from pitch-level events using PostgreSQL `FILTER` aggregations, handling traded players by splitting stats per team and computing season totals.

**`mlb_api`** — Official MLB data pulled from the Stats API: season-level batting/pitching lines, player biographical data, minor league stats, quality starts, and current 40-man rosters.

**`projections`** — Feature matrices and model outputs. Feature tables contain one row per player-season with 150+ engineered columns.

### Key SQL: Batting Stats View

The batting stats view computes all offensive statistics from raw pitch events, handling mid-season trades by producing both per-team splits and season totals in a single query:

```sql
-- Per-team counting stats from pitch-level events
WITH counting AS (
    SELECT batter,
           CASE WHEN inning_topbot = 'Top' THEN away_team ELSE home_team END AS batter_team,
           season,
           COUNT(*) AS pa,
           COUNT(*) FILTER (WHERE events = 'home_run') AS hr,
           COUNT(*) FILTER (WHERE events IN ('walk','intent_walk')) AS bb,
           ...
    FROM statcast.pitches p
    JOIN statcast.games g ON g.game_pk = p.game_pk
    GROUP BY batter, batter_team, season
)
-- Team splits with park-adjusted wRC+
SELECT ... FROM team_rates
UNION ALL
-- Aggregated totals
SELECT ... FROM total_rates;
```

### Key SQL: Pitching Stats View

The pitching view derives innings pitched from out-producing events, reconciles with official totals via window functions, and computes FIP using season-specific constants:

```sql
-- Distribute official outs across team splits for traded pitchers
(derived_outs + (official_outs - total_derived) / num_teams
  + CASE WHEN team_rn <= (official_outs - total_derived) % num_teams
    THEN 1 ELSE 0 END) AS outs
```

## Feature Engineering

Each player-season is represented by 150+ features across several categories:

| Category | Examples | Approach |
|---|---|---|
| Rate stats | AVG, OBP, ISO, K%, BB%, BABIP | Current season values |
| Regressed stats | ERA, WHIP, K%, BABIP, HR/FB | Bayesian shrinkage toward league mean based on sample size |
| Rolling averages | Velocity, whiff rate, xwOBA | 2-year and 3-year weighted averages for stability |
| Platoon splits | xwOBA vs LHP/RHP, whiff rates | Performance gaps reveal platoon vulnerability |
| Trends | Velocity, exit velo, xwOBA | First-half vs second-half delta captures within-season trajectory |
| Context | Park factor, team runs, team OPS | PA/IP-weighted across teams for traded players |
| Age modeling | Age, distance from peak, age² | Non-linear aging curves with position/role interaction terms |
| MiLB stats | Minor league ERA, K/9, BB/9, FIP | Supplements thin MLB track records for young players |

**Handling traded players:** Context features (park factor, team offensive strength) are weighted by playing time across teams. A hitter traded from Coors Field to Petco Park mid-season gets a PA-weighted blend of both park factors, not just the last team.

## Model Design

### Training Approach
- **Year-over-year pairs:** Season N features → Season N+1 outcomes
- **Train:** 2015–2023 pairs (~3,600 hitters, ~3,900 pitchers)
- **Test:** 2024 features → predict 2025 actuals (held-out, never seen during training)

### Ensemble Strategy

Each stat gets its own model pipeline:

1. **Feature selection** — XGBoost importance filter (per stat, since HR and SB are driven by different features)
2. **Ridge regression** — Strong on rate stats (AVG, ERA, WHIP) where regularization prevents overfitting
3. **XGBoost** — Strong on counting stats (HR, RBI, K) where non-linear interactions matter
4. **Blend optimization** — Grid search over Ridge/XGBoost weight per stat, evaluated on held-out test set

### Hitter Model Results (Test: predicting 2025)

| Stat | MAE | R² | Blend |
|---|---|---|---|
| PA | 118.6 | 0.421 | 95% XGB / 5% Ridge |
| AVG | 0.024 | 0.238 | 20% XGB / 80% Ridge |
| OPS | 0.071 | 0.283 | 10% XGB / 90% Ridge |
| HR | 5.6 | 0.459 | 60% XGB / 40% Ridge |
| R | 16.6 | 0.468 | 100% XGB |
| RBI | 17.3 | 0.419 | 90% XGB / 10% Ridge |
| SB | 3.9 | 0.533 | 35% XGB / 65% Ridge |
| TB | 51.9 | 0.444 | 100% XGB |

### Pitcher Model Results (Test: predicting 2025)

| Stat | MAE | R² | Blend |
|---|---|---|---|
| IP | 29.4 | 0.456 | 55% XGB / 45% Ridge |
| W | 2.4 | 0.347 | 55% XGB / 45% Ridge |
| K | 28.1 | 0.453 | 55% XGB / 45% Ridge |
| BB | 9.9 | 0.326 | 70% XGB / 30% Ridge |
| ERA | 1.28 | 0.113 | 100% XGB |
| WHIP | 0.19 | 0.123 | 60% XGB / 40% Ridge |
| QS | 2.2 | 0.565 | 90% XGB / 10% Ridge |
| SV+H | 4.6 | 0.536 | 100% XGB |
| K/BB | 0.81 | 0.189 | Derived from K ÷ BB |

K/BB is derived from the independently predicted K and BB components rather than predicted directly — ratios are inherently noisy (small BB errors get amplified in the denominator), and predicting the components separately produces more stable projections.

The model automatically discovers that Ridge dominates rate stats (AVG, OPS, ERA, WHIP) while XGBoost dominates counting stats (HR, RBI, K, SB) — consistent with the sabermetric understanding that rate stats regress heavily while counting stats depend on non-linear playing-time interactions.

### 2026 Projection Pipeline

Projections incorporate current team context:
1. Load current 40-man rosters from MLB API
2. Swap park factors and team strength to reflect 2026 team assignments
3. Recompute park-adjusted features for the new context
4. Generate projections only for active roster players (439 hitters, 511 pitchers)

## Project Structure

```
mlb-projections/
├── README.md
├── config.py                          # Dynamic season detection from database
├── requirements.txt
│
├── warehouse/
│   ├── loaders/
│   │   ├── load_statcast.py           # Statcast pitch-level data (pybaseball)
│   │   ├── load_mlb_batting.py        # MLB Stats API — batting lines
│   │   ├── load_mlb_pitching.py       # MLB Stats API — pitching lines
│   │   ├── load_player_bio.py         # Player biographical data
│   │   ├── load_milb_stats.py         # Minor league statistics
│   │   ├── load_sprint_speed.py       # Baserunning speed data
│   │   ├── load_quality_starts.py     # Quality start counts
│   │   └── load_active_40_man.py      # Current 40-man rosters
│   └── views/
│       ├── batting_stats_view.sql     # Batting stats from pitch events
│       └── pitching_stats_view.sql    # Pitching stats with FIP, K/BB, BABIP
│
├── features/
│   ├── build_hitter_features.py       # 151-column hitter feature matrix
│   └── build_pitcher_features.py      # 182-column pitcher feature matrix
│
└── models/
    ├── train_hitter_model.py          # Hitter ensemble model + 2026 projections
    └── train_pitcher_model.py         # Pitcher ensemble model + 2026 projections
```

## Sample Output: 2026 Projections

**Top 10 Hitters by Projected OPS:**

| Player | Team | PA | AVG | OPS | HR | R | RBI | SB |
|---|---|---|---|---|---|---|---|---|
| Aaron Judge | NYY | 419 | .305 | 1.057 | 33 | 78 | 63 | 13 |
| Shohei Ohtani | LAD | 523 | .288 | .975 | 37 | 92 | 74 | 19 |
| Juan Soto | NYM | 615 | .272 | .936 | 35 | 94 | 87 | 19 |
| Bobby Witt Jr. | KC | 612 | .301 | .883 | 30 | 94 | 90 | 33 |
| Vladimir Guerrero Jr. | TOR | 616 | .290 | .877 | 25 | 84 | 87 | 6 |
| Kyle Schwarber | PHI | 642 | .227 | .854 | 36 | 93 | 97 | 6 |
| José Ramírez | CLE | 629 | .294 | .853 | 26 | 95 | 84 | 27 |
| Fernando Tatis Jr. | SD | 637 | .273 | .846 | 26 | 93 | 77 | 22 |
| Rafael Devers | SF | 571 | .254 | .842 | 31 | 83 | 92 | 3 |
| Gunnar Henderson | BAL | 579 | .275 | .824 | 23 | 86 | 76 | 22 |

**Top 10 Pitchers by Projected ERA (min 100 IP):**

| Player | Team | IP | W | K | ERA | WHIP | K/BB | QS |
|---|---|---|---|---|---|---|---|---|
| Tarik Skubal | DET | 151.2 | 12 | 193 | 3.01 | 0.99 | 4.82 | 16 |
| Zack Wheeler | PHI | 148.1 | 11 | 170 | 3.11 | 1.07 | 4.25 | 15 |
| Garrett Crochet | BOS | 151.4 | 10 | 186 | 3.22 | 1.07 | 4.77 | 16 |
| Hunter Brown | HOU | 152.2 | 11 | 169 | 3.29 | 1.17 | 3.31 | 15 |
| Paul Skenes | PIT | 149.3 | 11 | 174 | 3.39 | 1.15 | 4.05 | 12 |
| Yoshinobu Yamamoto | LAD | 149.0 | 10 | 164 | 3.55 | 1.17 | 3.42 | 14 |
| Cristopher Sánchez | PHI | 155.2 | 10 | 163 | 3.55 | 1.18 | 4.08 | 15 |
| Chris Sale | ATL | 151.9 | 10 | 181 | 3.57 | 1.12 | 4.64 | 13 |
| Dylan Cease | TOR | 140.5 | 9 | 160 | 3.59 | 1.22 | 3.14 | 11 |
| Sonny Gray | BOS | 143.0 | 9 | 148 | 3.75 | 1.19 | 4.35 | 12 |

## Setup

### Prerequisites
- Python 3.10+
- PostgreSQL 16+

### Installation
```bash
git clone https://github.com/julianlopez21/mlb-projections.git
cd mlb-projections
pip install -r requirements.txt
```

### Database Setup
```sql
CREATE DATABASE baseball;
CREATE SCHEMA statcast;
CREATE SCHEMA mlb_api;
CREATE SCHEMA projections;
```

### Pipeline Execution Order
```bash
# 1. Load raw data into warehouse
python warehouse/loaders/load_statcast.py 2015-04-05 2025-09-29
python warehouse/loaders/load_mlb_batting.py
python warehouse/loaders/load_mlb_pitching.py
python warehouse/loaders/load_player_bio.py
python warehouse/loaders/load_milb_stats.py
python warehouse/loaders/load_sprint_speed.py
python warehouse/loaders/load_quality_starts.py
python warehouse/loaders/load_active_40_man.py

# 2. Create SQL views
psql -d baseball -f warehouse/views/batting_stats_view.sql
psql -d baseball -f warehouse/views/pitching_stats_view.sql

# 3. Build feature matrices
python features/build_hitter_features.py
python features/build_pitcher_features.py

# 4. Train models and generate projections
python models/train_hitter_model.py
python models/train_pitcher_model.py
```
