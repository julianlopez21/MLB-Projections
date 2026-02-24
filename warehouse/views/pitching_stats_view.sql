CREATE VIEW statcast.pitching_stats AS

WITH plate_appearances AS (
    SELECT p.pitcher,
           EXTRACT(year FROM g.game_date)::int AS season,
           g.home_team, g.away_team,
           p.events, p.inning_topbot, p.bb_type
    FROM statcast.pitches p
    JOIN statcast.games g ON g.game_pk = p.game_pk
    WHERE p.events IS NOT NULL
      AND p.events NOT IN ('truncated_pa','ejection','game_advisory')
      AND g.game_type = 'R'
),

counting AS (
    SELECT pitcher,
           CASE WHEN inning_topbot = 'Top' THEN home_team ELSE away_team END AS pitcher_team,
           season,
           COUNT(*) AS tbf,
           COUNT(*) FILTER (WHERE events IN ('single','double','triple','home_run')) AS h,
           COUNT(*) FILTER (WHERE events = 'single') AS "1b",
           COUNT(*) FILTER (WHERE events = 'double') AS "2b",
           COUNT(*) FILTER (WHERE events = 'triple') AS "3b",
           COUNT(*) FILTER (WHERE events = 'home_run') AS hr,
           COUNT(*) FILTER (WHERE events IN ('walk','intent_walk')) AS bb,
           COUNT(*) FILTER (WHERE events = 'intent_walk') AS ibb,
           COUNT(*) FILTER (WHERE events = 'hit_by_pitch') AS hbp,
           COUNT(*) FILTER (WHERE events IN ('strikeout','strikeout_double_play')) AS k,
           COUNT(*) FILTER (WHERE events IN ('sac_fly','sac_fly_double_play')) AS sf,
           COUNT(*) FILTER (WHERE events IN ('sac_bunt','sac_bunt_double_play')) AS sac,
           COUNT(*) FILTER (WHERE bb_type = 'fly_ball') AS fly_balls,
           COUNT(*) FILTER (WHERE events IN ('field_out','strikeout','force_out','sac_fly','sac_bunt',
               'fielders_choice','fielders_choice_out'))
             + 2 * COUNT(*) FILTER (WHERE events IN ('grounded_into_double_play','double_play',
               'strikeout_double_play','sac_fly_double_play','sac_bunt_double_play'))
             + 3 * COUNT(*) FILTER (WHERE events = 'triple_play') AS derived_outs
    FROM plate_appearances
    GROUP BY pitcher,
             CASE WHEN inning_topbot = 'Top' THEN home_team ELSE away_team END,
             season
),

with_adjustment AS (
    SELECT c.*,
           m.outs AS official_outs,
           SUM(c.derived_outs) OVER (PARTITION BY c.pitcher, c.season)::int AS total_derived,
           COUNT(*) OVER (PARTITION BY c.pitcher, c.season)::int AS num_teams,
           ROW_NUMBER() OVER (PARTITION BY c.pitcher, c.season ORDER BY c.pitcher_team)::int AS team_rn
    FROM counting c
    JOIN mlb_api.pitching_stats m ON m.player_id = c.pitcher AND m.season = c.season
),

team_rows AS (
    SELECT pitcher, pitcher_team, 'team'::text AS split_type, season,
           tbf, h, "1b", "2b", "3b", hr, bb, ibb, hbp, k, sf, sac, fly_balls,
           (derived_outs::int + (official_outs - total_derived) / num_teams
             + CASE WHEN team_rn <= (official_outs - total_derived) % num_teams THEN 1 ELSE 0 END) AS outs,
           NULL::int AS wins, NULL::int AS losses, NULL::int AS saves,
           NULL::int AS games, NULL::int AS games_started, NULL::int AS earned_runs
    FROM with_adjustment
),

total_rows AS (
    SELECT c.pitcher, 'TOT'::varchar(3) AS pitcher_team, 'total'::text AS split_type, c.season,
           SUM(c.tbf)::bigint AS tbf,
           SUM(c.h)::bigint AS h,
           SUM(c."1b")::bigint AS "1b",
           SUM(c."2b")::bigint AS "2b",
           SUM(c."3b")::bigint AS "3b",
           SUM(c.hr)::bigint AS hr,
           SUM(c.bb)::bigint AS bb,
           SUM(c.ibb)::bigint AS ibb,
           SUM(c.hbp)::bigint AS hbp,
           SUM(c.k)::bigint AS k,
           SUM(c.sf)::bigint AS sf,
           SUM(c.sac)::bigint AS sac,
           SUM(c.fly_balls)::bigint AS fly_balls,
           m.outs::bigint AS outs,
           m.wins, m.losses, m.saves,
           m.games_played AS games, m.games_started, m.earned_runs
    FROM counting c
    JOIN mlb_api.pitching_stats m ON m.player_id = c.pitcher AND m.season = c.season
    GROUP BY c.pitcher, c.season, m.outs, m.wins, m.losses, m.saves,
             m.games_played, m.games_started, m.earned_runs
),

combined AS (
    SELECT * FROM team_rows
    UNION ALL
    SELECT * FROM total_rows
),

with_rates AS (
    SELECT co.*,
           (co.outs / 3) + (co.outs % 3) * 0.1 AS ip,
           co.outs::numeric / 3.0 AS ip_numeric,
           lc.cfip
    FROM combined co
    JOIN statcast.league_constants lc ON lc.season = co.season
)

SELECT pitcher, pitcher_team, split_type, season,
       wins, losses, saves, games, games_started,
       tbf, ip, outs, h, "1b", "2b", "3b", hr, bb, ibb, hbp, k, sf, sac,
       ROUND(earned_runs::numeric * 9.0 / NULLIF(ip_numeric, 0), 2) AS era,
       ROUND(k::numeric / NULLIF(tbf, 0)::numeric * 100, 1) AS k_pct,
       ROUND(bb::numeric / NULLIF(tbf, 0)::numeric * 100, 1) AS bb_pct,
       ROUND(k::numeric / NULLIF(bb, 0)::numeric, 2) AS k_bb,
       ROUND(h::numeric / NULLIF(tbf - bb - hbp - sf - sac, 0)::numeric, 3) AS ba_against,
       ROUND((h - hr)::numeric / NULLIF(tbf - bb - hbp - sac - k - hr, 0)::numeric, 3) AS babip,
       ROUND((h + bb)::numeric / NULLIF(ip_numeric, 0), 2) AS whip,
       ROUND(k::numeric * 9.0 / NULLIF(ip_numeric, 0), 2) AS k_9,
       ROUND(bb::numeric * 9.0 / NULLIF(ip_numeric, 0), 2) AS bb_9,
       ROUND(hr::numeric * 9.0 / NULLIF(ip_numeric, 0), 2) AS hr_9,
       ROUND(hr::numeric / NULLIF(fly_balls, 0)::numeric, 3) AS hr_fb,
       ROUND((13 * hr + 3 * (bb + hbp) - 2 * k)::numeric / NULLIF(ip_numeric, 0) + cfip, 2) AS fip
FROM with_rates;
