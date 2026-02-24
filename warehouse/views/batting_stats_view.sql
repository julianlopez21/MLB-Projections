CREATE OR REPLACE VIEW statcast.batting_stats AS
WITH plate_appearances AS (
    SELECT p.batter,
        (EXTRACT(year FROM g.game_date))::integer AS season,
        g.home_team,
        g.away_team,
        p.events,
        p.inning_topbot
    FROM statcast.pitches p
    JOIN statcast.games g ON g.game_pk = p.game_pk
    WHERE p.events IS NOT NULL
      AND p.events <> ALL (ARRAY['truncated_pa','ejection','game_advisory'])
      AND g.game_type = 'R'
), counting AS (
    SELECT plate_appearances.batter,
        CASE
            WHEN plate_appearances.inning_topbot = 'Top' THEN plate_appearances.away_team
            ELSE plate_appearances.home_team
        END AS batter_team,
        plate_appearances.season,
        count(*) AS pa,
        count(*) FILTER (WHERE plate_appearances.events <> ALL (ARRAY['walk','intent_walk','hit_by_pitch','sac_fly','sac_fly_double_play','sac_bunt','sac_bunt_double_play','catcher_interf'])) AS ab,
        count(*) FILTER (WHERE plate_appearances.events = ANY (ARRAY['single','double','triple','home_run'])) AS h,
        count(*) FILTER (WHERE plate_appearances.events = 'single') AS "1b",
        count(*) FILTER (WHERE plate_appearances.events = 'double') AS "2b",
        count(*) FILTER (WHERE plate_appearances.events = 'triple') AS "3b",
        count(*) FILTER (WHERE plate_appearances.events = 'home_run') AS hr,
        count(*) FILTER (WHERE plate_appearances.events = ANY (ARRAY['walk','intent_walk'])) AS bb,
        count(*) FILTER (WHERE plate_appearances.events = 'intent_walk') AS ibb,
        count(*) FILTER (WHERE plate_appearances.events = 'hit_by_pitch') AS hbp,
        count(*) FILTER (WHERE plate_appearances.events = ANY (ARRAY['strikeout','strikeout_double_play'])) AS k,
        count(*) FILTER (WHERE plate_appearances.events = ANY (ARRAY['sac_fly','sac_fly_double_play'])) AS sf,
        count(*) FILTER (WHERE plate_appearances.events = ANY (ARRAY['sac_bunt','sac_bunt_double_play'])) AS sac,
        count(*) FILTER (WHERE plate_appearances.events = 'catcher_interf') AS ci,
        count(*) FILTER (WHERE plate_appearances.events = ANY (ARRAY['grounded_into_double_play','double_play','strikeout_double_play','sac_fly_double_play','sac_bunt_double_play','triple_play'])) AS gidp
    FROM plate_appearances
    GROUP BY plate_appearances.batter,
        CASE
            WHEN plate_appearances.inning_topbot = 'Top' THEN plate_appearances.away_team
            ELSE plate_appearances.home_team
        END,
        plate_appearances.season
), team_rates AS (
    SELECT c.*,
        'team'::text AS split_type,
        round(c.h::numeric / NULLIF(c.ab, 0)::numeric, 3) AS avg,
        round((c.h + c.bb + c.hbp)::numeric / NULLIF(c.ab + c.bb + c.hbp + c.sf, 0)::numeric, 3) AS obp,
        round((c."1b" + 2 * c."2b" + 3 * c."3b" + 4 * c.hr)::numeric / NULLIF(c.ab, 0)::numeric, 3) AS slg,
        (lc.wbb * (c.bb - c.ibb)::numeric + lc.whbp * c.hbp::numeric + lc.w1b * c."1b"::numeric + lc.w2b * c."2b"::numeric + lc.w3b * c."3b"::numeric + lc.whr * c.hr::numeric) AS woba_num,
        (c.ab + c.bb - c.ibb + c.sf + c.hbp)::numeric AS woba_denom,
        lc.lg_woba,
        lc.woba_scale,
        lc.lg_r_pa,
        pf.basic AS park_factor
    FROM counting c
    JOIN statcast.league_constants lc ON lc.season = c.season
    LEFT JOIN statcast.park_factors pf ON pf.season = c.season AND pf.team::text = c.batter_team::text
), totals AS (
    SELECT c.batter,
        'TOT'::text AS batter_team,
        'total'::text AS split_type,
        c.season,
        sum(c.pa)::bigint AS pa,
        sum(c.ab)::bigint AS ab,
        sum(c.h)::bigint AS h,
        sum(c."1b")::bigint AS "1b",
        sum(c."2b")::bigint AS "2b",
        sum(c."3b")::bigint AS "3b",
        sum(c.hr)::bigint AS hr,
        sum(c.bb)::bigint AS bb,
        sum(c.ibb)::bigint AS ibb,
        sum(c.hbp)::bigint AS hbp,
        sum(c.k)::bigint AS k,
        sum(c.sf)::bigint AS sf,
        sum(c.sac)::bigint AS sac,
        sum(c.ci)::bigint AS ci,
        sum(c.gidp)::bigint AS gidp
    FROM counting c
    GROUP BY c.batter, c.season
), total_rates AS (
    SELECT t.*,
        round(t.h::numeric / NULLIF(t.ab, 0)::numeric, 3) AS avg,
        round((t.h + t.bb + t.hbp)::numeric / NULLIF(t.ab + t.bb + t.hbp + t.sf, 0)::numeric, 3) AS obp,
        round((t."1b" + 2 * t."2b" + 3 * t."3b" + 4 * t.hr)::numeric / NULLIF(t.ab, 0)::numeric, 3) AS slg,
        (lc.wbb * (t.bb - t.ibb)::numeric + lc.whbp * t.hbp::numeric + lc.w1b * t."1b"::numeric + lc.w2b * t."2b"::numeric + lc.w3b * t."3b"::numeric + lc.whr * t.hr::numeric) AS woba_num,
        (t.ab + t.bb - t.ibb + t.sf + t.hbp)::numeric AS woba_denom,
        lc.lg_woba,
        lc.woba_scale,
        lc.lg_r_pa,
        NULL::numeric AS park_factor
    FROM totals t
    JOIN statcast.league_constants lc ON lc.season = t.season
)
-- Team rows
SELECT batter, batter_team, split_type, season,
    pa, ab, h, "1b", "2b", "3b", hr, bb, ibb, hbp, k, sf, sac, ci, gidp,
    avg, obp, slg,
    round(obp + slg, 3) AS ops,
    round(woba_num / NULLIF(woba_denom, 0), 3) AS woba,
    round(((woba_num / NULLIF(woba_denom, 0) - lg_woba) / woba_scale) * pa::numeric, 1) AS wraa,
    round(((((woba_num / NULLIF(woba_denom, 0) - lg_woba) / woba_scale) + lg_r_pa + (lg_r_pa - (park_factor / 100.0) * lg_r_pa)) / NULLIF(lg_r_pa, 0)) * 100, 0) AS wrc_plus,
    park_factor
FROM team_rates
UNION ALL
-- Total rows
SELECT batter, batter_team, split_type, season,
    pa, ab, h, "1b", "2b", "3b", hr, bb, ibb, hbp, k, sf, sac, ci, gidp,
    avg, obp, slg,
    round(obp + slg, 3) AS ops,
    round(woba_num / NULLIF(woba_denom, 0), 3) AS woba,
    round(((woba_num / NULLIF(woba_denom, 0) - lg_woba) / woba_scale) * pa::numeric, 1) AS wraa,
    NULL::numeric AS wrc_plus,
    park_factor
FROM total_rates;
