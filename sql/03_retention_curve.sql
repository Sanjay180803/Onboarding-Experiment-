-- N-day retention curve: share of users with a session exactly N days after signup.
WITH days AS (SELECT range AS day_n FROM range(1, 31)),
active AS (
    SELECT DISTINCT e.user_id,
           floor(date_diff('second', u.signup_ts, e.event_ts) / 86400.0)::INT AS day_n
    FROM events e JOIN users u USING (user_id)
    WHERE e.event_name = 'session_start'
),
base AS (SELECT user_id, {group_col} AS grp FROM user_metrics WHERE {where})
SELECT b.grp, d.day_n,
       count(DISTINCT a.user_id)::DOUBLE / (SELECT count(*) FROM base b2 WHERE b2.grp = b.grp) AS retention
FROM base b
CROSS JOIN days d
LEFT JOIN active a ON a.user_id = b.user_id AND a.day_n = d.day_n
GROUP BY 1, 2
ORDER BY 1, 2;
