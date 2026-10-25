-- One row per user with every metric used in the design and analysis.
-- Windows are measured from signup, so every user gets the same exposure time.
-- Inputs (DuckDB views): users, events, assignments
CREATE OR REPLACE TABLE user_metrics AS
WITH ev AS (
    SELECT e.user_id,
           e.event_name,
           e.revenue,
           date_diff('second', u.signup_ts, e.event_ts) / 86400.0 AS days_since_signup,
           floor(date_diff('second', u.signup_ts, e.event_ts) / 86400.0)::INT AS day_n
    FROM events e
    JOIN users u USING (user_id)
),
agg AS (
    SELECT user_id,
        max(event_name = 'onboarding_complete' AND days_since_signup < 7)::INT AS onboarded_7d,
        max(event_name = 'core_action'         AND days_since_signup < 7)::INT AS activated_7d,
        max(event_name = 'paywall_view'        AND days_since_signup < 8)::INT AS paywall_7d,
        max(event_name = 'trial_start'         AND days_since_signup < 8)::INT AS trial_7d,
        max(event_name = 'purchase'            AND days_since_signup < 30)::INT AS paid_30d,
        sum(CASE WHEN event_name = 'purchase'  AND days_since_signup < 30 THEN revenue ELSE 0 END) AS revenue_30d,
        max(event_name = 'session_start' AND day_n = 1)::INT  AS d1_retained,
        max(event_name = 'session_start' AND day_n = 7)::INT  AS d7_retained,
        max(event_name = 'session_start' AND day_n = 30)::INT AS d30_retained,
        sum((event_name = 'session_start' AND day_n BETWEEN 1 AND 30)::INT) AS sessions_30d,
        count(DISTINCT CASE WHEN event_name = 'session_start' AND day_n BETWEEN 1 AND 30 THEN day_n END) AS active_days_30d,
        max(event_name = 'support_ticket'      AND days_since_signup < 7)::INT AS support_ticket_7d,
        max(event_name = 'checklist_open')::INT AS opened_checklist,
        max(event_name = 'community_join')::INT AS joined_community
    FROM ev
    GROUP BY user_id
)
SELECT u.user_id, u.signup_ts, u.signup_day, u.cohort, u.platform, u.channel, u.goal,
       u.pre_visits, u.market,
       coalesce(a.variant, 'none') AS variant,
       (a.variant = 'treatment')::INT AS treat,
       agg.* EXCLUDE (user_id)
FROM users u
LEFT JOIN assignments a USING (user_id)
LEFT JOIN agg USING (user_id)
ORDER BY u.user_id;  -- deterministic row order => reproducible cross-fitting folds
