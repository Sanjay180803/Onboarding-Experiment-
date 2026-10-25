-- Strict (ordered) conversion funnel by group: a user counts at step k only if
-- they also completed every earlier step. Raw event counts are NOT a funnel --
-- e.g. many users see the paywall without ever activating.
-- {group_col} and {where} are substituted by the runner.
WITH f AS (
    SELECT {group_col} AS grp,
           count(*) AS s1_signup,
           sum(onboarded_7d) AS s2_onboarding_complete,
           sum(onboarded_7d * activated_7d) AS s3_activated,
           sum(onboarded_7d * activated_7d * paywall_7d) AS s4_paywall_view,
           sum(onboarded_7d * activated_7d * paywall_7d * trial_7d) AS s5_trial_start,
           sum(onboarded_7d * activated_7d * paywall_7d * trial_7d * paid_30d) AS s6_paid
    FROM user_metrics
    WHERE {where}
    GROUP BY 1
),
long AS (
    UNPIVOT f ON s1_signup, s2_onboarding_complete, s3_activated, s4_paywall_view, s5_trial_start, s6_paid
    INTO NAME step VALUE users
)
SELECT grp,
       substr(step, 2, 1)::INT AS step_order,
       substr(step, 4) AS step,
       users,
       users / first_value(users) OVER w AS pct_of_signups,
       users / lag(users) OVER w AS step_conversion
FROM long
WINDOW w AS (PARTITION BY grp ORDER BY step)
ORDER BY grp, step_order;
