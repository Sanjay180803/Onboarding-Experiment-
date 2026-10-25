-- Weekly signup cohorts x D1/D7/D30 retention (historical cohort): the classic
-- cohort table used to establish baselines before designing the test.
SELECT date_trunc('week', signup_ts)::DATE AS signup_week,
       count(*) AS users,
       avg(activated_7d)  AS activation_7d,
       avg(d1_retained)   AS d1,
       avg(d7_retained)   AS d7,
       avg(d30_retained)  AS d30,
       avg(paid_30d)      AS paid_30d
FROM user_metrics
WHERE cohort = 'historical'
GROUP BY 1
ORDER BY 1;
