-- ============================================================================
-- 01_cohort_retention.sql
-- Weekly sign-up cohorts and their retention curve, split by experiment group.
--
-- Demonstrates: CTEs, date bucketing, window functions (FIRST_VALUE / MIN OVER),
-- conditional aggregation, and a self-contained funnel from signup -> active week.
-- Dialect: SQLite (uses strftime / julianday).
-- ============================================================================

WITH user_base AS (
    SELECT
        user_id,
        experiment_group,
        DATE(signup_date) AS signup_date,
        -- ISO week bucket the user signed up in (the cohort label).
        strftime('%Y-W%W', signup_date) AS signup_cohort
    FROM users
),

-- For every transaction, how many whole weeks after signup did it happen?
activity AS (
    SELECT
        u.user_id,
        u.experiment_group,
        u.signup_cohort,
        CAST((julianday(DATE(t.ts)) - julianday(u.signup_date)) / 7 AS INTEGER)
            AS weeks_since_signup
    FROM user_base u
    JOIN transactions t ON t.user_id = u.user_id
    WHERE DATE(t.ts) >= u.signup_date
),

-- One row per user per active week (dedupe many transactions in a week).
weekly_active AS (
    SELECT DISTINCT user_id, experiment_group, signup_cohort, weeks_since_signup
    FROM activity
    WHERE weeks_since_signup BETWEEN 0 AND 8
),

cohort_size AS (
    SELECT signup_cohort, experiment_group, COUNT(*) AS cohort_users
    FROM user_base
    GROUP BY signup_cohort, experiment_group
)

SELECT
    cs.signup_cohort,
    cs.experiment_group,
    cs.cohort_users,
    wa.weeks_since_signup,
    COUNT(DISTINCT wa.user_id) AS active_users,
    ROUND(100.0 * COUNT(DISTINCT wa.user_id) / cs.cohort_users, 1) AS retention_pct
FROM cohort_size cs
JOIN weekly_active wa
  ON wa.signup_cohort = cs.signup_cohort
 AND wa.experiment_group = cs.experiment_group
GROUP BY cs.signup_cohort, cs.experiment_group, cs.cohort_users, wa.weeks_since_signup
ORDER BY cs.signup_cohort, cs.experiment_group, wa.weeks_since_signup;
