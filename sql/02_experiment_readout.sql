-- ============================================================================
-- 02_experiment_readout.sql
-- Per-user experiment metrics + a group-level readout of the A/B test.
--
-- Demonstrates: CTEs, LEFT JOIN to keep zero-activity users, conditional
-- aggregation, window function for revenue ranking, and computing the
-- primary / secondary / guardrail metrics in one pass.
--
-- Metric definitions
--   d30_retention            : made >=1 transaction on day 30..89 after signup
--   weekly_active_txns       : total transactions / weeks observed
--   avg_transaction_value    : guardrail - we must not "buy" activity by
--                              pushing users toward many tiny transactions
--   interchange_revenue_proxy: 0.20% of card spend (a simple revenue stand-in)
-- ============================================================================

WITH per_user AS (
    SELECT
        u.user_id,
        u.experiment_group,
        u.plan,
        u.country,
        COUNT(t.txn_id)                                          AS n_txns,
        COALESCE(SUM(CASE WHEN t.txn_type = 'card_payment'
                          THEN t.amount END), 0)                 AS card_spend,
        COALESCE(AVG(t.amount), 0)                               AS avg_txn_value,
        -- Retained = active in the day 30..89 window.
        MAX(CASE
                WHEN julianday(DATE(t.ts)) - julianday(DATE(u.signup_date))
                     BETWEEN 30 AND 89
                THEN 1 ELSE 0
            END)                                                 AS retained_d30
    FROM users u
    LEFT JOIN transactions t ON t.user_id = u.user_id
    GROUP BY u.user_id, u.experiment_group, u.plan, u.country
),

scored AS (
    SELECT
        *,
        0.002 * card_spend AS interchange_revenue_proxy,
        -- Rank users by revenue within their group (window function demo).
        RANK() OVER (
            PARTITION BY experiment_group
            ORDER BY 0.002 * card_spend DESC
        ) AS revenue_rank_in_group
    FROM per_user
)

SELECT
    experiment_group,
    COUNT(*)                                          AS users,
    ROUND(AVG(retained_d30) * 100, 2)                 AS d30_retention_pct,
    ROUND(AVG(n_txns / 12.85), 2)                     AS weekly_active_txns,  -- 90d ~= 12.85 wks
    ROUND(AVG(avg_txn_value), 2)                      AS avg_transaction_value,
    ROUND(AVG(interchange_revenue_proxy), 2)          AS avg_revenue_per_user,
    ROUND(SUM(interchange_revenue_proxy), 0)          AS total_revenue_proxy
FROM scored
GROUP BY experiment_group
ORDER BY experiment_group;
