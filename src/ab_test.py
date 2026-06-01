"""
A/B test statistics for the Smart Savings Vault experiment.

Implements a complete experiment readout the way a data scientist would defend
it in a review:

  * primary metric (retention) ........ two-proportion z-test + 95% CI on the lift
  * secondary metric (activity) ....... Welch's t-test + Mann-Whitney U (robust)
  * guardrail (avg txn value) ......... two-sided check that we did no harm
  * power / MDE ....................... was the experiment adequately powered?
  * Bayesian view ..................... posterior P(treatment > control)

Everything is computed from the SQLite warehouse so it stays consistent with
the SQL readout in ./sql.
"""
from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest, proportion_confint
from statsmodels.stats.power import NormalIndPower

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(ROOT, "data", "warehouse.db")

WEEKS_IN_WINDOW = 90 / 7.0


@dataclass
class MetricResult:
    name: str
    control: float
    treatment: float
    abs_lift: float
    rel_lift_pct: float
    ci95_low: float
    ci95_high: float
    p_value: float
    significant: bool
    test: str

    def as_dict(self) -> dict:
        return asdict(self)


def load_user_metrics(db_path: str = DB_PATH) -> pd.DataFrame:
    """One row per user with the metrics needed for the A/B analysis."""
    con = sqlite3.connect(db_path)
    df = pd.read_sql_query(
        """
        SELECT
            u.user_id,
            u.experiment_group,
            COUNT(t.txn_id)                                  AS n_txns,
            COALESCE(AVG(t.amount), 0)                       AS avg_txn_value,
            MAX(CASE
                    WHEN julianday(DATE(t.ts)) - julianday(DATE(u.signup_date))
                         BETWEEN 30 AND 89
                    THEN 1 ELSE 0
                END)                                         AS retained_d30
        FROM users u
        LEFT JOIN transactions t ON t.user_id = u.user_id
        GROUP BY u.user_id, u.experiment_group
        """,
        con,
    )
    con.close()
    df["weekly_txns"] = df["n_txns"] / WEEKS_IN_WINDOW
    return df


# ---------------------------------------------------------------------------
# Primary metric: retention (proportion)
# ---------------------------------------------------------------------------
def test_retention(df: pd.DataFrame) -> MetricResult:
    c = df[df.experiment_group == "control"]["retained_d30"]
    t = df[df.experiment_group == "treatment"]["retained_d30"]

    succ = np.array([t.sum(), c.sum()])
    nobs = np.array([len(t), len(c)])
    stat, p = proportions_ztest(succ, nobs)

    p_c, p_t = c.mean(), t.mean()
    abs_lift = p_t - p_c

    # 95% CI on the difference of two proportions (Wald).
    se = np.sqrt(p_c * (1 - p_c) / len(c) + p_t * (1 - p_t) / len(t))
    lo, hi = abs_lift - 1.96 * se, abs_lift + 1.96 * se

    return MetricResult(
        name="d30_retention",
        control=round(p_c * 100, 2),
        treatment=round(p_t * 100, 2),
        abs_lift=round(abs_lift * 100, 2),
        rel_lift_pct=round(abs_lift / p_c * 100, 2),
        ci95_low=round(lo * 100, 2),
        ci95_high=round(hi * 100, 2),
        p_value=round(float(p), 5),
        significant=bool(p < 0.05),
        test="two-proportion z-test",
    )


# ---------------------------------------------------------------------------
# Secondary metric: weekly active transactions (continuous)
# ---------------------------------------------------------------------------
def test_activity(df: pd.DataFrame) -> MetricResult:
    c = df[df.experiment_group == "control"]["weekly_txns"]
    t = df[df.experiment_group == "treatment"]["weekly_txns"]

    # Welch's t-test (unequal variance) is the primary read; Mann-Whitney as a
    # distribution-free robustness check on the same hypothesis.
    tstat, p_t_test = stats.ttest_ind(t, c, equal_var=False)
    _, p_mw = stats.mannwhitneyu(t, c, alternative="two-sided")

    abs_lift = t.mean() - c.mean()
    # CI on the difference of means (Welch).
    se = np.sqrt(t.var(ddof=1) / len(t) + c.var(ddof=1) / len(c))
    lo, hi = abs_lift - 1.96 * se, abs_lift + 1.96 * se

    return MetricResult(
        name="weekly_active_transactions",
        control=round(c.mean(), 3),
        treatment=round(t.mean(), 3),
        abs_lift=round(abs_lift, 3),
        rel_lift_pct=round(abs_lift / c.mean() * 100, 2),
        ci95_low=round(lo, 3),
        ci95_high=round(hi, 3),
        p_value=round(float(p_t_test), 5),
        significant=bool(p_t_test < 0.05),
        test=f"Welch t-test (Mann-Whitney p={p_mw:.4f})",
    )


# ---------------------------------------------------------------------------
# Guardrail metric: average transaction value (must not regress)
# ---------------------------------------------------------------------------
def test_guardrail(df: pd.DataFrame) -> MetricResult:
    sub = df[df.n_txns > 0]
    c = sub[sub.experiment_group == "control"]["avg_txn_value"]
    t = sub[sub.experiment_group == "treatment"]["avg_txn_value"]
    tstat, p = stats.ttest_ind(t, c, equal_var=False)
    abs_lift = t.mean() - c.mean()
    se = np.sqrt(t.var(ddof=1) / len(t) + c.var(ddof=1) / len(c))
    return MetricResult(
        name="avg_transaction_value_guardrail",
        control=round(c.mean(), 2),
        treatment=round(t.mean(), 2),
        abs_lift=round(abs_lift, 2),
        rel_lift_pct=round(abs_lift / c.mean() * 100, 2),
        ci95_low=round(abs_lift - 1.96 * se, 2),
        ci95_high=round(abs_lift + 1.96 * se, 2),
        p_value=round(float(p), 5),
        significant=bool(p < 0.05),
        test="Welch t-test (guardrail: expect NON-significant)",
    )


# ---------------------------------------------------------------------------
# Power analysis: what lift could this sample reliably detect?
# ---------------------------------------------------------------------------
def power_analysis(df: pd.DataFrame, baseline: float) -> dict:
    n_per_group = int(df.groupby("experiment_group").size().min())
    # Effect size (Cohen's h) for the minimum detectable retention lift at 80% power.
    analysis = NormalIndPower()
    h = analysis.solve_power(nobs1=n_per_group, alpha=0.05, power=0.80,
                             ratio=1.0, alternative="two-sided")
    # Convert Cohen's h back to an absolute proportion lift around the baseline.
    p1 = baseline
    phi1 = 2 * np.arcsin(np.sqrt(p1))
    p2 = np.sin((h + phi1) / 2) ** 2
    mde_abs = p2 - p1
    return {
        "n_per_group": n_per_group,
        "alpha": 0.05,
        "target_power": 0.80,
        "min_detectable_lift_pp": round(mde_abs * 100, 3),
    }


# ---------------------------------------------------------------------------
# Bayesian view of the primary metric (Beta-Binomial, uniform prior)
# ---------------------------------------------------------------------------
def bayesian_retention(df: pd.DataFrame, n_draws: int = 200_000,
                       seed: int = 7) -> dict:
    rng = np.random.default_rng(seed)
    c = df[df.experiment_group == "control"]["retained_d30"]
    t = df[df.experiment_group == "treatment"]["retained_d30"]
    # Beta(1+successes, 1+failures) posterior under a uniform Beta(1,1) prior.
    post_c = rng.beta(1 + c.sum(), 1 + (len(c) - c.sum()), n_draws)
    post_t = rng.beta(1 + t.sum(), 1 + (len(t) - t.sum()), n_draws)
    diff = post_t - post_c
    return {
        "prob_treatment_better": round(float((diff > 0).mean()), 4),
        "expected_lift_pp": round(float(diff.mean()) * 100, 3),
        "cred_interval_95_pp": [round(float(np.percentile(diff, 2.5)) * 100, 3),
                                round(float(np.percentile(diff, 97.5)) * 100, 3)],
    }


def run_full_readout(db_path: str = DB_PATH) -> dict:
    df = load_user_metrics(db_path)
    retention = test_retention(df)
    activity = test_activity(df)
    guardrail = test_guardrail(df)
    power = power_analysis(df, baseline=retention.control / 100)
    bayes = bayesian_retention(df)

    # Statistical vs PRACTICAL significance. With n~12k, a trivially small
    # guardrail change can be "statistically significant". We only treat the
    # guardrail as breached if the regression is both significant AND larger
    # than a 2% relative practical threshold.
    PRACTICAL_GUARDRAIL_PCT = 2.0
    guardrail_breached = (
        guardrail.significant
        and guardrail.abs_lift < 0
        and abs(guardrail.rel_lift_pct) > PRACTICAL_GUARDRAIL_PCT
    )
    ship = retention.significant and retention.abs_lift > 0 and not guardrail_breached
    decision = "SHIP" if ship else "DO NOT SHIP"

    decision_rationale = (
        f"Primary metric +{retention.abs_lift}pp (p={retention.p_value}, "
        f"95% CI [{retention.ci95_low}, {retention.ci95_high}]). "
        f"Guardrail moved {guardrail.rel_lift_pct:+}% "
        f"(p={guardrail.p_value}) - "
        + ("statistically significant but below the "
           f"{PRACTICAL_GUARDRAIL_PCT}% practical threshold, so not a blocker."
           if guardrail.significant and not guardrail_breached
           else "within noise."
           if not guardrail.significant
           else "a real regression — blocks the launch.")
    )

    return {
        "sample_sizes": df.groupby("experiment_group").size().to_dict(),
        "primary_metric": retention.as_dict(),
        "secondary_metric": activity.as_dict(),
        "guardrail_metric": guardrail.as_dict(),
        "power_analysis": power,
        "bayesian": bayes,
        "decision": decision,
        "decision_rationale": decision_rationale,
    }


if __name__ == "__main__":
    import json
    print(json.dumps(run_full_readout(), indent=2))
