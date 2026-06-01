"""
Synthetic neobank dataset generator (Revolut-style).

Produces three tables written as CSV files:
  - users:        one row per customer, with the A/B experiment assignment
  - transactions: card / transfer / top-up events over a 90-day window
  - experiment:   metadata describing the feature test

Scenario
--------
We launch a new "Smart Savings Vault" feature (round-up on every card payment
that is swept into a savings vault). It is rolled out to a randomly assigned
50% of newly-onboarded users (treatment) while the rest stay on the old app
(control). We want to know whether the feature increases:
  * 30-day retention (primary metric)
  * weekly active transactions per user (secondary)
without hurting our guardrail metric (avg transaction value).

The generator deliberately seeds a *modest, realistic* treatment effect so the
downstream statistical analysis has something true to recover.

Usage:
    python data/generate_data.py --users 12000 --seed 42
"""
from __future__ import annotations

import argparse
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
COUNTRIES = ["GB", "PL", "PT", "ES", "AE", "DE", "FR", "IE"]
COUNTRY_W = [0.28, 0.16, 0.10, 0.12, 0.06, 0.10, 0.10, 0.08]

PLANS = ["Standard", "Plus", "Premium", "Metal"]
PLAN_W = [0.62, 0.18, 0.14, 0.06]

CHANNELS = ["organic", "paid_social", "referral", "app_store", "influencer"]
CHANNEL_W = [0.34, 0.27, 0.18, 0.13, 0.08]

MERCHANT_CATS = [
    "groceries", "restaurants", "transport", "travel", "shopping",
    "entertainment", "utilities", "atm_withdrawal", "transfer", "top_up",
]
# Base probability of each category per transaction.
CAT_W = [0.20, 0.16, 0.12, 0.06, 0.16, 0.08, 0.07, 0.05, 0.06, 0.04]

WINDOW_DAYS = 90

# --- Seeded experiment effects (the "ground truth" we will try to recover) ---
BASE_D30_RETENTION = 0.42          # control 30-day retention
TREATMENT_RETENTION_LIFT = 0.035   # +3.5pp absolute lift from the feature
BASE_WEEKLY_TXNS = 6.0             # control weekly active transactions (mean)
TREATMENT_TXN_LIFT = 0.08          # +8% relative lift in activity


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def make_users(n: int, rng: np.random.Generator, start: datetime) -> pd.DataFrame:
    user_id = np.arange(1, n + 1)
    # Sign-ups spread across the first 30 days so everyone has >=60 days to be observed.
    signup_offset = rng.integers(0, 30, size=n)
    signup_date = [start + timedelta(days=int(d)) for d in signup_offset]

    country = rng.choice(COUNTRIES, size=n, p=COUNTRY_W)
    plan = rng.choice(PLANS, size=n, p=PLAN_W)
    channel = rng.choice(CHANNELS, size=n, p=CHANNEL_W)
    age = np.clip(rng.normal(31, 8, size=n).round().astype(int), 18, 70)

    # Randomised 50/50 experiment assignment.
    group = rng.choice(["control", "treatment"], size=n, p=[0.5, 0.5])

    return pd.DataFrame(
        {
            "user_id": user_id,
            "signup_date": [d.date().isoformat() for d in signup_date],
            "country": country,
            "plan": plan,
            "acquisition_channel": channel,
            "age": age,
            "experiment_group": group,
            "signup_dt_h": signup_date,  # helper, dropped before save
        }
    )


def make_transactions(users: pd.DataFrame, rng: np.random.Generator,
                      start: datetime) -> pd.DataFrame:
    rows = []
    for u in users.itertuples(index=False):
        signup = u.signup_dt_h
        observable_days = (start + timedelta(days=WINDOW_DAYS) - signup).days

        # --- Retention: does the user stay active past day 30? ---
        p_ret = BASE_D30_RETENTION
        if u.experiment_group == "treatment":
            p_ret += TREATMENT_RETENTION_LIFT
        # Premium tiers and referral users retain better (realistic confounders).
        if u.plan in ("Premium", "Metal"):
            p_ret += 0.10
        if u.acquisition_channel == "referral":
            p_ret += 0.05
        p_ret = min(p_ret, 0.95)
        retained = rng.random() < p_ret

        # --- Weekly activity intensity ---
        weekly = BASE_WEEKLY_TXNS
        if u.experiment_group == "treatment":
            weekly *= (1 + TREATMENT_TXN_LIFT)
        if u.plan in ("Premium", "Metal"):
            weekly *= 1.25
        weekly = max(weekly, 0.5)

        # Active lifespan in days within the window.
        if retained:
            active_days = min(observable_days, int(rng.uniform(35, WINDOW_DAYS)))
        else:
            active_days = min(observable_days, int(rng.uniform(1, 30)))
        active_days = max(active_days, 1)

        lam = weekly / 7.0  # daily transaction rate
        n_txn = rng.poisson(lam * active_days)

        for _ in range(n_txn):
            day = int(rng.integers(0, active_days))
            ts = signup + timedelta(days=day,
                                    hours=int(rng.integers(6, 23)),
                                    minutes=int(rng.integers(0, 60)))
            cat = rng.choice(MERCHANT_CATS, p=CAT_W)
            # Amounts are log-normal; ATM/transfer skew larger.
            base = rng.lognormal(mean=2.9, sigma=0.7)
            if cat in ("atm_withdrawal", "transfer", "travel"):
                base *= rng.uniform(1.5, 4.0)
            amount = round(float(base), 2)
            txn_type = (
                "atm" if cat == "atm_withdrawal"
                else "transfer" if cat == "transfer"
                else "top_up" if cat == "top_up"
                else "card_payment"
            )
            rows.append((u.user_id, ts.isoformat(sep=" ", timespec="minutes"),
                         amount, cat, txn_type))

    return pd.DataFrame(
        rows, columns=["user_id", "ts", "amount", "merchant_category", "txn_type"]
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate synthetic neobank data.")
    ap.add_argument("--users", type=int, default=12000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=os.path.dirname(os.path.abspath(__file__)))
    args = ap.parse_args()

    rng = _rng(args.seed)
    start = datetime(2025, 1, 1)

    users = make_users(args.users, rng, start)
    txns = make_transactions(users, rng, start)

    experiment = pd.DataFrame(
        [{
            "experiment_id": "exp_smart_savings_vault",
            "feature": "Smart Savings Vault (card round-ups)",
            "primary_metric": "d30_retention",
            "secondary_metric": "weekly_active_transactions",
            "guardrail_metric": "avg_transaction_value",
            "start_date": start.date().isoformat(),
            "window_days": WINDOW_DAYS,
            "allocation": "50/50 control/treatment",
        }]
    )

    out = args.out
    users.drop(columns="signup_dt_h").to_csv(os.path.join(out, "users.csv"), index=False)
    txns.to_csv(os.path.join(out, "transactions.csv"), index=False)
    experiment.to_csv(os.path.join(out, "experiment.csv"), index=False)

    print(f"users.csv         -> {len(users):,} rows")
    print(f"transactions.csv  -> {len(txns):,} rows")
    print(f"experiment.csv    -> {len(experiment)} row")
    print(f"output dir        -> {out}")


if __name__ == "__main__":
    main()
