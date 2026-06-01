"""
Smoke + correctness tests for the A/B analysis.

Run with:  pytest -q
These build a tiny in-memory warehouse so the tests are fast and deterministic
and don't depend on the full generated dataset.
"""
import os
import sqlite3
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import ab_test  # noqa: E402


@pytest.fixture
def tiny_db(tmp_path):
    """A controlled dataset where treatment clearly retains better."""
    db = tmp_path / "w.db"
    con = sqlite3.connect(db)
    con.executescript(
        """CREATE TABLE users(user_id INTEGER, signup_date TEXT,
                              experiment_group TEXT);
           CREATE TABLE transactions(txn_id INTEGER PRIMARY KEY AUTOINCREMENT,
                              user_id INTEGER, ts TEXT, amount REAL,
                              merchant_category TEXT, txn_type TEXT);"""
    )
    rng = np.random.default_rng(0)
    uid = 1
    for grp, p_ret in (("control", 0.40), ("treatment", 0.60)):
        for _ in range(800):
            con.execute("INSERT INTO users VALUES (?,?,?)",
                        (uid, "2025-01-01", grp))
            retained = rng.random() < p_ret
            # an early transaction for everyone
            con.execute("INSERT INTO transactions(user_id,ts,amount,"
                        "merchant_category,txn_type) VALUES (?,?,?,?,?)",
                        (uid, "2025-01-03 10:00", 20.0, "groceries", "card_payment"))
            if retained:  # a day 30..89 transaction
                con.execute("INSERT INTO transactions(user_id,ts,amount,"
                            "merchant_category,txn_type) VALUES (?,?,?,?,?)",
                            (uid, "2025-02-20 10:00", 30.0, "shopping",
                             "card_payment"))
            uid += 1
    con.commit(); con.close()
    return str(db)


def test_load_user_metrics_shape(tiny_db):
    df = ab_test.load_user_metrics(tiny_db)
    assert len(df) == 1600
    assert set(df.experiment_group.unique()) == {"control", "treatment"}
    assert df.retained_d30.isin([0, 1]).all()


def test_retention_detects_seeded_effect(tiny_db):
    df = ab_test.load_user_metrics(tiny_db)
    res = ab_test.test_retention(df)
    assert res.treatment > res.control          # treatment retains better
    assert res.abs_lift > 10                     # ~20pp seeded
    assert res.significant                       # large effect, big n
    assert res.ci95_low < res.abs_lift < res.ci95_high


def test_bayesian_agrees_with_frequentist(tiny_db):
    df = ab_test.load_user_metrics(tiny_db)
    b = ab_test.bayesian_retention(df, n_draws=50_000)
    assert b["prob_treatment_better"] > 0.99
    assert b["expected_lift_pp"] > 10


def test_full_readout_keys(tiny_db):
    out = ab_test.run_full_readout(tiny_db)
    for k in ("primary_metric", "secondary_metric", "guardrail_metric",
              "power_analysis", "bayesian", "decision"):
        assert k in out
