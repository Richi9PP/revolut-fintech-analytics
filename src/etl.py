"""
ETL: load the generated CSVs into a local SQLite warehouse.

SQLite is used so the SQL in ./sql is real and runnable by anyone cloning the
repo, with zero infrastructure. The schema mirrors how an analytics warehouse
(BigQuery / Snowflake) would model these events.

Usage:
    python src/etl.py
"""
from __future__ import annotations

import os
import sqlite3

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
DB_PATH = os.path.join(ROOT, "data", "warehouse.db")

SCHEMA = """
DROP TABLE IF EXISTS users;
DROP TABLE IF EXISTS transactions;
DROP TABLE IF EXISTS experiment;

CREATE TABLE users (
    user_id              INTEGER PRIMARY KEY,
    signup_date          TEXT NOT NULL,
    country              TEXT,
    plan                 TEXT,
    acquisition_channel  TEXT,
    age                  INTEGER,
    experiment_group     TEXT CHECK (experiment_group IN ('control','treatment'))
);

CREATE TABLE transactions (
    txn_id            INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id           INTEGER NOT NULL REFERENCES users(user_id),
    ts                TEXT NOT NULL,
    amount            REAL NOT NULL,
    merchant_category TEXT,
    txn_type          TEXT
);

CREATE TABLE experiment (
    experiment_id     TEXT PRIMARY KEY,
    feature           TEXT,
    primary_metric    TEXT,
    secondary_metric  TEXT,
    guardrail_metric  TEXT,
    start_date        TEXT,
    window_days       INTEGER,
    allocation        TEXT
);

CREATE INDEX idx_txn_user ON transactions(user_id);
CREATE INDEX idx_txn_ts   ON transactions(ts);
CREATE INDEX idx_users_grp ON users(experiment_group);
"""


def build() -> None:
    if not os.path.exists(os.path.join(DATA_DIR, "users.csv")):
        raise SystemExit("Missing data CSVs. Run `python data/generate_data.py` first.")

    con = sqlite3.connect(DB_PATH)
    con.executescript(SCHEMA)

    pd.read_csv(os.path.join(DATA_DIR, "users.csv")).to_sql(
        "users", con, if_exists="append", index=False)
    pd.read_csv(os.path.join(DATA_DIR, "transactions.csv")).to_sql(
        "transactions", con, if_exists="append", index=False)
    pd.read_csv(os.path.join(DATA_DIR, "experiment.csv")).to_sql(
        "experiment", con, if_exists="append", index=False)

    con.commit()
    counts = {
        t: con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        for t in ("users", "transactions", "experiment")
    }
    con.close()
    print(f"Built warehouse at {DB_PATH}")
    for t, n in counts.items():
        print(f"  {t:<13} {n:>10,} rows")


if __name__ == "__main__":
    build()
