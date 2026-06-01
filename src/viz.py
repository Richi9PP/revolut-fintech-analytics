"""Charts for the experiment readout. Saves PNGs into reports/figures/."""
from __future__ import annotations

import os
import sqlite3

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(ROOT, "data", "warehouse.db")
FIG_DIR = os.path.join(ROOT, "reports", "figures")

# Revolut-ish palette.
C_CONTROL = "#9aa0a6"
C_TREAT = "#2962ff"
plt.rcParams.update({
    "figure.dpi": 120,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 11,
})


def _con():
    return sqlite3.connect(DB_PATH)


def retention_curve() -> str:
    """Pooled weekly retention curve by experiment group."""
    q = """
    WITH ub AS (
        SELECT user_id, experiment_group, DATE(signup_date) sd FROM users
    ),
    wk AS (
        SELECT DISTINCT u.user_id, u.experiment_group,
            CAST((julianday(DATE(t.ts)) - julianday(u.sd))/7 AS INT) w
        FROM ub u JOIN transactions t ON t.user_id=u.user_id
        WHERE DATE(t.ts) >= u.sd
    ),
    sz AS (SELECT experiment_group, COUNT(*) n FROM ub GROUP BY experiment_group)
    SELECT wk.experiment_group, wk.w weeks,
           100.0*COUNT(DISTINCT wk.user_id)/sz.n pct
    FROM wk JOIN sz ON sz.experiment_group=wk.experiment_group
    WHERE wk.w BETWEEN 0 AND 8
    GROUP BY wk.experiment_group, wk.w, sz.n
    ORDER BY wk.experiment_group, wk.w;
    """
    df = pd.read_sql_query(q, _con())
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for grp, color in (("control", C_CONTROL), ("treatment", C_TREAT)):
        d = df[df.experiment_group == grp]
        ax.plot(d.weeks, d.pct, marker="o", color=color, label=grp.title(), lw=2.2)
    ax.set_title("Weekly retention curve by experiment group", fontweight="bold")
    ax.set_xlabel("Weeks since signup")
    ax.set_ylabel("Active users (%)")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    path = os.path.join(FIG_DIR, "retention_curve.png")
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    return path


def metric_comparison(readout: dict) -> str:
    """Bar chart of the three headline metrics, control vs treatment."""
    metrics = [
        ("D30 retention (%)", readout["primary_metric"]),
        ("Weekly txns", readout["secondary_metric"]),
        ("Avg txn value", readout["guardrail_metric"]),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(11, 4.1))
    for ax, (title, m) in zip(axes, metrics):
        vals = [m["control"], m["treatment"]]
        ax.bar(["Control", "Treatment"], vals, color=[C_CONTROL, C_TREAT])
        ax.set_title(title, fontsize=11, fontweight="bold", pad=26)
        # Headroom so labels never collide with bars.
        ax.set_ylim(0, max(vals) * 1.30)
        sig = "✓ p<0.05" if m["significant"] else "ns"
        # Lift annotation sits just under the title, above the bars.
        ax.text(0.5, 1.02, f'{m["rel_lift_pct"]:+.1f}%  ({sig})',
                transform=ax.transAxes, ha="center", fontsize=10,
                color=C_TREAT if m["significant"] else "#888")
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:g}", ha="center", va="bottom", fontsize=9)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Experiment readout: Smart Savings Vault", fontweight="bold",
                 y=1.02)
    path = os.path.join(FIG_DIR, "metric_comparison.png")
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    return path


def bayesian_posterior(readout: dict) -> str:
    """Visualise the Beta posteriors of the primary metric."""
    from scipy.stats import beta as beta_dist
    con = _con()
    g = pd.read_sql_query(
        """SELECT experiment_group,
                  SUM(CASE WHEN r=1 THEN 1 ELSE 0 END) s, COUNT(*) n FROM (
             SELECT u.experiment_group,
               MAX(CASE WHEN julianday(DATE(t.ts))-julianday(DATE(u.signup_date))
                        BETWEEN 30 AND 89 THEN 1 ELSE 0 END) r
             FROM users u LEFT JOIN transactions t ON t.user_id=u.user_id
             GROUP BY u.user_id, u.experiment_group)
           GROUP BY experiment_group""", con)
    con.close()
    x = np.linspace(0.30, 0.60, 600)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for _, row in g.iterrows():
        color = C_TREAT if row.experiment_group == "treatment" else C_CONTROL
        y = beta_dist.pdf(x, 1 + row.s, 1 + (row.n - row.s))
        ax.plot(x * 100, y, color=color, lw=2.2, label=row.experiment_group.title())
        ax.fill_between(x * 100, y, alpha=0.12, color=color)
    prob = readout["bayesian"]["prob_treatment_better"]
    ax.set_title(f"Posterior of D30 retention — P(treatment>control)={prob:.1%}",
                 fontweight="bold")
    ax.set_xlabel("D30 retention (%)"); ax.set_ylabel("density")
    ax.legend(frameon=False); ax.grid(axis="y", alpha=0.25)
    path = os.path.join(FIG_DIR, "bayesian_posterior.png")
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    return path
