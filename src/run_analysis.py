"""
End-to-end orchestrator:

    python data/generate_data.py     # 1. synthetic data
    python src/etl.py                # 2. load into SQLite
    python src/run_analysis.py       # 3. stats + figures + results.json  <-- this file

Produces:
    reports/results.json          machine-readable readout (used by the README)
    reports/figures/*.png         charts
and prints a human-readable summary.
"""
from __future__ import annotations

import json
import os

from ab_test import run_full_readout
import viz

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORTS = os.path.join(ROOT, "reports")


def main() -> None:
    readout = run_full_readout()

    os.makedirs(os.path.join(REPORTS, "figures"), exist_ok=True)
    viz.retention_curve()
    viz.metric_comparison(readout)
    viz.bayesian_posterior(readout)

    with open(os.path.join(REPORTS, "results.json"), "w", encoding="utf-8") as f:
        json.dump(readout, f, indent=2)

    p = readout["primary_metric"]
    s = readout["secondary_metric"]
    g = readout["guardrail_metric"]
    b = readout["bayesian"]
    pw = readout["power_analysis"]

    print("=" * 64)
    print(" SMART SAVINGS VAULT - A/B TEST READOUT")
    print("=" * 64)
    print(f" Sample: {readout['sample_sizes']}")
    print(f" Powered to detect >= {pw['min_detectable_lift_pp']}pp lift "
          f"(alpha=0.05, power=0.80)")
    print("-" * 64)
    print(f" PRIMARY  D30 retention : {p['control']}% -> {p['treatment']}%  "
          f"({p['abs_lift']:+}pp, {p['rel_lift_pct']:+}%)")
    print(f"          95% CI on lift: [{p['ci95_low']}, {p['ci95_high']}] pp  "
          f"p={p['p_value']}  {'SIG' if p['significant'] else 'ns'}")
    print(f"          Bayesian P(treat>control)={b['prob_treatment_better']:.1%}")
    print(f" SECONDARY weekly txns  : {s['control']} -> {s['treatment']}  "
          f"({s['rel_lift_pct']:+}%)  p={s['p_value']}  "
          f"{'SIG' if s['significant'] else 'ns'}")
    print(f" GUARDRAIL avg txn value: {g['control']} -> {g['treatment']}  "
          f"({g['rel_lift_pct']:+}%)  p={g['p_value']}  "
          f"{'(ok, ns)' if not g['significant'] else '(!! moved)'}")
    print("-" * 64)
    print(f" DECISION: {readout['decision']}")
    print(f" {readout['decision_rationale']}")
    print("=" * 64)
    print(f"Wrote {os.path.join(REPORTS, 'results.json')} and 3 figures.")


if __name__ == "__main__":
    main()
