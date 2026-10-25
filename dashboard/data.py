"""Data access and pure calculations for the dashboard.

Kept free of Streamlit so it can be unit-tested, and so the dashboard's
calculators are guaranteed to use the same math as the analysis.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
sys.path.insert(0, str(ROOT))

from abkit import power  # noqa: E402

TIER_ORDER = ["primary", "secondary", "guardrail", "validity"]


def load_json(name: str) -> dict:
    return json.loads((REPORTS / name).read_text())


def load_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(REPORTS / name)


def load_catalog() -> dict:
    return yaml.safe_load((ROOT / "metrics" / "catalog.yml").read_text())


def results_node(results: dict, key: str) -> dict:
    node = results
    for part in key.split("."):
        node = node[part]
    return node


def metric_table(results: dict, catalog: dict) -> pd.DataFrame:
    """One row per catalog metric that has an effect estimate (everything except
    the validity check), with absolute and relative effects, CIs and a verdict
    that respects each metric's desired direction."""
    rows = []
    for m in catalog["metrics"]:
        if m["tier"] == "validity":
            continue
        r = results_node(results, m["results_key"])
        c = r["control"]
        lo, hi = r["ci_low"], r["ci_high"]
        significant = lo > 0 or hi < 0
        good_dir = 1 if m["desired_direction"] == "increase" else -1
        if not significant:
            verdict = "no detectable change"
        elif r["diff"] * good_dir > 0:
            verdict = "better"
        else:
            verdict = "worse"
        rows.append({
            "id": m["id"], "metric": m["name"], "tier": m["tier"], "unit": m["unit"],
            "desired": m["desired_direction"], "control": c, "treatment": r["treatment"],
            "diff": r["diff"], "ci_low": lo, "ci_high": hi,
            "rel_lift": r["diff"] / c if c else float("nan"),
            "rel_ci_low": lo / c if c else float("nan"), "rel_ci_high": hi / c if c else float("nan"),
            "p_value": r["p_value"], "p_holm": r.get("p_holm"), "verdict": verdict,
        })
    df = pd.DataFrame(rows)
    df["tier"] = pd.Categorical(df["tier"], TIER_ORDER, ordered=True)
    return df.sort_values("tier", kind="stable").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Calculators
# --------------------------------------------------------------------------- #
@dataclass
class PlanInputs:
    baseline: float
    mde_abs: float
    alpha: float = 0.05
    power: float = 0.80
    daily_signups: float = 1000
    rho: float = 0.0           # correlation of a CUPED covariate with the metric


def plan(p: PlanInputs) -> dict:
    n = power.n_per_arm_proportions(p.baseline, p.mde_abs, p.alpha, p.power)
    n_adj = power.cuped_adjusted_n(n, p.rho) if p.rho else n
    return {"n_per_arm": n, "n_per_arm_adjusted": n_adj,
            "runtime_days": power.runtime_days(n_adj, p.daily_signups),
            "runtime_days_exact": power.runtime_days(n_adj, p.daily_signups, round_to_weeks=False)}


def power_curve(baseline: float, n_per_arm: int, alpha: float, mdes) -> pd.DataFrame:
    return pd.DataFrame({"mde_pp": [m * 100 for m in mdes],
                         "power": [power.power_proportions(baseline, baseline + m, n_per_arm, alpha) for m in mdes]})


def impact(lift_low: float, lift_point: float, lift_high: float, annual_signups: float,
           rollout_share: float, payer_link: float, ltv: float) -> pd.DataFrame:
    """Annualized impact for the CI low / point / CI high activation lift.
    Same arithmetic as analysis/02_analyze.py section I."""
    exposed = annual_signups * rollout_share
    rows = []
    for label, lift in (("CI low", lift_low), ("Point estimate", lift_point), ("CI high", lift_high)):
        act = lift * exposed
        rows.append({"scenario": label, "incremental_activated": act,
                     "incremental_payers": act * payer_link, "revenue": act * payer_link * ltv})
    return pd.DataFrame(rows)
