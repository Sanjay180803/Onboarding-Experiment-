"""STEP 3 -- Validate every estimator against the simulator's ground truth.

This is only possible because the data is synthetic, and it's the point:
it shows which methods recover the true causal effect and which naive
comparisons don't. (The analysis in step 2 never reads data/_truth.)
"""
import json

import pandas as pd

from common import DATA, REPORTS, dump_json
from abkit import uplift


def row(name, method, est, lo, hi, truth):
    covered = None if lo is None else bool(lo <= truth <= hi)
    return {"estimand": name, "method": method, "estimate": est, "ci_low": lo, "ci_high": hi,
            "truth": truth, "error": est - truth, "ci_covers_truth": covered}


def main():
    R = json.loads((REPORTS / "results.json").read_text())
    T = json.loads((DATA / "_truth" / "ground_truth.json").read_text())
    te = T["experiment"]
    p, sec = R["primary"], R["secondary"]
    rows = [
        row("ITT activation 7d", "Two-proportion z-test", p["diff"], p["ci_low"], p["ci_high"], te["itt_activation_7d"]),
        row("ITT activation 7d", "CUPAC-adjusted", R["cupac"]["adj_diff"], *R["cupac"]["adj_ci"], te["itt_activation_7d"]),
        row("ITT onboarding completion", "z-test", sec["onboarded_7d"]["diff"], sec["onboarded_7d"]["ci_low"], sec["onboarded_7d"]["ci_high"], te["itt_onboarding_complete"]),
        row("ITT D7 retention", "z-test", sec["d7_retained"]["diff"], sec["d7_retained"]["ci_low"], sec["d7_retained"]["ci_high"], te["itt_d7_retention"]),
        row("ITT D30 retention", "z-test", sec["d30_retained"]["diff"], sec["d30_retained"]["ci_low"], sec["d30_retained"]["ci_high"], te["itt_d30_retention"]),
        row("ITT paid 30d", "z-test", sec["paid_30d"]["diff"], sec["paid_30d"]["ci_low"], sec["paid_30d"]["ci_high"], te["itt_paid_30d"]),
        row("ITT revenue / user 30d", "Bootstrap", sec["revenue_30d"]["diff"], sec["revenue_30d"]["ci_low"], sec["revenue_30d"]["ci_high"], te["itt_revenue_30d"]),
        row("ITT support tickets", "z-test (guardrail)", R["guardrail_support"]["diff"], R["guardrail_support"]["ci_low"], R["guardrail_support"]["ci_high"], te["itt_support_ticket"]),
        row("CACE activation (compliers)", "IV / Wald", R["iv"]["cace"], R["iv"]["ci_low"], R["iv"]["ci_high"], te["cace_activation_7d"]),
        row("CACE activation (compliers)", "Naive: openers vs control", R["iv"]["per_protocol_naive"], None, None, te["cace_activation_7d"]),
        row("CACE activation (compliers)", "Naive: openers vs non-openers", R["iv"]["as_treated_naive"], None, None, te["cace_activation_7d"]),
        row("Community effect on D30", "Naive joiners vs non", R["community"]["naive"], None, None, T["community"]["ate_d30_retention"]),
        row("Community effect on D30", "IPW", R["community"]["ipw"], *R["community"]["ipw_ci"], T["community"]["ate_d30_retention"]),
        row("Community effect on D30", "AIPW (doubly robust)", R["community"]["aipw"], *R["community"]["aipw_ci"], T["community"]["ate_d30_retention"]),
        row("Streak reminders on WAU rate", "Naive post-period comparison", R["did"]["naive_post_comparison"], None, None, T["did"]["att_weekly_active_rate"]),
        row("Streak reminders on WAU rate", "Naive before/after", R["did"]["naive_before_after"], None, None, T["did"]["att_weekly_active_rate"]),
        row("Streak reminders on WAU rate", "DiD, CR1 clustered SE", R["did"]["att"], R["did"]["ci_low"], R["did"]["ci_high"], T["did"]["att_weekly_active_rate"]),
        row("Streak reminders on WAU rate", "DiD, wild cluster bootstrap", R["did_wild_bootstrap"]["att"], R["did_wild_bootstrap"]["ci_low"], R["did_wild_bootstrap"]["ci_high"], T["did"]["att_weekly_active_rate"]),
        row("Streak reminders on WAU rate", "DiD, randomization inference", R["did_randomization"]["att"], R["did_randomization"]["ci_low"], R["did_randomization"]["ci_high"], T["did"]["att_weekly_active_rate"]),
    ]
    df = pd.DataFrame(rows)
    pc = pd.read_parquet(REPORTS / "pred_cate.parquet").merge(pd.read_parquet(DATA / "_truth" / "user_truth.parquet"), on="user_id")
    out = {"table": df.to_dict("records"),
           "uplift_rank_corr_with_true_cate": uplift.spearman(pc["pred_cate"], pc["true_cate_activation"]),
           "true_complier_share": te["complier_share"]}
    dump_json(out, REPORTS / "validation.json")
    with pd.option_context("display.width", 200):
        print(df.round(4).to_string(index=False))
    print("uplift spearman vs true CATE:", round(out["uplift_rank_corr_with_true_cate"], 3))


if __name__ == "__main__":
    main()
