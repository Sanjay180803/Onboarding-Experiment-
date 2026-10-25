"""STEP 2 -- Analyze the experiment and the two observational questions,
then translate results into a business recommendation.

Sections
  A. Trust checks (SRM, guardrails)
  B. Primary metric: z-test, CI, CUPED, CUPAC
  C. Secondary metrics: proportions, bootstrap (revenue), delta method (ratio), Holm
  D. Sequential testing: the cost of peeking, mSPRT on the real test
  E. Noncompliance: ITT vs CACE (IV)
  F. Observational: does joining the community cause retention? (IPW / AIPW)
  G. Non-randomized rollout: streak reminders (DiD + event study)
  H. Heterogeneous effects: uplift modeling
  I. Business impact
"""
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

from common import (DATA, REPORTS, ROOT, connect, run_sql, style, save, dump_json,
                    C_CONTROL, C_TREAT, C_THIRD, INK, INK2, MUTED, VARIANT_COLOR)
from abkit import inference as inf, variance_reduction as vr, sequential as seq, causal, uplift

FEATURES = ["platform", "channel", "goal", "pre_visits"]
LTV_PER_PAYER = 65.0   # ASSUMPTION: first-year revenue per new payer -- replace with finance's number
COST_PER_TICKET = 6.0  # ASSUMPTION: fully loaded cost of one support ticket
ROLLOUT_SHARE = 0.95   # ship to 95% of new users, keep a 5% long-term control holdout


def main():
    style()
    design = json.loads((REPORTS / "design.json").read_text())
    con = connect()
    um = con.sql("SELECT * FROM user_metrics ORDER BY user_id").df()
    exp = um[um.cohort == "experiment"].copy()
    hist = um[um.cohort == "historical"].copy()
    t = exp["treat"].to_numpy(bool)
    R = {"design": design}

    # ---- A. trust checks ---------------------------------------------------
    R["srm"] = inf.srm_test([(~t).sum(), t.sum()])

    # ---- B. primary metric -------------------------------------------------
    y = exp["activated_7d"].to_numpy()
    R["primary"] = inf.two_proportion_test(y[t].sum(), t.sum(), y[~t].sum(), (~t).sum())
    R["cuped"] = vr.cuped(y, exp["pre_visits"], t)
    model, cols = vr.fit_cupac_model(hist, FEATURES, "activated_7d")
    R["cupac"] = vr.cuped(y, vr.cupac_covariate(model, cols, exp, FEATURES), t)

    # ---- C. secondary metrics + guardrail ---------------------------------
    sec = {}
    for m in ["onboarded_7d", "d1_retained", "d7_retained", "d30_retained", "paid_30d"]:
        v = exp[m].to_numpy()
        sec[m] = inf.two_proportion_test(v[t].sum(), t.sum(), v[~t].sum(), (~t).sum())
    rev = exp["revenue_30d"].to_numpy()
    sec["revenue_30d"] = {**inf.bootstrap_diff(rev[t], rev[~t]),
                          "control": rev[~t].mean(), "treatment": rev[t].mean()}
    sec["revenue_30d"]["rel_lift"] = sec["revenue_30d"]["diff"] / rev[~t].mean()
    sec["sessions_per_active_day"] = inf.delta_ratio_test(
        exp.loc[t, "sessions_30d"], exp.loc[t, "active_days_30d"],
        exp.loc[~t, "sessions_30d"], exp.loc[~t, "active_days_30d"])
    holm = inf.adjust_pvalues({k: v["p_value"] for k, v in sec.items()}, "holm")
    for k in sec:
        sec[k]["p_holm"] = holm[k]
    g = exp["support_ticket_7d"].to_numpy()
    R["guardrail_support"] = inf.two_proportion_test(g[t].sum(), t.sum(), g[~t].sum(), (~t).sum())
    R["secondary"] = sec

    # ---- D. sequential --------------------------------------------------------
    p0 = design["baseline_activation_7d"]
    R["peeking_sim_aa"] = seq.simulate_peeking(p=p0, n_per_day_per_arm=500, days=14, sims=5000, seed=1)
    R["peeking_sim_ab"] = seq.simulate_peeking(p=p0, n_per_day_per_arm=500, days=14, sims=3000,
                                               effect=0.025, seed=2)
    daily = exp.groupby(["signup_day", "treat"])["activated_7d"].agg(["sum", "count"]).unstack()
    cs = daily.cumsum()
    pt_, pc_ = cs[("sum", 1)] / cs[("count", 1)], cs[("sum", 0)] / cs[("count", 0)]
    var = pt_ * (1 - pt_) / cs[("count", 1)] + pc_ * (1 - pc_) / cs[("count", 0)]
    av_p = seq.always_valid_pvalues((pt_ - pc_).to_numpy(), var.to_numpy(), 0.02 ** 2)
    crossed = np.flatnonzero(av_p < 0.05)
    R["msprt_on_experiment"] = {"always_valid_p_by_day": av_p.tolist(),
                                "first_day_significant": int(crossed[0]) + 1 if len(crossed) else None}

    # ---- E. noncompliance -------------------------------------------------
    R["iv"] = causal.iv_cace(y, t, exp["opened_checklist"])

    # ---- F. community: observational --------------------------------------
    comm = causal.ipw_aipw(hist, "d30_retained", "joined_community", FEATURES, n_boot=200)
    base = hist.loc[hist.joined_community == 0, "d30_retained"].mean()
    rr = (base + comm["aipw"]) / base
    R["community"] = {k: comm[k] for k in ("naive", "ipw", "aipw", "ipw_ci", "aipw_ci")}
    R["community"].update({"baseline_nonjoiners": base, "risk_ratio": rr, "e_value": causal.e_value(rr),
                           "max_abs_smd_before": float(comm["balance_before"].abs().max()),
                           "max_abs_smd_after": float(comm["balance_after"].abs().max())})

    # ---- G. DiD ----------------------------------------------------------------
    panel = pd.read_parquet(DATA / "market_weeks.parquet")
    panel["wau_rate"] = panel["active_users"] / panel["users"]
    R["did"] = causal.did(panel)
    es = causal.event_study(panel, rollout_time=12)
    # Only 12 market clusters: CR1 SEs can be unreliable. Sensitivity checks,
    # both exact by enumeration, plus a null Monte Carlo of how each method behaves.
    R["did_wild_bootstrap"] = causal.did_wild_cluster_bootstrap(panel)
    ri = causal.did_randomization_inference(panel)
    R["did_randomization"] = {k: v for k, v in ri.items() if k != "placebo_estimates"}
    R["did_size_sim"] = causal.simulate_did_size(sims=300)

    # ---- H. uplift -------------------------------------------------------------
    cate = uplift.t_learner_cate(exp[FEATURES], y, t)
    exp["pred_cate"] = cate
    exp[["user_id", "pred_cate"]].to_parquet(REPORTS / "pred_cate.parquet", index=False)
    deciles = uplift.uplift_by_quantile(cate, y, t, q=5)
    R["uplift"] = {"qini": uplift.qini_coefficient(cate, y, t), "quintiles": deciles.to_dict("records")}
    seg = []
    for col in ["goal", "platform", "channel"]:
        for lvl, gdf in exp.groupby(col):
            tt = gdf.treat.astype(bool)
            r = inf.two_proportion_test(gdf.activated_7d[tt].sum(), tt.sum(), gdf.activated_7d[~tt].sum(), (~tt).sum())
            seg.append({"segment": f"{col}={lvl}", "n": len(gdf), "control": r["control"], "diff": r["diff"],
                        "ci_low": r["ci_low"], "ci_high": r["ci_high"]})
    lo = exp.pre_visits <= 1
    for name, mask in (("pre_visits<=1 (low intent)", lo), ("pre_visits>=2", ~lo)):
        gdf = exp[mask]; tt = gdf.treat.astype(bool)
        r = inf.two_proportion_test(gdf.activated_7d[tt].sum(), tt.sum(), gdf.activated_7d[~tt].sum(), (~tt).sum())
        seg.append({"segment": name, "n": len(gdf), "control": r["control"], "diff": r["diff"],
                    "ci_low": r["ci_low"], "ci_high": r["ci_high"]})
    R["segments"] = seg

    # ---- I. business impact -------------------------------------------------
    annual_signups = design["daily_signups"] * 365
    # Recommended rollout: 95% treatment, 5% long-term control holdout, so only
    # 95% of signups receive the checklist and generate incremental impact.
    exposed_signups = annual_signups * ROLLOUT_SHARE
    # Activation -> payer link. The raw gap (paid rate of activated minus
    # non-activated users) is confounded: motivated users both activate and pay.
    # Covariate adjustment removes the observed part; unobserved motivation
    # remains, so treat the adjusted link as an UPPER bound and show a
    # pessimistic scenario at half of it.
    raw_gap = hist.loc[hist.activated_7d == 1, "paid_30d"].mean() - hist.loc[hist.activated_7d == 0, "paid_30d"].mean()
    lpm = smf.ols("paid_30d ~ activated_7d + C(platform) + C(channel) + C(goal) + pre_visits", hist).fit(cov_type="HC1")
    link_hi = float(lpm.params["activated_7d"])
    links = {"optimistic": link_hi, "pessimistic": link_hi / 2}
    prim = R["primary"]
    grid = {}
    for lk, link in links.items():
        for k, lift in (("low", prim["ci_low"]), ("point", prim["diff"]), ("high", prim["ci_high"])):
            act = lift * exposed_signups
            grid[f"{lk}_{k}"] = {"incremental_activated": act, "incremental_payers": act * link,
                                 "revenue": act * link * LTV_PER_PAYER}
    tickets = R["guardrail_support"]["diff"] * exposed_signups
    R["impact"] = {"annual_signups": annual_signups, "rollout_share": ROLLOUT_SHARE,
                   "holdout_share": 1 - ROLLOUT_SHARE, "exposed_signups": exposed_signups,
                   "raw_activation_paid_gap": raw_gap,
                   "adjusted_link_upper": link_hi, "links": links,
                   "ltv_per_payer_assumption": LTV_PER_PAYER, "cost_per_ticket_assumption": COST_PER_TICKET,
                   "scenarios": grid,
                   "extra_tickets_per_year": tickets, "extra_ticket_cost": tickets * COST_PER_TICKET,
                   "direct_paid_itt": sec["paid_30d"]["diff"],
                   "direct_paid_ci": [sec["paid_30d"]["ci_low"], sec["paid_30d"]["ci_high"]],
                   "itt_if_80pct_open": R["iv"]["cace"] * 0.8,
                   "current_open_rate": R["iv"]["first_stage"]}

    # ---- figures & outputs ----------------------------------------------------
    figures(con, exp, R, es, comm, deciles)
    dump_json(R, REPORTS / "results.json")
    print_summary(R)


# --------------------------------------------------------------------------- #
def figures(con, exp, R, es, comm, deciles):
    # Experiment funnel
    f = run_sql(con, "02_funnel.sql", group_col="variant", where="cohort = 'experiment'")
    fig, ax = plt.subplots(figsize=(8, 3.6))
    steps = f[f.grp == "control"].sort_values("step_order")["step"].tolist()
    x = np.arange(len(steps)); w = 0.38
    for i, v in enumerate(["control", "treatment"]):
        vals = f[f.grp == v].sort_values("step_order")["pct_of_signups"].to_numpy()
        ax.bar(x + (i - 0.5) * w, vals, w - 0.02, color=VARIANT_COLOR[v], label=v)
    for j, s in enumerate(steps):
        c = f[(f.grp == "control") & (f.step == s)]["pct_of_signups"].iloc[0]
        tr = f[(f.grp == "treatment") & (f.step == s)]["pct_of_signups"].iloc[0]
        if j:
            ax.text(j, max(c, tr) + 0.03, f"{(tr - c) * 100:+.1f}pp", ha="center", color=INK2, fontsize=9)
    ax.set_xticks(x, [s.replace("_", "\n") for s in steps]); ax.yaxis.set_major_formatter("{x:.0%}")
    ax.grid(axis="x", visible=False); ax.legend(loc="upper right")
    ax.set_title("Experiment funnel by variant (share of signups reaching each step)")
    save(fig, "04_experiment_funnel.png")

    # Retention curves
    rc = run_sql(con, "03_retention_curve.sql", group_col="variant", where="cohort = 'experiment'")
    fig, ax = plt.subplots(figsize=(7, 3.4))
    for v in ["control", "treatment"]:
        d = rc[rc.grp == v]
        ax.plot(d.day_n, d.retention, color=VARIANT_COLOR[v], label=v)
    ax.yaxis.set_major_formatter("{x:.0%}"); ax.set_ylim(0, None); ax.legend()
    ax.set_xlabel("Days since signup"); ax.set_title("N-day retention by variant")
    save(fig, "05_retention_by_variant.png")

    # Forest plot of relative lifts
    rows = [("Activation 7d (primary)", R["primary"])]
    names = {"onboarded_7d": "Onboarding complete", "d1_retained": "D1 retention", "d7_retained": "D7 retention",
             "d30_retained": "D30 retention", "paid_30d": "Paid conversion 30d",
             "revenue_30d": "Revenue / user 30d (bootstrap)", "sessions_per_active_day": "Sessions / active day (delta)"}
    for k, lab in names.items():
        rows.append((lab, R["secondary"][k]))
    rows.append(("Support tickets (guardrail)", R["guardrail_support"]))
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for i, (lab, r) in enumerate(rows[::-1]):
        c = r["control"]
        lo, hi, pt = r["ci_low"] / c, r["ci_high"] / c, r["diff"] / c
        sig = lo > 0 or hi < 0
        col = C_TREAT if sig else MUTED
        ax.plot([lo, hi], [i, i], color=col, lw=2); ax.plot(pt, i, "o", color=col, ms=7)
        note = " (higher = worse)" if "guardrail" in lab else ""
        ax.text(max(hi, 0) + 0.01, i, f"{pt:+.1%}{note}", va="center", color=INK2, fontsize=9)
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows[::-1]])
    ax.xaxis.set_major_formatter("{x:.0%}"); ax.grid(axis="y", visible=False)
    ax.set_xlim(None, 0.62); ax.set_xlabel("Relative lift vs control (95% CI; orange = CI excludes 0)")
    ax.set_title("Treatment effects across metrics")
    save(fig, "06_metric_forest.png")

    # Peeking
    fig, ax = plt.subplots(figsize=(7, 3.4))
    fpr = R["peeking_sim_aa"]["fpr_by_looks"]
    ax.plot(range(1, len(fpr) + 1), fpr, color=C_TREAT, label="Naive daily peeking")
    ax.axhline(R["peeking_sim_aa"]["msprt"], color=C_THIRD, label=f"mSPRT ({R['peeking_sim_aa']['msprt']:.1%})")
    ax.axhline(0.05, color=MUTED, ls="--", lw=1, label="Nominal 5%")
    ax.yaxis.set_major_formatter("{x:.0%}"); ax.set_ylim(0, None); ax.legend()
    ax.set_xlabel("Number of daily looks"); ax.set_ylabel("False-positive rate (A/A)")
    ax.set_title(f"Peeking inflates false positives to {fpr[-1]:.0%}")
    save(fig, "07_peeking.png")

    # Love plot
    fig, ax = plt.subplots(figsize=(7, 3.8))
    b, a = comm["balance_before"], comm["balance_after"]
    order = b.abs().sort_values().index
    yy = np.arange(len(order))
    ax.plot(b[order].abs(), yy, "o", color=C_CONTROL, label="Before weighting", ms=7)
    ax.plot(a[order].abs(), yy, "o", color=C_TREAT, label="After IPW", ms=7)
    ax.axvline(0.1, color=MUTED, ls="--", lw=1)
    ax.set_yticks(yy, order); ax.legend(); ax.grid(axis="y", visible=False)
    ax.set_xlabel("|Standardized mean difference| (joiners vs non-joiners; < 0.1 = balanced)")
    ax.set_title("Covariate balance: community joiners vs non-joiners")
    save(fig, "08_ipw_balance.png")

    # Event study
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.fill_between(es.rel_week, es.ci_low * 100, es.ci_high * 100, color=C_CONTROL, alpha=0.18, lw=0)
    ax.plot(es.rel_week, es.coef * 100, "o-", color=C_CONTROL, ms=5)
    ax.axhline(0, color=INK, lw=0.8); ax.axvline(-0.5, color=MUTED, ls="--", lw=1)
    ax.set_xlabel("Weeks relative to rollout"); ax.set_ylabel("Effect on weekly active rate (pp)")
    ax.set_title("Event study: flat pre-trends, then a step up after rollout")
    save(fig, "09_event_study.png")

    # Uplift quintiles
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.bar(deciles["bin"], deciles["observed_effect"] * 100, color=C_TREAT, width=0.6)
    ax.errorbar(deciles["bin"], deciles["observed_effect"] * 100,
                yerr=[(deciles["observed_effect"] - deciles["ci_low"]) * 100, (deciles["ci_high"] - deciles["observed_effect"]) * 100],
                fmt="none", ecolor=INK2, capsize=3, lw=1)
    ax.plot(deciles["bin"], deciles["pred_cate"] * 100, "o", color=INK, ms=6, label="Predicted (T-learner)")
    ax.axhline(R["primary"]["diff"] * 100, color=MUTED, ls="--", lw=1, label="Average effect")
    ax.set_xticks(deciles["bin"], [f"Q{b}" for b in deciles["bin"]])
    ax.set_xlabel("Users grouped by predicted uplift (Q1 = highest)")
    ax.set_ylabel("Activation lift (pp)"); ax.legend(loc="upper right", fontsize=9); ax.grid(axis="x", visible=False)
    ax.set_title("Who benefits most? Observed lift by predicted-uplift quintile")
    save(fig, "10_uplift_quintiles.png")


def print_summary(R):
    p = R["primary"]
    print(f"SRM p={R['srm']['p_value']:.3f}")
    print(f"Primary: {p['control']:.4f} -> {p['treatment']:.4f}  diff={p['diff']*100:+.2f}pp "
          f"CI[{p['ci_low']*100:+.2f},{p['ci_high']*100:+.2f}] p={p['p_value']:.2g}")
    print(f"CUPED VR={R['cuped']['variance_reduction']:.1%}  CUPAC VR={R['cupac']['variance_reduction']:.1%}  "
          f"CUPAC diff={R['cupac']['adj_diff']*100:+.2f}pp CI[{R['cupac']['adj_ci'][0]*100:+.2f},{R['cupac']['adj_ci'][1]*100:+.2f}]")
    for k, v in R["secondary"].items():
        print(f"  {k:26s} diff={v['diff']:+.4f} CI[{v['ci_low']:+.4f},{v['ci_high']:+.4f}] p={v['p_value']:.3f} holm={v['p_holm']:.3f}")
    g = R["guardrail_support"]; print(f"Guardrail support diff={g['diff']:+.4f} p={g['p_value']:.3f}")
    print("Peeking A/A:", {k: v for k, v in R["peeking_sim_aa"].items() if k != "fpr_by_looks"})
    print("Peeking A/B power:", {k: v for k, v in R["peeking_sim_ab"].items() if k != "fpr_by_looks"})
    print("mSPRT first sig day:", R["msprt_on_experiment"]["first_day_significant"])
    iv = R["iv"]; print(f"IV: ITT={iv['itt']:.4f} first stage={iv['first_stage']:.3f} CACE={iv['cace']:.4f} "
                        f"CI[{iv['ci_low']:.4f},{iv['ci_high']:.4f}] per-protocol={iv['per_protocol_naive']:.4f} as-treated={iv['as_treated_naive']:.4f}")
    c = R["community"]; print(f"Community: naive={c['naive']:.4f} ipw={c['ipw']:.4f}{c['ipw_ci']} aipw={c['aipw']:.4f}{c['aipw_ci']} "
                              f"SMD {c['max_abs_smd_before']:.2f}->{c['max_abs_smd_after']:.2f} E={c['e_value']:.2f}")
    print('WCB:', R['did_wild_bootstrap']); print('RI:', R['did_randomization']); print('DiD size sim:', R['did_size_sim'])
    d = R["did"]; print(f"DiD: att={d['att']:.4f} CI[{d['ci_low']:.4f},{d['ci_high']:.4f}] naive_post={d['naive_post_comparison']:.4f} before_after={d['naive_before_after']:.4f}")
    print("Uplift qini:", round(R["uplift"]["qini"], 5))
    print(pd.DataFrame(R["uplift"]["quintiles"]).round(4).to_string())
    print(pd.DataFrame(R["segments"]).round(4).to_string())
    print(json.dumps(R["impact"], indent=1, default=float))


if __name__ == "__main__":
    main()
