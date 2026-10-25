"""STEP 1 -- Experiment design, using ONLY the pre-experiment (historical) cohort.

Mirrors how a real test is planned: baseline metrics, funnel diagnosis, power
analysis, CUPED/CUPAC covariate selection, runtime. Writes the numbers into
docs/preregistration.md before any experiment data is looked at.
"""
import numpy as np
import pandas as pd

from common import REPORTS, ROOT, connect, run_sql, style, save, dump_json, C_CONTROL, C_TREAT, C_THIRD, INK2, MUTED
import matplotlib.pyplot as plt
from abkit import power, variance_reduction as vr

ALPHA, POWER, MDE = 0.05, 0.80, 0.02
FEATURES = ["platform", "channel", "goal", "pre_visits"]


def main():
    style()
    con = connect()
    hist = con.sql("SELECT * FROM user_metrics WHERE cohort = 'historical' ORDER BY user_id").df()

    # ---- baselines & funnel ------------------------------------------------
    funnel = run_sql(con, "02_funnel.sql", group_col="cohort", where="cohort = 'historical'")
    cohorts = run_sql(con, "04_cohort_retention.sql")
    retention = run_sql(con, "03_retention_curve.sql", group_col="cohort", where="cohort = 'historical'")
    leak = funnel.dropna(subset=["step_conversion"]).sort_values("step_conversion").iloc[0]

    p0 = hist["activated_7d"].mean()
    daily = hist.groupby("signup_day").size().mean()

    # ---- power ---------------------------------------------------------------
    n = power.n_per_arm_proportions(p0, MDE, ALPHA, POWER)
    sim_power = power.simulate_power(p0, p0 + MDE, n, ALPHA, sims=10000)
    runtime = power.runtime_days(n, daily)

    # Covariates for variance reduction -- evaluated out-of-sample on historical data
    rho_cuped = np.corrcoef(hist["activated_7d"], hist["pre_visits"])[0, 1]
    half = hist.sample(frac=0.5, random_state=0)
    rest = hist.drop(half.index)
    model, cols = vr.fit_cupac_model(half, FEATURES, "activated_7d")
    pred = vr.cupac_covariate(model, cols, rest, FEATURES)
    rho_cupac = np.corrcoef(rest["activated_7d"], pred)[0, 1]

    design = {
        "baseline_activation_7d": p0, "baseline_d7": hist["d7_retained"].mean(),
        "baseline_d30": hist["d30_retained"].mean(), "baseline_paid_30d": hist["paid_30d"].mean(),
        "baseline_revenue_30d": hist["revenue_30d"].mean(), "daily_signups": daily,
        "alpha": ALPHA, "power": POWER, "mde_abs": MDE, "mde_rel": MDE / p0,
        "n_per_arm": n, "n_total": 2 * n, "simulated_power": sim_power, "runtime_days": runtime,
        "rho_cuped_pre_visits": rho_cuped, "n_per_arm_cuped": power.cuped_adjusted_n(n, rho_cuped),
        "rho_cupac": rho_cupac, "n_per_arm_cupac": power.cuped_adjusted_n(n, rho_cupac),
        "mde_at_planned_n_d30": power.mde_proportions(hist["d30_retained"].mean(), n),
        "mde_at_planned_n_paid": power.mde_proportions(hist["paid_30d"].mean(), n),
        "biggest_funnel_leak": {"from_step": funnel.loc[funnel.step_order == leak["step_order"] - 1, "step"].iloc[0],
                                "step": leak["step"], "step_conversion": leak["step_conversion"]},
    }
    dump_json(design, REPORTS / "design.json")
    funnel.to_csv(REPORTS / "historical_funnel.csv", index=False)
    cohorts.to_csv(REPORTS / "historical_cohorts.csv", index=False)

    # ---- figures -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 3.6))
    f = funnel.sort_values("step_order")
    labels = f["step"].str.replace("_", " ").tolist()
    ax.barh(labels[::-1], f["pct_of_signups"][::-1], color=C_CONTROL, height=0.6)
    for i, (pct, conv) in enumerate(zip(f["pct_of_signups"][::-1], f["step_conversion"][::-1])):
        txt = f"{pct:.0%}" + ("" if pd.isna(conv) else f"   ({conv:.0%} of prior step)")
        ax.text(pct + 0.01, i, txt, va="center", color=INK2, fontsize=9)
    ax.set_xlim(0, 1.25); ax.set_xticks([0, .25, .5, .75, 1]); ax.xaxis.set_major_formatter("{x:.0%}")
    ax.grid(axis="y", visible=False)
    prev = f.loc[f.step_order == leak["step_order"] - 1, "step"].iloc[0]
    ax.set_title(f"Historical funnel: biggest leak is {prev.replace('_', ' ')} \u2192 "
                 f"{leak['step'].replace('_', ' ')} ({leak['step_conversion']:.0%} convert)")
    save(fig, "01_historical_funnel.png")

    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.plot(retention["day_n"], retention["retention"], color=C_CONTROL)
    for d in (1, 7, 30):
        v = retention.loc[retention.day_n == d, "retention"].iloc[0]
        ax.plot(d, v, "o", color=C_CONTROL, ms=6)
        ax.annotate(f"D{d}: {v:.0%}", (d, v), xytext=(6, 6), textcoords="offset points", color=INK2, fontsize=9)
    ax.set_ylim(0, None); ax.yaxis.set_major_formatter("{x:.0%}")
    ax.set_xlabel("Days since signup"); ax.set_title("Historical N-day retention")
    save(fig, "02_historical_retention.png")

    fig, ax = plt.subplots(figsize=(7, 3.6))
    mdes = np.linspace(0.008, 0.04, 60)
    for label, rho, color in (("Plain z-test", 0, C_CONTROL), ("CUPED (pre_visits)", rho_cuped, C_TREAT),
                              ("CUPAC (ML covariate)", rho_cupac, C_THIRD)):
        ns = [power.n_per_arm_proportions(p0, m) * (1 - rho ** 2) for m in mdes]
        ax.plot(mdes * 100, ns, color=color, label=label)
    ax.axvline(MDE * 100, color=MUTED, ls="--", lw=1)
    ax.annotate(f"Planned MDE +{MDE*100:.0f}pp\n{n:,} users/arm", (MDE * 100, n), xytext=(10, 20),
                textcoords="offset points", color=INK2, fontsize=9)
    ax.set_yscale("log"); ax.yaxis.set_major_formatter("{x:,.0f}"); ax.yaxis.set_minor_formatter("")
    ax.set_yticks([2000, 5000, 10000, 20000, 40000])
    ax.text(0.99, 0.62, f"CUPED/CUPAC save only {1 - (1 - rho_cupac**2):.0%}:\nnew users have little pre-period data",
            transform=ax.transAxes, ha="right", color=INK2, fontsize=9)
    ax.set_xlabel("Minimum detectable effect (percentage points)")
    ax.set_ylabel("Users per arm"); ax.legend()
    ax.set_title("Sample size vs. MDE (alpha=0.05, power=0.80)")
    save(fig, "03_power_curve.png")

    write_prereg(design)
    print(pd.Series({k: v for k, v in design.items() if not isinstance(v, dict)}).to_string())


def write_prereg(d):
    text = f"""# Pre-registration: Guided Onboarding Checklist

*Written from the pre-experiment cohort only ({ROOT.name}/analysis/01_design.py), before any experiment data was analyzed.*

## Hypothesis
Users offered a guided onboarding checklist will complete their first core action (activation) within 7 days at a higher rate than users on the current onboarding flow.

Why: the historical funnel shows the largest leak between **{d['biggest_funnel_leak']['from_step'].replace('_', ' ')}** and **{d['biggest_funnel_leak']['step'].replace('_', ' ')}**: only {d['biggest_funnel_leak']['step_conversion']:.0%} of users make that step. The checklist targets exactly that step.

## Design
| Item | Decision |
|---|---|
| Unit of randomization | Seeded 50/50 Bernoulli assignment at the user level (at signup) |
| Population | All new signups, all platforms, during the test window |
| Primary metric | 7-day activation (core action within 7 days of signup) |
| Secondary metrics | Onboarding completion (7d), D7 retention, D30 retention, paid conversion (30d), revenue per user (30d), sessions per active day |
| Guardrails | Support tickets (7d); sample ratio must match 50/50 (SRM p > 0.001, otherwise the test is invalid) |
| Analysis | Intent-to-treat (everyone assigned, regardless of whether they opened the checklist) |
| Test | Two-sided two-proportion z-test, alpha = {d['alpha']}; CUPAC-adjusted estimate reported alongside |
| Multiple testing | Holm correction across secondary metrics; primary metric uncorrected |
| Decision rule | Ship if the primary metric is significant with CI lower bound > 0. A guardrail blocks launch if its point estimate is worse by more than +1pp absolute, or if the cost of the harm exceeds the value of the primary effect's CI lower bound. Otherwise a significant guardrail move triggers investigation and monitoring, not a block. |
| Peeking | No decisions before day {d['runtime_days']}; any interim look uses always-valid (mSPRT) p-values |

## Power analysis
| Input | Value |
|---|---|
| Baseline 7-day activation | {d['baseline_activation_7d']:.2%} |
| Minimum detectable effect | +{d['mde_abs']*100:.1f}pp absolute ({d['mde_rel']:.1%} relative) |
| alpha / power | {d['alpha']} / {d['power']} |
| **Users per arm** | **{d['n_per_arm']:,}** (simulated power check: {d['simulated_power']:.1%}) |
| Daily eligible signups | {d['daily_signups']:,.0f} |
| **Runtime** | **{d['runtime_days']} days** (rounded up to full weeks to balance weekdays) |
| CUPED with pre_visits (rho = {d['rho_cuped_pre_visits']:.2f}) | {d['n_per_arm_cuped']:,} per arm |
| CUPAC with ML covariate (rho = {d['rho_cupac']:.2f}, out-of-sample) | {d['n_per_arm_cupac']:,} per arm |

**Known limitation:** at this sample size the MDE for D30 retention is {d['mde_at_planned_n_d30']*100:.1f}pp and for paid conversion {d['mde_at_planned_n_paid']*100:.1f}pp. Downstream metrics are therefore *directional*; this test is powered for activation only.
"""
    (ROOT / "docs" / "preregistration.md").write_text(text)


if __name__ == "__main__":
    main()
