# Onboarding Experiment: A/B Testing & Causal Inference, End to End

A complete product-experimentation case study: a subscription app tests a **guided onboarding checklist**. The project covers the full lifecycle a product data scientist owns — funnel diagnosis → pre-registered design and power analysis → analysis with proper inference → causal inference where randomization breaks down → a business recommendation with dollar impact.

The data is **synthetic with known ground truth**, so every estimator is checked against the true causal effect. Naive methods are shown to fail and principled ones to succeed.

## Headline results

| Question | Method | Result | Truth |
|---|---|---|---|
| Does the checklist lift 7-day activation? | Two-proportion z-test, 14k users | **+2.45pp** [+1.07, +3.82], p = 0.0005 (+11.7% relative) | +2.90pp ✓ |
| Effect on users who actually *used* it? | IV / CACE (randomization as instrument) | **+4.1pp** [+1.8, +6.4]; naive comparisons say +6.6 to +7.6pp | +4.9pp ✓ |
| Does joining the community *cause* retention? | IPW and doubly robust AIPW | **+3.5pp** D30 retention; naive says +6.9pp | +3.4pp ✓ |
| Did streak reminders work? (non-random rollout) | Difference-in-differences + event study | **+2.1pp** weekly active rate; naive says +5.0pp | +2.0pp ✓ |
| Is daily peeking safe? | 5,000 simulated A/A tests | Naive peeking: **21.7% false positives**; mSPRT: 1.1% | — |
| Decision | Decision memo | **Ship**, with a 5% holdout; worth roughly $51k–$102k/yr | — |

## Skills demonstrated

| Skill | Where |
|---|---|
| Product funnels & retention | `sql/` (DuckDB): strict ordered funnel, N-day retention curves, weekly cohort table |
| A/B test design | `docs/preregistration.md`: hypothesis, primary/secondary/guardrail metrics, decision rule written before analysis |
| Power analysis | `abkit/power.py`: sample size, MDE, analytic power validated by simulation, runtime rounded to full weeks |
| Significance & confidence intervals | `abkit/inference.py`: SRM check, z-test, Welch, bootstrap (heavy-tailed revenue), delta method (ratio metrics), Holm correction |
| Variance reduction | `abkit/variance_reduction.py`: CUPED, and CUPAC with an ML covariate trained on the pre-period cohort |
| Sequential testing | `abkit/sequential.py`: peeking simulation, always-valid mSPRT p-values |
| Causal inference | `abkit/causal.py`: IV/CACE, IPW and cross-fitted AIPW with balance diagnostics and an E-value, DiD and event study |
| Heterogeneous effects | `abkit/uplift.py`: cross-fitted T-learner, uplift by quintile, Qini |
| Business recommendation | `docs/decision_memo.md`: annualized impact range, guardrail cost-benefit, next tests |

## Repo layout

```
sim/simulate.py          Event-log simulator with known causal effects (writes data/, ground truth to data/_truth/)
sql/                     DuckDB SQL: user-level metrics, funnel, retention curve, cohort table
abkit/                   Reusable, tested experimentation toolkit
analysis/01_design.py    Design from PRE-experiment data only -> docs/preregistration.md
analysis/02_analyze.py   Full analysis -> reports/results.json + figures
analysis/03_validate.py  Compare every estimate with ground truth
analysis/04_report.py    Generate reports/report.md and docs/decision_memo.md
tests/                   12 statistical correctness tests (CI coverage, FPR control, recovery of known effects)
```

## Run it

```bash
pip install -r requirements.txt
make all        # tests -> simulate -> design -> analyze -> validate -> report  (~2 min)
```

Every number in the report and memo is generated from code. Change the seed (`python sim/simulate.py --seed 11`) and the whole story regenerates.

## Things worth discussing in an interview

- **The guardrail false positive.** Support tickets rose significantly (p = 0.03), yet the true effect is zero. Across about 10 tests at alpha = 0.05, one miss is expected. The pre-registered rule (a block threshold plus a cost-benefit check) avoided killing a good launch over noise.
- **CUPED barely helped (4%).** New users have little pre-period behavior. Knowing *when* a technique won't help is as useful as knowing how to apply it.
- **Revenue moved the "wrong" way** while paid conversion rose. Revenue is heavy-tailed (annual vs monthly plans), and the test wasn't powered for it, as stated in the design. Hence the recommended 90-day holdout.
- **Translating activation into dollars needs a causal link**, not the raw "activated users pay more" gap, which is confounded by motivation. The memo shows a range rather than a falsely precise number.
- **Uplift modeling is weak at this sample size.** Rank correlation with the true individual effect is about 0.11. Segment-level effects are informative, but a per-user targeting model is not yet justified.

## Extending it
- Apply the same `abkit` analysis to a public dataset (e.g. the Kaggle *Cookie Cats* mobile-game retention A/B test).
- Add Bayesian analysis (beta-binomial posterior, expected loss) alongside the frequentist results.
- Replace the DiD with Callaway–Sant'Anna for a staggered rollout.
- Add a Streamlit dashboard over `reports/results.json`.
