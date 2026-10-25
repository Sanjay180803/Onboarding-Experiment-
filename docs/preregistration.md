# Pre-registration: Guided Onboarding Checklist

*Written from the pre-experiment cohort only (onboarding-experiment/analysis/01_design.py), before any experiment data was analyzed.*

## Hypothesis
Users offered a guided onboarding checklist will complete their first core action (activation) within 7 days at a higher rate than users on the current onboarding flow.

Why: the historical funnel shows the largest leak between **onboarding complete** and **activated**: only 31% of users make that step. The checklist targets exactly that step.

## Design
| Item | Decision |
|---|---|
| Unit of randomization | Seeded 50/50 Bernoulli assignment at the user level (at signup) |
| Population | All new signups, all platforms, during the test window |
| Primary metric | 7-day activation (core action within 7 days of signup) |
| Secondary metrics | Onboarding completion (7d), D7 retention, D30 retention, paid conversion (30d), revenue per user (30d), sessions per active day |
| Guardrails | Support tickets (7d); sample ratio must match 50/50 (SRM p > 0.001, otherwise the test is invalid) |
| Analysis | Intent-to-treat (everyone assigned, regardless of whether they opened the checklist) |
| Test | Two-sided two-proportion z-test, alpha = 0.05; CUPAC-adjusted estimate reported alongside |
| Multiple testing | Holm correction across secondary metrics; primary metric uncorrected |
| Decision rule | Ship if the primary metric is significant with CI lower bound > 0. A guardrail blocks launch if its point estimate is worse by more than +1pp absolute, or if the cost of the harm exceeds the value of the primary effect's CI lower bound. Otherwise a significant guardrail move triggers investigation and monitoring, not a block. |
| Peeking | No decisions before day 14; any interim look uses always-valid (mSPRT) p-values |

## Power analysis
| Input | Value |
|---|---|
| Baseline 7-day activation | 21.08% |
| Minimum detectable effect | +2.0pp absolute (9.5% relative) |
| alpha / power | 0.05 / 0.8 |
| **Users per arm** | **6,751** (simulated power check: 80.6%) |
| Daily eligible signups | 998 |
| **Runtime** | **14 days** (rounded up to full weeks to balance weekdays) |
| CUPED with pre_visits (rho = 0.18) | 6,521 per arm |
| CUPAC with ML covariate (rho = 0.22, out-of-sample) | 6,434 per arm |

**Known limitation:** at this sample size the MDE for D30 retention is 1.6pp and for paid conversion 1.3pp. Downstream metrics are therefore *directional*; this test is powered for activation only.
