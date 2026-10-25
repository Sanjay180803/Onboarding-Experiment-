# Metric review: how definitions were agreed

Metric definitions for this experiment live in [`metrics/catalog.yml`](../metrics/catalog.yml). This page records how they were set with stakeholders before launch, and the questions each definition answers. *(The scenario is simulated; the process is the one I would run on a real team.)*

## Design review agenda (before launch)

1. **What decision is this test for?** Ship the onboarding checklist, or not. Everyone agrees on one primary metric and one decision rule *before* seeing data.
2. **Primary metric.** Monetization asked for paid conversion. The power analysis showed it needs roughly 5x the traffic we have in two weeks, so we agreed on **7-day activation** as the primary metric: it moves within the test window, it is the funnel step the checklist targets, and historically it is strongly associated with paying. Paid conversion and D30 retention became secondary metrics, to be measured properly through a 5% long-term holdout.
3. **Definitions that caused debate:**

   | Question | Options discussed | Decision | Why |
   |---|---|---|---|
   | Activation window from signup or from first session? | signup / first session | **Signup** | First-session anchoring would drop users who never return, which is exactly the population the checklist targets. |
   | Denominator: everyone assigned, or only users who saw the checklist? | ITT / per-protocol | **Everyone assigned (ITT)** | Only ~60% open the checklist and openers are more motivated, so per-protocol is biased. The effect on openers is estimated separately with IV. |
   | Retention: exact-day or rolling? | exact day N / active on day N or later | **Exact day** | Matches the company's existing dashboards, so numbers are comparable. Caveat documented. |
   | Sessions per active day: ratio of sums or mean of ratios? | pooled / per-user mean | **Pooled ratio, delta-method CI** | Matches how the product dashboard reports it; per-user means overweight users with one active day. |
   | Support tickets guardrail: what blocks launch? | any significant rise / threshold | **+1pp threshold plus cost check** | With ~10 tests, one false positive at alpha 0.05 is expected. Support ops agreed a threshold tied to their staffing cost. |

4. **Owners.** Each metric has a named owner who approves definition changes (see the catalog).
5. **Sign-off.** Growth PM, Monetization PM, Support ops and Data science approved catalog v1.0 on 2026-07-20, and v1.1 (guardrail rule) on 2026-07-27, both before launch.

## Rules for changing a metric
- Never change a definition while a test that uses it is running.
- Bump the metric's `version`, add a changelog entry, and get the owner's approval.
- The catalog test in CI fails if a metric points to a column or result that no longer exists.
