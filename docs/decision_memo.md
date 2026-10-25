# Decision memo: Guided Onboarding Checklist

**Recommendation: Ship to 95% of new users while retaining a 5% long-term control holdout**, plus a support-ticket monitor.

## What we tested
A guided onboarding checklist, aimed at the funnel's biggest leak (only 31% of users who finish onboarding go on to activate). The test ran for 14 days with 14,082 new users, using seeded 50/50 Bernoulli assignment at the user level. It was pre-registered and powered for +2pp on 7-day activation.

## What we found
1. **Activation rose +2.4pp** (from 21.0% to 23.4%, a 12% relative lift). 95% CI [+1.07, +3.82], p = 0.0005. The result held under variance-reduced analysis and passed the sample-ratio check.
2. **For users who actually opened the checklist, the lift is about +4.1pp.** Only 60% opened it, so discoverability is the next lever.
3. **Downstream metrics (retention, paid, revenue) are directionally positive but not significant.** This was expected: the test was not sized to detect them.
4. **Support tickets ticked up +0.6pp** (p = 0.03). This is below our +1pp block threshold. If real, it would cost about $12,609 per year, well under the benefit below.

## Expected impact (annualized, 95% of ~364,166 signups = 345,957 exposed users)
| | CI low | Point | CI high |
|---|---|---|---|
| Additional activated users | 3,716 | 8,462 | 13,208 |
| Additional payers (pessimistic to optimistic link) | 328–656 | 747–1,493 | 1,165–2,331 |
| First-year revenue | $21,313–$42,626 | **$48,532–$97,064** | $75,751–$151,502 |

*Assumptions:* first-year value of $65 per payer, and $6 per support ticket. The activation-to-payer link comes from historical data, adjusted for user mix (0.176 payers per activated user), and is treated as an upper bound because motivated users both activate and pay. The pessimistic case uses half of it. The raw, unadjusted gap (0.189) would overstate impact.

## Next steps
1. **Ship to 95% of new users on all platforms and keep a 5% control holdout for at least 90 days**, to measure paid conversion and revenue directly. That is the only way to replace the assumption above with a measured number. Impact figures above already exclude the holdout.
2. **Run a follow-up test on checklist discoverability.** Raising opens from 60% to 80% is worth about +0.8pp more activation.
3. **Tag and monitor support tickets** from checklist users for 4 weeks. Revisit if the rate rises by 1pp or more.
4. **Do not exclude web users yet.** The negative web estimate is one of 11 unplanned segment cuts with a wide CI. Low-intent users may benefit most; test a tailored checklist for them.
5. **Separately, community features look causally valuable** (about +3.5pp D30 retention after adjusting for self-selection, versus +6.9pp naive). Confirm with a randomized encouragement test before investing.
