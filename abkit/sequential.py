"""The peeking problem and an always-valid fix (mixture SPRT).

Checking a fixed-horizon p-value every day and stopping at the first p < 0.05
inflates the false-positive rate well above 5%. mSPRT (Johari et al., 2017)
gives always-valid p-values: you may look as often as you like.
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def msprt_lambda(diff, var_diff, tau2: float):
    """Mixture likelihood ratio for H0: delta = 0 vs a N(0, tau2) mixture,
    with an (approximately) normal estimator `diff` of variance `var_diff`."""
    diff, v = np.asarray(diff, float), np.asarray(var_diff, float)
    return np.sqrt(v / (v + tau2)) * np.exp(diff ** 2 * tau2 / (2 * v * (v + tau2)))


def always_valid_pvalues(diffs, var_diffs, tau2: float) -> np.ndarray:
    lam = msprt_lambda(diffs, var_diffs, tau2)
    # running minimum over looks (last axis): once small, an always-valid p stays small
    return np.minimum.accumulate(np.minimum(1.0, 1.0 / np.maximum(lam, 1e-300)), axis=-1)


def simulate_peeking(p: float = 0.21, n_per_day_per_arm: int = 500, days: int = 14,
                     sims: int = 4000, alpha: float = 0.05, tau: float = 0.02,
                     effect: float = 0.0, seed: int = 0) -> dict:
    """A/A (effect=0) or A/B simulation of daily looks.
    Returns rejection rates for: one look at the end, naive daily peeking, mSPRT."""
    rng = np.random.default_rng(seed)
    xc = rng.binomial(n_per_day_per_arm, p, (sims, days)).cumsum(1)
    xt = rng.binomial(n_per_day_per_arm, p + effect, (sims, days)).cumsum(1)
    n = n_per_day_per_arm * np.arange(1, days + 1)
    pc, pt = xc / n, xt / n
    pp = (xc + xt) / (2 * n)
    se_pooled = np.sqrt(pp * (1 - pp) * 2 / n)
    pval = 2 * stats.norm.sf(np.abs(pt - pc) / se_pooled)
    var = (pc * (1 - pc) + pt * (1 - pt)) / n
    av_p = always_valid_pvalues(pt - pc, np.maximum(var, 1e-12), tau ** 2)
    return {
        "fixed_horizon": float(np.mean(pval[:, -1] < alpha)),
        "naive_peeking": float(np.mean((pval < alpha).any(1))),
        "msprt": float(np.mean((av_p < alpha).any(1))),
        "fpr_by_looks": [float(np.mean((pval[:, :k] < alpha).any(1))) for k in range(1, days + 1)],
    }
