"""Frequentist inference for A/B tests: SRM, proportions, means, bootstrap,
delta method for ratio metrics, multiple-testing correction."""
from __future__ import annotations

import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests


def srm_test(observed: list[int], expected_ratio: list[float] | None = None) -> dict:
    """Sample-ratio-mismatch chi-square test. A tiny p-value (< 0.001 is the
    usual threshold) means assignment/logging is broken -- don't trust results."""
    obs = np.asarray(observed, float)
    ratio = np.ones_like(obs) / len(obs) if expected_ratio is None else np.asarray(expected_ratio, float)
    exp = obs.sum() * ratio / ratio.sum()
    chi2, p = stats.chisquare(obs, exp)
    return {"observed": obs.astype(int).tolist(), "expected": exp.round(1).tolist(),
            "chi2": float(chi2), "p_value": float(p), "srm_detected": bool(p < 0.001)}


def two_proportion_test(x_t: int, n_t: int, x_c: int, n_c: int, alpha: float = 0.05) -> dict:
    """z-test (pooled SE) with Wald CI on the absolute difference (unpooled SE)
    and a delta-method CI on the relative lift."""
    p_t, p_c = x_t / n_t, x_c / n_c
    diff = p_t - p_c
    pp = (x_t + x_c) / (n_t + n_c)
    se_pooled = np.sqrt(pp * (1 - pp) * (1 / n_t + 1 / n_c))
    z = diff / se_pooled
    p = 2 * stats.norm.sf(abs(z))
    se = np.sqrt(p_t * (1 - p_t) / n_t + p_c * (1 - p_c) / n_c)
    zc = stats.norm.ppf(1 - alpha / 2)
    # relative lift: Var(pt/pc) ~ (1/pc^2)Var(pt) + (pt^2/pc^4)Var(pc)
    rel = diff / p_c
    se_rel = np.sqrt((p_t * (1 - p_t) / n_t) / p_c ** 2 + (p_t ** 2) * (p_c * (1 - p_c) / n_c) / p_c ** 4)
    return {"control": p_c, "treatment": p_t, "diff": diff, "se": se,
            "ci_low": diff - zc * se, "ci_high": diff + zc * se,
            "rel_lift": rel, "rel_ci_low": rel - zc * se_rel, "rel_ci_high": rel + zc * se_rel,
            "z": float(z), "p_value": float(p), "n_t": n_t, "n_c": n_c}


def mean_diff_test(y_t, y_c, alpha: float = 0.05) -> dict:
    """Welch t-test + CI for a difference in means."""
    y_t, y_c = np.asarray(y_t, float), np.asarray(y_c, float)
    diff = y_t.mean() - y_c.mean()
    se = np.sqrt(y_t.var(ddof=1) / len(y_t) + y_c.var(ddof=1) / len(y_c))
    t, p = stats.ttest_ind(y_t, y_c, equal_var=False)
    zc = stats.norm.ppf(1 - alpha / 2)
    return {"control": y_c.mean(), "treatment": y_t.mean(), "diff": diff, "se": se,
            "ci_low": diff - zc * se, "ci_high": diff + zc * se,
            "rel_lift": diff / y_c.mean() if y_c.mean() else np.nan,
            "p_value": float(p), "n_t": len(y_t), "n_c": len(y_c)}


def bootstrap_diff(y_t, y_c, stat=np.mean, n_boot: int = 4000, alpha: float = 0.05,
                   seed: int = 0) -> dict:
    """Percentile bootstrap CI for stat(treatment) - stat(control).
    Use for heavy-tailed metrics (revenue) where the CLT kicks in slowly."""
    rng = np.random.default_rng(seed)
    y_t, y_c = np.asarray(y_t, float), np.asarray(y_c, float)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        boots[i] = stat(y_t[rng.integers(0, len(y_t), len(y_t))]) - stat(y_c[rng.integers(0, len(y_c), len(y_c))])
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    point = stat(y_t) - stat(y_c)
    # two-sided bootstrap p-value: share of the centered distribution beyond |point|
    p = float(np.mean(np.abs(boots - boots.mean()) >= abs(point)))
    return {"diff": float(point), "ci_low": float(lo), "ci_high": float(hi),
            "p_value": max(p, 1 / n_boot), "n_boot": n_boot}


def _ratio_var(num, den):
    """Delta-method variance of sum(num)/sum(den) with user-level i.i.d. units."""
    n = len(num)
    mx, my = num.mean(), den.mean()
    r = mx / my
    vx, vy = num.var(ddof=1), den.var(ddof=1)
    cxy = np.cov(num, den, ddof=1)[0, 1]
    return r, (vx - 2 * r * cxy + r ** 2 * vy) / (my ** 2 * n)


def delta_ratio_test(num_t, den_t, num_c, den_c, alpha: float = 0.05) -> dict:
    """Ratio metric (e.g. sessions per active day) when the randomization unit
    (user) differs from the analysis unit (day). Naively treating days as
    independent understates the variance; the delta method does not."""
    rt, vt = _ratio_var(np.asarray(num_t, float), np.asarray(den_t, float))
    rc, vc = _ratio_var(np.asarray(num_c, float), np.asarray(den_c, float))
    diff, se = rt - rc, np.sqrt(vt + vc)
    zc = stats.norm.ppf(1 - alpha / 2)
    return {"control": rc, "treatment": rt, "diff": diff, "se": se,
            "ci_low": diff - zc * se, "ci_high": diff + zc * se,
            "rel_lift": diff / rc, "p_value": float(2 * stats.norm.sf(abs(diff / se)))}


def adjust_pvalues(pvals: dict[str, float], method: str = "holm") -> dict[str, float]:
    """Family-wise (holm) or FDR (fdr_bh) correction across secondary metrics."""
    keys = list(pvals)
    _, adj, _, _ = multipletests([pvals[k] for k in keys], method=method)
    return dict(zip(keys, adj.tolist()))
