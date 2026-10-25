"""Causal inference beyond the clean randomized comparison.

- Noncompliance: ITT vs CACE (instrumental variables, randomization as instrument)
- Self-selection: propensity-score weighting (IPW) and doubly-robust AIPW,
  with covariate balance diagnostics and an E-value sensitivity check
- Non-randomized rollout: difference-in-differences + event study
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import KFold

Z95 = stats.norm.ppf(0.975)


# --------------------------------------------------------------------------- #
# Instrumental variables / noncompliance
# --------------------------------------------------------------------------- #
def iv_cace(y, z, d) -> dict:
    """Wald / 2SLS estimate of the complier average causal effect.
    Assumptions: random Z, exclusion (Z affects Y only through D),
    monotonicity (no defiers -- guaranteed here: control can't see the checklist)."""
    y, z, d = (np.asarray(a, float) for a in (y, z, d))
    zt = z == 1
    itt_y = y[zt].mean() - y[~zt].mean()
    itt_d = d[zt].mean() - d[~zt].mean()
    cace = itt_y / itt_d
    e = y - cace * d                                 # influence-function residual
    se = np.sqrt(e[zt].var(ddof=1) / zt.sum() + e[~zt].var(ddof=1) / (~zt).sum()) / itt_d
    # common mistakes, for comparison
    per_protocol = y[zt & (d == 1)].mean() - y[~zt].mean()
    as_treated = y[d == 1].mean() - y[d == 0].mean()
    return {"itt": itt_y, "first_stage": itt_d, "cace": cace, "se": se,
            "ci_low": cace - Z95 * se, "ci_high": cace + Z95 * se,
            "per_protocol_naive": per_protocol, "as_treated_naive": as_treated}


# --------------------------------------------------------------------------- #
# Propensity scores
# --------------------------------------------------------------------------- #
def _design(df: pd.DataFrame, covariates: list[str]) -> pd.DataFrame:
    return pd.get_dummies(df[covariates], drop_first=True).astype(float)


def std_mean_diff(X: pd.DataFrame, t: np.ndarray, w: np.ndarray | None = None) -> pd.Series:
    w = np.ones(len(t)) if w is None else w
    out = {}
    for c in X.columns:
        x = X[c].to_numpy()
        m1 = np.average(x[t], weights=w[t]); m0 = np.average(x[~t], weights=w[~t])
        s = np.sqrt((x[t].var() + x[~t].var()) / 2)
        out[c] = (m1 - m0) / s if s > 0 else 0.0
    return pd.Series(out)


def ipw_aipw(df: pd.DataFrame, outcome: str, treatment: str, covariates: list[str],
             trim: tuple[float, float] = (0.02, 0.98), n_boot: int = 300, seed: int = 0) -> dict:
    """ATE via stabilized IPW and cross-fitted AIPW (doubly robust).
    Bootstrap CIs. Returns balance before/after weighting."""
    X = _design(df, covariates)
    y = df[outcome].to_numpy(float)
    t = df[treatment].to_numpy(bool)

    def estimate(idx, return_parts=False):
        Xi, yi, ti = X.iloc[idx].to_numpy(), y[idx], t[idx]
        ps = LogisticRegression(max_iter=2000).fit(Xi, ti).predict_proba(Xi)[:, 1]
        ps = np.clip(ps, *trim)
        pt = ti.mean()
        w = np.where(ti, pt / ps, (1 - pt) / (1 - ps))           # stabilized weights
        ipw = np.average(yi[ti], weights=w[ti]) - np.average(yi[~ti], weights=w[~ti])
        # AIPW with 2-fold cross-fitted outcome models
        mu1, mu0 = np.zeros(len(yi)), np.zeros(len(yi))
        for tr, te in KFold(2, shuffle=True, random_state=seed).split(Xi):
            for arm, mu in ((True, mu1), (False, mu0)):
                m = tr[ti[tr] == arm]
                clf = HistGradientBoostingClassifier(max_iter=150, max_depth=3, random_state=seed)
                mu[te] = clf.fit(Xi[m], yi[m]).predict_proba(Xi[te])[:, 1]
        aipw = np.mean(mu1 - mu0 + ti * (yi - mu1) / ps - (~ti) * (yi - mu0) / (1 - ps))
        if return_parts:
            return ipw, aipw, ps, w
        return ipw, aipw

    ipw, aipw, ps, w = estimate(np.arange(len(df)), return_parts=True)
    rng = np.random.default_rng(seed)
    boots = np.array([estimate(rng.integers(0, len(df), len(df))) for _ in range(n_boot)])
    naive = y[t].mean() - y[~t].mean()
    return {
        "naive": naive, "ipw": ipw, "aipw": aipw,
        "ipw_ci": np.quantile(boots[:, 0], [0.025, 0.975]).tolist(),
        "aipw_ci": np.quantile(boots[:, 1], [0.025, 0.975]).tolist(),
        "balance_before": std_mean_diff(X, t), "balance_after": std_mean_diff(X, t, w),
        "propensity": ps, "treated": t,
    }


def e_value(rr: float) -> float:
    """VanderWeele & Ding E-value: how strong (risk-ratio scale) an unmeasured
    confounder would need to be, with both treatment and outcome, to explain
    away the estimate."""
    rr = rr if rr >= 1 else 1 / rr
    return rr + np.sqrt(rr * (rr - 1))


# --------------------------------------------------------------------------- #
# Difference-in-differences
# --------------------------------------------------------------------------- #
def did(panel: pd.DataFrame, y: str = "wau_rate", unit: str = "market", time: str = "week",
        treated: str = "treated_market", post: str = "post") -> dict:
    """Two-way fixed effects DiD with SEs clustered by unit.
    (Single adoption date; for staggered rollouts use Callaway-Sant'Anna or
    Sun-Abraham -- TWFE can be biased there.)"""
    p = panel.copy()
    p["treat_post"] = p[treated] * p[post]
    m = smf.ols(f"{y} ~ treat_post + C({unit}) + C({time})", data=p).fit(
        cov_type="cluster", cov_kwds={"groups": p[unit]})
    naive_post = p.loc[(p[post] == 1) & (p[treated] == 1), y].mean() - p.loc[(p[post] == 1) & (p[treated] == 0), y].mean()
    pre_post = p.loc[(p[post] == 1) & (p[treated] == 1), y].mean() - p.loc[(p[post] == 0) & (p[treated] == 1), y].mean()
    return {"att": float(m.params["treat_post"]), "se": float(m.bse["treat_post"]),
            "ci_low": float(m.conf_int().loc["treat_post", 0]), "ci_high": float(m.conf_int().loc["treat_post", 1]),
            "p_value": float(m.pvalues["treat_post"]),
            "naive_post_comparison": float(naive_post), "naive_before_after": float(pre_post)}


def event_study(panel: pd.DataFrame, rollout_time: int, y: str = "wau_rate", unit: str = "market",
                time: str = "week", treated: str = "treated_market", ref: int = -1) -> pd.DataFrame:
    """Leads/lags around rollout. Flat pre-period coefficients support parallel trends."""
    p = panel.copy()
    p["rel"] = p[time] - rollout_time
    rels = sorted(r for r in p["rel"].unique() if r != ref)
    terms = []
    for r in rels:
        name = f"rel_m{abs(r)}" if r < 0 else f"rel_p{r}"
        p[name] = ((p["rel"] == r) & (p[treated] == 1)).astype(int)
        terms.append((r, name))
    m = smf.ols(f"{y} ~ {' + '.join(n for _, n in terms)} + C({unit}) + C({time})", data=p).fit(
        cov_type="cluster", cov_kwds={"groups": p[unit]})
    ci = m.conf_int()
    rows = [{"rel_week": r, "coef": m.params[n], "ci_low": ci.loc[n, 0], "ci_high": ci.loc[n, 1]} for r, n in terms]
    rows.append({"rel_week": ref, "coef": 0.0, "ci_low": 0.0, "ci_high": 0.0})
    return pd.DataFrame(rows).sort_values("rel_week").reset_index(drop=True)
