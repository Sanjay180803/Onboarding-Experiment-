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
             trim: tuple[float, float] = (0.02, 0.98), n_boot: int = 300, seed: int = 0,
             n_folds: int = 2) -> dict:
    """ATE via stabilized IPW and doubly robust AIPW, with bootstrap CIs.

    Nuisance-model fitting, stated precisely:
      - IPW: logistic propensity model fitted on the full analyzed sample
        (in-sample, the standard IPW estimator; also used for the balance check).
      - AIPW: BOTH nuisance models are cross-fitted (K folds). For each fold,
        the propensity model (logistic) and the two outcome models
        (gradient boosting, one per arm) are fitted on the other folds and
        predict on the held-out fold, so no unit's nuisance predictions come
        from a model that saw that unit.
    """
    X = _design(df, covariates)
    y = df[outcome].to_numpy(float)
    t = df[treatment].to_numpy(bool)

    def estimate(idx, return_parts=False):
        Xi, yi, ti = X.iloc[idx].to_numpy(), y[idx], t[idx]
        # --- IPW: full-sample propensity -----------------------------------
        ps_full = np.clip(LogisticRegression(max_iter=2000).fit(Xi, ti).predict_proba(Xi)[:, 1], *trim)
        pt = ti.mean()
        w = np.where(ti, pt / ps_full, (1 - pt) / (1 - ps_full))   # stabilized weights
        ipw = np.average(yi[ti], weights=w[ti]) - np.average(yi[~ti], weights=w[~ti])
        # --- AIPW: cross-fitted propensity AND outcome models ---------------
        ps_cf, mu1, mu0 = np.zeros(len(yi)), np.zeros(len(yi)), np.zeros(len(yi))
        for tr, te in KFold(n_folds, shuffle=True, random_state=seed).split(Xi):
            ps_cf[te] = LogisticRegression(max_iter=2000).fit(Xi[tr], ti[tr]).predict_proba(Xi[te])[:, 1]
            for arm, mu in ((True, mu1), (False, mu0)):
                m = tr[ti[tr] == arm]
                clf = HistGradientBoostingClassifier(max_iter=150, max_depth=3, random_state=seed)
                mu[te] = clf.fit(Xi[m], yi[m]).predict_proba(Xi[te])[:, 1]
        ps_cf = np.clip(ps_cf, *trim)
        aipw = np.mean(mu1 - mu0 + ti * (yi - mu1) / ps_cf - (~ti) * (yi - mu0) / (1 - ps_cf))
        ps = ps_full
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


# --------------------------------------------------------------------------- #
# Few-cluster inference for DiD
# --------------------------------------------------------------------------- #
# With only ~12 clusters, the usual cluster-robust (CR1) standard errors are
# downward-biased and their t-tests over-reject (Cameron, Gelbach & Miller 2008).
# Two standard sensitivity checks, both computed EXACTLY by full enumeration:
#   * wild cluster bootstrap-t with the null imposed (WCR), Rademacher weights
#   * randomization inference over every possible choice of treated units
# For a balanced panel, the TWFE coefficient equals a regression of two-way
# demeaned Y on two-way demeaned D (Frisch-Waugh-Lovell), which makes both fast.
from itertools import combinations, product


def _panel_arrays(panel, y, unit, time, treated, post):
    p = panel.sort_values([unit, time])
    units, times = p[unit].unique(), p[time].unique()
    G, T = len(units), len(times)
    if len(p) != G * T:
        raise ValueError("few-cluster DiD helpers require a balanced unit x time panel")
    Y = p[y].to_numpy(float).reshape(G, T)
    tr = p[treated].to_numpy().reshape(G, T)[:, 0].astype(bool)
    po = p[post].to_numpy().reshape(G, T)[0].astype(bool)
    return Y, tr, po


def _demean2(x):
    """Two-way demean over the last two axes (units, times)."""
    return x - x.mean(-1, keepdims=True) - x.mean(-2, keepdims=True) + x.mean((-2, -1), keepdims=True)


def _twfe_beta_se(Y, D):
    """TWFE estimate and CR1 cluster-robust SE. Y may carry leading batch dims."""
    G, T = D.shape
    dd = _demean2(D.astype(float))
    sdd = (dd ** 2).sum()
    dy = _demean2(Y)
    beta = (dy * dd).sum((-2, -1)) / sdd
    e = dy - beta[..., None, None] * dd
    score = (dd * e).sum(-1)                       # per-cluster score, shape (..., G)
    k = 1 + 1 + (G - 1) + (T - 1)                  # intercept + treat_post + unit FE + time FE (matches statsmodels)
    n = G * T
    adj = G / (G - 1) * (n - 1) / (n - k)
    se = np.sqrt(adj * (score ** 2).sum(-1)) / sdd
    return beta, se


def did_wild_cluster_bootstrap(panel, y="wau_rate", unit="market", time="week",
                               treated="treated_market", post="post",
                               ci: bool = True, alpha: float = 0.05, grid_size: int = 401) -> dict:
    """Wild cluster restricted bootstrap-t, enumerating all 2^G Rademacher draws.
    p-value for H0: beta = 0, and a CI by test inversion over a grid of nulls."""
    Y, tr, po = _panel_arrays(panel, y, unit, time, treated, post)
    G, T = Y.shape
    if G > 16:
        raise ValueError("full enumeration is for few clusters; sample weights instead")
    D = np.outer(tr, po).astype(float)
    W = np.array(list(product([-1.0, 1.0], repeat=G)))          # (2^G, G)
    beta, se = _twfe_beta_se(Y, D)

    def pval(b0):
        y0 = Y - b0 * D
        r = _demean2(y0)                     # residuals of the restricted (FE-only) model
        fitted = y0 - r
        ystar = b0 * D + fitted + W[:, :, None] * r          # (2^G, G, T)
        bs, ss = _twfe_beta_se(ystar, D)
        t_obs = (beta - b0) / se
        return float(np.mean(np.abs((bs - b0) / ss) >= abs(t_obs) - 1e-12))

    out = {"att": float(beta), "cr1_se": float(se), "n_clusters": G,
           "n_draws": len(W), "p_value": pval(0.0),
           # Rademacher draws come in +/- pairs, so p can never go below 2 / 2^G
           "min_attainable_p": 2 / len(W)}
    if ci:
        grid = np.linspace(beta - 8 * se, beta + 8 * se, grid_size)
        keep = grid[[pval(b) > alpha for b in grid]]
        out.update({"ci_low": float(keep.min()), "ci_high": float(keep.max())})
    return out


def did_randomization_inference(panel, y="wau_rate", unit="market", time="week",
                                treated="treated_market", post="post",
                                alpha: float = 0.05, grid_size: int = 401) -> dict:
    """Exact randomization (permutation) inference: re-run the TWFE DiD for every
    way of choosing the same number of treated units. Tests the sharp null of no
    effect for any market-week, treating assignment as if random across markets.
    A CI comes from inverting the test under constant-effect sharp nulls."""
    Y, tr, po = _panel_arrays(panel, y, unit, time, treated, post)
    G, T = Y.shape
    n_t = int(tr.sum())
    combos = list(combinations(range(G), n_t))
    Dk = np.zeros((len(combos), G, T))
    for i, c in enumerate(combos):
        Dk[i, list(c)] = po
    D_obs = np.outer(tr, po).astype(float)
    ddk = _demean2(Dk)
    sdd = (ddk ** 2).sum((-2, -1))
    dY, dDobs = _demean2(Y), _demean2(D_obs)
    b_y = (ddk * dY).sum((-2, -1)) / sdd            # each assignment's estimate on Y
    b_d = (ddk * dDobs).sum((-2, -1)) / sdd         # ... and on the observed D (for inversion)
    obs_idx = combos.index(tuple(np.flatnonzero(tr)))
    beta = b_y[obs_idx]

    def pval(b0):
        stat = b_y - b0 * b_d                        # estimates on Y - b0 * D_obs
        return float(np.mean(np.abs(stat) >= abs(stat[obs_idx]) - 1e-12))

    grid = np.linspace(beta - 5 * abs(beta) - 1e-3, beta + 5 * abs(beta) + 1e-3, grid_size)
    keep = grid[[pval(b) > alpha for b in grid]]
    return {"att": float(beta), "n_assignments": len(combos), "n_treated": n_t, "n_clusters": G,
            "p_value": pval(0.0),
            "min_attainable_p": float(np.mean(np.abs(b_y) >= np.abs(b_y).max() - 1e-12)),
            "ci_low": float(keep.min()) if len(keep) else None,
            "ci_high": float(keep.max()) if len(keep) else None,
            "placebo_estimates": b_y}


def simulate_did_size(n_treated_list=(6, 2), G: int = 12, T: int = 20, rollout: int = 12,
                      sims: int = 300, ar: float = 0.7, sd_treated: float = 0.012,
                      sd_control: float = 0.005, alpha: float = 0.05, seed: int = 3) -> list[dict]:
    """Null Monte Carlo (true effect = 0) of DiD panels shaped like ours, with
    serially correlated noise and noisier treated units. Returns the share of
    false rejections for CR1 cluster-robust SEs, the wild cluster bootstrap and
    randomization inference, for each number of treated clusters."""
    out = []
    for n_tr in n_treated_list:
        rng = np.random.default_rng(seed)
        rej = {"cr1": 0, "wild_bootstrap": 0, "randomization_inference": 0}
        for _ in range(sims):
            rows = []
            for m in range(G):
                a, e = rng.normal(0, 0.02), 0.0
                sd = sd_treated if m < n_tr else sd_control
                for w in range(1, T + 1):
                    e = ar * e + rng.normal(0, sd)
                    rows.append((m, w, int(m < n_tr), int(w >= rollout), 0.31 + a + e))
            p = pd.DataFrame(rows, columns=["market", "week", "treated_market", "post", "wau_rate"])
            rej["cr1"] += did(p)["p_value"] < alpha
            rej["wild_bootstrap"] += did_wild_cluster_bootstrap(p, ci=False)["p_value"] < alpha
            rej["randomization_inference"] += did_randomization_inference(p, grid_size=3)["p_value"] < alpha
        out.append({"n_treated": n_tr, "n_clusters": G, "sims": sims,
                    **{k: v / sims for k, v in rej.items()}})
    return out
