"""CUPED and CUPAC variance reduction.

CUPED (Deng et al., 2013): y_adj = y - theta * (x - mean(x)), theta = cov(y,x)/var(x),
where x is measured BEFORE treatment, so it cannot be affected by it. The
estimate stays unbiased and variance falls by corr(x, y)^2.

CUPAC (DoorDash, 2020): use an ML prediction of y from pre-treatment features as
the control variate x. The model is trained on a PRE-EXPERIMENT cohort, so no
experiment outcome leaks into the covariate.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor


def cuped(y, x, treat, alpha: float = 0.05) -> dict:
    y, x, treat = np.asarray(y, float), np.asarray(x, float), np.asarray(treat, bool)
    theta = np.cov(y, x, ddof=1)[0, 1] / x.var(ddof=1)
    y_adj = y - theta * (x - x.mean())

    def diff_se(v):
        a, b = v[treat], v[~treat]
        return a.mean() - b.mean(), np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))

    d_raw, se_raw = diff_se(y)
    d_adj, se_adj = diff_se(y_adj)
    zc = stats.norm.ppf(1 - alpha / 2)
    return {
        "theta": float(theta), "corr": float(np.corrcoef(y, x)[0, 1]),
        "raw_diff": d_raw, "raw_se": se_raw, "raw_ci": [d_raw - zc * se_raw, d_raw + zc * se_raw],
        "adj_diff": d_adj, "adj_se": se_adj, "adj_ci": [d_adj - zc * se_adj, d_adj + zc * se_adj],
        "adj_p_value": float(2 * stats.norm.sf(abs(d_adj / se_adj))),
        "variance_reduction": float(1 - (se_adj / se_raw) ** 2),
    }


def fit_cupac_model(hist: pd.DataFrame, features: list[str], target: str, seed: int = 0):
    X = pd.get_dummies(hist[features], drop_first=False)
    model = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_depth=4, random_state=seed)
    model.fit(X, hist[target])
    return model, X.columns


def cupac_covariate(model, columns, df: pd.DataFrame, features: list[str]) -> np.ndarray:
    X = pd.get_dummies(df[features], drop_first=False).reindex(columns=columns, fill_value=0)
    return model.predict(X)
