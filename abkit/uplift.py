"""Heterogeneous treatment effects with a cross-fitted T-learner.

Fit separate outcome models on treatment and control, predict both potential
outcomes for every user, CATE = mu1 - mu0. Cross-fitting means each user's
CATE comes from models that never saw that user, so the decile evaluation is
honest.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import KFold


def t_learner_cate(X: pd.DataFrame, y, t, n_splits: int = 5, seed: int = 0) -> np.ndarray:
    X = pd.get_dummies(X, drop_first=True).astype(float).to_numpy()
    y, t = np.asarray(y, int), np.asarray(t, bool)
    cate = np.zeros(len(y))
    for tr, te in KFold(n_splits, shuffle=True, random_state=seed).split(X):
        preds = []
        for arm in (True, False):
            m = tr[t[tr] == arm]
            clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_depth=3,
                                                 min_samples_leaf=100, random_state=seed)
            preds.append(clf.fit(X[m], y[m]).predict_proba(X[te])[:, 1])
        cate[te] = preds[0] - preds[1]
    return cate


def uplift_by_quantile(cate, y, t, q: int = 5) -> pd.DataFrame:
    """Observed treatment effect within bins of predicted CATE (highest first)."""
    df = pd.DataFrame({"cate": cate, "y": np.asarray(y, float), "t": np.asarray(t, bool)})
    df["bin"] = pd.qcut(df["cate"].rank(method="first", ascending=False), q, labels=range(1, q + 1))
    rows = []
    for b, g in df.groupby("bin", observed=True):
        a, c = g.loc[g.t, "y"], g.loc[~g.t, "y"]
        diff = a.mean() - c.mean()
        se = np.sqrt(a.var(ddof=1) / len(a) + c.var(ddof=1) / len(c))
        rows.append({"bin": int(b), "n": len(g), "pred_cate": g["cate"].mean(),
                     "observed_effect": diff, "ci_low": diff - 1.96 * se, "ci_high": diff + 1.96 * se})
    return pd.DataFrame(rows)


def qini_coefficient(cate, y, t) -> float:
    """Area between the Qini curve and random targeting (normalized by n)."""
    order = np.argsort(-np.asarray(cate))
    y, t = np.asarray(y, float)[order], np.asarray(t, bool)[order]
    ct, cc = np.cumsum(t), np.cumsum(~t)
    yt, yc = np.cumsum(y * t), np.cumsum(y * ~t)
    qini = yt - yc * np.divide(ct, cc, out=np.zeros_like(ct, float), where=cc > 0)
    random = qini[-1] * np.arange(1, len(y) + 1) / len(y)
    return float(np.mean(qini - random) / len(y))


def spearman(a, b) -> float:
    return float(stats.spearmanr(a, b).statistic)
