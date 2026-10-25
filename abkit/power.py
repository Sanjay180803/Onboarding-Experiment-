"""Power analysis and experiment sizing."""
from __future__ import annotations

import math

import numpy as np
from scipy import stats
from scipy.optimize import brentq


def _z(alpha: float, two_sided: bool) -> float:
    return stats.norm.ppf(1 - alpha / 2) if two_sided else stats.norm.ppf(1 - alpha)


def n_per_arm_proportions(p0: float, mde_abs: float, alpha: float = 0.05,
                          power: float = 0.8, two_sided: bool = True) -> int:
    """Users per arm to detect an absolute lift `mde_abs` from baseline `p0`
    with a two-proportion z-test (pooled variance under H0, unpooled under H1)."""
    p1 = p0 + mde_abs
    pbar = (p0 + p1) / 2
    za, zb = _z(alpha, two_sided), stats.norm.ppf(power)
    num = (za * math.sqrt(2 * pbar * (1 - pbar)) + zb * math.sqrt(p0 * (1 - p0) + p1 * (1 - p1))) ** 2
    return math.ceil(num / mde_abs ** 2)


def power_proportions(p0: float, p1: float, n_per_arm: int, alpha: float = 0.05,
                      two_sided: bool = True) -> float:
    """Analytic power of the two-proportion z-test."""
    d = abs(p1 - p0)
    pbar = (p0 + p1) / 2
    se0 = math.sqrt(2 * pbar * (1 - pbar) / n_per_arm)
    se1 = math.sqrt((p0 * (1 - p0) + p1 * (1 - p1)) / n_per_arm)
    za = _z(alpha, two_sided)
    return float(stats.norm.cdf((d - za * se0) / se1))


def mde_proportions(p0: float, n_per_arm: int, alpha: float = 0.05, power: float = 0.8) -> float:
    """Smallest absolute lift detectable with the given sample size."""
    return brentq(lambda d: power_proportions(p0, p0 + d, n_per_arm, alpha) - power, 1e-5, 1 - p0 - 1e-6)


def simulate_power(p0: float, p1: float, n_per_arm: int, alpha: float = 0.05,
                   sims: int = 5000, seed: int = 0) -> float:
    """Monte-Carlo check of the analytic power formula."""
    rng = np.random.default_rng(seed)
    xc = rng.binomial(n_per_arm, p0, sims)
    xt = rng.binomial(n_per_arm, p1, sims)
    pc, pt = xc / n_per_arm, xt / n_per_arm
    pp = (xc + xt) / (2 * n_per_arm)
    se = np.sqrt(pp * (1 - pp) * 2 / n_per_arm)
    z = (pt - pc) / np.where(se == 0, np.inf, se)
    return float(np.mean(np.abs(z) > _z(alpha, True)))


def cuped_adjusted_n(n: int, rho: float) -> int:
    """CUPED with a covariate of correlation rho cuts variance by rho^2."""
    return math.ceil(n * (1 - rho ** 2))


def runtime_days(n_per_arm: int, daily_eligible: float, n_arms: int = 2,
                 round_to_weeks: bool = True) -> int:
    """Days needed to enroll n_per_arm in every arm; rounded up to full weeks
    so every weekday is represented equally (avoids day-of-week bias)."""
    days = math.ceil(n_per_arm * n_arms / daily_eligible)
    return 7 * math.ceil(days / 7) if round_to_weeks else days
