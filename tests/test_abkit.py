"""Statistical correctness tests: each method is checked against simulation
or a known answer, not just "does it run"."""
import numpy as np
import pandas as pd
import pytest

from abkit import power, inference, variance_reduction, sequential, causal


def test_sample_size_matches_textbook():
    # 22% baseline, +2pp absolute, alpha .05, power .8 -> ~6.9k per arm
    n = power.n_per_arm_proportions(0.22, 0.02)
    assert 6800 < n < 7100


def test_analytic_power_matches_simulation():
    n = power.n_per_arm_proportions(0.22, 0.02)
    assert power.power_proportions(0.22, 0.24, n) == pytest.approx(0.80, abs=0.01)
    assert power.simulate_power(0.22, 0.24, n, sims=6000) == pytest.approx(0.80, abs=0.025)


def test_mde_inverts_sample_size():
    n = power.n_per_arm_proportions(0.22, 0.02)
    assert power.mde_proportions(0.22, n) == pytest.approx(0.02, abs=0.0005)


def test_runtime_rounds_to_full_weeks():
    assert power.runtime_days(7000, 1000) == 14
    assert power.runtime_days(7000, 1300) == 14  # 11 days -> 2 weeks


def test_srm_flags_broken_split():
    assert not inference.srm_test([5000, 5030])["srm_detected"]
    assert inference.srm_test([5000, 5400])["srm_detected"]


def test_ci_coverage_two_proportions():
    rng = np.random.default_rng(1)
    covered = 0
    for _ in range(1000):
        xt, xc = rng.binomial(5000, 0.24), rng.binomial(5000, 0.22)
        r = inference.two_proportion_test(xt, 5000, xc, 5000)
        covered += r["ci_low"] <= 0.02 <= r["ci_high"]
    assert 0.93 < covered / 1000 < 0.97


def test_delta_method_coverage_ratio_metric():
    rng = np.random.default_rng(2)
    true_ratio_diff = 0.0
    covered = 0
    for _ in range(400):
        den_t, den_c = rng.poisson(5, 3000) + 1, rng.poisson(5, 3000) + 1
        num_t = rng.poisson(den_t * 1.8 * rng.gamma(2, 0.5, 3000))  # user-level clustering
        num_c = rng.poisson(den_c * 1.8 * rng.gamma(2, 0.5, 3000))
        r = inference.delta_ratio_test(num_t, den_t, num_c, den_c)
        covered += r["ci_low"] <= true_ratio_diff <= r["ci_high"]
    assert 0.92 < covered / 400 < 0.98


def test_cuped_unbiased_and_reduces_variance():
    rng = np.random.default_rng(3)
    ests_raw, ests_adj = [], []
    for _ in range(300):
        x = rng.normal(size=4000)
        t = rng.random(4000) < 0.5
        y = 0.6 * x + rng.normal(size=4000) + 0.1 * t
        r = variance_reduction.cuped(y, x, t)
        ests_raw.append(r["raw_diff"]); ests_adj.append(r["adj_diff"])
    assert np.mean(ests_adj) == pytest.approx(0.1, abs=0.01)
    assert np.var(ests_adj) < 0.8 * np.var(ests_raw)


def test_peeking_inflates_fpr_and_msprt_controls_it():
    r = sequential.simulate_peeking(sims=3000, seed=4)
    assert r["fixed_horizon"] == pytest.approx(0.05, abs=0.015)
    assert r["naive_peeking"] > 0.12
    assert r["msprt"] <= 0.055


def test_iv_recovers_cace_when_naive_is_biased():
    rng = np.random.default_rng(5)
    n = 40000
    u = rng.normal(size=n)
    z = rng.random(n) < 0.5
    complier = rng.random(n) < 1 / (1 + np.exp(-(0.3 + u)))
    d = z & complier
    y = (rng.random(n) < 0.2 + 0.08 * u.clip(-1, 1) + 0.05 * d).astype(int)
    r = causal.iv_cace(y, z, d)
    assert r["ci_low"] < 0.05 < r["ci_high"]
    assert r["as_treated_naive"] > 0.08  # confounded by u


def test_did_recovers_effect():
    rng = np.random.default_rng(6)
    rows = []
    for m in range(10):
        a = rng.normal(0, 0.03) + 0.05 * (m < 5)
        for w in range(1, 17):
            rows.append(dict(market=m, week=w, treated_market=int(m < 5), post=int(w >= 9),
                             wau_rate=0.3 + a + 0.01 * np.sin(w) + 0.02 * (m < 5) * (w >= 9) + rng.normal(0, 0.004)))
    r = causal.did(pd.DataFrame(rows))
    assert r["ci_low"] < 0.02 < r["ci_high"]
    assert r["naive_post_comparison"] > 0.05


def test_e_value_known_value():
    # VanderWeele & Ding: RR=2 -> E-value 3.41
    assert causal.e_value(2.0) == pytest.approx(3.414, abs=0.01)


def _null_panel(rng, G=12, T=16, rollout=9, effect=0.0, ar=0.0):
    rows = []
    for m in range(G):
        a, e = rng.normal(0, 0.03), 0.0
        for w in range(1, T + 1):
            e = ar * e + rng.normal(0, 0.006)            # serially correlated noise
            rows.append(dict(market=m, week=w, treated_market=int(m < G // 2), post=int(w >= rollout),
                             wau_rate=0.3 + a + 0.01 * np.sin(w) + effect * (m < G // 2) * (w >= rollout) + e))
    return pd.DataFrame(rows)


def test_twfe_fast_path_matches_statsmodels():
    p = _null_panel(np.random.default_rng(7), effect=0.02)
    r, w = causal.did(p), causal.did_wild_cluster_bootstrap(p, ci=False)
    assert w["att"] == pytest.approx(r["att"], abs=1e-12)
    assert w["cr1_se"] == pytest.approx(r["se"], rel=1e-9)


def test_few_cluster_methods_recover_effect():
    p = _null_panel(np.random.default_rng(8), effect=0.02, ar=0.5)
    w = causal.did_wild_cluster_bootstrap(p)
    r = causal.did_randomization_inference(p)
    assert w["ci_low"] < 0.02 < w["ci_high"]
    assert r["ci_low"] < 0.02 < r["ci_high"]
    assert r["n_assignments"] == 924 and r["min_attainable_p"] == pytest.approx(2 / 924)


def test_wild_bootstrap_and_ri_hold_size_under_null():
    rng = np.random.default_rng(9)
    rej_w = rej_r = 0
    sims = 150
    for _ in range(sims):
        p = _null_panel(rng, ar=0.7)
        rej_w += causal.did_wild_cluster_bootstrap(p, ci=False)["p_value"] < 0.05
        rej_r += causal.did_randomization_inference(p, grid_size=3)["p_value"] < 0.05
    assert rej_w / sims < 0.10
    assert rej_r / sims < 0.10
