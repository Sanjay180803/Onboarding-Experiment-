"""
Synthetic product event-log simulator with KNOWN ground-truth causal effects.

Scenario
--------
"Habitly", a subscription habit-tracking app, tests a new guided onboarding
checklist (treatment) against the current onboarding flow (control).

The simulator produces:
  data/users.parquet        one row per signup (historical + experiment cohorts)
  data/assignments.parquet  experiment variant per experiment user
  data/events.parquet       raw event log (signup, sessions, funnel steps, revenue)
  data/market_weeks.parquet market x week panel for a separate rollout (DiD)
  data/_truth/              ground truth -- NEVER used by the analysis except
                            in the explicit "validate against truth" step

Every user gets potential outcomes under control and treatment generated with
COMMON random numbers, so the true (sample) causal effects are known exactly.

Causal structure (u = latent motivation, partly observed via proxies):
  u -> pre_visits, goal survey, compliance, activation, retention, purchase
  assignment Z -> checklist opened D (one-sided noncompliance, D depends on u)
  D -> onboarding completion -> activation -> retention / trial / purchase
  community opt-in (self-selected, confounded by pre_visits/goal/channel/u)
      -> retention
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit


@dataclass
class SimConfig:
    seed: int = 7
    hist_days: int = 28          # pre-experiment cohort (used for design/power)
    exp_days: int = 14           # experiment enrollment window (2 full weeks)
    signups_per_day: int = 1000
    horizon: int = 30            # days of follow-up per user
    n_markets: int = 12
    start_date: str = "2026-07-06"   # first historical signup day (a Monday)
    # treatment effects (on compliers, probability scale)
    ob_lift: float = 0.04            # onboarding completion lift
    tau_base: float = 0.025          # activation-given-onboarded lift, base
    tau_low_intent: float = 0.040    # extra lift for low-intent users
    tau_browsing: float = 0.020      # extra lift for "just browsing" goal
    community_effect: float = 0.35   # log-odds effect of community on daily activity
    did_effect: float = 0.020        # streak-reminder effect on weekly active rate


GOALS = np.array(["build_habit", "specific_goal", "just_browsing"])
PLATFORMS = np.array(["ios", "android", "web"])
CHANNELS = np.array(["organic", "paid_social", "referral"])


# --------------------------------------------------------------------------- #
# Users and covariates
# --------------------------------------------------------------------------- #
def make_users(rng: np.random.Generator, cfg: SimConfig) -> pd.DataFrame:
    days = np.arange(-cfg.hist_days, cfg.exp_days)
    # weekday seasonality in signups (+/- ~15%)
    n_by_day = rng.poisson(cfg.signups_per_day * (1 + 0.15 * np.sin(2 * np.pi * (days % 7) / 7)))
    signup_day = np.repeat(days, n_by_day)
    n = len(signup_day)

    u = rng.normal(size=n)
    platform = rng.choice(PLATFORMS, n, p=[0.45, 0.40, 0.15])
    channel = rng.choice(CHANNELS, n, p=[0.45, 0.40, 0.15])
    p_browse = expit(-1.1 - 0.9 * u)
    browse = rng.random(n) < p_browse
    goal = np.where(browse, "just_browsing",
                    np.where(rng.random(n) < 0.5, "build_habit", "specific_goal"))
    pre_visits = rng.poisson(np.exp(0.4 + 0.55 * u + 0.3 * (channel == "referral")))
    market = rng.integers(0, cfg.n_markets, n)

    start = pd.Timestamp(cfg.start_date) + pd.Timedelta(days=cfg.hist_days)  # day 0 = experiment start
    signup_ts = start + pd.to_timedelta(signup_day, unit="D") + pd.to_timedelta(rng.integers(0, 86400, n), unit="s")

    return pd.DataFrame({
        "user_id": np.arange(1, n + 1),
        "signup_ts": signup_ts,
        "signup_day": signup_day,
        "cohort": np.where(signup_day < 0, "historical", "experiment"),
        "platform": platform,
        "channel": channel,
        "goal": goal,
        "pre_visits": pre_visits,
        "market": market,
        "_u": u,
    })


# --------------------------------------------------------------------------- #
# Outcome model -- deterministic given probabilities + common random numbers
# --------------------------------------------------------------------------- #
@dataclass
class Draws:
    """Common random numbers shared by all potential-outcome worlds."""
    ob: np.ndarray
    act: np.ndarray
    act_day: np.ndarray
    comply: np.ndarray
    join: np.ndarray
    active: np.ndarray        # (n, horizon+1)
    sessions: np.ndarray      # (n, horizon+1)
    trial: np.ndarray
    buy: np.ndarray
    plan: np.ndarray
    ticket: np.ndarray


def draw(rng, n, horizon) -> Draws:
    return Draws(
        ob=rng.random(n), act=rng.random(n), act_day=rng.integers(0, 7, n),
        comply=rng.random(n), join=rng.random(n),
        active=rng.random((n, horizon + 1)),
        sessions=rng.poisson(0.7, (n, horizon + 1)),
        trial=rng.random(n), buy=rng.random(n), plan=rng.random(n), ticket=rng.random(n),
    )


def base_probs(df: pd.DataFrame, cfg: SimConfig) -> dict:
    u = df["_u"].to_numpy()
    web = (df["platform"] == "web").to_numpy()
    referral = (df["channel"] == "referral").to_numpy()
    paid = (df["channel"] == "paid_social").to_numpy()
    browse = (df["goal"] == "just_browsing").to_numpy()
    low_intent = (df["pre_visits"] <= 1).to_numpy()

    p_ob0 = expit(0.85 + 0.5 * u - 0.3 * web)
    p_act_ob0 = expit(-0.85 + 0.65 * u + 0.35 * referral - 0.25 * paid - 0.45 * browse)
    tau = cfg.tau_base + cfg.tau_low_intent * low_intent + cfg.tau_browsing * browse
    p_comply = expit(0.35 + 0.45 * u + 0.4 * web)
    p_join = expit(-2.3 + 0.32 * np.log1p(df["pre_visits"].to_numpy()) * 2
                   + 0.5 * referral + 0.45 * (~browse) + 0.15 * u)
    return dict(
        u=u, browse=browse,
        p_ob0=p_ob0, p_ob1=np.clip(p_ob0 + cfg.ob_lift, 0, 1),
        p_act_ob0=p_act_ob0, p_act_ob1=np.clip(p_act_ob0 + tau, 0, 1),
        p_comply=p_comply, p_join=p_join,
    )


def activation(P, d: Draws, treated_complier: np.ndarray):
    p_ob = np.where(treated_complier, P["p_ob1"], P["p_ob0"])
    p_ag = np.where(treated_complier, P["p_act_ob1"], P["p_act_ob0"])
    ob = d.ob < p_ob
    act = ob & (d.act < p_ag)
    return ob, act


def downstream(P, d: Draws, act: np.ndarray, join: np.ndarray, cfg: SimConfig) -> dict:
    """Retention, sessions, paywall, trial, purchase given activation + community."""
    u, browse = P["u"], P["browse"]
    days = np.arange(cfg.horizon + 1)
    logit = (-0.15 + 1.35 * act[:, None] + 0.5 * u[:, None] + cfg.community_effect * join[:, None]
             - 0.25 * browse[:, None] - 0.72 * np.log(np.maximum(days, 1))[None, :])
    active = d.active < expit(logit)
    active[:, 0] = True                                 # signup session
    sessions = np.where(active, 1 + d.sessions + (act[:, None] & (d.sessions > 0)), 0)

    paywall = active[:, 3:8].any(axis=1)
    trial = paywall & (d.trial < expit(-1.9 + 1.2 * act + 0.3 * u))
    paid = trial & (d.buy < expit(-0.6 + 0.8 * act + 0.4 * u))
    annual = d.plan < 0.3
    revenue = np.where(paid, np.where(annual, 79.99, 12.99), 0.0)
    return dict(active=active, sessions=sessions, paywall=paywall,
                trial=trial, paid=paid, annual=annual, revenue=revenue)


# --------------------------------------------------------------------------- #
# Event log
# --------------------------------------------------------------------------- #
def build_events(df, rng, out, ob, act, d: Draws, opened, join, ticket, cfg) -> pd.DataFrame:
    n = len(df)
    sig = df["signup_ts"].to_numpy()
    uid = df["user_id"].to_numpy()
    frames = []

    def add(mask, name, day_offset, revenue=None):
        idx = np.flatnonzero(mask)
        if len(idx) == 0:
            return
        off = np.asarray(day_offset)[idx] if np.ndim(day_offset) else np.full(len(idx), day_offset)
        secs = rng.integers(60, 3 * 3600, len(idx))
        ts = sig[idx] + (off * 86400 + secs).astype("timedelta64[s]")
        frames.append(pd.DataFrame({
            "user_id": uid[idx], "event_ts": ts, "event_name": name,
            "revenue": 0.0 if revenue is None else revenue[idx],
        }))

    add(np.ones(n, bool), "signup", 0)
    add(opened, "checklist_open", 0)
    add(join, "community_join", 0)
    add(ob, "onboarding_complete", 0)
    add(act, "core_action", d.act_day)
    add(ticket, "support_ticket", rng.integers(0, 7, n))

    # sessions: one row per session
    s = out["sessions"]
    rows, cols = np.nonzero(s)
    reps = s[rows, cols]
    r_u, r_d = np.repeat(rows, reps), np.repeat(cols, reps)
    ts = sig[r_u] + (r_d * 86400 + rng.integers(0, 86400, len(r_u))).astype("timedelta64[s]")
    frames.append(pd.DataFrame({"user_id": uid[r_u], "event_ts": ts,
                                "event_name": "session_start", "revenue": 0.0}))

    first_pw_day = np.where(out["paywall"], 3 + np.argmax(out["active"][:, 3:8], axis=1), 0)
    add(out["paywall"], "paywall_view", first_pw_day)
    add(out["trial"], "trial_start", first_pw_day)
    add(out["paid"], "purchase", first_pw_day + 7, revenue=out["revenue"])

    ev = pd.concat(frames, ignore_index=True)
    return ev.sort_values(["event_ts", "user_id"], kind="stable").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Market x week panel for the (non-randomized) streak-reminder rollout -> DiD
# --------------------------------------------------------------------------- #
def make_market_panel(rng, cfg: SimConfig) -> pd.DataFrame:
    weeks = np.arange(1, 21)
    rollout_week = 12
    m_effect = rng.normal(0, 0.02, cfg.n_markets)
    # ops chose the 6 strongest markets for rollout -> selection on LEVELS
    treated_markets = np.argsort(-m_effect)[:6]
    season = 0.015 * np.sin(2 * np.pi * weeks / 10) - 0.001 * weeks
    rows = []
    for m in range(cfg.n_markets):
        treated = m in treated_markets
        for w in weeks:
            post = w >= rollout_week
            wau = 0.31 + m_effect[m] + season[w - 1] + cfg.did_effect * (treated and post) + rng.normal(0, 0.005)
            users = int(rng.integers(3000, 9000))
            rows.append(dict(market=m, week=w, treated_market=int(treated), post=int(post),
                             active_users=int(round(wau * users)), users=users))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
def main(out_dir: Path, cfg: SimConfig):
    rng = np.random.default_rng(cfg.seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    truth_dir = out_dir / "_truth"
    truth_dir.mkdir(exist_ok=True)

    df = make_users(rng, cfg)
    n = len(df)
    P = base_probs(df, cfg)
    d = draw(rng, n, cfg.horizon)

    is_exp = (df["cohort"] == "experiment").to_numpy()
    z = np.zeros(n, bool)
    z[is_exp] = rng.random(is_exp.sum()) < 0.5          # 50/50 randomization
    complier = d.comply < P["p_comply"]                   # would open checklist if offered
    opened = z & complier
    join = d.join < P["p_join"]
    ticket = d.ticket < 0.03                              # guardrail: true effect = 0

    # --- potential outcomes (common random numbers) -------------------------
    ob0, act0 = activation(P, d, np.zeros(n, bool))
    ob1, act1 = activation(P, d, complier)                # world where everyone is assigned
    o0 = downstream(P, d, act0, join, cfg)
    o1 = downstream(P, d, act1, join, cfg)
    ob_obs, act_obs = activation(P, d, opened)
    obs = downstream(P, d, act_obs, join, cfg)
    # community counterfactuals (observed activation, flip community)
    oc1 = downstream(P, d, act_obs, np.ones(n, bool), cfg)
    oc0 = downstream(P, d, act_obs, np.zeros(n, bool), cfg)

    ev = build_events(df, rng, obs, ob_obs, act_obs, d, opened, join, ticket, cfg)

    # --- write observed data -------------------------------------------------
    users = df.drop(columns=["_u"])
    users.to_parquet(out_dir / "users.parquet", index=False)
    assign = pd.DataFrame({
        "user_id": df.loc[is_exp, "user_id"].to_numpy(),
        "variant": np.where(z[is_exp], "treatment", "control"),
        "assigned_ts": df.loc[is_exp, "signup_ts"].to_numpy(),
    })
    assign.to_parquet(out_dir / "assignments.parquet", index=False)
    ev.to_parquet(out_dir / "events.parquet", index=False)
    panel = make_market_panel(rng, cfg)
    panel.to_parquet(out_dir / "market_weeks.parquet", index=False)

    # --- ground truth (experiment cohort, sample estimands) ----------------
    e = is_exp
    def mean_diff(a, b, mask=e):
        return float(np.mean(a[mask].astype(float) - b[mask].astype(float)))

    p_c = P["p_comply"]
    exp_cate = p_c * (P["p_ob1"] * P["p_act_ob1"] - P["p_ob0"] * P["p_act_ob0"])
    truth = {
        "config": asdict(cfg),
        "n_users_total": int(n),
        "n_experiment": int(e.sum()),
        "experiment": {
            "control_activation_rate": float(act0[e].mean()),
            "itt_activation_7d": mean_diff(act1, act0),
            "complier_share": float(complier[e].mean()),
            "cace_activation_7d": float((act1[e & complier].astype(float) - act0[e & complier]).mean()),
            "itt_onboarding_complete": mean_diff(ob1, ob0),
            "itt_d7_retention": mean_diff(o1["active"][:, 7], o0["active"][:, 7]),
            "itt_d30_retention": mean_diff(o1["active"][:, 30], o0["active"][:, 30]),
            "itt_paid_30d": mean_diff(o1["paid"], o0["paid"]),
            "itt_revenue_30d": mean_diff(o1["revenue"], o0["revenue"]),
            "itt_support_ticket": 0.0,
        },
        "community": {
            "ate_d30_retention": float(np.mean(oc1["active"][:, 30].astype(float) - oc0["active"][:, 30])),
            "naive_note": "joiners differ systematically (pre_visits, goal, channel, u)",
        },
        "did": {"att_weekly_active_rate": cfg.did_effect},
    }
    (truth_dir / "ground_truth.json").write_text(json.dumps(truth, indent=2))
    pd.DataFrame({
        "user_id": df["user_id"], "u": P["u"], "complier": complier,
        "true_cate_activation": exp_cate,
    }).to_parquet(truth_dir / "user_truth.parquet", index=False)

    print(f"users={n:,}  experiment={e.sum():,}  events={len(ev):,}")
    print(json.dumps(truth["experiment"], indent=2))
    print("community ATE d30:", round(truth["community"]["ate_d30_retention"], 4))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    main(Path(args.out), SimConfig(seed=args.seed))
