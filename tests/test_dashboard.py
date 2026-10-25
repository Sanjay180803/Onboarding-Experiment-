"""Dashboard tests: every page renders without exceptions (Streamlit AppTest,
headless), and the calculators reproduce the analysis exactly at defaults."""
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "reports" / "results.json"
needs_outputs = pytest.mark.skipif(not RESULTS.exists(), reason="run the pipeline first")


@needs_outputs
def test_impact_calculator_matches_report():
    from dashboard import data as D
    R = json.loads(RESULTS.read_text())
    imp, P = R["impact"], R["primary"]
    for name, link in imp["links"].items():
        t = D.impact(P["ci_low"], P["diff"], P["ci_high"], imp["annual_signups"], imp["rollout_share"],
                     link, imp["ltv_per_payer_assumption"])
        for i, k in enumerate(["low", "point", "high"]):
            assert t.loc[i, "revenue"] == pytest.approx(imp["scenarios"][f"{name}_{k}"]["revenue"], rel=1e-12)


@needs_outputs
def test_power_calculator_matches_preregistration():
    from dashboard import data as D
    d = json.loads((ROOT / "reports" / "design.json").read_text())
    out = D.plan(D.PlanInputs(d["baseline_activation_7d"], d["mde_abs"], d["alpha"], d["power"], d["daily_signups"]))
    assert out["n_per_arm"] == d["n_per_arm"]
    assert out["runtime_days"] == d["runtime_days"]


@needs_outputs
def test_metric_table_verdicts_respect_direction():
    from dashboard import data as D
    mt = D.metric_table(json.loads(RESULTS.read_text()), D.load_catalog()).set_index("id")
    assert mt.loc["activation_7d", "verdict"] == "better"
    # tickets went up significantly; for a "decrease" metric that is "worse", not "better"
    assert mt.loc["support_ticket_7d", "verdict"] == "worse"
    assert mt.loc["d30_retained", "verdict"] == "no detectable change"


PAGES = ["Decision summary", "Metric results", "Funnel & retention", "Segments & uplift",
         "Power calculator", "Business impact", "Causal deep-dives", "Validation vs truth",
         "Metric definitions"]


@needs_outputs
@pytest.mark.parametrize("page", PAGES)
def test_page_renders(page):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "dashboard" / "app.py"), default_timeout=60)
    at.run()
    assert not at.exception, at.exception
    at.sidebar.radio(key="page").set_value(page).run()
    assert not at.exception, f"{page}: {at.exception}"
    assert at.title[0].value == page or page == "Decision summary"


@needs_outputs
def test_calculator_inputs_update_outputs():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "dashboard" / "app.py"), default_timeout=60)
    at.run()
    at.sidebar.radio(key="page").set_value("Power calculator").run()
    before = at.metric[0].value
    at.slider(key="mde").set_value(1.0).run()   # halve the MDE: 2pp -> 1pp
    assert not at.exception
    after = at.metric[0].value
    n_before, n_after = (int(v.replace(",", "")) for v in (before, after))
    assert 3.5 < n_after / n_before < 4.5      # n scales ~ 1 / MDE^2


@needs_outputs
def test_power_page_defaults_show_preregistered_sample_size():
    from streamlit.testing.v1 import AppTest
    d = json.loads((ROOT / "reports" / "design.json").read_text())
    at = AppTest.from_file(str(ROOT / "dashboard" / "app.py"), default_timeout=60)
    at.run()
    at.sidebar.radio(key="page").set_value("Power calculator").run()
    assert at.metric[0].value == f"{d['n_per_arm']:,}"
    assert at.metric[2].value == f"{d['runtime_days']} days"
