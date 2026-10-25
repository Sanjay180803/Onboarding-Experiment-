"""The metric catalog must stay in sync with the SQL and the analysis outputs."""
import json
import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
CATALOG = yaml.safe_load((ROOT / "metrics" / "catalog.yml").read_text())
SQL = (ROOT / "sql" / "01_user_metrics.sql").read_text()
REQUIRED = {"id", "name", "tier", "version", "definition", "numerator", "denominator", "unit",
            "desired_direction", "sql_column", "results_key", "analysis", "owner", "stakeholders",
            "decision_use"}


def _sql_columns() -> set[str]:
    return set(re.findall(r"\bAS\s+([a-z_][a-z0-9_]*)", SQL, flags=re.I)) | {"treat"}


def test_every_metric_has_required_fields():
    for m in CATALOG["metrics"]:
        missing = REQUIRED - m.keys()
        assert not missing, f"{m.get('id')}: missing {missing}"


def test_exactly_one_primary_metric_and_at_least_one_guardrail():
    tiers = [m["tier"] for m in CATALOG["metrics"]]
    assert tiers.count("primary") == 1
    assert tiers.count("guardrail") >= 1


def test_ids_are_unique():
    ids = [m["id"] for m in CATALOG["metrics"]]
    assert len(ids) == len(set(ids))


@pytest.mark.parametrize("metric", CATALOG["metrics"], ids=lambda m: m["id"])
def test_sql_column_exists(metric):
    cols = metric["sql_column"] if isinstance(metric["sql_column"], list) else [metric["sql_column"]]
    for c in cols:
        assert c in _sql_columns(), f"{metric['id']}: column {c} not produced by sql/01_user_metrics.sql"


@pytest.mark.parametrize("metric", CATALOG["metrics"], ids=lambda m: m["id"])
def test_results_key_exists(metric):
    path = ROOT / "reports" / "results.json"
    if not path.exists():
        pytest.skip("run the pipeline first")
    node = json.loads(path.read_text())
    for part in metric["results_key"].split("."):
        assert part in node, f"{metric['id']}: {metric['results_key']} not in results.json"
        node = node[part]


def test_catalog_version_matches_latest_changelog_entry():
    assert float(CATALOG["catalog_version"]) == max(float(c["version"]) for c in CATALOG["changelog"])
