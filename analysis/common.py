"""Shared helpers: build the DuckDB metric tables and a consistent chart style."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DATA, SQL, REPORTS = ROOT / "data", ROOT / "sql", ROOT / "reports"
FIG = REPORTS / "figures"

# Fixed categorical order: color follows the entity, never its rank.
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
C_CONTROL, C_TREAT, C_THIRD, C_FOURTH = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"
VARIANT_COLOR = {"control": C_CONTROL, "treatment": C_TREAT}


def connect() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    for t in ("users", "events", "assignments"):
        con.execute(f"CREATE VIEW {t} AS SELECT * FROM read_parquet('{DATA / (t + '.parquet')}')")
    con.execute((SQL / "01_user_metrics.sql").read_text())
    return con


def run_sql(con, name: str, **params):
    return con.sql((SQL / name).read_text().format(**params)).df()


def style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
        "axes.titleweight": "bold", "axes.titlesize": 12, "axes.titlelocation": "left",
        "font.size": 10, "legend.frameon": False, "lines.linewidth": 2, "figure.dpi": 130,
    })


def save(fig, name: str):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG / name, bbox_inches="tight")
    plt.close(fig)


def dump_json(obj, path: Path):
    def default(o):
        if hasattr(o, "item"):
            return o.item()
        if hasattr(o, "tolist"):
            return o.tolist()
        return str(o)
    path.write_text(json.dumps(obj, indent=2, default=default))
