"""Compare regenerated results against a committed copy, number by number.

Usage: python tools/compare_results.py COMMITTED.json REGENERATED.json [--tol 1e-9]

Prints every numeric leaf whose value changed by more than `tol`, the largest
absolute and relative change, and GitHub Actions annotations so the summary is
visible on the run page. Exits 0 always: tiny cross-platform floating-point
drift is expected (different CPUs / BLAS kernels); the workflow's ground-truth
check is the hard gate.
"""
import argparse
import json
import math
import os


def leaves(obj, path=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from leaves(v, f"{path}.{k}" if path else k)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from leaves(v, f"{path}[{i}]")
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        yield path, float(obj)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("committed")
    ap.add_argument("regenerated")
    ap.add_argument("--tol", type=float, default=1e-9)
    a = ap.parse_args()
    old = dict(leaves(json.load(open(a.committed))))
    new = dict(leaves(json.load(open(a.regenerated))))
    gh = os.environ.get("GITHUB_ACTIONS") == "true"
    name = os.path.basename(a.regenerated)

    changed = []
    for k in sorted(old.keys() & new.keys()):
        o, n = old[k], new[k]
        if math.isnan(o) and math.isnan(n):
            continue
        d = abs(n - o)
        if d > a.tol:
            changed.append((k, o, n, d, d / abs(o) if o else math.inf))
    missing = sorted(old.keys() ^ new.keys())

    print(f"{name}: {len(old.keys() & new.keys())} numeric values compared, "
          f"{len(changed)} changed by more than {a.tol:g}, {len(missing)} keys added/removed")
    top = sorted(changed, key=lambda r: -r[3])[:15]
    for k, o, n, d, r in top:
        print(f"  {k}: {o:.10g} -> {n:.10g}  (abs {d:.3g}, rel {r:.3g})")
    if gh:
        if changed:
            k, o, n, d, r = top[0]
            print(f"::notice title={name} drift::{len(changed)} values differ from the committed version; "
                  f"largest abs change {d:.3g} at {k} ({o:.6g} -> {n:.6g}); "
                  f"largest relative change {max(c[4] for c in changed):.3g}")
            for k, o, n, d, r in top[:5]:
                print(f"::notice title={name}::{k}: {o:.6g} -> {n:.6g} (abs {d:.2g})")
        else:
            print(f"::notice title={name}::identical to the committed version (tol {a.tol:g})")
        if missing:
            print(f"::warning title={name}::keys added/removed: {missing[:10]}")


if __name__ == "__main__":
    main()
