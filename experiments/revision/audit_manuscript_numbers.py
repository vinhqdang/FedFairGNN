"""For every number printed in the manuscript, ask whether any artifact contains it.

A number that appears in no file under results/ was either typed by hand,
computed by hand, or carried over from a superseded run. That is exactly how the
705% / 2212% slack figures survived: they trace to no artifact. This does not
decide anything -- a MISS can be perfectly innocent (a budget, a count, a year) --
it produces the shortlist a human then has to look at.

Matching is on the DIGITS, ignoring the decimal separator and trailing zeros, so
0.0488 matches 0.04880000000001 in an artifact and 72.95 matches 0.7295 only if
--loose is given (percentages in prose vs fractions in JSON).

Usage:
  python audit_manuscript_numbers.py --repo <FedFairGNN> [--min-digits 3] [--loose]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

# Thousands separators are part of the number, not a boundary: "21,742" is one
# value, and splitting it produced "21" and "742" as two unsourced numbers.
# Eleven of the reported misses in datasets.tex were this bug, which is enough
# noise to hide a real one.
NUM = re.compile(r"(?<![\w.,])(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)(?![\w,])")
# LaTeX constructs whose numbers are typography, not results. These are MASKED
# OUT of the line, never used to skip it: main.tex writes one paragraph per
# line, so skipping any line containing \ref threw away the real numbers in it.
# That is how the first version of this script failed to flag 705% and 2212%,
# the two figures already known to be stale -- which is exactly the acceptance
# test a tool like this has to pass before anyone trusts its silence.
MASK = re.compile(r"\\(?:label|ref|eqref|cite[a-z]*|includegraphics|usepackage|"
                  r"documentclass|setlength|hspace|vspace|resizebox|tabcolsep|"
                  r"arraystretch|cmidrule|multirow|multicolumn|input|linewidth)"
                  r"\s*(?:\[[^\]]*\])?\s*(?:\{[^{}]*\})*")


def artifact_index(results_dir: str, max_places: int = 6):
    """Index every artifact value at each rounding precision.

    A manuscript prints ROUNDED values, so "0.0155" in the paper must match
    0.015512... in a JSON file. Exact digit-string equality (the first version
    of this script) called 493 of 729 numbers missing, which is noise, not a
    finding. Index round(v, d) for each d instead and match at the precision the
    manuscript actually used.
    """
    idx = {d: set() for d in range(max_places + 1)}
    n = 0
    for path in glob.glob(os.path.join(results_dir, "**", "*.json"), recursive=True):
        try:
            blob = open(path, encoding="utf-8", errors="ignore").read()
        except OSError:
            continue
        for m in re.finditer(r"-?\d+\.?\d*(?:[eE][-+]?\d+)?", blob):
            try:
                v = abs(float(m.group(0)))
            except ValueError:
                continue
            n += 1
            for d in idx:
                idx[d].add(round(v, d))
    return idx, n


def _places(raw: str) -> int:
    raw = raw.replace(",", "")
    return len(raw.split(".")[1]) if "." in raw else 0


def _hit(raw: str, idx, loose: bool) -> bool:
    raw = raw.replace(",", "")
    v = abs(float(raw))
    d = min(_places(raw), max(idx))
    if v in idx[d]:
        return True
    if loose:
        # prose prints a percentage or a multiple of a fraction stored in JSON
        for s in (100.0, 1000.0, 0.01, 0.001):
            w = v * s
            dd = min(max(0, d + (2 if s in (100.0,) else 3 if s == 1000.0 else -2)), max(idx))
            if round(w, dd) in idx[dd]:
                return True
    return False


def scan(tex_path: str, idx, min_digits: int, loose: bool):
    rows = []
    for lineno, line in enumerate(open(tex_path, encoding="utf-8", errors="ignore"), 1):
        if line.lstrip().startswith("%"):
            continue
        line = MASK.sub(" ", line)
        for m in NUM.finditer(line):
            raw = m.group(1)
            if len(raw.replace(".", "").replace(",", "").lstrip("0")) < min_digits:
                continue
            hit = _hit(raw, idx, loose)
            rows.append({"file": tex_path, "line": lineno, "value": raw,
                         "found_in_results": hit,
                         "context": line.strip()[:160]})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--min-digits", type=int, default=3)
    ap.add_argument("--loose", action="store_true")
    ap.add_argument("--out-json", default="")
    a = ap.parse_args()

    results = os.path.join(a.repo, "results")
    if not os.path.isdir(results):
        sys.exit(f"no results dir at {results}")
    idx, n_vals = artifact_index(results)

    targets = [os.path.join(a.repo, "manuscript_neurocomputing", "main.tex")]
    targets += sorted(glob.glob(os.path.join(a.repo, "manuscript_neurocomputing",
                                             "tables", "**", "*.tex"), recursive=True))

    rows = []
    for t in targets:
        if os.path.exists(t):
            rows += scan(t, idx, a.min_digits, a.loose)

    miss = [r for r in rows if not r["found_in_results"]]
    print(f"artifact numeric literals indexed : {n_vals}")
    print(f"manuscript numbers examined       : {len(rows)}")
    print(f"NOT found in any artifact         : {len(miss)}")
    print()
    by_file = {}
    for r in miss:
        by_file.setdefault(os.path.basename(r["file"]), []).append(r)
    for f, rs in sorted(by_file.items(), key=lambda kv: -len(kv[1])):
        print(f"--- {f}: {len(rs)} ---")
        for r in rs:
            print(f"  L{r['line']:>4} {r['value']:>12}   {r['context']}")
    if a.out_json:
        with open(a.out_json, "w") as f:
            json.dump({"n_indexed": n_vals, "rows": rows, "misses": miss}, f, indent=1)
        print(f"\n[+] {a.out_json}")


if __name__ == "__main__":
    main()
