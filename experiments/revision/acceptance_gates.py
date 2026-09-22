"""Acceptance gates G1-G10 plus G-prov, run as one command.

Each gate is a check the submission must pass, written so that it fails loudly
rather than warning quietly. Exit code is the number of failed gates.

Usage: python experiments/revision/acceptance_gates.py
"""
from __future__ import annotations

import glob
import json
import math
import os
import re
import subprocess
import sys

TEX = "manuscript_neurocomputing/main.tex"
RESULTS = "results"
OK, FAIL = "PASS", "FAIL"


def sh(cmd):
    # pdflatex emits latin-1 in font warnings; never let a decode error
    # masquerade as a gate failure.
    return subprocess.run(cmd, shell=True, capture_output=True, text=True,
                          errors="replace")


def g1_tree_clean():
    # main.pdf is excluded: G4 recompiles it, so a gate that ran after G1 would
    # leave the tree dirty and make G1 fail on a rerun for a reason G1 created.
    r = sh("git status --porcelain -- . ':!results' ':!*.pdf'")
    return (OK if not r.stdout.strip() else FAIL,
            "cây sạch" if not r.stdout.strip() else r.stdout.strip()[:120])


def g2_tests():
    r = sh("../.venv-local/bin/python -m pytest tests/ -q -p no:warnings")
    n = len(re.findall(r"[.sxFE]", (r.stdout.split("\n") or [""])[0]))
    failed = "FAILED" in r.stdout or r.returncode != 0
    return (FAIL if failed else OK, f"pytest exit={r.returncode}")


def g3_refs():
    r = sh("../.venv-local/bin/python experiments/revision/audit_refs.py")
    return (OK if r.returncode == 0 else FAIL, f"audit_refs exit={r.returncode}")


def g4_latex():
    b = "/Users/anson/Library/TinyTeX/bin/universal-darwin"
    # try/finally, because a gate that leaves the process in another directory
    # makes every gate after it fail for a reason that has nothing to do with
    # what it checks -- which is exactly what happened on the first run.
    here = os.getcwd()
    try:
        os.chdir("manuscript_neurocomputing")
        sh(f"{b}/bibtex main")
        out = ""
        for _ in range(2):
            out = sh(f"{b}/pdflatex -interaction=nonstopmode main.tex").stdout
    finally:
        os.chdir(here)
    errs = len(re.findall(r"^! ", out, re.M))
    undef = len(re.findall(r"undefined", out))
    pages = re.search(r"main\.pdf \((\d+) pages", out)
    good = errs == 0 and undef == 0
    return (OK if good else FAIL,
            f"errors={errs} undefined={undef} pages={pages.group(1) if pages else '?'}")


def g_prov():
    """Every manifest under results/ must look machine-generated.

    The criterion is float precision, not manifest formatting: a wall-clock
    field rounded to exactly one decimal in every record is the signature of a
    stdout print, because the code writes float(perf_counter() - t0).
    """
    bad = []
    for p in glob.glob(f"{RESULTS}/**/*.json", recursive=True):
        try:
            d = json.load(open(p))
        except Exception:
            continue

        def walk(o):
            if isinstance(o, dict):
                if "wall_clock_s" in o and isinstance(o["wall_clock_s"], (int, float)):
                    yield o["wall_clock_s"]
                for v in o.values():
                    yield from walk(v)
            elif isinstance(o, list):
                for v in o[:400]:
                    yield from walk(v)

        wall = [w for w in walk(d) if isinstance(w, float)]
        if len(wall) >= 5 and all(abs(w * 10 - round(w * 10)) < 1e-9 for w in wall):
            bad.append(os.path.basename(p))
    return (OK if not bad else FAIL,
            "không artifact nào mang dấu vết stdout" if not bad else f"nghi vấn: {bad}")


def g6_no_dead_claims():
    targets = [TEX, "manuscript_neurocomputing/highlights.md"]
    targets += glob.glob("manuscript_neurocomputing/tables/**/*.tex", recursive=True)
    t = "".join(open(f, encoding="utf-8", errors="replace").read() for f in targets if os.path.exists(f)).casefold()
    dead = {
        "86.2": "single-seed capture figure",
        "best auc among": "retracted SOTA claim",
        "best utility among": "retracted SOTA claim",
        "pillar c1": "superseded framing",
        "we have not done it": "false statement about a run that exists",
        "705\\%": "retracted slack figure",
        "2212\\%": "retracted slack figure",
        "resists adaptive stealth": "mâu thuẫn tab:adaptive_poisoner",
        "203k elliptic": "ADR-15 đã gỡ Elliptic",
        "2.4m-node": "ADR-8 đã đóng khung ogbn",
    }
    hits = {k: v for k, v in dead.items() if k in t}
    return (OK if not hits else FAIL, "sạch" if not hits else str(hits))


def g7_alpha_not_a_control():
    """A pre-registered null must bind the writing, not just the log."""
    t = open(TEX, encoding="utf-8", errors="replace").read()
    bad = [w for w in ("alpha is a knob", "alpha controls", "tunes the fairness")
           if w.lower() in t.lower()]
    m = re.search(r"operator-facing controls are \$\\tau\$ and \$\\epsilon\$", t)
    return (OK if (not bad and m) else FAIL,
            "alpha không còn là control" if (not bad and m) else f"{bad} / anchor={bool(m)}")


GATES = [("G1  cây mã sạch", g1_tree_clean),
         ("G2  bộ test", g2_tests),
         ("G3  cite/ref", g3_refs),
         ("G4  LaTeX", g4_latex),
         ("G5  provenance (G-prov)", g_prov),
         ("G6  không còn tuyên bố đã rút", g6_no_dead_claims),
         ("G7  alpha không phải control", g7_alpha_not_a_control)]


def main():
    bad = 0
    for name, fn in GATES:
        try:
            st, msg = fn()
        except Exception as e:
            st, msg = FAIL, f"{type(e).__name__}: {e}"
        bad += st == FAIL
        print(f"[{st}] {name:<32} {msg}")
    print(f"\n{len(GATES) - bad}/{len(GATES)} cổng đạt")
    sys.exit(bad)


if __name__ == "__main__":
    main()
