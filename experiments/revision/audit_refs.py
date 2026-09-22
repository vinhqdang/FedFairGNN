"""Check every \\cite and \\ref in the manuscript against what actually exists.

Citation and cross-reference errors are cheap to make and expensive to be caught
on: a wrong key still compiles, still typesets, and still looks right. Two real
ones were found by hand in this project -- a PoPETs row citing a different
paper, and a venue label two years off -- so this runs the check by machine.

Usage: python experiments/revision/audit_refs.py [--tex main.tex] [--bib ref.bib]
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys

CITE = re.compile(r"\\cite[a-z]*\*?(?:\[[^\]]*\])*\{([^}]*)\}")
REF = re.compile(r"\\(?:ref|eqref|autoref|cref|Cref)\{([^}]*)\}")
LABEL = re.compile(r"\\label\{([^}]*)\}")
BIBKEY = re.compile(r"^@\w+\{([^,]+),", re.M)


def read_all(tex_path: str) -> str:
    """main.tex plus every file it \\input s, since tables carry cites too."""
    base = os.path.dirname(tex_path)
    text = open(tex_path, encoding="utf-8", errors="ignore").read()
    for inc in re.findall(r"\\input\{([^}]*)\}", text):
        p = os.path.join(base, inc if inc.endswith(".tex") else inc + ".tex")
        if os.path.exists(p):
            text += "\n" + open(p, encoding="utf-8", errors="ignore").read()
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tex", default="manuscript_neurocomputing/main.tex")
    ap.add_argument("--bib", default="manuscript_neurocomputing/ref.bib")
    a = ap.parse_args()

    text = read_all(a.tex)
    bib = open(a.bib, encoding="utf-8", errors="ignore").read()

    keys = BIBKEY.findall(bib)
    dupes = {k for k in keys if keys.count(k) > 1}
    bibkeys = set(keys)

    cited = {k.strip() for m in CITE.findall(text) for k in m.split(",") if k.strip()}
    refs = {r.strip() for m in REF.findall(text) for r in m.split(",") if r.strip()}
    labels = LABEL.findall(text)
    dup_labels = {l for l in labels if labels.count(l) > 1}
    labelset = set(labels)

    problems = {
        "cite -> no bib entry": sorted(cited - bibkeys),
        "ref -> no label": sorted(refs - labelset),
        "duplicate bib keys": sorted(dupes),
        "duplicate labels": sorted(dup_labels),
    }
    # orphans are reported separately: they are untidy, not broken
    orphan_labels = sorted(l for l in labelset - refs
                           if not l.startswith(("sec:", "app:")))
    uncited = sorted(bibkeys - cited)

    print(f"cites {len(cited)} distinct | refs {len(refs)} | labels {len(labelset)} | bib entries {len(bibkeys)}")
    bad = 0
    for name, items in problems.items():
        print(f"\n[{'FAIL' if items else ' ok '}] {name}: {len(items)}")
        for i in items:
            print(f"    {i}")
        bad += len(items)
    print(f"\n[warn] labels never referenced (excluding sec:/app:): {len(orphan_labels)}")
    for l in orphan_labels:
        print(f"    {l}")
    print(f"[warn] bib entries never cited: {len(uncited)}")
    for u in uncited:
        print(f"    {u}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
