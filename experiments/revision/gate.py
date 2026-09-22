#!/usr/bin/env python3
"""gate.py -- Unified tiered acceptance gate runner per AGENT_PROTOCOL.md §7.

Modes:
  gate.py [--scope auto]   Per-edit mode (~4s). Runs static lints, refs, LaTeX checks,
                           provenance, dead claims. Runs pytest ONLY if git diff touches
                           src/ or tests/. Skips G1 (clean tree).
  gate.py --full           Tier-closing mode (~114s). Runs ALL 7 gates (G1 clean tree,
                           G2 pytest, G3 refs, G4 LaTeX, G5 prov, G6 dead claims, G7 alpha)
                           PLUS tex_lint, preflight_handoff, and audit_manuscript_numbers.
  gate.py --scope test     Force running test suite (G2).
  gate.py --scope tex      Fast LaTeX/manuscript only check (~3s).

Exit code: 0 if all active gates pass, number of failed gates otherwise.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
VENV_PYTHON = os.path.abspath(os.path.join(BASE_DIR, "..", ".venv-local", "bin", "python"))
if not os.path.exists(VENV_PYTHON):
    VENV_PYTHON = sys.executable

OK, FAIL, SKIP, WARN = "PASS", "FAIL", "SKIP", "WARN"


def sh(cmd: str, cwd: str = BASE_DIR) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, shell=True, capture_output=True, text=True,
                          errors="replace", cwd=cwd)


def get_tier_base_commit() -> str | None:
    """Read the most recent tier closing commit from HANDOFF.md."""
    handoff_path = os.path.abspath(os.path.join(BASE_DIR, "..", "docs", "review", "HANDOFF.md"))
    if not os.path.exists(handoff_path):
        return None
    try:
        with open(handoff_path, "r", encoding="utf-8") as f:
            text = f.read()
        matches = re.findall(r"Commit kết tầng \(FedFairGNN\):\*{0,2}\s*`?([a-f0-9]{7,40})`?", text)
        if matches:
            for sha in reversed(matches):
                chk = sh(f"git rev-parse --verify {sha}^{{commit}}")
                if chk.returncode == 0:
                    return sha
    except Exception:
        pass
    return None


def diff_touches_code(base_commit: str | None = None) -> tuple[bool, str]:
    """Check if diff against tier base commit or staged/working tree touches src/ or tests/."""
    resolved_base = base_commit or get_tier_base_commit()
    diff_target = resolved_base if resolved_base else "HEAD"
    # Verify git target is valid
    chk = sh(f"git rev-parse --verify {diff_target}^{{commit}}")
    if chk.returncode != 0:
        diff_target = "HEAD"

    r1 = sh(f"git diff --name-only {diff_target}")
    r2 = sh("git diff --cached --name-only")
    r3 = sh("git status --porcelain")
    all_files = set((r1.stdout + "\n" + r2.stdout + "\n" + r3.stdout).split())
    for f in all_files:
        f_clean = f.strip().lstrip("M").lstrip("A").lstrip("D").lstrip("?").strip()
        if f_clean.startswith(("src/", "tests/", "FedFairGNN/src/", "FedFairGNN/tests/")):
            return True, diff_target
    return False, diff_target


def gate_tree_clean(is_full: bool) -> tuple[str, str]:
    if not is_full:
        return SKIP, "chỉ kiểm ở --full (tránh tín hiệu rỗng giữa chừng)"
    r = sh("git status --porcelain -- . ':!results' ':!*.pdf'")
    out = r.stdout.strip()
    if not out:
        return OK, "cây sạch"
    return FAIL, out[:120]


def gate_tests(is_full: bool, force_test: bool, base_commit: str | None = None) -> tuple[str, str]:
    touches, base_used = diff_touches_code(base_commit)
    should_run = is_full or force_test or touches
    if not should_run:
        return SKIP, f"diff với biên tầng ({base_used[:8]}) không chạm src/ hoặc tests/"
    r = sh(f"{VENV_PYTHON} -m pytest tests/ -q -p no:warnings")
    failed = "FAILED" in r.stdout or r.returncode != 0
    return (FAIL if failed else OK, f"pytest exit={r.returncode}")


def gate_refs() -> tuple[str, str]:
    r = sh(f"{VENV_PYTHON} experiments/revision/audit_refs.py")
    return (OK if r.returncode == 0 else FAIL, f"audit_refs exit={r.returncode}")


def gate_tex_lint() -> tuple[str, str]:
    r = sh(f"{sys.executable} experiments/revision/tex_lint.py")
    return (OK if r.returncode == 0 else FAIL, "tất cả tex_lint đạt" if r.returncode == 0 else "có vi phạm")


def gate_latex() -> tuple[str, str]:
    tinytex_bin = "/Users/anson/Library/TinyTeX/bin/universal-darwin"
    ms_dir = os.path.join(BASE_DIR, "manuscript_neurocomputing")
    # Multi-pass compile
    sh(f"{tinytex_bin}/bibtex main", cwd=ms_dir)
    out = ""
    for _ in range(2):
        out = sh(f"{tinytex_bin}/pdflatex -interaction=nonstopmode main.tex", cwd=ms_dir).stdout
    errs = len(re.findall(r"^! ", out, re.M))
    undef = len(re.findall(r"undefined", out))
    pages = re.search(r"main\.pdf \((\d+) pages", out)
    p_count = pages.group(1) if pages else "?"
    # Z-5: Page count is no longer a hard gate; only errors and undefined references block, pages monitored.
    good = (errs == 0 and undef == 0)
    return (OK if good else FAIL, f"errors={errs} undefined={undef} pages={p_count}")


def gate_provenance() -> tuple[str, str]:
    import glob
    import json
    bad = []
    for p in glob.glob(f"{BASE_DIR}/results/**/*.json", recursive=True):
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


def gate_dead_claims() -> tuple[str, str]:
    import glob
    tex_path = os.path.join(BASE_DIR, "manuscript_neurocomputing", "main.tex")
    targets = [tex_path, os.path.join(BASE_DIR, "manuscript_neurocomputing", "highlights.md")]
    targets += glob.glob(os.path.join(BASE_DIR, "manuscript_neurocomputing", "tables", "**", "*.tex"), recursive=True)
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


def gate_alpha_control() -> tuple[str, str]:
    tex_path = os.path.join(BASE_DIR, "manuscript_neurocomputing", "main.tex")
    t = open(tex_path, encoding="utf-8", errors="replace").read()
    bad = [w for w in ("alpha is a knob", "alpha controls", "tunes the fairness")
           if w.lower() in t.lower()]
    m = re.search(r"operator-facing controls are \$\\tau\$ and \$\\epsilon\$", t)
    return (OK if (not bad and m) else FAIL,
            "alpha không còn là control" if (not bad and m) else f"{bad} / anchor={bool(m)}")


def run_preflight() -> tuple[str, str]:
    r = sh(f"{VENV_PYTHON} experiments/revision/preflight_handoff.py")
    lines = [l for l in r.stdout.splitlines() if l.strip()]
    res_line = lines[-1] if lines else "Result: unknown"
    return (OK if "3/3 checks passed" in r.stdout else FAIL, res_line)


def run_audit_numbers() -> tuple[str, str]:
    r = sh(f"{VENV_PYTHON} experiments/revision/audit_manuscript_numbers.py --repo . --min-digits 3 --loose")
    m = re.search(r"NOT found in any artifact\s*:\s*(\d+)", r.stdout)
    not_found = int(m.group(1)) if m else -1
    return (OK if not_found == 0 else FAIL, f"NOT found in artifacts = {not_found}")


def run_audit_table_configs() -> tuple[str, str]:
    r = sh(f"{VENV_PYTHON} experiments/revision/audit_table_configs.py")
    lines = [l for l in r.stdout.splitlines() if l.strip()]
    res_line = ""
    for l in lines:
        if "Result:" in l:
            res_line = l.strip()
            break
    if not res_line and lines:
        res_line = lines[-1].strip()
    return (OK if r.returncode == 0 else FAIL, res_line or f"exit={r.returncode}")


def main():
    parser = argparse.ArgumentParser(description="Unified acceptance gate runner per AGENT_PROTOCOL §7")
    parser.add_argument("--full", action="store_true", help="Run full suite for tier closure (~114s)")
    parser.add_argument("--scope", choices=["auto", "test", "tex"], default="auto",
                        help="Scope: auto (default), test (force pytest), tex (LaTeX/prose only)")
    parser.add_argument("--base", default=None, help="Base commit to diff against (defaults to last tier commit in HANDOFF.md)")
    args = parser.parse_args()

    is_full = args.full
    force_test = args.scope == "test"
    is_tex_only = args.scope == "tex" and not is_full
    base_commit = args.base

    print("=" * 60)
    print(f"ACCEPTANCE GATES ({'FULL TIER-CLOSING' if is_full else 'PER-EDIT: ' + args.scope})")
    print("=" * 60)

    gates = [
        ("G1  cây mã sạch", lambda: gate_tree_clean(is_full)),
        ("G2  bộ test", lambda: gate_tests(is_full, force_test, base_commit) if not is_tex_only else (SKIP, "bỏ qua per --scope tex")),
        ("G3  cite/ref", gate_refs),
        ("G4  LaTeX (errors/undef)", gate_latex),
        ("G5  provenance (G-prov)", gate_provenance),
        ("G6  không tuyên bố đã rút", gate_dead_claims),
        ("G7  alpha không phải control", gate_alpha_control),
        ("L1  tex_lint (tĩnh 0.01s)", gate_tex_lint),
    ]

    if is_full:
        gates.append(("CP-1 preflight handoff", run_preflight))
        gates.append(("CP-2 audit numbers loose", run_audit_numbers))
        gates.append(("CP-3 table configs audit", run_audit_table_configs))

    failed = 0
    passed = 0
    skipped = 0

    for name, fn in gates:
        try:
            st, msg = fn()
        except Exception as e:
            st, msg = FAIL, f"{type(e).__name__}: {e}"

        if st == FAIL:
            failed += 1
        elif st == OK:
            passed += 1
        elif st == SKIP:
            skipped += 1

        print(f"[{st}] {name:<32} {msg}")

    print("-" * 60)
    total_evaluated = passed + failed
    print(f"Kết quả: {passed}/{total_evaluated} cổng được đánh giá đạt ({skipped} bỏ qua)")
    if is_full and failed == 0:
        print("✅ TOÀN BỘ CỔNG FULL PASS -- ĐỦ ĐIỀU KIỆN ĐÓNG TẦNG")
    sys.exit(failed)


if __name__ == "__main__":
    main()
