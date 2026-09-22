#!/usr/bin/env python3
"""tex_lint.py -- Fast static linter for LaTeX and manuscript surfaces (< 0.05s).

Checks (per AGENT_PROTOCOL.md §7.2):
1. \\word in .tex (e.g. \\\\tau) which LaTeX accepts as newline+word, breaking math.
2. \\input{...} pointing to non-existent files.
3. Blacklist dictionary (06 §2.2): outperforms, state-of-the-art, best AUC, etc.
4. Highlights bullets <= 85 characters including spaces (Elsevier rule).
5. Limitations section has exactly 15 items.
6. Number repeated twice in the same clause/sentence (appending instead of replacing).

Exit code: 0 if all clean, number of failed checks otherwise.
"""
from __future__ import annotations

import glob
import os
import re
import sys
from collections import Counter

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MANUSCRIPT_DIR = os.path.join(BASE_DIR, "manuscript_neurocomputing")
MAIN_TEX = os.path.join(MANUSCRIPT_DIR, "main.tex")
HIGHLIGHTS_MD = os.path.join(MANUSCRIPT_DIR, "highlights.md")
TABLES_DIR = os.path.join(MANUSCRIPT_DIR, "tables")


def get_tex_files() -> list[str]:
    files = [MAIN_TEX] if os.path.exists(MAIN_TEX) else []
    files += sorted(glob.glob(os.path.join(TABLES_DIR, "**", "*.tex"), recursive=True))
    return files


def check_double_backslash(files: list[str]) -> tuple[bool, list[str]]:
    """Check for \\\\[a-zA-Z] in .tex files (e.g. \\\\tau)."""
    pattern = re.compile(r'\\\\[a-zA-Z]')
    violations = []
    for f in files:
        rel = os.path.relpath(f, BASE_DIR)
        with open(f, "r", encoding="utf-8", errors="replace") as fp:
            for idx, line in enumerate(fp, 1):
                # Ignore LaTeX comment lines
                clean_line = re.sub(r'(?<!\\)%.*$', '', line).strip()
                if pattern.search(clean_line):
                    violations.append(f"{rel}:{idx}: {clean_line[:80]}")
    return len(violations) == 0, violations


def check_missing_inputs() -> tuple[bool, list[str]]:
    """Check for \\input{...} pointing to missing files."""
    if not os.path.exists(MAIN_TEX):
        return False, [f"Missing {MAIN_TEX}"]
    pattern = re.compile(r'\\input\{([^}]+)\}')
    violations = []
    with open(MAIN_TEX, "r", encoding="utf-8", errors="replace") as fp:
        for idx, line in enumerate(fp, 1):
            clean_line = re.sub(r'(?<!\\)%.*$', '', line).strip()
            for match in pattern.findall(clean_line):
                target = match if match.endswith(".tex") else f"{match}.tex"
                full_path = os.path.join(MANUSCRIPT_DIR, target)
                if not os.path.exists(full_path):
                    violations.append(f"main.tex:{idx}: \\input{{{match}}} -> {target} not found")
    return len(violations) == 0, violations


def check_blacklist(files: list[str]) -> tuple[bool, list[str]]:
    """Check for forbidden terms per 06 §2.2."""
    patterns = [
        (re.compile(r'\boutperform[a-z]*\b', re.IGNORECASE), "outperforms"),
        (re.compile(r'\bstate-of-the-art\b', re.IGNORECASE), "state-of-the-art"),
        (re.compile(r'\bbest\s+(auc|utility|accuracy)\b', re.IGNORECASE), "best AUC/utility"),
        (re.compile(r'\bimmune\s+to\s+byzantine\b', re.IGNORECASE), "immune to Byzantine"),
        (re.compile(r'\b(?:end-to-end|zero)\s+dp\s+leakage\b', re.IGNORECASE), "end-to-end DP"),
        (re.compile(r'\bformally\s+verified\b', re.IGNORECASE), "formally verified (use machine-checked)"),
    ]
    violations = []
    for f in files:
        rel = os.path.relpath(f, BASE_DIR)
        with open(f, "r", encoding="utf-8", errors="replace") as fp:
            for idx, line in enumerate(fp, 1):
                clean_line = re.sub(r'(?<!\\)%.*$', '', line).strip()
                for pat, label in patterns:
                    if pat.search(clean_line):
                        violations.append(f"{rel}:{idx} [{label}]: {clean_line[:80]}")
    return len(violations) == 0, violations


def check_highlights() -> tuple[bool, list[str]]:
    """Check bullet highlights <= 85 characters."""
    if not os.path.exists(HIGHLIGHTS_MD):
        return False, [f"Missing {HIGHLIGHTS_MD}"]
    violations = []
    bullet_count = 0
    with open(HIGHLIGHTS_MD, "r", encoding="utf-8", errors="replace") as fp:
        for idx, line in enumerate(fp, 1):
            line_str = line.strip()
            if line_str.startswith(("-", "*")):
                bullet_count += 1
                bullet_text = line_str.lstrip("-*").strip()
                length = len(bullet_text)
                if length > 85:
                    violations.append(f"highlights.md:{idx}: {length} chars (>85): '{bullet_text}'")
    if bullet_count < 3 or bullet_count > 5:
        violations.append(f"highlights.md: expected 3-5 bullets, found {bullet_count}")
    return len(violations) == 0, violations


def check_limitations_count() -> tuple[bool, list[str]]:
    """Check limitations items == 15."""
    if not os.path.exists(MAIN_TEX):
        return False, [f"Missing {MAIN_TEX}"]
    with open(MAIN_TEX, "r", encoding="utf-8", errors="replace") as fp:
        text = fp.read()
    m = re.search(r'\\section\{Limitations\}(.*?)(?:\\section\{|\\bibliographystyle|\Z)', text, re.DOTALL)
    if not m:
        return False, ["\\section{Limitations} not found in main.tex"]
    sec_content = m.group(1)
    # Count items in enumerate
    items = re.findall(r'\\item\b', sec_content)
    count = len(items)
    if count != 15:
        return False, [f"Limitations has {count} items, expected exactly 15"]
    return True, []


def check_duplicate_numbers_in_clauses() -> tuple[bool, list[str]]:
    """Check for duplicate multi-digit/percentage numbers within the same clause."""
    if not os.path.exists(MAIN_TEX):
        return False, [f"Missing {MAIN_TEX}"]
    with open(MAIN_TEX, "r", encoding="utf-8", errors="replace") as fp:
        text = fp.read()

    # Strip comments and large non-prose environments
    clean = re.sub(r'(?<!\\)%.*$', '', text, flags=re.M)
    clean = re.sub(r'\\begin\{(tabular|tikzpicture|table\*?|figure\*?|equation\*?|align\*?)\}.*?\\end\{\1\}',
                   '', clean, flags=re.DOTALL)

    # Split into sentences / major clauses
    sentences = re.split(r'(?<=[.?!;])\s+|\n\n+', clean)
    violations = []
    # Match numbers like 39.5%, 39.5, 0.7445, 10.5
    num_pattern = re.compile(r'(?<![a-zA-Z\\])\b(\d+(?:\.\d+)?%?)\b(?![a-zA-Z])')

    for s in sentences:
        s_clean = s.strip()
        if not s_clean:
            continue
        # Strip internal citation, label, and ref tags
        s_clean = re.sub(r'\\(cite|label|ref|eqref)\{[^}]*\}', '', s_clean)
        # Find multi-digit numbers or percentages
        tokens = num_pattern.findall(s_clean)
        meaningful_nums = [t for t in tokens if ('.' in t or '%' in t) and not t.endswith('.')]
        counts = Counter(meaningful_nums)
        for num, cnt in counts.items():
            if cnt > 1:
                # Check distance between occurrences in the sentence
                escaped = re.escape(num)
                m = re.search(escaped + r'(.{1,120}?)' + escaped, s_clean)
                if m:
                    span = m.group(1)
                    # Exclude comparisons like "from X to X" or identical endpoints
                    if "bounded away" in span or "at least" in span or ("," in span and "to" not in span):
                        violations.append(f"Duplicate number '{num}' in clause: '{s_clean[:100]}...'")

    return len(violations) == 0, violations


def run_all_lints() -> int:
    tex_files = get_tex_files()
    checks = [
        ("\\\\[a-zA-Z] syntax", lambda: check_double_backslash(tex_files)),
        ("\\input missing files", check_missing_inputs),
        ("Blacklist terms", lambda: check_blacklist(tex_files)),
        ("Highlights <= 85 chars", check_highlights),
        ("Limitations count == 15", check_limitations_count),
        ("Duplicate clause numbers", check_duplicate_numbers_in_clauses),
    ]

    failed = 0
    for name, fn in checks:
        passed, details = fn()
        if passed:
            print(f"[PASS] {name:<30} OK")
        else:
            failed += 1
            print(f"[FAIL] {name:<30} {len(details)} violation(s)")
            for d in details[:5]:
                print(f"       -> {d}")
            if len(details) > 5:
                print(f"       -> ... and {len(details)-5} more")

    if failed == 0:
        print("\nAll tex_lint checks passed.")
    else:
        print(f"\n{failed} tex_lint check(s) failed.")
    return failed


if __name__ == "__main__":
    sys.exit(run_all_lints())
