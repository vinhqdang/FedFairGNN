#!/usr/bin/env python3
"""
lint_manuscript_blacklist.py - Automated linter for manuscript phrase blacklist (Phụ Lục B & D5).

Scans documentation and manuscript files to ensure zero violations of overclaim rules,
absolute language, and unauthorized compliance claims.
"""

import os
import re
import sys
from pathlib import Path

# Blacklist definitions: (pattern, description, allowed_exceptions_regex)
BLACKLIST_VIETNAMESE = [
    (r"khẳng định 100%", "Overclaim thống kê tuyệt đối", None),
    (r"quy luật tất yếu không thể chối cãi", "Tuyên bố giáo điều phi khoa học", None),
    (r"Phù hợp toàn diện", "Tuyên bố tuân thủ pháp lý tuyệt đối (EU AI Act)", None),
    (r"Đáp ứng đầy đủ các yêu cầu", "Tuyên bố tuân thủ pháp lý tuyệt đối (EU AI Act)", None),
    (r"không thể và không bao giờ", "Tuyệt đối hoá sai sự thật", None),
    (r"chống đỡ tuyệt vời trước mọi kẻ địch", "Tuyên bố sai với Bảng 2.3c", None),
    (r"đập tan mối lo ngại", "Văn phong phi học thuật / overclaim", None),
    (r"không mở ra bất kỳ bề mặt tấn công mới nào", "Khái quát hoá quá mức từ 1 kiểm định", None),
    (r"Ưu thế tuyệt đối", "Tuyệt đối hoá hiệu năng vận hành", None),
    (r"vượt trội hoàn toàn", "Tuyệt đối hoá khi so sánh baseline", None),
    (r"Tính ưu việt đồng thời", "Overclaim chọn lọc", None),
]

BLACKLIST_ENGLISH = [
    (r"\bstate-of-the-art\b", "Unqualified SOTA claim (use competitive instead)", None),
    (r"\boutperformed\s+all\b", "Overclaim across baselines", None),
    (r"\bsuperiority\b", "Overclaim", r"(?:not\s+(?:claim\s+)?superiority|not\s+superiority)"),
    (r"\bguarantees\s+robustness\b", "Mathematical term used for empirical observation", None),
    (r"\bimmune\s+to\s+byzantine\b", "Falsified by white-box T1 adversary", None),
    (r"\bend-to-end\s+dp\b", "Misleading privacy claim (only disparity statistic is privatised)", None),
]

def scan_file(filepath: Path):
    violations = []
    text = filepath.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()

    is_vietnamese = filepath.suffix in [".md"]
    rules = BLACKLIST_VIETNAMESE if is_vietnamese else BLACKLIST_ENGLISH

    for line_idx, line in enumerate(lines, 1):
        # Skip markdown tables in Phụ Lục B / Blacklist dictionaries where the phrases are listed as prohibited examples
        if "Cụm từ BỊ CẤM" in line or "Bảng Tra Cứu Nội Bộ" in line or "❌ Cấm viết" in line or "blacklist" in line.lower():
            continue
        if line.strip().startswith("|") and ("Lý do cấm" in line or "overclaim" in line.lower() or "tuyệt đối" in line.lower()):
            continue
        if "Phụ Lục B" in line or "blacklist_dictionary" in line.lower():
            continue

        for pattern, reason, exception_pat in rules:
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                if exception_pat and re.search(exception_pat, line, re.IGNORECASE):
                    continue
                violations.append((filepath, line_idx, match.group(0), reason, line.strip()))
    return violations

def main():
    project_root = Path(__file__).resolve().parent.parent.parent
    targets = [
        project_root / "docs" / "review" / "CAU_CHUYEN.md",
        project_root / "docs" / "review" / "CAU_CHUYEN_short.md",
        project_root / "docs" / "review" / "KE_HOACH_VIET_LAI_MANUSCRIPT.md",
    ]

    # Also check if manuscript_v2 directory exists
    manuscript_v2_dir = project_root / "FedFairGNN" / "manuscript_v2"
    if manuscript_v2_dir.exists():
        for ext in ["*.tex", "*.md"]:
            targets.extend(manuscript_v2_dir.rglob(ext))

    # Also check if manuscript_neurocomputing directory exists
    manuscript_dir = project_root / "FedFairGNN" / "manuscript_neurocomputing"
    if manuscript_dir.exists():
        for ext in ["*.tex", "*.md"]:
            targets.extend(manuscript_dir.rglob(ext))

    total_violations = 0
    print("=" * 70)
    print("🔍 RUNNING MANUSCRIPT BLACKLIST LINTER")
    print("=" * 70)

    for target in targets:
        if not target.exists():
            continue
        violations = scan_file(target)
        if violations:
            print(f"\n❌ {target.name} ({len(violations)} violations):")
            for _, line_no, matched, reason, snippet in violations:
                print(f"   Line {line_no}: [{matched}] -> {reason}")
                print(f"      Snippet: {snippet[:100]}...")
            total_violations += len(violations)
        else:
            print(f"✅ {target.name}: CLEAN (0 violations)")

    print("\n" + "=" * 70)
    if total_violations == 0:
        print("🎉 AUDIT PASSED: 0 blacklist violations found across all documents!")
        print("=" * 70)
        sys.exit(0)
    else:
        print(f"⚠️ AUDIT FAILED: {total_violations} blacklist violations detected.")
        print("=" * 70)
        sys.exit(1)

if __name__ == "__main__":
    main()
