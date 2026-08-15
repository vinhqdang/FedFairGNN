# Revision 1 submission package

Everything needed to submit **Revision 1** to *Discover Artificial Intelligence*
(MSID `ca5f4c68-82d0-46f5-99ec-5304715a69a6`), plus the Revision 0 review
documents it responds to.

## What to upload

| Springer item | File |
|---|---|
| Manuscript (LaTeX source) | `TrustFedGNN_revision1_latex.zip` |
| Manuscript (PDF, clean) | `revision1/main.pdf` |
| Manuscript (tracked changes) | `revision1_tracked/main_tracked.pdf` |
| Response to reviewers | `response_letter/response_letter.pdf` |
| Figures + tables (one archive) | `TrustFedGNN_revision1_figures_tables.zip` |
| — figures only | `TrustFedGNN_revision1_figures.zip` |
| — tables only | `TrustFedGNN_revision1_tables.zip` |

The three artwork ZIPs are alternatives, not additions: upload the combined one
unless the system asks for figures and tables separately. Each contains
`MANIFEST.txt`, which maps `Fig1`–`Fig6` to their manuscript numbers and captions,
lists the table order, states the formats, and records what changed this revision.

## Contents

```
revision0_review/
├── Reviewer1_review.pdf              incoming: reviewer 1 (7 comments)
├── reviewer2_review.txt              incoming: reviewer 2 (2 comments)
├── editor_comment.txt                incoming: editor + editorial requirements
├── Discover AI_minorRev_Minh.xlsx    incoming: corresponding author's notes
├── main_revision0.tex                snapshot of the submitted (rev 0) source
│
├── revision1/                        CLEAN revised manuscript, compiles as-is
│   ├── main.tex  main.pdf  main.bbl  ref.bib
│   ├── sn-jnl.cls  sn-mathphys-num.bst
│   └── tables/  figures/
│
├── revision1_tracked/                TRACKED-CHANGES version (latexdiff)
│   ├── main_tracked.tex  main_tracked.pdf
│   └── (same class/bib/tables/figures)
│
├── response_letter/
│   ├── response_letter.pdf           formatted, 10 pp — upload this
│   ├── response_letter.tex
│   └── response_letter.md            plain text, for pasting into the form
│
├── springer_upload/                  production artwork + tables (unzipped)
│   ├── MANIFEST.txt                  figure/table map, formats, what changed
│   ├── figures/Fig1..Fig6 .pdf/.eps/.tif/.png
│   └── tables/Tables_all.pdf + Tables_all.tex + tables/
│
├── TrustFedGNN_revision1_latex.zip           LaTeX source (verified to compile)
├── TrustFedGNN_revision1_figures_tables.zip  artwork + tables (combined)
├── TrustFedGNN_revision1_figures.zip         artwork only
└── TrustFedGNN_revision1_tables.zip          tables only
```

## Figure numbering

| File | Manuscript figure | Source |
|---|---|---|
| `Fig1` | Fig. 1 — TrustFedGNN architecture | TikZ, extracted standalone from `main.tex` |
| `Fig2` | Fig. 2 — utility/fairness vs. ε (Bail) | `figures/privacy_bail.pdf` |
| `Fig3` | Fig. 3 — attribute-inference attack | `figures/privacy_attack.pdf` **(re-rendered this revision)** |
| `Fig4` | Fig. 4 — fairness–utility Pareto frontier | `figures/pareto.pdf` |
| `Fig5` | Fig. 5 — AUC vs. number of Byzantine clients | `figures/robustness_byz.pdf` |
| `Fig6` | Fig. 6 — convergence on Bail | `figures/convergence.pdf` |

Formats per Springer artwork guidelines: **PDF and EPS are vector** (preferred for
line art); **TIFF is 600 dpi, LZW-compressed**; PNG at 600 dpi is included for
convenience only. Vector files should be preferred wherever the system accepts them.

## Still to do at submission time

- [ ] Select **"yes"** under the data availability declaration in the system, and
      paste the *Data availability* statement from the manuscript verbatim so the
      two match exactly (editorial requirement 2).
- [x] `https://github.com/vinhqdang/FedFairGNN` is **public** — confirmed, and the
      Code availability statement cites it directly.
- [ ] *Optional but worthwhile:* the raw per-run logs (`results/`) are not in the
      repository and never were, so the Data/Code availability statements now say
      they are available from the corresponding author on request. If the logs can
      be recovered from the machine that ran the matrix, committing them (or
      depositing them on Zenodo for a DOI) would let both statements be
      strengthened to "included in the repository".

## Notes on this revision

- **No new experiments were run.** Every reviewer point was wording, scope, or
  consistency; Reviewer 1 explicitly permitted moderating the robustness claims
  rather than adding seeds.
- Two artifacts were regenerated **from already-logged results** (no re-training):
  Fig. 3 (the chance line was mislabelled) and the energy columns of Tables 10/11
  (arithmetic on wall-clock already in Table 10). Both generator scripts
  (`experiments/privacy_attack.py`, `experiments/trust_eval.py`,
  `experiments/report.py`) were patched so a future full rerun reproduces the
  corrected output.
- Nine text/table inconsistencies were fixed: the five Reviewer 1 found, plus four
  the re-audit surfaced (listed at the end of the response letter).
