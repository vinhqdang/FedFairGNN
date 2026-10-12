# TrustFedGNN Manuscript (Elsevier Neurocomputing)

Official LaTeX manuscript repository for the paper:  
**"TrustFedGNN: A Byzantine-Robust and Differentially Private Federated Graph Neural Network with Fairness Constraints"**  
Submitted to: **Elsevier *Neurocomputing*** (Specialized in Neural Networks, Deep Learning, and Trustworthy AI Systems).

---

## 👤 Author Information
- **Ngoc-Son-An Nguyen** (corresponding author) --- Industrial University of Ho Chi Minh City, Ho Chi Minh City, Vietnam --- `annns25871@pgr.iuh.edu.vn`
- **Quang-Vinh Dang** --- British University Vietnam, Hung Yen, Vietnam --- `vinh.dq4@buv.edu.vn`

---

## 📁 Repository Structure

```
.
├── elsarticle.cls              # Official Elsevier document class (v3.5, 2026)
├── elsarticle-num.bst          # Numbered citation style
├── elsarticle-harv.bst         # Author-year citation style
├── main.tex                    # Primary LaTeX manuscript
├── highlights.md               # Mandatory Elsevier Highlights (<= 85 chars/bullet)
├── credit_statement.md         # Formal CRediT authorship contribution statement
├── declaration_of_interest.md  # Declaration of competing interest
├── cover_letter.md             # Formal cover letter to the Editor-in-Chief
├── ref.bib                     # Comprehensive bibliography
├── figures/                    # High-resolution vector PDF figures (Fig 1 - Fig 6)
└── tables/                     # LaTeX tables (including all 9 revision tables)
```

---

## ⚙️ Compilation Instructions

To compile the manuscript locally via command line:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Or import directly into **Overleaf** by linking this GitHub repository (`https://github.com/ngngsonan/trustfedgnn_manuscript`).

---

## 🔗 Code & Data Availability
The complete open-source reference implementation, automated test suites, and reproduction scripts are available at:  
👉 [https://github.com/vinhqdang/FedFairGNN](https://github.com/vinhqdang/FedFairGNN) (branch `main_trustfed`).
