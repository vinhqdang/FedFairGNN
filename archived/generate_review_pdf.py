#!/usr/bin/env python3
"""
Convert review_request_document.md into a publication-grade, professionally formatted PDF.
Uses reportlab Platypus with custom NumberedCanvas for page numbering and running headers.
Ensures 100% clean typography and standard ASCII / HTML-supported sub/superscript entities.
"""

import os
import sys
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether, HRFlowable
)
from reportlab.pdfgen import canvas

class NumberedCanvas(canvas.Canvas):
    """Two-pass canvas to dynamically compute and print 'Page X of Y' and running headers."""
    def __init__(self, *args, **kwargs):
        super(NumberedCanvas, self).__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super(NumberedCanvas, self).showPage()
        super(NumberedCanvas, self).save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748B"))
        
        # Header (pages > 1)
        if self._pageNumber > 1:
            self.drawString(45, 11 * inch - 36, "TrustFedGNN — Pre-Submission Review Document")
            self.drawRightString(8.5 * inch - 45, 11 * inch - 36, "Springer Discover AI (Q1 Submission)")
            self.setStrokeColor(colors.HexColor("#CBD5E1"))
            self.setLineWidth(0.5)
            self.line(45, 11 * inch - 42, 8.5 * inch - 45, 11 * inch - 42)

        # Footer (all pages)
        self.setStrokeColor(colors.HexColor("#E2E8F0"))
        self.setLineWidth(0.5)
        self.line(45, 40, 8.5 * inch - 45, 40)
        
        self.drawString(45, 28, "CONFIDENTIAL — FOR PRE-SUBMISSION PEER REVIEW ONLY")
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(8.5 * inch - 45, 28, page_str)
        self.restoreState()


def build_pdf(output_pdf_path):
    doc = SimpleDocTemplate(
        output_pdf_path,
        pagesize=letter,
        leftMargin=45,
        rightMargin=45,
        topMargin=48,
        bottomMargin=48
    )

    styles = getSampleStyleSheet()

    # Custom Color Palette
    PRIMARY = colors.HexColor("#1E3A8A")     # Deep Blue
    SECONDARY = colors.HexColor("#0D9488")   # Teal
    DARK_TEXT = colors.HexColor("#0F172A")   # Slate 900
    MUTED_TEXT = colors.HexColor("#475569")  # Slate 600
    BG_LIGHT = colors.HexColor("#F8FAFC")    # Slate 50
    BG_BOX = colors.HexColor("#F1F5F9")      # Slate 100
    ACCENT_AMBER = colors.HexColor("#B45309")# Amber 700

    # Typography styles
    style_title = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        textColor=PRIMARY,
        spaceAfter=3
    )

    style_subtitle = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=15,
        textColor=DARK_TEXT,
        spaceAfter=7
    )

    style_meta = ParagraphStyle(
        'DocMeta',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=12,
        textColor=MUTED_TEXT,
        spaceAfter=4
    )

    style_h1 = ParagraphStyle(
        'SectionH1',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=11.5,
        leading=15,
        textColor=PRIMARY,
        spaceBefore=10,
        spaceAfter=4,
        keepWithNext=True
    )

    style_h2 = ParagraphStyle(
        'SectionH2',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9.5,
        leading=13,
        textColor=DARK_TEXT,
        spaceBefore=6,
        spaceAfter=2,
        keepWithNext=True
    )

    style_body = ParagraphStyle(
        'Body',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=12,
        textColor=DARK_TEXT,
        spaceAfter=4
    )

    style_abstract = ParagraphStyle(
        'AbstractBody',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=8,
        leading=11.5,
        textColor=DARK_TEXT
    )

    style_bullet = ParagraphStyle(
        'BulletItem',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        leading=11,
        textColor=DARK_TEXT,
        leftIndent=12,
        firstLineIndent=-8,
        spaceAfter=2
    )

    style_q_title = ParagraphStyle(
        'QuestionTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8.5,
        leading=11.5,
        textColor=PRIMARY,
        spaceBefore=4,
        spaceAfter=1,
        keepWithNext=True
    )

    style_q_body = ParagraphStyle(
        'QuestionBody',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        leading=11,
        textColor=DARK_TEXT,
        leftIndent=6,
        spaceAfter=3
    )

    style_th = ParagraphStyle(
        'TableHead',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=7.5,
        leading=9.5,
        textColor=colors.white,
        alignment=1  # Centered
    )

    style_td = ParagraphStyle(
        'TableData',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.5,
        leading=9.5,
        textColor=DARK_TEXT,
        alignment=0  # Left
    )

    style_td_center = ParagraphStyle(
        'TableDataCenter',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.5,
        leading=9.5,
        textColor=DARK_TEXT,
        alignment=1
    )

    style_td_bold = ParagraphStyle(
        'TableDataBold',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=7.5,
        leading=9.5,
        textColor=DARK_TEXT,
        alignment=0
    )

    story = []

    # Title & Metadata Banner
    story.append(Paragraph("Pre-Submission Review Package", style_title))
    story.append(Paragraph("TrustFedGNN: Trustworthy Federated Graph Neural Networks", style_subtitle))
    
    meta_text = """
    <b>Target Venue:</b> Springer <i>Discover Artificial Intelligence</i> (Q1 Journal) — Special Collection: <i>Trustworthy and Responsible Federated Learning</i><br/>
    <b>Manuscript Specs:</b> ~30 pages (Springer Nature <code>sn-jnl</code> template) · 14 Tables · 5 Figures · 3 Appendices · 1,779 lines LaTeX<br/>
    <b>Author:</b> Ngoc-Son-An Nguyen · <b>Date:</b> September 2026
    """
    story.append(Paragraph(meta_text, style_meta))
    story.append(HRFlowable(width="100%", thickness=1.2, color=PRIMARY, spaceBefore=2, spaceAfter=5))

    # Abstract Box
    story.append(Paragraph("Abstract", style_h1))
    abstract_p1 = "<b>Background:</b> Federated Learning (FL) enables institutions to train shared models over sensitive, decentralised graph data—transactions, credit records, social ties—without exchanging raw data. Yet accuracy alone does not make such systems deployable: recent regulatory and scientific frameworks (EU AI Act, NIST AI Risk Management Framework) demand that automated decision systems be simultaneously <i>fair</i>, <i>private</i>, <i>robust</i>, and <i>transparent</i>. Existing work addresses these properties in isolation, and no prior method combines fairness, differential privacy, and Byzantine robustness for federated graph neural networks (GNNs)."
    abstract_p2 = "<b>Framework:</b> We present <b>TrustFedGNN</b>, a unified trustworthy federated graph-learning framework built from three components: (i) <i>Fairness-Sensitive Edge Reweighting</i> (FSER), which suppresses biased cross-group message passing; (ii) <i>Fairness-Targeted Gradient Decomposition</i> (FTGD), which applies differential privacy to the released fairness statistic by privatising only the low-dimensional demographic-parity means—so strong DP for the fairness signal costs O(1) noise where full-gradient DP-SGD injects O(sqrt(|theta|)) and destroys utility; and (iii) <i>Bi-objective Frank-Wolfe Aggregation</i> (BFWA) and its Byzantine-robust variant, which impose an operator-chosen constraint on the aggregated client-disparity statistic while screening malicious clients, including a new fairness-poisoning threat."
    abstract_p3 = "<b>Results:</b> We further instrument the framework with predictive uncertainty, GNN attention audits, a composite trust score, sustainability accounting, and an EU AI Act / NIST RMF compliance mapping. Across five real datasets spanning credit, recidivism, social, and large-scale crypto (Elliptic Bitcoin, 203,769 nodes) domains, TrustFedGNN is the only method among 16 baselines to combine competitive fairness and utility with sensitive-attribute DP and Byzantine robustness in a single federated GNN. All tables and figures are regenerated from logged runs by a single script."

    abstract_content = [
        [Paragraph(abstract_p1 + "<br/><br/>" + abstract_p2 + "<br/><br/>" + abstract_p3, style_abstract)]
    ]
    t_abstract = Table(abstract_content, colWidths=[7.25 * inch])
    t_abstract.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), BG_LIGHT),
        ('BOX', (0,0), (-1,-1), 0.75, colors.HexColor("#CBD5E1")),
        ('LINELEFT', (0,0), (-1,-1), 3.5, PRIMARY),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(t_abstract)
    story.append(Spacer(1, 6))

    # Paper Structure Section
    story.append(Paragraph("1. Paper Structure Overview", style_h1))
    structure_data = [
        [Paragraph("Section", style_th), Paragraph("Core Technical Content & Scope", style_th)],
        [Paragraph("<b>§1 Introduction</b>", style_td), Paragraph("Banking fraud motivation; multi-stakeholder tensions; gap analysis; 5 formal contributions.", style_td)],
        [Paragraph("<b>§2 Related Work</b>", style_td), Paragraph("Taxonomy across 5 perspectives: Fair GNN (7), Fair FL (12), DP-FL (4), Byzantine (7), Trust metrics (4).", style_td)],
        [Paragraph("<b>§3 Preliminaries</b>", style_td), Paragraph("Cross-silo formulation; DPD, EOD, EqOdds definitions; (epsilon, delta)-DP and RDP accountant; threat models.", style_td)],
        [Paragraph("<b>§4 Method</b>", style_td), Paragraph("FSER attention debiasing; FTGD statistic DP & Proposition 1; BFWA Frank-Wolfe optimization; Algorithm 1.", style_td)],
        [Paragraph("<b>§5 Experiments</b>", style_td), Paragraph("16 baselines on 5 datasets; Main comparisons; Ablations; Privacy sweeps & attack; Robustness; 2.4M scaling.", style_td)],
        [Paragraph("<b>§6 Trust Analysis</b>", style_td), Paragraph("Composite trust score (power-mean); Attention explainability audit; EU AI Act / NIST RMF mapping.", style_td)],
        [Paragraph("<b>§7 Discussion</b>", style_td), Paragraph("Why FTGD works; Comparison with 2025-2026 SOTA; <b>11 explicit limitations</b>; Broader impact.", style_td)],
        [Paragraph("<b>Appendices A-C</b>", style_td), Paragraph("Full hyperparameters; Baseline reimplementation fidelity notes (kept vs. dropped); Dataset construction.", style_td)]
    ]
    t_struct = Table(structure_data, colWidths=[1.4 * inch, 5.85 * inch])
    t_struct.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), PRIMARY),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#E2E8F0")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, BG_LIGHT]),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 2.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
        ('LEFTPADDING', (0,0), (-1,-1), 5),
        ('RIGHTPADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(t_struct)
    story.append(Spacer(1, 6))

    # Five Key Contributions Section
    story.append(Paragraph("2. Five Key Contributions & Empirical Evidence", style_h1))
    
    story.append(Paragraph("<b>Contribution 1: FSER — Fairness-Sensitive Edge Reweighting</b>", style_h2))
    story.append(Paragraph("• <i>Mechanism:</i> Corrects attention weights via e_tilde_{ij} = e_{ij} - beta &middot; I(s_i != s_j) &middot; max(0, cos(h_i, h_j)), where beta in [0, 5] is learned per layer.", style_bullet))
    story.append(Paragraph("• <i>Empirical Grounding:</i> On Pokec-z (67.8k nodes, natural social graph), FSER delivers +0.0201 AUC gain (p = 0.0059*, Holm-Bonferroni significant).", style_bullet))
    story.append(Paragraph("• <i>Honest Boundary:</i> Described as a <b>learnable structural regularizer</b> rather than a certified causal debiaser.", style_bullet))

    story.append(Paragraph("<b>Contribution 2: FTGD — Fairness-Targeted Gradient Decomposition</b>", style_h2))
    story.append(Paragraph("• <i>Mechanism:</i> Exploits the fact that sensitive attribute s touches objective solely via 2D means (mu_0, mu_1), injecting O(1) noise instead of O(sqrt(|theta|)).", style_bullet))
    story.append(Paragraph("• <i>Empirical Grounding:</i> Retains full utility (Delta AUC &lt; 0.001) while Full DP-SGD collapses to AUC = 0.556 on ogbn-products (2.4M nodes).", style_bullet))
    story.append(Paragraph("• <i>Adversarial Validation:</i> Exact release yields 100% attribute inference; FTGD at epsilon=8 drives inference down to 51% (random chance).", style_bullet))

    story.append(Paragraph("<b>Contribution 3: BFWA — Bi-objective Frank-Wolfe Aggregation</b>", style_h2))
    story.append(Paragraph("• <i>Mechanism:</i> Solves max Sum w_k &middot; Perf_k s.t. Sum w_k &middot; DPD_k &le; tau on Simplex Delta_K with O(K) complexity and distance-to-median Byzantine filtering.", style_bullet))
    story.append(Paragraph("• <i>Empirical Grounding:</i> robust_bfwa is the <b>only aggregator</b> sustaining AUC &ge; 0.95 across Gaussian, ALIE, and Fairness-Poisoning attacks simultaneously.", style_bullet))

    story.append(Paragraph("<b>Contribution 4: Trustworthiness Instrumentation Layer</b>", style_h2))
    story.append(Paragraph("• Multi-faceted audit suite: MC-dropout uncertainty & calibration (ECE, Brier), attention bias ratios, transparent power-mean trust score, and EU AI Act (Arts 10, 13-15) / NIST RMF design mappings + auto-generated model card.", style_bullet))

    story.append(Paragraph("<b>Contribution 5: Large-Scale Rigorous Benchmarking</b>", style_h2))
    story.append(Paragraph("• 5 real datasets, 16 baselines (including 5 reimplemented 2025-2026 methods), ogbn-products 2.4M scaling, paired Wilcoxon signed-rank tests (n=10 seeds), and single-script reproducibility.", style_bullet))
    story.append(Spacer(1, 6))

    # Main Results Tables
    story.append(Paragraph("3. Verified Main Benchmark Results", style_h1))
    story.append(Paragraph("<b>Table 1: Main Benchmark on Pokec-z Graph (N = 67,796 nodes, 1.24M edges, n = 10 independent random seeds)</b>", style_meta))
    
    pokec_data = [
        [Paragraph("Method", style_th), Paragraph("Venue", style_th), Paragraph("DP", style_th), Paragraph("AUC-ROC ↑", style_th), Paragraph("DPD_hard ↓", style_th), Paragraph("EOD ↓", style_th), Paragraph("Omega_w ↓", style_th)],
        [Paragraph("FedAvg-GCN", style_td), Paragraph("AISTATS'17", style_td_center), Paragraph("✗", style_td_center), Paragraph("0.7272 ± 0.0122*", style_td_center), Paragraph("0.0400 ± 0.0116*", style_td_center), Paragraph("0.0588 ± 0.0314*", style_td_center), Paragraph("0.0000", style_td_center)],
        [Paragraph("FairGNN", style_td), Paragraph("WSDM'21", style_td_center), Paragraph("✗", style_td_center), Paragraph("0.5839 ± 0.0893*", style_td_center), Paragraph("0.0200 ± 0.0226", style_td_center), Paragraph("0.0075 ± 0.0101", style_td_center), Paragraph("0.0000", style_td_center)],
        [Paragraph("FairSIN", style_td), Paragraph("WWW'24", style_td_center), Paragraph("✗", style_td_center), Paragraph("0.7212 ± 0.0302*", style_td_center), Paragraph("0.0341 ± 0.0149*", style_td_center), Paragraph("0.0370 ± 0.0259", style_td_center), Paragraph("0.0000", style_td_center)],
        [Paragraph("FairFed", style_td), Paragraph("AAAI'23", style_td_center), Paragraph("✗", style_td_center), Paragraph("0.7232 ± 0.0162*", style_td_center), Paragraph("<b>0.0065 ± 0.0040*</b>", style_td_center), Paragraph("0.0168 ± 0.0237", style_td_center), Paragraph("0.0592", style_td_center)],
        [Paragraph("FairGFL", style_td), Paragraph("IEEE TPDS'26", style_td_center), Paragraph("✗", style_td_center), Paragraph("0.7318 ± 0.0066*", style_td_center), Paragraph("0.0399 ± 0.0095*", style_td_center), Paragraph("0.0618 ± 0.0376*", style_td_center), Paragraph("0.0000", style_td_center)],
        [Paragraph("FedGraph-Fair", style_td), Paragraph("InfoSci'26", style_td_center), Paragraph("✗", style_td_center), Paragraph("0.7159 ± 0.0103*", style_td_center), Paragraph("0.0367 ± 0.0067*", style_td_center), Paragraph("0.0567 ± 0.0251*", style_td_center), Paragraph("0.0165", style_td_center)],
        [Paragraph("CGSV", style_td), Paragraph("NeurIPS'21", style_td_center), Paragraph("✗", style_td_center), Paragraph("0.7390 ± 0.0054*", style_td_center), Paragraph("0.0435 ± 0.0077*", style_td_center), Paragraph("0.0540 ± 0.0341*", style_td_center), Paragraph("0.1467", style_td_center)],
        [Paragraph("Ours w/o FSER", style_td), Paragraph("Ablation", style_td_center), Paragraph("✓", style_td_center), Paragraph("0.7662 ± 0.0173*", style_td_center), Paragraph("0.0161 ± 0.0124", style_td_center), Paragraph("0.0246 ± 0.0144", style_td_center), Paragraph("0.1592", style_td_center)],
        [Paragraph("<b>TrustFedGNN (Ours)</b>", style_td_bold), Paragraph("Proposed", style_td_center), Paragraph("<b>✓</b>", style_td_center), Paragraph("<b>0.7862 ± 0.0105</b>", style_td_center), Paragraph("<b>0.0149 ± 0.0088</b>", style_td_center), Paragraph("<b>0.0219 ± 0.0148</b>", style_td_center), Paragraph("0.6348", style_td_center)]
    ]
    t_pokec = Table(pokec_data, colWidths=[1.3 * inch, 0.9 * inch, 0.4 * inch, 1.25 * inch, 1.2 * inch, 1.2 * inch, 0.9 * inch])
    t_pokec.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), PRIMARY),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E1")),
        ('ROWBACKGROUNDS', (0,1), (-1,-2), [colors.white, BG_LIGHT]),
        ('BACKGROUND', (0,-1), (-1,-1), colors.HexColor("#EFF6FF")),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 2.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
    ]))
    story.append(t_pokec)
    story.append(Paragraph("<font size=7 color='#64748B'>* Statistically significant difference vs. TrustFedGNN (two-sided Wilcoxon signed-rank test with family-wise Holm-Bonferroni correction, p &lt; 0.05). TrustFedGNN is the only method with active (epsilon=8.0, delta=10<sup>-5</sup>)-DP.</font>", style_body))
    story.append(Spacer(1, 5))

    # Boundary Analysis & Robustness
    story.append(Paragraph("<b>Table 2: Boundary Analysis (Credit 30k) & Byzantine Robustness Retention (Bail 18.8k)</b>", style_meta))
    
    robust_data = [
        [Paragraph("Aggregator", style_th), Paragraph("Gaussian AUC ↑", style_th), Paragraph("ALIE AUC ↑", style_th), Paragraph("Fair-Poison AUC ↑", style_th), Paragraph("Fair-Poison EOD ↓", style_th)],
        [Paragraph("FedAvg", style_td), Paragraph("0.511†", style_td_center), Paragraph("0.935", style_td_center), Paragraph("0.945", style_td_center), Paragraph("0.037", style_td_center)],
        [Paragraph("BFWA", style_td), Paragraph("0.483†", style_td_center), Paragraph("<b>0.991</b>", style_td_center), Paragraph("0.689", style_td_center), Paragraph("0.126 (High Bias)", style_td_center)],
        [Paragraph("Krum", style_td), Paragraph("0.986", style_td_center), Paragraph("0.710", style_td_center), Paragraph("<b>0.991</b>", style_td_center), Paragraph("<b>0.007</b>", style_td_center)],
        [Paragraph("<b>robust_bfwa (Ours)</b>", style_td_bold), Paragraph("<b>0.992</b>", style_td_center), Paragraph("<b>0.953</b>", style_td_center), Paragraph("<b>0.990</b>", style_td_center), Paragraph("<b>0.024</b>", style_td_center)]
    ]
    t_robust = Table(robust_data, colWidths=[1.75 * inch, 1.35 * inch, 1.35 * inch, 1.45 * inch, 1.35 * inch])
    t_robust.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), SECONDARY),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E1")),
        ('ROWBACKGROUNDS', (0,1), (-1,-2), [colors.white, BG_LIGHT]),
        ('BACKGROUND', (0,-1), (-1,-1), colors.HexColor("#F0FDF4")),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 2.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
    ]))
    story.append(t_robust)
    story.append(Paragraph("<font size=7 color='#64748B'>† Near-chance model collapse. robust_bfwa is the <b>only aggregator</b> sustaining AUC &ge; 0.95 across all three attack families.</font>", style_body))
    story.append(Spacer(1, 6))

    # 11 Limitations Section
    story.append(Paragraph("4. Transparently Acknowledged Limitations (11 Items)", style_h1))
    limits_text = """
    <b>1. DP Scope:</b> FTGD guarantees DP for released fairness statistics (mu_0, mu_1), not the entire parameter gradient update.<br/>
    <b>2. BFWA Surrogate:</b> tau bounds client-reported disparity; non-linear global aggregation means it is an operator target, not a certificate.<br/>
    <b>3. Metric Tension:</b> Soft-DPD optimization does not guarantee EOD improvement; base-rate disparities require domain alignment.<br/>
    <b>4. Computational Overhead:</b> FSER + FTGD adds ~2.4x wall-clock overhead over vanilla FedAvg (communication bytes remain identical).<br/>
    <b>5. Binary Protected Attributes:</b> Multi-group attributes currently handled via one-vs-rest reduction.<br/>
    <b>6. Structural Proxies:</b> Elliptic Bitcoin and ogbn-products evaluate temporal/degree subgroups, not protected demographics.<br/>
    <b>7. Simulated Silos:</b> Cross-silo subgraphs are partitioned from central benchmarks (cross-client edge modeling is future work).<br/>
    <b>8. Baseline Reimplementations:</b> Competitors ported to standardized pipeline; exact nuances documented in Appendix B.<br/>
    <b>9. Known Byzantine Count:</b> Screening relies on upper bound estimate f &lt; K/2.<br/>
    <b>10. Composite Score Calibration:</b> Trust score ranking is sensitive to reference constants and power-mean exponent p.<br/>
    <b>11. Attention Faithfulness:</b> FSER attention is an audit diagnostic of model behavior, not a certified causal explanation.
    """
    t_limits = Table([[Paragraph(limits_text, style_body)]], colWidths=[7.25 * inch])
    t_limits.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), BG_BOX),
        ('BOX', (0,0), (-1,-1), 0.75, colors.HexColor("#CBD5E1")),
        ('LINELEFT', (0,0), (-1,-1), 3.5, ACCENT_AMBER),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING', (0,0), (-1,-1), 7),
        ('RIGHTPADDING', (0,0), (-1,-1), 7),
    ]))
    story.append(t_limits)
    story.append(Spacer(1, 6))

    # Questions for Reviewers Section
    story.append(Paragraph("5. Specific Reviewer Feedback Requests (8 Questions)", style_h1))

    questions = [
        ("Q1 (Scope of Privacy Guarantee)", 
         "Our DP guarantee covers the released fairness statistics (2D) rather than the entire model update. We explicitly document this distinction throughout the manuscript. Does this transparent scoping satisfy technical rigor, or would additional update-level empirical leakage audits be required for a top-tier Q1 assessment?"),
        
        ("Q2 (Scientific Reframing of Hypotheses)", 
         "Of our four pre-registered hypotheses, three were refuted under strict multi-seed testing (e.g. FSER benefit is driven by structural regularization rather than homophily h_s). We report all refutations openly. Does this rigorous empirical honesty strengthen scientific credibility in your review?"),
        
        ("Q3 (FSER Characterization as Heuristic)", 
         "FSER is framed as a learnable structural regularization heuristic rather than a certified causal debiaser. Given its verified empirical utility (+0.0201 AUC on Pokec-z, p=0.0059*), is this empirical justification sufficient without closed-form PAC bounds?"),
        
        ("Q4 (Baseline Reimplementation Fidelity)", 
         "Five 2025-2026 competitors (FairGFL, FedGraph-Fair, PUFFLE, FedFACT, PoPETs'25) were reimplemented from published specifications (no public GFL code exists). Appendix B itemizes all preserved and adapted components. Are these controls sufficient?"),
        
        ("Q5 (Simulated Federation Paradigm)", 
         "Experiments use Dirichlet-partitioned graph benchmarks where cross-client edges are dropped. While standard in federated graph literature, how critically would reviewers view this relative to real-world deployment?"),
        
        ("Q6 (Temporal/Structural Proxy Attributes)", 
         "On Elliptic Bitcoin (203.7k nodes) and ogbn-products (2.4M nodes), we use documented temporal and degree subgroups as fairness proxies. Is our careful framing as 'subgroup-disparity analysis' appropriate?"),
        
        ("Q7 (FU-Shapley Incentive Alignment)", 
         "With exact Shapley approximation refuted (L_1 = 0.76 > 0.15), we reposition FU-Shapley as an O(KP) first-order ranking heuristic (Spearman rho = 0.69). Does this pragmatic repositioning sound solid?"),
        
        ("Q8 (Overall Manuscript Balance & Length)", 
         "At ~30 pages (1,779 lines LaTeX with appendices), the manuscript is comprehensive across 5 trust dimensions. Should specific sections (e.g. Trustworthiness analysis or Related Work) be condensed for publication?"),
    ]

    for q_title, q_desc in questions:
        story.append(Paragraph(f"<b>{q_title}</b>", style_q_title))
        story.append(Paragraph(q_desc, style_q_body))

    story.append(Spacer(1, 4))

    # Reproducibility Footer
    repro_text = "<b>Reproducibility & Provenance:</b> All code, configs, random seeds, unit tests (46/46 passing), and JSON run manifests (Git commit tagged) are fully automated and available for audit. Tables & figures regenerate via <code>python -m experiments.report</code>."
    t_repro = Table([[Paragraph(repro_text, style_body)]], colWidths=[7.25 * inch])
    t_repro.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#F8FAFC")),
        ('BOX', (0,0), (-1,-1), 0.75, colors.HexColor("#E2E8F0")),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING', (0,0), (-1,-1), 7),
        ('RIGHTPADDING', (0,0), (-1,-1), 7),
    ]))
    story.append(t_repro)

    # Build PDF
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Successfully generated PDF at: {output_pdf_path}")

if __name__ == "__main__":
    out_path = sys.argv[1] if len(sys.argv) > 1 else "review_request_document.pdf"
    build_pdf(out_path)
