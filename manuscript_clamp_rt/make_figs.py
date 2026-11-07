"""Figures of the CLAMP-RT manuscript, from the summary files used by make_tables.py.

Usage (from the repository root):  python3 manuscript_clamp_rt/make_figs.py
Writes manuscript_clamp_rt/figures/fig_liars.pdf and fig_privacy.pdf.
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_tables import ATT, ATT_NAME, DS_NAME, parse_e4_nodb, parse_summary  # noqa: E402

FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(FIG, exist_ok=True)
# Fixed categorical order (blue, orange, aqua, yellow, magenta); markers give a second encoding.
COL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
MRK = ["o", "s", "^", "D", "v"]
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3dd"

plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                     "axes.spines.right": False, "legend.frameon": False, "lines.linewidth": 1.5})


def fig_liars():
    s = parse_summary("db_e4.txt")
    old = parse_e4_nodb()
    dss = ["credit", "pokec_n"]
    fig, axes = plt.subplots(2, 2, figsize=(6.4, 4.2), sharey=True)
    for i, ds in enumerate(dss):
        for j, K in enumerate((5, 10)):
            ax = axes[i, j]
            b = 1 if K == 5 else 2
            fs = [0, 1, 2] if K == 5 else [0, 1, 2, 3]
            ax.axvspan(b + 0.5, fs[-1] + 0.3, color=GRID, zorder=0)
            ax.text(b + 0.55, 0.30, "f > b", color=MUTED, fontsize=7, va="bottom")
            for n, a in enumerate(ATT):
                y = [s[(ds, K, "none", 0)]["dpd"][0]] + [s[(ds, K, a, f)]["dpd"][0] for f in fs[1:]]
                ax.plot(fs, y, color=COL[n], marker=MRK[n], ms=4, label=f"CLAMP-RT, {ATT_NAME[a]}")
            yo = [old[(ds, K, "clamp-f", "none", 0)]["dpd"]] + \
                 [old[(ds, K, "clamp-f", "stat_falsify", f)]["dpd"] for f in fs[1:]]
            ax.plot(fs, yo, color=COL[4], marker=MRK[4], ms=4, ls="--",
                    label="CLAMP (sums), swap")
            ax.set_xticks(fs)
            ax.grid(axis="y", color=GRID, lw=0.6)
            ax.set_title(f"{DS_NAME[ds]}, K = {K} (b = {b})", fontsize=8, color=INK)
            if j == 0:
                ax.set_ylabel("test DPD")
            if i == 1:
                ax.set_xlabel("number of lying clients f")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=7, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(os.path.join(FIG, "fig_liars.pdf"), bbox_inches="tight")
    plt.close(fig)


def fig_privacy():
    s = parse_summary("db_e5.txt")
    dss = ["credit", "pokec_z", "pokec_n"]
    eps = [1.0, 2.0, 4.0, 8.0]
    fig, axes = plt.subplots(1, 3, figsize=(6.4, 2.2), sharey=True)
    for i, ds in enumerate(dss):
        ax = axes[i]
        for n, P in enumerate((1, 5, 10)):
            m = [s[(ds, e, P)]["dpd"][0] for e in eps]
            sd = [s[(ds, e, P)]["dpd"][1] for e in eps]
            x = [e * (1 + 0.06 * (n - 1)) for e in eps]
            ax.errorbar(x, m, yerr=sd, color=COL[n], marker=MRK[n], ms=4, capsize=2, lw=1.5,
                        elinewidth=0.8, label=f"P = {P}")
        ax.axhline(0.03, color=MUTED, ls=":", lw=1)
        ax.set_xscale("log", base=2)
        ax.set_xticks(eps)
        ax.set_xticklabels(["1", "2", "4", "8"])
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_title(DS_NAME[ds], fontsize=8, color=INK)
        ax.set_xlabel(r"$\varepsilon$")
        if i == 0:
            ax.set_ylabel("test DPD")
    axes[0].legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_privacy.pdf"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig_liars()
    fig_privacy()
    print("wrote", sorted(os.listdir(FIG)))
