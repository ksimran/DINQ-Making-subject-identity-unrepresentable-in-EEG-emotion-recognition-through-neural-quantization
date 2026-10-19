"""Regenerate the paper figures and tables from results/.

Usage:  python scripts/make_paper_figures.py
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch, Patch, Polygon, Rectangle
from matplotlib.text import Text
from matplotlib.lines import Line2D

REPO = Path(__file__).resolve().parents[1]
FIG = REPO / "paper" / "figures"
TAB = REPO / "paper" / "tables"
MANIFEST: list[tuple[str, str, str]] = []  # (artifact, anchor, status)

# Okabe-Ito colour-blind-safe palette.
OI = {"orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73", "yellow": "#F0E442",
      "blue": "#0072B2", "vermillion": "#D55E00", "purple": "#CC79A7", "black": "#000000"}
BASE_C, DINQ_C = OI["vermillion"], OI["blue"]
CHANCE_C, CEIL_C = "#8a8a8a", "#404040"

# Elsevier artwork widths. Figures are drawn at final size so \includegraphics
# places them unscaled and the font sizes below are the printed sizes.
MM = 1 / 25.4
W_1COL, W_15COL = 90 * MM, 140 * MM
MIN_FONT_PT = 7.0
CORPUS_NAME = {"deap": "DEAP", "seedv": "SEED-V", "faced": "FACED"}


def style():
    plt.rcParams.update({
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        # Liberation Sans is metric-compatible with Arial (Elsevier's preferred face).
        "font.family": "sans-serif", "font.sans-serif": ["Liberation Sans", "Arial", "DejaVu Sans"],
        "mathtext.fontset": "custom", "mathtext.rm": "Liberation Sans",
        "mathtext.it": "Liberation Sans:italic", "mathtext.bf": "Liberation Sans:bold",
        "mathtext.cal": "Liberation Sans:italic",
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.8, "figure.dpi": 150, "legend.frameon": False,
        # 'standard' bbox keeps the exact figsize width (a tight bbox would crop it).
        "figure.constrained_layout.use": True, "savefig.bbox": "standard",
    })


def _min_font(fig) -> float:
    sizes = [t.get_fontsize() for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip()]
    return min(sizes) if sizes else float("nan")


def save(fig, name: str, anchor: str):
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "svg"):
        fig.savefig(FIG / f"{name}.{ext}")
    w_mm, h_mm = fig.get_size_inches() / MM
    mf = _min_font(fig)
    plt.close(fig)
    MANIFEST.append((f"{name}.pdf/.svg", anchor, "BUILT"))
    warn = f"  WARNING: font below {MIN_FONT_PT} pt" if mf < MIN_FONT_PT else ""
    print(f"[built] {name}  -> {anchor}  ({w_mm:.0f} x {h_mm:.0f} mm, min font {mf:.1f} pt){warn}")


def flag(name: str, anchor: str, missing: str):
    MANIFEST.append((name, anchor, f"BLOCKED — needs: {missing}"))
    print(f"[FLAG]  {name}: missing {missing}")


def ci95(vals):
    """(mean, lo, hi) 95% t-CI; degenerate n<2 -> zero-width."""
    v = np.asarray(vals, float)
    m = float(v.mean())
    if v.size < 2:
        return m, m, m
    from scipy import stats
    h = float(stats.t.ppf(0.975, v.size - 1) * v.std(ddof=1) / np.sqrt(v.size))
    return m, m - h, m + h


def load_cells(subdir: str):
    """{config: [cell,...]} from results/<subdir>/cells/*__split*.json."""
    out: dict[str, list] = {}
    for f in sorted(glob.glob(str(REPO / "results" / subdir / "cells" / "*.json"))):
        c = json.load(open(f))
        out.setdefault(c["config"], []).append(c)
    return out


def agg(cells, path):
    """Aggregate a nested field (e.g. 'recov_heldout.hgb' or 'emotion_unseen') across splits."""
    def get(c):
        o = c
        for p in path.split("."):
            o = o[p]
        return o
    return ci95([get(c) for c in cells])


def maxp(c):
    """Held-out subject recoverability: max over the linear, MLP and HGB probes."""
    r = c["recov_heldout"]
    return max(r["linear"], r["mlp"], r["hgb"])


MATCHED = REPO / "results" / "faced_epochs_check" / "like_for_like_60ep.json"


def _dinq_matched():
    """Per-split DINQ (vq512_conU) results at 60 epochs / cap 8000, matching every other
    FACED row: the seeded faced_permtest z_c, with emotion probed by faced_epochs_check.py.
    The 80-epoch cells in results/dinq_sweep_faced are archival and not used."""
    if not MATCHED.exists():
        return None
    return json.loads(MATCHED.read_text())["rows"]


def fig_headline():
    cells = _dinq_matched()
    if cells is None:
        return flag("Fig-HEADLINE", "§4.3", f"{MATCHED.relative_to(REPO)} (run scripts/faced_epochs_check.py)")
    dinq_emo = agg(cells, "emotion_unseen")
    K = int(cells[0]["K"])
    # Recentring bars from the 8-split `recentering` cells so both carry a 95% CI.
    bl = _subj_baseline_cells()
    if not bl or "recentering" not in bl:
        return flag("Fig-HEADLINE", "§4.3", "results/baselines_faced_subject/cells/recentering__split*.json")
    rc = bl["recentering"]
    base_recov, base_lo, base_hi = ci95([maxp(c) for c in rc])
    base_emo = agg(rc, "emotion_unseen")
    # DINQ recoverability and the null ceiling come from the permutation test (verdict.json).
    pt = _permtest()
    if pt is None:
        return flag("Fig-HEADLINE", "§4.3", "results/faced_permtest/verdict.json (settled numbers)")
    v = pt[0]
    dinq_recov = (v["observed_mean"], v["observed_ci95"][0], v["observed_ci95"][1])
    ceiling = float(v["null_95pct"])

    fig, (a, b) = plt.subplots(1, 2, figsize=(W_15COL, 62 * MM))
    # Panel A: held-out subject recoverability. The x-ticks name the bars, so the legend
    # carries only the reference lines.
    xs = [0, 1]
    a.bar(0, base_recov, 0.55, color=BASE_C)
    a.bar(1, dinq_recov[0], 0.55, color=DINQ_C)
    a.errorbar(0, base_recov, yerr=[[base_recov - base_lo], [base_hi - base_recov]],
               fmt="none", ecolor=OI["black"], elinewidth=1.1, capsize=3)
    a.errorbar(1, dinq_recov[0], yerr=[[dinq_recov[0] - dinq_recov[1]], [dinq_recov[2] - dinq_recov[0]]],
               fmt="none", ecolor=OI["black"], elinewidth=1.1, capsize=3)
    a.axhline(ceiling, ls="--", lw=1.2, color=CEIL_C, label=f"Null ceiling ({ceiling:.3f})")
    a.axhline(1 / K, ls=":", lw=1.0, color=CHANCE_C, label=f"Chance (1/{K})")
    a.set_xticks(xs); a.set_xticklabels(["Recentring", "DINQ"])
    a.set_ylabel("Held-out subject recoverability\n(max over probes)")
    a.text(0, base_hi + 0.010, f"{base_recov:.3f}", ha="center", fontsize=7.5)
    a.text(1, dinq_recov[2] + 0.010, f"{dinq_recov[0]:.3f}", ha="center", fontsize=7.5)
    a.legend(loc="upper right")
    a.set_ylim(0, max(base_recov * 1.25, 0.3))

    # Panel B: emotion accuracy on unseen subjects
    b.bar(0, base_emo[0], 0.55, color=BASE_C)
    b.bar(1, dinq_emo[0], 0.55, color=DINQ_C)
    for xi, e in ((0, base_emo), (1, dinq_emo)):
        b.errorbar(xi, e[0], yerr=[[e[0] - e[1]], [e[2] - e[0]]],
                   fmt="none", ecolor=OI["black"], elinewidth=1.1, capsize=3)
        b.text(xi, e[2] + 0.012, f"{e[0]:.3f}", ha="center", fontsize=7.5)
    b.axhline(0.5, ls=":", lw=1.0, color=CHANCE_C, label="Chance (0.50)")
    b.set_xticks(xs); b.set_xticklabels(["Recentring", "DINQ"])
    b.set_ylabel("Emotion accuracy\n(unseen subjects)")
    b.set_ylim(0.45, 0.72); b.legend(loc="upper right")

    save(fig, "fig_headline", "§4.3")
    MANIFEST.append(("Fig-HEADLINE note", "§4.3",
                     f"recentring recov {base_recov:.3f} [{base_lo:.3f}, {base_hi:.3f}], emotion "
                     f"{base_emo[0]:.3f} [{base_emo[1]:.3f}, {base_emo[2]:.3f}] (8 recentering cells); DINQ recov "
                     f"{dinq_recov[0]:.3f} [{dinq_recov[1]:.3f}, {dinq_recov[2]:.3f}] (faced_permtest), emotion "
                     f"{dinq_emo[0]:.3f} [{dinq_emo[1]:.3f}, {dinq_emo[2]:.3f}] (matched 60 ep / cap 8000); "
                     f"ceiling {ceiling:.3f}; recov = max over probes"))


CODEBOOK_ORDER = ["vq128", "vq256", "vq512", "vq1024", "fsq512", "vq512_conU"]
CODEBOOK_FACED = "codebook_sweep_faced"  # 60 epochs, cap 8000, seeded per cell
# Label positions in data coordinates: (x, y, horizontal alignment). The points sit
# close together, so labels go in clear gutters with leader lines.
CODEBOOK_LAB_SEEDV = {"vq128": (0.245, 0.645, "right"), "vq1024": (0.245, 0.718, "right"),
                      "vq512": (0.362, 0.738, "left"), "vq256": (0.362, 0.716, "left"),
                      "vq512_conU": (0.362, 0.648, "left"), "fsq512": (0.345, 0.572, "left")}


# FACED: five configurations land within ~0.003 of each other with overlapping error
# bars, so any leader line would cross a point; they get distinct markers and a legend.
CODEBOOK_LAB_FACED = None
CODEBOOK_XLIM_FACED = (0.025, 0.08)
CODEBOOK_MARKER = {"vq128": "o", "vq256": "s", "vq512": "^", "vq1024": "D", "fsq512": "v", "vq512_conU": "P"}


def _codebook_complete(sweep, n_splits=8):
    return all(len(sweep.get(c, [])) >= n_splits for c in CODEBOOK_ORDER)


def fig_codebook(subdir, out, anchor, lab, xlim=None, use_max=False):
    """Emotion vs held-out subject recoverability per codebook configuration.
    SEED-V (archival): HGB probe against its per-cell HGB permutation null.
    FACED (use_max): max over probes against the shared max-over-probes null of
    faced_permtest, i.e. the same quantities as Table 3."""
    sweep = load_cells(subdir)
    if not sweep:
        return flag(out, anchor, f"results/{subdir}/cells")
    fig, ax = plt.subplots(figsize=(W_1COL, 72 * MM))
    ceilings, chances = [], []
    order = [c for c in CODEBOOK_ORDER if c in sweep]
    name = {"vq128": "K=128", "vq256": "K=256", "vq512": "K=512", "vq1024": "K=1024",
            "fsq512": "FSQ", "vq512_conU": "K=512,\ncontrastive on u"}
    recov = maxp if use_max else (lambda c: c["recov_heldout"]["hgb"])
    cfg_handles = []
    for i, cfg in enumerate(order):
        cells = sweep[cfg]
        r = ci95([recov(c) for c in cells]); e = agg(cells, "emotion_unseen")
        ceilings.append(np.mean([c["ceiling_heldout"] for c in cells]))
        chances.append(1 / cells[0]["recov_heldout"]["K"])
        col = OI["green"] if cfg.startswith("fsq") else (OI["orange"] if "conU" in cfg else DINQ_C)
        if lab is None:  # hollow markers keep near-coincident configurations visible
            ax.errorbar(r[0], e[0], xerr=[[r[0] - r[1]], [r[2] - r[0]]], yerr=[[e[0] - e[1]], [e[2] - e[0]]],
                        fmt=CODEBOOK_MARKER[cfg], ms=5.5, mfc="none", mec=col, mew=1.3, color=col,
                        ecolor=col, elinewidth=0.8, capsize=2, zorder=3)
            cfg_handles.append(Line2D([], [], ls="none", marker=CODEBOOK_MARKER[cfg], ms=5.5, mfc="none",
                                      mec=col, mew=1.3, label=name.get(cfg, cfg).replace("\n", " ")))
        else:
            ax.errorbar(r[0], e[0], xerr=[[r[0] - r[1]], [r[2] - r[0]]],
                        yerr=[[e[0] - e[1]], [e[2] - e[0]]], fmt="o", ms=5, color=col,
                        ecolor=col, elinewidth=1, capsize=2, zorder=3)
            lx, ly, ha = lab.get(cfg, (r[0] + 0.01, e[0] + 0.01, "left"))
            ax.annotate(name.get(cfg, cfg), (r[0], e[0]), xytext=(lx, ly), textcoords="data",
                        ha=ha, va="center", fontsize=7, zorder=5,
                        arrowprops=dict(arrowstyle="-", lw=0.5, color="#9a9a9a", shrinkA=1, shrinkB=3))
    if use_max:
        pt = _permtest(); ceil = pt[0]["null_95pct"] if pt else SHARED_NULL_K31
        hn = ax.axvline(ceil, ls="--", lw=1.2, color=CEIL_C, label=f"Null ceiling ({ceil:.3f})")
        hc = ax.axvline(np.mean(chances), ls=":", lw=1.0, color=CHANCE_C,
                        label=f"Chance (1/{round(1 / np.mean(chances))})")
        ax.set_xlabel("Held-out subject recoverability\n(max over probes, lower is better)")
    else:
        hn = ax.axvline(np.mean(ceilings), ls="--", lw=1.2, color=CEIL_C,
                        label=f"Permutation-null ceiling (mean {np.mean(ceilings):.2f})")
        hc = ax.axvline(np.mean(chances), ls=":", lw=1.0, color=CHANCE_C, label=f"Chance ({np.mean(chances):.2f})")
        ax.set_xlabel("Held-out subject recoverability (lower is better)")
    ax.axhline(0.5, ls=":", lw=1.0, color=CHANCE_C)
    if xlim:
        ax.set_xlim(*xlim)
    ax.set_ylabel("Emotion accuracy (unseen subjects)")
    if lab is None:
        ax.legend(handles=cfg_handles, loc="center right", bbox_to_anchor=(1.0, 0.42),
                  handletextpad=0.4, borderaxespad=0.3)
    fig.legend(handles=[hn, hc], loc="outside upper center", ncols=2, columnspacing=1.0, handlelength=1.8)
    save(fig, out, anchor)


def fig_gen():
    seed, faced = load_cells("dinq_sweep"), load_cells("dinq_sweep_faced")
    cfg = "vq1024"
    if cfg not in seed or cfg not in faced:
        return flag("Fig-GEN", "appendix", f"{cfg} in both dinq_sweep and dinq_sweep_faced")
    fig, ax = plt.subplots(figsize=(4.0, 3.2))
    for i, (name, cells) in enumerate([("SEED-V\n(K=4, weak)", seed[cfg]), ("FACED\n(K=31, strong)", faced[cfg])]):
        r = agg(cells, "recov_heldout.hgb"); K = cells[0]["recov_heldout"]["K"]
        ceil = float(np.mean([c["ceiling_heldout"] for c in cells]))
        ax.bar(i, r[0], 0.5, color=DINQ_C, zorder=2)
        ax.errorbar(i, r[0], yerr=[[r[0] - r[1]], [r[2] - r[0]]], fmt="none",
                    ecolor=OI["black"], elinewidth=1.1, capsize=3, zorder=3)
        ax.hlines(1 / K, i - 0.3, i + 0.3, ls=":", lw=1.4, color=CHANCE_C, zorder=4)
        ax.hlines(ceil, i - 0.3, i + 0.3, ls="--", lw=1.4, color=CEIL_C, zorder=4)
        ax.text(i, r[0] + 0.01, f"{r[0]:.3f}", ha="center", fontsize=8)
        ax.text(i + 0.33, 1 / K, f"chance {1/K:.3f}", va="center", fontsize=7, color=CHANCE_C)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["SEED-V\n(K=4)", "FACED\n(K=31)"])
    ax.set_ylabel("DINQ held-out subject-recoverability")
    ax.set_title("SEED-V residual closes under the strong (K=31) test")
    ax.plot([], [], "--", color=CEIL_C, label="perm-null ceiling")
    ax.plot([], [], ":", color=CHANCE_C, label="chance 1/K")
    ax.legend(loc="upper right", fontsize=7)
    save(fig, "fig_gen", "appendix")


def fig_xcorp():
    files = sorted(glob.glob(str(REPO / "results" / "cross_corpus" / "*.json")))
    if not files:
        return flag("Fig-XCORP", "§5.2", "results/cross_corpus")
    pairs, base, dinq, nulls = [], [], [], []
    for f in files:
        o = json.load(open(f))
        if "dinq" not in o:
            continue
        pairs.append(f"{CORPUS_NAME.get(o['source'], o['source'])}\n"
                     f"to {CORPUS_NAME.get(o['target'], o['target'])}")
        base.append(o["baseline"]["domain_deep_acc"]["point"])
        dinq.append(o["dinq"]["domain_deep_acc"]["point"])
        nulls.append(o["domain_perm_null"]["p95"])
    x = np.arange(len(pairs)); w = 0.38
    fig, ax = plt.subplots(figsize=(W_15COL, 66 * MM))
    hb = ax.bar(x - w / 2, base, w, color=BASE_C)
    hd = ax.bar(x + w / 2, dinq, w, color=DINQ_C)
    hs = ax.axhspan(min(nulls), max(nulls), color=CHANCE_C, alpha=0.18, zorder=0)
    hn, = ax.plot([], [], ls="--", lw=1.2, color=CEIL_C)
    ax.axhline(np.mean(nulls), ls="--", lw=1.2, color=CEIL_C)
    hc = ax.axhline(0.5, ls=":", lw=1.0, color=CHANCE_C)
    ax.set_xticks(x); ax.set_xticklabels(pairs)
    ax.set_ylabel("Dataset recoverability (max over probes)")
    ax.set_ylim(0.4, 1.0)
    # Bars fill every column, so the legend sits above the axes.
    fig.legend([hb, hd, (hs, hn), hc],
               ["Recentred baseline", "DINQ",
                f"Permutation-null 95th percentile ({min(nulls):.2f}–{max(nulls):.2f})", "Chance (0.50)"],
               loc="outside upper center", ncols=2)
    save(fig, "fig_xcorp", "§5.2")


def fig_arch():
    # Schematic in the paper's Computer Modern font: feature pipeline on top, the
    # model and the probes below. The axes span the figure in mm, so coordinates are
    # printed millimetres. Signals, matrices and codes are illustrative (seeded).
    rc = {"font.family": "serif", "font.serif": ["cmr10"], "mathtext.fontset": "cm",
          "axes.formatter.use_mathtext": True, "axes.unicode_minus": False}
    bold = FontProperties(family="cmb10", size=8.5)
    panel_c = {"pipe": "#EAF2FB", "model": "#EEF6E8", "eval": "#FFF6DE"}
    edge_c = {"pipe": "#5B8DB8", "model": "#6A9F58", "eval": "#C9A227"}
    box_c = {"model": "#D5E8C8", "eval": "#FCE8B2"}
    ink, grey, code_c = "#333333", "#555555", OI["vermillion"]
    blues = ["#9DC3E6", "#BDD7EE", "#6FA8DC", "#DEEBF7", "#2E75B6"]
    rng = np.random.default_rng(3)
    W, H = 140, 79

    with plt.rc_context(rc):
        fig = plt.figure(figsize=(W * MM, H * MM))
        fig.set_layout_engine("none")  # absolute mm layout; layout="none" alone loses to the rc default
        ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
        ax.set_xlim(0, W); ax.set_ylim(0, H)

        def panel(x0, y0, x1, y1, key, title):
            ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0,rounding_size=1.5",
                                        fc=panel_c[key], ec=edge_c[key], lw=0.6, zorder=0))
            ax.text(x0 + 5.5, y1 - 2.2, title, fontproperties=bold, color=ink, va="top", zorder=5)

        def arrow(p, q, **kw):
            style = dict(arrowstyle="-|>", mutation_scale=7, lw=0.8, color=ink, shrinkA=0, shrinkB=0, zorder=4)
            style.update(kw)
            ax.add_patch(FancyArrowPatch(p, q, **style))

        def label(x, y, s, size=7.5, color=ink, ha="center"):
            ax.text(x, y, s, ha=ha, va="center", fontsize=size, color=color, zorder=6)

        def vec(x0, y0, w, h, n, colors):
            for i in range(n):
                ax.add_patch(Rectangle((x0, y0 + i * h / n), w, h / n, fc=colors[i % len(colors)],
                                       ec=ink, lw=0.4, zorder=3))

        def trapezoid(x0, x1, yc, h0, h1, name, role):
            ax.add_patch(Polygon([(x0, yc - h0 / 2), (x1, yc - h1 / 2), (x1, yc + h1 / 2), (x0, yc + h0 / 2)],
                                 closed=True, fc=box_c["model"], ec=edge_c["model"], lw=0.8, zorder=3))
            label((x0 + x1) / 2, yc + 1.2, name, size=8.5)
            label((x0 + x1) / 2, yc - 2.3, role, size=7)

        def rbox(x0, y0, w, h, key, text=""):
            ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=0,rounding_size=0.9",
                                        fc=box_c[key], ec=edge_c[key], lw=0.7, zorder=3))
            if text:
                label(x0 + w / 2, y0 + h / 2, text, size=8)

        # Feature pipeline.
        panel(1.5, 46.5, 138.5, 77.5, "pipe", "Feature pipeline")
        yc = 62.5
        t = np.linspace(0, 1, 200)
        ax.add_patch(Rectangle((4, 54), 19, 17, fc="white", ec=edge_c["pipe"], lw=0.6, zorder=2))
        for k in range(5):
            sig = (0.9 * np.sin(2 * np.pi * (3 + 2 * k) * t + k) + 0.5 * np.sin(2 * np.pi * 11 * t + 2 * k)
                   + 0.35 * rng.standard_normal(t.size))
            ax.plot(5 + 17 * t, 56.5 + 3 * k + 0.55 * sig, lw=0.45, color=["#1f4e79", "#2e75b6"][k % 2], zorder=3)
        label(13.5, 51.4, "EEG trial"); label(13.5, 48.6, r"$x \in \mathbb{R}^{C \times T}$")
        arrow((23.6, yc), (28.6, yc))
        ax.add_patch(Rectangle((29, 54), 20, 17, fc="white", ec=edge_c["pipe"], lw=0.6, zorder=2))
        for k, (band, cycles) in enumerate(zip([r"$\gamma$", r"$\beta$", r"$\alpha$", r"$\theta$"], [7, 4, 2.4, 1.4])):
            y = 57 + 3.8 * k
            label(31.2, y, band)
            ax.plot(33 + 14.5 * t, y + 1.1 * np.sin(2 * np.pi * cycles * t), lw=0.6, color="#2e75b6", zorder=3)
        label(39, 51.4, "Band-pass"); label(39, 48.6, "filter bank")
        arrow((49.6, yc), (54.6, yc))
        for k in range(4):
            a = rng.standard_normal((8, 8)); s = a @ a.T + 3 * np.eye(8); d = np.sqrt(np.diag(s))
            corr = s / np.outer(d, d)
            x0, y0, cell = 55 + 1.8 * k, 58.5 - 1.8 * k, 11 / 8
            for i in range(8):
                for j in range(8):
                    ax.add_patch(Rectangle((x0 + j * cell, y0 + (7 - i) * cell), cell, cell, lw=0,
                                           fc=plt.cm.Blues((corr[i, j] + 0.4) / 1.4), zorder=3 + k))
            ax.add_patch(Rectangle((x0, y0), 11, 11, fc="none", ec=ink, lw=0.4, zorder=3 + k))
        label(64.5, 51.4, "Per-band"); label(64.5, 48.6, r"covariances $\Sigma_b$")
        arrow((71, yc), (76, yc))
        centre = np.array([88.5, yc + 0.5])
        ax.add_patch(Ellipse(tuple(centre), 23, 16, fc="white", ec=edge_c["pipe"], lw=0.6, zorder=2))
        for mu, col in (((82, 66), OI["blue"]), ((95, 59), OI["orange"]), ((84, 58), OI["green"])):
            pts = np.array(mu) + rng.normal(0, 0.9, (7, 2))
            ax.scatter(pts[:, 0], pts[:, 1], s=2.5, color=col, zorder=4, linewidths=0)
            arrow(mu, tuple(centre + (np.array(mu) - centre) * 0.18), lw=0.6, mutation_scale=5, color=col)
        ax.plot(*centre, "o", ms=2.2, color=ink, zorder=5)
        label(centre[0] + 0.2, centre[1] + 2.2, r"$I$")
        label(88.5, 51.4, "Per-subject"); label(88.5, 48.6, "recentring")
        arrow((100.4, yc), (105.4, yc))
        vec(107, 54, 3.2, 17, 10, blues)
        label(112.5, yc + 3, r"$v \in \mathbb{R}^{1860}$", ha="left")
        label(112.5, yc - 1.5, "tangent vector", size=7, color=grey, ha="left")
        label(112.5, yc - 4.6, "(4 bands, 30 ch.)", size=7, color=grey, ha="left")

        # DINQ: encoder, discrete content code z^c, continuous residual z^r, decoder.
        panel(1.5, 1.5, 103.5, 43.5, "model", "DINQ")
        panel(105.5, 1.5, 138.5, 43.5, "eval", "Evaluation")
        yb = 23
        ax.plot([108.6, 108.6, 4.5, 4.5], [54, 45, 45, yb], color=ink, lw=0.8, zorder=4)
        arrow((4.5, yb), (7.6, yb))
        trapezoid(8, 23, yb, 22, 10, r"$f_{\mathrm{enc}}$", "encoder")
        arrow((23, yb), (26, yb), arrowstyle="-")
        ax.plot([26, 26], [14, 32], color=ink, lw=0.8, zorder=4)
        arrow((26, 32), (28.6, 32)); arrow((26, 14), (28.6, 14))
        rbox(29, 29.5, 7, 5, "model", r"$g_{\mathrm{c}}$")
        rbox(29, 11.5, 7, 5, "model", r"$g_{\mathrm{r}}$")
        arrow((36, 32), (39.6, 32))
        vec(40, 28.5, 2.6, 7, 5, ["#F4B183", "#F8CBAD", "#FBE5D6"])
        label(41.3, 37.3, r"$u$", size=8)
        label(41.3, 40.2, r"$\mathcal{L}_{\mathrm{con}}$", size=8, color=code_c)
        shade = rng.uniform(0.25, 0.85, (4, 8))
        for i in range(4):
            for j in range(8):
                ax.add_patch(Rectangle((47 + j * 2.35, 26.5 + i * 2.35), 2.0, 2.0, ec="white", lw=0.3, zorder=3,
                                       fc=plt.cm.Oranges(shade[i, j] * 0.6)))
        hx, hy = 47 + 5 * 2.35, 26.5 + 2 * 2.35
        ax.add_patch(Rectangle((hx, hy), 2.0, 2.0, fc=code_c, ec=ink, lw=0.7, zorder=4))
        arrow((42.8, 32), (46.6, 32), lw=0.7)
        label(56.2, 24.2, r"codebook, $K = 512$", size=7)
        label(56.2, 21.2, r"nearest code; $\mathcal{L}_{\mathrm{vq}}$", size=7)
        arrow((hx + 2.0, hy + 1.0), (66.6, 32), lw=0.7, connectionstyle="arc3,rad=-0.2")
        vec(67, 28.5, 2.6, 7, 5, [code_c, "#E9915A", code_c, "#F2B48A", "#E9915A"])
        label(68.3, 37.3, r"$z^{\mathrm{c}}$", size=8)
        arrow((36, 14), (66.6, 14))
        label(51.5, 15.8, "continuous residual", size=7, color=grey)
        vec(67, 11.5, 2.6, 5, 3, ["#D9D9D9", "#BFBFBF", "#E7E6E6"])
        label(68.3, 9.3, r"$z^{\mathrm{r}}$", size=8)
        arrow((69.6, 32), (75.6, yb + 1.5), lw=0.7); arrow((69.6, 14), (75.6, yb - 1.5), lw=0.7)
        trapezoid(76, 91, yb, 10, 22, r"$f_{\mathrm{dec}}$", "decoder")
        arrow((91, yb), (94.6, yb))
        vec(95, 14.5, 3.2, 17, 10, blues)
        label(96.6, 34.2, r"$\hat{v}$", size=8)
        label(83.5, 6, r"$\mathcal{L}_{\mathrm{rec}} = \|v - \hat{v}\|^2$", color=code_c)

        # Probes read z^c only.
        ax.plot([69.6, 72.5, 72.5, 104.5, 104.5], [34.5, 34.5, 39.8, 39.8, 15], color=ink, lw=0.8, zorder=4)
        arrow((104.5, 31.75), (107.6, 31.75)); arrow((104.5, 15), (107.6, 15))
        rbox(108, 27, 28, 9.5, "eval")
        label(122, 33.6, r"Emotion probe on $z^{\mathrm{c}}$")
        label(122, 29.6, "unseen subjects", size=7, color=grey)
        rbox(108, 6, 28, 18, "eval")
        label(122, 20.6, r"Subject probes on $z^{\mathrm{c}}$")
        label(122, 16.4, "linear, MLP, HGB", size=7, color=grey)
        label(122, 12.9, "(max over probes)", size=7, color=grey)
        label(122, 9.1, "vs. permutation null", size=7, color=grey)
        save(fig, "fig_arch", "§3 / graphical-abstract")


def _texesc(x):
    """Escape LaTeX specials so the .tex compiles (underscores, arrows, &, %, #)."""
    s = str(x).replace("→", r"$\rightarrow$")
    # Backslashes are left as they are: the arrow replacement above introduces them.
    for a, b in (("_", r"\_"), ("&", r"\&"), ("%", r"\%"), ("#", r"\#")):
        s = s.replace(a, b)
    return s


def _write_table(name, header, rows, caption, label, anchor):
    TAB.mkdir(parents=True, exist_ok=True)
    # CSV (raw values, no escaping)
    import csv
    with open(TAB / f"{name}.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)
    # LaTeX booktabs (escaped)
    col = "l" + "r" * (len(header) - 1)
    tex = ["\\begin{table}[t]", "\\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
           f"\\begin{{tabular}}{{{col}}}", "\\toprule",
           " & ".join(_texesc(h) for h in header) + " \\\\", "\\midrule"]
    tex += [" & ".join(_texesc(c) for c in r) + " \\\\" for r in rows]
    tex += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (TAB / f"{name}.tex").write_text("\n".join(tex))
    MANIFEST.append((f"{name}.tex/.csv", anchor, "BUILT"))
    print(f"[built] {name}  -> {anchor}")


def table_xcorp():
    files = sorted(glob.glob(str(REPO / "results" / "cross_corpus" / "*.json")))
    rows = []
    for f in files:
        o = json.load(open(f))
        for arm in ("baseline", "dinq"):
            if arm not in o:
                continue
            c = o[arm]; g = lambda k: round(c.get(k, {}).get("point", float("nan")), 2)
            rows.append([f"{o['source']}→{o['target']}", arm, g("task_target_acc"),
                         g("domain_linear_acc"), g("domain_deep_acc"), g("residual_domain_acc"),
                         g("conditional_domain_acc"), c.get("category", "")])
    _write_table("table_xcorp", ["pair", "arm", "emotion", "dom_lin", "dom_deep", "resid", "cond", "cell"],
                 rows, "Cross-dataset secondary: recentered baseline vs DINQ (subject-grouped CV, "
                 "max-over-probes domain, frozen cross-corpus thresholds).", "tab:xcorp", "§5.2")


def table_abl_seedv():
    sweep = load_cells("dinq_sweep")
    if not sweep:
        return flag("table_abl_seedv", "§4.4", "results/dinq_sweep/cells")
    rows = []
    for cfg in [c for c in CODEBOOK_ORDER if c in sweep]:
        cells = sweep[cfg]
        e = agg(cells, "emotion_unseen"); r = agg(cells, "recov_heldout.hgb")
        rows.append([cfg, f"{e[0]:.3f}", f"{r[0]:.3f}",
                     f"{np.mean([c['ceiling_heldout'] for c in cells]):.3f}", len(cells)])
    _write_table("table_abl_seedv", ["config", "emotion", "subj_recov", "perm_ceiling", "n_splits"],
                 rows, "Codebook configurations on SEED-V (held-out subjects): codebook size, FSQ, "
                 "contrastive loss on u.",
                 "tab:abl_seedv", "§4.4 (archival)")


def table_abl_faced():
    sweep = load_cells(CODEBOOK_FACED)
    if not _codebook_complete(sweep):
        return flag("table_abl", "§4.5", f"all 6 configs x 8 splits in results/{CODEBOOK_FACED}/cells")
    rows = []
    for cfg in CODEBOOK_ORDER:
        cells = sweep[cfg]
        e = agg(cells, "emotion_unseen"); rm = ci95([maxp(c) for c in cells])
        rows.append([cfg, f"{e[0]:.3f} [{e[1]:.3f}, {e[2]:.3f}]", f"{agg(cells, 'recov_heldout.linear')[0]:.3f}",
                     f"{rm[0]:.3f} [{rm[1]:.3f}, {rm[2]:.3f}]", f"{agg(cells, 'recov_heldout.hgb')[0]:.3f}",
                     f"{np.mean([c['ceiling_heldout'] for c in cells]):.3f}", len(cells)])
    pt = _permtest(); ceil = pt[0]["null_95pct"] if pt else SHARED_NULL_K31
    _write_table("table_abl", ["config", "emotion", "subj_recov_linear", "subj_recov_max",
                               "subj_recov_hgb", "perm_ceiling_hgb", "n_splits"],
                 rows, "Codebook configurations on FACED (31 held-out subjects per partition, chance "
                 f"{CHANCE_K31:.3f}, eight partitions; 60 epochs, cap 8000, seeded per cell). Means with 95% CIs. "
                 f"Max-over-probes recoverability compares with the shared null ceiling {ceil:.3f} (Table 3); "
                 "perm_ceiling_hgb is the per-cell HGB permutation null (15 permutations).",
                 "tab:abl", "§4.5")


def table_thresh():
    rec = yaml.safe_load((REPO / "configs" / "thresholds_recentered.yaml").read_text())["thresholds_recentered"]
    faced = yaml.safe_load((REPO / "configs" / "thresholds_faced.yaml").read_text())["faced_recentered"]
    cc = rec["cross_corpus"]
    rows = [
        ["cross-corpus", "task_floor", cc["task_floor"], "recalibrated (perm-null / transfer dist)"],
        ["cross-corpus", "task_good", cc["task_good"], "0.55 (was 0.60; cross-corpus ceiling)"],
        ["cross-corpus", "dom_low", cc["dom_low"], "chance+0.10"],
        ["cross-corpus", "dom_high", cc["dom_high"], "60th-pct reference deep"],
        ["cross-corpus", "residual_low", cc["residual_low"], "chance+0.10"],
    ]
    pt = _permtest()
    if pt is not None:
        v = pt[0]
        rows += [
            ["faced (subject)", "null_95pct ceiling", round(v["null_95pct"], 3), "stabilized max-over-probes perm-null (n=1000, K=31)"],
            ["faced (subject)", "dinq_recov (max)", round(v["observed_mean"], 3),
             f"[{v['observed_ci95'][0]:.3f}, {v['observed_ci95'][1]:.3f}]; {v['label']} (perm p={v['perm_p_value']:.3f})"],
            ["faced (subject)", "chance 1/K", round(v["chance_1_over_K"], 3), "K=31 held-out subjects"],
            ["faced (subject)", "task_floor_permnull", faced.get("task_floor_permnull", "-"),
             f"n_perm_pooled={v.get('n_perm_pooled', '-')}"],
        ]
    _write_table("table_thresh", ["setting", "threshold", "value", "rationale / perm-null"], rows,
                 "Frozen thresholds and permutation-null ceilings per setting.",
                 "tab:thresh", "appendix")


def _subj_baseline_cells():
    return load_cells("baselines_faced_subject")


def fig_mech():
    b = _subj_baseline_cells()
    faced = load_cells("dinq_sweep_faced")  # only the 60-epoch `continuous` cells are used
    dinq = _dinq_matched()
    if "recentering" not in b or dinq is None:
        return flag("Fig-MECH", "§4.3", "results/baselines_faced_subject/cells + " + str(MATCHED.relative_to(REPO)))
    base = b["recentering"]
    lin_b, lin_d = agg(base, "recov_heldout.linear"), agg(dinq, "recov_heldout.linear")
    deep_b = ci95([maxp(c) for c in base])
    deep_d = ci95([maxp(c) for c in dinq])
    fig, (a, ins) = plt.subplots(1, 2, figsize=(W_15COL, 62 * MM), gridspec_kw={"width_ratios": [2, 1]})
    x = np.arange(2); w = 0.38
    a.bar(x - w / 2, [lin_b[0], deep_b[0]], w, color=BASE_C, label="Recentring baseline")
    a.bar(x + w / 2, [lin_d[0], deep_d[0]], w, color=DINQ_C, label="DINQ")
    for xi, (bb, dd) in enumerate([(lin_b, lin_d), (deep_b, deep_d)]):
        a.errorbar(xi - w / 2, bb[0], yerr=[[bb[0] - bb[1]], [bb[2] - bb[0]]], fmt="none", ecolor="k", capsize=2)
        a.errorbar(xi + w / 2, dd[0], yerr=[[dd[0] - dd[1]], [dd[2] - dd[0]]], fmt="none", ecolor="k", capsize=2)
    K = int(dinq[0]["K"])
    a.axhline(1 / K, ls=":", lw=1.0, color=CHANCE_C, label=f"Chance (1/{K})")
    a.set_xticks(x); a.set_xticklabels(["Linear probe", "Max over probes"])
    a.set_ylabel("Held-out subject recoverability")
    h, l = a.get_legend_handles_labels()  # bars first, then the chance line
    a.legend(h[1:] + h[:1], l[1:] + l[:1], loc="upper left")
    # Right panel: matched continuous bottleneck (same encoder and contrastive loss,
    # no quantizer) against DINQ.
    pt = _permtest(); null = pt[0]["null_95pct"] if pt else 0.041
    if "continuous" in faced:
        deep_cont = ci95([maxp(c) for c in faced["continuous"]])
        vals = [deep_cont[0], deep_d[0]]
        ins.bar([0, 1], vals, 0.6, color=[OI["green"], DINQ_C])
        for xi, vv in [(0, deep_cont), (1, deep_d)]:
            ins.errorbar(xi, vv[0], yerr=[[vv[0] - vv[1]], [vv[2] - vv[0]]], fmt="none", ecolor="k", capsize=2)
            ins.text(xi, vv[2] + 0.004, f"{vv[0]:.3f}", ha="center", fontsize=7.5)
        ins.axhline(null, ls="--", lw=1.2, color=CEIL_C, label=f"Null ceiling ({null:.3f})")
        ins.set_xticks([0, 1]); ins.set_xticklabels(["Continuous\nbottleneck", "Quantized\n(DINQ)"])
        ins.set_ylim(0, max(vals) * 1.5); ins.legend(loc="upper right")
        ins.set_ylabel("Held-out subject recoverability\n(max over probes)")
    else:
        ins.axis("off"); ins.text(0.5, 0.5, "continuous\nablation\npending", ha="center", va="center", fontsize=8)
    save(fig, "fig_mech", "§4.3")


def _permtest():
    """FACED permutation test: verdict dict and the pooled null array."""
    pt = REPO / "results" / "faced_permtest"
    vf = pt / "verdict.json"
    if not vf.exists():
        return None
    v = json.loads(vf.read_text())
    null = np.concatenate([np.asarray(json.loads(p.read_text())["null_max"])
                           for p in sorted(pt.glob("split*.json"))])
    return v, null


def fig_perm():
    pt = _permtest()
    if pt is None:
        return flag("Fig-PERM", "§4.3", "results/faced_permtest/{verdict.json, split*.json}")
    v, null = pt
    obs, lo, hi = v["observed_mean"], v["observed_ci95"][0], v["observed_ci95"][1]
    ceil, chance = v["null_95pct"], v["chance_1_over_K"]
    fig, ax = plt.subplots(figsize=(W_1COL, 70 * MM))
    ax.hist(null, bins=30, color=OI["sky"], alpha=0.85)
    hc = ax.axvline(chance, ls=":", lw=1.0, color=CHANCE_C)
    hn = ax.axvline(ceil, ls="--", lw=1.5, color=CEIL_C)
    hs = ax.axvspan(lo, hi, color=DINQ_C, alpha=0.18)
    ho = ax.axvline(obs, ls="-", lw=2.2, color=DINQ_C)
    for p in v["observed_per_split"]:
        ax.plot(p, 0.5, marker="|", ms=9, color=DINQ_C, mew=1.4)
    ax.set_xlabel("Held-out subject recoverability (max over probes)")
    ax.set_ylabel("Permutation count")
    # The observed line and CI band span the full height, so the legend sits above the axes.
    fig.legend([Patch(color=OI["sky"], alpha=0.85), hc, hn, (hs, ho)],
               [f"Permutation null (n = {null.size})", f"Chance (1/{round(1 / chance)})",
                f"Null 95th percentile ({ceil:.3f})", f"DINQ mean ({obs:.3f}), 95% CI"],
               loc="outside upper center", ncols=2, columnspacing=1.0, handlelength=1.8)
    save(fig, "fig_perm", "§4.3/appendix")


# Shared permutation-null ceiling for 31 held-out subjects at matched 64 dimensions
# (cap 8000). Shuffling subject labels at trial granularity removes any subject
# structure, so the null reflects the probe rather than the representation: the
# recentred tangent gives 0.0434 and the DINQ z_c 0.041 (faced_permtest, n=1000).
SHARED_NULL_K31 = 0.043
CHANCE_K31 = 1.0 / 31  # 0.032


def _above_null(deep_lo, null):
    """'yes' if the lower 95% CI bound of the recoverability exceeds the null."""
    return "yes" if deep_lo > null else "near"


def table_main():
    b = _subj_baseline_cells(); faced = load_cells("dinq_sweep_faced"); dinq = _dinq_matched()
    if not b or dinq is None or "continuous" not in faced:
        return flag("Table-MAIN", "§4.3", "results/baselines_faced_subject + dinq_sweep_faced/continuous + "
                    + str(MATCHED.relative_to(REPO)))
    rows = []

    def _row(label, cs, null):
        mean, lo, hi = ci95([maxp(c) for c in cs])
        margin = mean - null  # recoverable subject signal above the ceiling
        return [label, f"{agg(cs,'emotion_unseen')[0]:.3f}", f"{agg(cs,'recov_heldout.linear')[0]:.3f}",
                f"{mean:.3f}", f"{null:.3f}", f"{margin:+.3f}", _above_null(lo, null), len(cs)]

    order = ["recentering", "random", "task_only", "grl_subject"]
    for name in [n for n in order if n in b] + [c for c in b if c not in order]:
        rows.append(_row(name, b[name], SHARED_NULL_K31))
    # DINQ rows use the faced_permtest null (0.041), the other rows the shared 0.043.
    # Every row is 60 epochs / cap 8000; the archived 80-epoch cells are excluded.
    pt = _permtest()
    dinq_null = pt[0]["null_95pct"] if pt and "null_95pct" in pt[0] else SHARED_NULL_K31
    rows.append(_row("DINQ (continuous)", faced["continuous"], dinq_null))
    rows.append(_row("DINQ (vq512_conU)", dinq, dinq_null))

    _write_table("table_main",
                 ["method", "emotion(unseen)", "subj_recov_linear", "subj_recov_max",
                  "null(K=31)", "margin(max-null)", "above_null", "n_splits"],
                 rows,
                 "Headline (FACED cross-subject): emotion transfer + multi-class subject-recoverability "
                 f"(K=31, chance {CHANCE_K31:.3f}), max over linear/MLP/HGB probes; every row 60 epochs, cap 8000. "
                 "'null(K=31)' is the permutation-null "
                 "ceiling at matched 64-dim (shared across methods, verified: recentering 0.043 = DINQ 0.041 "
                 "from faced_permtest n=1000); 'margin' is recoverable subject signal above that ceiling; "
                 "'above_null' = 'yes' if the max-recov 95%CI lower bound clears the null, else 'near' "
                 "(near-C1).",
                 "tab:main", "§4.3")


def main():
    style()
    fig_headline(); fig_gen(); fig_xcorp(); fig_arch()
    fig_codebook("dinq_sweep", "fig_codebook_seedv", "§4.4 (archival)", CODEBOOK_LAB_SEEDV, xlim=(0.195, 0.43))
    if _codebook_complete(load_cells(CODEBOOK_FACED)):
        fig_codebook(CODEBOOK_FACED, "fig_codebook", "§4.5", CODEBOOK_LAB_FACED, xlim=CODEBOOK_XLIM_FACED,
                     use_max=True)
    else:
        flag("fig_codebook", "§4.5", f"all 6 configs x 8 splits in results/{CODEBOOK_FACED}/cells")
    fig_mech(); fig_perm()
    table_xcorp(); table_abl_seedv(); table_abl_faced(); table_thresh(); table_main()

    (REPO / "paper").mkdir(exist_ok=True)
    lines = ["# Paper artifacts — manifest\n",
             "| artifact | outline anchor | status |", "|---|---|---|"]
    lines += [f"| {a} | {anc} | {s} |" for a, anc, s in MANIFEST]
    (REPO / "paper" / "MANIFEST.md").write_text("\n".join(lines) + "\n")
    built = sum(1 for _, _, s in MANIFEST if s == "BUILT")
    blocked = sum(1 for _, _, s in MANIFEST if s.startswith("BLOCKED"))
    print(f"\n[manifest] paper/MANIFEST.md — {built} built, {blocked} blocked (see flags)")


if __name__ == "__main__":
    main()
