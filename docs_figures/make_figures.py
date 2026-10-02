#!/usr/bin/env python3
"""Figures and table numbers for the "Performance" pages of the Baysor docs.

One command regenerates every figure of ``docs/performance/`` in a Baysor
checkout and the numbers of its tables::

    .deps/bench/bin/python docs_figures/make_figures.py \\
        --docs /path/to/Baysor/docs/performance

Inputs (all read-only, under ``$BAYSOR_BENCH_DATA``, default
``/home/vpetukhov/Projects/Baysor/.bench-data``):

* ``rc/2026-10-01-1636267/tables/{runtime,quality_sim,quality_real}.csv`` and
  ``runs/{rc1636,v0083}-{real,sim,t1,t1b}/`` - the release-candidate A/B
  benchmark (cpp-0.9.0 candidate vs cpp-0.8.3, 77 datasets);
* ``docs-figures/remeasure/runs.tsv`` - the one-process-at-a-time re-measurement written by
  ``remeasure.py`` (thread sweep and the real full-tier datasets);
* ``profiling/reports/2026-10-01-20bc45c/`` and
  ``profiling/reports/2026-09-30-e45fddc/`` - the profiling summaries (after /
  before the optimisation);
* ``real/<id>/meta.json`` and ``sim/<id>/meta.json`` - dataset facts.

Outputs:

* ``<docs>/img/*.svg|png`` - charts in a light and a dark variant
  (``*-light.svg`` / ``*-dark.svg``, selected by MkDocs Material's
  ``#only-light`` / ``#only-dark``); the segmentation mosaic is one PNG on a
  dark background that reads in both themes;
* ``docs_figures/generated/tables.md`` - every number quoted in the pages,
  as Markdown tables, each with the file it comes from;
* ``docs_figures/generated/numbers.json`` - the same numbers, machine-readable.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = Path(os.environ.get("BAYSOR_BENCH_DATA", "/home/vpetukhov/Projects/Baysor/.bench-data"))
RC = DATA / "rc/2026-10-01-1636267"
RUNS = DATA / "runs"
PROF = DATA / "profiling/reports/2026-10-01-20bc45c"
PROF_BEFORE = DATA / "profiling/reports/2026-09-30-e45fddc"
REMEASURE = DATA / "docs-figures/remeasure/runs.tsv"
GEN = HERE / "generated"

NEW, OLD = "cpp-0.9.0", "cpp-0.8.3"

# ---------------------------------------------------------------- theme ----
# Reference palette of the dataviz guidelines (validated: first three slots
# all-pairs in both modes; see README.md in this directory).
THEMES = {
    "light": dict(ink="#0b0b0b", ink2="#52514e", muted="#898781", grid="#e1e0d9",
                  axis="#c3c2b7", s=["#2a78d6", "#eb6834", "#1baf7a", "#eda100"],
                  base="#898781", surface="#ffffff"),
    "dark": dict(ink="#ffffff", ink2="#c3c2b7", muted="#898781", grid="#2c2c2a",
                 axis="#4a4a46", s=["#3987e5", "#d95926", "#199e70", "#c98500"],
                 base="#9a9890", surface="#1e2129"),
}
FONT = "system-ui, -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"

SOURCES: dict[str, set] = defaultdict(set)   # figure/table -> source files
NUMBERS: dict = {}


def src(key: str, *paths):
    for p in paths:
        SOURCES[key].add(str(Path(p)).replace(str(DATA), "$BAYSOR_BENCH_DATA"))


def apply_theme(mode: str) -> dict:
    t = THEMES[mode]
    plt.rcdefaults()
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9.5,
        "axes.edgecolor": t["axis"], "axes.labelcolor": t["ink2"],
        "axes.titlecolor": t["ink"], "axes.titlesize": 10.5, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "axes.titlepad": 8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": t["grid"], "grid.linewidth": 0.6,
        "axes.axisbelow": True, "xtick.color": t["muted"], "ytick.color": t["muted"],
        "xtick.labelcolor": t["ink2"], "ytick.labelcolor": t["ink2"],
        "legend.frameon": False, "legend.labelcolor": t["ink2"], "legend.fontsize": 8.5,
        "text.color": t["ink"], "lines.linewidth": 2, "lines.markersize": 6,
        "svg.fonttype": "none", "figure.facecolor": "none", "axes.facecolor": "none",
        "savefig.facecolor": "none", "savefig.transparent": True,
    })
    return t


def save(fig, out: Path, name: str, mode: str, fmt: str = "svg", dpi: int = 110):
    path = out / f"{name}-{mode}.{fmt}"
    if fmt == "svg":
        buf = io.StringIO()
        fig.savefig(buf, format="svg", bbox_inches="tight", metadata={"Date": None})
        text = buf.getvalue()
        text = re.sub(r"font-family:\s*'?DejaVu Sans'?", f"font-family: {FONT}", text)
        text = text.replace('font-family="DejaVu Sans"', f'font-family="{FONT}"')
        # drop the matplotlib comment header with the creation date
        text = re.sub(r"<metadata>.*?</metadata>\s*", "", text, flags=re.S)
        path.write_text(text)
    else:
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def both_modes(fn):
    """Run a figure function for the light and the dark theme."""
    def run(out: Path):
        for mode in ("light", "dark"):
            fn(out, mode, apply_theme(mode))
    run.__name__ = fn.__name__
    return run


def log_axis(ax, axis="both"):
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

    def fmt(v, _):
        if v >= 1e6:
            return f"{v / 1e6:g}M"
        if v >= 1e3:
            return f"{v / 1e3:g}k"
        return f"{v:g}"
    for a in ("x", "y"):
        if axis in (a, "both"):
            getattr(ax, f"set_{a}scale")("log")
            ax_ = getattr(ax, f"{a}axis")
            ax_.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
            ax_.set_major_formatter(FuncFormatter(fmt))
            ax_.set_minor_formatter(NullFormatter())


# ----------------------------------------------------------------- data ----

def read_meta(ds: str) -> dict:
    for kind in ("real", "sim"):
        p = DATA / kind / ds / "meta.json"
        if p.is_file():
            return json.loads(p.read_text())
    raise FileNotFoundError(ds)


def platform_of(ds: str) -> str:
    m = read_meta(ds)
    return m.get("platform") or ("simulated" if m.get("kind") == "sim" else "?")


def metrics(run: str, ds: str) -> dict:
    return json.loads((RUNS / run / ds / "metrics.json").read_text())


def runtime_table() -> pd.DataFrame:
    p = RC / "tables/runtime.csv"
    df = pd.read_csv(p)
    rows = []
    for _, r in df.iterrows():
        m = metrics(r.rc_run, r.dataset)
        rows.append({"n_molecules": m["dataset"]["n_molecules"], "n_genes": m["dataset"]["n_genes"],
                     "platform": platform_of(r.dataset)})
    df = pd.concat([df, pd.DataFrame(rows)], axis=1)
    # the 1-thread quick tier ran twice (rc1636-t1, rc1636-t1b); keep the
    # campaign with the lower mean load for each dataset
    df["load_pair"] = (df.load_rc + df.load_old) / 2
    df = (df.sort_values("load_pair").drop_duplicates(["dataset", "threads"])
            .sort_values(["threads", "kind", "dataset"], ascending=[False, True, True]))
    return df


def remeasured() -> pd.DataFrame:
    if not REMEASURE.is_file():
        return pd.DataFrame(columns=["exp", "dataset", "version", "threads", "wall_s"])
    df = pd.read_csv(REMEASURE, sep="\t")
    df = df[df.exit == 0].copy()
    df["cpu_s"] = df.user_s + df.sys_s
    df["rss_mib"] = df.rss_kb / 1024
    return df


PLATFORM_ORDER = ["Xenium", "CosMx SMI", "MERFISH", "other"]
PLATFORM_LABEL = {"Xenium": "Xenium", "CosMx SMI": "CosMx", "MERFISH": "MERFISH",
                  "other": "ISS, osmFISH, STARmap"}
PLATFORM_MARKER = {"Xenium": "o", "CosMx SMI": "s", "MERFISH": "^", "other": "D"}


def platform_group(p: str) -> str:
    return p if p in ("Xenium", "CosMx SMI", "MERFISH") else "other"


def platform_color(t: dict, g: str) -> str:
    return {"Xenium": t["s"][0], "CosMx SMI": t["s"][1], "MERFISH": t["s"][2],
            "other": t["ink2"]}[g]


SHORT = {
    "xenium_prime5k_ovarian": "Xenium Prime 5K", "cosmx_wtx_colon": "CosMx WTx",
    "xenium_pancreas_377": "Xenium pancreas", "cosmx_nsclc_lung5_rep1": "CosMx lung",
    "xenium_breast_rep1_dense": "Xenium breast", "merfish_ileum": "MERFISH ileum",
    "xenium_lung_cancer": "Xenium lung", "xenium_mouse_brain_ff": "Xenium brain",
    "xenium_breast_rep1_stroma": "Xenium breast stroma",
    "iss_mouse_hippocampus": "ISS hippocampus", "xenium_breast_rep2_dense": "Xenium breast rep2",
}


def short(ds: str) -> str:
    base = re.sub(r"_(quick|full|admix)$", "", ds)
    return SHORT.get(base, base)


# -------------------------------------------------------------- figures ----

@both_modes
def fig_runtime_vs_molecules(out, mode, t):
    df = runtime_table()
    d = df[(df.kind == "real") & (df.threads == 6)].copy()
    d["group"] = d.platform.map(platform_group)
    src("runtime_vs_molecules", RC / "tables/runtime.csv", RUNS / "rc1636-real",
        RUNS / "v0083-real", DATA / "real")
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.6))
    for ax, (new, old, ylab, title) in zip(axes, [
            ("wall_rc", "wall_old", "wall time, s", "Wall time (6 threads)"),
            ("rss_rc_mib", "rss_old_mib", "peak memory (RSS), MiB", "Peak memory (6 threads)")]):
        for g in PLATFORM_ORDER:
            s = d[d.group == g]
            if s.empty:
                continue
            c = platform_color(t, g)
            segs = [[(x, y0), (x, y1)] for x, y0, y1 in zip(s.n_molecules, s[old], s[new])]
            ax.add_collection(LineCollection(segs, colors=c, linewidths=1, alpha=0.45))
            ax.scatter(s.n_molecules, s[old], marker=PLATFORM_MARKER[g], s=30,
                       facecolors="none", edgecolors=c, linewidths=1.1, alpha=0.8, zorder=3)
            ax.scatter(s.n_molecules, s[new], marker=PLATFORM_MARKER[g], s=40, color=c,
                       edgecolors=t["surface"], linewidths=0.8, zorder=4,
                       label=PLATFORM_LABEL[g])
        # label the gene-rich panels, they sit above the trend
        for _, r in d[d.n_genes >= 4000].iterrows():
            ax.annotate(f"{short(r.dataset)}\n{r.n_genes:,} genes", (r.n_molecules, r[new]),
                        xytext=(6, -2), textcoords="offset points", fontsize=7.5,
                        color=t["ink2"], va="top")
        log_axis(ax)
        ax.set_xlabel("molecules")
        ax.set_ylabel(ylab)
        ax.set_title(title)
    handles = [Line2D([], [], marker=PLATFORM_MARKER[g], ls="", color=platform_color(t, g),
                      label=PLATFORM_LABEL[g]) for g in PLATFORM_ORDER]
    handles += [Line2D([], [], marker="o", ls="", color=t["ink2"], label=NEW + " (filled)"),
                Line2D([], [], marker="o", ls="", markerfacecolor="none",
                       markeredgecolor=t["ink2"], label=OLD + " (open)")]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.13),
               handletextpad=0.3, columnspacing=1.2)
    fig.tight_layout()
    save(fig, out, "runtime_vs_molecules", mode)


@both_modes
def fig_threads(out, mode, t):
    df = remeasured()
    sw = df[df.exp == "sweep"]
    src("threads", REMEASURE)
    if sw.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.3))
    dsets = list(dict.fromkeys(sw.dataset))
    for i, ds in enumerate(dsets):
        c = t["s"][i]
        for v, ls, mf in (("rc", "-", c), ("old", "--", "none")):
            s = sw[(sw.dataset == ds) & (sw.version == v)].sort_values("threads")
            if s.empty:
                continue
            lab = f"{short(ds)} ({s.n_molecules.iloc[0] if 'n_molecules' in s else ''})"
            lab = f"{short(ds)}, {NEW if v == 'rc' else OLD}"
            for ax, col in zip(axes, ("wall_s", "cpu_s")):
                ax.plot(s.threads, s[col], ls=ls, color=c, marker="o", markersize=5,
                        markerfacecolor=mf, markeredgecolor=c, label=lab)
        # ideal scaling from the 1-thread wall time of cpp-0.9.0
        s = sw[(sw.dataset == ds) & (sw.version == "rc")].sort_values("threads")
        if not s.empty:
            t1 = s.wall_s.iloc[0]
            th = np.array([1, 16])
            axes[0].plot(th, t1 / th, ls=":", lw=1, color=t["muted"])
    t1 = sw[(sw.dataset == dsets[0]) & (sw.version == "rc") & (sw.threads == 1)].wall_s.iloc[0]
    axes[0].text(9.5, t1 / 9.5 * 0.62, "ideal", fontsize=7.5, color=t["muted"], ha="center")
    for ax, ylab, title in zip(axes, ("wall time, s", "CPU time (user + sys), s"),
                               ("Wall time vs threads", "CPU time vs threads")):
        log_axis(ax)
        ax.set_xticks([1, 2, 4, 8, 16])
        ax.set_xticklabels(["1", "2", "4", "8", "16"])
        ax.set_xlabel("threads (8 physical cores)")
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.axvline(8, color=t["axis"], lw=0.8, zorder=0)
    h, l = axes[1].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.17))
    fig.tight_layout()
    save(fig, out, "threads", mode)


GENE_FAMILIES = [
    ("simulated circles, 92k molecules",
     ["sim_circles_gaps_g100", "sim_circles_gaps_g1000", "sim_circles_gaps_g5000"]),
    ("simulated tiles, 116k molecules",
     ["sim_tiled_distinct_g100", "sim_tiled_distinct_g1000", "sim_tiled_distinct_g5000"]),
    ("simulated tissue (strec), 64k molecules",
     ["strec_dense_s2_disjoint", "strec_dense_s2_merfish", "strec_dense_s2_xenium",
      "strec_dense_s2_prime5k1000", "strec_dense_s2_prime5k5000"]),
]


@both_modes
def fig_genes(out, mode, t):
    df = runtime_table()
    d = df[df.threads == 6].set_index("dataset")
    src("time_vs_genes", RC / "tables/runtime.csv", RUNS / "rc1636-sim", DATA / "sim")
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.5))
    for i, (lab, dss) in enumerate(GENE_FAMILIES):
        s = d.loc[dss].sort_values("n_genes")
        c = t["s"][i]
        for ax, (new, old) in zip(axes, (("wall_rc", "wall_old"), ("rss_rc_mib", "rss_old_mib"))):
            ax.plot(s.n_genes, s[new], color=c, marker="o", markersize=5, label=f"{lab}, {NEW}")
            ax.plot(s.n_genes, s[old], color=c, ls="--", marker="o", markersize=5,
                    markerfacecolor="none", label=f"{lab}, {OLD}")
    for ax, ylab, title in zip(axes, ("wall time, s", "peak memory (RSS), MiB"),
                               ("Wall time vs gene panel (6 threads)",
                                "Peak memory vs gene panel (6 threads)")):
        log_axis(ax)
        ax.set_xlabel("genes in the panel")
        ax.set_ylabel(ylab)
        ax.set_title(title)
    h = [Line2D([], [], color=t["s"][i], lw=2, label=lab) for i, (lab, _) in enumerate(GENE_FAMILIES)]
    h += [Line2D([], [], color=t["ink2"], marker="o", label=NEW + " (solid)"),
          Line2D([], [], color=t["ink2"], ls="--", marker="o", markerfacecolor="none",
                 label=OLD + " (dashed)")]
    fig.legend(handles=h, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.2))
    fig.tight_layout()
    save(fig, out, "time_vs_genes", mode)


@both_modes
def fig_accuracy(out, mode, t):
    p = RC / "tables/quality_sim.csv"
    q = pd.read_csv(p)
    q = q[q.threads == 6]
    src("accuracy_sim", p, RUNS / "rc1636-sim", RUNS / "v0083-sim")
    fig, ax = plt.subplots(figsize=(4.6, 4.3))
    gen = q.dataset.str.startswith("strec").map({True: "strec", False: "trivial"})
    for i, (g, lab) in enumerate((("trivial", "geometric scenes (trivial.py)"),
                                  ("strec", "tissue-like (strec.py)"))):
        s = q[gen == g]
        ax.scatter(s.accuracy_1to1_old, s.accuracy_1to1_rc, s=34, color=t["s"][i],
                   marker="os"[i], edgecolors=t["surface"], linewidths=0.6, label=lab, zorder=3)
    lo = min(q.accuracy_1to1_old.min(), q.accuracy_1to1_rc.min()) - 0.03
    ax.plot([lo, 1], [lo, 1], color=t["muted"], lw=1, ls=":", zorder=1)
    ax.set_xlim(lo, 1.0)
    ax.set_ylim(lo, 1.0)
    ax.set_aspect("equal")
    ax.set_xlabel(f"{OLD}: 1-to-1 accuracy (fraction)")
    ax.set_ylabel(f"{NEW}: 1-to-1 accuracy (fraction)")
    ax.set_title("Accuracy vs ground truth (simulated)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    save(fig, out, "accuracy_sim", mode)


@both_modes
def fig_determinism(out, mode, t):
    ds = "iss_mouse_hippocampus_quick"
    rows = []
    for run, lab, v in (("v0083-real", f"{OLD}\n6 threads", "old"), ("v0083-t1", f"{OLD}\n1 thread", "old"),
                        ("rc1636-real", f"{NEW}\n6 threads", "rc"), ("rc1636-t1", f"{NEW}\n1 thread", "rc")):
        m = metrics(run, ds)
        src("determinism", RUNS / run / ds / "metrics.json")
        for r in m["reps"]:
            rows.append((lab, v, r["n_cells"], r["rep"]))
    fig, ax = plt.subplots(figsize=(6.4, 2.5))
    labels = list(dict.fromkeys(r[0] for r in rows))
    for i, lab in enumerate(labels[::-1]):
        pts = [r for r in rows if r[0] == lab]
        c = t["s"][0] if pts[0][1] == "rc" else t["base"]
        xs = [p[2] for p in pts]
        jit = np.linspace(-0.12, 0.12, len(xs)) if len(set(xs)) > 1 else np.zeros(len(xs))
        ax.scatter(xs, i + jit, s=46, color=c, edgecolors=t["surface"], linewidths=0.8, zorder=3)
        txt = f"{len(xs)} runs: " + ", ".join(f"{x:,}" for x in xs) if len(xs) > 1 else f"{xs[0]:,}"
        if len(xs) > 1 and len(set(xs)) == 1:
            txt = f"{len(xs)} runs, all {xs[0]:,} (identical output)"
        ax.text(max(xs) + 40, i, txt, va="center", fontsize=8, color=t["ink2"])
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels[::-1], fontsize=8.5)
    ax.set_ylim(-0.6, len(labels) - 0.4)
    ax.set_xlim(9900, 11250)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("cells found (ISS mouse hippocampus, 82k molecules)")
    ax.set_title("Same input, repeated runs")
    fig.tight_layout()
    save(fig, out, "determinism_iss", mode)


ADMIX_LABEL = {
    "xenium_prime5k_ovarian_admix": "Xenium Prime 5K ovarian (550k)",
    "xenium_prime5k_ovarian_full": "Xenium Prime 5K ovarian (2M)",
    "xenium_prime5k_ovarian_quick": "Xenium Prime 5K ovarian (130k)",
    "xenium_pancreas_377_full": "Xenium pancreas (2M)",
    "xenium_pancreas_377_quick": "Xenium pancreas (130k)",
    "xenium_breast_rep1_dense_full": "Xenium breast (2M)",
    "xenium_breast_rep1_dense_admix": "Xenium breast (550k)",
    "xenium_breast_rep1_stroma_admix": "Xenium breast stroma (550k)",
    "xenium_breast_rep1_stroma_quick": "Xenium breast stroma (130k)",
    "xenium_lung_cancer_admix": "Xenium lung (550k)",
    "xenium_lung_cancer_quick": "Xenium lung (130k)",
    "xenium_mouse_brain_ff_admix": "Xenium mouse brain (550k)",
    "xenium_mouse_brain_ff_edge_quick": "Xenium mouse brain edge (130k)",
    "xenium_mouse_brain_ff_quick": "Xenium mouse brain (130k)",
    "xenium_breast_rep2_dense_quick": "Xenium breast rep2 (130k)",
    "xenium_breast_rep1_dense_quick": "Xenium breast (130k)",
    "cosmx_nsclc_lung5_rep1_full": "CosMx lung (2M)",
    "merfish_ileum_full": "MERFISH ileum (820k)",
    "merfish_ileum_quick": "MERFISH ileum (150k)",
    "iss_mouse_hippocampus_quick": "ISS hippocampus (82k)",
}


def admix_rows() -> pd.DataFrame:
    p = RC / "tables/quality_real.csv"
    q = pd.read_csv(p)
    q = q[(q.threads == 6) & (q.admix_status == "ok/ok")].drop_duplicates("dataset")
    # 2,000+ Baysor cells: the audit's power threshold (the tiny-crop rates are noise)
    q = q[q.cells_rc >= 2000]
    return q.sort_values("admix_rc")


@both_modes
def fig_celladmix(out, mode, t):
    q = admix_rows()
    src("celladmix", RC / "tables/quality_real.csv")
    fig, axes = plt.subplots(2, 1, figsize=(7.0, 7.4), gridspec_kw={"height_ratios": [len(q), 8]})
    ax = axes[0]
    y = np.arange(len(q))
    for i, (_, r) in enumerate(q.iterrows()):
        ax.plot([r.admix_old * 100, r.admix_rc * 100], [i, i], color=t["axis"], lw=1.5, zorder=1)
    ax.scatter(q.admix_old * 100, y, s=40, facecolors="none", edgecolors=t["base"], linewidths=1.3,
               label=OLD, zorder=2)
    ax.scatter(q.admix_rc * 100, y, s=40, color=t["s"][0], label=NEW, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([ADMIX_LABEL.get(d, d) for d in q.dataset], fontsize=8)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("admixed molecules, % of assigned (lower is cleaner)")
    ax.set_title("cellAdmix: total admixture rate")
    ax.legend(loc="lower right")

    # per-pair rates for one dataset (cell-type pairs are fixed across versions)
    ax = axes[1]
    ds = "xenium_lung_cancer_admix"
    pairs = defaultdict(lambda: {"rc": [], "old": []})
    for run, v in (("rc1636-real", "rc"), ("v0083-real", "old")):
        src("celladmix", RUNS / run / ds / "metrics.json")
        for r in metrics(run, ds)["reps"]:
            for pr in r["celladmix"]["pairs"]:
                pairs[(pr["source"], pr["target"])][v].append(pr["rate"])
    rows = [(k, np.mean(v["rc"]) if v["rc"] else np.nan, np.mean(v["old"]) if v["old"] else np.nan)
            for k, v in pairs.items()]
    rows = sorted(rows, key=lambda r: np.nan_to_num(r[1]))[-8:]
    y = np.arange(len(rows))
    for i, (_, a, b) in enumerate(rows):
        ax.plot([b * 100, a * 100], [i, i], color=t["axis"], lw=1.5, zorder=1)
    ax.scatter([r[2] * 100 for r in rows], y, s=40, facecolors="none", edgecolors=t["base"],
               linewidths=1.3, zorder=2)
    ax.scatter([r[1] * 100 for r in rows], y, s=40, color=t["s"][0], zorder=3)
    lab = lambda c: c.replace("cluster_", "")  # noqa: E731
    ax.set_yticks(y)
    ax.set_yticklabels([f"type {lab(s)} → type {lab(tg)}" for (s, tg), _, _ in rows], fontsize=8)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("molecules of the source type in target cells, %")
    ax.set_title("Xenium lung (550k): top source → target pairs")
    fig.tight_layout()
    save(fig, out, "celladmix", mode)
    NUMBERS["celladmix"] = {d: {"rc": float(a), "old": float(b)}
                            for d, a, b in zip(q.dataset, q.admix_rc, q.admix_old)}


# ---- segmentation mosaic and UMAPs (images) ----

SEG_EXAMPLES = [
    ("xenium_pancreas_377_quick", "Xenium · human pancreas", 100),
    ("merfish_ileum_quick", "MERFISH · mouse ileum", 100),
    ("cosmx_nsclc_lung5_rep1_quick", "CosMx · human lung cancer", 100),
    ("iss_mouse_hippocampus_quick", "ISS · mouse hippocampus", 150),
]
SEG_THEMES = {
    "dark": dict(bg="#14161b", ink="#e8e6df", sub="#b5b3aa", noise="#62656c", line="#ffffff",
                 line_alpha=0.7, cells=["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181",
                                        "#2fbf2f", "#9085e9", "#e66767", "#5fc3d6", "#c0b23a"]),
    "light": dict(bg="#fcfcfb", ink="#0b0b0b", sub="#52514e", noise="#b5b3aa", line="#1b1b1b",
                  line_alpha=0.6, cells=["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
                                         "#008300", "#4a3aa7", "#e34948", "#1f9fb8", "#9a8a00"]),
}
SEG_COLORS = SEG_THEMES["dark"]["cells"]


def densest_window(x, y, size, step=None):
    step = step or size / 4
    xs = np.arange(x.min(), x.max() - size, step)
    ys = np.arange(y.min(), y.max() - size, step)
    best, arg = -1, (x.min(), y.min())
    for x0 in xs:
        sx = (x >= x0) & (x < x0 + size)
        yy = y[sx]
        for y0 in ys:
            n = int(((yy >= y0) & (yy < y0 + size)).sum())
            if n > best:
                best, arg = n, (x0, y0)
    return arg


def color_cells(centers: pd.DataFrame) -> dict:
    """Greedy colouring so that neighbouring cells get different colours."""
    from scipy.spatial import cKDTree
    ids = list(centers.index)
    if not ids:
        return {}
    pts = centers[["x", "y"]].to_numpy()
    tree = cKDTree(pts)
    k = min(9, len(ids))
    _, nn = tree.query(pts, k=k)
    nn = np.atleast_2d(nn)
    rng = np.random.default_rng(7)
    col = {}
    for i in rng.permutation(len(ids)):
        used = {col.get(ids[j]) for j in nn[i][1:]}
        free = [c for c in range(len(SEG_COLORS)) if c not in used]
        col[ids[i]] = (free or list(range(len(SEG_COLORS))))[rng.integers(len(free) if free else len(SEG_COLORS))]
    return {c: SEG_COLORS[v] for c, v in col.items()}


def seg_window(ds: str, size: float) -> dict:
    """Molecules, cell colours and polygons of the densest size x size window."""
    import shapely
    base = RUNS / "rc1636-real" / ds / "rep0" / "seg"
    src("segmentation_examples", base / "molecules.parquet", base / "cell_boundaries.parquet")
    m = pd.read_parquet(base / "molecules.parquet")
    is3d = "z" in m.columns
    x0, y0 = densest_window(m.x.to_numpy(), m.y.to_numpy(), size)
    w = m[(m.x >= x0) & (m.x < x0 + size) & (m.y >= y0) & (m.y < y0 + size)]
    if is3d:  # 3-D data: show the most populated z-plane
        zs = w.z.value_counts()
        w = w[w.z == float(zs.index[zs.argmax()])]
    cells = w[~w.is_noise & w.cell.notna()]
    b = pd.read_parquet(base / "cell_boundaries.parquet")
    geoms = shapely.from_wkb(b.geometry.to_numpy())
    keep = shapely.intersects(geoms, shapely.box(x0, y0, x0 + size, y0 + size))
    if is3d:  # 2-D outlines of the cells with >= 8 molecules in this plane
        n = cells.cell.value_counts()
        keep &= b.cell.isin(set(n.index[n >= 8])).to_numpy()
    rings = []
    for g in geoms[keep]:  # drawn whole; the axes clip them to the window
        for part in getattr(g, "geoms", [g]):
            if part.geom_type == "Polygon" and not part.is_empty:
                rings.append(np.asarray(part.exterior.coords))
    return dict(x0=x0, y0=y0, w=w, cells=cells, noise=w[w.is_noise | w.cell.isna()], is3d=is3d,
                order=color_cells(cells.groupby("cell")[["x", "y"]].mean()), rings=rings)


def fig_segmentation(out: Path):
    from PIL import Image
    wins = {ds: seg_window(ds, size) for ds, _, size in SEG_EXAMPLES}
    for mode, th in SEG_THEMES.items():
        plt.rcdefaults()
        plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
        bg, ink = th["bg"], th["ink"]
        fig, axes = plt.subplots(2, 2, figsize=(7.6, 8.1), facecolor=bg)
        for ax, (ds, title, size) in zip(axes.ravel(), SEG_EXAMPLES):
            d = wins[ds]
            x0, y0, w = d["x0"], d["y0"], d["w"]
            ms = float(np.clip(2.3e4 / max(len(w), 1), 0.8, 9))
            ax.scatter(d["noise"].x, d["noise"].y, s=ms, color=th["noise"], linewidths=0)
            col = {c: th["cells"][SEG_COLORS.index(v)] for c, v in d["order"].items()}
            ax.scatter(d["cells"].x, d["cells"].y, s=ms, c=d["cells"].cell.map(col), linewidths=0)
            for r in d["rings"]:
                ax.plot(r[:, 0], r[:, 1], color=th["line"], lw=0.6, alpha=th["line_alpha"])
            ax.set_xlim(x0, x0 + size)
            ax.set_ylim(y0, y0 + size)
            ax.set_aspect("equal")
            ax.axis("off")
            ax.set_title(title, color=ink, fontsize=10, fontweight="bold", loc="left")
            note = f"{size} × {size} µm" + (", one z-plane" if d["is3d"] else "")
            ax.text(0.0, -0.03, f"{note} · {len(w):,} mol. · {read_meta(ds)['stats']['n_genes']} genes",
                    transform=ax.transAxes, color=th["sub"], fontsize=8, va="top")
            sb = 20  # 20 µm scale bar on a backing plate
            xe, yb = x0 + size * 0.96, y0 + size * 0.05
            ax.add_patch(plt.Rectangle((xe - sb - size * 0.03, yb - size * 0.03), sb + size * 0.06,
                                       size * 0.13, color=bg, alpha=0.85, lw=0, zorder=5))
            ax.plot([xe - sb, xe], [yb, yb], color=ink, lw=2.5, solid_capstyle="butt", zorder=6)
            ax.text(xe - sb / 2, yb + size * 0.015, f"{sb} µm", color=ink, ha="center",
                    va="bottom", fontsize=8, zorder=6)
            NUMBERS.setdefault("segmentation_examples", {})[ds] = {
                "window_um": [float(x0), float(y0), size], "molecules": int(len(w)),
                "noise_fraction": float(len(d["noise"]) / max(len(w), 1)),
                "cells_with_molecules": int(d["cells"].cell.nunique()), "polygons": len(d["rings"])}
        fig.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.04, wspace=0.06, hspace=0.16)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=130, facecolor=bg)
        plt.close(fig)
        # palettised PNG: a fraction of the size, no visible loss for flat colours
        Image.open(buf).convert("RGB").quantize(colors=96, method=Image.Quantize.MEDIANCUT,
                                                dither=Image.Dither.NONE).save(
            out / f"segmentation_examples-{mode}.png", optimize=True)


UMAP_DATASETS = [("xenium_lung_cancer_admix", "Xenium · human lung cancer"),
                 ("xenium_breast_rep1_dense_full", "Xenium · human breast cancer")]
UMAP_COLORS = {"light": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300",
                         "#4a3aa7", "#e34948"],
               "dark": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300",
                        "#9085e9", "#e66767"]}


def umap_embedding(ds: str) -> pd.DataFrame:
    """Cells x genes counts from Baysor's molecule output -> UMAP coordinates."""
    cache = GEN / f"umap_{ds}.csv"
    base = RUNS / "rc1636-real" / ds / "rep0"
    src("umap", base / "seg/molecules.parquet", base / "celltypes.parquet")
    if cache.is_file():
        return pd.read_csv(cache)
    import anndata as ad
    import scanpy as sc
    import scipy.sparse as sp
    m = pd.read_parquet(base / "seg/molecules.parquet", columns=["cell", "gene", "is_noise"])
    m = m[~m.is_noise & m.cell.notna()]
    cells = pd.Index(sorted(m.cell.unique()))
    genes = pd.Index(sorted(m.gene.unique()))
    X = sp.coo_matrix((np.ones(len(m), dtype=np.float32),
                       (cells.get_indexer(m.cell), genes.get_indexer(m.gene))),
                      shape=(len(cells), len(genes))).tocsr()
    a = ad.AnnData(X)
    a.obs_names, a.var_names = cells.astype(str), genes.astype(str)
    sc.pp.normalize_total(a, target_sum=1e3)
    sc.pp.log1p(a)
    sc.pp.pca(a, n_comps=30, random_state=0)
    sc.pp.neighbors(a, n_neighbors=15, random_state=0)
    sc.tl.umap(a, random_state=0)
    ct = pd.read_parquet(base / "celltypes.parquet")
    ct["cell"] = "cell_" + ct.cell.astype(str)
    lab = ct.set_index("cell").celltype
    df = pd.DataFrame(a.obsm["X_umap"], columns=["u1", "u2"], index=a.obs_names)
    df["celltype"] = lab.reindex(df.index).fillna("untyped").to_numpy()
    df = df.reset_index().rename(columns={"index": "cell"})
    GEN.mkdir(exist_ok=True)
    df.to_csv(cache, index=False, float_format="%.4f")
    return df


@both_modes
def fig_umap(out, mode, t):
    fig, axes = plt.subplots(1, len(UMAP_DATASETS), figsize=(7.6, 4.0))
    for ax, (ds, title) in zip(axes, UMAP_DATASETS):
        df = umap_embedding(ds)
        types = df[df.celltype != "untyped"].celltype.value_counts().index.tolist()
        cols = UMAP_COLORS[mode]
        un = df[df.celltype == "untyped"]
        ax.scatter(un.u1, un.u2, s=1.2, color=t["muted"], linewidths=0, rasterized=True)
        for i, ctp in enumerate(types):
            s = df[df.celltype == ctp]
            ax.scatter(s.u1, s.u2, s=1.6, color=cols[i % len(cols)], linewidths=0, rasterized=True)
            cx, cy = np.median(s.u1), np.median(s.u2)
            ax.text(cx, cy, ctp.replace("cluster_", ""), fontsize=8.5, fontweight="bold",
                    color=t["ink"], ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.18", fc=t["surface"], ec="none", alpha=0.75))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for sp_ in ax.spines.values():
            sp_.set_visible(False)
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.set_title(f"{title}\n{len(df):,} cells, {len(types)} cell types", fontsize=10)
        NUMBERS.setdefault("umap", {})[ds] = {"cells": int(len(df)), "types": len(types),
                                              "untyped": int(len(un))}
    fig.tight_layout()
    save(fig, out, "umap", mode, fmt="png", dpi=140)


# ---- profiling figures ----

LADDERS = {"lung": "Xenium lung, 377 genes", "prime5k": "Xenium Prime 5K, 5,078 genes"}


def scaling_jobs(report: Path) -> pd.DataFrame:
    p = report / "summary-scaling-run/scaling_jobs.csv"
    src("scaling", p)
    j = pd.read_csv(p)
    j = j[(j.status == "ok") & (j.tool == "gperf")].copy()
    j["slide"] = j.dataset.str.extract(r"^(lung|prime5k)_")[0]
    return j


@both_modes
def fig_scaling(out, mode, t):
    after, before = scaling_jobs(PROF), scaling_jobs(PROF_BEFORE)
    fig, grid = plt.subplots(2, 2, figsize=(7.6, 6.6))
    axes = [grid[0, 0], grid[0, 1], grid[1, 0]]
    grid[1, 1].axis("off")
    for i, sl in enumerate(LADDERS):
        c = t["s"][i]
        a8 = after[(after.slide == sl) & (after.threads == 8)].sort_values("molecules_loaded")
        a1 = after[(after.slide == sl) & (after.threads == 1)].sort_values("molecules_loaded")
        b8 = before[(before.slide == sl) & (before.threads == 8)].sort_values("molecules_loaded")
        ax = axes[0]
        ax.plot(a8.molecules_loaded, a8.cpu_s, color=c, marker="o", markersize=5)
        ax.plot(b8.molecules_loaded, b8.cpu_s, color=c, ls="--", marker="o", markersize=5,
                markerfacecolor="none")
        ax = axes[1]
        ax.plot(a8.molecules_loaded, a8.wall_s, color=c, marker="o", markersize=5)
        if len(a1):
            ax.plot(a1.molecules_loaded, a1.wall_s, color=c, ls=":", marker="s", markersize=4.5,
                    markerfacecolor="none")
        ax = axes[2]
        ax.plot(a8.molecules_loaded, a8.peak_rss_kb / 2**20, color=c, marker="o", markersize=5)
        ax.plot(b8.molecules_loaded, b8.peak_rss_kb / 2**20, color=c, ls="--", marker="o",
                markersize=5, markerfacecolor="none")
    # whole-transcriptome slide as a single point
    w = after[after.dataset == "cosmx_wtx_colon_full"]
    wb = before[before.dataset == "cosmx_wtx_colon_full"]
    for ax, col, f in ((axes[0], "cpu_s", 1), (axes[2], "peak_rss_kb", 2**-20)):
        ax.scatter(w.molecules_loaded, w[col] * f, marker="D", s=34, color=t["s"][2], zorder=4)
        ax.scatter(wb.molecules_loaded, wb[col] * f, marker="D", s=30, facecolors="none",
                   edgecolors=t["s"][2], zorder=4)
    axes[1].scatter(w.molecules_loaded, w.wall_s, marker="D", s=34, color=t["s"][2], zorder=4)
    for ax, ylab, title in zip(axes, ("CPU time, s", "wall time, s", "peak memory (RSS), GiB"),
                               ("CPU time (8 threads)", "Wall time, 8 vs 1 thread",
                                "Peak memory (8 threads)")):
        log_axis(ax)
        ax.set_xlabel("molecules")
        ax.set_ylabel(ylab)
        ax.set_title(title)
    # ~linear guide on the CPU panel
    xs = np.array([1e5, 1e7])
    axes[0].plot(xs, 80 * xs / 1e5, color=t["muted"], lw=1, ls=":")
    axes[0].text(2.2e6, 80 * 2.2e6 / 1e5 * 1.3, "linear", color=t["muted"], fontsize=7.5, rotation=30)
    h = [Line2D([], [], color=t["s"][i], lw=2, label=lab) for i, lab in enumerate(LADDERS.values())]
    h += [Line2D([], [], color=t["s"][2], marker="D", ls="", label="CosMx WTx, 18,935 genes"),
          Line2D([], [], color=t["ink2"], marker="o", label=f"{NEW}, 8 threads"),
          Line2D([], [], color=t["ink2"], ls="--", marker="o", markerfacecolor="none",
                 label="before optimisation (e45fddc), 8 threads"),
          Line2D([], [], color=t["ink2"], ls=":", marker="s", markerfacecolor="none",
                 label=f"{NEW}, 1 thread")]
    fig.tight_layout()
    grid[1, 1].legend(handles=h, loc="center left", bbox_to_anchor=(0.0, 0.5))
    save(fig, out, "scaling", mode)


PHASE_GROUPS = [("BMM iterations", {"bmm_iterations", "bmm_init"}),
                ("molecule clustering", {"mol_clustering"}),
                ("NCV colours", {"ncv_colors"}),
                ("everything else", None)]
CROP_LABEL = {
    "osmfish_g35_20k": "osmFISH, 35 genes", "xenium_pancreas_g377_20k": "Xenium pancreas, 328 genes",
    "merfish_ileum_3d_20k": "MERFISH 3D, 208 genes", "cosmx_nsclc_g960_20k": "CosMx, 922 genes",
    "sim_circles_g1000_20k": "simulated, 995 genes", "xenium_prime5k_20k": "Xenium Prime 5K, 3,335 genes",
    "sim_circles_g5000_10k": "simulated, 2,793 genes (10k)", "cosmx_wtx_20k": "CosMx WTx, 8,407 genes",
}


def group_of(phase: str) -> str:
    for name, s in PHASE_GROUPS:
        if s and phase in s:
            return name
    return "everything else"


@both_modes
def fig_phases(out, mode, t):
    p = PROF / "summary-report-run/phases.csv"
    src("phases", p, PROF / "summary-scaling-run/scaling_phases.csv")
    ph = pd.read_csv(p)
    ph = ph[ph.job.isin([f"callgrind-{d}-t1" for d in CROP_LABEL])]  # plain 1-thread job per crop
    ph = ph[~ph.phase.isin(["other_threads", "pool_worker_runtime"])]
    ph["group"] = ph.phase.map(group_of)
    crop = ph.groupby(["dataset", "group"]).Ir.sum().unstack(fill_value=0) / 1e9
    crop = crop.loc[[d for d in CROP_LABEL if d in crop.index]][[g for g, _ in PHASE_GROUPS]]

    sp_ = pd.read_csv(PROF / "summary-scaling-run/scaling_phases.csv")
    sp_ = sp_[(sp_.tool == "gperf") & (sp_.threads == 8)]
    sp_["group"] = sp_.phase.map(group_of)
    real = sp_.groupby(["dataset", "group"]).cpu_s.sum().unstack(fill_value=0)
    order = ["lung_100k", "lung_1M", "lung_4M", "lung_all", "prime5k_100k", "prime5k_1M", "prime5k_8M",
             "cosmx_wtx_colon_full"]
    real = real.loc[[o for o in order if o in real.index]][[g for g, _ in PHASE_GROUPS]]
    real_share = real.div(real.sum(axis=1), axis=0) * 100
    mol = sp_.groupby("dataset").molecules.first()

    def rung_label(d):
        name = {"lung_all": "lung, whole slide", "cosmx_wtx_colon_full": "CosMx WTx slide"}.get(
            d, d.split("_")[0])
        n = mol[d]
        return f"{name} ({n / 1e6:.1f}M)" if n >= 1e6 else f"{name} ({n / 1e3:.0f}k)"

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 7.0))
    cols = [t["s"][0], t["s"][1], t["s"][3], t["base"]]
    for ax, data, xlab, title, labels in (
            (axes[0], crop, "instructions, 10⁹ (callgrind, 1 thread)",
             "20k-molecule crops: where the instructions go", [CROP_LABEL[d] for d in crop.index]),
            (axes[1], real_share, "share of CPU time, % (8 threads)",
             "Real sizes: where the CPU time goes", [rung_label(d) for d in real_share.index])):
        left = np.zeros(len(data))
        y = np.arange(len(data))[::-1]
        for (g, _), c in zip(PHASE_GROUPS, cols):
            v = data[g].to_numpy()
            ax.barh(y, v, left=left, color=c, height=0.68, edgecolor=t["surface"], linewidth=1.2,
                    label=g)
            left += v
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel(xlab)
        ax.set_title(title)
    axes[1].set_xlim(0, 100)
    h = [plt.Rectangle((0, 0), 1, 1, color=c, label=g) for (g, _), c in zip(PHASE_GROUPS, cols)]
    fig.legend(handles=h, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout()
    save(fig, out, "phases", mode)
    NUMBERS["phases_real_share"] = real_share.round(1).to_dict(orient="index")
    NUMBERS["phases_crop_GIr"] = crop.round(2).to_dict(orient="index")


# --------------------------------------------------------------- tables ----

TABLE_DATASETS = [
    "xenium_pancreas_377_quick", "xenium_lung_cancer_admix", "xenium_breast_rep1_dense_full",
    "xenium_pancreas_377_full", "xenium_prime5k_ovarian_quick", "xenium_prime5k_ovarian_full",
    "cosmx_nsclc_lung5_rep1_full", "cosmx_wtx_colon_quick", "merfish_ileum_full",
    "iss_mouse_hippocampus_quick", "osmfish_somatosensory_quick", "starmap_visual_cortex_quick",
]


def fmt_s(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    if v >= 600:
        return f"{v / 60:.1f} min"
    return f"{v:.0f} s" if v >= 20 else f"{v:.1f} s"


def fmt_mem(mib):
    if mib is None or (isinstance(mib, float) and math.isnan(mib)):
        return "—"
    return f"{mib / 1024:.2f} GiB" if mib >= 1024 else f"{mib:.0f} MiB"


def tables() -> str:
    df = runtime_table()
    rm = remeasured()
    out = []
    six = df[df.threads == 6].set_index("dataset")
    one = df[df.threads == 1].set_index("dataset")
    src("table_datasets", RC / "tables/runtime.csv", DATA / "real", RUNS / "rc1636-real")
    out.append("## Per-dataset runtime (A/B campaign, median of replicates)\n")
    out.append("Source: `$BAYSOR_BENCH_DATA/rc/2026-10-01-1636267/tables/runtime.csv` "
               "(wall_rc/wall_old, rss_rc_mib/rss_old_mib, load_rc/load_old), molecules and "
               "genes from `runs/rc1636-real/<id>/metrics.json`, platform from "
               "`real/<id>/meta.json`.\n")
    out.append("| dataset | platform | molecules | genes | wall 6 thr new | wall 6 thr 0.8.3 | "
               "RSS 6 thr new | RSS 6 thr 0.8.3 | wall 1 thr new | wall 1 thr 0.8.3 | load |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|")
    tab = {}
    for ds in TABLE_DATASETS:
        r = six.loc[ds]
        o = one.loc[ds] if ds in one.index else None
        row = dict(platform=r.platform, molecules=int(r.n_molecules), genes=int(r.n_genes),
                   wall6_new=r.wall_rc, wall6_old=r.wall_old, rss6_new=r.rss_rc_mib,
                   rss6_old=r.rss_old_mib, wall1_new=None if o is None else o.wall_rc,
                   wall1_old=None if o is None else o.wall_old, load=r.load_pair)
        tab[ds] = row
        out.append(f"| {ds} | {r.platform} | {row['molecules']:,} | {row['genes']:,} | "
                   f"{fmt_s(r.wall_rc)} | {fmt_s(r.wall_old)} | {fmt_mem(r.rss_rc_mib)} | "
                   f"{fmt_mem(r.rss_old_mib)} | {fmt_s(row['wall1_new'])} | {fmt_s(row['wall1_old'])} | "
                   f"{r.load_pair:.1f} |")
    NUMBERS["table_datasets"] = tab

    # aggregate ratios over all 6-thread datasets
    s6 = df[df.threads == 6]
    agg = {
        "n_datasets": int(len(s6)),
        "n_real": int((s6.kind == "real").sum()), "n_sim": int((s6.kind == "sim").sum()),
        "wall_ratio_median": float(s6.wall_ratio.median()),
        "wall_ratio_min": float(s6.wall_ratio.min()), "wall_ratio_max": float(s6.wall_ratio.max()),
        "rss_ratio_median": float(s6.rss_ratio.median()),
        "cpu_ratio_median": float(s6.cpu_ratio.median()),
        "n_slower": int((s6.wall_ratio > 1).sum()),
        "slower": s6[s6.wall_ratio > 1].dataset.tolist(),
        "load_min": float(s6[["load_rc", "load_old"]].min().min()),
        "load_max": float(s6[["load_rc", "load_old"]].max().max()),
    }
    NUMBERS["aggregate_6thr"] = agg
    out.append("\n## Aggregate over all 6-thread datasets (runtime.csv)\n")
    out.append("| measure | value |\n|---|---|")
    for k, v in agg.items():
        out.append(f"| {k} | {v if not isinstance(v, float) else round(v, 3)} |")

    # re-measurement, one process at a time
    if not rm.empty:
        src("remeasure", REMEASURE)
        out.append("\n## Re-measurement, one process at a time (remeasure.py)\n")
        out.append(f"Source: `{str(REMEASURE).replace(str(DATA), '$BAYSOR_BENCH_DATA')}`.\n")
        out.append("| exp | dataset | threads | wall new | wall 0.8.3 | CPU new | CPU 0.8.3 | "
                   "RSS new | RSS 0.8.3 | load (start, both runs) |")
        out.append("|---|---|---|---|---|---|---|---|---|---|")
        rmn = {}
        for (exp, ds, th), g in rm.groupby(["exp", "dataset", "threads"], sort=False):
            n = g[g.version == "rc"].iloc[-1] if (g.version == "rc").any() else None
            o = g[g.version == "old"].iloc[-1] if (g.version == "old").any() else None
            get = lambda r, c: None if r is None else float(r[c])  # noqa: E731
            rmn[f"{exp}/{ds}/t{th}"] = {k: get(r, c) for k, r, c in (
                ("wall_new", n, "wall_s"), ("wall_old", o, "wall_s"), ("cpu_new", n, "cpu_s"),
                ("cpu_old", o, "cpu_s"), ("rss_new_mib", n, "rss_mib"), ("rss_old_mib", o, "rss_mib"))}
            loads = ", ".join(str(r["load1_start"]) for r in (n, o) if r is not None)
            out.append(f"| {exp} | {ds} | {th} | {fmt_s(get(n, 'wall_s'))} | {fmt_s(get(o, 'wall_s'))} | "
                       f"{fmt_s(get(n, 'cpu_s'))} | {fmt_s(get(o, 'cpu_s'))} | "
                       f"{fmt_mem(get(n, 'rss_mib'))} | {fmt_mem(get(o, 'rss_mib'))} | {loads} |")
        NUMBERS["remeasure"] = rmn

    # determinism: identical replicate outputs per dataset (3 replicates, 6 threads)
    det = {}
    for v, runs in (("rc", ("rc1636-real", "rc1636-sim")), ("old", ("v0083-real", "v0083-sim"))):
        same = total = 0
        for run in runs:
            for p in sorted((RUNS / run).glob("*/metrics.json")):
                reps = json.loads(p.read_text())["reps"]
                hs = {r.get("assignment_sha256") for r in reps if r.get("status") == "ok"}
                if len(reps) >= 2:
                    total += 1
                    same += len(hs) == 1
            src("determinism", RUNS / run)
        det[v] = {"identical": same, "datasets": total}
    NUMBERS["determinism"] = det
    out.append("\n## Run-to-run identical output (all replicates same assignment sha256, 6 threads)\n")
    out.append("Source: `runs/{rc1636,v0083}-{real,sim}/*/metrics.json` (`reps[].assignment_sha256`).\n")
    out.append("| version | datasets with identical replicates | datasets |\n|---|---|---|")
    for v, d in det.items():
        out.append(f"| {NEW if v == 'rc' else OLD} | {d['identical']} | {d['datasets']} |")
    iss = {run: [r["n_cells"] for r in metrics(run, "iss_mouse_hippocampus_quick")["reps"]]
           for run in ("v0083-real", "v0083-t1", "rc1636-real", "rc1636-t1")}
    NUMBERS["iss_cells"] = iss
    out.append("\nISS cells per replicate: " + "; ".join(f"{k}: {v}" for k, v in iss.items()))

    # accuracy summary on simulated data
    q = pd.read_csv(RC / "tables/quality_sim.csv")
    q = q[q.threads == 6]
    acc = {"n": int(len(q)), "delta_median": float(q.accuracy_1to1_delta.median()),
           "delta_min": float(q.accuracy_1to1_delta.min()),
           "delta_max": float(q.accuracy_1to1_delta.max()),
           "n_better_gt_0.005": int((q.accuracy_1to1_delta > 0.005).sum()),
           "n_worse_gt_0.005": int((q.accuracy_1to1_delta < -0.005).sum()),
           "worst": q.loc[q.accuracy_1to1_delta.idxmin(), "dataset"],
           "best": q.loc[q.accuracy_1to1_delta.idxmax(), "dataset"],
           "ari_delta_median": float(q.ari_assigned_delta.median())}
    NUMBERS["accuracy_sim"] = acc
    src("accuracy_sim", RC / "tables/quality_sim.csv")
    out.append("\n## Accuracy on simulated data (quality_sim.csv, 6 threads, mean of 3)\n")
    out.append("| measure | value |\n|---|---|")
    for k, v in acc.items():
        out.append(f"| {k} | {v if not isinstance(v, float) else round(v, 4)} |")

    # profiling numbers
    a, b = scaling_jobs(PROF), scaling_jobs(PROF_BEFORE)
    out.append("\n## Real-size ladders (profiling scaling tier, gperftools jobs)\n")
    out.append("Source: `profiling/reports/2026-10-01-20bc45c/summary-scaling-run/scaling_jobs.csv` "
               "(after) and `profiling/reports/2026-09-30-e45fddc/summary-scaling-run/scaling_jobs.csv` "
               "(before).\n")
    out.append("| rung | threads | molecules | genes | wall | CPU | CPU before | peak RSS | RSS before | "
               "RSS per molecule | load at start |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|")
    lad = {}
    for _, r in a.sort_values(["threads", "dataset"]).iterrows():
        rb = b[(b.dataset == r.dataset) & (b.threads == r.threads)]
        cb = float(rb.cpu_s.iloc[0]) if len(rb) else None
        rssb = float(rb.peak_rss_kb.iloc[0]) / 1024 if len(rb) else None
        per = r.peak_rss_kb * 1024 / r.molecules_loaded
        lad[f"{r.dataset}/t{r.threads}"] = dict(molecules=int(r.molecules_loaded), genes=int(r.genes_loaded),
                                                wall=float(r.wall_s), cpu=float(r.cpu_s), cpu_before=cb,
                                                rss_mib=float(r.peak_rss_kb / 1024), rss_before_mib=rssb,
                                                bytes_per_molecule=float(per))
        out.append(f"| {r.dataset} | {r.threads} | {int(r.molecules_loaded):,} | {int(r.genes_loaded):,} | "
                   f"{fmt_s(r.wall_s)} | {fmt_s(r.cpu_s)} | {fmt_s(cb)} | {fmt_mem(r.peak_rss_kb / 1024)} | "
                   f"{fmt_mem(rssb)} | {per:,.0f} B | {r.loadavg_start} |")
    NUMBERS["ladders"] = lad

    fits = pd.read_csv(PROF / "summary-scaling-run/scaling_fits.csv")
    src("fits", PROF / "summary-scaling-run/scaling_fits.csv")
    f = fits[(fits.what.isin(["total", "phase_cpu"])) & (fits.tool == "gperf")]
    out.append("\n## Scaling exponents (scaling_fits.csv: value ∝ molecules^exponent)\n")
    out.append("| slide | threads | molecules | what | name | exponent |\n|---|---|---|---|---|---|")
    fx = {}
    for _, r in f.iterrows():
        if r.what == "phase_cpu" and r["name"] not in ("mol_clustering", "bmm_iterations", "ncv_colors",
                                                       "polygons", "molecule_graph", "confidence"):
            continue
        fx[f"{r.slide}/t{r.threads}/{r['name']}"] = float(r.exponent)
        out.append(f"| {r.slide} | {r.threads} | {int(r.molecules_min):,}–{int(r.molecules_max):,} | "
                   f"{r.what} | {r['name']} | {r.exponent} |")
    NUMBERS["exponents"] = fx

    nat = pd.read_csv(PROF / "summary-report-run/native.csv")
    src("crop_threads", PROF / "summary-report-run/native.csv")
    nat = nat[(nat.dataset.isin(["xenium_pancreas_g377_20k", "xenium_prime5k_20k"])) & (nat.wait_policy == "default")]
    out.append("\n## Crop thread series (native.csv, min wall of 3, default wait policy; includes NCV colours)\n")
    out.append("| crop | threads | wall min | CPU median | load mean |\n|---|---|---|---|---|")
    cr = {}
    for _, r in nat.sort_values(["dataset", "threads"]).iterrows():
        cr[f"{r.dataset}/t{r.threads}"] = dict(wall_min=float(r.wall_s_min), cpu_median=float(r.cpu_s_median))
        out.append(f"| {r.dataset} | {r.threads} | {r.wall_s_min:.2f} s | {r.cpu_s_median:.1f} s | "
                   f"{r.loadavg_1m_mean} |")
    NUMBERS["crop_threads"] = cr
    return "\n".join(out) + "\n"


def write_sources() -> str:
    lines = ["\n## Figure and table sources\n", "| figure / table | files |", "|---|---|"]
    for k in sorted(SOURCES):
        lines.append(f"| {k} | " + "<br>".join(f"`{p}`" for p in sorted(SOURCES[k])) + " |")
    return "\n".join(lines) + "\n"


FIGURES = [fig_runtime_vs_molecules, fig_threads, fig_genes, fig_accuracy, fig_determinism,
           fig_celladmix, fig_umap, fig_scaling, fig_phases]


def main():
    ap = argparse.ArgumentParser(description="Regenerate the Baysor docs performance figures.")
    ap.add_argument("--docs", type=Path,
                    default=Path(os.environ.get("BAYSOR_DOCS_PERF",
                                                "/home/vpetukhov/Projects/Baysor/docs/performance")),
                    help="docs/performance directory of a Baysor checkout (figures go to <docs>/img)")
    ap.add_argument("--only", nargs="*", help="figure function names to run (default: all)")
    args = ap.parse_args()
    img = args.docs / "img"
    img.mkdir(parents=True, exist_ok=True)
    GEN.mkdir(exist_ok=True)
    for fn in FIGURES + [fig_segmentation]:
        if args.only and fn.__name__ not in args.only:
            continue
        print("figure:", fn.__name__, flush=True)
        fn(img)
    md = "# Numbers quoted in docs/performance (generated by make_figures.py)\n\n" + tables() + write_sources()
    (GEN / "tables.md").write_text(md)
    (GEN / "numbers.json").write_text(json.dumps(NUMBERS, indent=1, default=float) + "\n")
    total = sum(p.stat().st_size for p in args.docs.rglob("*") if p.is_file())
    print(f"wrote {img} ; docs/performance total {total / 1e6:.2f} MB ; tables {GEN / 'tables.md'}")


if __name__ == "__main__":
    main()
