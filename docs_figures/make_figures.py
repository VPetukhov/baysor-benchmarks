#!/usr/bin/env python3
"""Figures, chart data and tables of the "Performance" pages of the Baysor docs.

One command regenerates every figure of ``docs/performance/`` in a Baysor
checkout, the data of its interactive charts, its per-dataset table and the
owner-only profiling report::

    .deps/bench/bin/python docs_figures/make_figures.py \\
        --docs /path/to/Baysor/docs/performance

Inputs (all read-only, under ``$BAYSOR_BENCH_DATA``, default
``/home/vpetukhov/Projects/Baysor/.bench-data``):

* ``rc/2026-10-01-1636267/tables/{runtime,quality_sim,quality_real}.csv`` and
  ``runs/{rc1636,v0083}-{real,sim,t1,t1b}/`` - the release-candidate A/B
  benchmark (cpp-0.9.0 candidate vs cpp-0.8.3, 77 datasets);
* ``docs-figures/remeasure/runs.tsv`` - the one-process-at-a-time re-measurement written by
  ``remeasure.py`` (thread sweep and the real full-tier datasets);
* ``docs-figures/ncv/<dataset>/`` - the segmentation-example crops re-run with
  NCV colours on by ``ncv_runs.py``;
* ``profiling/reports/2026-10-01-20bc45c/`` and
  ``profiling/reports/2026-09-30-e45fddc/`` - the profiling summaries (after /
  before the optimisation);
* ``real/<id>/meta.json`` and ``sim/<id>/meta.json`` - dataset facts.

Outputs:

* ``<docs>/data/*.json`` - the specs of the interactive charts, drawn in the
  browser by ``docs/assets/perfcharts/perfcharts.js`` of the Baysor repository;
* ``<docs>/img/*-{light,dark}.png`` - the scatterplots of biological data
  (segmentation examples, UMAPs), one image per Material colour scheme
  (``#only-light`` / ``#only-dark``);
* the per-dataset table of ``<docs>/profiling.md`` (between the
  ``docs_figures:runtime-table`` markers);
* ``docs_figures/generated/owner_report.md`` and ``generated/owner/`` - the
  owner-only "where the time goes" / "current bottlenecks" report (not in the docs);
* ``docs_figures/generated/tables.md`` - every number quoted in the pages,
  as Markdown tables, each with the file it comes from;
* ``docs_figures/generated/numbers.json`` - the same numbers, machine-readable.
"""
from __future__ import annotations

import argparse
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


# --------------------------------------------------- interactive charts ----
#
# The benchmark and statistics charts of the docs are drawn in the browser by
# docs/javascripts/perfcharts.js (vendored in the Baysor repository) from one
# JSON spec per chart, written to <docs>/data/<name>.json:
#
#   {"columns": 1|2, "legend": [{name, color, marker, line, open}],
#    "panels": [{"title", "height", "marginLeft", "hover": "point"|"x",
#                "x"/"y": {"label", "log", "fmt": "si"|"plain"|"pct", "ticks", "min", "max",
#                          "type": "band", "categories"},
#                "series": [{name, color, marker, line, open,
#                            points: [{x, y, title, rows: [[value, label]], xTitle, yText, rowLabel}]}],
#                "refs": [{"type": "vline", x, label} | {"type": "line", points, dash, label, labelAt}],
#                "labels": [{x, y, text, dx, dy}], "links": [{x1, x2, y}]}],
#    "table": {"columns": [...], "rows": [[...]]}}
#
# Colours are slot numbers (0-2: the categorical palette) or token names
# ("ink2", "base"); the stylesheet maps them to the light or dark theme.

NAME = {
    "xenium_pancreas_377": "Xenium pancreas", "xenium_lung_cancer": "Xenium lung cancer",
    "xenium_breast_rep1_dense": "Xenium breast cancer", "xenium_breast_rep1_stroma": "Xenium breast stroma",
    "xenium_breast_rep1_imageprior": "Xenium breast cancer, image prior",
    "xenium_breast_rep1_z": "Xenium breast cancer, z-stack", "xenium_breast_rep2_dense": "Xenium breast cancer rep2",
    "xenium_mouse_brain_ff": "Xenium mouse brain", "xenium_mouse_brain_ff_edge": "Xenium mouse brain, tissue edge",
    "xenium_prime5k_ovarian": "Xenium Prime 5K ovarian", "cosmx_nsclc_lung5_rep1": "CosMx lung cancer",
    "cosmx_wtx_colon": "CosMx WTx colon", "merfish_ileum": "MERFISH ileum (3D)",
    "iss_mouse_hippocampus": "ISS hippocampus", "osmfish_somatosensory": "osmFISH cortex",
    "starmap_visual_cortex": "STARmap cortex (3D)",
}
PLATFORM_SLOT = {"Xenium": 0, "CosMx SMI": 1, "MERFISH": 2, "other": "ink2"}
PLATFORM_SHAPE = {"Xenium": "circle", "CosMx SMI": "square", "MERFISH": "triangle", "other": "diamond"}


def name(ds: str) -> str:
    base = re.sub(r"_(quick|full|admix)$", "", ds)
    return NAME.get(base, base)


def size_tag(n: float) -> str:
    return f"{n / 1e6:.1f}M" if n >= 9.5e5 else f"{n / 1e3:.0f}k"


def r4(v):
    """Round floats for compact JSON."""
    if isinstance(v, float):
        return float(f"{v:.4g}")
    if isinstance(v, dict):
        return {k: r4(x) for k, x in v.items()}
    if isinstance(v, list):
        return [r4(x) for x in v]
    return v


def write_chart(out: Path, name_: str, spec: dict):
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name_}.json").write_text(json.dumps(r4(spec), separators=(",", ":"), ensure_ascii=False) + "\n")


def load_txt(v) -> str:
    return "—" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.1f}"


def chart_runtime_vs_molecules(out: Path):
    """Figure 1 of Profiling: wall time and peak RSS vs molecules, real crops, current version."""
    df = runtime_table()
    d = df[(df.kind == "real") & (df.threads == 6)].copy()
    d["group"] = d.platform.map(platform_group)
    src("runtime_vs_molecules", RC / "tables/runtime.csv", RUNS / "rc1636-real", DATA / "real")
    panels = []
    for col, ylab, title, fmt in (("wall_rc", "wall time, s", "Wall time (6 threads)", fmt_s),
                                  ("rss_rc_mib", "peak memory (RSS), MiB", "Peak memory (6 threads)", fmt_mem)):
        series = []
        for g in PLATFORM_ORDER:
            s = d[d.group == g].sort_values("n_molecules")
            pts = []
            for _, r in s.iterrows():
                pts.append(dict(x=int(r.n_molecules), y=float(r[col]),
                                title=f"{name(r.dataset)}, {size_tag(r.n_molecules)} molecules",
                                rows=[[fmt(r[col]), title.lower()],
                                      [f"{int(r.n_molecules):,}", "molecules"], [f"{int(r.n_genes):,}", "genes"],
                                      [fmt_s(r.cpu_rc), "CPU time"] if col == "wall_rc" else
                                      [fmt_s(r.wall_rc), "wall time"],
                                      [f"{int(r.n_rc)}", "runs, median shown"],
                                      [load_txt(r.load_rc), "mean 1-min load"],
                                      [r.dataset, "benchmark id"]]))
            series.append(dict(name=PLATFORM_LABEL[g], color=PLATFORM_SLOT[g], marker=PLATFORM_SHAPE[g],
                               size=4.5, points=pts))
        labels = [dict(x=int(r.n_molecules), y=float(r[col]), dx=9, dy=4,
                       text=f"{name(r.dataset)}, {int(r.n_genes):,} genes")
                  for _, r in d[d.n_genes >= 4000].iterrows()]
        panels.append(dict(title=title, height=330, x=dict(label="molecules", log=True),
                           y=dict(label=ylab, log=True, fmt="plain"), series=series, labels=labels))
    rows = [[name(r.dataset), r.platform, f"{int(r.n_molecules):,}", f"{int(r.n_genes):,}", fmt_s(r.wall_rc),
             fmt_s(r.cpu_rc), fmt_mem(r.rss_rc_mib), int(r.n_rc), load_txt(r.load_rc), r.dataset]
            for _, r in d.sort_values("n_molecules").iterrows()]
    write_chart(out, "runtime_vs_molecules", dict(
        columns=1,
        legend=[dict(name=PLATFORM_LABEL[g], color=PLATFORM_SLOT[g], marker=PLATFORM_SHAPE[g])
                for g in PLATFORM_ORDER],
        panels=panels,
        table=dict(columns=["dataset", "platform", "molecules", "genes", "wall time", "CPU time", "peak RSS",
                            "runs", "load", "benchmark id"], rows=rows)))


def chart_threads(out: Path):
    """Thread sweep (remeasure.py), current version."""
    df = remeasured()
    sw = df[(df.exp == "sweep") & (df.version == "rc")]
    src("threads", REMEASURE)
    if sw.empty:
        return
    dsets = list(dict.fromkeys(sw.dataset))
    mol = {ds: metrics("rc1636-real", ds)["dataset"]["n_molecules"] for ds in dsets}
    label = {ds: f"{name(ds)}, {size_tag(mol[ds])}" for ds in dsets}
    panels, rows = [], []
    for col, ylab, title in (("wall_s", "wall time, s", "Wall time vs threads"),
                             ("cpu_s", "CPU time (user + sys), s", "CPU time vs threads")):
        series, refs = [], []
        for i, ds in enumerate(dsets):
            s = sw[sw.dataset == ds].sort_values("threads")
            t1 = float(s[s.threads == 1][col].iloc[0])
            pts = []
            for _, r in s.iterrows():
                v = float(r[col])
                extra = (f"{t1 / v:.1f}× vs 1 thread" if col == "wall_s"
                         else f"{(v / t1 - 1) * 100:+.0f} % vs 1 thread")
                pts.append(dict(x=int(r.threads), y=v, xTitle=f"{int(r.threads)} thread" +
                                ("s" if r.threads > 1 else ""), yText=fmt_s(v),
                                rowLabel=f"{label[ds]} · {extra} · load {r.load1_start:.1f}"))
            series.append(dict(name=label[ds], color=i, marker="circle", line="solid", points=pts))
            if col == "wall_s":
                refs.append(dict(type="line", points=[[1, t1], [16, t1 / 16]], dash="dotted",
                                 label="ideal" if i == 0 else None, labelAt=[2.6, t1 / 2.6]))
        refs.append(dict(type="vline", x=8, label="8 cores"))
        panels.append(dict(title=title, height=300, hover="x",
                           x=dict(label="threads", log=True, fmt="plain", ticks=[1, 2, 4, 8, 16], min=0.85, max=18),
                           y=dict(label=ylab, log=True, fmt="plain"), series=series, refs=refs))
    for _, r in sw.sort_values(["dataset", "threads"]).iterrows():
        rows.append([label[r.dataset], int(r.threads), fmt_s(r.wall_s), fmt_s(r.cpu_s), fmt_mem(r.rss_mib),
                     f"{r.load1_start:.1f}"])
    write_chart(out, "threads", dict(
        columns=2, legend=[dict(name=label[ds], color=i, marker="circle", line="solid")
                           for i, ds in enumerate(dsets)] +
        [dict(name="ideal scaling", color="muted", line="dotted")],
        panels=panels,
        table=dict(columns=["dataset", "threads", "wall time", "CPU time", "peak RSS", "load at start"],
                   rows=rows)))
    NUMBERS["threads_sweep"] = {f"{ds}/t{int(r.threads)}": dict(wall=float(r.wall_s), cpu=float(r.cpu_s))
                                for ds in dsets for _, r in sw[sw.dataset == ds].iterrows()}


GENE_FAMILIES = [
    ("simulated circles, 92k molecules",
     ["sim_circles_gaps_g100", "sim_circles_gaps_g1000", "sim_circles_gaps_g5000"]),
    ("simulated tiles, 116k molecules",
     ["sim_tiled_distinct_g100", "sim_tiled_distinct_g1000", "sim_tiled_distinct_g5000"]),
    ("simulated tissue (strec), 64k molecules",
     ["strec_dense_s2_disjoint", "strec_dense_s2_merfish", "strec_dense_s2_xenium",
      "strec_dense_s2_prime5k1000", "strec_dense_s2_prime5k5000"]),
]


def chart_genes(out: Path):
    """Wall time and peak RSS vs gene-panel size, simulated families, current version."""
    df = runtime_table()
    d = df[df.threads == 6].set_index("dataset")
    src("time_vs_genes", RC / "tables/runtime.csv", RUNS / "rc1636-sim", DATA / "sim")
    panels, rows = [], []
    for col, ylab, title, fmt in (("wall_rc", "wall time, s", "Wall time vs gene panel (6 threads)", fmt_s),
                                  ("rss_rc_mib", "peak memory (RSS), MiB", "Peak memory vs gene panel (6 threads)",
                                   fmt_mem)):
        series = []
        for i, (fam, dss) in enumerate(GENE_FAMILIES):
            s = d.loc[dss].sort_values("n_genes")
            pts = [dict(x=int(r.n_genes), y=float(r[col]), title=ds,
                        rows=[[fmt(r[col]), title.split(" vs")[0].lower()], [f"{int(r.n_genes):,}", "genes"],
                              [f"{int(r.n_molecules):,}", "molecules"],
                              [fmt_s(r.wall_rc) if col != "wall_rc" else fmt_mem(r.rss_rc_mib),
                               "wall time" if col != "wall_rc" else "peak RSS"],
                              [f"{int(r.n_rc)}", "runs, median shown"], [load_txt(r.load_rc), "mean 1-min load"],
                              [fam, "family"]])
                   for ds, r in s.iterrows()]
            series.append(dict(name=fam, color=i, marker="circle", line="solid", points=pts))
        panels.append(dict(title=title, height=300, x=dict(label="genes in the panel", log=True),
                           y=dict(label=ylab, log=True, fmt="plain"), series=series))
    for fam, dss in GENE_FAMILIES:
        for ds, r in d.loc[dss].sort_values("n_genes").iterrows():
            rows.append([fam, ds, f"{int(r.n_genes):,}", f"{int(r.n_molecules):,}", fmt_s(r.wall_rc),
                         fmt_mem(r.rss_rc_mib), int(r.n_rc)])
    write_chart(out, "time_vs_genes", dict(
        columns=2, legend=[dict(name=f, color=i, marker="circle", line="solid")
                           for i, (f, _) in enumerate(GENE_FAMILIES)],
        panels=panels,
        table=dict(columns=["family", "dataset", "genes", "molecules", "wall time", "peak RSS", "runs"],
                   rows=rows)))


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


def chart_celladmix(out: Path):
    """cellAdmix total rate per dataset and the top pairs on the Xenium lung crop."""
    q = admix_rows()
    src("celladmix", RC / "tables/quality_real.csv")
    cats = [ADMIX_LABEL.get(d, d) for d in q.dataset]
    old_pts, new_pts, links, rows = [], [], [], []
    for c, (_, r) in zip(cats, q.iterrows()):
        tip = [[f"{r.admix_rc * 100:.2f} %", NEW], [f"{r.admix_old * 100:.2f} %", OLD],
               [f"{int(r.cells_rc):,}", f"cells ({NEW})"], [f"{int(r.n_rc)}", "runs, mean shown"],
               [r.dataset, "benchmark id"]]
        new_pts.append(dict(x=float(r.admix_rc * 100), y=c, title=c, rows=tip))
        old_pts.append(dict(x=float(r.admix_old * 100), y=c, title=c, rows=[tip[1], tip[0]] + tip[2:]))
        links.append(dict(x1=float(r.admix_old * 100), x2=float(r.admix_rc * 100), y=c))
        rows.append([c, f"{r.admix_rc * 100:.2f} %", f"{r.admix_old * 100:.2f} %", f"{int(r.cells_rc):,}",
                     int(r.n_rc)])
    top = dict(title="cellAdmix: total admixture rate", height=len(cats) * 24 + 60, marginLeft=200,
               x=dict(label="admixed molecules, % of assigned (lower is cleaner)", fmt="plain"),
               y=dict(type="band", categories=cats), links=links,
               series=[dict(name=OLD, color="base", marker="circle", open=True, points=old_pts),
                       dict(name=NEW, color=0, marker="circle", points=new_pts)])

    ds = "xenium_lung_cancer_admix"
    pairs = defaultdict(lambda: {"rc": [], "old": []})
    for run, v in (("rc1636-real", "rc"), ("v0083-real", "old")):
        src("celladmix", RUNS / run / ds / "metrics.json")
        for r in metrics(run, ds)["reps"]:
            for pr in r["celladmix"]["pairs"]:
                pairs[(pr["source"], pr["target"])][v].append(pr["rate"])
    prs = [(k, np.mean(v["rc"]) if v["rc"] else np.nan, np.mean(v["old"]) if v["old"] else np.nan)
           for k, v in pairs.items()]
    prs = sorted(prs, key=lambda r: np.nan_to_num(r[1]))[-8:]
    lab = lambda c: c.replace("cluster_", "")  # noqa: E731
    pcats = [f"type {lab(s)} → type {lab(t)}" for (s, t), _, _ in prs]
    po, pn, pl = [], [], []
    for c, (_, a, b) in zip(pcats, prs):
        tip = [[f"{a * 100:.2f} %", NEW], [f"{b * 100:.2f} %", OLD], ["Xenium lung (550k)", "dataset"]]
        pn.append(dict(x=float(a * 100), y=c, title=c, rows=tip))
        po.append(dict(x=float(b * 100), y=c, title=c, rows=[tip[1], tip[0], tip[2]]))
        pl.append(dict(x1=float(b * 100), x2=float(a * 100), y=c))
        rows.append([f"Xenium lung (550k): {c}", f"{a * 100:.2f} %", f"{b * 100:.2f} %", "", ""])
    bottom = dict(title="Xenium lung (550k): top source → target pairs", height=len(pcats) * 24 + 60,
                  marginLeft=200, x=dict(label="molecules of the source type in target cells, %", fmt="plain"),
                  y=dict(type="band", categories=pcats), links=pl,
                  series=[dict(name=OLD, color="base", marker="circle", open=True, points=po),
                          dict(name=NEW, color=0, marker="circle", points=pn)])
    write_chart(out, "celladmix", dict(
        columns=1, legend=[dict(name=NEW, color=0, marker="circle"),
                           dict(name=OLD, color="base", marker="circle", open=True)],
        panels=[top, bottom],
        table=dict(columns=["dataset / pair", f"rate {NEW}", f"rate {OLD}", "cells", "runs"], rows=rows)))
    NUMBERS["celladmix"] = {d: {"rc": float(a), "old": float(b)}
                            for d, a, b in zip(q.dataset, q.admix_rc, q.admix_old)}


# ---- segmentation mosaic and UMAPs (images) ----

SEG_EXAMPLES = [
    ("xenium_pancreas_377_quick", "Xenium · human pancreas", 100),
    ("merfish_ileum_quick", "MERFISH · mouse ileum", 100),
    ("cosmx_nsclc_lung5_rep1_quick", "CosMx · human lung cancer", 100),
    ("iss_mouse_hippocampus_quick", "ISS · mouse hippocampus", 150),
]
NCV_RUNS = DATA / "docs-figures/ncv"   # ncv_runs.py: the same runs with NCV colours on
SEG_THEMES = {
    "dark": dict(bg="#14161b", ink="#e8e6df", sub="#b5b3aa", line="#ffffff", line_alpha=0.75),
    "light": dict(bg="#fcfcfb", ink="#0b0b0b", sub="#52514e", line="#1b1b1b", line_alpha=0.7),
}
NOISE_MUTE = 0.78   # noise molecules: their NCV colour blended this far toward the background


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


def hex_rgb(colors: pd.Series) -> np.ndarray:
    h = colors.str.lstrip("#")
    return np.stack([h.str[i:i + 2].map(lambda v: int(v, 16)).to_numpy() for i in (0, 2, 4)], axis=1) / 255


def seg_window(ds: str, size: float) -> dict:
    """Molecules with NCV colours and the cell polygons of the densest size x size window."""
    import shapely
    base = NCV_RUNS / ds / "seg"
    if not (base / "molecules.parquet").is_file():
        raise SystemExit(f"{base}: missing; run docs_figures/ncv_runs.py first")
    src("segmentation_examples", base / "molecules.parquet", base / "cell_boundaries.parquet",
        NCV_RUNS / ds / "run.json")
    m = pd.read_parquet(base / "molecules.parquet")
    is3d = "z" in m.columns
    x0, y0 = densest_window(m.x.to_numpy(), m.y.to_numpy(), size)
    w = m[(m.x >= x0) & (m.x < x0 + size) & (m.y >= y0) & (m.y < y0 + size)]
    if is3d:  # 3-D data: show the most populated z-plane
        zs = w.z.value_counts()
        w = w[w.z == float(zs.index[zs.argmax()])]
    noise = w.is_noise | w.cell.isna()
    cells = w[~noise]
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
    return dict(x0=x0, y0=y0, w=w, cells=cells, noise=w[noise], is3d=is3d, rings=rings)


def fig_segmentation(out: Path):
    from PIL import Image
    wins = {ds: seg_window(ds, size) for ds, _, size in SEG_EXAMPLES}
    for mode, th in SEG_THEMES.items():
        plt.rcdefaults()
        plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
        bg, ink = th["bg"], th["ink"]
        bg_rgb = np.array(matplotlib.colors.to_rgb(bg))
        fig, axes = plt.subplots(2, 2, figsize=(7.6, 8.1), facecolor=bg)
        for ax, (ds, title, size) in zip(axes.ravel(), SEG_EXAMPLES):
            d = wins[ds]
            x0, y0, w = d["x0"], d["y0"], d["w"]
            ms = float(np.clip(2.3e4 / max(len(w), 1), 0.8, 9))
            nz = d["noise"]
            muted = hex_rgb(nz.ncv_color) * (1 - NOISE_MUTE) + bg_rgb * NOISE_MUTE
            ax.scatter(nz.x, nz.y, s=ms, c=muted, linewidths=0)
            ax.scatter(d["cells"].x, d["cells"].y, s=ms, c=hex_rgb(d["cells"].ncv_color), linewidths=0)
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
                "noise_fraction": float(len(nz) / max(len(w), 1)),
                "cells_with_molecules": int(d["cells"].cell.nunique()), "polygons": len(d["rings"])}
        fig.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.04, wspace=0.06, hspace=0.16)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=130, facecolor=bg)
        plt.close(fig)
        # palettised PNG: a fraction of the size, no visible loss at this marker size
        Image.open(buf).convert("RGB").quantize(colors=256, method=Image.Quantize.MEDIANCUT,
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




# ---- profiling ----

LADDERS = {"lung": "Xenium lung slide, 377 genes", "prime5k": "Xenium Prime 5K slide, 5,078 genes"}
WTX = "cosmx_wtx_colon_full"


def scaling_jobs(report: Path) -> pd.DataFrame:
    p = report / "summary-scaling-run/scaling_jobs.csv"
    src("scaling", p)
    j = pd.read_csv(p)
    j = j[(j.status == "ok") & (j.tool == "gperf")].copy()
    j["slide"] = j.dataset.str.extract(r"^(lung|prime5k)_")[0]
    return j


def rung_name(ds: str, molecules: float) -> str:
    if ds == WTX:
        return "CosMx WTx colon slide"
    slide = {"lung": "Xenium lung slide", "prime5k": "Xenium Prime 5K slide"}[ds.split("_")[0]]
    return f"{slide} (whole)" if ds.endswith("_all") else f"{slide}, {size_tag(molecules)} subset"


def load1(v) -> str:
    try:
        return f"{json.loads(v)[0]:.1f}"
    except Exception:  # noqa: BLE001
        return str(v)


def chart_scaling(out: Path):
    """Real-size ladders: CPU time, wall time and peak RSS vs molecules."""
    after, before = scaling_jobs(PROF), scaling_jobs(PROF_BEFORE)
    ver = {"after": f"{NEW} (profiling build 20bc45c)", "before": "before optimisation (e45fddc)"}

    def pts(df, col, f, which, threads):
        out_ = []
        for _, r in df.sort_values("molecules_loaded").iterrows():
            v = float(r[col]) * f
            vtxt = fmt_mem(v * 1024) if col == "peak_rss_kb" else fmt_s(v)
            out_.append(dict(x=int(r.molecules_loaded), y=v, title=rung_name(r.dataset, r.molecules_loaded),
                             rows=[[vtxt, {"cpu_s": "CPU time", "wall_s": "wall time",
                                           "peak_rss_kb": "peak RSS"}[col]],
                                   [f"{int(r.molecules_loaded):,}", "molecules"],
                                   [f"{int(r.genes_loaded):,}", "genes"],
                                   [str(threads), "threads"], [ver[which], "code"],
                                   [load1(r.loadavg_start), "1-min load at start"]]))
        return out_

    panels = []
    for col, f, ylab, title in (("cpu_s", 1, "CPU time, s", "CPU time (8 threads)"),
                                ("wall_s", 1, "wall time, s", "Wall time, 8 and 1 thread"),
                                ("peak_rss_kb", 2**-20, "peak memory (RSS), GiB", "Peak memory (8 threads)")):
        series = []
        for i, sl in enumerate(LADDERS):
            a8 = after[(after.slide == sl) & (after.threads == 8)]
            series.append(dict(name=f"{LADDERS[sl]}, 8 threads", color=i, marker="circle", line="solid",
                               points=pts(a8, col, f, "after", 8)))
            if col == "wall_s":
                a1 = after[(after.slide == sl) & (after.threads == 1)]
                series.append(dict(name=f"{LADDERS[sl]}, 1 thread", color=i, marker="square", line="dotted",
                                   open=True, size=3.5, points=pts(a1, col, f, "after", 1)))
            else:
                b8 = before[(before.slide == sl) & (before.threads == 8)]
                series.append(dict(name=f"{LADDERS[sl]}, before optimisation", color=i, marker="circle",
                                   line="dashed", open=True, size=3.5, points=pts(b8, col, f, "before", 8)))
        series.append(dict(name="CosMx WTx, 18,935 genes", color=2, marker="diamond", size=4.5,
                           points=pts(after[after.dataset == WTX], col, f, "after", 8)))
        if col != "wall_s":
            series.append(dict(name="CosMx WTx, before optimisation", color=2, marker="diamond", open=True,
                               size=4, points=pts(before[before.dataset == WTX], col, f, "before", 8)))
        refs = []
        if col == "cpu_s":
            refs.append(dict(type="line", points=[[1e5, 80], [1.1e7, 80 * 110]], dash="dotted", label="linear",
                             labelAt=[3e6, 1100]))
        panels.append(dict(title=title, height=290, x=dict(label="molecules", log=True),
                           y=dict(label=ylab, log=True, fmt="plain"), series=series, refs=refs))
    rows = []
    for which, df in (("after", after), ("before", before)):
        for _, r in df.sort_values(["slide", "threads", "molecules_loaded"]).iterrows():
            rows.append([rung_name(r.dataset, r.molecules_loaded), ver[which], int(r.threads),
                         f"{int(r.molecules_loaded):,}", f"{int(r.genes_loaded):,}", fmt_s(r.cpu_s),
                         fmt_s(r.wall_s), fmt_mem(r.peak_rss_kb / 1024), load1(r.loadavg_start)])
    write_chart(out, "scaling", dict(
        columns=2,
        legend=[dict(name=LADDERS[sl], color=i, line="solid", marker="circle") for i, sl in enumerate(LADDERS)] +
               [dict(name="CosMx WTx colon slide, 18,935 genes", color=2, marker="diamond"),
                dict(name="before optimisation (open, dashed)", color="ink2", line="dashed", marker="circle",
                     open=True),
                dict(name="1 thread (open, dotted)", color="ink2", line="dotted", marker="square", open=True)],
        panels=panels,
        table=dict(columns=["rung", "code", "threads", "molecules", "genes", "CPU time", "wall time", "peak RSS",
                            "load"], rows=rows)))


# ---- merged per-dataset table of the Profiling page ----

TABLE_DATASETS = [
    "xenium_pancreas_377_quick", "xenium_lung_cancer_admix", "xenium_breast_rep1_dense_full",
    "xenium_pancreas_377_full", "xenium_prime5k_ovarian_quick", "xenium_prime5k_ovarian_full",
    "cosmx_nsclc_lung5_rep1_full", "cosmx_wtx_colon_quick", "merfish_ileum_full",
    "iss_mouse_hippocampus_quick", "osmfish_somatosensory_quick", "starmap_visual_cortex_quick",
]
TABLE_RUNGS = ["lung_1M", "lung_2M", "lung_all", "prime5k_1M", "prime5k_8M", WTX]
TABLE_BEGIN = "<!-- docs_figures:runtime-table begin (generated by make_figures.py; do not edit) -->"
TABLE_END = "<!-- docs_figures:runtime-table end -->"


def runtime_table_md() -> str:
    """Benchmark crops (6 threads) and profiling ladders (8 threads), datasets in rows."""
    df = runtime_table()
    six = df[df.threads == 6].set_index("dataset")
    one = df[df.threads == 1].set_index("dataset")
    a, b = scaling_jobs(PROF), scaling_jobs(PROF_BEFORE)
    src("runtime_table", RC / "tables/runtime.csv", DATA / "real", RUNS / "rc1636-real",
        PROF / "summary-scaling-run/scaling_jobs.csv", PROF_BEFORE / "summary-scaling-run/scaling_jobs.csv")
    rows = []
    for ds in TABLE_DATASETS:
        r = six.loc[ds]
        o = one.loc[ds] if ds in one.index else None
        rows.append((int(r.n_molecules), [name(ds), f"{int(r.n_molecules):,}", f"{int(r.n_genes):,}", "6",
                                          fmt_s(r.wall_rc), fmt_s(r.cpu_rc), fmt_mem(r.rss_rc_mib),
                                          f"{r.rss_rc_mib * 2**20 / r.n_molecules:,.0f} B",
                                          fmt_s(None if o is None else o.wall_rc), "", ""]))
    for ds in TABLE_RUNGS:
        r = a[(a.dataset == ds) & (a.threads == 8)].iloc[0]
        r1 = a[(a.dataset == ds) & (a.threads == 1)]
        rb = b[(b.dataset == ds) & (b.threads == 8)]
        rows.append((int(r.molecules_loaded), [
            rung_name(ds, r.molecules_loaded), f"{int(r.molecules_loaded):,}", f"{int(r.genes_loaded):,}", "8",
            fmt_s(r.wall_s), fmt_s(r.cpu_s), fmt_mem(r.peak_rss_kb / 1024),
            f"{r.peak_rss_kb * 1024 / r.molecules_loaded:,.0f} B",
            fmt_s(float(r1.wall_s.iloc[0]) if len(r1) else None),
            fmt_s(float(rb.cpu_s.iloc[0])) if len(rb) else "",
            fmt_mem(float(rb.peak_rss_kb.iloc[0]) / 1024) if len(rb) else ""]))
    head = ["dataset", "molecules", "genes", "threads", "wall time", "CPU time", "peak RSS",
            "RSS per molecule", "wall time, 1 thread", "CPU time before", "peak RSS before"]
    lines = ["| " + " | ".join(head) + " |", "|---|" + "---:|" * (len(head) - 1)]
    for _, r in sorted(rows, key=lambda x: x[0]):
        lines.append("| " + " | ".join(r) + " |")
    return "\n".join(lines)


def update_page_table(docs: Path):
    page = docs / "profiling.md"
    if not page.is_file():
        print("skip table: no", page)
        return
    text = page.read_text()
    if TABLE_BEGIN not in text or TABLE_END not in text:
        raise SystemExit(f"{page}: table markers not found")
    pre, rest = text.split(TABLE_BEGIN, 1)
    _, post = rest.split(TABLE_END, 1)
    page.write_text(pre + TABLE_BEGIN + "\n\n" + runtime_table_md() + "\n\n" + TABLE_END + post)


# ---- owner-only report: where the time goes, current bottlenecks ----

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
    for name_, s in PHASE_GROUPS:
        if s and phase in s:
            return name_
    return "everything else"


def phase_tables():
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
    order = ["lung_100k", "lung_1M", "lung_4M", "lung_all", "prime5k_100k", "prime5k_1M", "prime5k_8M", WTX]
    real = real.loc[[o for o in order if o in real.index]][[g for g, _ in PHASE_GROUPS]]
    real_share = real.div(real.sum(axis=1), axis=0) * 100
    mol = sp_.groupby("dataset").molecules.first()
    NUMBERS["phases_real_share"] = real_share.round(1).to_dict(orient="index")
    NUMBERS["phases_crop_GIr"] = crop.round(2).to_dict(orient="index")
    return crop, real_share, mol


@both_modes
def fig_phases(out, mode, t):
    crop, real_share, mol = phase_tables()

    def rung_label(d):
        nm = {"lung_all": "lung, whole slide", WTX: "CosMx WTx slide"}.get(d, d.split("_")[0])
        n = mol[d]
        return f"{nm} ({n / 1e6:.1f}M)" if n >= 1e6 else f"{nm} ({n / 1e3:.0f}k)"

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
            ax.barh(y, v, left=left, color=c, height=0.68, edgecolor=t["surface"], linewidth=1.2, label=g)
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


OWNER_TEXT = """\
# Owner report: where the time goes, current bottlenecks

Generated by `docs_figures/make_figures.py` (not part of the public docs).
Profiling run: `{prof}`. The figure and the tables under "Data" are
regenerated from that run's summaries; the prose was written for the
2026-10-01 run (`20bc45c`) and quotes its numbers; re-check it against the
tables when the run changes.

## Where the time goes

![Stacked bars: instructions per phase on 20k-molecule crops and CPU share per phase at real sizes](owner/phases-light.svg)

**Figure.** Top: instructions per phase on 20k-molecule crops (callgrind,
1 thread). Bottom: share of the CPU time per phase at real sizes (gperftools,
8 threads). "BMM iterations" is the segmentation itself; "molecule
clustering" is the initial assignment of molecules to cell types (the MRF
clustering or, for whole-transcriptome panels, the neighbourhood-graph
clustering); "NCV colours" is the colour embedding used by the plots and the
`ncv_color` output column.

- On **small crops** the colour embedding dominates: 66–84 % of the
  instructions on every panel below 1,000 genes, mostly umappp's
  single-threaded layout optimisation. It costs a fixed amount (at most
  20,000 anchors), so it fades at real sizes: {ncv_lung_all:.0f} % of the CPU time on the whole
  lung slide. Skip it with `--skip-ncv-color` if you do not need the colours.
- On **real slides** (1M molecules and more) the BMM iterations take
  {bmm_min:.0f}–{bmm_max:.0f} % of the CPU time and molecule clustering {mc_min:.0f}–{mc_max:.0f} %; on the
  18,935-gene CosMx slide molecule clustering takes {mc_wtx:.0f} %.
- The **dense ICA** behind the 2,793-gene crop's huge clustering bar is the
  1,000–3,000-gene case of the docs' Profiling › Gene panel size.

## Current bottlenecks

Ranked by their share of the CPU time at real sizes (whole lung slide and
Prime 5K 8M, 8 threads).

1. **BMM E-step** — about 35 % of the CPU on the whole lung slide (the E-step
   chunk, `CategoricalSmoothed::pdf` and `exp`). Linear and 99 % parallel:
   this is the algorithm's core work, not overhead.
2. **Molecule clustering** — {mc_lung_all:.0f} % (lung) and {mc_p8:.0f} % (Prime 5K 8M) of the CPU,
   super-linear (exponent {exp_mc:.2f}), because the MRF clustering needs more
   iterations on larger slides.
3. **BMM bookkeeping** (splitting and grouping components, hash-map updates)
   — about 22 % of the CPU on the whole slide, mildly super-linear
   (exponents 1.2–1.4).
4. **NCV colour embedding** on small data — single-threaded, 61 % of the
   instructions on a 20k crop; negligible on slides. Use `--skip-ncv-color`
   when the colours are not needed.
5. **Gene-rich panels** — the neighbourhood k-NN with k = genes / 10 takes
   65 % of the CPU on the 18,935-gene CosMx slide, and panels of
   1,000–3,000 genes still run the dense O(genes³) ICA (87 % of the
   instructions on the 2,793-gene crop, single-threaded).

Memory at the whole-slide peak (5.06 GiB heap) is spread over the molecule
adjacency list (787 MiB), the assignment history (728 MiB), the MRF clustering
state (383 MiB) and three copies of the molecule positions (3 × 170 MiB).

## Data

Share of the CPU time per phase group, % (gperftools, 8 threads):

{real_table}

Instructions per phase group, 10⁹ (callgrind, 1 thread, 20k-molecule crops):

{crop_table}

Scaling exponents (value ∝ molecules^b, gperftools, 8 threads):

{exp_table}
"""


def md_table(df: pd.DataFrame, index_name: str, fmt="{:.1f}") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join([index_name] + cols) + " |", "|---|" + "---:|" * len(cols)]
    for i, r in df.iterrows():
        lines.append("| " + " | ".join([str(i)] + [fmt.format(v) for v in r]) + " |")
    return "\n".join(lines)


def owner_report(path: Path):
    """Write the owner-only report (outside the docs) with its figure."""
    owner = path.parent / "owner"
    owner.mkdir(parents=True, exist_ok=True)
    fig_phases(owner)
    crop, share, _ = phase_tables()
    fits = pd.read_csv(PROF / "summary-scaling-run/scaling_fits.csv")
    f = fits[fits.what.isin(["total", "phase_cpu"]) & (fits.tool == "gperf") & (fits.threads == 8)]
    ex = f.pivot_table(index="name", columns="slide", values="exponent", aggfunc="first")
    big = [d for d in share.index if d in ("lung_1M", "lung_4M", "lung_all", "prime5k_1M", "prime5k_8M")]
    path.write_text(OWNER_TEXT.format(
        prof=str(PROF).replace(str(DATA), "$BAYSOR_BENCH_DATA"),
        ncv_lung_all=share.loc["lung_all", "NCV colours"],
        bmm_min=share.loc[big, "BMM iterations"].min(), bmm_max=share.loc[big, "BMM iterations"].max(),
        mc_min=share.loc[big, "molecule clustering"].min(), mc_max=share.loc[big, "molecule clustering"].max(),
        mc_wtx=share.loc[WTX, "molecule clustering"], mc_lung_all=share.loc["lung_all", "molecule clustering"],
        mc_p8=share.loc["prime5k_8M", "molecule clustering"],
        exp_mc=float(ex.loc["mol_clustering", "lung"]) if "mol_clustering" in ex.index else float("nan"),
        real_table=md_table(share, "rung"), crop_table=md_table(crop, "crop", "{:.2f}"),
        exp_table=md_table(ex, "phase", "{:.2f}")))


# --------------------------------------------------------------- tables ----

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



FIGURES = [chart_runtime_vs_molecules, chart_threads, chart_genes, chart_scaling, chart_celladmix]
IMAGES = [fig_umap, fig_segmentation]


def main():
    ap = argparse.ArgumentParser(description="Regenerate the Baysor docs performance figures.")
    ap.add_argument("--docs", type=Path,
                    default=Path(os.environ.get("BAYSOR_DOCS_PERF",
                                                "/home/vpetukhov/Projects/Baysor/docs/performance")),
                    help="docs/performance directory of a Baysor checkout (charts go to <docs>/data, "
                         "images to <docs>/img, the per-dataset table into <docs>/profiling.md)")
    ap.add_argument("--only", nargs="*", help="figure function names to run (default: all)")
    args = ap.parse_args()
    img, data = args.docs / "img", args.docs / "data"
    img.mkdir(parents=True, exist_ok=True)
    GEN.mkdir(exist_ok=True)
    for fn in FIGURES + IMAGES:
        if args.only and fn.__name__ not in args.only:
            continue
        print("figure:", fn.__name__, flush=True)
        fn(data if fn in FIGURES else img)
    update_page_table(args.docs)
    owner_report(GEN / "owner_report.md")
    md = "# Numbers quoted in docs/performance (generated by make_figures.py)\n\n" + tables() + write_sources()
    (GEN / "tables.md").write_text(md)
    (GEN / "numbers.json").write_text(json.dumps(NUMBERS, indent=1, default=float) + "\n")
    total = sum(p.stat().st_size for p in args.docs.rglob("*") if p.is_file())
    print(f"wrote {img}, {data}, the table of {args.docs / 'profiling.md'} ; docs/performance total "
          f"{total / 1e6:.2f} MB ; tables {GEN / 'tables.md'} ; owner report {GEN / 'owner_report.md'}")


if __name__ == "__main__":
    main()
