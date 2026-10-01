#!/usr/bin/env python3
"""Self-contained HTML before/after profiling report.

    python profiling/report_html.py --before REPORT_DIR_A --after REPORT_DIR_B \
        [--notes notes.md] [--out REPORT_DIR_B/report.html] [--md REPORT_DIR_B/REPORT.md]

REPORT_DIR_A / REPORT_DIR_B are report directories
(`$BAYSOR_BENCH_DATA/profiling/reports/<date>-<sha>/`) that hold copies of
the summaries of one run of each tier:

    summary-report-run/    profile.py --suite report   (summary/ of the run)
    summary-quick-run/     profile.py --suite quick
    summary-scaling-run/   scaling.py summarize
    compare-*.csv          optional compare.py outputs (linked)

Everything is rendered into ONE html file without network access: inline
CSS, charts as inline SVG (matplotlib's svg backend, hover tooltips as SVG
<title> elements), a few lines of inline JS for sortable tables and tabs.
Links point to files next to the report (relative), to the "before" report
directory, and to github.com.

Narrative text comes from an optional notes file (Markdown subset:
headings, paragraphs, lists, tables, `code`, **bold**, links), split into
sections by lines `<!-- section: ID -->`; each ID is rendered at its place
in the report (ids: subtitle, summary, bottlenecks_top, where_time, area_ncv,
area_bmm, area_clustering, area_memory, area_threading, threads, scaling,
memory, bottlenecks, next_steps, method). Every number in the generated tables and
charts comes from the summary files.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import html
import io
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------

DATASETS = [  # (id, label) in gene-panel order, as REPORT.md of 2026-09-30
    ("osmfish_g35_20k", "osmFISH 28 g"),
    ("xenium_pancreas_g377_10k", "pancreas 10k"),
    ("sim_circles_g100_20k", "sim g100"),
    ("xenium_pancreas_g377_20k", "pancreas 20k ★"),
    ("xenium_pancreas_g377_40k", "pancreas 40k"),
    ("merfish_ileum_3d_20k", "MERFISH 3D"),
    ("xenium_prime5k_20k", "prime5k 20k"),
    ("cosmx_nsclc_g960_20k", "CosMx 922 g"),
    ("sim_circles_g1000_20k", "sim g1000"),
    ("sim_circles_g5000_10k", "sim g5000 (2,793 g)"),
    ("cosmx_wtx_20k", "WTx 8,407 g"),
]
DS_LABEL = dict(DATASETS)
REP = "xenium_pancreas_g377_20k"          # representative crop

# phase groups for stacked bars (fixed order = fixed colour slots)
PHASE_GROUPS = ["confidence", "mol_clustering", "bmm_iterations", "ncv_colors", "polygons",
                "report", "other"]
MAIN_PHASES = ["confidence", "molecule_graph", "mol_clustering", "bmm_init", "bmm_iterations",
               "ncv_colors", "polygons", "report"]
# reference categorical palette (dataviz skill, references/palette.md), light mode
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
GROUP_COLOR = dict(zip(PHASE_GROUPS, SLOTS))
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
BEFORE_GRAY = "#a3a29c"

GITHUB = "https://github.com/VPetukhov/baysor-benchmarks"
# thread entry points and pool/OpenMP machinery: inclusive shares say nothing
NOISE_FN = re.compile(r"^(__clone3?|start_thread|execute_native_thread_routine|std::thread::_State_impl|"
                      r"gomp_thread_start|GOMP_parallel|main|_start|__libc_start|cmd_run|"
                      r"baysor::(run_parallel_chunks|parallel_region|ParallelRegion::)|.*\.pool_region$|"
                      r"std::_Function_handler::_M_invoke$)")


# ---------------------------------------------------------------------------
# data loading
# ---------------------------------------------------------------------------

def num(v):
    if v is None or v == "":
        return None
    if isinstance(v, (int, float)):
        return v
    try:
        f = float(v)
    except ValueError:
        if v in ("True", "False"):
            return v == "True"
        return v
    return int(f) if f.is_integer() and "." not in v and "e" not in v.lower() else f


def read_csv(path: Path) -> list:
    if not path.is_file():
        return []
    with open(path, newline="") as fh:
        return [{k: num(v) for k, v in r.items()} for r in csv.DictReader(fh)]


def read_json(path: Path):
    return json.loads(path.read_text()) if path.is_file() else None


class Tier:
    """Summary files of one quick-tier run (report or quick suite)."""

    def __init__(self, d: Path):
        self.dir = d
        self.ok = (d / "summary.json").is_file()
        self.summary = read_json(d / "summary.json") or {"jobs": {}, "suite": {}, "load_samples": []}
        self.phases = read_csv(d / "phases.csv")
        self.functions = read_csv(d / "functions.csv")
        self.lines = read_csv(d / "lines.csv")
        self.regions = read_csv(d / "omp_regions.csv")
        self.native = read_csv(d / "native.csv")
        self.native_phases = read_csv(d / "native_phases.csv")
        self.dhat_sites = read_csv(d / "dhat_sites.csv")
        self.cache = read_csv(d / "cache.csv")
        self._cg = {}

    def jobs(self):
        return self.summary.get("jobs", {})

    def cg(self, job: str):
        if job not in self._cg:
            self._cg[job] = read_json(self.dir / "callgrind" / f"{job}.json")
        return self._cg[job]

    def dhat(self, job: str):
        return read_json(self.dir / "dhat" / f"{job}.json")

    def total_ir(self, job):
        j = self.jobs().get(job)
        return j.get("total_Ir") if j else None

    def phase_ir(self, job) -> dict:
        return {r["phase"]: r for r in self.phases if r["job"] == job}

    def native_row(self, ds, threads, policy="default"):
        for r in self.native:
            if r["dataset"] == ds and r["threads"] == threads and (r.get("wait_policy") or "default") == policy:
                return r
        return None

    def native_phase(self, ds, threads, phase, policy="default"):
        for r in self.native_phases:
            if (r["dataset"] == ds and r["threads"] == threads and r["phase"] == phase
                    and (r.get("wait_policy") or "default") == policy):
                return r
        return None


class Scaling:
    def __init__(self, d: Path):
        self.dir = d
        self.ok = (d / "scaling_jobs.csv").is_file()
        self.jobs = [r for r in read_csv(d / "scaling_jobs.csv")]
        self.phases = read_csv(d / "scaling_phases.csv")
        self.fits = read_csv(d / "scaling_fits.csv")
        self.parallel = read_csv(d / "scaling_parallel.csv")
        self.heaptrack = read_csv(d / "scaling_heaptrack.csv")
        self.summary = read_json(d / "scaling_summary.json") or {}
        self.partial = {p.name: read_json(p) for p in d.glob("heaptrack-*.partial.json")}

    def job(self, tool, ds, thr):
        for r in self.jobs:
            if r["tool"] == tool and r["dataset"] == ds and r["threads"] == thr:
                return r
        return None

    def ok_job(self, tool, ds, thr):
        r = self.job(tool, ds, thr)
        return r if r and r.get("ok") is True else None

    def phase(self, ds, thr, phase, tool="gperf"):
        for r in self.phases:
            if r["tool"] == tool and r["dataset"] == ds and r["threads"] == thr and r["phase"] == phase:
                return r
        return None

    def fit(self, slide, thr, what, name):
        for r in self.fits:
            if (r["slide"] == slide and r["threads"] == thr and r["what"] == what and r["name"] == name
                    and r["tool"] == "gperf"):
                return r
        return None


def group_of(phase: str) -> str:
    return phase if phase in PHASE_GROUPS[:-1] else "other"


# ---------------------------------------------------------------------------
# formatting helpers
# ---------------------------------------------------------------------------

def esc(s) -> str:
    return html.escape("" if s is None else str(s))


def fmt(v, nd=1, unit=""):
    if v is None or v == "" or isinstance(v, bool):
        return "—"
    if isinstance(v, str):
        return esc(v)
    if abs(v) >= 1e4 and nd <= 1:
        return f"{v:,.0f}{unit}"
    return f"{v:,.{nd}f}{unit}"


def G(v):
    return "—" if v is None else f"{v / 1e9:,.2f}"


def delta(a, b, lower_is_better=True, nd=1):
    """'−37.2 %' with a class for good/bad."""
    if a in (None, 0) or b is None:
        return '<span class="d">—</span>'
    p = 100.0 * (b - a) / a
    good = (p < 0) == lower_is_better
    cls = "good" if good and abs(p) >= 1 else ("bad" if not good and abs(p) >= 1 else "flat")
    sign = "−" if p < 0 else "+"
    return f'<span class="d {cls}">{sign}{abs(p):.{nd}f} %</span>'


def ratio(a, b):
    if not a or not b:
        return "—"
    r = a / b
    return f"{r:.2f}×" if r < 10 else f"{r:.1f}×"


def code(s):
    return f"<code>{esc(s)}</code>"


def table(headers, rows, cls="sortable", caption=None, num_cols=None, raw=False):
    """rows: lists of cells (already HTML if raw, else escaped)."""
    num_cols = set(num_cols or [])
    out = [f'<div class="tw"><table class="{cls}">']
    if caption:
        out.append(f"<caption>{caption}</caption>")
    out.append("<thead><tr>" + "".join(
        f'<th{" class=n" if i in num_cols else ""}>{h}</th>' for i, h in enumerate(headers)) + "</tr></thead><tbody>")
    for r in rows:
        cells = []
        for i, c in enumerate(r):
            v = c if raw else esc(c)
            cells.append(f'<td{" class=n" if i in num_cols else ""}>{v}</td>')
        out.append("<tr>" + "".join(cells) + "</tr>")
    out.append("</tbody></table></div>")
    return "\n".join(out)


def stat_tile(label, before, after, unit="", lower_is_better=True, nd=1, note=""):
    d = delta(before, after, lower_is_better)
    b = fmt(before, nd)
    a = fmt(after, nd)
    return (f'<div class="tile"><div class="tl">{label}</div>'
            f'<div class="tv">{a}<span class="tu">{unit}</span></div>'
            f'<div class="tb">before {b}{unit} {d}</div>'
            + (f'<div class="tn">{note}</div>' if note else "") + "</div>")


# ---------------------------------------------------------------------------
# markdown subset (notes)
# ---------------------------------------------------------------------------

def md_inline(s: str) -> str:
    s = esc(s)
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    s = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", s)
    s = re.sub(r"(?<![\w*])\*([^*\s][^*]*)\*(?![\w*])", r"<em>\1</em>", s)
    s = re.sub(r"\[([^\]]+)\]\(([^)\s]+)\)",
               lambda m: f'<a href="{m.group(2)}">{m.group(1)}</a>', s)
    return s


def md(text: str) -> str:
    out, para, lst, tbl = [], [], None, []

    def flush():
        nonlocal para, lst, tbl
        if para:
            out.append("<p>" + md_inline(" ".join(para)) + "</p>")
            para = []
        if lst:
            tag, items = lst
            out.append(f"<{tag}>" + "".join(f"<li>{md_inline(i)}</li>" for i in items) + f"</{tag}>")
            lst = None
        if tbl:
            rows = [[c.strip() for c in r.strip().strip("|").split("|")] for r in tbl]
            if len(rows) >= 2 and set(rows[1][0].replace(":", "")) <= {"-"}:
                head, body = rows[0], rows[2:]
            else:
                head, body = rows[0], rows[1:]
            out.append(table([md_inline(h) for h in head], [[md_inline(c) for c in r] for r in body],
                             cls="sortable", raw=True))
            tbl = []

    for line in text.splitlines():
        s = line.rstrip()
        if not s.strip():
            flush()
            continue
        if s.lstrip().startswith("|"):
            if para or lst:
                flush()
            tbl.append(s)
            continue
        if tbl:
            flush()
        m = re.match(r"^(#{2,5})\s+(.*)$", s)
        if m:
            flush()
            lvl = len(m.group(1)) + 1
            out.append(f"<h{lvl}>{md_inline(m.group(2))}</h{lvl}>")
            continue
        m = re.match(r"^\s*([*-]|\d+\.)\s+(.*)$", s)
        if m:
            if para:
                flush()
            tag = "ol" if m.group(1)[0].isdigit() else "ul"
            if lst and lst[0] != tag:
                flush()
            if not lst:
                lst = (tag, [])
            lst[1].append(m.group(2))
            continue
        if lst and line.startswith("  "):
            lst[1][-1] += " " + s.strip()
            continue
        para.append(s.strip())
    flush()
    return "\n".join(out)


def load_notes(path) -> dict:
    if not path or not Path(path).is_file():
        return {}
    sec, cur = {}, None
    for line in Path(path).read_text().splitlines():
        m = re.match(r"^<!--\s*section:\s*([\w-]+)\s*-->\s*$", line.strip())
        if m:
            cur = m.group(1)
            sec[cur] = []
            continue
        if cur:
            sec[cur].append(line)
    return {k: md("\n".join(v)) for k, v in sec.items()} | {f"raw:{k}": "\n".join(v).strip() for k, v in sec.items()}


# ---------------------------------------------------------------------------
# charts (matplotlib svg, inlined; hover text as <title>)
# ---------------------------------------------------------------------------

_plt = None


def plt():
    global _plt
    if _plt is None:
        import matplotlib
        matplotlib.use("svg")
        import matplotlib.pyplot as p
        p.rcParams.update({
            "svg.fonttype": "none", "font.family": "sans-serif",
            "font.sans-serif": ["Inter", "Helvetica Neue", "Arial", "DejaVu Sans"],
            "font.size": 9.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
            "xtick.color": INK2, "ytick.color": INK2, "axes.titlesize": 10.5,
            "axes.titleweight": "semibold", "axes.titlecolor": INK, "axes.spines.top": False,
            "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
            "axes.axisbelow": True, "legend.frameon": False, "legend.fontsize": 8.5,
            "svg.hashsalt": "baysor-profiling",
        })
        _plt = p
    return _plt


class Tips:
    """Collects hover texts; artists get a gid, the svg gets <title>s."""

    def __init__(self):
        self.items = {}

    def __call__(self, artist, text):
        gid = f"tt{id(self)}_{len(self.items)}"
        artist.set_gid(gid)
        self.items[gid] = text
        return artist


_SVG_N = [0]


def svg_of(fig, tips: Tips | None = None, alt="") -> str:
    buf = io.StringIO()
    fig.savefig(buf, format="svg", bbox_inches="tight", metadata={"Date": None})
    plt().close(fig)
    s = buf.getvalue()
    s = s[s.index("<svg"):]
    _SVG_N[0] += 1
    # unique ids per chart (several svgs share one document)
    pref = f"c{_SVG_N[0]}_"
    s = re.sub(r'id="([^"]+)"', lambda m: f'id="{pref}{m.group(1)}"', s)
    s = re.sub(r'(url\(#|xlink:href="#|href="#)([^")]+)', lambda m: f"{m.group(1)}{pref}{m.group(2)}", s)
    if tips:
        for gid, text in tips.items.items():
            s = s.replace(f'<g id="{pref}{gid}">', f'<g id="{pref}{gid}" class="hov"><title>{esc(text)}</title>', 1)
    s = re.sub(r'<svg ([^>]*?)width="([\d.]+)pt" height="([\d.]+)pt"',
               lambda m: f'<svg {m.group(1)}role="img" aria-label="{esc(alt)}" '
                         f'style="width:100%;max-width:{float(m.group(2)) * 1.333:.0f}px;height:auto"', s, 1)
    return f'<figure class="chart">{s}</figure>'


def chart_phase_stack(before: Tier, after: Tier, title: str) -> str:
    """Horizontal stacked bars of G Ir per phase group, before/after per dataset."""
    p = plt()
    rows = []
    for ds, lab in DATASETS:
        job = f"callgrind-{ds}-t1"
        pa, pb = before.phase_ir(job), after.phase_ir(job)
        if not pa and not pb:
            continue
        rows.append((lab, "after", pb))
        rows.append(("", "before", pa))
    n = len(rows)
    fig, ax = p.subplots(figsize=(9.2, 0.27 * n + 1.2))
    tips = Tips()
    ylabels = []
    for i, (lab, which, ph) in enumerate(rows):
        y = n - 1 - i
        left = 0.0
        tot = sum(r["Ir"] for r in ph.values()) or 1
        agg = defaultdict(float)
        for name, r in ph.items():
            agg[group_of(name)] += r["Ir"]
        for g in PHASE_GROUPS:
            v = agg.get(g, 0) / 1e9
            if v <= 0:
                continue
            b = ax.barh(y, v, left=left, height=0.78 if which == "after" else 0.62,
                        color=GROUP_COLOR[g], alpha=1.0 if which == "after" else 0.45,
                        edgecolor="white", linewidth=1.2)
            tips(b.patches[0], f"{lab or rows[i - 1][0]} ({which}): {g} {v:.2f} G Ir = "
                               f"{100 * agg[g] / tot:.1f} % of the run")
            left += v
        ax.text(left + 0.6, y, f"{left:.1f} G", va="center", fontsize=8, color=INK2)
        ylabels.append((y, f"{lab}  {which}" if lab else f"{which}"))
    ax.set_yticks([y for y, _ in ylabels], [l for _, l in ylabels], fontsize=8.5)
    ax.set_xlabel("instructions, G Ir (callgrind, 1 thread)")
    ax.grid(axis="y", visible=False)
    ax.set_title(title, loc="left")
    handles = [p.Rectangle((0, 0), 1, 1, color=GROUP_COLOR[g]) for g in PHASE_GROUPS]
    ax.legend(handles, PHASE_GROUPS, ncol=1, loc="lower left", bbox_to_anchor=(1.01, 0.0), fontsize=8.5,
              handlelength=1.0, title="phase", title_fontsize=8.5)
    return svg_of(fig, tips, title)


def chart_lines(series, title, xlabel, ylabel, logx=False, logy=False, ref_slope=None, xticks=None,
                ideal=None, figsize=(4.6, 3.3), legend_loc="best", ylim=None):
    """series: list of dict(label, xs, ys, color, before: bool, fmt)."""
    p = plt()
    fig, ax = p.subplots(figsize=figsize)
    tips = Tips()
    for s in series:
        pts = [(x, y) for x, y in zip(s["xs"], s["ys"]) if x is not None and y is not None]
        if not pts:
            continue
        xs, ys = zip(*pts)
        bef = s.get("before", False)
        ln, = ax.plot(xs, ys, color=s["color"], lw=1.6 if bef else 2.0, ls="--" if bef else "-",
                      marker="o", ms=5.5 if bef else 6.5, mfc="white" if bef else s["color"],
                      mec=s["color"], mew=1.5, label=s["label"], alpha=0.85 if bef else 1.0)
        for x, y in pts:
            pt, = ax.plot([x], [y], ls="none", marker="o", ms=11, alpha=0.0)
            tips(pt, f"{s['label']}: x={s.get('xfmt', lambda v: f'{v:,.0f}')(x)}, "
                     f"{ylabel}={s.get('yfmt', lambda v: f'{v:,.3g}')(y)}")
    if ideal:
        xs, ys = ideal
        ax.plot(xs, ys, color=MUTED, lw=1, ls=":", label="ideal")
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    if xticks:
        ax.set_xticks(xticks, [str(x) for x in xticks])
        ax.minorticks_off()
    if ylim:
        ax.set_ylim(*ylim)
    ax.set_title(title, loc="left")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc=legend_loc, fontsize=7.5)
    return svg_of(fig, tips, title)


def chart_grouped_bars(cats, groups, title, ylabel, figsize=(9, 3.2), fmt_v=lambda v: f"{v:.3g}"):
    """cats: labels; groups: list of (label, values, color, alpha)."""
    p = plt()
    fig, ax = p.subplots(figsize=figsize)
    tips = Tips()
    k = len(groups)
    w = 0.8 / k
    for gi, (lab, vals, color, alpha) in enumerate(groups):
        for ci, v in enumerate(vals):
            if v is None:
                continue
            b = ax.bar(ci - 0.4 + w * (gi + 0.5), v, width=w * 0.92, color=color, alpha=alpha,
                       edgecolor="white", linewidth=1)
            tips(b.patches[0], f"{cats[ci]} — {lab}: {fmt_v(v)} {ylabel}")
    ax.set_xticks(range(len(cats)), cats, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel(ylabel)
    ax.grid(axis="x", visible=False)
    ax.set_title(title, loc="left")
    handles = [p.Rectangle((0, 0), 1, 1, color=c, alpha=a) for _, _, c, a in groups]
    ax.legend(handles, [g[0] for g in groups], fontsize=8)
    return svg_of(fig, tips, title)


# ---------------------------------------------------------------------------
# sections
# ---------------------------------------------------------------------------

class Report:
    def __init__(self, bdir: Path, adir: Path, notes: dict, out: Path, extra=None):
        self.bdir, self.adir, self.notes, self.out = bdir, adir, notes, out
        # optional intermediate commit (label, report dir), shown in the NCV area
        self.extra = (extra[0], Tier(extra[1] / "summary-report-run"), Scaling(extra[1] / "summary-scaling-run")) \
            if extra else None
        self.B = Tier(bdir / "summary-report-run")
        self.A = Tier(adir / "summary-report-run")
        self.Bq = Tier(bdir / "summary-quick-run")
        self.Aq = Tier(adir / "summary-quick-run")
        self.BS = Scaling(bdir / "summary-scaling-run")
        self.AS = Scaling(adir / "summary-scaling-run")
        if not self.BS.parallel:
            # the earlier run predates scaling_parallel.csv: a re-summary of it
            # (scaling.py summarize <before run> --out ...) may be kept next to the new report
            self.BS.parallel = read_csv(adir / "before-scaling_parallel.csv")
        self.toc = []

    # -- helpers -----------------------------------------------------------
    def rel(self, p: Path) -> str:
        try:
            return str(p.relative_to(self.out.parent))
        except ValueError:
            import os
            return os.path.relpath(p, self.out.parent)

    def note(self, key):
        return f'<div class="note">{self.notes[key]}</div>' if key in self.notes else ""

    # -- REPORT.md -----------------------------------------------------------
    def markdown(self) -> str:
        A, B, AS, BS = self.A, self.B, self.AS, self.BS
        ma, mb = self.meta(A), self.meta(B)

        def pc(a, b):
            return "—" if not a or b is None else f"{100 * (b - a) / a:+.1f} %"

        def f(v, nd=1):
            return "—" if v is None else f"{v:,.{nd}f}"
        o = [f"# Baysor CPU and memory profile — {(ma.get('started') or '')[:10]}, commit "
             f"{(ma.get('git_sha') or '')[:7]} vs {(mb.get('git_sha') or '')[:7]}", "",
             "Headline numbers of the HTML report [`report.html`](report.html) (same directory; it has every "
             "table, chart and the method). Summaries behind every number: `summary-report-run/`, "
             "`summary-quick-run/`, `summary-scaling-run/` next to this file; before/after deltas: "
             "`compare-*.txt`/`.csv`. Generated by `profiling/report_html.py --md`.", ""]
        if "raw:summary" in self.notes:
            o += [self.notes["raw:summary"], ""]
        o += ["## Representative crop `xenium_pancreas_g377_20k` (19,297 molecules, 141 genes)", "",
              "| measure | before | after | Δ |", "|---|---|---|---|"]
        job = f"callgrind-{REP}-t1"
        ta, tb = B.total_ir(job), A.total_ir(job)
        o.append(f"| instructions, 1 thread (G Ir, callgrind) | {f(ta and ta / 1e9, 2)} | {f(tb and tb / 1e9, 2)} | {pc(ta, tb)} |")
        for t in (1, 8, 16):
            x, y = B.native_row(REP, t), A.native_row(REP, t)
            o.append(f"| wall, {t} thr (s, min of 3; load-sensitive) | {f(x and x['wall_s_min'], 2)} | "
                     f"{f(y and y['wall_s_min'], 2)} | {pc(x and x['wall_s_min'], y and y['wall_s_min'])} |")
        for t in (1, 8, 16):
            x, y = B.native_row(REP, t), A.native_row(REP, t)
            o.append(f"| CPU time, {t} thr (s, median) | {f(x and x['cpu_s_median'], 1)} | "
                     f"{f(y and y['cpu_s_median'], 1)} | {pc(x and x['cpu_s_median'], y and y['cpu_s_median'])} |")
        x, y = B.native_row(REP, 1), A.native_row(REP, 1)
        o.append(f"| peak RSS, 1 thr (MiB) | {f(x and x['peak_rss_kb'] / 1024, 0)} | {f(y and y['peak_rss_kb'] / 1024, 0)} | "
                 f"{pc(x and x['peak_rss_kb'], y and y['peak_rss_kb'])} |")
        da, db = B.jobs().get(f"dhat-{REP}-t1"), A.jobs().get(f"dhat-{REP}-t1")
        if da and db:
            o.append(f"| peak heap (MiB, DHAT) | {f(da['peak_bytes'] / 2**20, 1)} | {f(db['peak_bytes'] / 2**20, 1)} | "
                     f"{pc(da['peak_bytes'], db['peak_bytes'])} |")
            o.append(f"| heap allocations (M, DHAT) | {f(da['total_blocks'] / 1e6, 2)} | {f(db['total_blocks'] / 1e6, 2)} | "
                     f"{pc(da['total_blocks'], db['total_blocks'])} |")
        ca, cb = B.cg(job), A.cg(job)
        if ca and cb:
            o.append(f"| serial share / Amdahl bound at 16 thr (callgrind) | {100 * ca['serial_frac']:.1f} % / "
                     f"{ca['amdahl']['16']:.2f}× | {100 * cb['serial_frac']:.1f} % / {cb['amdahl']['16']:.2f}× | |")
        o += ["", "## Instructions per crop (1 thread, G Ir)", "",
              "| dataset | before | after | Δ | mol_clustering after (Δ) | bmm_iterations after (Δ) | ncv_colors after (Δ) |",
              "|---|---|---|---|---|---|---|"]
        for ds, lab in DATASETS:
            j = f"callgrind-{ds}-t1"
            pa, pb = B.phase_ir(j), A.phase_ir(j)
            if not pb:
                continue
            cells = []
            for ph in ("mol_clustering", "bmm_iterations", "ncv_colors"):
                a, b = pa.get(ph, {}).get("Ir"), pb.get(ph, {}).get("Ir")
                cells.append(f"{f(b and b / 1e9, 2)} ({pc(a, b)})")
            o.append(f"| `{ds}` | {f(B.total_ir(j) and B.total_ir(j) / 1e9, 2)} | {f(A.total_ir(j) / 1e9, 2)} | "
                     f"{pc(B.total_ir(j), A.total_ir(j))} | " + " | ".join(cells) + " |")
        o += ["", "## Real sizes (scaling tier, 8 threads)", "",
              "| rung | molecules | CPU s before | after | Δ | wall s before | after (load-sens.) | peak RSS GiB before | after | Δ |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for ds in ("lung_100k", "lung_1M", "lung_2M", "lung_4M", "lung_8M", "lung_all", "prime5k_1M", "prime5k_2M",
                   "prime5k_8M", "cosmx_wtx_colon_full"):
            x, y = BS.ok_job("gperf", ds, 8), AS.ok_job("gperf", ds, 8)
            if not x and not y:
                continue
            m = (y or x).get("molecules_loaded") or (y or x)["molecules"]
            o.append(f"| `{ds}` | {m:,} | {f(x and x['cpu_s'], 0)} | {f(y and y['cpu_s'], 0)} | {pc(x and x['cpu_s'], y and y['cpu_s'])} | "
                     f"{f(x and x['wall_s'], 0)} | {f(y and y['wall_s'], 0)} | {f(x and x['peak_rss_kb'] / 2**20, 2)} | "
                     f"{f(y and y['peak_rss_kb'] / 2**20, 2)} | {pc(x and x['peak_rss_kb'], y and y['peak_rss_kb'])} |")
        for key, title in (("bottlenecks", "Remaining bottlenecks, ranked"), ("next_steps", "Next steps"),
                           ("method", "Method notes")):
            if f"raw:{key}" in self.notes:
                o += ["", f"## {title}", "", self.notes[f"raw:{key}"]]
        return "\n".join(o) + "\n"

    def section(self, sid, title, body):
        self.toc.append((sid, title))
        return f'<section id="{sid}"><h2>{esc(title)}</h2>\n{body}\n</section>'

    def meta(self, tier: Tier):
        return tier.summary.get("suite", {})

    # -- header ------------------------------------------------------------
    def header(self):
        ma, mb = self.meta(self.A), self.meta(self.B)
        sa = (self.A.summary.get("suite") or {})
        sha_a = (ma.get("git_sha") or "")[:7]
        sha_b = (mb.get("git_sha") or "")[:7]
        return f"""
<header>
<div class="kicker">Baysor · CPU &amp; memory profile · before / after</div>
<h1>Where Baysor spends CPU and memory after the optimizations</h1>
<p class="sub"><code>perf-optimization</code> @ <strong>{esc(sha_a)}</strong>
({md_inline(self.notes.get("raw:subtitle") or "own thread pool + NCV, BMM, clustering and memory optimizations")}) vs. the
pre-optimization profile @ <strong>{esc(sha_b)}</strong>
({esc((mb.get('started') or '')[:10])}). Host {esc(ma.get('cpu') or mb.get('cpu'))},
{esc(ma.get('nproc'))} logical CPUs, shared. Generated {dt.datetime.utcnow():%Y-%m-%d %H:%M} UTC by
<code>profiling/report_html.py</code>.</p>
</header>"""

    # -- 1. executive summary ----------------------------------------------
    def exec_summary(self):
        A, B, AS, BS = self.A, self.B, self.AS, self.BS
        job = f"callgrind-{REP}-t1"
        tiles = [stat_tile("Instructions, pancreas 20k, 1 thread", (B.total_ir(job) or 0) / 1e9 or None,
                           (A.total_ir(job) or 0) / 1e9 or None, " G Ir", nd=1,
                           note="callgrind, load-independent")]
        for t in (1, 8, 16):
            nb, na = B.native_row(REP, t), A.native_row(REP, t)
            tiles.append(stat_tile(f"Wall, pancreas 20k, {t} thr", nb and nb["wall_s_min"],
                                   na and na["wall_s_min"], " s", nd=1, note="min of 3 reps, load-sensitive"))
        for t in (1, 16):
            nb, na = B.native_row(REP, t), A.native_row(REP, t)
            tiles.append(stat_tile(f"CPU time, pancreas 20k, {t} thr", nb and nb["cpu_s_median"],
                                   na and na["cpu_s_median"], " s", nd=1, note="median of 3 reps"))
        nb, na = B.native_row(REP, 1), A.native_row(REP, 1)
        tiles.append(stat_tile("Peak RSS, pancreas 20k, 1 thr", nb and nb["peak_rss_kb"] / 1024,
                               na and na["peak_rss_kb"] / 1024, " MiB", nd=0))
        cb_, ca_ = B.cg(job), A.cg(job)
        if cb_ and ca_:
            tiles.append(stat_tile("Amdahl bound at 16 thr (whole run)", cb_["amdahl"]["16"], ca_["amdahl"]["16"],
                                   "×", lower_is_better=False, nd=2,
                                   note=f"serial share {100 * cb_['serial_frac']:.0f} % → {100 * ca_['serial_frac']:.0f} %"))
        db_, da_ = B.jobs().get(f"dhat-{REP}-t1"), A.jobs().get(f"dhat-{REP}-t1")
        if db_ and da_:
            tiles.append(stat_tile("Heap allocations, 1 thr", db_["total_blocks"] / 1e6, da_["total_blocks"] / 1e6,
                                   " M", nd=2, note=f"peak heap {db_['peak_bytes'] / 2**20:.1f} → "
                                                    f"{da_['peak_bytes'] / 2**20:.1f} MiB (DHAT)"))
        jp = f"{job}-plot"
        if B.total_ir(jp) and A.total_ir(jp):
            tiles.append(stat_tile("Instructions with --plot", B.total_ir(jp) / 1e9, A.total_ir(jp) / 1e9, " G Ir",
                                   nd=1, note="HTML reports included"))
        crop = '<div class="tiles">' + "".join(tiles) + "</div>"

        real = []
        for ds, lab in (("lung_2M", "lung 2M"), ("lung_all", "lung whole slide"),
                        ("prime5k_8M", "prime5k 8M"), ("cosmx_wtx_colon_full", "WTx 3M (18,935 g)")):
            jb, ja = BS.ok_job("gperf", ds, 8), AS.ok_job("gperf", ds, 8)
            real.append([f"<strong>{esc(lab)}</strong>",
                         fmt(jb and jb["cpu_s"], 0), fmt(ja and ja["cpu_s"], 0),
                         delta(jb and jb["cpu_s"], ja and ja["cpu_s"]),
                         fmt(jb and jb["wall_s"], 0), fmt(ja and ja["wall_s"], 0),
                         delta(jb and jb["wall_s"], ja and ja["wall_s"]),
                         fmt(jb and jb["peak_rss_kb"] / 2**20, 2), fmt(ja and ja["peak_rss_kb"] / 2**20, 2),
                         delta(jb and jb["peak_rss_kb"], ja and ja["peak_rss_kb"]),
                         f"{fmt((jb or {}).get('loadavg_start', '[]') and _load0(jb), 0)} / "
                         f"{fmt(_load0(ja), 0)}"])
        rt = table(["real size, 8 threads", "CPU s before", "CPU s after", "Δ", "wall s before",
                    "wall s after", "Δ (load-sens.)", "peak RSS GiB before", "after", "Δ", "load 1m b / a"],
                   real, raw=True, num_cols=range(1, 11))
        return self.section("summary", "1 · Executive summary",
                            self.note("summary") + "<h3>Representative crop (19,297 molecules, 141 genes)</h3>"
                            + crop + "<h3>Real data sizes (scaling tier, gperftools + /proc)</h3>" + rt
                            + "<h3>Five biggest remaining bottlenecks</h3>" + self.note("bottlenecks_top"))

    # -- 2. where time goes ------------------------------------------------
    def where_time(self):
        A, B = self.A, self.B
        body = [self.note("where_time")]
        body.append(chart_phase_stack(B, A, "Instructions per phase, 1 thread: after (solid) vs before (faded)"))
        # phase table
        rows = []
        for ds, lab in DATASETS:
            job = f"callgrind-{ds}-t1"
            pa, pb = B.phase_ir(job), A.phase_ir(job)
            if not pb:
                continue
            ta, tb = B.total_ir(job), A.total_ir(job)
            r = [esc(lab), G(ta), G(tb), delta(ta, tb)]
            for ph in ("mol_clustering", "bmm_iterations", "ncv_colors"):
                a = pa.get(ph, {}).get("Ir")
                b = pb.get(ph, {}).get("Ir")
                r += [G(b), delta(a, b)]
            gl = pb.get("ncv_colors", {}).get("pct")
            rows.append(r)
        body.append(table(["dataset", "total G Ir before", "after", "Δ", "mol_clustering after", "Δ",
                           "bmm_iterations after", "Δ", "ncv_colors after", "Δ"], rows, raw=True,
                          num_cols=range(1, 10), caption="Phase instructions (G Ir, 1 thread, callgrind)"))
        # share table after
        rows = []
        for ds, lab in DATASETS:
            ph = A.phase_ir(f"callgrind-{ds}-t1")
            if not ph:
                continue
            agg = defaultdict(float)
            for n, r in ph.items():
                agg[group_of(n)] += r["pct"]
            rows.append([lab] + [f"{agg.get(g, 0):.1f}" for g in PHASE_GROUPS if g != "report"])
        body.append(table(["dataset (after)"] + [g for g in PHASE_GROUPS if g != "report"], rows,
                          num_cols=range(1, 7), caption="Share of instructions per phase after (%)"))
        # top functions with tabs per dataset
        tabs = []
        for ds, lab in DATASETS:
            job = f"callgrind-{ds}-t1"
            if not A.cg(job):
                continue
            tabs.append((lab, self.fn_tables(job)))
        body.append("<h3>Top functions and hot lines (after), per dataset</h3>"
                    "<p class='cap'>Exclusive = the function's own instructions; inclusive = with callees. "
                    "Before = same short name in the 2026-09-30 run (— = not in its top 60). "
                    "Pool chunks (<code>.pool_chunk</code>) are the bodies of parallel loops of the new "
                    "thread pool (the former <code>._omp_fn.N</code>).</p>")
        body.append(tabset("fn", tabs))
        return self.section("where", "2 · Where the time goes now", "\n".join(body))

    def fn_tables(self, job):
        A, B = self.A, self.B
        ca, cb = A.cg(job), B.cg(job)
        before = {}
        if cb:
            for r in cb["top_self"] + cb["top_incl"]:
                before.setdefault(r["name"], r)
        rows = []
        for r in ca["top_self"][:20]:
            b = before.get(r["name"])
            rows.append([code(r["name"]), code(r.get("hot_line") or ""), f"{r['self_pct']:.2f}",
                         f"{r['incl_pct']:.2f}", fmt(r["calls"], 0),
                         f"{b['self_pct']:.2f}" if b else "—",
                         delta(b and b["self_Ir"], r["self_Ir"]),
                         esc(" < ".join((r.get("hot_path") or [])[:4]))])
        t1 = table(["function", "hottest line", "excl %", "incl %", "calls", "excl % before", "Δ Ir",
                    "dominant call path"], rows, raw=True, num_cols={2, 3, 4, 5, 6},
                   caption=f"By exclusive Ir — total {G(ca['total_Ir'])} G Ir")
        rows = []
        for r in ca["top_incl"][:15]:
            b = before.get(r["name"])
            rows.append([code(r["name"]), code(r.get("file") or ""), f"{r['incl_pct']:.2f}",
                         f"{b['incl_pct']:.2f}" if b else "—", delta(b and b["incl_Ir"], r["incl_Ir"])])
        t2 = table(["function", "file", "incl %", "incl % before", "Δ Ir"], rows, raw=True,
                   num_cols={2, 3, 4}, caption="By inclusive Ir")
        rows = []
        for r in ca["top_lines"][:14]:
            rows.append([code(r["fn"]), code(f"{r['file']}:{r['line']}"), f"{r['pct_of_total']:.2f}",
                         fmt(r.get("pct_of_fn"), 1)])
        t3 = table(["function", "source line", "% of total Ir", "% of function"], rows, raw=True,
                   num_cols={2, 3}, caption="Hot source lines")
        return t1 + t2 + t3

    # -- 3. per optimization area ------------------------------------------
    def phase_delta_rows(self, phase, tier_b=None, tier_a=None, datasets=None):
        B, A = tier_b or self.B, tier_a or self.A
        rows = []
        for ds, lab in DATASETS:
            if datasets and ds not in datasets:
                continue
            job = f"callgrind-{ds}-t1"
            pa, pb = B.phase_ir(job).get(phase), A.phase_ir(job).get(phase)
            if not pa and not pb:
                continue
            na, nb = B.native_phase(ds, 1, phase), A.native_phase(ds, 1, phase)
            rows.append([esc(lab), G(pa and pa["Ir"]), G(pb and pb["Ir"]), delta(pa and pa["Ir"], pb and pb["Ir"]),
                         ratio(pa and pa["Ir"], pb and pb["Ir"]),
                         fmt(pa and pa["pct"], 1), fmt(pb and pb["pct"], 1),
                         fmt(pa and 100 * pa["serial_frac"], 1), fmt(pb and 100 * pb["serial_frac"], 1),
                         fmt(na and na["cpu_s_median"], 2), fmt(nb and nb["cpu_s_median"], 2)])
        return table(["dataset", "G Ir before", "after", "Δ", "speed-up", "% of run before", "after",
                      "serial % before", "after", "CPU s before (native 1 thr, load-sens.)", "after"], rows, raw=True,
                     num_cols=range(1, 11), caption=f"Phase <code>{phase}</code>, 1 thread")

    def scaling_phase_rows(self, phase, datasets, thr=8):
        rows = []
        for ds in datasets:
            pa, pb = self.BS.phase(ds, thr, phase), self.AS.phase(ds, thr, phase)
            ja, jb = self.BS.ok_job("gperf", ds, thr), self.AS.ok_job("gperf", ds, thr)
            if not pa and not pb:
                continue
            rows.append([code(ds), fmt(pa and pa["cpu_s"], 1), fmt(pb and pb["cpu_s"], 1),
                         delta(pa and pa["cpu_s"], pb and pb["cpu_s"]),
                         fmt(pa and ja and 100 * pa["cpu_s"] / ja["cpu_s"], 1),
                         fmt(pb and jb and 100 * pb["cpu_s"] / jb["cpu_s"], 1),
                         fmt(pa and pa.get("cpu_per_wall"), 2), fmt(pb and pb.get("cpu_per_wall"), 2)])
        return table(["rung", "CPU s before", "after", "Δ", "% of run CPU before", "after", "CPU/wall before",
                      "after (load-sens.)"], rows, raw=True, num_cols=range(1, 8),
                     caption=f"Phase <code>{phase}</code> at real sizes, {thr} threads (/proc CPU s)")

    def areas(self):
        A, B = self.A, self.B
        body = []
        # NCV
        ncv = [self.note("area_ncv"), self.phase_delta_rows("ncv_colors")]
        jb, ja = "callgrind-xenium_pancreas_g377_20k-t1-plot", "callgrind-xenium_pancreas_g377_20k-t1-plot"
        pb, pa = B.phase_ir(jb), A.phase_ir(ja)
        if pa:
            rows = []
            for ph in ("ncv_colors", "report", "bmm_iterations"):
                x, y = pb.get(ph), pa.get(ph)
                rows.append([code(ph), G(x and x["Ir"]), G(y and y["Ir"]), delta(x and x["Ir"], y and y["Ir"]),
                             fmt(x and 100 * x["serial_frac"], 1), fmt(y and 100 * y["serial_frac"], 1)])
            rows.append(["<strong>whole run</strong>", G(B.total_ir(jb)), G(A.total_ir(ja)),
                         delta(B.total_ir(jb), A.total_ir(ja)), "", ""])
            ncv.append(table(["phase", "G Ir before", "after", "Δ", "serial % before", "after"], rows, raw=True,
                             num_cols=range(1, 6), caption="<code>--plot</code> run (pancreas 20k, 1 thread)"))
        ncv.append(self.scaling_phase_rows("ncv_colors", ["lung_100k", "lung_1M", "lung_all", "prime5k_8M"]))
        if self.extra:
            ncv.append(self.extra_ncv_table())
        body.append(("NCV colours and --plot", "\n".join(ncv)))
        # BMM
        bmm = [self.note("area_bmm"), self.phase_delta_rows("bmm_iterations"),
               self.scaling_phase_rows("bmm_iterations", ["lung_100k", "lung_1M", "lung_2M", "lung_4M", "lung_8M",
                                                          "lung_all", "prime5k_8M"])]
        bmm.append(self.dhat_compare(["xenium_pancreas_g377_20k", "xenium_pancreas_g377_40k"], churn=True))
        body.append(("BMM loop", "\n".join(bmm)))
        # clustering
        cl = [self.note("area_clustering"), self.phase_delta_rows("mol_clustering"),
              self.scaling_phase_rows("mol_clustering", ["lung_100k", "lung_2M", "lung_all", "prime5k_100k",
                                                         "prime5k_8M", "cosmx_wtx_colon_full"])]
        nb = [(ds, B.native_row(ds, 1), A.native_row(ds, 1)) for ds in ("sim_circles_g5000_10k",
                                                                         "sim_circles_g5000_20k", "cosmx_wtx_20k")]
        cl.append(table(["dataset", "wall s before (native 1 thr)", "after", "Δ", "peak RSS MiB before", "after"],
                        [[code(ds), fmt(b and b["wall_s_min"], 1), fmt(a and a["wall_s_min"], 1),
                          delta(b and b["wall_s_min"], a and a["wall_s_min"]),
                          fmt(b and b["peak_rss_kb"] / 1024, 0), fmt(a and a["peak_rss_kb"] / 1024, 0)]
                         for ds, b, a in nb], raw=True, num_cols=range(1, 6),
                        caption="Gene-rich crops, native (load-sensitive wall)"))
        body.append(("Molecule clustering and large panels", "\n".join(cl)))
        # memory
        mem = [self.note("area_memory"), self.rss_table(), self.heaptrack_table()]
        body.append(("Memory at scale", "\n".join(mem)))
        # threading
        th = [self.note("area_threading"), self.thread_cpu_table()]
        body.append(("Threading (own pool instead of OpenMP)", "\n".join(th)))
        html_ = "".join(f"<h3>{esc(t)}</h3>{b}" for t, b in body)
        return self.section("areas", "3 · Per optimization area: what changed, measured effect", html_)

    def extra_ncv_table(self):
        lab, X, XS = self.extra
        rows = []
        for ds, dl in DATASETS:
            job = f"callgrind-{ds}-t1"
            b, x, a = (T.phase_ir(job).get("ncv_colors") for T in (self.B, X, self.A))
            if not a:
                continue
            # native times only for the representative crop: the other 1-thread
            # native jobs share a lane with the Valgrind pool (load-sensitive)
            nb, nx, na = ((T.native_phase(ds, 1, "ncv_colors") for T in (self.B, X, self.A))
                          if ds == REP else (None, None, None))
            rows.append([esc(dl), G(b and b["Ir"]), G(x and x["Ir"]), G(a["Ir"]),
                         fmt(nb and nb["cpu_s_median"], 2), fmt(nx and nx["cpu_s_median"], 2),
                         fmt(na and na["cpu_s_median"], 2)])
        for t in (8,):
            nb, nx, na = (T.native_phase(REP, t, "ncv_colors") for T in (self.B, X, self.A))
            rows.append([esc(f"pancreas 20k, {t} thr (wall s)"), "", "", "", fmt(nb and nb["wall_s_median"], 2),
                         fmt(nx and nx["wall_s_median"], 2), fmt(na and na["wall_s_median"], 2)])
        jp = f"callgrind-{REP}-t1-plot"
        rows.append(["<strong>--plot run, whole (G Ir)</strong>", G(self.B.total_ir(jp)), G(X.total_ir(jp)),
                     G(self.A.total_ir(jp)), "", "", ""])
        for ds in ("lung_all", "prime5k_8M"):
            pb, px, pa = self.BS.phase(ds, 8, "ncv_colors"), XS.phase(ds, 8, "ncv_colors"), self.AS.phase(ds, 8, "ncv_colors")
            rows.append([code(f"{ds} 8 thr (CPU s)"), "", "", "", fmt(pb and pb["cpu_s"], 0), fmt(px and px["cpu_s"], 0),
                         fmt(pa and pa["cpu_s"], 0)])
        return table(["dataset", "G Ir before", f"with {esc(lab)}", "now", "CPU s before (native 1 thr, thread series)",
                      f"with {esc(lab)}", "now"], rows, raw=True, num_cols=range(1, 7),
                     caption=f"Phase <code>ncv_colors</code>: before, with {esc(lab)}, and now")

    def dhat_compare(self, datasets, churn=False):
        rows = []
        for ds in datasets:
            jb = jbj = f"dhat-{ds}-t1"
            a, b = self.B.jobs().get(jb), self.A.jobs().get(jbj)
            if not a and not b:
                continue
            rows.append([code(ds), fmt(a and a["peak_bytes"] / 2**20, 1), fmt(b and b["peak_bytes"] / 2**20, 1),
                         delta(a and a["peak_bytes"], b and b["peak_bytes"]),
                         fmt(a and a["total_bytes"] / 2**30, 2), fmt(b and b["total_bytes"] / 2**30, 2),
                         delta(a and a["total_bytes"], b and b["total_bytes"]),
                         fmt(a and a["total_blocks"] / 1e6, 2), fmt(b and b["total_blocks"] / 1e6, 2),
                         delta(a and a["total_blocks"], b and b["total_blocks"])])
        return table(["dataset", "peak heap MiB before", "after", "Δ", "allocated GiB before", "after", "Δ",
                      "allocations M before", "after", "Δ"], rows, raw=True, num_cols=range(1, 10),
                     caption="DHAT, 1 thread")

    def rss_table(self):
        rows = []
        for ds in ["lung_100k", "lung_500k", "lung_1M", "lung_2M", "lung_4M", "lung_8M", "lung_all",
                   "prime5k_1M", "prime5k_2M", "prime5k_4M", "prime5k_8M", "cosmx_wtx_colon_full"]:
            a, b = self.BS.ok_job("gperf", ds, 8), self.AS.ok_job("gperf", ds, 8)
            if not a and not b:
                continue
            mols = (b or a).get("molecules_loaded") or (b or a)["molecules"]
            rows.append([code(ds), fmt(mols, 0), fmt(a and a["peak_rss_kb"] / 2**20, 2),
                         fmt(b and b["peak_rss_kb"] / 2**20, 2), delta(a and a["peak_rss_kb"], b and b["peak_rss_kb"]),
                         fmt(a and 1024 * a["peak_rss_kb"] / mols, 0), fmt(b and 1024 * b["peak_rss_kb"] / mols, 0)])
        return table(["rung (8 thr)", "molecules", "peak RSS GiB before", "after", "Δ", "bytes/molecule before",
                      "after"], rows, raw=True, num_cols=range(1, 7), caption="Peak RSS (/proc VmHWM)")

    def heaptrack_table(self):
        rows = []
        for ds in ("lung_2M", "prime5k_2M", "lung_all"):
            a, b = self.BS.ok_job("heaptrack", ds, 8), self.AS.ok_job("heaptrack", ds, 8)
            ra = self.BS.job("heaptrack", ds, 8)
            rb = self.AS.job("heaptrack", ds, 8)
            st = lambda r, ok: "ok" if ok else ("stopped/partial" if r else "not run")  # noqa: E731
            rows.append([code(ds), fmt(a and a.get("heaptrack_peak_heap", 0) / 2**30, 2),
                         fmt(b and b.get("heaptrack_peak_heap", 0) / 2**30, 2),
                         delta(a and a.get("heaptrack_peak_heap"), b and b.get("heaptrack_peak_heap")),
                         fmt(a and a.get("heaptrack_calls", 0) / 1e6, 0), fmt(b and b.get("heaptrack_calls", 0) / 1e6, 0),
                         delta(a and a.get("heaptrack_calls"), b and b.get("heaptrack_calls")),
                         esc(st(ra, a)), esc(st(rb, b))])
        return table(["rung (8 thr)", "peak heap GiB before", "after", "Δ", "allocations M before", "after", "Δ",
                      "status before", "after"], rows, raw=True, num_cols=range(1, 7), caption="heaptrack")

    def thread_cpu_table(self):
        rows = []
        for ds in (REP, "xenium_prime5k_20k"):
            for t in (1, 2, 4, 8, 16):
                a, b = self.B.native_row(ds, t), self.A.native_row(ds, t)
                if not a and not b:
                    continue
                rows.append([code(ds), t, fmt(a and a["cpu_s_median"], 1), fmt(b and b["cpu_s_median"], 1),
                             delta(a and a["cpu_s_median"], b and b["cpu_s_median"]),
                             fmt(a and a["cpu_per_wall_median"], 2), fmt(b and b["cpu_per_wall_median"], 2)])
        return table(["dataset", "threads", "CPU s before", "after", "Δ", "CPU/wall before", "after"], rows,
                     raw=True, num_cols=range(1, 7), caption="Native CPU time vs threads (median of 3)")

    # -- 4. threads --------------------------------------------------------
    def threads(self):
        A, B = self.A, self.B
        body = [self.note("threads")]
        charts = []
        for ds, lab in ((REP, "pancreas 20k"), ("xenium_prime5k_20k", "prime5k 20k")):
            ser = []
            for tier, which, bef in ((B, "before", True), (A, "after", False)):
                xs, wall, cpu, sp = [], [], [], []
                base = tier.native_row(ds, 1)
                for t in (1, 2, 4, 8, 16):
                    r = tier.native_row(ds, t)
                    if r:
                        xs.append(t)
                        wall.append(r["wall_s_min"])
                        cpu.append(r["cpu_s_median"])
                        sp.append(base["wall_s_min"] / r["wall_s_min"] if base else None)
                ser.append((which, bef, xs, wall, cpu, sp))
            c1 = chart_lines([{"label": f"{w}", "xs": xs, "ys": wl, "color": SLOTS[0], "before": bf}
                              for w, bf, xs, wl, _, _ in ser], f"{lab}: wall (min of 3)", "threads", "wall s",
                             logx=True, xticks=[1, 2, 4, 8, 16])
            c2 = chart_lines([{"label": f"{w}", "xs": xs, "ys": sp, "color": SLOTS[1], "before": bf}
                              for w, bf, xs, _, _, sp in ser], f"{lab}: speed-up vs 1 thread", "threads",
                             "speed-up", logx=True, xticks=[1, 2, 4, 8, 16])
            c3 = chart_lines([{"label": f"{w}", "xs": xs, "ys": cp, "color": SLOTS[2], "before": bf}
                              for w, bf, xs, _, cp, _ in ser], f"{lab}: CPU time (median)", "threads", "CPU s",
                             logx=True, xticks=[1, 2, 4, 8, 16])
            charts.append(f'<div class="grid3">{c1}{c2}{c3}</div>')
        body.append("<h3>Native thread series (load-sensitive; load average in the table)</h3>" + "".join(charts))
        rows = []
        for ds in (REP, "xenium_prime5k_20k"):
            for t in (1, 2, 4, 8, 16):
                a, b = B.native_row(ds, t), A.native_row(ds, t)
                a1, b1 = B.native_row(ds, 1), A.native_row(ds, 1)
                if not a and not b:
                    continue
                spa = a1["wall_s_min"] / a["wall_s_min"] if a and a1 else None
                spb = b1["wall_s_min"] / b["wall_s_min"] if b and b1 else None
                rows.append([code(ds), t, fmt(a and a["wall_s_min"], 2), fmt(b and b["wall_s_min"], 2),
                             delta(a and a["wall_s_min"], b and b["wall_s_min"]),
                             fmt(spa, 2), fmt(spb, 2), fmt(spa and 100 * spa / t, 0), fmt(spb and 100 * spb / t, 0),
                             _loads(a), _loads(b)])
        body.append(table(["dataset", "thr", "wall min before", "after", "Δ", "speed-up before", "after",
                           "efficiency % before", "after", "load before", "load after"], rows, raw=True,
                          num_cols=range(1, 9), caption="Wall, speed-up and parallel efficiency (speed-up / threads)"))
        pa = A.native_row(REP, 8, "passive")
        pb = B.native_row(REP, 8, "passive")
        if pa or pb:
            body.append(table(["8 threads, pancreas 20k", "wall min before", "after", "CPU s before", "after"],
                              [["default wait", fmt(B.native_row(REP, 8)["wall_s_min"], 2),
                                fmt(A.native_row(REP, 8)["wall_s_min"], 2),
                                fmt(B.native_row(REP, 8)["cpu_s_median"], 1), fmt(A.native_row(REP, 8)["cpu_s_median"], 1)],
                               ["passive (before: OMP_WAIT_POLICY=passive; after: BAYSOR_POOL_SPIN_US=0)",
                                fmt(pb and pb["wall_s_min"], 2), fmt(pa and pa["wall_s_min"], 2),
                                fmt(pb and pb["cpu_s_median"], 1), fmt(pa and pa["cpu_s_median"], 1)]],
                              raw=True, num_cols=range(1, 5), caption="Wait policy"))
        # Amdahl per phase
        job = f"callgrind-{REP}-t1"
        pa, pb = B.phase_ir(job), A.phase_ir(job)
        rows = []
        for ph in MAIN_PHASES:
            x, y = pa.get(ph), pb.get(ph)
            if not x and not y:
                continue
            rows.append([code(ph), fmt(x and x["pct"], 1), fmt(y and y["pct"], 1),
                         fmt(x and 100 * x["serial_frac"], 1), fmt(y and 100 * y["serial_frac"], 1),
                         fmt(x and x["amdahl_8"], 2), fmt(y and y["amdahl_8"], 2),
                         fmt(x and x["amdahl_16"], 2), fmt(y and y["amdahl_16"], 2)])
        ca, cb = B.cg(job), A.cg(job)
        if ca and cb:
            rows.append(["<strong>whole run</strong>", "100", "100", fmt(100 * ca["serial_frac"], 1),
                         fmt(100 * cb["serial_frac"], 1), fmt(ca["amdahl"]["8"], 2), fmt(cb["amdahl"]["8"], 2),
                         fmt(ca["amdahl"]["16"], 2), fmt(cb["amdahl"]["16"], 2)])
        body.append(table(["phase", "% Ir before", "after", "serial % before", "after", "Amdahl 8 thr before",
                           "after", "Amdahl 16 thr before", "after"], rows, raw=True, num_cols=range(1, 9),
                          caption="Serial fraction and Amdahl bound per phase, pancreas 20k (callgrind, 1 thread; "
                                  "serial = main-thread Ir outside parallel-region bodies)"))
        rows = []
        for ds, lab in DATASETS:
            j = f"callgrind-{ds}-t1"
            ca, cb = B.cg(j), A.cg(j)
            if not ca or not cb:
                continue
            rows.append([esc(lab), fmt(100 * ca["serial_frac"], 1), fmt(100 * cb["serial_frac"], 1),
                         fmt(ca["amdahl"]["8"], 2), fmt(cb["amdahl"]["8"], 2), fmt(ca["amdahl"]["16"], 2),
                         fmt(cb["amdahl"]["16"], 2)])
        body.append(table(["dataset", "serial % before", "after", "Amdahl 8 before", "after", "Amdahl 16 before",
                           "after"], rows, raw=True, num_cols=range(1, 7), caption="Whole-run serial share, all crops"))
        # per-phase native
        rows = []
        for ph in ("confidence", "mol_clustering", "bmm_iterations", "ncv_colors", "polygons"):
            r = [code(ph)]
            for t in (1, 8, 16):
                x, y = B.native_phase(REP, t, ph), A.native_phase(REP, t, ph)
                r += [f"{fmt(x and x['wall_s_median'], 2)} / {fmt(x and x['cpu_s_median'], 2)}",
                      f"{fmt(y and y['wall_s_median'], 2)} / {fmt(y and y['cpu_s_median'], 2)}"]
            rows.append(r)
        body.append(table(["phase (wall / CPU s, median)", "1 thr before", "after", "8 thr before", "after",
                           "16 thr before", "after"], rows, raw=True,
                          caption="Per phase, native, pancreas 20k (load-sensitive)"))
        # callgrind thread series
        rows = []
        for t in (1, 2, 4, 8, 16):
            j = f"callgrind-{REP}-t{t}"
            ca, cb = B.cg(j), A.cg(j)
            if not ca and not cb:
                continue
            def mainpct(c):
                if not c:
                    return None
                th = c["threads"]
                return 100 * th.get("1", 0) / sum(th.values())
            rows.append([t, G(ca and ca["total_Ir"]), G(cb and cb["total_Ir"]), fmt(mainpct(ca), 1), fmt(mainpct(cb), 1),
                         fmt(ca and _phase_val(ca, "omp_worker_runtime"), 3) if ca else "—",
                         fmt(cb and (_phase_val(cb, "pool_worker_runtime") or 0), 3) if cb else "—"])
        body.append(table(["threads", "total G Ir before", "after", "main thread % before", "after",
                           "worker runtime G Ir before (OpenMP)", "after (pool)"], rows, raw=True, num_cols=range(0, 7),
                          caption="Instructions vs threads (callgrind, pancreas 20k). Under Valgrind threads run "
                                  "one at a time, so the split of dynamically scheduled chunks over threads is "
                                  "not the native one."))
        # regions table (after) at 16 threads
        j16 = f"callgrind-{REP}-t16"
        cr = A.cg(j16)
        if cr:
            rows = []
            for r in cr["omp_regions"][:14]:
                rows.append([code(r["region"]), f"{r['pct']:.2f}", fmt(r["calls_main"], 0), r["n_threads_active"],
                             fmt(r["max_over_mean"], 2)])
            body.append(table(["parallel region (pool chunk)", "% Ir", "entries (main thread)", "threads with work",
                               "max/mean over threads"], rows, raw=True, num_cols={1, 2, 3, 4},
                              caption="Parallel regions at 16 threads, after (callgrind)"))
        # real size parallel
        body.append(self.real_parallel())
        tabs = []
        for ds, thr in (("lung_2M", 1), ("lung_1M", 1), ("prime5k_500k", 1), ("lung_2M", 8), ("lung_all", 8),
                        ("prime5k_8M", 8), ("cosmx_wtx_colon_full", 8)):
            rows = [r for r in self.AS.parallel if r["dataset"] == ds and r["threads"] == thr
                    and r["phase"] != "(whole run)" and (r["samples"] or 0) >= 0.002 * max(
                        (x["samples"] for x in self.AS.parallel if x["dataset"] == ds and x["threads"] == thr), default=1)]
            if not rows:
                continue
            tot = sum(r["samples"] for r in rows) or 1
            rows.sort(key=lambda r: -r["samples"])
            tabs.append((f"{ds}, {thr} thr", table(
                ["phase", "% of CPU samples", "parallel %", "wait %", "serial % of non-wait CPU", "Amdahl 8", "Amdahl 16"],
                [[code(r["phase"]), fmt(100 * r["samples"] / tot, 1), fmt(r["parallel_pct"], 1), fmt(r["wait_pct"], 1),
                  fmt(None if r["serial_frac"] is None else 100 * r["serial_frac"], 1), fmt(r["amdahl_8"], 2),
                  fmt(r["amdahl_16"], 2)] for r in rows], raw=True, num_cols=range(1, 7))))
        if tabs:
            body.append("<h3>Per phase at real sizes (after; gperftools stack classification)</h3>"
                        "<p class='cap'>Phase = outermost phase function on the sample's stack; worker samples "
                        "are mapped to phases through the main thread's samples of the same pool loop. At 1 thread "
                        "the serial share is Amdahl's <em>s</em> for that phase at this size.</p>" + tabset("rp", tabs))
        return self.section("threads", "4 · Threads: 1 → 16", "\n".join(body))

    def real_parallel(self):
        rows = []
        for ds in ("lung_100k", "lung_500k", "lung_1M", "lung_2M", "prime5k_100k", "prime5k_500k"):
            a1, b1 = self.BS.ok_job("gperf", ds, 1), self.AS.ok_job("gperf", ds, 1)
            a8, b8 = self.BS.ok_job("gperf", ds, 8), self.AS.ok_job("gperf", ds, 8)
            pa = next((r for r in self.BS.parallel if r["dataset"] == ds and r["threads"] == 1
                       and r["phase"] == "(whole run)"), None)
            pb = next((r for r in self.AS.parallel if r["dataset"] == ds and r["threads"] == 1
                       and r["phase"] == "(whole run)"), None)
            wa = next((r for r in self.BS.parallel if r["dataset"] == ds and r["threads"] == 8
                       and r["phase"] == "(whole run)"), None)
            wb = next((r for r in self.AS.parallel if r["dataset"] == ds and r["threads"] == 8
                       and r["phase"] == "(whole run)"), None)
            rows.append([code(ds), fmt(a8 and a1 and a8["cpu_s"] / a1["cpu_s"], 2),
                         fmt(b8 and b1 and b8["cpu_s"] / b1["cpu_s"], 2),
                         fmt(a8 and a8["cpu_s"] / a8["wall_s"], 2), fmt(b8 and b8["cpu_s"] / b8["wall_s"], 2),
                         fmt(wa and wa["wait_pct"], 1), fmt(wb and wb["wait_pct"], 1),
                         fmt(pa and 100 * pa["serial_frac"], 1), fmt(pb and 100 * pb["serial_frac"], 1),
                         fmt(pa and pa["amdahl_8"], 2), fmt(pb and pb["amdahl_8"], 2)])
        return table(["rung", "CPU 8 thr / CPU 1 thr before", "after", "CPU/wall 8 thr before", "after (load-sens.)",
                      "wait % of CPU, 8 thr, before", "after", "serial % 1 thr before", "after", "Amdahl 8 before",
                      "after"], rows, raw=True, num_cols=range(1, 11),
                     caption="Real sizes (gperftools): extra CPU for threading, waiting and serial share. "
                             "Before: OpenMP wait = libgomp barrier/spin leaf samples; serial share from the "
                             "inclusive samples of the <code>._omp_fn</code> bodies. After: stack classification "
                             "(pool chunks = parallel; pool hand-off/barrier machinery and Eigen pool idling = wait).")

    # -- 5. scaling --------------------------------------------------------
    def scaling(self):
        AS, BS = self.AS, self.BS
        body = [self.note("scaling")]

        def series(S, slide, thr, key, scale=1.0, phase=None):
            pts = []
            for r in S.jobs:
                if r["tool"] != "gperf" or r.get("ok") is not True or r["threads"] != thr:
                    continue
                if not re.match(rf"^{slide}_(\d+k|\d+M|all)$", r["dataset"]):
                    continue
                x = r.get("molecules_loaded") or r["molecules"]
                if phase:
                    p = S.phase(r["dataset"], thr, phase)
                    y = p and p.get(key)
                else:
                    y = r.get(key)
                if y:
                    pts.append((x, y * scale))
            pts.sort()
            return [p[0] for p in pts], [p[1] for p in pts]

        charts = []
        for key, lab, scale in (("cpu_s", "CPU s", 1.0), ("peak_rss_kb", "peak RSS GiB", 1 / 2**20),
                                ("wall_s", "wall s (load-sens.)", 1.0)):
            ss = []
            for i, (slide, thr) in enumerate((("lung", 8), ("prime5k", 8), ("lung", 1), ("prime5k", 1))):
                for S, bef in ((BS, True), (AS, False)):
                    xs, ys = series(S, slide, thr, key, scale)
                    ss.append({"label": f"{slide} {thr} thr {'before' if bef else 'after'}", "xs": xs, "ys": ys,
                               "color": SLOTS[i], "before": bef})
            charts.append(chart_lines(ss, f"{lab} vs molecules", "molecules", lab, logx=True, logy=True,
                                      figsize=(4.7, 3.6)))
        body.append(f'<div class="grid3">{"".join(charts)}</div>')
        # per phase small multiples, lung 8 thr & prime5k 8 thr
        for slide in ("lung", "prime5k"):
            ch = []
            for ph in ("bmm_iterations", "mol_clustering", "ncv_colors", "confidence", "polygons", "molecule_graph"):
                ss = []
                for S, bef in ((BS, True), (AS, False)):
                    xs, ys = series(S, slide, 8, "cpu_s", phase=ph)
                    ss.append({"label": "before" if bef else "after", "xs": xs, "ys": ys,
                               "color": GROUP_COLOR.get(group_of(ph), SLOTS[6]), "before": bef})
                ch.append(chart_lines(ss, f"{slide} 8 thr: {ph}", "molecules", "CPU s", logx=True, logy=True,
                                      figsize=(3.4, 2.7)))
            body.append(f"<h3>{slide}, 8 threads: CPU seconds per phase (log-log)</h3>"
                        f'<div class="grid3">{"".join(ch)}</div>')
        # exponents table
        rows = []
        for slide, thr in (("lung", 8), ("lung", 1), ("prime5k", 8), ("prime5k", 1)):
            for what, name in [("total", "cpu_s"), ("total", "peak_rss_kb")] + \
                              [("phase_cpu", p) for p in ("confidence", "molecule_graph", "mol_clustering", "bmm_init",
                                                         "bmm_iterations", "ncv_colors", "polygons", "out_molecules")]:
                a, b = BS.fit(slide, thr, what, name), AS.fit(slide, thr, what, name)
                if not a and not b:
                    continue
                ea, eb = a and a["exponent"], b and b["exponent"]
                sup_b = isinstance(eb, (int, float)) and eb >= 1.15
                if sup_b and (b.get("superlinear") is True or what == "total"):
                    flag = '<span class="flag">super-linear</span>'
                elif sup_b:
                    flag = '<span class="flag old">super-linear, < 2 % of CPU</span>'
                else:
                    flag = ""
                was = ('<span class="flag old">was super-linear</span>'
                       if isinstance(ea, (int, float)) and ea >= 1.15 and not sup_b else "")
                sh = lambda r: (f"{100 * r['share_first']:.1f} → {100 * r['share_last']:.1f}"  # noqa: E731
                                if r and isinstance(r.get("share_first"), (int, float)) else "")
                rows.append([esc(slide), thr, esc(what.replace("_cpu", "")), code(name), fmt(ea, 2), fmt(eb, 2),
                             esc(sh(a)), esc(sh(b)), flag + was])
        body.append(table(["slide", "thr", "what", "name", "exponent before", "after", "% of CPU first → last rung "
                           "before", "after", ""], rows, raw=True, num_cols={1, 4, 5},
                          caption="Scaling exponents (least-squares slope of log CPU s / log RSS over log molecules; "
                                  "≥ 1.15 flagged)"))
        # function exponents after, super-linear or growing
        rows = []
        for f in AS.fits:
            if not str(f["what"]).startswith("fn_") or f["tool"] != "gperf":
                continue
            shares = json.loads(f["shares"]) if isinstance(f.get("shares"), str) else (f.get("shares") or [])
            if not shares or max(shares) < 0.02:
                continue
            if not (f.get("superlinear") is True or f.get("growing_share") is True):
                continue
            if NOISE_FN.match(f["name"]):
                continue
            rows.append([esc(f["slide"]), f["threads"], esc(f["what"][3:]), code(f["name"]), fmt(f["exponent"], 2),
                         f"{100 * f['share_first']:.1f} → {100 * f['share_last']:.1f}",
                         "yes" if f.get("superlinear") is True else "", "yes" if f.get("growing_share") is True else ""])
        rows.sort(key=lambda r: (r[0], r[1], r[2]))
        body.append(table(["slide", "thr", "kind", "function", "exponent", "% CPU first → last rung", "super-linear",
                           "growing share"], rows, raw=True, num_cols={1, 4},
                          caption="Functions (after) that are super-linear or grow their CPU share ≥ 1.5× "
                                  "(gperftools share × CPU s)"))
        # ladder table
        rows = []
        for r in sorted((r for r in AS.jobs if r["tool"] == "gperf"), key=lambda r: (r["dataset"].split("_")[0],
                                                                                      r["threads"], r["molecules"])):
            b = BS.job("gperf", r["dataset"], r["threads"])
            rows.append([code(r["dataset"]), r["threads"], fmt(r.get("molecules_loaded") or r["molecules"], 0),
                         fmt(r.get("genes_loaded"), 0), fmt(b and b.get("cpu_s"), 0), fmt(r.get("cpu_s"), 0),
                         delta(b and b.get("cpu_s"), r.get("cpu_s")), fmt(b and b.get("wall_s"), 0),
                         fmt(r.get("wall_s"), 0), fmt(b and b.get("peak_rss_kb") and b["peak_rss_kb"] / 2**20, 2),
                         fmt(r.get("peak_rss_kb") and r["peak_rss_kb"] / 2**20, 2),
                         f"{fmt(_load0(b), 0)} / {fmt(_load0(r), 0)}", esc(r.get("status"))])
        body.append(table(["rung", "thr", "molecules", "genes", "CPU s before", "after", "Δ", "wall s before",
                           "after", "RSS GiB before", "after", "load b / a", "status"], rows, raw=True,
                          num_cols=range(1, 12), caption="The whole ladder (gperftools jobs)"))
        return self.section("scaling", "5 · Real-size scaling", "\n".join(body))

    # -- 6. memory ---------------------------------------------------------
    def memory(self):
        A, B = self.A, self.B
        body = [self.note("memory")]
        body.append(self.dhat_compare([ds for ds, _ in DATASETS]))
        rows = []
        for ds, lab in DATASETS + [("sim_circles_g5000_20k", "sim g5000 20k (3,889 g)")]:
            a, b = B.native_row(ds, 1), A.native_row(ds, 1)
            if not a and not b:
                continue
            rows.append([esc(lab), fmt(a and a["peak_rss_kb"] / 1024, 1), fmt(b and b["peak_rss_kb"] / 1024, 1),
                         delta(a and a["peak_rss_kb"], b and b["peak_rss_kb"])])
        body.append(table(["dataset", "peak RSS MiB before (native 1 thr)", "after", "Δ"], rows, raw=True,
                          num_cols={1, 2, 3}))
        tabs = []
        for ds in (REP, "xenium_pancreas_g377_40k", "cosmx_wtx_20k", "xenium_prime5k_20k"):
            jb = f"dhat-{ds}-t1"
            da, db = B.dhat(jb), A.dhat(jb)
            if not db:
                continue
            bsite = {s["site"]: s for s in (da or {}).get("top_live_at_peak", []) + (da or {}).get("top_churn_blocks", [])
                     + (da or {}).get("top_churn_bytes", [])}
            r1 = [[code(s["site"]), code(s["loc"]), fmt(s["gb"] / 2**20, 2), fmt(100 * s["gb"] / db["peak_bytes"], 1),
                   fmt(s["tb"] / 2**20, 1), fmt(s["tbk"], 0)] for s in db["top_live_at_peak"][:10]]
            r2 = [[code(s["site"]), code(s["loc"]), fmt(s["tbk"], 0),
                   fmt(bsite.get(s["site"], {}).get("tbk"), 0),
                   fmt(s["tb"] / 2**20, 1), fmt(bsite.get(s["site"], {}).get("tb") and bsite[s["site"]]["tb"] / 2**20, 1)]
                  for s in db["top_churn_blocks"][:10]]
            tabs.append((DS_LABEL.get(ds, ds),
                         f"<p class='cap'>Peak heap {db['peak_bytes'] / 2**20:.1f} MiB (before "
                         f"{(da or {}).get('peak_bytes', 0) / 2**20:.1f}), {db['total_blocks'] / 1e6:.2f} M "
                         f"allocations (before {(da or {}).get('total_blocks', 0) / 1e6:.2f} M).</p>"
                         + table(["allocation site", "location", "live at peak MiB", "% of peak", "allocated MiB",
                                  "allocations"], r1, raw=True, num_cols={2, 3, 4, 5}, caption="Live at the heap peak")
                         + table(["allocation site", "location", "allocations", "before (same site)", "MiB",
                                  "before"], r2, raw=True, num_cols={2, 3, 4, 5}, caption="Churn (allocation count)")
                         + (table(["allocation site (before)", "location", "allocations", "MiB"],
                                  [[code(s["site"]), code(s["loc"]), fmt(s["tbk"], 0), fmt(s["tb"] / 2**20, 1)]
                                   for s in da["top_churn_blocks"][:8]], raw=True, num_cols={2, 3},
                                  caption="Churn before (2026-09-30): the sites of the old code") if da else "")))
        body.append("<h3>DHAT: what is live at the peak, churn by site (after)</h3>" + tabset("dhat", tabs))
        # heaptrack after
        for h in self.AS.heaptrack:
            pass
        ht = []
        for ds in ("lung_2M", "prime5k_2M", "lung_all"):
            j = self.AS.summary and next((r for r in self.AS.summary.get("heaptrack", []) if r["dataset"] == ds), None)
            jbf = self.BS.summary and next((r for r in self.BS.summary.get("heaptrack", []) if r["dataset"] == ds), None)
            if not j:
                continue
            bsite = {x["site"]: x for x in (jbf or {}).get("heaptrack_top_peak", [])}
            rows = [[code(x["site"])[:400], fmt(x.get("peak_bytes", 0) / 2**20, 1),
                     fmt(bsite.get(x["site"], {}).get("peak_bytes") and bsite[x["site"]]["peak_bytes"] / 2**20, 1),
                     fmt(100 * x.get("peak_bytes", 0) / j["heaptrack_peak_heap"], 1) if j.get("heaptrack_peak_heap") else "—"]
                    for x in j.get("heaptrack_top_peak", [])[:10]]
            ht.append((ds, f"<p class='cap'>peak heap {j.get('heaptrack_peak_heap', 0) / 2**30:.2f} GiB "
                           f"(before {((jbf or {}).get('heaptrack_peak_heap') or 0) / 2**30:.2f}), "
                           f"{(j.get('heaptrack_calls') or 0) / 1e6:.0f} M allocations "
                           f"(before {((jbf or {}).get('heaptrack_calls') or 0) / 1e6:.0f} M)</p>"
                       + table(["live at the heap peak (innermost user frames)", "MiB", "before (same site)",
                                "% of peak"], rows, raw=True, num_cols={1, 2, 3})))
        if ht:
            body.append("<h3>heaptrack at real sizes (after)</h3>" + tabset("ht", ht))
        rows = []
        for t in (1, 2, 4, 8, 16):
            a, b = B.native_row(REP, t), A.native_row(REP, t)
            if a or b:
                rows.append([t, fmt(a and a["peak_rss_kb"] / 1024, 1), fmt(b and b["peak_rss_kb"] / 1024, 1)])
        body.append(table(["threads", "peak RSS MiB before (pancreas 20k)", "after"], rows, num_cols={0, 1, 2}))
        return self.section("memory", "6 · Memory", "\n".join(body))

    # -- 7. bottlenecks ----------------------------------------------------
    def bottlenecks(self):
        body = [self.note("bottlenecks"), self.note("next_steps")]
        tabs = []
        for slide, thr, lab in (("lung", 8, "lung, 8 thr (largest rung: whole slide)"),
                                ("prime5k", 8, "prime5k, 8 thr (largest rung: 8M)"),
                                ("lung", 1, "lung, 1 thr (largest rung: 2M)")):
            for kind in ("fn_excl", "fn_incl"):
                after = [f for f in self.AS.fits if f["slide"] == slide and f["threads"] == thr
                         and f["what"] == kind and f["tool"] == "gperf"]
                before = {f["name"]: f for f in self.BS.fits if f["slide"] == slide and f["threads"] == thr
                          and f["what"] == kind and f["tool"] == "gperf"}
                after = [f for f in after if not NOISE_FN.match(f["name"])]
                after.sort(key=lambda f: -(f["share_last"] or 0))
                rows = []
                for f in after[:15]:
                    b = before.get(f["name"])
                    rows.append([code(f["name"]), fmt(100 * f["share_last"], 1),
                                 fmt(b and 100 * b["share_last"], 1), fmt(100 * f["share_first"], 1),
                                 fmt(f["exponent"], 2), fmt(b and b["exponent"], 2),
                                 '<span class="flag">super-linear</span>' if f.get("superlinear") is True else ""])
                tabs.append((f"{lab}, {'exclusive' if kind == 'fn_excl' else 'inclusive'}",
                             table(["function", "% CPU at largest rung after", "before", "% CPU at smallest rung after",
                                    "exponent after", "before", ""], rows, raw=True, num_cols={1, 2, 3, 4, 5})))
        if any("<tr>" in t[1] for t in tabs):
            body.append("<h3>Supporting data: top functions at real sizes (gperftools CPU share)</h3>"
                        + tabset("hot", tabs))
        return self.section("bottlenecks", "7 · Remaining bottlenecks, ranked, and what to do next", "\n".join(body))

    # -- 8. method ---------------------------------------------------------
    def method(self):
        body = [self.note("method")]
        rows = []
        for lab, T in (("report suite before", self.B), ("report suite after", self.A),
                       ("quick suite before", self.Bq), ("quick suite after", self.Aq)):
            m = T.summary.get("suite") or {}
            ls = [s[1] for s in T.summary.get("load_samples", [])]
            rows.append([esc(lab), esc(m.get("run_id")), esc((m.get("git_sha") or "")[:10]),
                         esc((m.get("baysor_sha256") or "")[:16]), esc(m.get("started")),
                         fmt(m.get("suite_wall_s") and m["suite_wall_s"] / 60, 1),
                         fmt(min(ls) if ls else None, 1), fmt(sorted(ls)[len(ls) // 2] if ls else None, 1),
                         fmt(max(ls) if ls else None, 1), len(T.jobs())])
        for lab, S in (("scaling before", self.BS), ("scaling after", self.AS)):
            ls = []
            st = S.summary
            rows.append([esc(lab), esc(Path(st.get("run_dir", "")).name), "", "", "", "",
                         fmt(min((_load0(j) for j in S.jobs if _load0(j) is not None), default=None), 1), "",
                         fmt(max((_load0(j) for j in S.jobs if _load0(j) is not None), default=None), 1), len(S.jobs)])
        body.append(table(["run", "run id", "git sha", "binary sha256", "started", "wall min", "load 1m min",
                           "median", "max", "jobs"], rows, raw=True, num_cols=range(5, 10),
                          caption="Runs (scaling rows: load at job starts)"))
        # load charts
        ss = []
        for i, (lab, T) in enumerate((("report before", self.B), ("report after", self.A),
                                      ("quick before", self.Bq), ("quick after", self.Aq))):
            smp = T.summary.get("load_samples", [])
            if not smp:
                continue
            t0 = smp[0][0]
            ss.append({"label": lab, "xs": [(s[0] - t0) / 60 for s in smp], "ys": [s[1] for s in smp],
                       "color": SLOTS[i // 2], "before": i % 2 == 0, "xfmt": lambda v: f"{v:.0f} min"})
        if ss:
            body.append(chart_lines(ss, "1-minute load average during the suites", "minutes since start", "load",
                                    figsize=(8, 3)))
        # reproducibility
        rows = []
        for lab, T in (("before", self.B), ("after", self.A), ("quick before", self.Bq), ("quick after", self.Aq)):
            for ds in (REP, "xenium_pancreas_g377_10k"):
                a, b = T.total_ir(f"callgrind-{ds}-t1"), T.total_ir(f"callgrind-{ds}-t1-rerun")
                if a and b:
                    rows.append([esc(lab), code(ds), f"{a:,}", f"{b:,}", f"{b - a:+,}", f"{100 * (b - a) / a:+.4f} %"])
        # cross-suite (report vs quick, after)
        for lab, X, Y in (("before", self.B, self.Bq), ("after", self.A, self.Aq)):
            com = [j for j in X.jobs() if j in Y.jobs() and j.startswith("callgrind-") and j.endswith("-t1")]
            if com:
                mx = max(com, key=lambda j: abs(Y.total_ir(j) / X.total_ir(j) - 1))
                d = 100 * (Y.total_ir(mx) / X.total_ir(mx) - 1)
                rows.append([esc(f"{lab}: report vs quick suite"), esc(f"{len(com)} shared 1-thread jobs"), "", "",
                             esc(f"largest: {mx}"), f"{d:+.4f} %"])
        body.append(table(["run", "dataset", "Ir", "Ir re-run", "difference", "%"], rows, raw=True,
                          num_cols={2, 3, 4, 5}, caption="Reproducibility of instruction counts"))
        # links
        links = []
        for p in sorted(self.out.parent.glob("*")):
            if p.name in (self.out.name,):
                continue
            links.append(f'<li><a href="{esc(self.rel(p))}">{esc(p.name)}</a>{"/" if p.is_dir() else ""}</li>')
        links.append(f'<li><a href="{esc(self.rel(self.bdir / "REPORT.md"))}">before: {esc(self.bdir.name)}/REPORT.md</a></li>')
        runs = self.out.parent.parent.parent / "runs"
        for rid in sorted({(T.summary.get('suite') or {}).get('run_id') for T in (self.A, self.Aq, self.B, self.Bq)} - {None}):
            links.append(f'<li><a href="{esc(self.rel(runs / rid / "raw"))}">raw: runs/{esc(rid)}/raw/</a></li>')
        for S in (self.AS, self.BS):
            rd = S.summary.get("run_dir")
            if rd:
                links.append(f'<li><a href="{esc(self.rel(Path(rd) / "raw"))}">raw: runs/{esc(Path(rd).name)}/raw/</a></li>')
        links.append(f'<li><a href="{GITHUB}/tree/main/profiling">profiling suite on GitHub</a></li>')
        body.append("<h3>Files</h3><ul class='files'>" + "".join(links) + "</ul>")
        return self.section("method", "8 · Method, load and reproducibility", "\n".join(body))

    # -- page --------------------------------------------------------------
    def render(self) -> str:
        parts = [self.exec_summary(), self.where_time(), self.areas(), self.threads(), self.scaling(),
                 self.memory(), self.bottlenecks(), self.method()]
        nav = "".join(f'<a href="#{sid}">{esc(t)}</a>' for sid, t in self.toc)
        return PAGE.replace("{{HEADER}}", self.header()).replace("{{NAV}}", nav).replace(
            "{{BODY}}", "\n".join(parts))


def _load0(r):
    if not r:
        return None
    v = r.get("loadavg_start")
    if isinstance(v, str):
        try:
            v = json.loads(v)
        except ValueError:
            return None
    if isinstance(v, list) and v:
        return v[0]
    return v if isinstance(v, (int, float)) else None


def _loads(r) -> str:
    v = r and r.get("loadavg_1m_at_start")
    if isinstance(v, str):
        try:
            v = json.loads(v)
        except ValueError:
            return esc(v)
    return ", ".join(f"{x:.0f}" for x in v) if isinstance(v, list) else "—"


def _phase_val(c, name):
    for p in c["phases"]:
        if p["phase"] == name:
            return p["Ir"] / 1e9
    return None


def tabset(key, tabs):
    if not tabs:
        return ""
    btn = "".join(f'<button class="tab{" on" if i == 0 else ""}" data-t="{key}{i}">{esc(lab)}</button>'
                  for i, (lab, _) in enumerate(tabs))
    pan = "".join(f'<div class="pane{" on" if i == 0 else ""}" id="{key}{i}">{body}</div>'
                  for i, (_, body) in enumerate(tabs))
    return f'<div class="tabs" role="tablist">{btn}</div>{pan}'


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Baysor profile: before / after optimizations</title>
<style>
:root{color-scheme:light;--surface:#fcfcfb;--card:#ffffff;--ink:#0b0b0b;--ink2:#52514e;--muted:#8a8984;
--line:#e4e3df;--accent:#2a78d6;--good:#008300;--bad:#e34948;--code:#f3f2ef}
*{box-sizing:border-box}
body{margin:0;background:var(--surface);color:var(--ink);font:14px/1.5 Inter,"Helvetica Neue",Arial,sans-serif}
header{padding:28px 40px 8px;max-width:1280px;margin:0 auto}
.kicker{color:var(--accent);font-weight:600;letter-spacing:.04em;text-transform:uppercase;font-size:12px}
h1{font-size:26px;margin:6px 0 8px;line-height:1.2}
.sub{color:var(--ink2);max-width:980px}
nav{position:sticky;top:0;z-index:5;background:rgba(252,252,251,.95);backdrop-filter:blur(4px);
border-bottom:1px solid var(--line);padding:8px 40px;display:flex;gap:18px;flex-wrap:wrap;font-size:13px}
nav a{color:var(--ink2);text-decoration:none}nav a:hover{color:var(--accent)}
main{max-width:1280px;margin:0 auto;padding:0 40px 60px}
section{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:22px 26px;margin:22px 0}
h2{font-size:20px;margin:0 0 12px}h3{font-size:15.5px;margin:22px 0 8px}h4{font-size:14px;margin:16px 0 6px}
p{margin:6px 0 10px;max-width:1050px}
code{font:12px/1.4 "JetBrains Mono",Menlo,Consolas,monospace;background:var(--code);padding:1px 4px;border-radius:4px;
overflow-wrap:anywhere}
td:first-child{min-width:150px}td:first-child code{white-space:normal}
td code{display:inline-block;max-width:520px}
.tiles{display:grid;grid-template-columns:repeat(auto-fill,minmax(205px,1fr));gap:12px;margin:10px 0 6px}
.tile{border:1px solid var(--line);border-radius:10px;padding:12px 14px;background:var(--surface)}
.tl{color:var(--ink2);font-size:12.5px}.tv{font-size:26px;font-weight:650;margin:2px 0}
.tu{font-size:14px;font-weight:500;color:var(--ink2);margin-left:2px}
.tb{font-size:12.5px;color:var(--ink2)}.tn{font-size:11.5px;color:var(--muted);margin-top:2px}
.d{font-weight:600;white-space:nowrap}.d.good{color:var(--good)}.d.bad{color:var(--bad)}.d.flat{color:var(--ink2)}
.tw{overflow-x:auto;margin:8px 0 14px}
table{border-collapse:collapse;font-size:12.5px;min-width:50%}
caption{text-align:left;color:var(--ink2);font-size:12.5px;padding:2px 0 6px;caption-side:top}
th,td{padding:5px 9px;border-bottom:1px solid var(--line);vertical-align:top;text-align:left}
th{font-weight:600;color:var(--ink2);background:#f7f6f3;position:sticky;top:0;white-space:nowrap}
th.n,td.n{text-align:right;font-variant-numeric:tabular-nums}
table.sortable th{cursor:pointer}table.sortable th:hover{color:var(--accent)}
th.asc::after{content:" ▲";font-size:9px}th.desc::after{content:" ▼";font-size:9px}
tbody tr:hover{background:#f7f6f3}
.flag{background:#fdeceb;color:#a32b2a;border-radius:4px;padding:1px 6px;font-size:11.5px;white-space:nowrap}
.flag.old{background:#eef5ee;color:#1f6b1f}
figure.chart{margin:8px 0 14px}figure.chart svg{display:block}
.hov:hover{opacity:.82}
.grid3{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:8px 18px}
.tabs{display:flex;gap:6px;flex-wrap:wrap;margin:8px 0}
.tab{border:1px solid var(--line);background:var(--surface);border-radius:999px;padding:4px 12px;cursor:pointer;
font:inherit;font-size:12.5px;color:var(--ink2)}
.tab.on{background:var(--accent);border-color:var(--accent);color:#fff}
.pane{display:none}.pane.on{display:block}
.note{border-left:3px solid var(--accent);padding:2px 0 2px 14px;margin:8px 0 14px}
.note ul,.note ol{margin:4px 0 10px;padding-left:22px;max-width:1050px}.note li{margin:3px 0}
.cap{color:var(--ink2);font-size:12.5px}
ul.files{columns:2;font-size:13px}
a{color:var(--accent)}
footer{color:var(--muted);font-size:12px;text-align:center;padding:10px 0 30px}
@media print{nav{display:none}section{break-inside:avoid-page;border:none}.pane{display:block}}
</style></head>
<body>
{{HEADER}}
<nav>{{NAV}}</nav>
<main>
{{BODY}}
</main>
<footer>Self-contained report: no external resources. Instruction counts (callgrind) and heap numbers (DHAT,
heaptrack) are load-independent; wall time is load-sensitive (load averages are listed next to it).</footer>
<script>
document.querySelectorAll('.tab').forEach(function(b){b.addEventListener('click',function(){
 var box=b.parentNode, id=b.getAttribute('data-t');
 box.querySelectorAll('.tab').forEach(function(x){x.classList.toggle('on',x===b)});
 var n=box.nextElementSibling;
 while(n&&n.classList.contains('pane')){n.classList.toggle('on',n.id===id);n=n.nextElementSibling}
})});
function key(td){var t=td.textContent.trim().replace(/[,\\s%×]/g,'').replace('−','-');
 var v=parseFloat(t);return isNaN(v)||!/^[-+]?[\\d.]/.test(t)?td.textContent.trim().toLowerCase():v}
document.querySelectorAll('table.sortable').forEach(function(tb){
 tb.querySelectorAll('th').forEach(function(th,i){th.addEventListener('click',function(){
  var asc=!th.classList.contains('asc');
  tb.querySelectorAll('th').forEach(function(x){x.classList.remove('asc','desc')});
  th.classList.add(asc?'asc':'desc');
  var body=tb.tBodies[0],rows=Array.prototype.slice.call(body.rows);
  rows.sort(function(a,b){var x=key(a.cells[i]),y=key(b.cells[i]);
   if(typeof x!==typeof y){return (typeof x==='number'?-1:1)*(asc?1:-1)}
   return (x<y?-1:x>y?1:0)*(asc?1:-1)});
  rows.forEach(function(r){body.appendChild(r)});
 })})});
</script>
</body></html>
"""


def check_offline(page: str) -> list:
    """External URLs in src=/href= other than plain GitHub links."""
    bad = []
    for m in re.finditer(r'(src|href)\s*=\s*"([^"]+)"', page):
        url = m.group(2)
        if re.match(r"^(https?:)?//", url) and not re.match(r"^https://github\.com/", url):
            bad.append(url)
        if m.group(1) == "src" and re.match(r"^(https?:)?//", url):
            bad.append(url)
    for m in re.finditer(r"@import|url\((?!#)", page):
        bad.append(m.group(0))
    return bad


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--before", required=True, help="report directory of the earlier commit")
    ap.add_argument("--after", required=True, help="report directory of the later commit")
    ap.add_argument("--notes", default=None, help="narrative (Markdown subset with section markers); "
                                                  "default <after>/notes.md if present")
    ap.add_argument("--out", default=None, help="default <after>/report.html")
    ap.add_argument("--extra", default=None, metavar="LABEL=DIR",
                    help="intermediate report directory shown next to before/after in the NCV area")
    ap.add_argument("--md", default=None, help="also write the headline numbers as Markdown "
                                               "(e.g. <after>/REPORT.md)")
    args = ap.parse_args(argv)
    bdir, adir = Path(args.before).resolve(), Path(args.after).resolve()
    out = Path(args.out).resolve() if args.out else adir / "report.html"
    notes_path = args.notes or (adir / "notes.md")
    extra = None
    if args.extra:
        lab, _, d = args.extra.partition("=")
        extra = (lab, Path(d).resolve())
    rep = Report(bdir, adir, load_notes(notes_path), out, extra)
    page = rep.render()
    bad = check_offline(page)
    if bad:
        print(f"external references: {bad[:5]}", file=sys.stderr)
        return 1
    out.write_text(page)
    print(f"wrote {out} ({len(page) / 1024:.0f} KiB)")
    if args.md:
        Path(args.md).write_text(rep.markdown())
        print(f"wrote {args.md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
