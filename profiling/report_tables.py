#!/usr/bin/env python3
"""Render markdown tables for REPORT.md from a summarized profiling run.

    python profiling/report_tables.py <run_dir> [--rep DATASET] [-o tables.md]

Every number in the tables comes from <run_dir>/summary/ (summary.json,
callgrind/*.json, dhat/*.json); the output is a building block for the
hand-written report, not the report itself.
"""
from __future__ import annotations

import argparse
import math
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import profcommon as common  # noqa: E402

PHASE_ORDER = ["startup", "loading", "prior", "confidence", "molecule_graph", "mol_clustering",
               "bmm_init", "bmm_iterations", "ncv_colors", "out_molecules",
               "glue_after_out_molecules", "out_cell_stats", "polygons", "out_polygons",
               "glue_after_out_polygons", "out_counts", "report", "shutdown",
               "omp_worker_runtime", "other_threads"]


def G(v):
    return "" if v is None else f"{v / 1e9:.2f}"


def MB(v):
    return "" if v is None else f"{v / 2**20:.1f}"


def pct(v, digits=1):
    return "" if v is None else f"{v:.{digits}f}"


def table(headers, rows) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        out.append("| " + " | ".join("" if c is None else str(c) for c in r) + " |")
    return "\n".join(out) + "\n"


def slope(xs, ys):
    """Least-squares slope of log(y) on log(x)."""
    pts = [(math.log(x), math.log(y)) for x, y in zip(xs, ys) if x and y and x > 0 and y > 0]
    if len(pts) < 2:
        return None
    mx = sum(p[0] for p in pts) / len(pts)
    my = sum(p[1] for p in pts) / len(pts)
    den = sum((p[0] - mx) ** 2 for p in pts)
    return sum((p[0] - mx) * (p[1] - my) for p in pts) / den if den else None


class Run:
    def __init__(self, run_dir: Path):
        self.dir = run_dir
        self.sdir = run_dir / "summary"
        self.s = common.read_json(self.sdir / "summary.json")
        self.jobs = self.s["jobs"]
        self._cg = {}
        self._dh = {}

    def cg(self, jid):
        if jid not in self._cg:
            p = self.sdir / "callgrind" / f"{jid}.json"
            self._cg[jid] = common.read_json(p) if p.is_file() else None
        return self._cg[jid]

    def dh(self, jid):
        if jid not in self._dh:
            p = self.sdir / "dhat" / f"{jid}.json"
            self._dh[jid] = common.read_json(p) if p.is_file() else None
        return self._dh[jid]

    def ids(self, tool, threads=None, plot=False, tag=""):
        out = []
        for jid, j in self.jobs.items():
            if j["tool"] != tool or bool(j.get("plot")) != plot or (j.get("tag") or "") != tag:
                continue
            if threads is not None and j["threads"] != threads:
                continue
            out.append(jid)
        return sorted(out, key=lambda i: (self.jobs[i].get("genes_loaded") or 0,
                                          self.jobs[i].get("molecules_loaded") or 0))


def section_method(run: Run) -> str:
    su = run.s["suite"]
    loads = [s[1] for s in run.s.get("load_samples", []) if len(s) > 1]
    txt = ["### Run\n"]
    rows = [("run id", su.get("run_id")), ("git sha", su.get("git_sha")),
            ("binary sha256", su.get("baysor_sha256")),
            ("host", f"{su.get('cpu')}, {su.get('nproc')} CPUs, "
                     f"{(su.get('mem_total_kb') or 0) / 2**20:.0f} GB RAM"),
            ("kernel / glibc", f"{su.get('kernel')} / {su.get('glibc')}"),
            ("valgrind", su.get("valgrind")), ("compiler", su.get("compiler")),
            ("build flags", ", ".join(f"{k}={v}" for k, v in (su.get("build") or {}).items()
                                      if "FLAGS" in k or k == "CMAKE_BUILD_TYPE")),
            ("suite wall", f"{(su.get('suite_wall_s') or 0) / 60:.1f} min "
                           f"({su.get('started')} .. {su.get('finished')})"),
            ("load average (1 min) during the suite",
             f"min {min(loads):.1f} / median {statistics.median(loads):.1f} / max {max(loads):.1f}"
             f" ({len(loads)} samples, 30 s apart)" if loads else "n/a"),
            ("max concurrent heavy processes", su.get("jobs_parallel"))]
    txt.append(table(["item", "value"], rows))
    # datasets
    seen = {}
    for jid, j in run.jobs.items():
        d = j["dataset"]
        if d not in seen or (j.get("genes_loaded") and not seen[d].get("genes_loaded")):
            seen[d] = j
    rows = []
    for d, j in sorted(seen.items(), key=lambda kv: (kv[1].get("genes_loaded") or 0)):
        rows.append((f"`{d}`", j.get("source"), j.get("n_molecules"), j.get("molecules_loaded"),
                     j.get("n_genes_panel"), j.get("genes_loaded"),
                     "3D" if j.get("is_3d") else "2D", j.get("prior")))
    txt.append("\n### Datasets\n")
    txt.append(table(["dataset", "source", "molecules (crop)", "molecules (after Baysor filters)",
                      "genes in crop", "genes used by Baysor", "dim", "prior"], rows))
    return "\n".join(txt)


def section_phases(run: Run) -> str:
    ids = run.ids("callgrind", threads=1)
    phases = []
    data = {}
    for jid in ids:
        c = run.cg(jid)
        if not c:
            continue
        data[jid] = {p["phase"]: p for p in c["phases"]}
        for p in c["phases"]:
            if p["phase"] not in phases:
                phases.append(p["phase"])
    phases.sort(key=lambda p: PHASE_ORDER.index(p) if p in PHASE_ORDER else 99)
    # drop phases < 0.5% everywhere into "other"
    major = [p for p in phases if any(data[j].get(p, {}).get("pct", 0) >= 0.5 for j in data)]
    rows = []
    for jid in data:
        j = run.jobs[jid]
        c = run.cg(jid)
        other = sum(v["pct"] for k, v in data[jid].items() if k not in major)
        rows.append([f"`{j['dataset']}`", j.get("molecules_loaded"), j.get("genes_loaded"),
                     G(c["total_Ir"])] + [pct(data[jid].get(p, {}).get("pct")) for p in major]
                    + [pct(other)])
    return table(["dataset", "mols", "genes", "total Ir (G)"] + major + ["other"], rows)


def section_top_functions(run: Run, jid: str, n=20, kind="top_self") -> str:
    c = run.cg(jid)
    rows = []
    for r in c[kind][:n]:
        rows.append((f"`{r['name']}`", r.get("hot_line") or r.get("file"), pct(r["self_pct"], 2),
                     pct(r["incl_pct"], 2), r["calls"],
                     " < ".join(r.get("hot_path", [])[:4])))
    return table(["function", "hottest line", "excl %", "incl %", "calls", "dominant call path"], rows)


def section_cross_functions(run: Run, n=12) -> str:
    """Function x dataset matrix of exclusive % for the union of top functions."""
    ids = run.ids("callgrind", threads=1)
    names = []
    per = {}
    for jid in ids:
        c = run.cg(jid)
        if not c:
            continue
        per[jid] = {}
        for r in c["top_self"]:
            per[jid][r["name"]] = per[jid].get(r["name"], 0) + r["self_pct"]
        for r in c["top_self"][:n]:
            if r["name"] not in names:
                names.append(r["name"])
    names.sort(key=lambda nm: -max(per[j].get(nm, 0) for j in per))
    rows = []
    for nm in names[:30]:
        rows.append([f"`{nm}`"] + [pct(per[j].get(nm), 1) if per[j].get(nm) else "·" for j in per])
    hdr = ["function (excl % Ir)"] + [run.jobs[j]["dataset"].replace("_20k", "").replace("xenium_", "x_")
                                      for j in per]
    return table(hdr, rows)


def section_lines(run: Run, jid: str, n_fn=8, n_lines=4) -> str:
    c = run.cg(jid)
    rows = []
    fns = []
    for r in c["top_lines"]:
        if r["fn"] not in fns:
            fns.append(r["fn"])
    for fn in fns[:n_fn]:
        for r in [x for x in c["top_lines"] if x["fn"] == fn][:n_lines]:
            rows.append((f"`{fn}`", f"{r['file']}:{r['line']}", pct(r["pct_of_total"], 2),
                         pct(r["pct_of_fn"], 1)))
    return table(["function", "source line", "% of total Ir", "% of function"], rows)


def section_scaling(run: Run, datasets: list[str], label: str, xkey: str) -> str:
    ids = [f"callgrind-{d}-t1" for d in datasets if f"callgrind-{d}-t1" in run.jobs]
    if len(ids) < 2:
        return f"(not enough {label} points)\n"
    xs = [run.jobs[j].get(xkey) for j in ids]
    phases = []
    for jid in ids:
        for p in run.cg(jid)["phases"]:
            if p["phase"] not in phases:
                phases.append(p["phase"])
    phases.sort(key=lambda p: PHASE_ORDER.index(p) if p in PHASE_ORDER else 99)
    rows = []
    tot = [run.cg(j)["total_Ir"] for j in ids]
    rows.append(["**total**"] + [G(v) for v in tot] + [pct(slope(xs, tot), 2)])
    for ph in phases:
        vals = [next((p["Ir"] for p in run.cg(j)["phases"] if p["phase"] == ph), 0) for j in ids]
        if max(vals) < 0.005 * max(tot):
            continue
        rows.append([ph] + [G(v) for v in vals] + [pct(slope(xs, vals), 2)])
    hdr = ["phase (G Ir)"] + [f"{run.jobs[j]['dataset']} ({xkey.split('_')[0]}={x})"
                              for j, x in zip(ids, xs)] + [f"exponent vs {xkey.split('_')[0]}"]
    return table(hdr, rows)


def section_cache(run: Run) -> str:
    ids = run.ids("cachegrind", threads=1)
    if not ids:
        return "(no cache-simulation job)\n"
    c = run.cg(ids[0])
    t = c["totals"]
    d_acc = t.get("Dr", 0) + t.get("Dw", 0)
    d1 = t.get("D1mr", 0) + t.get("D1mw", 0)
    ll = t.get("DLmr", 0) + t.get("DLmw", 0)
    txt = [f"Dataset `{run.jobs[ids[0]]['dataset']}`; simulated caches: "
           + "; ".join(c.get("cache_desc", [])) + "\n",
           f"Totals: Ir {G(t.get('Ir'))} G, data accesses {G(d_acc)} G, D1 misses {G(d1)} G "
           f"({100 * d1 / d_acc:.2f} % of accesses), LL data misses {ll / 1e6:.1f} M "
           f"({100 * ll / d_acc:.3f} % of accesses), LL instruction misses {t.get('ILmr', 0) / 1e6:.2f} M.\n"]
    rows = []
    for r in c["cache_top_LL"][:15]:
        rows.append((f"`{r['name']}`", pct(r["self_pct"], 2), f"{r['self_LLmiss'] / 1e6:.2f}",
                     pct(100 * r["self_LLmiss"] / ll if ll else None, 1),
                     pct(r["LL_miss_per_kIr"], 3), pct(100 * (r["D1_miss_rate"] or 0), 2),
                     f"{r['self_D1miss'] / 1e6:.1f}"))
    txt.append(table(["function", "excl % Ir", "LL data misses (M)", "% of all LL misses",
                      "LL misses / 1000 Ir", "D1 miss rate %", "D1 misses (M)"], rows))
    return "\n".join(txt)


def section_memory(run: Run) -> str:
    rows = []
    for jid in run.ids("dhat"):
        j = run.jobs[jid]
        nat = run.jobs.get(f"native-{j['dataset']}-t{j['threads']}")
        rows.append((f"`{j['dataset']}`", j["threads"], j.get("molecules_loaded"), j.get("genes_loaded"),
                     MB(j["peak_bytes"]), f"{j['total_bytes'] / 2**30:.2f}",
                     f"{j['total_blocks'] / 1e6:.2f}",
                     MB(nat["peak_rss_kb"] * 1024) if nat else ""))
    return table(["dataset", "threads", "mols", "genes", "peak heap (MiB, DHAT t-gmax)",
                  "total allocated (GiB)", "allocations (M)", "peak RSS (MiB, native)"], rows)


def section_dhat_sites(run: Run, jid: str, key="top_live_at_peak", n=12) -> str:
    d = run.dh(jid)
    rows = []
    for s in d[key][:n]:
        rows.append((f"`{s['site']}`", s["loc"], MB(s["gb"]), pct(100 * s["gb"] / d["peak_bytes"], 1)
                     if d["peak_bytes"] else "", f"{s['tb'] / 2**20:.1f}", f"{s['tbk']:,}"))
    return table(["allocation site (innermost user frames)", "location", "live at peak (MiB)",
                  "% of peak", "allocated in total (MiB)", "allocations"], rows)


def section_threads(run: Run, dataset: str) -> str:
    rows = []
    base = None
    for t in (1, 2, 4, 8, 16):
        jid = f"callgrind-{dataset}-t{t}"
        c = run.cg(jid)
        if not c:
            continue
        if base is None:
            base = c["total_Ir"]
        thr = {int(k): v for k, v in c["threads"].items()}
        main = thr.get(1, 0)
        others = sorted((v for k, v in thr.items() if k != 1), reverse=True)
        rows.append((t, G(c["total_Ir"]), pct(100 * (c["total_Ir"] - base) / base, 2), len(thr),
                     G(main), pct(100 * main / c["total_Ir"], 1),
                     ", ".join(G(v) for v in others[:16])))
    return table(["threads", "total Ir (G)", "vs 1 thread %", "OS threads", "main thread Ir (G)",
                  "main thread %", "other threads Ir (G)"], rows)


def section_amdahl(run: Run, jid: str) -> str:
    c = run.cg(jid)
    rows = []
    for p in c["phases"]:
        if p["pct"] < 0.2:
            continue
        rows.append((p["phase"], pct(p["pct"], 1), G(p["serial_Ir"]), G(p["parallel_Ir"]),
                     pct(100 * p["serial_frac"], 1), p["amdahl"]["4"], p["amdahl"]["8"],
                     p["amdahl"]["16"]))
    rows.append(("**whole run**", "100", G(c["serial_Ir"]), G(c["parallel_Ir"]),
                 pct(100 * c["serial_frac"], 1), c["amdahl"]["4"], c["amdahl"]["8"], c["amdahl"]["16"]))
    return table(["phase", "% Ir", "serial Ir (G)", "parallel-region Ir (G)", "serial %",
                  "Amdahl 4 thr", "Amdahl 8 thr", "Amdahl 16 thr"], rows)


def section_phase_threads(run: Run, dataset: str, top=8) -> str:
    """Per phase: Ir at 1 thread vs 16 and the max/mean over OS threads."""
    c1 = run.cg(f"callgrind-{dataset}-t1")
    cs = {t: run.cg(f"callgrind-{dataset}-t{t}") for t in (2, 4, 8, 16)}
    rows = []
    for p in c1["phases"]:
        if p["pct"] < 0.5:
            continue
        r = [p["phase"], G(p["Ir"])]
        for t in (2, 4, 8, 16):
            c = cs.get(t)
            q = next((x for x in c["phases"] if x["phase"] == p["phase"]), None) if c else None
            if q is None:
                r.append("")
                continue
            vals = [v for v in q["per_thread"].values() if v > 0]
            r.append(f"{G(q['Ir'])} ({len(vals)} thr, max/mean {q['thread_max_over_mean']})")
        rows.append(r)
    return table(["phase", "1 thr G Ir", "2 thr", "4 thr", "8 thr", "16 thr"], rows)


def section_omp(run: Run, dataset: str, threads=16, n=12) -> str:
    c = run.cg(f"callgrind-{dataset}-t{threads}")
    if not c:
        return ""
    rows = []
    for r in c["omp_regions"][:n]:
        vals = sorted(r["per_thread"].values(), reverse=True)
        rows.append((f"`{r['region']}`", pct(r["pct"], 2), r["calls_main"], r["n_threads_active"],
                     r["max_over_mean"], G(vals[0]) if vals else "", G(vals[-1]) if vals else ""))
    return table(["OpenMP region", "% Ir", "entries (main)", "threads with work",
                  "max/mean over threads", "max thread (G Ir)", "min thread (G Ir)"], rows)


def section_native_threads(run: Run, dataset: str) -> str:
    rows = []
    base = None
    for t in (1, 2, 4, 8, 16):
        j = run.jobs.get(f"native-{dataset}-t{t}")
        if not j:
            continue
        if base is None:
            base = j["wall_s_min"]
        rows.append((t, f"{j['wall_s_min']:.2f}", f"{j['wall_s_median']:.2f}",
                     f"{j['cpu_s_median']:.2f}", f"{base / j['wall_s_min']:.2f}",
                     f"{j['cpu_per_wall_median']:.2f}", MB(j["peak_rss_kb"] * 1024),
                     ", ".join(f"{v:.1f}" for v in j["loadavg_1m_at_start"])))
    j = run.jobs.get(f"native-{dataset}-t8-passive")
    if j:
        rows.append(("8 (passive)", f"{j['wall_s_min']:.2f}", f"{j['wall_s_median']:.2f}",
                     f"{j['cpu_s_median']:.2f}", f"{base / j['wall_s_min']:.2f}",
                     f"{j['cpu_per_wall_median']:.2f}", MB(j["peak_rss_kb"] * 1024),
                     ", ".join(f"{v:.1f}" for v in j["loadavg_1m_at_start"])))
    return table(["threads", "wall min (s)", "wall median (s)", "CPU median (s)", "speedup (min wall)",
                  "CPU/wall", "peak RSS (MiB)", "load avg 1 min at start of each rep"], rows)


def section_native_all(run: Run) -> str:
    rows = []
    for jid in run.ids("native", threads=1):
        j = run.jobs[jid]
        rows.append((f"`{j['dataset']}`", j.get("molecules_loaded"), j.get("genes_loaded"), j["n_reps"],
                     f"{j['wall_s_min']:.2f}", f"{j['wall_s_median']:.2f}", f"{j['cpu_s_median']:.2f}",
                     MB(j["peak_rss_kb"] * 1024), pct(j.get("loadavg_1m_mean"), 1)))
    return table(["dataset", "mols", "genes", "reps", "wall min (s)", "wall median (s)", "CPU median (s)",
                  "peak RSS (MiB)", "load avg"], rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir")
    ap.add_argument("--rep", default="xenium_pancreas_g377_20k", help="representative dataset")
    ap.add_argument("-o", "--out", default=None, help="default: <run_dir>/summary/tables.md")
    args = ap.parse_args(argv)
    run = Run(Path(args.run_dir).resolve())
    rep = args.rep
    parts = ["# Tables (generated by report_tables.py)\n", section_method(run),
             "\n## Phase breakdown, % of Ir (1 thread)\n", section_phases(run),
             f"\n## Top functions by exclusive Ir: {rep}\n",
             section_top_functions(run, f"callgrind-{rep}-t1"),
             f"\n## Top functions by inclusive Ir: {rep}\n",
             section_top_functions(run, f"callgrind-{rep}-t1", kind="top_incl"),
             "\n## Exclusive % Ir across datasets\n", section_cross_functions(run),
             f"\n## Hot lines: {rep}\n", section_lines(run, f"callgrind-{rep}-t1"),
             "\n## Molecule scaling (Xenium pancreas crops)\n",
             section_scaling(run, ["xenium_pancreas_g377_10k", "xenium_pancreas_g377_20k",
                                   "xenium_pancreas_g377_40k"], "molecules", "molecules_loaded"),
             "\n## Gene scaling, simulated (same geometry)\n",
             section_scaling(run, ["sim_circles_g100_20k", "sim_circles_g1000_20k",
                                   "sim_circles_g5000_10k"], "genes", "genes_loaded"),
             "\n## Cache simulation\n", section_cache(run),
             "\n## Memory\n", section_memory(run),
             "\n## Native 1-thread runs (load-sensitive except RSS)\n", section_native_all(run)]
    for jid in run.ids("dhat", threads=1):
        parts += [f"\n### Live at peak: {run.jobs[jid]['dataset']}\n", section_dhat_sites(run, jid)]
    for jid in run.ids("dhat", threads=1):
        parts += [f"\n### Churn (bytes): {run.jobs[jid]['dataset']}\n",
                  section_dhat_sites(run, jid, key="top_churn_bytes")]
    parts += [f"\n## Threads (callgrind): {rep}\n", section_threads(run, rep),
              f"\n## Per phase vs threads: {rep}\n", section_phase_threads(run, rep),
              f"\n## Serial fraction and Amdahl bound: {rep}\n", section_amdahl(run, f"callgrind-{rep}-t1"),
              f"\n## OpenMP regions at 16 threads: {rep}\n", section_omp(run, rep),
              f"\n## Native thread series: {rep}\n", section_native_threads(run, rep)]
    for other in ("xenium_prime5k_20k",):
        if f"native-{other}-t2" in run.jobs:
            parts += [f"\n## Native thread series: {other}\n", section_native_threads(run, other)]
    out = Path(args.out) if args.out else run.sdir / "tables.md"
    out.write_text("\n".join(parts))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
