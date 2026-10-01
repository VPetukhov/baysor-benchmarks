#!/usr/bin/env python3
"""Scaling tier: Baysor at real data sizes with low-overhead measurements.

    BENCH=/path/to/baysor-benchmarks          # this repository
    PY=$BENCH/.deps/bench/bin/python
    $PY profiling/scaling_crops.py            # ladders (once)
    $PY profiling/scaling.py plan             # job list + estimates
    $PY profiling/scaling.py run [--run-id ID] [--max-procs 8] [--cores 16] \
        --baysor /path/to/baysor/build/profiling/baysor
    $PY profiling/scaling.py summarize <run_dir>

Jobs (profiling.yaml `scaling.jobs`), one Baysor run each, resumable (a job
with job.json status ok is skipped):

  gperf      native run with the gperftools CPU profiler preloaded
             (ITIMER_PROF samples = CPU time, libunwind stacks) and /proc
             sampling every second (CPU seconds, RSS, per phase via the log)
  heaptrack  heaptrack run: peak heap, live-at-peak and allocation counts by
             call stack (collapsed stacks from heaptrack_print)
  callgrind  same as the quick tier, on the lowest rungs: cross-check of the
             sampled CPU shares against instruction counts

The scheduler keeps at most --max-procs heavy processes and --cores
threads busy (a T-thread job counts T cores), longest job first.
`summarize` fits per slide and thread count the exponent of CPU seconds and
peak RSS vs molecules (log-log least squares) for the whole run, every phase
and the top functions, flags super-linear parts (exponent >= 1.15) and
growing shares, and writes plots.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.append(str(HERE.parent / "harness"))   # ../harness (the benchmark harness)
import gperf  # noqa: E402
import procmon  # noqa: E402
import profcommon as common  # noqa: E402

SUPERLINEAR = 1.15


@dataclass
class SJob:
    tool: str
    dataset: str
    threads: int
    molecules: int
    est_s: float = 0.0

    @property
    def id(self) -> str:
        return f"{self.tool}-{self.dataset}-t{self.threads}"


def scaling_datasets(root: Path, spec: dict) -> dict:
    """id -> (dataset dir, meta) for ladder rungs and extra datasets."""
    out = {}
    base = root / "profiling" / "scaling" / "data"
    for d in sorted(base.glob("*")) if base.is_dir() else []:
        if (d / "meta.json").is_file():
            out[d.name] = (d, common.read_json(d / "meta.json"))
    for ds in spec.get("extra_datasets", []):
        p = common.find_source_dataset(root, ds)
        out[ds] = (p, common.read_json(p / "meta.json"))
    return out


def expand_jobs(spec: dict, dsets: dict) -> list[SJob]:
    jobs = []
    js = spec["jobs"]
    for tool in ("gperf", "heaptrack", "callgrind"):
        for entry in js.get(tool, []) or []:
            names = []
            if "slides" in entry:
                for name, (_, meta) in dsets.items():
                    sl = (meta.get("scaling") or {}).get("slide")
                    if sl in entry["slides"]:
                        n = meta["stats"]["n_molecules"]
                        if n <= int(entry.get("max_molecules", 10**12)):
                            names.append(name)
            if entry.get("extra"):
                names += list(spec.get("extra_datasets", []))
            names += entry.get("datasets", [])
            for name in names:
                if name not in dsets:
                    print(f"warning: dataset {name} not found (run scaling_crops.py)")
                    continue
                n = dsets[name][1]["stats"]["n_molecules"]
                for t in entry["threads"]:
                    jobs.append(SJob(tool, name, int(t), n))
    # estimated wall seconds (ordering only): CPU-s per molecule from the
    # benchmark baselines, parallel efficiency ~50 %, tool overheads
    for j in jobs:
        genes = dsets[j.dataset][1]["stats"].get("n_genes", 400)
        cpu = j.molecules * (0.7e-3 if genes < 1000 else (1.1e-3 if genes < 10000 else 3.0e-3))
        eff = 1.0 if j.threads == 1 else 0.5 * j.threads
        factor = {"gperf": 1.05, "heaptrack": 1.5, "callgrind": 25.0}[j.tool]
        j.est_s = cpu * factor / eff
    seen, out = set(), []
    for j in jobs:
        if j.id not in seen:
            seen.add(j.id)
            out.append(j)
    return out


# ---------------------------------------------------------------------------
# job execution
# ---------------------------------------------------------------------------

def proftools_prefix(cli: str | None) -> Path:
    """gperftools/heaptrack prefix: --proftools, $PROFTOOLS, else <repo>/.deps/proftools."""
    found = common.find_proftools(cli)
    if found is None:
        raise SystemExit("gperftools/heaptrack prefix not found: pass --proftools or set PROFTOOLS")
    return found


def base_env(threads: int) -> dict:
    env = os.environ.copy()
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                "BAYSOR_NUM_THREADS"):
        env[var] = str(threads)
    env.pop("OMP_WAIT_POLICY", None)
    env.pop("BAYSOR_POOL_SPIN_US", None)   # native jobs: the pool's default spin-then-block
    return env


def run_job(j: SJob, jd: Path, ds_dir: Path, ctx: dict) -> dict:
    hcommon, hrun = ctx["harness"]
    ds = hcommon.load_dataset(ds_dir)
    for old in list(jd.glob("cpu.prof*")) + list(jd.glob("callgrind.out*")) + list(jd.glob("heaptrack.*")):
        old.unlink()
    cmd = hrun.build_command(ctx["baysor"], ds, jd / "seg", ctx["probe"], ctx["repo"],
                             skip_ncv_color=False)
    env = base_env(j.threads)
    interval = 1.0
    if j.tool == "gperf":
        env["LD_PRELOAD"] = str(ctx["proftools"] / "lib" / "libprofiler.so")
        env["CPUPROFILE"] = str(jd / "cpu.prof")
        env["CPUPROFILE_FREQUENCY"] = "100"
        env["TCMALLOC_STACKTRACE_METHOD"] = "libunwind"
        full = cmd
    elif j.tool == "heaptrack":
        full = [str(ctx["proftools"] / "bin" / "heaptrack"), "-o", str(jd / "heaptrack")] + cmd
    elif j.tool == "callgrind":
        import profile as qp
        spec_q = common.load_spec(ctx["spec_path"])
        qjob = qp.Job(tool="callgrind", dataset=j.dataset, threads=j.threads)
        full = qp.valgrind_args(qjob, jd, spec_q, ctx["valgrind"]) + cmd
        env["OMP_WAIT_POLICY"] = "passive"    # as profile.thread_env(valgrind=True)
        env["BAYSOR_POOL_SPIN_US"] = "0"
        interval = 10.0
    else:
        raise ValueError(j.tool)
    res = procmon.run_monitored(full, env, jd / "baysor.log", interval=interval, cwd=str(ctx["repo"]))
    res["phases"] = procmon.phase_table(res, jd / "baysor.log")
    common.write_json(jd / "procmon.json", res)
    shutil.rmtree(jd / "seg", ignore_errors=True)
    out = {"status": "ok" if res["exit_code"] == 0 else "failed", "command": full,
           "env": {k: env.get(k) for k in ("OMP_NUM_THREADS", "OMP_WAIT_POLICY", "LD_PRELOAD",
                                           "CPUPROFILE_FREQUENCY", "TCMALLOC_STACKTRACE_METHOD",
                                           "BAYSOR_NUM_THREADS", "BAYSOR_POOL_SPIN_US")},
           **{k: res[k] for k in ("exit_code", "wall_s", "cpu_s", "cpu_user_s", "cpu_sys_s",
                                  "peak_rss_kb", "loadavg_start", "loadavg_end", "t_start", "t_end")}}
    if out["status"] == "ok" and j.tool == "gperf":
        prof = jd / "cpu.prof"
        g = gperf.summarize(prof, top=80, phases=common.load_spec(ctx["spec_path"])["phases"])
        common.write_json(jd / "gperf.json", g)
        out["gperf_samples"] = g["total_samples"]
    if out["status"] == "ok" and j.tool == "heaptrack":
        out.update(analyze_heaptrack(jd, ctx["proftools"]))
    return out


_HT_SECTIONS = {"MOST CALLS TO ALLOCATION FUNCTIONS": "allocations",
                "PEAK MEMORY CONSUMERS": "peak",
                "MOST TEMPORARY ALLOCATIONS": "temporary"}
_SIZE_RE = r"([\d.]+)([KMGT]?)B?"


def _size(v: str, unit: str) -> int:
    return int(float(v) * {"K": 2**10, "M": 2**20, "G": 2**30, "T": 2**40}.get(unit, 1))


def parse_heaptrack_text(txt: str) -> dict:
    """Entries of `heaptrack_print --merge-backtraces 0` grouped per section.

    Each entry: {"value": calls | peak bytes | temporary allocations, "calls",
    "peak_bytes", "frames": [(function, location)], innermost first}.
    """
    import summarize as qs
    out = {v: [] for v in _HT_SECTIONS.values()}
    section, cur = None, None
    for line in txt.splitlines():
        if line.strip() in _HT_SECTIONS:
            section, cur = _HT_SECTIONS[line.strip()], None
            continue
        if section is None:
            continue
        if not line.strip():
            cur = None
            continue
        if not line.startswith(" "):
            m1 = re.match(r"^(\d+) calls to allocation functions with " + _SIZE_RE + r" peak consumption from", line)
            m2 = re.match(r"^" + _SIZE_RE + r" peak memory consumed over (\d+) calls from", line)
            m3 = re.match(r"^(\d+) temporary allocations of (\d+) allocations in total", line)
            if m1:
                cur = {"calls": int(m1.group(1)), "peak_bytes": _size(m1.group(2), m1.group(3))}
                cur["value"] = cur["calls"]
            elif m2:
                cur = {"peak_bytes": _size(m2.group(1), m2.group(2)), "calls": int(m2.group(3))}
                cur["value"] = cur["peak_bytes"]
            elif m3:
                cur = {"temporary": int(m3.group(1)), "calls": int(m3.group(2))}
                cur["value"] = cur["temporary"]
            else:
                cur = None
                continue
            cur["frames"] = []
            out[section].append(cur)
            continue
        if cur is None:
            continue
        if line.startswith("    at "):
            if cur["frames"] and not cur["frames"][-1][1]:
                cur["frames"][-1] = (cur["frames"][-1][0], line[7:].strip())
        elif line.startswith("  ") and not line.startswith("    "):
            cur["frames"].append((line.strip(), ""))
    for sec, entries in out.items():
        for e in entries:
            names = [f for f, _ in e["frames"]]
            key, idx = qs.site_from_frames(names)
            e["site"] = key
            e["loc"] = Path(e["frames"][idx][1]).name if idx is not None and e["frames"][idx][1] else ""
            del e["frames"]
    return out


def analyze_heaptrack(jd: Path, prefix: Path) -> dict:
    files = sorted(jd.glob("heaptrack.*"))
    trace = next((f for f in files if f.suffix in (".zst", ".gz")), None)
    if trace is None:
        return {"heaptrack_error": "no trace file"}
    hp = str(prefix / "bin" / "heaptrack_print")
    res = {"heaptrack_trace": trace.name}
    t0 = time.time()
    txt = subprocess.run([hp, "-f", str(trace), "--merge-backtraces", "0", "-n", "80",
                          "--print-peaks", "1", "--print-allocators", "1",
                          "--print-temporary", "1", "--print-leaks", "0"],
                         capture_output=True, text=True).stdout
    res["heaptrack_print_s"] = round(time.time() - t0, 1)
    (jd / "heaptrack_print.txt").write_text(txt)
    for key, rx in (("calls", r"calls to allocation functions: (\d+)"),
                    ("temporary", r"temporary memory allocations: (\d+)")):
        m = re.search(rx, txt)
        if m:
            res[f"heaptrack_{key}"] = int(m.group(1))
    for key, rx in (("peak_heap", r"peak heap memory consumption: " + _SIZE_RE),
                    ("peak_rss", r"peak RSS \(including heaptrack overhead\): " + _SIZE_RE),
                    ("leaked", r"total memory leaked: " + _SIZE_RE)):
        m = re.search(rx, txt)
        if m:
            res[f"heaptrack_{key}"] = _size(m.group(1), m.group(2))
    parsed = parse_heaptrack_text(txt)
    for sec, entries in parsed.items():
        agg = defaultdict(lambda: {"value": 0, "calls": 0, "peak_bytes": 0, "loc": ""})
        for e in entries:
            a = agg[e["site"]]
            a["value"] += e["value"]
            a["calls"] += e.get("calls", 0)
            a["peak_bytes"] += e.get("peak_bytes", 0)
            a["loc"] = a["loc"] or e["loc"]
        top = sorted(agg.items(), key=lambda kv: -kv[1]["value"])[:40]
        res[f"heaptrack_top_{sec}"] = [{"site": k, **v} for k, v in top]
    return res


def job_cores(j: SJob, cores: int) -> int:
    """Cores a job occupies: its threads, but heaptrack serializes allocations
    (a whole-slide run uses ~1.5 cores at 8 threads), so it counts as 2."""
    return min(2 if j.tool == "heaptrack" else j.threads, cores)


def schedule(jobs: list[SJob], run_dir: Path, dsets: dict, ctx: dict, max_procs: int,
             cores: int, force: bool) -> None:
    # multi-threaded jobs first (longest first), so that single-thread jobs
    # filling free cores can not starve them; then single-thread jobs
    pending = sorted([j for j in jobs if force or not job_done(run_dir / "raw" / j.id)],
                     key=lambda j: (j.threads == 1, -j.est_s))
    for j in jobs:
        if j not in pending:
            print(f"[skip] {j.id}", flush=True)
    lock = threading.Lock()
    running: dict[str, SJob] = {}
    free = {"cores": cores}

    def worker(j: SJob):
        jd = run_dir / "raw" / j.id
        jd.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        try:
            res = run_job(j, jd, dsets[j.dataset][0], ctx)
        except Exception as exc:  # record and continue
            res = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
        res.update({"job": j.id, "tool": j.tool, "dataset": j.dataset, "threads": j.threads,
                    "molecules": j.molecules, "est_s": round(j.est_s)})
        common.write_json(jd / "job.json", res)
        print(f"[{j.tool}] {j.id}: {res['status']} {time.time() - t0:.0f} s "
              f"(est {j.est_s:.0f} s) {time.strftime('%H:%M:%S')} load {procmon.loadavg()}", flush=True)
        with lock:
            free["cores"] += job_cores(j, cores)
            running.pop(j.id, None)

    threads = []
    while pending or running:
        started = False
        with lock:
            if len(running) < max_procs:
                for j in pending:
                    need = job_cores(j, cores)
                    if need <= free["cores"]:
                        pending.remove(j)
                        free["cores"] -= need
                        running[j.id] = j
                        th = threading.Thread(target=worker, args=(j,), daemon=True)
                        th.start()
                        threads.append(th)
                        print(f"[start] {j.id} (est {j.est_s / 60:.0f} min) running={len(running)} "
                              f"free cores={free['cores']}", flush=True)
                        started = True
                        break
        if not started:
            time.sleep(5)
    for th in threads:
        th.join()


def job_done(jd: Path) -> bool:
    p = jd / "job.json"
    try:
        return p.is_file() and common.read_json(p).get("status") == "ok"
    except (OSError, json.JSONDecodeError):
        return False


# ---------------------------------------------------------------------------
# summarize
# ---------------------------------------------------------------------------

def fit(xs, ys):
    pts = [(math.log(x), math.log(y)) for x, y in zip(xs, ys) if x and y and x > 0 and y > 0]
    if len(pts) < 2:
        return None
    mx = sum(p[0] for p in pts) / len(pts)
    my = sum(p[1] for p in pts) / len(pts)
    den = sum((p[0] - mx) ** 2 for p in pts)
    return round(sum((p[0] - mx) * (p[1] - my) for p in pts) / den, 3) if den else None


def short(fn: str) -> str:
    import summarize as qs
    return qs.short_name(fn)


def summarize_run(run_dir: Path, spec_path: Path, top_fns: int = 15, out: Path | None = None) -> dict:
    import csv
    out = out or run_dir / "summary"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for jd in sorted((run_dir / "raw").iterdir()):
        if not (jd / "job.json").is_file():
            continue
        job = common.read_json(jd / "job.json")
        if job.get("status") != "ok":
            rows.append({**job, "ok": False})
            continue
        pm = common.read_json(jd / "procmon.json")
        m = re.search(r"Loaded (\d+) transcripts, (\d+) genes",
                      (jd / "baysor.log").read_text(errors="replace"))
        rec = {**job, "ok": True, "molecules_loaded": int(m.group(1)) if m else None,
               "genes_loaded": int(m.group(2)) if m else None,
               "phases": pm.get("phases", [])}
        if (jd / "gperf.json").is_file():
            g = common.read_json(jd / "gperf.json")
            rec["gperf"] = {"total_samples": g["total_samples"], "period_us": g["period_us"],
                            "excl": g["excl_all"], "incl": g["incl_all"]}
            if "classes" in g:
                rec["gperf"]["classes"] = g["classes"]
        rows.append(rec)
    good = [r for r in rows if r.get("ok")]
    # slide of each dataset
    def slide_of(ds):
        m = re.match(r"^(.*)_(\d+k|\d+(?:\.\d+)?M|all)$", ds)
        return m.group(1) if m else ds

    series = defaultdict(list)       # (slide, tool, threads) -> rows
    for r in good:
        series[(slide_of(r["dataset"]), r["tool"], r["threads"])].append(r)
    fits = []
    fn_rows = []
    phase_rows = []
    for (slide, tool, thr), lst in sorted(series.items()):
        lst.sort(key=lambda r: r["molecules"])
        xs = [r["molecules_loaded"] or r["molecules"] for r in lst]
        for r in lst:
            for p in r["phases"]:
                phase_rows.append({"slide": slide, "tool": tool, "threads": thr, "dataset": r["dataset"],
                                   "molecules": r["molecules_loaded"] or r["molecules"], **p})
        if len(lst) < 2 or tool in ("callgrind", "heaptrack"):
            continue   # tool overhead distorts CPU/RSS; only native gperf runs are fitted
        base = {"slide": slide, "tool": tool, "threads": thr, "n_points": len(lst),
                "molecules_min": min(xs), "molecules_max": max(xs)}
        fits.append({**base, "what": "total", "name": "cpu_s",
                     "exponent": fit(xs, [r["cpu_s"] for r in lst]),
                     "values": [r["cpu_s"] for r in lst]})
        fits.append({**base, "what": "total", "name": "wall_s",
                     "exponent": fit(xs, [r["wall_s"] for r in lst]),
                     "values": [r["wall_s"] for r in lst]})
        fits.append({**base, "what": "total", "name": "peak_rss_kb",
                     "exponent": fit(xs, [r["peak_rss_kb"] for r in lst]),
                     "values": [r["peak_rss_kb"] for r in lst]})
        # phases: CPU seconds and RSS peak
        names = []
        for r in lst:
            for p in r["phases"]:
                if p["phase"] not in names:
                    names.append(p["phase"])
        for ph in names:
            vals = [next((p["cpu_s"] for p in r["phases"] if p["phase"] == ph), None) for r in lst]
            shares = [v / r["cpu_s"] if v is not None and r["cpu_s"] else None for v, r in zip(vals, lst)]
            e = fit(xs, vals)
            fits.append({**base, "what": "phase_cpu", "name": ph, "exponent": e, "values": vals,
                         "share_first": shares[0], "share_last": shares[-1],
                         "superlinear": e is not None and e >= SUPERLINEAR
                         and (max(v or 0 for v in vals) >= 0.02 * max(r["cpu_s"] for r in lst))})
            rss = [next((p["rss_peak_kb"] for p in r["phases"] if p["phase"] == ph), None) for r in lst]
            fits.append({**base, "what": "phase_rss", "name": ph, "exponent": fit(xs, rss), "values": rss})
        # functions (gperf): CPU seconds = share of samples x total CPU seconds
        if tool == "gperf" and all("gperf" in r for r in lst):
            last = lst[-1]["gperf"]
            cand = sorted(last["incl"], key=lambda f: -last["incl"][f])
            cand_ex = sorted(last["excl"], key=lambda f: -last["excl"][f])
            # also functions that dominate at the smallest rung
            first = lst[0]["gperf"]
            cand_ex += sorted(first["excl"], key=lambda f: -first["excl"][f])[:top_fns]
            for kind, fnl in (("excl", list(dict.fromkeys(cand_ex))[:2 * top_fns]),
                              ("incl", cand[:3 * top_fns])):
                for fn in fnl:
                    if kind == "incl" and re.match(r"^(main|_start|__libc_start|cmd_run|auto cmd_run|"
                                                   r"start_thread|clone|GOMP_parallel|gomp_thread_start)", fn):
                        continue
                    vals, shares = [], []
                    for r in lst:
                        g = r["gperf"]
                        n = g[kind].get(fn, 0)
                        sh = n / g["total_samples"] if g["total_samples"] else 0
                        shares.append(sh)
                        vals.append(sh * r["cpu_s"] if sh else None)
                    e = fit(xs, vals)
                    rec = {**base, "what": f"fn_{kind}", "name": short(fn), "fn": fn, "exponent": e,
                           "values": vals, "shares": shares, "share_first": shares[0],
                           "share_last": shares[-1],
                           "superlinear": e is not None and e >= SUPERLINEAR and max(shares) >= 0.02,
                           "growing_share": shares[-1] >= 1.5 * max(shares[0], 1e-9) and shares[-1] >= 0.03}
                    fits.append(rec)
                    fn_rows.append(rec)
    # callgrind cross-check: callgrind Ir share vs gperf CPU share (same rung, same thread count)
    xcheck = []
    import summarize as qs
    for r in good:
        if r["tool"] != "callgrind":
            continue
        jd = run_dir / "raw" / r["job"]
        spec = common.load_spec(spec_path)
        try:
            cg = qs.summarize_callgrind(jd, spec["phases"], spec.get("phase_callers", ["cmd_run(*"]))
        except Exception as exc:
            xcheck.append({"job": r["job"], "error": str(exc)})
            continue
        common.write_json(out / f"callgrind-{r['dataset']}-t{r['threads']}.json", cg)
        gp = next((x for x in good if x["tool"] == "gperf" and x["dataset"] == r["dataset"]
                   and x["threads"] == r["threads"]), None)
        if not gp:
            continue
        g = gp["gperf"]
        for kind, cgkey, cgpct in (("excl", "top_self", "self_pct"), ("incl", "top_incl", "incl_pct")):
            for f in cg[cgkey][:15]:
                name = f["fn"]
                n = g[kind].get(name)
                if n is None:   # gperf names may differ by clone suffix / return type
                    alt = [k for k in g[kind] if short(k) == f["name"]]
                    n = sum(g[kind][k] for k in alt) if alt else 0
                xcheck.append({"dataset": r["dataset"], "threads": r["threads"], "kind": kind,
                               "fn": f["name"], "callgrind_pct_Ir": f[cgpct],
                               "gperf_pct_cpu": round(100.0 * n / g["total_samples"], 2)
                               if g["total_samples"] else None})
        for p in cg["phases"]:
            q = next((x for x in gp["phases"] if x["phase"] == p["phase"]), None)
            xcheck.append({"dataset": r["dataset"], "threads": r["threads"], "kind": "phase",
                           "fn": p["phase"], "callgrind_pct_Ir": p["pct"],
                           "gperf_pct_cpu": round(100.0 * q["cpu_s"] / gp["cpu_s"], 2)
                           if q and gp["cpu_s"] else None})
    # heaptrack
    heap = []
    for r in good:
        if r["tool"] == "heaptrack":
            heap.append({k: v for k, v in r.items() if k.startswith("heaptrack") or k in
                         ("job", "dataset", "threads", "molecules", "molecules_loaded", "peak_rss_kb",
                          "wall_s", "cpu_s")})
    summary = {"run_dir": str(run_dir), "jobs": [{k: v for k, v in r.items() if k not in ("gperf",)}
                                                  for r in rows],
               "fits": fits, "xcheck": xcheck, "heaptrack": heap}
    common.write_json(out / "scaling_summary.json", summary)

    def wcsv(path, rs):
        if not rs:
            return
        keys = []
        for r in rs:
            for k in r:
                if k not in keys:
                    keys.append(k)
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            for r in rs:
                w.writerow({k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in r.items()})
    wcsv(out / "scaling_jobs.csv", [{k: v for k, v in r.items() if k not in ("phases", "gperf", "command",
                                                                             "env")} for r in rows])
    wcsv(out / "scaling_phases.csv", phase_rows)
    wcsv(out / "scaling_fits.csv", fits)
    wcsv(out / "scaling_xcheck.csv", xcheck)
    par_rows = parallel_rows(good)
    if par_rows:
        wcsv(out / "scaling_parallel.csv", par_rows)
    wcsv(out / "scaling_heaptrack.csv", [{**{k: v for k, v in h.items() if not k.startswith("heaptrack_top")}}
                                         for h in heap])
    try:
        plots(out, good, fits)
    except Exception as exc:  # plotting is optional
        print(f"plots failed: {exc}")
    (out / "scaling_tables.md").write_text(tables_md(good, fits, xcheck, heap))
    print(f"scaling summary -> {out}")
    return summary


OMP_WAIT_RE = gperf.OMP_WAIT_RE


def omp_wait_share(g: dict) -> float | None:
    """Share of CPU samples in thread waiting: libgomp barrier/spin-wait
    functions (leaf, OpenMP builds); with the stack classification of
    gperf.classify (pool builds) also the pool's hand-off/barrier machinery
    and Eigen's GEMM pool idling."""
    if not g or not g.get("total_samples"):
        return None
    c = g.get("classes")
    if c:
        return (c["wait_omp"] + c["wait_pool"] + c["wait_eigen"]) / g["total_samples"]
    n = sum(v for fn, v in g["excl"].items() if OMP_WAIT_RE.match(fn))
    return n / g["total_samples"]


def parallel_rows(good: list) -> list:
    """Per gperf job: CPU share inside parallel-region bodies and the serial
    share of the non-waiting CPU (Amdahl's s at 1 thread). Pool builds: from
    the per-sample stack classification (also per phase); OpenMP builds
    without it: inclusive samples of the outlined bodies (`*._omp_fn.N`,
    no nested regions in these builds), whole run only."""
    import cgparse
    rows = []
    for r in good:
        g = r.get("gperf")
        if r["tool"] != "gperf" or not g or not g.get("total_samples"):
            continue
        tot = g["total_samples"]
        c = g.get("classes")
        if c:
            wait = c["wait_omp"] + c["wait_pool"] + c["wait_eigen"]
            par, method = c["parallel"], "stack classification"
        else:
            wait = sum(v for fn, v in g["excl"].items() if OMP_WAIT_RE.match(fn))
            par = sum(v for fn, v in g["incl"].items() if cgparse.is_region_fn(fn))
            method = "inclusive samples of region functions"
        work = tot - wait
        s = 1 - par / work if work else None
        base = {"job": r["job"], "dataset": r["dataset"], "threads": r["threads"],
                "molecules": r.get("molecules_loaded") or r["molecules"], "method": method}
        rows.append({**base, "phase": "(whole run)", "samples": tot, "wait_pct": round(100 * wait / tot, 2),
                     "parallel_pct": round(100 * par / tot, 2),
                     "serial_frac": round(s, 4) if s is not None else None,
                     **{f"amdahl_{p}": round(1 / (s + (1 - s) / p), 3) if s is not None else None
                        for p in (8, 16)}})
        for ph, v in ((c or {}).get("phases") or {}).items():
            w = v["samples"] - v["wait"]
            sp = 1 - v["parallel"] / w if w > 0 else None
            rows.append({**base, "phase": ph, "samples": round(v["samples"], 1),
                         "wait_pct": round(100 * v["wait"] / v["samples"], 2) if v["samples"] else None,
                         "parallel_pct": round(100 * v["parallel"] / v["samples"], 2) if v["samples"] else None,
                         "serial_frac": round(sp, 4) if sp is not None else None,
                         **{f"amdahl_{p}": round(1 / (sp + (1 - sp) / p), 3) if sp is not None else None
                            for p in (8, 16)}})
    return rows


def tables_md(good: list, fits: list, xcheck: list, heap: list) -> str:
    """Markdown building blocks for REPORT.md section 6."""
    def slide_of(ds):
        m = re.match(r"^(.*)_(\d+k|\d+(?:\.\d+)?M|all)$", ds)
        return m.group(1) if m else ds

    def t(headers, rows):
        o = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
        o += ["| " + " | ".join("" if c is None else str(c) for c in r) + " |" for r in rows]
        return "\n".join(o) + "\n"

    parts = ["# Scaling tables (generated by scaling.py summarize)\n", "## Ladder\n"]
    rows = []
    for r in sorted(good, key=lambda r: (r["tool"], slide_of(r["dataset"]), r["threads"], r["molecules"])):
        ow = omp_wait_share(r.get("gperf"))
        rows.append((r["tool"], f"`{r['dataset']}`", r["threads"], f"{r.get('molecules_loaded') or r['molecules']:,}",
                     r.get("genes_loaded"), f"{r['wall_s']:.0f}", f"{r['cpu_s']:.0f}",
                     f"{r['cpu_s'] / r['wall_s']:.2f}" if r["wall_s"] else "",
                     f"{(r.get('heaptrack_peak_rss') or 0) / 2**30:.2f} (with heaptrack)"
                     if r["tool"] == "heaptrack" else f"{r['peak_rss_kb'] / 2**20:.2f}",
                     f"{100 * ow:.1f}" if ow is not None else "",
                     ", ".join(f"{v:.1f}" for v in (r.get("loadavg_start") or [])[:1] + (r.get("loadavg_end") or [])[:1])))
    pool = any((r.get("gperf") or {}).get("classes") for r in good)
    parts.append(t(["tool", "dataset", "thr", "molecules", "genes", "wall s (load-sens.)", "CPU s", "CPU/wall",
                    "peak RSS GiB", "wait % of CPU (pool/OpenMP)" if pool else "OpenMP wait % of CPU",
                    "load 1m start, end"], rows))
    parts.append("\n## Exponents (log-log slope vs molecules)\n")
    rows = []
    for f in fits:
        if f["what"] in ("total", "phase_cpu", "phase_rss"):
            if f["what"] != "total" and not any(v for v in f["values"] if v):
                continue
            if f["what"] == "phase_cpu" and max((v or 0) for v in f["values"]) < 1.0:
                continue
            rows.append((f["slide"], f["threads"], f["what"], f["name"], f["exponent"],
                         f"{f.get('share_first'):.3f}" if f.get("share_first") is not None else "",
                         f"{f.get('share_last'):.3f}" if f.get("share_last") is not None else "",
                         "**yes**" if f.get("superlinear") else "",
                         ", ".join("" if v is None else (f"{v:.0f}" if v >= 10 else f"{v:.2f}") for v in f["values"])))
    parts.append(t(["slide", "thr", "what", "name", "exponent", "share first", "share last", "super-linear",
                    "values (CPU s / RSS kB) by rung"], rows))
    parts.append("\n## Functions (gperftools CPU share x CPU s)\n")
    rows = []
    for f in sorted((f for f in fits if f["what"].startswith("fn_")), key=lambda f: (f["slide"], f["threads"],
                                                                                    f["what"], -(f["share_last"] or 0))):
        if max(f["shares"]) < 0.02:
            continue
        rows.append((f["slide"], f["threads"], f["what"][3:], f"`{f['name']}`", f["exponent"],
                     f"{100 * f['share_first']:.1f}", f"{100 * f['share_last']:.1f}",
                     "**yes**" if f.get("superlinear") else "", "**yes**" if f.get("growing_share") else ""))
    parts.append(t(["slide", "thr", "kind", "function", "exponent", "% CPU first rung", "% CPU last rung",
                    "super-linear", "growing share"], rows))
    parts.append("\n## callgrind cross-check (same rung, 1 thread)\n")
    parts.append(t(["dataset", "kind", "function / phase", "callgrind % Ir", "gperftools % CPU"],
                   [(x.get("dataset"), x.get("kind"), f"`{x.get('fn')}`", x.get("callgrind_pct_Ir"),
                     x.get("gperf_pct_cpu")) for x in xcheck if "error" not in x]))
    parts.append("\n## heaptrack\n")
    for h in heap:
        parts.append(f"\n### {h['dataset']} ({h['threads']} thr): peak heap "
                     f"{(h.get('heaptrack_peak_heap') or 0) / 2**30:.2f} GiB, "
                     f"{h.get('heaptrack_calls', 0):,} allocations, "
                     f"{h.get('heaptrack_temporary', 0):,} temporary, peak RSS with heaptrack "
                     f"{(h.get('heaptrack_peak_rss') or 0) / 2**30:.2f} GiB\n")
        tot = h.get("heaptrack_peak_heap") or 1
        parts.append(t(["live at peak: site", "location", "MiB", "% of peak"],
                       [(f"`{x['site']}`", x["loc"], f"{x['value'] / 2**20:.1f}", f"{100 * x['value'] / tot:.1f}")
                        for x in h.get("heaptrack_top_peak", [])[:12]]))
        calls = h.get("heaptrack_calls") or 1
        parts.append(t(["allocations: site", "location", "allocations", "% of all"],
                       [(f"`{x['site']}`", x["loc"], f"{x['value']:,}", f"{100 * x['value'] / calls:.1f}")
                        for x in h.get("heaptrack_top_allocations", [])[:12]]))
    return "\n".join(parts)


def plots(out: Path, good: list, fits: list) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def slide_of(ds):
        m = re.match(r"^(.*)_(\d+k|\d+(?:\.\d+)?M|all)$", ds)
        return m.group(1) if m else ds
    series = defaultdict(list)
    for r in good:
        if r["tool"] == "gperf":
            series[(slide_of(r["dataset"]), r["threads"])].append(r)
    # 1. total CPU / wall / RSS vs molecules
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for (slide, thr), lst in sorted(series.items()):
        lst.sort(key=lambda r: r["molecules"])
        xs = [r["molecules_loaded"] or r["molecules"] for r in lst]
        lab = f"{slide} {thr} thr"
        axes[0].loglog(xs, [r["cpu_s"] for r in lst], "o-", label=lab)
        axes[1].loglog(xs, [r["wall_s"] for r in lst], "o-", label=lab)
        axes[2].loglog(xs, [r["peak_rss_kb"] / 2**20 for r in lst], "o-", label=lab)
    for ax, t in zip(axes, ("CPU seconds", "wall seconds (load-sensitive)", "peak RSS (GiB)")):
        ax.set_xlabel("molecules")
        ax.set_title(t)
        ax.grid(True, which="both", alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "scaling_totals.png", dpi=110)
    plt.close(fig)
    # 2. per-phase CPU seconds vs molecules, per series
    for (slide, thr), lst in sorted(series.items()):
        if len(lst) < 2:
            continue
        lst.sort(key=lambda r: r["molecules"])
        xs = [r["molecules_loaded"] or r["molecules"] for r in lst]
        fig, ax = plt.subplots(1, 2, figsize=(13, 5))
        names = []
        for r in lst:
            for p in r["phases"]:
                if p["phase"] not in names:
                    names.append(p["phase"])
        for ph in names:
            vals = [next((p["cpu_s"] for p in r["phases"] if p["phase"] == ph), 0) for r in lst]
            if max(vals) < 0.02 * max(r["cpu_s"] for r in lst):
                continue
            ax[0].loglog(xs, vals, "o-", label=ph)
            ax[1].semilogx(xs, [v / r["cpu_s"] for v, r in zip(vals, lst)], "o-", label=ph)
        ax[0].set_title(f"{slide}, {thr} thr: CPU s per phase")
        ax[1].set_title("share of CPU time")
        for a in ax:
            a.set_xlabel("molecules")
            a.grid(True, which="both", alpha=0.3)
        ax[0].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out / f"scaling_phases_{slide}_t{thr}.png", dpi=110)
        plt.close(fig)
    # 3. parallel efficiency: CPU/wall per phase at 8 threads
    for (slide, thr), lst in sorted(series.items()):
        if thr == 1 or len(lst) < 2:
            continue
        lst.sort(key=lambda r: r["molecules"])
        xs = [r["molecules_loaded"] or r["molecules"] for r in lst]
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.semilogx(xs, [r["cpu_s"] / r["wall_s"] for r in lst], "ko-", label="whole run")
        for ph in ("bmm_iterations", "ncv_colors", "mol_clustering", "confidence", "polygons"):
            vals = [next((p["cpu_per_wall"] for p in r["phases"] if p["phase"] == ph), None) for r in lst]
            if any(v for v in vals):
                ax.semilogx(xs, [v or float("nan") for v in vals], "o-", label=ph)
        ax.set_title(f"{slide}: CPU/wall at {thr} threads (load-sensitive)")
        ax.set_xlabel("molecules")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out / f"scaling_parallelism_{slide}_t{thr}.png", dpi=110)
        plt.close(fig)


# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["plan", "run", "summarize"])
    ap.add_argument("run_dir", nargs="?")
    ap.add_argument("--spec", default=str(HERE / "profiling.yaml"))
    ap.add_argument("--baysor-src", default=None,
                    help="optional Baysor source checkout (git sha of the run, default "
                         "binary location); default $BAYSOR_SRC")
    ap.add_argument("--baysor", default=os.environ.get("BAYSOR_BIN"),
                    help="profiling build of baysor; default $BAYSOR_BIN, or "
                         "<baysor-src>/build/profiling/baysor when --baysor-src is given")
    ap.add_argument("--valgrind", default=None,
                    help="valgrind executable (default $VALGRIND, else <repo>/.deps/vgenv, "
                         "else valgrind on PATH)")
    ap.add_argument("--proftools", default=None,
                    help="gperftools/heaptrack prefix (default $PROFTOOLS, else "
                         "<repo>/.deps/proftools)")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--max-procs", type=int, default=8)
    ap.add_argument("--cores", type=int, default=18,
                    help="thread budget (default 18: two 8-thread jobs + two 1-thread jobs)")
    ap.add_argument("--only", default=None, help="comma-separated tools")
    ap.add_argument("--filter", default=None, help="glob on job ids, e.g. 'gperf-lung_*-t8'")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--out", default=None,
                    help="summarize: output directory (default <run_dir>/summary)")
    args = ap.parse_args(argv)
    if args.max_procs > 8:
        ap.error("--max-procs is capped at 8")
    spec_path = Path(args.spec)
    spec_all = common.load_spec(spec_path)
    spec = spec_all["scaling"]
    src = common.baysor_src(args.baysor_src)
    root = common.bench_root(args.data_root)
    repo = common.repo_root()
    if args.baysor:
        baysor = Path(args.baysor).expanduser().resolve()
    elif src is not None:
        baysor = (src / "build" / "profiling" / "baysor").resolve()
    else:
        baysor = None
    valgrind = common.find_valgrind(args.valgrind or os.environ.get("VALGRIND"))
    proftools = common.find_proftools(args.proftools)

    if args.cmd == "summarize":
        if not args.run_dir:
            ap.error("summarize needs <run_dir>")
        summarize_run(Path(args.run_dir).resolve(), spec_path,
                      out=Path(args.out).resolve() if args.out else None)
        return 0

    dsets = scaling_datasets(root, spec)
    jobs = expand_jobs(spec, dsets)
    if args.only:
        jobs = [j for j in jobs if j.tool in set(args.only.split(","))]
    if args.filter:
        import fnmatch
        jobs = [j for j in jobs if fnmatch.fnmatchcase(j.id, args.filter)]
    if args.cmd == "plan":
        print("resolved:")
        print(f"  spec:       {spec_path}")
        print(f"  data root:  {root}")
        print(f"  ladders:    {root / 'profiling' / 'scaling' / 'data'} ({len(dsets)} datasets)")
        print(f"  runs dir:   {root / 'profiling' / 'runs'}")
        print(f"  baysor:     {baysor or 'not given (--baysor / BAYSOR_BIN)'}")
        print(f"  baysor-src: {src or 'not given (only used for the run git sha)'}")
        print(f"  valgrind:   {valgrind or 'not found (--valgrind / VALGRIND)'}")
        print(f"  proftools:  {proftools or 'not found (--proftools / PROFTOOLS)'}")
        print(f"  configs:    {repo / 'baysor-configs'} (vendored, resolved by the harness)")
        print("jobs:")
        tot = 0.0
        for j in sorted(jobs, key=lambda j: -j.est_s):
            tot += j.est_s * j.threads
            print(f"  {j.id:40s} {j.molecules:>9d} mols  est {j.est_s / 60:6.1f} min")
        print(f"{len(jobs)} jobs, ~{tot / 3600:.1f} core-hours (rough)")
        return 0

    if baysor is None:
        ap.error("baysor binary unknown: pass --baysor or set BAYSOR_BIN (build it, see README.md)")
    if not baysor.is_file():
        ap.error(f"baysor binary not found: {baysor} (build it, see README.md)")
    sha = common.run_sha(src, baysor)
    run_id = args.run_id or f"{time.strftime('%Y-%m-%d')}-{sha}-scaling"
    run_dir = root / "profiling" / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    import common as hcommon   # harness
    import run as hrun
    # a missing valgrind stays the raw name: only callgrind jobs need it and
    # they record the exec error as a failed job (as before)
    valgrind_s = str(valgrind) if valgrind else (
        args.valgrind or os.environ.get("VALGRIND") or "valgrind")
    ctx = {"harness": (hcommon, hrun), "baysor": baysor, "probe": hrun.probe_binary(baysor),
           "repo": repo, "proftools": proftools_prefix(args.proftools), "valgrind": valgrind_s,
           "spec_path": spec_path}
    t0 = time.time()
    import profile as qp
    info = qp.host_info(baysor, valgrind_s, src)
    info.update({"run_id": run_id, "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                 "max_procs": args.max_procs, "cores": args.cores, "argv": sys.argv,
                 "jobs": [j.id for j in jobs]})
    sampler = qp.LoadSampler(60.0)
    sampler.start()
    schedule(jobs, run_dir, dsets, ctx, args.max_procs, args.cores, args.force)
    sampler.stop()
    info["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    info["wall_s"] = round(time.time() - t0, 1)
    info["load_samples"] = sampler.samples
    prev = run_dir / "suite.json"
    if prev.is_file():   # keep the load history of earlier (resumed) invocations
        old = common.read_json(prev)
        info["previous_invocations"] = old.get("previous_invocations", []) + [
            {k: old.get(k) for k in ("started", "finished", "wall_s")}]
        info["load_samples"] = old.get("load_samples", []) + info["load_samples"]
    common.write_json(prev, info)
    summarize_run(run_dir, spec_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
