#!/usr/bin/env python3
"""Turn the raw outputs of a profiling run into machine-readable summaries.

    python profiling/summarize.py <run_dir> [--jobs 8]

Reads <run_dir>/raw/<job>/ (callgrind.out.*, dhat.out.json, procmon.json,
job.json) and writes <run_dir>/summary/:

  summary.json        index: per job the headline numbers (+ suite.json info)
  jobs.csv            one row per job: tool, dataset, threads, wall of the job
  callgrind/<job>.json  full per-job callgrind summary (phases, functions,
                      lines, OpenMP regions, threads, cache events)
  phases.csv          callgrind per-phase Ir, % of total, serial/parallel Ir,
                      Amdahl bounds (one row per job x phase)
  functions.csv       top functions per job: exclusive/inclusive Ir (+ cache
                      events for cache-sim jobs)
  lines.csv           hottest source lines inside the top functions
  omp_regions.csv     per OpenMP region: Ir per thread, imbalance
  threads.csv         per job x thread x phase Ir
  dhat/<job>.json, dhat_sites.csv   heap: peak, live-at-peak and churn by site
  native.csv, native_phases.csv     wall/CPU/RSS per run and per phase
                                    (load-sensitive except peak RSS)
"""
from __future__ import annotations

import argparse
import csv
import fnmatch
import json
import re
import statistics
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cgparse  # noqa: E402
import profcommon as common  # noqa: E402

REPO = common.repo_root()
TOP_N = 60
ENTRY_FNS = re.compile(r"^(\(below main\)|main|__libc_start_main.*|__libc_start_call_main|"
                       r"0x[0-9a-f]+|cmd_run\(.*|auto cmd_run\(.*|_start|"
                       r"GOMP_parallel|start_thread|clone3?|gomp_thread_start|"
                       r"std::thread::_State_impl<.*baysor::\(anonymous namespace\)::ThreadPool.*>::_M_run\(\)|"
                       r"baysor::\(anonymous namespace\)::ThreadPool::worker_loop.*)$")
SYSTEM_HDR = re.compile(r"(intrin\.h|/include/c\+\+/|/bits/|xmmintrin|emmintrin)")
ALLOC_FRAME = re.compile(r"(\bmalloc\b|\bcalloc\b|\brealloc\b|operator new|"
                         r"memalign|posix_memalign|aligned_alloc|"
                         r"__gnu_cxx::new_allocator|std::allocator|"
                         r"std::__new_allocator|allocator_traits|_M_allocate|"
                         r"_M_create_storage|_M_realloc|Eigen::internal::aligned_malloc|"
                         r"Eigen::internal::conditional_aligned|"
                         r"Eigen::DenseStorage|Eigen::PlainObjectBase<.*>::resize|"
                         r"std::vector<.*>::_M_default_append|std::_Vector_base|"
                         r"arrow::.*Allocate|arrow::PoolBuffer|je_arrow|mallocx)")


# ---------------------------------------------------------------------------
# name / path helpers
# ---------------------------------------------------------------------------

def strip_templates(s: str) -> str:
    out, depth = [], 0
    i = 0
    while i < len(s):
        c = s[i]
        if s.startswith("operator<", i) or s.startswith("operator>", i):
            out.append(s[i:i + 9]); i += 9; continue
        if c == "<":
            depth += 1
        elif c == ">" and depth:
            depth -= 1
        elif depth == 0:
            out.append(c)
        i += 1
    return "".join(out)


POOL_KIND_SUFFIX = ((cgparse.POOL_CHUNK_PREFIX, ".pool_chunk"), (cgparse.POOL_BODY_PREFIX, ".pool_region"),
                    (cgparse.POOL_SINGLE_PREFIX, ".pool_single"))


@lru_cache(maxsize=None)
def pool_parent(fn: str):
    """Short name of the function that encloses the user callable of a pool
    handler (chunk / region body / single block), e.g. 'baysor::apply_phase'
    for a ParallelRegion::for_each loop inside apply_phase; None otherwise."""
    if not any(fn.startswith(p) for p, _ in POOL_KIND_SUFFIX):
        return None
    r = cgparse.pool_callable(re.sub(r"\s*\[clone [^\]]+\]$", "", fn))
    return short_name(r[0]) if r else None


@lru_cache(maxsize=None)
def short_name(fn: str) -> str:
    """'void ns::f<int>(int) [clone ._omp_fn.0]' -> 'ns::f._omp_fn.0'.

    Pool handlers: 'std::_Function_handler<void (long, long, int),
    ParallelRegion::for_each<ns::f(...)::{lambda(int)#2}>(...)::{lambda(...)#1}>::_M_invoke'
    -> 'ns::f::{lambda#2}.pool_chunk' (.pool_region: persistent-region body,
    .pool_single: single block)."""
    for prefix, suffix in POOL_KIND_SUFFIX:
        if fn.startswith(prefix) and "::_M_invoke(" in fn:
            m = re.search(r"\[clone (\.[^\]]+)\]$", fn)
            r = cgparse.pool_callable(fn[:m.start()].strip() if m else fn)
            if r:
                return (short_name(r[0]) + (f"::{{lambda#{r[1]}}}" if r[1] else "") + suffix
                        + (m.group(1) if m else ""))
    fn = fn.replace("(anonymous namespace)", "{anon}")
    clone = ""
    m = re.search(r"\[clone (\.[^\]]+)\]", fn)
    if m:
        clone = m.group(1)
        fn = fn[:m.start()].strip()
    s = strip_templates(fn)
    # drop argument list (outermost parentheses at the end), keep lambdas readable
    depth, cut = 0, None
    for i, c in enumerate(s):
        if c == "(":
            if depth == 0 and cut is None and not s[:i].endswith("operator"):
                cut = i
            depth += 1
        elif c == ")":
            depth -= 1
    head = s[:cut] if cut is not None else s
    tail = ""
    if "::{lambda" in s:
        tail = "::{lambda}"
    # drop return type: last space outside of names
    head = head.strip()
    if " " in head:
        head = head.rsplit(" ", 1)[-1]
    head = head.replace("[abi:cxx11]", "")
    return (head + tail + clone) or fn[:80]


def short_path(p: str) -> str:
    if not p or p == "???":
        return "???"
    s = str(p)
    for marker in ("/Baysor/", "/_deps/"):
        if marker in s:
            idx = s.rindex(marker)
            rest = s[idx + len(marker):]
            return ("_deps/" + rest) if marker == "/_deps/" else rest
    parts = Path(s).parts
    return "/".join(parts[-3:])


# ---------------------------------------------------------------------------
# callgrind
# ---------------------------------------------------------------------------

def summarize_callgrind(job_dir: Path, phases_spec: list, top_callers: list) -> dict:
    files = sorted(p for p in job_dir.glob("callgrind.out*") if p.stat().st_size > 0)
    if not files:   # --tool=cachegrind: one file, self costs only, no phases
        files = sorted(p for p in job_dir.glob("cachegrind.out*") if p.stat().st_size > 0)
    profs = [cgparse.parse(p) for p in files]
    if not profs:
        raise RuntimeError(f"no callgrind output in {job_dir}")
    events = profs[0].events
    by_part = defaultdict(dict)
    for p in profs:
        by_part[p.part][p.thread] = p
    parts = sorted(by_part)
    agg = cgparse.Aggregate(events)
    for p in profs:
        agg.add(p)
    IR = 0
    total = agg.totals[IR]
    thread_ids = sorted({p.thread for p in profs})

    # ---- phases -----------------------------------------------------------
    # Callgrind's --dump-before writes only the costs of the thread that hit
    # the trigger (the main thread); the other threads are written once, at
    # program end. Therefore:
    #  * main thread: every part starts at the entry of a phase function and
    #    callgrind writes the in-part inclusive cost of every call arc
    #    (including calls still active at the dump), so a phase's main-thread
    #    cost is the sum over parts of the arcs from a top-level caller
    #    (cmd_run / its segmentation lambda, spec `phase_callers`) into the
    #    phase function. A phase function called from elsewhere (e.g.
    #    build_molecule_graph inside the confidence estimate) stays in its
    #    caller's phase; main-thread cost outside phase calls is glue.
    #  * OpenMP regions (`f [clone ._omp_fn.N]`) belong to the phase from
    #    whose function the outlining parent `f` is reachable in that part's
    #    main-thread call graph (not expanding GOMP_* runtime nodes).
    #  * worker threads: each region's worker cost is split over phases in
    #    proportion to the main thread's cost of that region per phase;
    #    workers' cost outside regions (thread start, barriers) is reported
    #    as `omp_worker_runtime`, non-OpenMP threads (Arrow I/O pool) as
    #    `other_threads`.
    def phase_name(fn: str):
        for ph in phases_spec:
            if fnmatch.fnmatchcase(fn, ph["pattern"]):
                return ph["name"]
        return None

    def is_top(fn: str) -> bool:
        return any(fnmatch.fnmatchcase(fn, pat) for pat in top_callers)

    def parent_of(region: str) -> str:
        return re.sub(r"\s*\[clone \._omp_fn\.\d+\]$", "", region)

    def is_entry(a: str, b: str) -> bool:
        """Arc a -> b enters a parallel-region body: an OpenMP outlined
        function entered from the runtime, or a pool chunk (see cgparse)."""
        return ((cgparse.is_omp_fn(b) and not cgparse.is_omp_fn(a))
                or (cgparse.is_pool_chunk(b) and not cgparse.is_pool_chunk(a)))

    def expand(seen: set, stack: list, succ: dict) -> None:
        # runtime nodes (GOMP_*, the pool's machinery) are not expanded:
        # they connect every region with every caller
        while stack:
            f = stack.pop()
            if f.startswith("GOMP_") or cgparse.is_pool_runtime(f):
                continue
            for g in succ.get(f, []):
                if g not in seen:
                    seen.add(g)
                    stack.append(g)

    phase_rows = defaultdict(lambda: {"Ir": 0, "main_Ir": 0, "workers_Ir": 0,
                                      "par_main_Ir": 0, "par_workers_Ir": 0,
                                      "per_thread": defaultdict(int), "n_parts": 0})
    order = []
    ambiguous = set()
    region_main = defaultdict(lambda: defaultdict(int))   # region -> phase -> main-thread Ir
    last_phase = None

    def add(label, **kw):
        if label not in order:
            order.append(label)
        row = phase_rows[label]
        for k, v in kw.items():
            if k == "thread":
                continue
            row[k] += v
        return row

    for k in parts:
        main = by_part[k].get(1)
        if main is None:
            continue
        main_tot = main.total("Ir")
        succ = defaultdict(list)
        ph_cost = defaultdict(int)
        ph_fns = defaultdict(set)
        for (a, b), (_, c) in main.arcs.items():
            succ[a].append(b)
            if c[IR] and is_top(a) and not is_top(b):
                name = phase_name(b)
                if name:
                    ph_cost[name] += c[IR]
                    ph_fns[name].add(b)
        reach = {}
        for name, fns in ph_fns.items():
            seen, stack = set(fns), list(fns)
            expand(seen, stack, succ)
            reach[name] = seen
        # pool handlers are called from the pool's machinery, which is not
        # expanded: link each handler to the phase that reaches the function
        # enclosing its lambda (by short name; repeated for nested regions)
        handlers = sorted({b for (a, b) in main.arcs if pool_parent(b) is not None})
        if handlers:
            for name, seen in reach.items():
                shorts = {short_name(f) for f in seen}
                changed = True
                while changed:
                    changed = False
                    for h in handlers:
                        if h not in seen and pool_parent(h) in shorts:
                            seen.add(h)
                            before = set(seen)
                            expand(seen, [h], succ)
                            shorts |= {short_name(f) for f in seen - before} | {short_name(h)}
                            changed = True
        if ph_cost:
            main_phase = max(ph_cost, key=ph_cost.get)
        else:
            main_phase = None
        glue_label = ("startup" if last_phase is None and main_phase is None else
                      ("shutdown" if k == parts[-1] else f"glue_after_{main_phase or last_phase}"))
        for name, cost in ph_cost.items():
            r = add(name, Ir=cost, main_Ir=cost)
            r["per_thread"][1] += cost
            r["n_parts"] += 1
        glue = main_tot - sum(ph_cost.values())
        if glue > 0:
            if last_phase is None and main_phase is None:
                glue_label = "startup"
            r = add(glue_label, Ir=glue, main_Ir=glue)
            r["per_thread"][1] += glue
            r["n_parts"] += 1
        if main_phase is not None:
            last_phase = main_phase
        # main-thread parallel regions of this part
        for (a, b), (_, c) in main.arcs.items():
            if not is_entry(a, b) or not c[IR]:
                continue
            par = b if cgparse.is_pool_chunk(b) else parent_of(b)
            owners = [n for n, rs in reach.items() if par in rs]
            if len(owners) > 1:
                ambiguous.add(short_name(b))
            if not owners:
                owners = [main_phase or glue_label]
            for n in owners:
                share = c[IR] / len(owners)
                region_main[b][n] += share
                phase_rows[n]["par_main_Ir"] += share

    # worker threads (summed over their files; normally only the final one)
    workers = defaultdict(lambda: cgparse.Aggregate(events))
    for p in profs:
        if p.thread != 1:
            workers[p.thread].add(p)
    for t, wa in sorted(workers.items()):
        tot_t = wa.totals[IR]
        in_regions = 0
        pool_thread = False
        for (a, b), (_, c) in wa.arcs.items():
            if not is_entry(a, b) or not c[IR]:
                continue
            pool_thread = pool_thread or cgparse.is_pool_chunk(b)
            in_regions += c[IR]
            split = region_main.get(b)
            if not split:
                split = {"unattributed_regions": 1}
            s = sum(split.values())
            for n, w in split.items():
                v = c[IR] * w / s
                r = add(n, Ir=v, workers_Ir=v, par_workers_Ir=v)
                r["per_thread"][t] += v
        rest = tot_t - in_regions
        if rest > 0:
            # pool workers: hand-off/barrier waits and replicated code of
            # persistent-region bodies outside the work-shared loops
            label = (("pool_worker_runtime" if pool_thread else "omp_worker_runtime")
                     if in_regions else "other_threads")
            r = add(label, Ir=rest, workers_Ir=rest)
            r["per_thread"][t] += rest
    for r in phase_rows.values():
        r["par_main_Ir"] = min(r["par_main_Ir"], r["main_Ir"])
        r["Ir"] = int(round(r["Ir"]))
    shared_parts = sorted(ambiguous)

    phases = []
    for name in order:
        r = phase_rows[name]
        serial = r["main_Ir"] - r["par_main_Ir"]
        par = r["par_main_Ir"] + r["par_workers_Ir"]
        work = serial + par
        s_frac = serial / work if work else 1.0
        amdahl = {str(p): round(1.0 / (s_frac + (1 - s_frac) / p), 3) for p in (2, 4, 8, 16)}
        per_thread = dict(sorted(r["per_thread"].items()))
        vals = [v for v in per_thread.values()]
        phases.append({
            "phase": name, "Ir": r["Ir"], "pct": round(100.0 * r["Ir"] / total, 3) if total else 0,
            "main_Ir": r["main_Ir"], "workers_Ir": r["workers_Ir"],
            "serial_Ir": int(round(serial)), "parallel_Ir": int(round(par)),
            "serial_frac": round(s_frac, 4), "amdahl": amdahl,
            "per_thread": {str(k): int(round(v)) for k, v in per_thread.items()},
            "thread_max_over_mean": round(max(vals) / (sum(vals) / len(vals)), 3)
            if vals and sum(vals) else None,
            "n_parts": r["n_parts"],
        })

    if not any(p.arcs for p in profs):
        phases = []      # --tool=cachegrind: no call graph, no phases
    # ---- whole-run serial / parallel ------------------------------------------
    ser_total = sum(p["serial_Ir"] for p in phases)
    par_total = sum(p["parallel_Ir"] for p in phases)
    s_all = ser_total / (ser_total + par_total) if ser_total + par_total else 1.0
    amdahl_all = {str(p): round(1.0 / (s_all + (1 - s_all) / p), 3) for p in (2, 4, 8, 16)}

    # ---- functions -------------------------------------------------------------
    inc = agg.inclusive()
    callers = agg.callers()
    calls = agg.call_counts()
    ev_idx = {e: i for i, e in enumerate(events)}

    fn_line_best = {}
    lines_by_fn = defaultdict(list)
    for (fn, fl, ln), c in agg.line_cost.items():
        lines_by_fn[fn].append((c[IR], fl, ln, c))
    for fn, lst in lines_by_fn.items():
        lst.sort(key=lambda x: -x[0])
        own = [x for x in lst if not SYSTEM_HDR.search(x[1] or "")]
        fn_line_best[fn] = own[0] if own and own[0][0] >= 0.2 * lst[0][0] else lst[0]

    def fn_record(fn: str) -> dict:
        s = agg.self_cost.get(fn, [0] * len(events))
        i = inc.get(fn, s)
        best = fn_line_best.get(fn)
        rec = {
            "fn": fn, "name": short_name(fn),
            "file": short_path(agg.fn_file.get(fn, "")),
            "object": Path(agg.fn_obj.get(fn, "") or "").name,
            "hot_line": f"{short_path(best[1])}:{best[2]}" if best else None,
            "self_Ir": s[IR], "self_pct": round(100.0 * s[IR] / total, 3),
            "incl_Ir": i[IR], "incl_pct": round(100.0 * i[IR] / total, 3),
            "calls": calls.get(fn, 0),
            "omp_region": cgparse.is_omp_fn(fn),
        }
        if cgparse.is_pool_chunk(fn):
            rec["pool_chunk"] = True
        if "D1mr" in ev_idx:
            g = lambda v, e: v[ev_idx[e]] if e in ev_idx else 0  # noqa: E731
            d_acc = g(s, "Dr") + g(s, "Dw")
            d1 = g(s, "D1mr") + g(s, "D1mw")
            ll = g(s, "DLmr") + g(s, "DLmw")
            rec.update({"self_Dacc": d_acc, "self_D1miss": d1, "self_LLmiss": ll,
                        "self_ILmiss": g(s, "ILmr"), "self_I1miss": g(s, "I1mr"),
                        "D1_miss_rate": round(d1 / d_acc, 5) if d_acc else None,
                        "LL_miss_per_kIr": round(1000.0 * ll / s[IR], 4) if s[IR] else None,
                        "incl_LLmiss": g(i, "DLmr") + g(i, "DLmw"),
                        "incl_D1miss": g(i, "D1mr") + g(i, "D1mw")})
        return rec

    self_sorted = sorted(agg.self_cost, key=lambda f: -agg.self_cost[f][IR])
    top_self = [fn_record(f) for f in self_sorted[:TOP_N]]
    incl_sorted = sorted((f for f in inc if not ENTRY_FNS.match(f)), key=lambda f: -inc[f][IR])
    top_incl = [fn_record(f) for f in incl_sorted[:TOP_N]]
    for rec in top_self[:25] + top_incl[:25]:
        rec["hot_path"] = [short_name(f) for f in agg.hot_path(rec["fn"], IR, callers=callers)]

    top_lines = []
    for rec in top_self[:15]:
        fn = rec["fn"]
        for ir, fl, ln, c in lines_by_fn.get(fn, [])[:8]:
            if ir <= 0:
                continue
            top_lines.append({"fn": rec["name"], "file": short_path(fl), "line": ln,
                              "Ir": ir, "pct_of_total": round(100.0 * ir / total, 3),
                              "pct_of_fn": round(100.0 * ir / rec["self_Ir"], 2) if rec["self_Ir"] else None})

    cache_top = []
    if "DLmr" in ev_idx:
        ll_sorted = sorted(agg.self_cost, key=lambda f: -(agg.self_cost[f][ev_idx["DLmr"]]
                                                          + agg.self_cost[f][ev_idx["DLmw"]]))
        cache_top = [fn_record(f) for f in ll_sorted[:30]]
        d1_sorted = sorted(agg.self_cost, key=lambda f: -(agg.self_cost[f][ev_idx["D1mr"]]
                                                          + agg.self_cost[f][ev_idx["D1mw"]]))
        cache_top_d1 = [fn_record(f) for f in d1_sorted[:30]]
    else:
        cache_top_d1 = []

    # ---- OpenMP regions: Ir per thread -------------------------------------------
    regions = defaultdict(lambda: defaultdict(int))
    region_calls = defaultdict(int)
    for p in profs:
        for (a, b), (n, c) in p.arcs.items():
            if is_entry(a, b):
                regions[b][p.thread] += c[IR]
                if p.thread == 1:
                    region_calls[b] += n
    omp_regions = []
    for fn, per in regions.items():
        vals = list(per.values())
        tot = sum(vals)
        n_thr = len(thread_ids)
        mean_all = tot / max(1, n_thr)
        rec = {
            "region": short_name(fn), "fn": fn, "Ir": tot,
            "pct": round(100.0 * tot / total, 3), "calls_main": region_calls.get(fn, 0),
            "per_thread": {str(k): v for k, v in sorted(per.items())},
            "n_threads_active": sum(1 for v in vals if v > 0),
            "max_over_mean": round(max(vals) / mean_all, 3) if mean_all else None,
        }
        if cgparse.is_pool_chunk(fn):
            rec["kind"] = "pool"
        omp_regions.append(rec)
    omp_regions.sort(key=lambda r: -r["Ir"])

    threads_tot = defaultdict(int)
    for p in profs:
        threads_tot[p.thread] += p.total("Ir")

    trig_callers = {}
    for ph in phases_spec:
        for fn in agg.self_cost:
            if fnmatch.fnmatchcase(fn, ph["pattern"]):
                trig_callers[short_name(fn)] = sorted({short_name(a) for a, _, _ in callers.get(fn, [])})

    return {
        "events": events,
        "totals": dict(zip(events, agg.totals)),
        "total_Ir": total,
        "n_parts": len(parts), "n_files": len(files),
        "threads": {str(k): v for k, v in sorted(threads_tot.items())},
        "phases": phases,
        "serial_Ir": ser_total, "parallel_Ir": par_total,
        "serial_frac": round(s_all, 4), "amdahl": amdahl_all,
        "top_self": top_self, "top_incl": top_incl, "top_lines": top_lines,
        "cache_top_LL": cache_top, "cache_top_D1": cache_top_d1,
        "cache_desc": [d for d in profs[0].header.get("desc", []) if "cache" in d],
        "omp_regions": omp_regions,
        "nested_parallel": agg.omp_split()["nested"] + pool_nested(agg),
        "regions_in_several_phases": shared_parts,
        "trigger_callers": trig_callers,
    }


def pool_nested(agg) -> list:
    """Pool chunks that call into the pool again (nested parallel calls run
    serially inline; their chunk cost is then also inside the outer chunk)."""
    callees = agg.callees()
    out = []
    for f in {b for (_, b) in agg.arcs if cgparse.is_pool_chunk(b)}:
        seen, stack = {f}, [f]
        hit = False
        while stack and not hit:
            g = stack.pop()
            for h, _, _ in callees.get(g, []):
                if cgparse.POOL_ENTRY_RE.match(h):
                    hit = True
                    break
                if h not in seen and not cgparse.is_pool_runtime(h):
                    seen.add(h)
                    stack.append(h)
        if hit:
            out.append(f)
    return sorted(out)


# ---------------------------------------------------------------------------
# DHAT
# ---------------------------------------------------------------------------

STL_UNQUALIFIED = re.compile(
    r"^(allocate|deallocate|construct|destroy|emplace_back|emplace|emplace_hint|push_back|resize|"
    r"reserve|insert|assign|append|operator\[\]|operator=|operator new|operator new\[\]|"
    r"vector|basic_string|_Vector_base|_M_\w+|_S_\w+|conditional_aligned\w*|aligned_\w+|"
    r"DenseStorage|PlainObjectBase|Matrix|Array|resize_if_allowed|call_assignment\w*|"
    r"_Temporary_buffer|_Impl|__get_temporary_buffer|get_temporary_buffer|_Hashtable\w*|_Map_base|"
    r"__uninitialized\w*|uninitialized_\w+|__new_allocator|new_allocator|allocator|"
    r"malloc|calloc|realloc|free|memalign|posix_memalign|aligned_alloc|mallocx|"
    r"__libc_\w+|_int_\w+|new|delete|_Scoped_node|_Hash_node\w*|_Rb_tree\w*|pair|tuple|"
    r"resizeLike|_init\d*|_set_noalias|lazyAssign|evalTo|run|"
    r"__do_uninit_copy|__relocate_a\w*)$")


@lru_cache(maxsize=None)
def is_alloc_plumbing(fn: str) -> bool:
    """True for allocator / STL / Eigen storage frames (not an allocation *site*)."""
    sn = short_name(fn)
    if sn.startswith(("std::", "__gnu_cxx::", "Eigen::", "__cxxabiv1", "operator new")):
        return True
    return bool(STL_UNQUALIFIED.match(sn) or ALLOC_FRAME.search(sn) or sn.startswith("GOMP_"))


def site_from_frames(fns: list) -> tuple:
    """(site key, index of the site frame) from innermost-first function names."""
    user = [i for i, f in enumerate(fns) if f and f != "[root]" and not is_alloc_plumbing(f)]
    if not user:
        return "(allocator)", None
    stack = [short_name(fns[i]) for i in user[:3]]
    return " < ".join(stack), user[0]


@lru_cache(maxsize=None)
def _frame_fn(frame: str) -> str:
    # "0x4C2A1B3: fn (in /lib/x.so)" or "0x..: fn (file.cpp:12)"
    m = re.match(r"^0x[0-9A-Fa-f]+: (.*?)(?: \((?:in )?([^()]*)\))?$", frame)
    if not m:
        return frame
    return m.group(1)


def _frame_loc(frame: str) -> str:
    m = re.search(r"\(([^()]*:\d+)\)$", frame)
    return short_path(m.group(1).rsplit(":", 1)[0]) + ":" + m.group(1).rsplit(":", 1)[1] if m else ""


def summarize_dhat(job_dir: Path) -> dict:
    path = job_dir / "dhat.out.json"
    d = json.loads(path.read_text())
    ftbl = d["ftbl"]
    pps = d["pps"]
    tot_b = sum(pp["tb"] for pp in pps)
    tot_bk = sum(pp["tbk"] for pp in pps)
    gmax_b = sum(pp.get("gb", 0) for pp in pps)
    gmax_bk = sum(pp.get("gbk", 0) for pp in pps)
    end_b = sum(pp.get("eb", 0) for pp in pps)

    def site(pp) -> tuple[str, str, list]:
        frames = [ftbl[i] for i in pp["fs"]]
        fns = [_frame_fn(f) for f in frames]
        key, idx = site_from_frames(fns)
        if idx is None:
            return key, "", []
        stack = [short_name(f) for f in fns[idx:idx + 4]]
        return key, _frame_loc(frames[idx]), stack

    by_site = defaultdict(lambda: {"tb": 0, "tbk": 0, "gb": 0, "gbk": 0, "mb": 0, "eb": 0,
                                   "rb": 0, "wb": 0, "loc": "", "stack": [], "n_pp": 0})
    for pp in pps:
        key, loc, stack = site(pp)
        s = by_site[key]
        for k in ("tb", "tbk", "gb", "gbk", "mb", "eb"):
            s[k] += pp.get(k, 0)
        s["rb"] += pp.get("rb", 0) or 0
        s["wb"] += pp.get("wb", 0) or 0
        s["n_pp"] += 1
        if not s["loc"]:
            s["loc"], s["stack"] = loc, stack
    sites = [{"site": k, **v} for k, v in by_site.items()]

    def top(key, n=40):
        return sorted(sites, key=lambda s: -s[key])[:n]

    return {
        "cmd": d.get("cmd"), "t_end_instrs": d.get("te"), "t_gmax_instrs": d.get("tg"),
        "total_bytes": tot_b, "total_blocks": tot_bk,
        "peak_bytes": gmax_b, "peak_blocks": gmax_bk, "end_bytes": end_b,
        "n_program_points": len(pps),
        "top_live_at_peak": top("gb"), "top_churn_bytes": top("tb"),
        "top_churn_blocks": top("tbk"),
    }


# ---------------------------------------------------------------------------
# native
# ---------------------------------------------------------------------------

def summarize_native(job_dir: Path, job: dict) -> dict:
    reps = []
    for rd in sorted(job_dir.glob("rep*")):
        p = rd / "procmon.json"
        if p.is_file():
            r = common.read_json(p)
            hwm = [smp[3] for smp in r.get("samples", []) if smp[3]]
            if hwm:   # ru_maxrss of the child includes the forked Python image
                r["peak_rss_kb"] = max(hwm)
            r.pop("samples", None)
            reps.append(r)
    walls = [r["wall_s"] for r in reps]
    cpus = [r["cpu_s"] for r in reps]
    loads = [r["loadavg_start"][0] for r in reps if r.get("loadavg_start")]
    phase_names = []
    per_phase = defaultdict(lambda: {"wall_s": [], "cpu_s": [], "rss_peak_kb": []})
    for r in reps:
        for ph in r.get("phases", []):
            if ph["phase"] not in phase_names:
                phase_names.append(ph["phase"])
            per_phase[ph["phase"]]["wall_s"].append(ph["wall_s"])
            per_phase[ph["phase"]]["cpu_s"].append(ph["cpu_s"])
            if ph.get("rss_peak_kb"):
                per_phase[ph["phase"]]["rss_peak_kb"].append(ph["rss_peak_kb"])
    phases = []
    for name in phase_names:
        v = per_phase[name]
        w = statistics.median(v["wall_s"])
        c = statistics.median(v["cpu_s"])
        phases.append({"phase": name, "wall_s_median": round(w, 3), "wall_s_min": min(v["wall_s"]),
                       "cpu_s_median": round(c, 3), "cpu_per_wall": round(c / w, 2) if w > 0.05 else None,
                       "rss_peak_kb": max(v["rss_peak_kb"]) if v["rss_peak_kb"] else None})
    return {
        "n_reps": len(reps), "wall_s": walls, "cpu_s": cpus,
        "wall_s_min": min(walls) if walls else None,
        "wall_s_median": statistics.median(walls) if walls else None,
        "cpu_s_min": min(cpus) if cpus else None,
        "cpu_s_median": statistics.median(cpus) if cpus else None,
        "cpu_per_wall_median": round(statistics.median(c / w for c, w in zip(cpus, walls)), 3)
        if walls else None,
        "peak_rss_kb": max(r["peak_rss_kb"] for r in reps) if reps else None,
        "peak_rss_kb_all": [r["peak_rss_kb"] for r in reps],
        "loadavg_1m_at_start": loads,
        "loadavg_1m_mean": round(sum(loads) / len(loads), 2) if loads else None,
        "wait_policy": job.get("wait_policy") or "default",
        "phases": phases,
    }


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def _work(args):
    jd, phases_spec, top_callers = args
    job = common.read_json(jd / "job.json")
    if job.get("status") != "ok":
        return job, None, f"job status {job.get('status')!r} (not summarized)"
    try:
        if job["tool"] in ("callgrind", "cachegrind"):
            return job, summarize_callgrind(jd, phases_spec, top_callers), None
        if job["tool"] == "dhat":
            return job, summarize_dhat(jd), None
        if job["tool"] == "native":
            return job, summarize_native(jd, job), None
    except Exception as exc:  # keep going; report the failure
        return job, None, f"{type(exc).__name__}: {exc}"
    return job, None, "unknown tool"


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (json.dumps(v) if isinstance(v, (dict, list)) else v)
                        for k, v in r.items()})


def dataset_info(run_dir: Path, ds: str) -> dict:
    root = run_dir.parents[1] / "data" / ds
    try:
        meta = common.read_json(root / "meta.json")
    except OSError:
        return {}
    return {"n_molecules": meta["stats"]["n_molecules"], "n_genes_panel": meta["stats"]["n_genes"],
            "source": meta.get("profiling_crop", {}).get("spec", {}).get("source"),
            "prior": meta.get("baysor", {}).get("prior"),
            "is_3d": "z" in (meta.get("baysor", {}).get("extra_args") or []) or ds.find("3d") >= 0}


def loaded_genes(job_dir: Path) -> tuple[int | None, int | None]:
    """(molecules, genes) after Baysor's own filters, from its log."""
    for name in ("valgrind.log", "rep0/baysor.log"):
        p = job_dir / name
        if p.is_file():
            m = re.search(r"Loaded (\d+) transcripts, (\d+) genes", p.read_text(errors="replace"))
            if m:
                return int(m.group(1)), int(m.group(2))
    return None, None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--spec", default=str(HERE / "profiling.yaml"))
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--out", default=None,
                    help="output directory (default <run_dir>/summary); e.g. to re-summarize "
                         "an old run without overwriting its summary")
    args = ap.parse_args(argv)
    run_dir = Path(args.run_dir).resolve()
    spec = common.load_spec(Path(args.spec))
    out = Path(args.out).resolve() if args.out else run_dir / "summary"
    (out / "callgrind").mkdir(parents=True, exist_ok=True)
    (out / "dhat").mkdir(parents=True, exist_ok=True)

    job_dirs = sorted(d for d in (run_dir / "raw").iterdir() if (d / "job.json").is_file())
    work = [(d, spec["phases"], spec.get("phase_callers", ["*cmd_run(*"])) for d in job_dirs]
    with ProcessPoolExecutor(max_workers=max(1, min(args.jobs, 8))) as pool:
        results = list(pool.map(_work, work))

    index, jobs_rows = {}, []
    phase_rows, fn_rows, line_rows, omp_rows, thr_rows = [], [], [], [], []
    dhat_rows, native_rows, native_phase_rows, cache_rows = [], [], [], []
    errors = {}
    for (jd, _, _), (job, summ, err) in zip(work, results):
        jid = job["job"]
        mol, genes = loaded_genes(jd)
        base = {"job": jid, "tool": job["tool"], "dataset": job["dataset"],
                "threads": job["threads"], "plot": job.get("plot", False), "tag": job.get("tag", ""),
                "molecules_loaded": mol, "genes_loaded": genes,
                **dataset_info(run_dir, job["dataset"])}
        jobs_rows.append({**base, "status": job.get("status"),
                          "job_wall_s": job.get("wall_s"),
                          "loadavg_start": (job.get("loadavg_start") or [None])[0],
                          "loadavg_end": (job.get("loadavg_end") or [None])[0],
                          "error": err})
        if err:
            errors[jid] = err
            continue
        if job["tool"] in ("callgrind", "cachegrind"):
            common.write_json(out / "callgrind" / f"{jid}.json", {**base, **summ})
            index[jid] = {**base, "total_Ir": summ["total_Ir"], "serial_frac": summ["serial_frac"],
                          "amdahl": summ["amdahl"], "threads_Ir": summ["threads"],
                          "phases": {p["phase"]: p["Ir"] for p in summ["phases"]},
                          "totals": summ["totals"]}
            for p in summ["phases"]:
                phase_rows.append({**base, **{k: v for k, v in p.items()
                                              if k not in ("per_thread", "amdahl")},
                                   **{f"amdahl_{k}": v for k, v in p["amdahl"].items()}})
                for t, v in p["per_thread"].items():
                    thr_rows.append({**base, "phase": p["phase"], "thread": int(t), "Ir": v})
            for kind, lst in (("self", summ["top_self"]), ("incl", summ["top_incl"])):
                for rank, r in enumerate(lst, 1):
                    fn_rows.append({**base, "rank_by": kind, "rank": rank,
                                    **{k: v for k, v in r.items() if k != "hot_path"},
                                    "hot_path": " < ".join(r.get("hot_path", [])[:6])})
            for r in summ["top_lines"]:
                line_rows.append({**base, **r})
            for r in summ["omp_regions"]:
                omp_rows.append({**base, **r})
            for rank, r in enumerate(summ["cache_top_LL"], 1):
                cache_rows.append({**base, "rank_by": "LL", "rank": rank, **r})
            for rank, r in enumerate(summ["cache_top_D1"], 1):
                cache_rows.append({**base, "rank_by": "D1", "rank": rank, **r})
        elif job["tool"] == "dhat":
            common.write_json(out / "dhat" / f"{jid}.json", {**base, **summ})
            index[jid] = {**base, **{k: summ[k] for k in ("total_bytes", "total_blocks",
                                                          "peak_bytes", "peak_blocks", "end_bytes")}}
            for kind, lst in (("live_at_peak", summ["top_live_at_peak"]),
                              ("churn_bytes", summ["top_churn_bytes"]),
                              ("churn_blocks", summ["top_churn_blocks"])):
                for rank, s in enumerate(lst, 1):
                    dhat_rows.append({**base, "rank_by": kind, "rank": rank,
                                      **{k: v for k, v in s.items() if k != "stack"},
                                      "stack": " < ".join(s["stack"])})
        elif job["tool"] == "native":
            index[jid] = {**base, **{k: v for k, v in summ.items() if k != "phases"}}
            native_rows.append({**base, **{k: v for k, v in summ.items() if k != "phases"}})
            for p in summ["phases"]:
                native_phase_rows.append({**base, "wait_policy": summ["wait_policy"], **p})

    suite = {}
    if (run_dir / "suite.json").is_file():
        suite = common.read_json(run_dir / "suite.json")
    common.write_json(out / "summary.json", {"run_dir": str(run_dir), "suite": {
        k: v for k, v in suite.items() if k != "load_samples"}, "load_samples": suite.get(
        "load_samples", []), "jobs": index, "errors": errors})
    write_csv(out / "jobs.csv", jobs_rows)
    write_csv(out / "phases.csv", phase_rows)
    write_csv(out / "functions.csv", fn_rows)
    write_csv(out / "lines.csv", line_rows)
    write_csv(out / "omp_regions.csv", omp_rows)
    write_csv(out / "threads.csv", thr_rows)
    write_csv(out / "cache.csv", cache_rows)
    write_csv(out / "dhat_sites.csv", dhat_rows)
    write_csv(out / "native.csv", native_rows)
    write_csv(out / "native_phases.csv", native_phase_rows)
    print(f"summarized {len(results)} jobs -> {out}" + (f"; {len(errors)} errors: {errors}" if errors else ""))
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
