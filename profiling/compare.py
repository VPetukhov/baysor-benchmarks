#!/usr/bin/env python3
"""Compare two profiling runs (e.g. before/after an optimization).

    python profiling/compare.py RUN_A RUN_B [--jobs PATTERN]
        [--top 15] [--min-pct 0.5] [--csv out.csv]

RUN_A / RUN_B are run directories (or their summary/ subdirectories);
summaries are created with summarize.py first if missing. For every job
present in both runs it prints:

  * callgrind: total Ir, per-phase Ir and per-function exclusive/inclusive Ir
    with % deltas (B vs A); functions are the union of both runs' top lists;
  * dhat: peak heap, total allocated bytes/blocks;
  * native: min/median wall and CPU seconds and peak RSS (wall/CPU marked as
    load-sensitive, with each run's load average).

Deltas of instruction counts are load-independent; a single-threaded job
re-run on the same binary differs by < 0.1 %.
"""
from __future__ import annotations

import argparse
import csv
import fnmatch
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import profcommon as common  # noqa: E402


def summary_dir(p: str) -> Path:
    d = Path(p).resolve()
    if (d / "summary.json").is_file():
        return d
    if (d / "summary" / "summary.json").is_file():
        return d / "summary"
    if (d / "raw").is_dir():
        import summarize
        summarize.main([str(d)])
        return d / "summary"
    raise SystemExit(f"{p}: no summary.json and no raw/ directory")


def pct(a, b):
    if a in (None, 0) or b is None:
        return None
    return 100.0 * (b - a) / a


def fmt_pct(v):
    return "   n/a" if v is None else f"{v:+7.2f}%"


def fmt_n(v):
    if v is None:
        return "n/a"
    v = float(v)
    for unit, div in (("G", 1e9), ("M", 1e6), ("k", 1e3)):
        if abs(v) >= div:
            return f"{v / div:.3f}{unit}"
    return f"{v:.0f}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_a")
    ap.add_argument("run_b")
    ap.add_argument("--jobs", default="*", help="glob on job ids (default: all common jobs)")
    ap.add_argument("--top", type=int, default=15, help="functions per job")
    ap.add_argument("--min-pct", type=float, default=0.5,
                    help="only functions with >= this %% of total Ir in A or B")
    ap.add_argument("--csv", default=None, help="write all deltas to this CSV")
    args = ap.parse_args(argv)

    da, db = summary_dir(args.run_a), summary_dir(args.run_b)
    sa, sb = common.read_json(da / "summary.json"), common.read_json(db / "summary.json")
    common_jobs = sorted(j for j in sa["jobs"] if j in sb["jobs"]
                         and fnmatch.fnmatch(j, args.jobs))
    print(f"A: {da}\nB: {db}\n{len(common_jobs)} common jobs\n")
    rows = []
    for jid in common_jobs:
        ja, jb = sa["jobs"][jid], sb["jobs"][jid]
        tool = ja["tool"]
        if tool in ("callgrind", "cachegrind"):
            ca = common.read_json(da / "callgrind" / f"{jid}.json")
            cb = common.read_json(db / "callgrind" / f"{jid}.json")
            ta, tb = ca["total_Ir"], cb["total_Ir"]
            print(f"== {jid}: total Ir {fmt_n(ta)} -> {fmt_n(tb)} {fmt_pct(pct(ta, tb))}"
                  f"  (abs diff {tb - ta:+d})")
            rows.append({"job": jid, "kind": "total", "name": "Ir", "a": ta, "b": tb,
                         "delta_pct": pct(ta, tb)})
            pa = {p["phase"]: p for p in ca["phases"]}
            pb = {p["phase"]: p for p in cb["phases"]}
            names = list(pa) + [n for n in pb if n not in pa]
            print(f"   {'phase':28s} {'A Ir':>12s} {'B Ir':>12s} {'delta':>9s}  {'A %':>6s} {'B %':>6s}")
            for n in names:
                a = pa.get(n, {}).get("Ir")
                b = pb.get(n, {}).get("Ir")
                print(f"   {n:28s} {fmt_n(a):>12s} {fmt_n(b):>12s} {fmt_pct(pct(a, b)):>9s}  "
                      f"{pa.get(n, {}).get('pct', 0):6.2f} {pb.get(n, {}).get('pct', 0):6.2f}")
                rows.append({"job": jid, "kind": "phase", "name": n, "a": a, "b": b,
                             "delta_pct": pct(a, b)})
            fa = {r["fn"]: r for r in ca["top_self"] + ca["top_incl"]}
            fb = {r["fn"]: r for r in cb["top_self"] + cb["top_incl"]}
            fns = [f for f in dict.fromkeys(list(fa) + list(fb))
                   if max(fa.get(f, {}).get("self_pct", 0), fb.get(f, {}).get("self_pct", 0),
                          fa.get(f, {}).get("incl_pct", 0), fb.get(f, {}).get("incl_pct", 0))
                   >= args.min_pct]
            fns.sort(key=lambda f: -max(fa.get(f, {}).get("self_Ir", 0), fb.get(f, {}).get("self_Ir", 0)))
            print(f"   {'function (by exclusive Ir)':50s} {'excl A':>10s} {'excl B':>10s} {'delta':>9s}"
                  f" {'incl A':>10s} {'incl B':>10s} {'delta':>9s}")
            for f in fns[:args.top]:
                ra, rb = fa.get(f, {}), fb.get(f, {})
                name = (ra or rb)["name"][:50]
                print(f"   {name:50s} {fmt_n(ra.get('self_Ir')):>10s} {fmt_n(rb.get('self_Ir')):>10s} "
                      f"{fmt_pct(pct(ra.get('self_Ir'), rb.get('self_Ir'))):>9s} "
                      f"{fmt_n(ra.get('incl_Ir')):>10s} {fmt_n(rb.get('incl_Ir')):>10s} "
                      f"{fmt_pct(pct(ra.get('incl_Ir'), rb.get('incl_Ir'))):>9s}")
            for f in fns:
                ra, rb = fa.get(f, {}), fb.get(f, {})
                rows.append({"job": jid, "kind": "function_self", "name": (ra or rb)["name"],
                             "a": ra.get("self_Ir"), "b": rb.get("self_Ir"),
                             "delta_pct": pct(ra.get("self_Ir"), rb.get("self_Ir"))})
                rows.append({"job": jid, "kind": "function_incl", "name": (ra or rb)["name"],
                             "a": ra.get("incl_Ir"), "b": rb.get("incl_Ir"),
                             "delta_pct": pct(ra.get("incl_Ir"), rb.get("incl_Ir"))})
            print()
        elif tool == "dhat":
            print(f"== {jid}:")
            for k in ("peak_bytes", "total_bytes", "total_blocks"):
                print(f"   {k:14s} {fmt_n(ja[k]):>10s} -> {fmt_n(jb[k]):>10s} {fmt_pct(pct(ja[k], jb[k]))}")
                rows.append({"job": jid, "kind": "dhat", "name": k, "a": ja[k], "b": jb[k],
                             "delta_pct": pct(ja[k], jb[k])})
            print()
        elif tool == "native":
            print(f"== {jid} (wall/CPU are load-sensitive; load A {ja.get('loadavg_1m_mean')}, "
                  f"B {jb.get('loadavg_1m_mean')}):")
            for k in ("wall_s_min", "wall_s_median", "cpu_s_min", "cpu_s_median", "peak_rss_kb"):
                print(f"   {k:14s} {ja.get(k)!s:>10s} -> {jb.get(k)!s:>10s} {fmt_pct(pct(ja.get(k), jb.get(k)))}")
                rows.append({"job": jid, "kind": "native", "name": k, "a": ja.get(k),
                             "b": jb.get(k), "delta_pct": pct(ja.get(k), jb.get(k))})
            print()
    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["job", "kind", "name", "a", "b", "delta_pct"])
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
