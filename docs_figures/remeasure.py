#!/usr/bin/env python3
"""Re-measurement, one process at a time, of the wall times plotted in the Baysor docs.

The release-candidate A/B runs (``rc/2026-10-01-1636267``) ran while other
jobs loaded the host, which inflates wall time.  This script re-runs, one
process at a time, the exact harness commands of those runs (taken from
``runs/<run_id>/<dataset>/metrics.json``) for

* a thread sweep (1, 2, 4, 8, 16 threads) on two datasets, and
* the real full-tier datasets at the harness default of 6 threads,

for both binaries (RC = cpp-0.9.0 candidate, and cpp-0.8.3), interleaving the
two versions so that load drift affects both alike.  Threads are set the way
the harness sets them (``OMP_NUM_THREADS`` & co.; the RC reads it as its
fallback thread count).  Each run is timed with ``/usr/bin/time -v``; the
1-minute load average is recorded at start and end.  Results are appended to
``$BAYSOR_BENCH_DATA/docs-figures/remeasure/runs.tsv``; finished rows are
skipped on restart.  Segmentation outputs are deleted after each run (only
``run.log`` is kept).

Usage:  remeasure.py [--dry-run] [--only sweep|full]
"""
import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

DATA = Path(os.environ.get("BAYSOR_BENCH_DATA",
                           "/home/vpetukhov/Projects/Baysor/.bench-data"))
OUT = DATA / "docs-figures" / "remeasure"
VERSIONS = {"rc": "rc1636-real", "old": "v0083-real"}
SWEEP = [("xenium_pancreas_377_quick", [1, 2, 4, 8, 16]),
         ("merfish_ileum_full", [1, 2, 4, 8, 16])]
FULL = ["merfish_ileum_full", "xenium_breast_rep1_dense_full",
        "xenium_pancreas_377_full", "cosmx_nsclc_lung5_rep1_full",
        "xenium_prime5k_ovarian_full"]
FIELDS = ["exp", "dataset", "version", "threads", "start", "end", "load1_start",
          "load1_end", "exit", "wall_s", "user_s", "sys_s", "rss_kb", "binary"]


def harness_command(version: str, dataset: str) -> list:
    m = json.loads((DATA / "runs" / VERSIONS[version] / dataset / "metrics.json").read_text())
    return list(m["reps"][0]["command"])


def load1() -> str:
    return Path("/proc/loadavg").read_text().split()[0]


def parse_time_v(text: str) -> dict:
    def grab(pat):
        mm = re.search(pat, text)
        return mm.group(1) if mm else ""
    wall = grab(r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): ([\d:.]+)")
    secs = 0.0
    for part in wall.split(":"):
        secs = secs * 60 + float(part)
    return {"wall_s": f"{secs:.2f}", "user_s": grab(r"User time \(seconds\): ([\d.]+)"),
            "sys_s": grab(r"System time \(seconds\): ([\d.]+)"),
            "rss_kb": grab(r"Maximum resident set size \(kbytes\): (\d+)")}


def plan():
    for ds, threads in SWEEP:
        for t in threads:
            for v in ("rc", "old"):
                yield "sweep", ds, v, t
    for ds in FULL:
        for v in ("rc", "old"):
            yield "full", ds, v, 6


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", choices=["sweep", "full"])
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    tsv = OUT / "runs.tsv"
    done = set()
    if tsv.is_file():
        for r in csv.DictReader(open(tsv), delimiter="\t"):
            if r["exit"] == "0":
                done.add((r["exp"], r["dataset"], r["version"], int(r["threads"])))
    for exp, ds, v, t in plan():
        if args.only and exp != args.only:
            continue
        if (exp, ds, v, t) in done:
            continue
        cmd = harness_command(v, ds)
        odir = OUT / exp / ds / f"{v}-t{t}"
        if odir.exists():
            shutil.rmtree(odir)
        odir.mkdir(parents=True)
        cmd[cmd.index("-o") + 1] = str(odir / "seg")
        env = dict(os.environ)
        for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            env[var] = str(t)
        print(f"[{time.strftime('%H:%M:%S')}] {exp} {ds} {v} t={t} load={load1()}", flush=True)
        if args.dry_run:
            print("  " + " ".join(cmd))
            continue
        l0, t0 = load1(), int(time.time())
        p = subprocess.run(["/usr/bin/time", "-v"] + cmd, env=env, cwd=odir,
                           stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        l1, t1 = load1(), int(time.time())
        (odir / "time.txt").write_text(p.stderr[-4000:])
        log = odir / "seg" / "run.log"
        if log.is_file():
            shutil.copy(log, odir / "run.log")
        shutil.rmtree(odir / "seg", ignore_errors=True)
        row = {"exp": exp, "dataset": ds, "version": v, "threads": t, "start": t0,
               "end": t1, "load1_start": l0, "load1_end": l1, "exit": p.returncode,
               "binary": cmd[0], **parse_time_v(p.stderr)}
        new = not tsv.is_file()
        with open(tsv, "a", newline="") as fh:
            w = csv.DictWriter(fh, FIELDS, delimiter="\t")
            if new:
                w.writeheader()
            w.writerow(row)
        print(f"  -> exit {p.returncode} wall {row['wall_s']} s rss {row['rss_kb']} kB "
              f"load {l0}->{l1}", flush=True)


if __name__ == "__main__":
    main()
