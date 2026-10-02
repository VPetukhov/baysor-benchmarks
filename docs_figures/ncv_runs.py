#!/usr/bin/env python3
"""Re-run the segmentation-example crops with NCV colours on.

The benchmark campaign ran Baysor with ``--skip-ncv-color``, so its outputs
have no ``ncv_color`` column. This script re-runs the exact harness command of
``runs/rc1636-real/<dataset>/rep0`` for the crops shown in the docs'
segmentation examples (``SEG_EXAMPLES`` in ``make_figures.py``), with the
release-candidate binary and without ``--skip-ncv-color``, one process at a
time. Outputs go to ``$BAYSOR_BENCH_DATA/docs-figures/ncv/<dataset>/``;
finished datasets are skipped.

    python3 docs_figures/ncv_runs.py            # ~3 min on the E5-2670 host
    python3 docs_figures/ncv_runs.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

DATA = Path(os.environ.get("BAYSOR_BENCH_DATA", "/home/vpetukhov/Projects/Baysor/.bench-data"))
BINARY = DATA / "binaries/baysor-rc-cba8ca3"   # output-identical to cpp-0.9.0
OUT = DATA / "docs-figures/ncv"
DATASETS = ["xenium_pancreas_377_quick", "merfish_ileum_quick", "cosmx_nsclc_lung5_rep1_quick",
            "iss_mouse_hippocampus_quick"]


def command(ds: str, out: Path) -> list[str]:
    rep = json.loads((DATA / "runs/rc1636-real" / ds / "metrics.json").read_text())["reps"][0]
    cmd = list(rep["command"])
    cmd[0] = str(BINARY)
    cmd = [c for c in cmd if c != "--skip-ncv-color"]
    cmd[cmd.index("-o") + 1] = str(out / "seg")
    return cmd


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--threads", type=int, default=6)
    args = ap.parse_args()
    for ds in DATASETS:
        out = OUT / ds
        cmd = command(ds, out)
        if args.dry_run:
            print(" ".join(cmd))
            continue
        if (out / "run.json").is_file():
            print("done:", ds)
            continue
        (out / "seg").mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        with open(out / "baysor.log", "w") as log:
            res = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT,
                                 env={**os.environ, "OMP_NUM_THREADS": str(args.threads)})
        if res.returncode != 0:
            raise SystemExit(f"{ds}: exit {res.returncode}, see {out / 'baysor.log'}")
        (out / "run.json").write_text(json.dumps({
            "dataset": ds, "binary": str(BINARY), "command": cmd, "threads": args.threads,
            "wall_s": round(time.time() - t0, 2), "loadavg_start": os.getloadavg()[0]}, indent=1))
        print(f"{ds}: {time.time() - t0:.1f} s", flush=True)


if __name__ == "__main__":
    main()
