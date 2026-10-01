#!/usr/bin/env python3
"""One-command Baysor profiling suite (quick tier).

    BENCH=/path/to/baysor-benchmarks          # this repository
    PY=$BENCH/.deps/bench/bin/python
    $PY profiling/profile.py --baysor /path/to/baysor/build/profiling/baysor
    $PY profiling/profile.py --baysor <binary> --run-id myrun --jobs 8
    $PY profiling/profile.py --baysor <binary> --only callgrind \
        --datasets xenium_pancreas_g377_20k
    $PY profiling/profile.py --baysor <binary> --dry-run   # print the job plan

Steps: (1) derive the profiling crops (make_crops.py, cached); (2) run the
native jobs one at a time (load-sensitive wall/CPU time, peak RSS, per-phase
CPU seconds from /proc sampling); (3) run the Valgrind jobs (callgrind,
callgrind cache simulation, DHAT) in a pool of at most --jobs processes,
longest first; (4) summarize (summarize.py) into <run>/summary/.

Raw outputs go to the data root's profiling/runs/<run-id>/raw/<job>/
($BAYSOR_BENCH_DATA, default <repo>/.bench-data; ignored by git). A finished
job (job.json with exit_code 0) is skipped on re-runs unless --force, so an
interrupted suite resumes where it stopped.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.append(str(HERE.parent / "harness"))   # ../harness (the benchmark harness)
import profcommon as common  # noqa: E402
import make_crops  # noqa: E402
import procmon  # noqa: E402

BENCH: Path = common.repo_root()   # configs (vendored) and subprocess cwd
DEFAULT_VALGRIND = "valgrind"
# rough Valgrind slowdown vs native 1-thread wall, only used to order jobs
TOOL_FACTOR = {"callgrind": 25.0, "cachegrind": 40.0, "dhat": 30.0}


@dataclass
class Job:
    tool: str
    dataset: str
    threads: int
    plot: bool = False
    tag: str = ""
    reps: int = 1
    wait_policy: str = ""     # OMP_WAIT_POLICY for native jobs ("" = libgomp default)
    when: str = "lane"        # native: "before" the Valgrind pool or in the side "lane"
    est_s: float = 0.0
    extra: dict = field(default_factory=dict)

    @property
    def id(self) -> str:
        s = f"{self.tool}-{self.dataset}-t{self.threads}"
        if self.plot:
            s += "-plot"
        if self.wait_policy:
            s += f"-{self.wait_policy}"
        if self.tag:
            s += f"-{self.tag}"
        return s


def load_harness_modules():
    """../harness/{common,run}.py: the harness' command builder."""
    import common as hcommon   # harness/common.py (sys.path appended above)
    import run as hrun
    return hcommon, hrun


def expand_suite(spec: dict, suite: str, only_tools, only_datasets) -> list[Job]:
    s = spec["suites"][suite]
    all_ids = [d["id"] for d in spec["datasets"]]
    jobs: list[Job] = []
    for tool in ("native", "callgrind", "cachegrind", "dhat"):
        if only_tools and tool not in only_tools:
            continue
        for entry in s.get(tool, []) or []:
            dsets = all_ids if entry["datasets"] == "all" else entry["datasets"]
            for ds in dsets:
                if only_datasets and ds not in only_datasets:
                    continue
                for t in entry["threads"]:
                    jobs.append(Job(tool=tool, dataset=ds, threads=int(t),
                                    plot=bool(entry.get("plot", False)),
                                    tag=str(entry.get("tag", "")),
                                    reps=int(entry.get("reps", 1)),
                                    wait_policy=str(entry.get("wait_policy", "")),
                                    when=str(entry.get("when", "lane"))))
    # de-duplicate (a dataset may be listed twice)
    seen, out = set(), []
    for j in jobs:
        if j.id not in seen:
            seen.add(j.id)
            out.append(j)
    return out


def thread_env(threads: int, valgrind: bool, wait_policy: str = "") -> dict:
    env = os.environ.copy()
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS"):
        env[var] = str(threads)
    env.pop("OMP_WAIT_POLICY", None)
    if valgrind:
        # Valgrind serializes threads: spinning workers would burn simulated
        # instructions while the thread holding the work is descheduled
        env["OMP_WAIT_POLICY"] = "passive"
    elif wait_policy:
        env["OMP_WAIT_POLICY"] = wait_policy
    return env


def valgrind_args(job: Job, jd: Path, spec: dict, valgrind: str) -> list[str]:
    if job.tool == "cachegrind":
        # Valgrind's cachegrind: simulated I1/D1/LL caches (host geometry),
        # per function and line; no call graph, much cheaper than
        # callgrind --cache-sim=yes
        return [valgrind, "--tool=cachegrind", "--cache-sim=yes",
                f"--cachegrind-out-file={jd}/cachegrind.out"]
    if job.tool == "callgrind":
        args = [valgrind, "--tool=callgrind", "--separate-threads=yes",
                f"--callgrind-out-file={jd}/callgrind.out"]
        if job.threads > 1:
            args += ["--fair-sched=yes"]
        seen = set()
        for ph in spec["phases"]:
            if ph["pattern"] not in seen:
                seen.add(ph["pattern"])
                args.append(f"--dump-before={ph['pattern']}")
        return args
    if job.tool == "dhat":
        args = [valgrind, "--tool=dhat", f"--dhat-out-file={jd}/dhat.out.json"]
        if job.threads > 1:
            args += ["--fair-sched=yes"]
        return args
    raise ValueError(job.tool)


def job_done(jd: Path) -> bool:
    p = jd / "job.json"
    if not p.is_file():
        return False
    try:
        return common.read_json(p).get("status") == "ok"
    except (OSError, json.JSONDecodeError):
        return False


def run_native_job(job: Job, jd: Path, baysor: Path, ds_dir: Path, probe, hrun, hcommon) -> dict:
    reps = []
    for r in range(job.reps):
        rd = jd / f"rep{r}"
        if rd.exists():
            shutil.rmtree(rd)
        rd.mkdir(parents=True)
        ds = hcommon.load_dataset(ds_dir)
        cmd = hrun.build_command(baysor, ds, rd / "seg", probe, BENCH,
                                   skip_ncv_color=False)
        if job.plot:
            cmd.append("--plot")
        env = thread_env(job.threads, valgrind=False, wait_policy=job.wait_policy)
        res = procmon.run_monitored(cmd, env, rd / "baysor.log", interval=0.1, cwd=str(BENCH))
        res["phases"] = procmon.phase_table(res, rd / "baysor.log")
        res["rep"] = r
        common.write_json(rd / "procmon.json", res)
        reps.append({k: res[k] for k in ("rep", "exit_code", "wall_s", "cpu_s", "cpu_user_s",
                                          "cpu_sys_s", "peak_rss_kb", "loadavg_start",
                                          "loadavg_end")})
        shutil.rmtree(rd / "seg", ignore_errors=True)
    ok = all(r["exit_code"] == 0 for r in reps)
    return {"status": "ok" if ok else "failed", "reps": reps,
            "command": cmd, "env": {k: env.get(k) for k in ("OMP_NUM_THREADS", "OMP_WAIT_POLICY")}}


def run_valgrind_job(job: Job, jd: Path, baysor: Path, ds_dir: Path, probe, hrun, hcommon,
                     spec: dict, valgrind: str) -> dict:
    for p in list(jd.glob("callgrind.out*")) + list(jd.glob("cachegrind.out*")):
        p.unlink()
    ds = hcommon.load_dataset(ds_dir)
    cmd = hrun.build_command(baysor, ds, jd / "seg", probe, BENCH,
                             skip_ncv_color=False)
    if job.plot:
        cmd.append("--plot")
    vg = valgrind_args(job, jd, spec, valgrind)
    env = thread_env(job.threads, valgrind=True)
    res = procmon.run_monitored(vg + cmd, env, jd / "valgrind.log", interval=5.0, cwd=str(BENCH))
    shutil.rmtree(jd / "seg", ignore_errors=True)
    res.pop("samples", None)
    return {"status": "ok" if res["exit_code"] == 0 else "failed",
            "command": vg + cmd,
            "env": {k: env.get(k) for k in ("OMP_NUM_THREADS", "OMP_WAIT_POLICY")},
            **{k: res[k] for k in ("exit_code", "wall_s", "cpu_s", "peak_rss_kb",
                                   "loadavg_start", "loadavg_end", "t_start", "t_end")}}


class LoadSampler(threading.Thread):
    """Samples /proc/loadavg every `interval` s for the whole suite."""

    def __init__(self, interval=30.0):
        super().__init__(daemon=True)
        self.interval, self.samples, self._halt = interval, [], threading.Event()

    def run(self):
        while not self._halt.is_set():
            self.samples.append([round(time.time(), 1)] + procmon.loadavg())
            self._halt.wait(self.interval)

    def stop(self):
        self._halt.set()


def host_info(baysor: Path, valgrind: str, src: Optional[Path] = None) -> dict:
    info = {"hostname": platform.node(), "nproc": os.cpu_count(),
            "kernel": platform.release(), "python": platform.python_version()}
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                info["cpu"] = line.split(":", 1)[1].strip()
                break
        mem = Path("/proc/meminfo").read_text().split("\n")[0]
        info["mem_total_kb"] = int(mem.split()[1])
    except OSError:
        pass
    for key, cmd in (("glibc", ["ldd", "--version"]),
                     ("valgrind", [valgrind, "--version"])):
        try:
            info[key] = subprocess.run(cmd, capture_output=True, text=True).stdout.splitlines()[0]
        except (OSError, IndexError):
            info[key] = None
    cache = baysor.parent / "CMakeCache.txt"
    if cache.is_file():
        keep = ("CMAKE_BUILD_TYPE", "CMAKE_CXX_FLAGS", "CMAKE_CXX_FLAGS_RELEASE",
                "CMAKE_C_FLAGS", "CMAKE_CXX_COMPILER", "BAYSOR_WITH_REPORTING")
        flags = {}
        for line in cache.read_text().splitlines():
            k = line.split(":", 1)[0]
            if k in keep:
                flags[k] = line.split("=", 1)[1]
        info["build"] = flags
        cxx = flags.get("CMAKE_CXX_COMPILER")
        if cxx:
            try:
                info["compiler"] = subprocess.run([cxx, "--version"], capture_output=True,
                                                  text=True).stdout.splitlines()[0]
            except (OSError, IndexError):
                pass
    if src is not None:      # git sha of the profiled Baysor sources (optional)
        info["baysor_src"] = str(src)
        try:
            info["git_sha"] = subprocess.run(["git", "-C", str(src), "rev-parse", "HEAD"],
                                             capture_output=True, text=True).stdout.strip()
            info["git_dirty"] = bool(subprocess.run(
                ["git", "-C", str(src), "status", "--porcelain", "--untracked-files=no"],
                capture_output=True, text=True).stdout.strip())
        except OSError:
            pass
    return info


def print_job_plan(spec_path: Path, root: Path, run_dir: Path, baysor: Path,
                   valgrind: Optional[Path], src: Optional[Path], suite: str,
                   jobs: list[Job], needed: list[str]) -> None:
    """--dry-run: the fully resolved plan (data dirs, tools, configs, jobs)."""
    data_dir = root / "profiling" / "data"
    print("resolved:")
    print(f"  suite:     {suite}")
    print(f"  spec:      {spec_path}")
    print(f"  data root: {root}")
    print(f"  crops:     {data_dir}")
    print(f"  run dir:   {run_dir}")
    print(f"  baysor:    {baysor}")
    print(f"  baysor-src: {src or 'not given (only used for the run git sha)'}")
    print(f"  valgrind:  {valgrind or 'not found'}")
    cfg_dir = BENCH / "baysor-configs"
    print(f"  configs:   {cfg_dir} (vendored, resolved by the harness)")
    print("  datasets:")
    try:
        import common as hcommon   # harness/common.py (sys.path appended above)
    except ImportError as exc:
        hcommon = None
        print(f"    (harness import failed: {exc}; config resolution skipped)")
    for ds in needed:
        meta_p = data_dir / ds / "meta.json"
        if not meta_p.is_file():
            print(f"    {ds}: crop not built yet (make_crops.py runs first)")
            continue
        cfg_ref = (common.read_json(meta_p).get("baysor") or {}).get("config")
        if not cfg_ref:
            print(f"    {ds}: no -c config")
            continue
        if hcommon is None:
            print(f"    {ds}: {cfg_ref}")
            continue
        cfg_path = hcommon.baysor_config_path(str(cfg_ref), repo=hcommon.repo_root())
        mark = "" if cfg_path.is_file() else "  [MISSING]"
        print(f"    {ds}: {cfg_ref} -> {cfg_path}{mark}")
    print("jobs:")
    for j in jobs:
        print(f"  {j.id}  (reps={j.reps})" if j.tool == "native" else f"  {j.id}")
    print(f"{len(jobs)} jobs, datasets: {', '.join(needed)}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", default="quick")
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
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--run-id", default=None, help="default: <date>-<shortsha>")
    ap.add_argument("--jobs", type=int, default=8, help="max concurrent Valgrind processes (<= 8)")
    ap.add_argument("--only", default=None, help="comma-separated tools: native,callgrind,cachegrind,dhat")
    ap.add_argument("--datasets", default=None, help="comma-separated dataset ids")
    ap.add_argument("--filter", default=None, help="glob on job ids, e.g. 'callgrind-*-t1'")
    ap.add_argument("--force", action="store_true", help="re-run finished jobs")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-summary", action="store_true")
    args = ap.parse_args(argv)
    if args.jobs > 8:
        ap.error("--jobs is capped at 8 to keep the shared host usable")

    only_tools = set(args.only.split(",")) if args.only else None
    t_suite0 = time.time()
    src = common.baysor_src(args.baysor_src)
    spec = common.load_spec(Path(args.spec))
    root = common.bench_root(args.data_root)
    if args.baysor:
        baysor = Path(args.baysor).expanduser().resolve()
    elif src is not None:
        baysor = (src / "build" / "profiling" / "baysor").resolve()
    else:
        ap.error("baysor binary unknown: pass --baysor or set BAYSOR_BIN (build it, see README.md)")
    valgrind = common.find_valgrind(args.valgrind or os.environ.get("VALGRIND"))
    if valgrind is None:
        if not args.dry_run and any(t != "native" for t in (only_tools or {"callgrind"})):
            ap.error("valgrind not found: pass --valgrind or set VALGRIND (see README.md)")
    else:
        args.valgrind = str(valgrind)
    if not baysor.is_file():
        ap.error(f"baysor binary not found: {baysor} (build it, see README.md)")
    sha = common.run_sha(src, baysor)
    run_id = args.run_id or f"{time.strftime('%Y-%m-%d')}-{sha}"
    run_dir = root / "profiling" / "runs" / run_id
    only_ds = set(args.datasets.split(",")) if args.datasets else None
    jobs = expand_suite(spec, args.suite, only_tools, only_ds)
    if args.filter:
        import fnmatch
        jobs = [j for j in jobs if fnmatch.fnmatchcase(j.id, args.filter)]

    needed = sorted({j.dataset for j in jobs})
    by_id = {d["id"]: d for d in spec["datasets"]}
    if args.dry_run:
        print_job_plan(Path(args.spec), root, run_dir, baysor, valgrind, src,
                       args.suite, jobs, needed)
        return 0

    # 1. crops
    data_dir = root / "profiling" / "data"
    for ds in needed:
        make_crops.make_crop(by_id[ds], root, data_dir, force=False)

    hcommon, hrun = load_harness_modules()
    probe = hrun.probe_binary(baysor)
    run_dir.mkdir(parents=True, exist_ok=True)
    info = host_info(baysor, args.valgrind or DEFAULT_VALGRIND, src)
    info.update({"run_id": run_id, "suite": args.suite, "baysor": str(baysor),
                 "baysor_sha256": probe["sha256"], "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                 "jobs_parallel": args.jobs, "argv": sys.argv})
    sampler = LoadSampler(30.0)
    sampler.start()

    def record(job: Job, res: dict, jd: Path) -> None:
        res.update({"job": job.id, "tool": job.tool, "dataset": job.dataset,
                    "threads": job.threads, "plot": job.plot, "tag": job.tag,
                    "wait_policy": job.wait_policy})
        common.write_json(jd / "job.json", res)

    def run_native(j: Job) -> str:
        jd = run_dir / "raw" / j.id
        if job_done(jd) and not args.force:
            return f"[skip] {j.id}"
        jd.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        res = run_native_job(j, jd, baysor, data_dir / j.dataset, probe, hrun, hcommon)
        record(j, res, jd)
        return (f"[native] {j.id}: {res['status']} wall "
                f"{[r['wall_s'] for r in res['reps']]} s ({time.time() - t0:.0f} s)")

    # 2. native jobs marked `when: before`, one at a time, nothing else running
    for j in (j for j in jobs if j.tool == "native" and j.when == "before"):
        print(run_native(j), time.strftime("%H:%M:%S"), procmon.loadavg(), flush=True)

    # 3. valgrind jobs, pool of (jobs - 1) longest first, plus one native lane
    native_wall = {}
    for d in (run_dir / "raw").glob("native-*-t1"):
        try:
            jj = common.read_json(d / "job.json")
            native_wall[jj["dataset"]] = min(r["wall_s"] for r in jj["reps"])
        except (OSError, KeyError, ValueError):
            pass
    vjobs = [j for j in jobs if j.tool != "native"]
    for j in vjobs:
        base = native_wall.get(j.dataset) or by_id[j.dataset].get(
            "native_s_hint", 20.0 * by_id[j.dataset]["n_molecules"] / 20000)
        j.est_s = base * TOOL_FACTOR[j.tool] * (1.0 + 0.05 * (j.threads - 1)) * (1.3 if j.plot else 1.0)
    vjobs.sort(key=lambda j: -j.est_s)
    lane = [j for j in jobs if j.tool == "native" and j.when != "before"]

    def run_one(j: Job) -> str:
        jd = run_dir / "raw" / j.id
        if job_done(jd) and not args.force:
            return f"[skip] {j.id}"
        jd.mkdir(parents=True, exist_ok=True)
        res = run_valgrind_job(j, jd, baysor, data_dir / j.dataset, probe, hrun, hcommon,
                               spec, args.valgrind)
        res["est_s"] = round(j.est_s, 1)
        record(j, res, jd)
        return f"[{j.tool}] {j.id}: {res['status']} {res['wall_s']:.0f} s (est {j.est_s:.0f} s)"

    n_pool = max(1, args.jobs - (1 if lane else 0))
    sem = threading.Semaphore(n_pool)

    def run_lane() -> str:
        for j in lane:
            print(run_native(j), time.strftime("%H:%M:%S"), procmon.loadavg(), flush=True)
        if args.jobs > n_pool:
            sem.release()          # the lane's slot goes to the Valgrind pool
        return "[lane] native lane done"

    with cf.ThreadPoolExecutor(max_workers=n_pool + 1) as pool:
        futs = []
        if lane:
            futs.append(pool.submit(run_lane))

        def gated(j: Job) -> str:
            with sem:
                return run_one(j)
        with cf.ThreadPoolExecutor(max_workers=args.jobs) as vpool:
            futs += [vpool.submit(gated, j) for j in vjobs]
            for f in cf.as_completed(futs):
                print(f.result(), time.strftime("%H:%M:%S"), procmon.loadavg(), flush=True)

    sampler.stop()
    info["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    info["suite_wall_s"] = round(time.time() - t_suite0, 1)
    info["load_samples"] = sampler.samples
    info["jobs"] = [j.id for j in jobs]
    prev = run_dir / "suite.json"
    if prev.is_file():   # a resumed / extended run: keep the earlier invocations
        old = common.read_json(prev)
        info["previous_invocations"] = old.pop("previous_invocations", []) + [
            {k: v for k, v in old.items() if k != "load_samples"}]
        info["load_samples"] = old.get("load_samples", []) + info["load_samples"]
    common.write_json(run_dir / "suite.json", info)
    print(f"suite wall: {info['suite_wall_s'] / 60:.1f} min -> {run_dir}")

    # 4. summaries
    if not args.no_summary:
        import summarize
        summarize.main([str(run_dir), "--spec", args.spec])
    return 0


if __name__ == "__main__":
    sys.exit(main())
