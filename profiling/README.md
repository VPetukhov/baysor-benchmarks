# Baysor profiling suite

Load-robust CPU and memory profiling of `baysor run` on a shared, busy host.
The numbers that conclusions rest on do not depend on how busy the machine
is:

| measure | tool | load-sensitive? |
|---|---|---|
| instructions (Ir) per phase / function / source line, per thread | Valgrind callgrind | no (deterministic at 1 thread; < 0.1 % run-to-run) |
| simulated I1/D1/LL cache misses per function and line | Valgrind cachegrind `--cache-sim=yes` | no |
| heap peak, live-at-peak and churn by allocation site | Valgrind DHAT; heaptrack (scaling tier) | no |
| peak RSS | native run, `wait4` rusage | practically no |
| CPU seconds per phase, CPU share per function | `/proc` sampling; gperftools CPU profiler (scaling tier) | mildly |
| wall time, speedup | native runs, repeated, min/median + load average | **yes** |

Two tiers:

* **quick** (`profile.py`, default, < 60 min wall at load ~15–20):
  small deterministic crops (5–40k molecules) of the benchmark datasets
  under callgrind / cachegrind / DHAT, plus native timing and thread series.
  `--suite report` is the extended job set behind the first full report
  (~80 min at that load: thread series, `--plot` and re-run on the 20k crop,
  a 20k whole-transcriptome crop, DHAT on most datasets).
* **scaling** (`scaling.py`, ≤ 4 h, unattended, resumable per rung): nested
  size ladders of whole Xenium slides (0.1M → 12M molecules) with
  low-overhead sampling (gperftools, `/proc`), heaptrack at selected rungs,
  and callgrind on the lowest rungs as a cross-check.

Nothing here writes into the Baysor source tree: crops, raw tool outputs,
summaries and reports go to `$BAYSOR_BENCH_DATA/profiling/` (default
`<repo>/.bench-data/profiling/`, git-ignored, never committed).

## Layout

```
profiling/
  profiling.yaml      dataset spec (crops), suites (job sets), phase functions,
                      scaling ladders
  make_crops.py       deterministic quick-tier crops -> profiling/data/<id>/
  scaling_crops.py    nested size ladders from vendor transcripts.parquet
  profile.py          quick-tier runner (one command)
  scaling.py          scaling-tier runner + summarizer (exponent fits, plots)
  summarize.py        raw outputs -> summary/*.json, *.csv
  report_tables.py    summary -> markdown tables for REPORT.md
  compare.py          two runs -> per-phase / per-function % deltas
  cgparse.py          callgrind output parser
  gperf.py            gperftools CPU profile parser + symbolizer
  procmon.py          native runner with /proc CPU/RSS sampling + phase alignment
  profcommon.py       paths (data root, tools, optional Baysor checkout), spec loading
```

Output layout under `$BAYSOR_BENCH_DATA/profiling/`:

```
data/<id>/                     quick-tier crops (molecules.parquet + meta.json)
scaling/data/<slide>_<rung>/   scaling ladders
runs/<run-id>/raw/<job>/       raw callgrind.out.*, dhat.out.json, cpu.prof,
                               heaptrack.*, baysor/valgrind logs, job.json
runs/<run-id>/summary/         summary.json + CSVs + per-job JSON
reports/<YYYY-MM-DD>-<sha>/    REPORT.md, summary copies, plots
```

## Setup

Paths the tools need (all overridable):

| what | option | env | default |
|---|---|---|---|
| profiling binary | `--baysor` | `BAYSOR_BIN` | `<baysor-src>/build/profiling/baysor` when a source checkout is given, else required |
| Baysor source checkout (git sha of runs; default binary location) | `--baysor-src` | `BAYSOR_SRC` | none — optional; configs, data and tools never come from it |
| data root | `--data-root` | `BAYSOR_BENCH_DATA` | `<repo>/.bench-data` |
| valgrind | `--valgrind` | `VALGRIND` | `<repo>/.deps/vgenv/bin/valgrind`, else `valgrind` on PATH |
| gperftools / heaptrack prefix | `--proftools` | `PROFTOOLS` | `<repo>/.deps/proftools` |

Baysor configs are **not** taken from a Baysor tree: the suite runs datasets
through `../harness`, which resolves each dataset's `configs/<name>.toml`
reference to this repository's vendored byte-identical copies in
[`../baysor-configs/`](../baysor-configs/) (see the top-level README,
"Relation to Baysor").

The Python environment is the benchmark one (`environment.yml`, `.deps/bench`);
the scripts import `../harness` (command construction, exactly as the
benchmark harness runs each dataset) and `../fetch` (transcript filters for
the scaling ladders).

### Tools (no root needed)

From the repository root, with any conda-forge capable `micromamba`/`conda`:

```bash
# Valgrind 3.27 (conda-forge) works on glibc 2.43 / kernel 7.0
micromamba create -y -p .deps/vgenv -c conda-forge valgrind
# gperftools (libprofiler, libtcmalloc) + heaptrack for the scaling tier
micromamba create -y -p .deps/proftools -c conda-forge gperftools heaptrack
```

`perf` is not needed (it is unusable on the shared host:
`perf_event_paranoid=4`).

### Profiling build

In a Baysor checkout, with its `profiling` preset (Release + `-g`):

```bash
cmake --preset profiling && cmake --build --preset profiling
# -> build/profiling/baysor
```

Release code generation plus `-g` (identical machine code: the `.text` of
`build/profiling/baysor` and a plain Release `build/baysor` have the same
sha256). `-fno-omit-frame-pointer` is **not** used: Valgrind does not need
frame pointers and gperftools unwinds with libunwind, so the profiled code
stays identical to Release. Equivalent, when dependencies should be
bootstrapped by `configure.sh`:

```bash
CXXFLAGS=-g CFLAGS=-g ./configure.sh --deps=conda --deps-dir=.deps \
    --build-dir=build/profiling --build
```

## Running

```bash
BENCH=/path/to/baysor-benchmarks      # this repository
PY=$BENCH/.deps/bench/bin/python
BAYSOR=/path/to/baysor/build/profiling/baysor   # the profiling build
cd $BENCH

$PY profiling/profile.py --baysor $BAYSOR --dry-run   # job plan: data dirs,
                                                      # tools, configs, jobs
$PY profiling/profile.py --baysor $BAYSOR             # quick suite (run id <date>-<sha>)
$PY profiling/profile.py --baysor $BAYSOR --run-id rerun1 \
    --only callgrind --datasets xenium_pancreas_g377_20k   # a subset
$PY profiling/report_tables.py .bench-data/profiling/runs/<run-id>

$PY profiling/scaling_crops.py              # ladders (once, ~15 min)
$PY profiling/scaling.py plan --baysor $BAYSOR        # job list + estimates
$PY profiling/scaling.py run --baysor $BAYSOR [--run-id ID]   # resumable
$PY profiling/scaling.py summarize .bench-data/profiling/runs/<ID>
```

(`.bench-data` above is the default data root; with `BAYSOR_BENCH_DATA`
set, use `$BAYSOR_BENCH_DATA/profiling/...` instead.)

Re-running a command skips finished jobs (`job.json` with `status: ok`);
`--force` re-runs them.

### What the quick suite does

1. `make_crops.py`: crops listed in `profiling.yaml` (cached; recreated only
   when the spec entry changes).
2. Native thread series (`when: before`): 1/2/4/8/16 threads × 3 reps on
   two datasets, alone on the host (as far as a shared host allows), plus
   8 threads with `OMP_WAIT_POLICY=passive`.
3. Valgrind pool: at most 7 concurrent Valgrind processes, longest first
   (ordering from `native_s_hint` in the spec), plus one lane of native
   1-thread runs (8 heavy processes in total; the lane's slot goes to the
   pool when the lane is done):
   * callgrind, 1 thread, every dataset (`--separate-threads=yes`,
     `--dump-before=<phase function>`);
   * callgrind at 2/4/8/16 threads on one dataset;
   * callgrind with `--plot` (HTML report phase) and an identical re-run
     (reproducibility check);
   * cachegrind `--cache-sim=yes` (I1/D1/LL misses with the host's cache
     geometry; `callgrind --cache-sim=yes` was ~5× slower);
   * DHAT on several datasets and at 16 threads.
4. `summarize.py` → `runs/<run-id>/summary/`.

Valgrind jobs run with `OMP_WAIT_POLICY=passive`: Valgrind serializes the
threads of a process, so spinning OpenMP workers would burn simulated
instructions while the thread that owns the work waits for its time slice.
Multi-threaded jobs add `--fair-sched=yes`. Instruction counts do not depend
on how many Valgrind jobs run concurrently.

## Reading the results

* `summary/phases.csv`: per job and phase, Ir, % of the run, serial vs
  parallel-region Ir and the Amdahl bound. Phases are delimited by
  `--dump-before` on the phase functions in `profiling.yaml` (patterns on
  demangled names; mind `[abi:cxx11]` tags and return types of templates, and
  avoid patterns that also match per-iteration lambdas). A phase's
  main-thread cost is the inclusive cost of the calls from `cmd_run` (or its
  segmentation lambda, `phase_callers`) into the phase function, summed over
  the dump parts (callgrind writes the in-part cost of calls still active at
  a dump), so a phase function called elsewhere (e.g. `build_molecule_graph`
  inside the confidence estimate) stays in its caller's phase;
  `glue_after_<phase>` is `cmd_run` code between phases (cell statistics,
  count-matrix assembly). Callgrind writes only the triggering (main) thread
  at a dump and the other threads once at exit, so OpenMP regions are mapped
  to phases through their outlining function (reachable from the phase
  function in the main thread's call graph) and worker costs are split in
  proportion to the main thread's cost of the region per phase; worker time
  outside regions is `omp_worker_runtime`, non-OpenMP threads (Arrow I/O)
  `other_threads`. The phases add up to the total Ir.
  Note: `--dump-before` and `--dump-after` on the same function do not
  combine in callgrind 3.27 (only one of them fires), hence before-only.
* *Serial* = main-thread instructions outside OpenMP outlined bodies
  (`*._omp_fn.N`); *parallel* = instructions inside them (main + workers).
  Amdahl bound = 1 / (s + (1 − s)/p) with s = serial share of the 1-thread
  run. Arrow's own thread pool (parquet I/O) is not OpenMP and counts as
  serial.
* `summary/functions.csv`: top 60 functions by exclusive and by inclusive
  Ir per job, with the hottest source line and the dominant call path
  (heaviest caller chain); `lines.csv`: hottest lines of the top functions.
* `summary/omp_regions.csv`: per parallel region, Ir per thread and
  max/mean over threads (load imbalance). Caveat: under Valgrind threads run
  one at a time, so `schedule(dynamic)` chunk distribution is not the native
  one; static schedules are faithful.
* `summary/cache.csv`: per function D1/LL misses (cachegrind job; no call
  graph, so no phases).
* `summary/dhat_sites.csv`: allocation sites (innermost non-allocator frames)
  by bytes live at the global heap peak (`t-gmax`), total bytes and number of
  allocations (churn).
* `summary/native.csv`, `native_phases.csv`: wall and CPU seconds (min,
  median of the reps), CPU/wall, peak RSS (`VmHWM` of the process: the
  `wait4` rusage of a child forked from Python also counts the Python image
  before `exec`), the 1-minute load average at the start of each rep, and
  per phase wall/CPU from `/proc` sampling aligned with Baysor's log lines.
  Wall and CPU/wall are load-sensitive.

Open a raw profile interactively with `callgrind_annotate` /
`kcachegrind` (`runs/<run-id>/raw/callgrind-*/callgrind.out.*`, one file per
dump part and thread) or DHAT's viewer (`dh_view.html`, load
`dhat.out.json`).

## Scaling tier

```bash
$PY profiling/scaling_crops.py               # ladders, once (~2 min lung, ~2 min prime5k)
$PY profiling/scaling.py plan --baysor $BAYSOR      # jobs + rough estimates + resolved tools
$PY profiling/scaling.py run --baysor $BAYSOR [--run-id ID]   # resumable: finished jobs are skipped
$PY profiling/scaling.py summarize .bench-data/profiling/runs/<ID>
```

* Ladders (`profiling.yaml` → `scaling.slides`): nested squares around the
  median molecule of a whole Xenium slide, cut from the vendor
  `transcripts.parquet` with the benchmark fetch filters (`qv >= 20`, real
  genes, vendor nucleus prior); `all` = the whole slide. Baysor flags and
  scale come from the slide's reference dataset (`xenium_lung_cancer_admix`,
  `xenium_prime5k_ovarian_full`), so every rung runs with identical
  parameters. `extra_datasets` (e.g. `cosmx_wtx_colon_full`) run as single
  large points.
* `gperf` jobs: native run with `LD_PRELOAD=libprofiler.so`,
  `CPUPROFILE_FREQUENCY=100`, `TCMALLOC_STACKTRACE_METHOD=libunwind` (the
  default frame-pointer unwinder gives 1–2-frame stacks without frame
  pointers). Samples are ITIMER_PROF, i.e. CPU time of all threads (the sample
  count matches rusage CPU time within 1 %). `gperf.py` symbolizes with
  `nm` and the separate debug files of stripped system libraries
  (`/usr/lib/debug/.build-id`, needed for libm's `pow`/`exp` internals).
  The OpenMP runtime's barrier/spin functions are reported as
  "OpenMP wait % of CPU" (the default wait policy spins).
* `/proc` sampling every second gives per-phase CPU seconds, CPU/wall and
  RSS, aligned with the phase log lines (short phases get interpolated CPU).
* `heaptrack` jobs: heaptrack serializes allocations (a whole-slide run uses
  ~1.5 cores at 8 threads, and is several times slower than native), so the
  scheduler counts them as 2 cores. Analysis is one
  `heaptrack_print --merge-backtraces 0` text pass (fast); allocation sites
  are the innermost non-STL/allocator frames, as for DHAT.
* `callgrind` jobs on the lowest rungs cross-check the sampled CPU shares
  against instruction counts (`scaling_xcheck.csv`).
* Scheduler: at most `--max-procs` (8) processes and `--cores` threads
  (default 18: two 8-thread jobs + two 1-thread jobs); multi-threaded jobs
  are started first so single-thread jobs cannot starve them.
* `summarize` writes `scaling_summary.json`, CSVs (jobs, phases, fits,
  cross-check, heaptrack), `scaling_tables.md` and PNG plots: exponents are
  least-squares slopes of log(CPU seconds) and log(peak RSS) over
  log(molecules) per slide and thread count, for the whole run, every phase
  and the top functions; exponent ≥ 1.15 is flagged super-linear, a function
  whose CPU share grows ≥ 1.5× over the ladder is flagged as growing.

## Comparing two commits

```bash
# build each commit's profiling binary (see "Profiling build"), then
$PY profiling/profile.py --baysor /path/A/build/profiling/baysor --run-id A
$PY profiling/profile.py --baysor /path/B/build/profiling/baysor --run-id B
$PY profiling/compare.py .bench-data/profiling/runs/A \
    .bench-data/profiling/runs/B [--jobs 'callgrind-*-t1'] [--csv deltas.csv]
```

`compare.py` prints for every common job the total Ir delta, per-phase Ir
deltas and per-function exclusive/inclusive deltas (union of both runs' top
lists), DHAT peak/total deltas and native timings (flagged as
load-sensitive). A 1-thread callgrind job of an unchanged binary reproduces
its total instruction count to < 0.1 %, so smaller deltas are noise.
For a quicker loop restrict to the jobs that matter, e.g.
`--only callgrind --datasets xenium_pancreas_g377_20k`.
