# Baysor benchmark suite

Regression and quality benchmarks for the Baysor segmentation algorithm on
cropped real datasets and simulated datasets with known ground truth.

The suite answers two questions:

1. **No algorithm change** (refactoring, performance work, bug fixes that must
   not alter results): simulated-data metrics stay within the noise floor of the
   stored baseline, and real-data segmentations stay essentially the same as
   the stored baseline segmentation.
2. **Algorithm change**: accuracy on simulated data goes up, and the cellAdmix
   admixture audit on real data is not worse than the baseline.

Data is never committed. Code, dataset manifests and suite definitions are;
benchmark/validation **results and baselines are local** — they live under
`$BAYSOR_BENCH_DATA` (see [Results and baselines are
local](#results-and-baselines-are-local)).

## Layout

```
baysor-benchmarks/
  README.md              this file: layout, workflow and the dataset contract
  DATASETS.md            generated inventory of every dataset + coverage matrix
  LICENSE                same license as Baysor (copied from it)
  environment.yml        Python environment for the suite
  datasets/              manifests: one YAML per dataset group (sim, real_xenium,
                         real_other) + suites.yaml (the regular/release suites)
  baysor-configs/        vendored copies of Baysor's shipped configs
                         (see "Relation to Baysor")
  simulate/              generators for simulated datasets
  fetch/                 download + crop scripts for real datasets
  harness/               runner, metrics, baseline comparison, reports
  celladmix/             cellAdmix admixture audit on a segmentation
  profiling/             load-robust CPU/memory profiling suite (see
                         "Profiling" below)
```

## Relation to Baysor

This repository is the benchmark suite split out of the [Baysor](https://github.com/VPetukhov/baysor)
repository (formerly its `benchmarks/` directory; the history was preserved
with `git filter-repo --subdirectory-filter benchmarks`): <https://github.com/VPetukhov/baysor-benchmarks>.
It works standalone — nothing in this tree expects a Baysor source checkout:

* **The Baysor binary is always explicit**: every run takes `--baysor` /
  `--binary` or the `BAYSOR_BIN` environment variable; there is no built-in
  default binary.
* **Baysor configs are vendored**: dataset specs (`datasets/*.yaml`) and
  `meta.json` reference Baysor's shipped configs as `configs/<name>.toml`,
  and the harness resolves those references to the byte-identical copies in
  [`baysor-configs/`](baysor-configs/) — copied from Baysor commit
  `69be9805b8d8854e6c30c4e4068d56197c9dfeff` (`cpp-dev-llm`):
  `xenium.toml`, `iss.toml`, `osm_fish.toml`, `starmap.toml`,
  `example_config.toml`. A change of Baysor's shipped configs therefore
  never silently changes benchmark runs.
* **cellAdmix patches live outside git**: the four patches needed to build
  the pinned cellAdmix-core (see [`celladmix/INSTALL.md`](celladmix/INSTALL.md))
  are applied by `celladmix/install.sh` from `$CELLADMIX_PATCHES` (default
  `<data dir>/celladmix/patches/`); patches 0002 and 0003 are also proposed
  upstream in [kharchenkolab/cellAdmix-core#2](https://github.com/kharchenkolab/cellAdmix-core/issues/2).
* **License**: [`LICENSE`](LICENSE) is copied from Baysor (MIT).

Baysor's own testing docs point back to this repository; keep the two in
sync only through explicit binary + baseline choices, never through shared
trees.

Baseline metric JSONs, `resources.csv`, `SUMMARY.md`, run outputs and
validation reports are **not** in this tree: they live under
`$BAYSOR_BENCH_DATA` (see [Data location](#data-location)).

## Dataset inventory

[`DATASETS.md`](DATASETS.md) lists every dataset (id, kind, tier,
platform/generator, tissue/scenario, genes, molecules, area, cells/mm²,
density and gene-panel class, 2D/3D, prior, images, `admixture_capable`,
source, notes) grouped by kind and platform, plus the density × gene-panel
coverage matrix per kind. It also carries the measured **resource columns**
(6-thread CPU time ± SD, 6-thread wall and peak RAM, 1-thread wall/RAM,
cellAdmix audit time — from the local
`$BAYSOR_BENCH_DATA/baselines/bugfixes-35e8a7e/resources.csv`,
regenerated with [`harness/resources.py`](harness/resources.py) from the
existing runs; `TODO` = never measured, never guessed) and a **Suite**
column (membership in [`datasets/suites.yaml`](datasets/suites.yaml)).
Regenerate it with:

```bash
$PY harness/inventory.py            # writes DATASETS.md
$PY harness/inventory.py --check    # exit 1 when stale
```

Validate the datasets against this contract (columns/dtypes/sort, meta
fields and enums, stats consistency, image/prior/config references,
manifest sha256) with:

```bash
$PY harness/validate_datasets.py    # exit 1 on metadata errors
$PY harness/validate_datasets.py --strict   # data findings too
```

## Workflow

See [`harness/README.md`](harness/README.md) for the benchmark workflow:
running datasets (`run.py`), metric definitions, baselines and the measured
noise floor (including Baysor's determinism findings), and the
`--expect same` / `--expect improved` comparison (`compare.py`, one-shot
`bench.sh`).

## Profiling

Load-robust CPU and memory profiling of `baysor run` lives in
[`profiling/`](profiling/) — quick and report tiers under Valgrind
(callgrind/cachegrind/DHAT), a scaling tier with gperftools/heaptrack on
whole-slide ladders, and a `compare.py` for before/after runs. Start with
[`profiling/README.md`](profiling/README.md) (tool install into `.deps/`,
profiling build of Baysor, tiers, output layout). Results go to
`$BAYSOR_BENCH_DATA/profiling/` (default `<repo>/.bench-data/profiling/`)
and are never committed.

## How to test a change

Official baselines of the current algorithm (stored under
`$BAYSOR_BENCH_DATA/baselines/`, never in git):

| baseline | flavor | contents |
|---|---|---|
| `$BAYSOR_BENCH_DATA/baselines/cpp-dev-llm-b8bba9d-t1/` | `identical` | quick tier, 1 thread, 1 replicate, no cellAdmix, NCV colours skipped; binary `.bench-data/binaries/baysor-cpp-dev-llm-b8bba9d` (Baysor `cpp-dev-llm` @ b8bba9d, includes the edge-order fix, so results do not depend on output paths or run-id length); used by the `regular` and `release` identical steps |
| `$BAYSOR_BENCH_DATA/baselines/bugfixes-35e8a7e-t1/` | `identical` | previous 1-thread baseline (binary `baysor-bugfixes-35e8a7e`, before the edge-order fix: path-length sensitive); kept for reference |
| `$BAYSOR_BENCH_DATA/baselines/bugfixes-35e8a7e/` | noise floor | quick + full tier, 6 threads, 3 replicates (full tier: see its README), cellAdmix audit with stable typing |

### Deterministic baselines

Baysor on the `perf-optimization` branch is bitwise deterministic at any
thread count: replicates of such a run are identical and carry no noise
information (SD = 0). A baseline created from a single replicate of that
deterministic binary can gate `--expect same` (and `improved`) on sim and
real datasets, with the tolerances taken from the calibrated floors alone:

```bash
# freeze a deterministic run (verifies pairwise-identical replicates when
# there are >= 2, accepts a single one, records "deterministic": true)
$PY harness/baseline.py create --run-id <run> --name <name> \
    --deterministic [--force]
# compare against it: a 1-replicate baseline satisfies the replicate-count
# requirement for sim and real datasets; tolerance = floor (k*SD, SD = 0)
$PY harness/compare.py --run-id <run> --baseline <name> --expect same
```

Reports state `deterministic baseline: tolerance = floor` (meta line,
gated-metrics table note and gated check rows). Baselines created without
`--deterministic` behave exactly as before (replicate-count requirement,
pooled-SD tolerances). `$BAYSOR_BENCH_DATA/baselines/perf-7c2b936/` is such
a baseline, frozen from the run `perf7c2b936-noise`; details in
[`harness/README.md`](harness/README.md) "Deterministic baselines".

Setup used by every command below (from the repository root):

```bash
PY=.deps/bench/bin/python           # see "Environment" below
B=/path/to/baysor                   # explicit build of the Baysor sources
```

**NCV colours are skipped by default.** Every benchmark command line the
harness builds (`run.py`/`bench.sh`, `fetch/sanity_run.sh`,
`fetch/other.py smoke`, `simulate/sanity.py`, the cellAdmix validation
scripts) passes `--skip-ncv-color` to Baysor when the binary supports it:
the neighbourhood composition colours are never compared, hashed or read
by any metric, yet cost 58–76 % of instructions on panels below 1,000 genes
(`$BAYSOR_BENCH_DATA/profiling/reports/`). Opt out with `--ncv-color`
(`run.py`, `bench.sh`, `fetch/other.py smoke`, `simulate/sanity.py`) or
`NCV_COLOR=1` (shell scripts); the effective choice is recorded per
replicate in `run.json` (`skip_ncv_color`). Comparisons ignore colours
entirely — `--expect identical` hashes the normalized assignment table — so
runs with and without colours are comparable; metrics and thresholds are
unchanged. One caveat from the preserved binary's layout sensitivity (see
the harness README "Determinism findings"): on a few datasets
(`xenium_pancreas_377_quick`, `xenium_breast_rep1_stroma_quick`,
`xenium_mouse_brain_ff_quick`) the flag's presence alone changes the
1-thread segmentation, so `--expect identical` against a *colour-created*
baseline only holds on flag-stable datasets (verified: `sim_circles_*`,
`strec_dense_s2_disjoint`, `iss_mouse_hippocampus_quick`); recreate
baselines under the new default or opt the step out with `ncv_color: true`.

### The suites (`datasets/suites.yaml`)

Two committed suites (schema in [`harness/suites.py`](harness/suites.py),
resolution via `run.py --suite` / `compare.py --suite`). The times are
estimates from
`$BAYSOR_BENCH_DATA/baselines/bugfixes-35e8a7e/resources.csv`
(measured Baysor wall/CPU × replicates + cellAdmix audit; reproduce with
`$PY harness/suites.py --suite <name>`):

| suite | steps (compare mode) | coverage | est. wall | est. CPU |
|---|---|---|---|---|
| `regular` | `exact`: 1 thr × 1 rep → `identical` vs `-t1`; `noise`: 6 thr × 1 rep + audit → `same` vs `bugfixes-35e8a7e` | 4-dataset bitwise subset; 23-dataset coverage list | 3.8 + 14.4 = **18.1 min** core (+ ~1–2 min metrics/typing ≈ **~20 min**) | ~45 CPU-min |
| `release` | `quick6`+`full6`: 6 thr × 3 rep + audit → `same` (one run folder); `quick1`: 1 thr × 1 rep → `identical` | every dataset (78 = 65 quick + 13 full); the quick tier again at 1 thread | 156.7 + 232.4 + 87.9 = **477 min ≈ 8 h** (+ metrics bookkeeping; the historical `benchbase-b` quick+full passes observed ≈ 7 h against the 389 min core) | ~24 CPU-h |

* `regular` runs with `--no-ami`: AMI is informational (no gate reads it)
  but costs ~30–60 s of metrics time per sim replicate; `release` computes
  AMI so regenerated baselines keep their current contents.
* `regular` coverage: every `trivial.py` scenario **with and without
  prior** (12 datasets); st-recoverability sparse and dense; one 3D
  simulation; every real platform at quick tier (Xenium, CosMx, MERFISH,
  ISS, osmFISH, STARmap); gene-panel classes `tiny`–`huge` through cheap
  quick crops (`strec_*` tiny, `sim_circles_gaps_g100`/ISS small, pancreas
  medium, CosMx/STARmap large, `xenium_prime5k_ovarian_quick` huge).
* **The `*_admix` crop: yes, one fits** — `xenium_lung_cancer_admix`
  (~97 s/rep Baysor + ~6 s audit ≈ 103 s of the ~20 min budget) joins the
  noise step, so the cellAdmix audit and the admixture gate
  (`compare --expect improved`, `--celltypes-from bugfixes-35e8a7e` fixed
  typing/pairs) are exercised on a full-size admixture crop in every
  regular run; the quick Xenium crops in the list exercise the audit on
  quick data too. Without it the gate would still evaluate on
  `admixture_capable` quick crops, but never on the crop class the audit
  was calibrated for.
* Steps sharing a `group` run in the same folder `runs/<id>`; the group
  holding the suite's `identical` step keeps the bare `--run-id`
  (1-thread output-path-length sensitivity: keep it ≤ 17 characters),
  other groups get `<id>-<group>` (`<id>-noise`).

Resolve both suites without running Baysor (validates dataset ids,
baselines, run-ids and prints the estimates):

```bash
$PY harness/run.py --suite regular --run-id dry --dry-run
$PY harness/run.py --suite release --run-id dry --dry-run
$PY harness/suites.py --suite regular     # plan + estimates only
```

### Step 0: the C++ unit tests (part of `regular`)

```bash
cmake --preset tests && cmake --build --preset tests
ctest --test-dir build/tests --output-on-failure
# ~8 s (Release) / ~90 s (coverage build)
```

### Every change/PR: the `regular` suite (~20 min)

```bash
harness/bench.sh --baysor $B --preset regular --run-id reg-1
```

Runs both steps and compares every group (`exact` → `identical`, `noise`
→ `same`); exit 0 = pass, 1 = regression, 2 = setup error. `same` keeps
its strict *unchanged-binary* rule per check, but the **suite verdict
downgrades a `same` group whose only failing checks are the
`binary_sha256` provenance rows** (the normal result of a rebuilt binary
whose metrics all stayed within the noise floor) and prints a note — the
bitwise `identical` gate is then the verdict on behaviour.

Variants (override the non-bitwise group's mode):

* **algorithm change** (assignments are *supposed* to change):
  `bench.sh --preset regular --expect improved` — judges only the noise
  group (mean gain > noise + no per-dataset regression + admixture gate
  on `xenium_lung_cancer_admix`); the bitwise group is skipped by design.
* **harness/data/config change, binary untouched**:
  `bench.sh --preset regular --expect same` — strict mode, the sha gate
  fails if the binary really changed.
* **refactoring/bug fix that must not change results**: the default
  invocation; `identical` must pass and the noise metrics must stay
  within tolerance.

The legacy single-step presets `refactor` (1 thr × 1 rep, `identical`) and
`algorithm` (6 thr × 3 rep, `improved`) still exist for ad-hoc runs — see
[`harness/README.md`](harness/README.md).

### Before a release: the `release` suite (~8 h)

```bash
# algorithm release: full noise-floor run judged on improvement
harness/bench.sh --baysor $B --preset release --run-id rel-1 \
    --expect improved
# unchanged-algorithm release: default (--expect same) + bitwise gate
harness/bench.sh --baysor $B --preset release --run-id rel-2
```

Everything: all quick and full datasets at 6 threads × 3 replicates with
the cellAdmix audit (on the `*_admix` crops as on every real dataset),
plus 1-thread `identical` over the whole quick tier. After it passes,
freeze the new baselines (below).

Validate suite resolution without running Baysor (also shown above):
`run.py --suite <name> --dry-run` prints every step's datasets, threads,
replicates, timeouts and estimated time.

### Updating the baselines after an accepted change

```bash
# rerun both configurations with the new binary
$PY harness/run.py --baysor $B --datasets quick --run-id benchbase-b \
    --replicates 3 --threads 6 --timeout 1800 --skip-existing \
    --celltypes-from bugfixes-35e8a7e --label <new-sha>
$PY harness/run.py --baysor $B --datasets quick --run-id benchbase-t1 \
    --replicates 1 --threads 1 --timeout 1800 --no-celladmix \
    --skip-existing --label <new-sha>
# freeze them (adds --allow-incomplete/--identical as needed; see baseline.py)
$PY harness/baseline.py create --run-id benchbase-b \
    --name bugfixes-<new-sha> --force
$PY harness/baseline.py create --run-id benchbase-t1 \
    --name bugfixes-<new-sha>-t1 --force --identical
$PY harness/baseline_summary.py --baseline bugfixes-<new-sha>
```

(`--skip-existing` only reuses replicates whose binary sha256, thread
count and scale factor match, so a new binary reruns everything; the full
tier runs are appended by repeating the command with `--datasets full`
`--timeout 5400`.) See
`$BAYSOR_BENCH_DATA/baselines/bugfixes-35e8a7e/README.md`
for the exact official-baseline invocations of this binary.

## Results and baselines are local

Nothing committed to this repository is a run result: datasets, run outputs, baseline
metric JSONs and validation reports all live under `$BAYSOR_BENCH_DATA`
(gitignored, never committed — see [Data location](#data-location)):

```
$BAYSOR_BENCH_DATA/
  baselines/<name>/        baseline metric JSONs + README/SUMMARY.md +
                           resources.csv, and <dataset>/rep<k>/ assignment
                           tables, celltypes.parquet, fixed_pairs.json
  results/celladmix/       cellAdmix validation summaries + audit JSONs
  results/simulate/        sanity.py reports (sanity_check*.json)
```

The harness reads and writes every one of these files there; the
`regular`/`release` suites and their `--expect identical|same|improved`
comparisons work unchanged.

**Recreate a baseline** with the release suite and the preserved official
binary (kept in the data dir and passed explicitly): the 6-thread group
freezes as `<name>`, the 1-thread bitwise group as `<name>-t1`:

```bash
harness/bench.sh --suite release \
    --baysor .bench-data/binaries/baysor-bugfixes-35e8a7e \
    --create-baseline bugfixes-35e8a7e
```

`--create-baseline` implies `--force`: `baseline.py` swaps the new baseline
in atomically and keeps the old one on any error. To only resolve the plan
— datasets, run-ids, baseline and resources-CSV locations — without running
Baysor: `bench.sh --suite release --dry-run`.

## Data location

All data lives under `$BAYSOR_BENCH_DATA` (default: `<repo>/.bench-data`,
gitignored), in one directory per dataset:

```
$BAYSOR_BENCH_DATA/
  sim/<dataset_id>/...
  real/<dataset_id>/...
  runs/<run_id>/<dataset_id>/...     harness outputs (Baysor results, metrics)
  baselines/<name>/...               baseline metric JSONs, resources.csv,
                                     assignment tables, cell types (see above)
  results/                           validation reports (celladmix, simulate)
  cache/                             raw downloads kept for re-cropping (may be deleted)
```

## Dataset contract

Every dataset directory contains:

### `molecules.parquet` (required)

| column | type | required | meaning |
|---|---|---|---|
| `x`, `y` | float64 | yes | coordinates in µm |
| `z` | float64 | no | µm; present only for 3D datasets |
| `gene` | string (dictionary ok) | yes | gene name; control probes and blank codewords already removed |
| `qv` | float32 | no | vendor quality value, if available (already filtered by the fetch script) |
| `prior` | int32 | no | prior segmentation label for Baysor's `:prior` option (0 = no prior), e.g. the vendor nucleus id |
| `cell_vendor` | string | real only, if available | the vendor's cell assignment (empty = unassigned), kept for reference comparisons |
| `cell` | int32 | sim only | ground-truth cell id (0 = background / noise molecule) |
| `interior` | bool | sim only, optional | molecule counts toward accuracy metrics; excludes edge effects. Defaults to all true |
| `celltype` | string | sim only, optional | ground-truth cell type of the true cell |

Rows are sorted by (`y`, `x`). No other columns are required. Generators may
add extra columns prefixed with `aux_`.

### `meta.json` (required)

```json
{
  "id": "xenium_pancreas_377_crop1",
  "kind": "real",
  "tier": "quick",
  "platform": "Xenium",
  "source": {"url": "...", "doi": "...", "license": "...", "original_dataset": "...", "retrieved": "2026-09-29"},
  "crop": {"bbox_um": [x0, y0, x1, y1], "z_range_um": null, "note": "why this region"},
  "stats": {"n_molecules": 0, "n_genes": 0, "area_um2": 0.0, "molecules_per_um2": 0.0,
            "n_vendor_cells": 0, "vendor_cells_per_mm2": 0.0},
  "difficulty": {"cell_density": "sparse|medium|dense", "gene_panel": "tiny|small|medium|large|huge", "notes": ""},
  "baysor": {"scale_um": 5.0, "scale_std": "25%", "min_molecules_per_cell": 20,
             "prior": "none" ,
             "prior_confidence": 0.5,
             "config": "configs/xenium.toml",
             "extra_args": []},
  "images": [{"name": "dapi", "file": "images/dapi.tif", "pixel_size_um": 0.2125, "origin_um": [x0, y0]}],
  "truth": null
}
```

- `kind` is either `real` or `sim`.
- `tier` is either `quick` (at most about 150k molecules, runs in about a
  minute) or `full` (at most about 3M molecules).
- `baysor.prior` is one of:
  - `"none"`
  - `"column"`: use the `prior` column as `:prior`
  - `"image:<relative path>"`: a label TIFF in the dataset directory, same
    pixel frame as `images`
- `images` is optional. It holds cropped DAPI and membrane/boundary stains, when
  the source has them. They are used by cellAdmix membrane scoring and by future
  image-aware methods.
- For `sim`, `truth` holds the generator name and version or commit, all
  parameters, the seed and, when available, the oracle (best achievable)
  assignment accuracy. For `real` it stays `null`.
- `gene_panel` classes:

  | class | genes |
  |---|---|
  | `tiny` | < 50 |
  | `small` | 50–250 |
  | `medium` | 250–700 |
  | `large` | 700–2000 |
  | `huge` | > 2000 |

- `cell_density` classes, by nucleus or cell density:

  | class | cells/mm² |
  |---|---|
  | `sparse` | < 2500 |
  | `medium` | 2500–7000 |
  | `dense` | > 7000 |

### Optional files

- `images/*.tif`: cropped stains, single-channel, uint16 or uint8.
- `reference/`: vendor cell or nucleus boundaries for the crop (parquet), or a
  reference annotation.
- `README.md`: provenance notes specific to the dataset.

## Environment and local state

The Python environment used by the whole suite is created from the
repository root:

```bash
micromamba create -p .deps/bench -f environment.yml
```

Two locations in this repository hold local state and are gitignored
(`.gitignore`); create them fresh or symlink existing ones:

```bash
# Python env: create from environment.yml (above), or reuse an existing one
ln -s /path/to/existing/.deps/bench .deps/bench

# Data dir: either symlink an existing data dir (datasets, runs, baselines)
ln -s /path/to/existing/.bench-data .bench-data
# ...or leave it absent: fetch/ and simulate/ create <repo>/.bench-data/
# on first use (fetch downloads the datasets, simulate generates them),
# and harness/ runs + baselines are written there too.
```

Everything else (the Baysor binary, micromamba itself, the cellAdmix build
toolchain) comes in explicitly: `--baysor` / `BAYSOR_BIN`,
`BENCH_PYTHON`/`BENCH_PY`, and `BENCH_DEPS` for `celladmix/install.sh`.
