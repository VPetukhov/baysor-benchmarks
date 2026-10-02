# docs_figures — figures and numbers of the Baysor "Performance" docs

Regenerates every figure of the Baysor documentation pages
`docs/performance/benchmarks.md` and `docs/performance/profiling.md`, and the
numbers quoted in their tables, from the benchmark and profiling results under
`$BAYSOR_BENCH_DATA` (default `/home/vpetukhov/Projects/Baysor/.bench-data`).

```bash
# all figures -> <Baysor checkout>/docs/performance/img/, numbers -> generated/
.deps/bench/bin/python docs_figures/make_figures.py --docs /path/to/Baysor/docs/performance

# a subset
.deps/bench/bin/python docs_figures/make_figures.py --docs ... --only fig_scaling fig_phases
```

The run takes about two minutes (the UMAP embeddings are cached in
`generated/umap_*.csv`; delete them to recompute).

## Inputs

| input | produced by |
|---|---|
| `rc/2026-10-01-1636267/tables/{runtime,quality_sim,quality_real}.csv` | the cpp-0.9.0 release-candidate A/B campaign vs cpp-0.8.3 (`scripts/aggregate.py` in that directory; its `REPORT.md` describes the runs) |
| `runs/{rc1636,v0083}-{real,sim,t1,t1b}/<dataset>/` | `harness/run.py` (per-replicate `metrics.json`, segmentation outputs, cellAdmix audits) |
| `docs-figures/remeasure/runs.tsv` | `docs_figures/remeasure.py` (below) |
| `profiling/reports/2026-10-01-20bc45c/summary-*-run/` | the profiling suite (`profiling/`), after the optimisation |
| `profiling/reports/2026-09-30-e45fddc/summary-scaling-run/` | the same suite, before the optimisation |
| `real/<id>/meta.json`, `sim/<id>/meta.json` | dataset facts (platform, genes) |

## Outputs

* `<docs>/img/<name>-light.svg` and `<name>-dark.svg` for every chart (MkDocs
  Material shows one or the other via `#only-light` / `#only-dark`),
  `umap-{light,dark}.png`, and `segmentation_examples.png` (one dark-background
  image for both themes, palettised to keep it small);
* `generated/tables.md`: every number quoted in the pages, as tables, plus a
  table mapping each figure/table to its source files;
* `generated/numbers.json`: the same numbers, machine-readable.

Colours follow the dataviz reference palette (blue / orange / aqua for up to
three categories, which validates all-pairs for colour-vision deficiencies in
both themes; secondary encoding by marker shape or line style; the previous
version in neutral grey or open marks).

## Re-measurement, one process at a time

The A/B campaign ran two benchmark streams at once while other jobs loaded the
shared host (1-minute load 5–21), which inflates wall times. `remeasure.py` re-runs the exact harness
command lines of that campaign, one process at a time and alternating the two
versions, for

* a thread sweep (1, 2, 4, 8, 16 threads) on `xenium_pancreas_377_quick` and
  `merfish_ileum_full`, and
* the real 2M-molecule crops and `merfish_ileum_full` at 6 threads,

and appends wall/CPU/RSS and the load average to
`$BAYSOR_BENCH_DATA/docs-figures/remeasure/runs.tsv` (finished rows are
skipped on restart; segmentation outputs are deleted, `run.log` is kept).

```bash
python3 docs_figures/remeasure.py            # ~1.5 h on the E5-2670 host
python3 docs_figures/remeasure.py --dry-run  # print the commands
```

In the run of 2026-10-01/02 (23:28–01:08) the thread sweep saw a 1-minute
load of 2–13 (below 5 for the whole Xenium sweep). Other jobs then raised the
load to 6–22 during the 6-thread full-tier runs, so those are a second sample
at about the A/B campaign's load, not a quiet-host measurement; the docs say
so.
