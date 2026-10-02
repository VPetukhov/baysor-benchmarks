# docs_figures — figures, chart data and tables of the Baysor "Performance" docs

Regenerates everything data-driven on the Baysor documentation pages
`docs/performance/benchmarks.md` and `docs/performance/profiling.md`, plus an
owner-only profiling report, from the benchmark and profiling results under
`$BAYSOR_BENCH_DATA` (default `/home/vpetukhov/Projects/Baysor/.bench-data`).

```bash
# once: re-run the four segmentation-example crops with NCV colours on (~2 min)
python3 docs_figures/ncv_runs.py

# everything: chart data, images, the per-dataset table, the owner report
.deps/bench/bin/python docs_figures/make_figures.py --docs /path/to/Baysor/docs/performance

# a subset
.deps/bench/bin/python docs_figures/make_figures.py --docs ... --only chart_scaling fig_umap
```

The run takes about 15 s (the UMAP embeddings are cached in
`generated/umap_*.csv`; delete them to recompute).

## Inputs

| input | produced by |
|---|---|
| `rc/2026-10-01-1636267/tables/{runtime,quality_sim,quality_real}.csv` | the cpp-0.9.0 release-candidate A/B campaign vs cpp-0.8.3 (`scripts/aggregate.py` in that directory; its `REPORT.md` describes the runs) |
| `runs/{rc1636,v0083}-{real,sim,t1,t1b}/<dataset>/` | `harness/run.py` (per-replicate `metrics.json`, segmentation outputs, cellAdmix audits) |
| `docs-figures/remeasure/runs.tsv` | `docs_figures/remeasure.py` (below) |
| `docs-figures/ncv/<dataset>/` | `docs_figures/ncv_runs.py`: the harness command of `runs/rc1636-real/<dataset>/rep0` with the release-candidate binary `baysor-rc-cba8ca3` and without `--skip-ncv-color` (the segmentation is identical; the outputs gain `ncv_color`) |
| `profiling/reports/2026-10-01-20bc45c/summary-*-run/` | the profiling suite (`profiling/`), after the optimisation |
| `profiling/reports/2026-09-30-e45fddc/summary-scaling-run/` | the same suite, before the optimisation |
| `real/<id>/meta.json`, `sim/<id>/meta.json` | dataset facts (platform, genes) |

## Outputs

* `<docs>/data/<chart>.json` — one spec per interactive chart
  (`runtime_vs_molecules`, `scaling`, `threads`, `time_vs_genes`,
  `celladmix`). The Baysor docs draw them in the browser with the vendored
  `docs/assets/perfcharts/perfcharts.js` (plain SVG + JS, no dependencies,
  no CDN): hover/keyboard tooltips with the dataset, x, y and run details, a
  legend, a sortable data table, light/dark colours from CSS variables. The
  spec format is documented at the top of the "interactive charts" section of
  `make_figures.py`. Only the current version is plotted, except cellAdmix
  (cpp-0.9.0 vs cpp-0.8.3) and the profiling ladders (before/after the
  optimisation).
* `<docs>/img/{segmentation_examples,umap}-{light,dark}.png` — the
  scatterplots of biological data (MkDocs Material shows one or the other via
  `#only-light` / `#only-dark`); molecules in the segmentation examples are
  coloured by `ncv_color`, noise molecules muted, cell polygons as outlines.
* the per-dataset table of `<docs>/profiling.md`, rewritten between the
  `docs_figures:runtime-table` markers (benchmark crops and profiling ladders,
  datasets in rows; the page makes it sortable by column).
* `generated/owner_report.md` + `generated/owner/phases-{light,dark}.svg` —
  **owner only, not in the docs**: "Where the time goes" and "Current
  bottlenecks" with the phase figure and the phase-share / exponent tables.
* `generated/tables.md`: every number quoted in the pages, as tables, plus a
  table mapping each figure/table to its source files;
  `generated/numbers.json`: the same numbers, machine-readable.

Colours follow the dataviz reference palette (blue / orange / aqua for up to
three categories, which validates all-pairs for colour-vision deficiencies in
both themes; a neutral grey for "other" or the previous version; secondary
encoding by marker shape or line style).

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
