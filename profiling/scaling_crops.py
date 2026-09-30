#!/usr/bin/env python3
"""Build the nested size ladders of the scaling tier (profiling.yaml `scaling`).

For every slide: read the vendor transcripts.parquet row group by row group
with the benchmark fetch code (fetch/xenium_common.py: qv >= min_qv
and real-gene filter, vendor nucleus prior, contract columns), keep the square
around the median molecule position that holds the largest rung (or the whole
slide for `all`), then cut every smaller rung as the smallest square with the
same centre holding >= N molecules (exact order statistic of the Chebyshev
distance, so the rungs are nested and have the slide's local density).

Output: $BAYSOR_BENCH_DATA/profiling/scaling/data/<slide>_<rung>/
{molecules.parquet, meta.json}; meta.json copies the `baysor` section
(flags, scale) of the slide's reference dataset so every rung runs with
identical parameters.

    python profiling/scaling_crops.py [--only lung] [--force]
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.append(str(HERE.parent / "fetch"))       # ../fetch
import profcommon as common  # noqa: E402
import xenium_common as xc  # noqa: E402


def rung_label(n) -> str:
    if n == "all":
        return "all"
    n = int(n)
    return f"{n / 1e6:g}M" if n >= 1_000_000 else f"{n // 1000}k"


def slide_bounds(pf: pq.ParquetFile) -> list[float]:
    names = pf.schema_arrow.names
    ix, iy = names.index("x_location"), names.index("y_location")
    x0 = y0 = math.inf
    x1 = y1 = -math.inf
    for rg in range(pf.metadata.num_row_groups):
        md = pf.metadata.row_group(rg)
        sx, sy = md.column(ix).statistics, md.column(iy).statistics
        if sx is None or sy is None or sx.min is None or sy.min is None:
            t = pf.read_row_group(rg, columns=["x_location", "y_location"])
            if t.num_rows == 0:
                continue
            xs = t.column("x_location").to_numpy()
            ys = t.column("y_location").to_numpy()
            lo_x, hi_x, lo_y, hi_y = xs.min(), xs.max(), ys.min(), ys.max()
        else:
            lo_x, hi_x, lo_y, hi_y = sx.min, sx.max, sy.min, sy.max
        x0, x1 = min(x0, lo_x), max(x1, hi_x)
        y0, y1 = min(y0, lo_y), max(y1, hi_y)
    return [float(x0), float(y0), float(x1) + 1e-6, float(y1) + 1e-6]


def median_from_hist(counts: np.ndarray, lo: float, bin_um: float) -> float:
    c = np.cumsum(counts)
    k = int(np.searchsorted(c, c[-1] / 2.0))
    return lo + (k + 0.5) * bin_um


def half_side_for(hist: np.ndarray, bounds, bin_um: float, cx: float, cy: float, target: int) -> float:
    """Smallest half-side (bin resolution) whose square around (cx, cy) holds >= target."""
    ny, nx = hist.shape
    xc_ = bounds[0] + (np.arange(nx) + 0.5) * bin_um
    yc_ = bounds[1] + (np.arange(ny) + 0.5) * bin_um
    d = np.maximum(np.abs(xc_[None, :] - cx), np.abs(yc_[:, None] - cy)).ravel()
    order = np.argsort(d, kind="stable")
    cum = np.cumsum(hist.ravel()[order])
    k = int(np.searchsorted(cum, target))
    k = min(k, len(order) - 1)
    return float(d[order[k]] + bin_um)          # one bin of margin


def build_slide(slide: dict, root: Path, out_root: Path, force: bool) -> list[dict]:
    rungs = slide["rungs"]
    todo = [r for r in rungs if force or not (out_root / f"{slide['id']}_{rung_label(r)}" / "meta.json").is_file()]
    metas = []
    if not todo:
        for r in rungs:
            metas.append(common.read_json(out_root / f"{slide['id']}_{rung_label(r)}" / "meta.json"))
        return metas
    path = root / slide["transcripts"]
    ref_dir = common.find_source_dataset(root, slide["reference_dataset"])
    ref_meta = json.loads((ref_dir / "meta.json").read_text())
    min_qv = float(slide.get("min_qv", 20))
    pf = pq.ParquetFile(str(path))
    t0 = time.time()
    numeric = [int(r) for r in rungs if r != "all"]
    if "all" in rungs:
        big_bbox = None
    else:
        bounds = slide_bounds(pf)
        bin_um = 25.0
        hist = xc.transcript_histogram(path, min_qv, bounds, bin_um=bin_um)
        cx = median_from_hist(hist.sum(axis=0), bounds[0], bin_um)
        cy = median_from_hist(hist.sum(axis=1), bounds[1], bin_um)
        h = half_side_for(hist, bounds, bin_um, cx, cy, int(max(numeric) * 1.03))
        big_bbox = [cx - h, cy - h, cx + h, cy + h]
        print(f"[{slide['id']}] histogram {time.time() - t0:.0f} s, centre ({cx:.0f}, {cy:.0f}), "
              f"largest square half-side {h:.0f} um", flush=True)
    if big_bbox is None:
        bounds = slide_bounds(pf)
        big_bbox = bounds
    crop = xc.read_transcript_crops(path, [big_bbox], min_qv)[0]
    table = xc.build_molecule_table(crop)
    del crop
    x = table.column("x").to_numpy()
    y = table.column("y").to_numpy()
    cx, cy = float(np.median(x)), float(np.median(y))
    dist = np.maximum(np.abs(x - cx), np.abs(y - cy))
    print(f"[{slide['id']}] read {table.num_rows} filtered molecules in {time.time() - t0:.0f} s", flush=True)
    for r in rungs:
        label = rung_label(r)
        out_dir = out_root / f"{slide['id']}_{label}"
        if r == "all":
            mask = np.ones(len(x), dtype=bool)
            half = float(dist.max())
        else:
            n = int(r)
            if n > len(x):
                print(f"[{slide['id']}] rung {label}: only {len(x)} molecules, skipped")
                continue
            half = float(np.partition(dist, n - 1)[n - 1])
            mask = dist <= half
        sub = table.filter(pa.array(mask))
        out_dir.mkdir(parents=True, exist_ok=True)
        pq.write_table(sub, out_dir / "molecules.parquet")
        genes = pc.unique(sub.column("gene"))
        bbox = [cx - half, cy - half, cx + half, cy + half]
        area = float((min(bbox[2], x.max()) - max(bbox[0], x.min())) *
                     (min(bbox[3], y.max()) - max(bbox[1], y.min())))
        meta = copy.deepcopy(ref_meta)
        meta.update({"id": out_dir.name, "tier": "profiling-scaling", "images": [],
                     "crop": {"bbox_um": [round(v, 3) for v in bbox], "z_range_um": None,
                              "note": f"scaling ladder rung {label} of {slide['transcripts']}: "
                                      f"median-centred square, qv >= {min_qv}"}})
        meta["stats"] = {"n_molecules": int(sub.num_rows), "n_genes": int(len(genes)),
                         "area_um2": round(area, 1),
                         "molecules_per_um2": round(sub.num_rows / area, 4) if area else None,
                         "n_prior_labels": int(len(np.unique(sub.column("prior").to_numpy()))) - 1}
        meta["scaling"] = {"slide": slide["id"], "rung": r if r == "all" else int(r),
                           "reference_dataset": slide["reference_dataset"],
                           "transcripts": str(path), "min_qv": min_qv}
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
        metas.append(meta)
        print(f"[{slide['id']}] {out_dir.name}: {sub.num_rows} molecules, {len(genes)} genes, "
              f"{area / 1e6:.2f} mm2", flush=True)
    return metas


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", default=str(HERE / "profiling.yaml"))
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--only", default=None, help="comma-separated slide ids")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    spec = common.load_spec(Path(args.spec))["scaling"]
    root = common.bench_root(args.data_root)
    out_root = root / "profiling" / "scaling" / "data"
    only = set(args.only.split(",")) if args.only else None
    for slide in spec["slides"]:
        if only and slide["id"] not in only:
            continue
        build_slide(slide, root, out_root, args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
