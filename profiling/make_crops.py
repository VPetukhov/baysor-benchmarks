#!/usr/bin/env python3
"""Derive the small profiling crops listed in profiling.yaml.

Each crop is an axis-aligned x/y square centred on the median molecule
position of its source dataset, with the smallest side that holds at least
``n_molecules`` molecules (bisection; deterministic, no randomness). Rows
keep the source order and schema, all z are kept. The output goes to
``$BAYSOR_BENCH_DATA/profiling/data/<id>/{molecules.parquet,meta.json}``;
nothing is written to the repository.

    python profiling/make_crops.py            # all datasets
    python profiling/make_crops.py --only cosmx_wtx_20k --force
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import profcommon as common  # noqa: E402


def crop_square(x: np.ndarray, y: np.ndarray, target: int):
    """Return (mask, bbox) of the smallest median-centred square with >= target rows."""
    cx, cy = float(np.median(x)), float(np.median(y))
    dist = np.maximum(np.abs(x - cx), np.abs(y - cy))   # Chebyshev distance
    if target >= len(x):
        half = float(dist.max())
    else:
        # the smallest half-side holding >= target molecules is the target-th
        # order statistic of the Chebyshev distance (ties may add a few rows)
        half = float(np.partition(dist, target - 1)[target - 1])
    mask = dist <= half
    return mask, [cx - half, cy - half, cx + half, cy + half]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def make_crop(spec: dict, bench_root: Path, out_root: Path, force: bool) -> dict:
    src_dir = common.find_source_dataset(bench_root, spec["source"])
    out_dir = out_root / spec["id"]
    meta_path = out_dir / "meta.json"
    if meta_path.is_file() and not force:
        meta = json.loads(meta_path.read_text())
        if meta.get("profiling_crop", {}).get("spec") == spec:
            return meta
    src_meta = json.loads((src_dir / "meta.json").read_text())
    table = pq.read_table(src_dir / "molecules.parquet")
    x = table.column("x").to_numpy()
    y = table.column("y").to_numpy()
    mask, bbox = crop_square(x, y, int(spec["n_molecules"]))
    sub = table.filter(mask)

    out_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(sub, out_dir / "molecules.parquet")

    genes = pc.unique(sub.column("gene").combine_chunks().cast("string"))
    area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    meta = copy.deepcopy(src_meta)
    meta["id"] = spec["id"]
    meta["tier"] = "profiling"
    meta["images"] = []          # images are not cropped (no image priors used)
    meta["crop"] = {"bbox_um": [round(v, 6) for v in bbox], "z_range_um": None,
                    "note": f"profiling crop of {spec['source']}: median-centred "
                            f"square with >= {spec['n_molecules']} molecules"}
    stats = meta.setdefault("stats", {})
    stats.update({"n_molecules": int(sub.num_rows), "n_genes": int(len(genes)),
                  "area_um2": round(area, 3),
                  "molecules_per_um2": round(sub.num_rows / area, 6) if area else None})
    if "prior" in sub.column_names:
        prior = sub.column("prior").to_numpy()
        stats["n_prior_labels"] = int(len(np.unique(prior[prior > 0])))
    meta["profiling_crop"] = {
        "spec": spec, "source_dir": str(src_dir),
        "source_sha256": sha256_file(src_dir / "molecules.parquet"),
        "sha256": sha256_file(out_dir / "molecules.parquet"),
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", default=str(HERE / "profiling.yaml"))
    ap.add_argument("--data-root", default=None,
                    help="benchmark data root (default: $BAYSOR_BENCH_DATA or <repo>/.bench-data)")
    ap.add_argument("--only", default=None, help="comma-separated dataset ids")
    ap.add_argument("--force", action="store_true", help="re-crop even if up to date")
    args = ap.parse_args(argv)

    spec = common.load_spec(Path(args.spec))
    bench_root = common.bench_root(args.data_root)
    out_root = bench_root / "profiling" / "data"
    only = set(args.only.split(",")) if args.only else None
    for ds in spec["datasets"]:
        if only and ds["id"] not in only:
            continue
        meta = make_crop(ds, bench_root, out_root, args.force)
        st = meta["stats"]
        print(f"{ds['id']:28s} {st['n_molecules']:7d} molecules {st['n_genes']:6d} genes "
              f"area {st['area_um2']:.0f} um2  <- {ds['source']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
