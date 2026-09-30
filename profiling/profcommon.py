"""Shared helpers for the profiling suite (paths, spec loading).

Layout inside this repository (the benchmarks repo, `profiling/` at its root):

  <repo>/profiling/        this suite
  <repo>/harness/          the benchmark harness (command construction)
  <repo>/fetch/            dataset code (scaling ladders)
  <repo>/baysor-configs/   vendored Baysor configs, resolved by the harness
  <repo>/.bench-data/      data root: crops, runs, summaries ($BAYSOR_BENCH_DATA)
  <repo>/.deps/vgenv       Valgrind ($VALGRIND / --valgrind override)
  <repo>/.deps/proftools   gperftools + heaptrack ($PROFTOOLS / --proftools override)

The Baysor *source* checkout is optional (`--baysor-src` / `$BAYSOR_SRC`):
it is only needed for the git sha recorded with a run and to locate a default
profiling binary (`<src>/build/profiling/baysor`). The binary itself is
explicit (`--baysor` / `$BAYSOR_BIN`); Baysor configs come from the vendored
`baysor-configs/` via the harness resolver, never from a Baysor tree.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional

import yaml

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent            # this repository: ../harness is the harness


def repo_root() -> Path:
    """Root of this benchmarks repository (profiling/'s home)."""
    return BENCH


def baysor_src(cli_value: Optional[str] = None) -> Optional[Path]:
    """Optional Baysor source checkout: `--baysor-src`, else `$BAYSOR_SRC`.

    Needed only for the git sha recorded with runs and for the default
    profiling binary location; configs, data and tools all live in this
    repository. None when neither is given.
    """
    if cli_value:
        return Path(cli_value).resolve()
    env = os.environ.get("BAYSOR_SRC")
    if env:
        return Path(env).resolve()
    return None


def bench_root(cli_value: Optional[str] = None) -> Path:
    """Data root: `--data-root`, else `$BAYSOR_BENCH_DATA`, else
    `<repo>/.bench-data` — the same rule as harness/common.py data_root().
    A git worktree without its own `.bench-data` falls back to the main
    checkout's (where the symlink normally lives)."""
    if cli_value:
        return Path(cli_value).expanduser().resolve()
    env = os.environ.get("BAYSOR_BENCH_DATA")
    if env:
        return Path(env).expanduser().resolve()
    local = BENCH / ".bench-data"
    if local.exists():
        return local
    git = BENCH / ".git"
    if git.is_file():                       # worktree: "gitdir: <main>/.git/worktrees/<name>"
        try:
            gitdir = Path(git.read_text().split(":", 1)[1].strip())
            main = gitdir.parents[2] / ".bench-data"
            if main.exists():
                return main
        except (OSError, IndexError, ValueError):
            pass
    return local


def vgenv_bin(name: str) -> Path:
    """`<repo>/.deps/vgenv/bin/<name>` (the Valgrind conda env)."""
    return BENCH / ".deps" / "vgenv" / "bin" / name


def find_valgrind(specified: Optional[str] = None) -> Optional[Path]:
    """Resolve the valgrind executable.

    An explicit `--valgrind` / `$VALGRIND` (anything other than the bare
    default `valgrind`) wins when it exists; otherwise the repository's
    `.deps/vgenv`, then `valgrind` on PATH. None when nothing is found.
    """
    if specified and specified != "valgrind":
        if Path(specified).is_file():
            return Path(specified)
        found = shutil.which(specified)
        return Path(found) if found else None
    local = vgenv_bin("valgrind")
    if local.is_file():
        return local
    found = shutil.which("valgrind")
    return Path(found) if found else None


def find_proftools(specified: Optional[str] = None) -> Optional[Path]:
    """gperftools/heaptrack prefix: `--proftools`, else `$PROFTOOLS`, else
    `<repo>/.deps/proftools`. None when no candidate holds libprofiler.so."""
    for cand in (specified, os.environ.get("PROFTOOLS"),
                 str(BENCH / ".deps" / "proftools")):
        if cand and (Path(cand) / "lib" / "libprofiler.so").is_file():
            return Path(cand)
    return None


def run_sha(src: Optional[Path], baysor: Optional[Path] = None) -> str:
    """Short id for default run-ids: the Baysor checkout's git sha when a
    source checkout is known, else the binary's sha256 prefix, else `nogit`."""
    if src is not None:
        try:
            out = subprocess.run(["git", "-C", str(src), "rev-parse", "--short=7", "HEAD"],
                                 capture_output=True, text=True, timeout=15).stdout.strip()
            if out:
                return out
        except OSError:
            pass
    if baysor is not None and Path(baysor).is_file():
        h = hashlib.sha256()
        with open(baysor, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()[:7]
    return "nogit"


def load_spec(path: Path) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


def find_source_dataset(root: Path, ds_id: str) -> Path:
    for kind in ("real", "sim"):
        p = root / kind / ds_id
        if (p / "meta.json").is_file():
            return p
    raise FileNotFoundError(f"source dataset {ds_id} not found under {root}/{{real,sim}}")


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=1, allow_nan=True)
        fh.write("\n")


def read_json(path: Path):
    with open(path) as fh:
        return json.load(fh)
