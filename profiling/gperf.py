"""Parse and symbolize gperftools CPU profiles (legacy binary format).

A profile written by libprofiler (CPUPROFILE=<file>) is a sequence of
native-endian machine words:

    header   0, 3, 0, <sampling period in us>, 0
    records  <count>, <n pcs>, pc_1 .. pc_n        (pc_1 = sampled pc,
                                                    pc_2.. = return addresses)
    trailer  0, 1, 0
    text     a copy of /proc/self/maps

Samples are taken on ITIMER_PROF, i.e. per consumed CPU time (not wall), so
per-function shares are shares of CPU time. Symbols come from ``nm -C`` of
every mapped object (PIE and shared objects are relocated via the mapping
offset and the PT_LOAD segments from ``readelf -lW``).
"""
from __future__ import annotations

import bisect
import re
import struct
import subprocess
from collections import defaultdict
from functools import lru_cache
from pathlib import Path


def read_profile(path: Path):
    data = Path(path).read_bytes()
    w = 8
    words = struct.unpack_from(f"<{5}Q", data, 0)
    if words[0] != 0 or words[1] != 3:
        raise ValueError(f"{path}: not a gperftools CPU profile")
    period_us = words[3]
    off = 5 * w
    samples = []
    while off + 2 * w <= len(data):
        count, n = struct.unpack_from("<2Q", data, off)
        off += 2 * w
        pcs = struct.unpack_from(f"<{n}Q", data, off) if n else ()
        off += n * w
        if count == 0 and n == 1 and pcs == (0,):
            break
        samples.append((count, pcs))
    maps_text = data[off:].decode(errors="replace")
    maps = []
    for line in maps_text.splitlines():
        parts = line.split(None, 5)
        if len(parts) < 6 or "x" not in parts[1]:
            continue
        a, b = (int(v, 16) for v in parts[0].split("-"))
        maps.append((a, b, int(parts[2], 16), parts[5].strip()))
    maps.sort()
    return period_us, samples, maps


@lru_cache(maxsize=None)
def _load_segments(obj: str):
    """[(p_offset, p_vaddr, p_filesz)] of executable PT_LOAD segments; ELF type."""
    try:
        out = subprocess.run(["readelf", "-lhW", obj], capture_output=True, text=True).stdout
    except OSError:
        return [], "?"
    etype = "DYN" if "DYN (" in out else ("EXEC" if "EXEC (" in out else "?")
    segs = []
    for line in out.splitlines():
        line = line.strip()
        if line.startswith("LOAD"):
            f = line.split()
            segs.append((int(f[1], 16), int(f[2], 16), int(f[4], 16)))
    return segs, etype


def _debug_file(obj: str):
    """Separate debug file (/usr/lib/debug/.build-id/xx/yyyy.debug), if installed."""
    try:
        out = subprocess.run(["readelf", "-nW", obj], capture_output=True, text=True).stdout
    except OSError:
        return None
    m = re.search(r"Build ID: ([0-9a-f]+)", out)
    if not m:
        return None
    bid = m.group(1)
    p = Path("/usr/lib/debug/.build-id") / bid[:2] / f"{bid[2:]}.debug"
    return str(p) if p.is_file() else None


@lru_cache(maxsize=None)
def _symbols(obj: str):
    """Sorted (addr, name) of function symbols of an object (static + dynamic,
    plus the separate debug file of stripped system libraries)."""
    syms = {}
    dbg = _debug_file(obj)
    sources = ([(dbg, [])] if dbg else []) + [(obj, []), (obj, ["-D"])]
    for path, extra in sources:
        try:
            out = subprocess.run(["nm", "-C", "--defined-only", "-n", *extra, path],
                                 capture_output=True, text=True).stdout
        except OSError:
            continue
        for line in out.splitlines():
            parts = line.split(" ", 2)
            if len(parts) == 3 and parts[1] in "tTwWiI":
                try:
                    addr = int(parts[0], 16)
                except ValueError:
                    continue
                if addr and addr not in syms:
                    syms[addr] = parts[2]
    items = sorted(syms.items())
    return [a for a, _ in items], [n for _, n in items]


class Symbolizer:
    def __init__(self, maps):
        self.maps = maps
        self.starts = [m[0] for m in maps]
        self.cache = {}

    def __call__(self, pc: int) -> tuple[str, str]:
        if pc in self.cache:
            return self.cache[pc]
        i = bisect.bisect_right(self.starts, pc) - 1
        res = ("??", "??")
        if i >= 0:
            a, b, off, obj = self.maps[i]
            if a <= pc < b and obj.startswith("/"):
                file_off = pc - a + off
                segs, etype = _load_segments(obj)
                vaddr = None
                for p_off, p_vaddr, p_filesz in segs:
                    if p_off <= file_off < p_off + max(p_filesz, 1):
                        vaddr = file_off - p_off + p_vaddr
                        break
                if vaddr is None:
                    vaddr = pc if etype == "EXEC" else file_off
                addrs, names = _symbols(obj)
                j = bisect.bisect_right(addrs, vaddr) - 1
                name = names[j] if j >= 0 else f"{Path(obj).name}+{vaddr:#x}"
                res = (name, Path(obj).name)
        self.cache[pc] = res
        return res


# ---------------------------------------------------------------------------
# Thread-model classification of samples (OpenMP builds and builds with
# Baysor's own pool, include/baysor/utils/thread_pool.h)
# ---------------------------------------------------------------------------
# libgomp barrier/spin-wait functions (leaf), as in scaling.py's original
# "OpenMP wait % of CPU"
OMP_WAIT_RE = re.compile(r"^(gomp_\w*(barrier|wait|spin)\w*|do_wait|do_spin|futex_wait)")
# Low-level frames under a waiting function: clock reads of the spin
# deadline, futex/condition-variable/mutex syscalls and their wrappers.
WAIT_HELPER_RE = re.compile(
    r"^(__vdso_\w+|clock_gettime\w*|__clock_gettime\w*|syscall|__GI_syscall|\w*futex\w*|"
    r"\w*pthread_cond_\w+|\w*pthread_mutex_\w+|__lll_\w+|lll_\w+|std::condition_variable::\w+.*|"
    r"std::__condvar::\w+.*|\w*sched_yield|std::chrono::_V2::(steady|system)_clock::now.*|"
    r"__GI___\w+|\?\?|.*\+0x[0-9a-f]+)$")
# Eigen's GEMM thread pool (EIGEN_GEMM_THREADPOOL) idling/spinning.
EIGEN_WAIT_RE = re.compile(r"^Eigen::(ThreadPoolTempl<.*>::(WaitForWork|Steal|LocalSteal|GlobalSteal|"
                           r"NonEmptyQueueIndex|WorkerLoop)|EventCount::|RunQueue<)")


def _phase_of(frames, patterns):
    """Outermost frame matching a phase pattern (closest to main)."""
    import fnmatch
    hit = None
    for name in frames:
        for pat, ph in patterns:
            if fnmatch.fnmatchcase(name, pat):
                hit = ph
                break
    return hit


def classify(samples_frames, phases=None, region_name=None) -> dict:
    """Split CPU samples by thread model.

    * ``parallel``: samples inside a parallel-region body (an OpenMP outlined
      function ``*._omp_fn.N`` or a pool chunk, the std::function<void(long,
      long, int)> invoker of a work-shared loop; see cgparse.py);
    * ``wait_pool``: the first frame above clock/futex/mutex helpers is the
      pool's machinery (hand-off spin and block, region barrier, chunk
      scheduling); ``wait_omp``: leaf in libgomp's barrier/spin functions;
      ``wait_eigen``: Eigen's GEMM thread pool idling;
    * per phase (outermost phase function on the stack; worker samples
      without one are distributed over phases in proportion to the main
      thread's samples of the same region): samples, parallel, wait.
    """
    import cgparse
    pats = [(p["pattern"], p["name"]) for p in (phases or [])]
    tot = {"samples": 0, "parallel": 0, "wait_pool": 0, "wait_omp": 0, "wait_eigen": 0}
    regions = defaultdict(int)
    ph = defaultdict(lambda: {"samples": 0, "parallel": 0, "wait": 0})
    region_phase = defaultdict(lambda: defaultdict(int))
    orphan = defaultdict(lambda: {"samples": 0, "parallel": 0, "wait": 0})   # region -> worker samples
    for count, frames in samples_frames:
        tot["samples"] += count
        leaf = frames[0] if frames else "??"
        kind = None
        if OMP_WAIT_RE.match(leaf):
            kind = "wait_omp"
        else:
            first = next((f for f in frames if not WAIT_HELPER_RE.match(f)), None)
            if first is not None and cgparse.is_pool_runtime(first):
                kind = "wait_pool"
            elif first is not None and EIGEN_WAIT_RE.match(first):
                kind = "wait_eigen"
        reg = None
        for f in frames:                 # outermost region frame
            if cgparse.is_region_fn(f):
                reg = f
        if kind:
            tot[kind] += count
        if reg is not None and not kind:
            tot["parallel"] += count
            regions[reg] += count
        if pats:
            phase = _phase_of(frames, pats)
            rec_kind = "wait" if kind else ("parallel" if reg is not None else None)
            if phase is not None:
                r = ph[phase]
                if reg is not None:
                    region_phase[reg][phase] += count
            elif reg is not None:
                r = orphan[reg]
            else:
                r = ph["(no phase: " + ("wait" if kind else "other") + ")"]
            r["samples"] += count
            if rec_kind:
                r[rec_kind] += count
    if pats:
        for reg, o in orphan.items():
            split = region_phase.get(reg) or {"(unattributed regions)": 1}
            s = sum(split.values())
            for name, w in split.items():
                for k in ("samples", "parallel", "wait"):
                    ph[name][k] += o[k] * w / s
    name = region_name or (lambda f: f)
    out = {**tot,
           "regions": {name(f): n for f, n in sorted(regions.items(), key=lambda kv: -kv[1])[:40]}}
    if pats:
        out["phases"] = {k: {kk: round(vv, 2) for kk, vv in v.items()} for k, v in ph.items()}
    return out


def summarize(path: Path, top: int = 60, phases=None) -> dict:
    """Per-function exclusive (leaf) and inclusive (on stack) sample shares;
    with ``phases`` (profiling.yaml phase list) also the thread-model
    classification (``classes``, see classify)."""
    period_us, samples, maps = read_profile(path)
    sym = Symbolizer(maps)
    excl = defaultdict(int)
    incl = defaultdict(int)
    obj_of = {}
    total = 0
    stacks = []
    for count, pcs in samples:
        if not pcs:
            continue
        total += count
        frames = [sym(pcs[0])] + [sym(pc - 1) for pc in pcs[1:]]
        leaf = frames[0][0]
        excl[leaf] += count
        obj_of.setdefault(leaf, frames[0][1])
        for name, obj in dict.fromkeys(frames):
            incl[name] += count
            obj_of.setdefault(name, obj)
        if phases is not None:
            stacks.append((count, [f[0] for f in frames]))
    def rows(d):
        return [{"fn": fn, "object": obj_of.get(fn, ""), "samples": n,
                 "pct": round(100.0 * n / total, 3) if total else 0.0}
                for fn, n in sorted(d.items(), key=lambda kv: -kv[1])[:top]]
    return {"period_us": period_us, "total_samples": total,
            "cpu_s_sampled": round(total * period_us / 1e6, 2),
            "top_excl": rows(excl), "top_incl": rows(incl),
            "excl_all": {fn: n for fn, n in excl.items() if n >= max(1, total // 2000)},
            "incl_all": {fn: n for fn, n in incl.items() if n >= max(1, total // 2000)},
            **({"classes": classify(stacks, phases, _short)} if phases is not None else {})}


def _short(fn: str) -> str:
    import summarize as qs   # lazy: summarize imports cgparse only
    return qs.short_name(fn)
