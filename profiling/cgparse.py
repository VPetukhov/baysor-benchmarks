"""Minimal parser for callgrind output files (format version 1).

Handles name compression (``fn=(12) name`` / ``fn=(12)``), relative
position compression (``+N``, ``-N``, ``*``), inlined-file switches
(``fi=``/``fe=``), call arcs (``calls=`` followed by the inclusive cost
line), several dump parts (``--dump-before/--dump-after``) and several
threads (``--separate-threads=yes``: one file per part and thread).

Only ``positions: line`` (the default) is supported; costs are kept as
integer lists in the order of the ``events:`` header.
"""
from __future__ import annotations

import gzip
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

OMP_FN_RE = re.compile(r"\._omp_fn\.\d+")


def is_omp_fn(name: str) -> bool:
    return bool(OMP_FN_RE.search(name))


def _add(dst: list, src: list) -> None:
    if len(dst) < len(src):
        dst.extend([0] * (len(src) - len(dst)))
    for i, v in enumerate(src):
        dst[i] += v


@dataclass
class Profile:
    """Costs of one callgrind output file (one part of one thread)."""
    path: str
    events: list = field(default_factory=list)
    header: dict = field(default_factory=dict)
    totals: list = field(default_factory=list)
    self_cost: dict = field(default_factory=lambda: defaultdict(list))      # fn -> costs
    fn_file: dict = field(default_factory=dict)                            # fn -> file
    fn_obj: dict = field(default_factory=dict)                             # fn -> object
    line_cost: dict = field(default_factory=lambda: defaultdict(list))     # (fn, file, line) -> costs
    arcs: dict = field(default_factory=dict)                               # (caller, callee) -> [calls, costs]

    @property
    def part(self) -> int:
        return int(self.header.get("part", 1))

    @property
    def thread(self) -> int:
        return int(self.header.get("thread", 1))

    @property
    def trigger(self) -> str:
        for d in self.header.get("desc", []):
            if d.startswith("Trigger:"):
                return d[len("Trigger:"):].strip()
        return ""

    def total(self, event: str = "Ir") -> int:
        i = self.events.index(event)
        return self.totals[i] if i < len(self.totals) else 0


def _open(path: Path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", errors="replace")
    return open(path, errors="replace")


_NAME_RE = re.compile(r"^\((\d+)\)(?:\s(.*))?$")


def parse(path: Path) -> Profile:
    prof = Profile(path=str(path))
    names = {k: {} for k in ("fl", "fn", "ob")}   # compression tables (fi/fe/cfi share fl)

    def resolve(kind: str, value: str) -> str:
        m = _NAME_RE.match(value)
        if not m:
            return value
        idx, name = m.group(1), m.group(2)
        table = names[kind]
        if name is not None:
            table[idx] = name
            return name
        return table.get(idx, f"({idx})")

    cur_ob = cur_fl = cur_file = cur_fn = ""
    cfn = cfl = cob = None
    pending_call = None       # [calls] waiting for its cost line
    last_line = 0
    ncost = 0
    desc = []
    with _open(path) as fh:
        for raw in fh:
            line = raw.rstrip("\n")
            if not line or line[0] == "#":
                continue
            c0 = line[0]
            if c0.isdigit() or c0 in "+-*":
                parts = line.split()
                pos = parts[0]
                if pos[0] == "+":
                    ln = last_line + int(pos[1:])
                elif pos[0] == "-":
                    ln = last_line - int(pos[1:])
                elif pos == "*":
                    ln = last_line
                else:
                    ln = int(pos)
                last_line = ln
                costs = [int(v) for v in parts[1:]]
                if pending_call is not None:
                    key = (cur_fn, cfn)
                    rec = prof.arcs.get(key)
                    if rec is None:
                        prof.arcs[key] = [pending_call, costs]
                    else:
                        rec[0] += pending_call
                        _add(rec[1], costs)
                    pending_call = None
                    cfn = cfl = cob = None
                else:
                    _add(prof.self_cost[cur_fn], costs)
                    _add(prof.line_cost[(cur_fn, cur_file, ln)], costs)
                continue
            key, _, value = line.partition("=")
            if key == "fn":
                cur_fn = resolve("fn", value)
                cur_file = cur_fl
                prof.fn_file.setdefault(cur_fn, cur_fl)
                prof.fn_obj.setdefault(cur_fn, cur_ob)
            elif key == "fl":
                cur_fl = cur_file = resolve("fl", value)
            elif key in ("fi", "fe"):
                cur_file = resolve("fl", value)
            elif key == "ob":
                cur_ob = resolve("ob", value)
            elif key == "cfn":
                cfn = resolve("fn", value)
            elif key in ("cfi", "cfl"):
                cfl = resolve("fl", value)
            elif key == "cob":
                cob = resolve("ob", value)
            elif key == "calls":
                pending_call = int(value.split()[0])
                if cfn is not None:
                    prof.fn_file.setdefault(cfn, cfl if cfl is not None else cur_fl)
                    prof.fn_obj.setdefault(cfn, cob if cob is not None else cur_ob)
            elif key in ("jump", "jcnd"):
                pending_call = None
            else:
                hk, sep, hv = line.partition(":")
                if not sep:
                    continue
                hv = hv.strip()
                if hk == "events":
                    prof.events = hv.split()
                    ncost = len(prof.events)
                elif hk in ("totals", "summary"):
                    prof.totals = [int(v) for v in hv.split()]
                elif hk == "desc":
                    desc.append(hv)
                elif hk in ("version", "creator", "pid", "cmd", "part", "thread",
                            "positions"):
                    prof.header[hk] = hv
    prof.header["desc"] = desc
    for costs in prof.self_cost.values():
        if len(costs) < ncost:
            costs.extend([0] * (ncost - len(costs)))
    if not prof.totals:
        tot: list = [0] * ncost
        for costs in prof.self_cost.values():
            _add(tot, costs)
        prof.totals = tot
    return prof


class Aggregate:
    """Sum of several Profiles (parts and/or threads) with derived metrics."""

    def __init__(self, events: list):
        self.events = list(events)
        self.self_cost = defaultdict(lambda: [0] * len(self.events))
        self.line_cost = defaultdict(lambda: [0] * len(self.events))
        self.arcs = {}
        self.fn_file: dict = {}
        self.fn_obj: dict = {}
        self.totals = [0] * len(self.events)

    def add(self, prof: Profile) -> None:
        idx = [prof.events.index(e) if e in prof.events else None for e in self.events]

        def remap(c):
            return [c[i] if i is not None and i < len(c) else 0 for i in idx]

        _add(self.totals, remap(prof.totals))
        for fn, c in prof.self_cost.items():
            _add(self.self_cost[fn], remap(c))
        for k, c in prof.line_cost.items():
            _add(self.line_cost[k], remap(c))
        for k, (n, c) in prof.arcs.items():
            rec = self.arcs.get(k)
            if rec is None:
                self.arcs[k] = [n, remap(c)]
            else:
                rec[0] += n
                _add(rec[1], remap(c))
        for fn, f in prof.fn_file.items():
            self.fn_file.setdefault(fn, f)
        for fn, o in prof.fn_obj.items():
            self.fn_obj.setdefault(fn, o)

    def ev(self, name: str) -> int:
        return self.events.index(name)

    def inclusive(self) -> dict:
        """fn -> inclusive costs (self + non-recursive outgoing arcs)."""
        inc = {fn: list(c) for fn, c in self.self_cost.items()}
        for (caller, callee), (_, c) in self.arcs.items():
            if caller == callee:
                continue
            _add(inc.setdefault(caller, [0] * len(self.events)), c)
        # clamp to the grand total (recursion through several functions can
        # otherwise count a cost more than once)
        for fn, c in inc.items():
            for i, v in enumerate(c):
                if v > self.totals[i]:
                    c[i] = self.totals[i]
        return inc

    def callers(self) -> dict:
        out = defaultdict(list)
        for (caller, callee), (n, c) in self.arcs.items():
            out[callee].append((caller, n, c))
        return out

    def callees(self) -> dict:
        out = defaultdict(list)
        for (caller, callee), (n, c) in self.arcs.items():
            out[caller].append((callee, n, c))
        return out

    def call_counts(self) -> dict:
        n = defaultdict(int)
        for (caller, callee), (cnt, _) in self.arcs.items():
            n[callee] += cnt
        return n

    def hot_path(self, fn: str, ev: int = 0, max_depth: int = 12, callers=None) -> list:
        """Heaviest caller chain above ``fn`` (following the costliest arc)."""
        callers = callers if callers is not None else self.callers()
        path, seen, cur = [], {fn}, fn
        for _ in range(max_depth):
            cands = [(c[ev], caller) for caller, _, c in callers.get(cur, [])
                     if caller not in seen and caller != cur]
            if not cands:
                break
            _, best = max(cands)
            path.append(best)
            seen.add(best)
            cur = best
            if best in ("main", "(below main)"):
                break
        return path

    def omp_split(self) -> dict:
        """Instruction split for the main-thread view of OpenMP.

        ``parallel``: inclusive cost of outlined parallel-region bodies
        (``*._omp_fn.N``) entered from the OpenMP runtime; ``nested`` lists
        outlined bodies that are also reachable from inside another
        outlined body (their cost is then counted once, as part of the
        outer region).
        """
        ir = 0
        callees = self.callees()
        omp_fns = {fn for fn in set(self.self_cost) | {k[1] for k in self.arcs} if is_omp_fn(fn)}
        # functions reachable from any outlined body (context-insensitive)
        inside = set()
        stack = list(omp_fns)
        while stack:
            f = stack.pop()
            for g, _, _ in callees.get(f, []):
                if g not in inside:
                    inside.add(g)
                    stack.append(g)
        nested = sorted(f for f in omp_fns if f in inside)
        par = 0
        per_region = {}
        for (caller, callee), (n, c) in self.arcs.items():
            if is_omp_fn(callee) and not is_omp_fn(caller) and callee not in inside:
                par += c[ir]
                per_region[callee] = per_region.get(callee, 0) + c[ir]
        return {"parallel": par, "nested": nested, "per_region": per_region}
