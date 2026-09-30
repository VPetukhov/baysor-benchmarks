"""Run a command natively and sample its CPU time / RSS from /proc.

Used by the native timing jobs of the quick tier and by the scaling tier.
The sampler reads ``/proc/<pid>/stat`` (utime+stime of the whole process,
including exited threads) and ``/proc/<pid>/status`` (VmRSS, VmHWM,
Threads) every ``interval`` seconds. ``phase_table`` aligns the samples with
Baysor's spdlog phase lines (same wall clock), giving CPU seconds, wall
seconds, CPU/wall (effective parallelism) and peak RSS per phase.
"""
from __future__ import annotations

import os
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path

CLK_TCK = os.sysconf("SC_CLK_TCK")

# Baysor log line -> phase that starts at that line (order of cmd_run)
LOG_PHASES = [
    ("loading", re.compile(r"Loading data from")),
    ("confidence", re.compile(r"Estimating confidence")),
    ("molecule_graph", re.compile(r"Building molecule graph")),
    ("mol_clustering", re.compile(r"Clustering molecules")),
    ("bmm_init", re.compile(r"Initializing BmmData")),
    ("bmm_iterations", re.compile(r"Running segmentation")),
    ("ncv_colors", re.compile(r"Computing neighborhood composition colors")),
    ("out_molecules", re.compile(r"Saving segmented molecule table")),
    ("out_cell_stats", re.compile(r"Saving cell stats")),
    ("polygons", re.compile(r"Estimating boundary polygons")),
    ("out_polygons", re.compile(r"Saving cell polygons")),
    ("out_counts", re.compile(r"Saving count matrix")),
    ("report", re.compile(r"Generating HTML run report")),
    ("shutdown", re.compile(r"Results saved to")),
]
_TS_RE = re.compile(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d+)\]")


def loadavg() -> list[float]:
    try:
        return [float(v) for v in Path("/proc/loadavg").read_text().split()[:3]]
    except OSError:
        return []


def _read_sample(pid: int):
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
        status = Path(f"/proc/{pid}/status").read_text()
    except OSError:
        return None
    fields = stat[stat.rindex(")") + 2:].split()
    cpu = (int(fields[11]) + int(fields[12])) / CLK_TCK      # utime + stime
    vals = {}
    for line in status.splitlines():
        k, _, v = line.partition(":")
        if k in ("VmRSS", "VmHWM", "Threads"):
            vals[k] = int(v.split()[0])
    return cpu, vals.get("VmRSS", 0), vals.get("VmHWM", 0), vals.get("Threads", 0)


def run_monitored(cmd: list[str], env: dict, log_path: Path, interval: float = 1.0,
                  cwd: str | None = None, timeout: float | None = None) -> dict:
    """Run ``cmd``, stdout+stderr to ``log_path``; return timing + samples."""
    samples = []
    la0 = loadavg()
    t0 = time.time()
    with open(log_path, "w") as log:
        proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, cwd=cwd)
        status, ru = None, None
        next_t = t0
        while True:
            pid, st, r = os.wait4(proc.pid, os.WNOHANG)
            if pid:
                status, ru = st, r
                break
            now = time.time()
            if now >= next_t:
                s = _read_sample(proc.pid)
                if s is not None:
                    samples.append([round(now, 3), s[0], s[1], s[2], s[3]])
                next_t = now + interval
            if timeout and now - t0 > timeout:
                proc.kill()
            time.sleep(min(interval, 0.05))
        proc.returncode = os.waitstatus_to_exitcode(status)
    t1 = time.time()
    # ru_maxrss of a Popen child also counts the forked Python image before
    # exec; VmHWM of the exec'd process (monotonic, sampled) is the real peak
    hwm = max((smp[3] for smp in samples if smp[3]), default=0)
    return {
        "command": cmd,
        "exit_code": proc.returncode,
        "t_start": t0, "t_end": t1, "wall_s": round(t1 - t0, 3),
        "cpu_user_s": round(ru.ru_utime, 3), "cpu_sys_s": round(ru.ru_stime, 3),
        "cpu_s": round(ru.ru_utime + ru.ru_stime, 3),
        "peak_rss_kb": int(hwm or ru.ru_maxrss),
        "ru_maxrss_kb": int(ru.ru_maxrss),
        "loadavg_start": la0, "loadavg_end": loadavg(),
        "sample_interval_s": interval,
        "samples_columns": ["t", "cpu_s", "rss_kb", "hwm_kb", "threads"],
        "samples": samples,
    }


def parse_log_phases(log_path: Path) -> list[tuple[str, float]]:
    """[(phase, epoch_start), ...] from Baysor's timestamped log lines."""
    out = []
    seen = set()
    with open(log_path, errors="replace") as fh:
        for line in fh:
            m = _TS_RE.match(line)
            if not m:
                continue
            for name, rx in LOG_PHASES:
                if name not in seen and rx.search(line):
                    ts = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S.%f").timestamp()
                    out.append((name, ts))
                    seen.add(name)
                    break
    return out


def _interp(samples, t, col):
    """Linear interpolation of a monotone sample column at time t."""
    if not samples:
        return 0.0
    if t <= samples[0][0]:
        return samples[0][col]
    for a, b in zip(samples, samples[1:]):
        if a[0] <= t <= b[0]:
            if b[0] == a[0]:
                return b[col]
            return a[col] + (b[col] - a[col]) * (t - a[0]) / (b[0] - a[0])
    return samples[-1][col]


def phase_table(run: dict, log_path: Path) -> list[dict]:
    """Per-phase wall, CPU seconds, CPU/wall and peak RSS from one monitored run."""
    marks = parse_log_phases(log_path)
    samples = run["samples"]
    # append the final rusage totals as the closing sample
    final = [run["t_end"], run["cpu_s"], 0, run["peak_rss_kb"], 0]
    samp = [[run["t_start"], 0.0, 0, 0, 0]] + samples + [final]
    bounds = [("startup", run["t_start"])] + marks + [("__end__", run["t_end"])]
    rows = []
    for (name, ts), (_, te) in zip(bounds, bounds[1:]):
        ts = max(ts, run["t_start"])
        te = min(max(te, ts), run["t_end"])
        cpu = _interp(samp, te, 1) - _interp(samp, ts, 1)
        inside = [s for s in samples if ts <= s[0] <= te]
        rss_peak = max([s[2] for s in inside], default=None)
        hwm = max([s[3] for s in inside], default=None)
        wall = te - ts
        rows.append({
            "phase": name, "wall_s": round(wall, 3), "cpu_s": round(max(cpu, 0.0), 3),
            "cpu_per_wall": round(cpu / wall, 3) if wall > 0.05 else None,
            "rss_peak_kb": rss_peak, "hwm_end_kb": hwm, "n_samples": len(inside),
        })
    return rows

