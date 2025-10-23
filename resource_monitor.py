#!/usr/bin/env python3

import argparse
import csv
import datetime as dt
import os
import psutil
import sys
import time
from typing import Dict, Iterable, List, Set, Tuple

def _mib(b: int) -> float:
    return b / (1024.0 * 1024.0)

def _find_procs(name_substrings: List[str], pid_whitelist: Set[int]) -> Dict[int, psutil.Process]:
    matches: Dict[int, psutil.Process] = {}
    lowered = [s.lower() for s in name_substrings] if name_substrings else []

    for p in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            if pid_whitelist and p.info["pid"] in pid_whitelist:
                matches[p.pid] = p
                continue
            if lowered:
                hay = " ".join(p.info.get("cmdline") or []) + " " + (p.info.get("name") or "")
                hay = hay.lower()
                if any(sub in hay for sub in lowered):
                    matches[p.pid] = p
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    add_children: Dict[int, psutil.Process] = {}
    for p in list(matches.values()):
        try:
            for c in p.children(recursive=True):
                add_children[c.pid] = c
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    matches.update(add_children)
    return matches

def _prime_cpu(procs: Iterable[psutil.Process]) -> None:
    for p in procs:
        try:
            p.cpu_percent(interval=None)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

def _sample(procs: Dict[int, psutil.Process], detailed: bool):
    system_cpu = psutil.cpu_percent(interval=None)
    vm = psutil.virtual_memory()
    system_mem_percent = vm.percent

    tracked_cpu = 0.0
    tracked_rss = 0
    tracked_vms = 0
    detail_rows = []
    dead = []

    for pid, p in procs.items():
        try:
            cpu = p.cpu_percent(interval=None)
            mem = p.memory_info()
            tracked_cpu += cpu
            tracked_rss += mem.rss
            tracked_vms += mem.vms
            if detailed:
                detail_rows.append({
                    "pid": pid,
                    "name": p.name(),
                    "cpu": round(cpu, 3),
                    "rss": round(_mib(mem.rss), 3),
                    "vms": round(_mib(mem.vms), 3),
                })
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            dead.append(pid)

    for pid in dead:
        procs.pop(pid, None)

    aggregate = {
        "system_cpu": round(system_cpu, 3),
        "system_mem": round(system_mem_percent, 3),
        "tracked_cpu": round(tracked_cpu, 3),
        "tracked_rss": round(_mib(tracked_rss), 3),
        "tracked_vms": round(_mib(tracked_vms), 3),
        "n_procs": len(procs),
    }
    return aggregate, detail_rows

def main():
    ap = argparse.ArgumentParser(description="Log CPU & RAM for Unity ML-Agents to CSV")
    ap.add_argument("--names", nargs="*", default=[], help="Process name substrings (e.g., Unity mlagents-learn python)")
    ap.add_argument("--pids", nargs="*", type=int, default=[], help="Specific PIDs to track (children included)")
    ap.add_argument("--interval", type=float, default=1.0, help="Sampling interval seconds (default 1.0)")
    ap.add_argument("--outfile", type=str, default="performance_log.csv", help="Output CSV path")
    ap.add_argument("--run-id", type=str, default="", help="Optional run id column")
    ap.add_argument("--detailed", action="store_true", help="Also log one row per process per sample")
    ap.add_argument("--max-seconds", type=float, default=0.0, help="Stop after this many seconds (0 = unlimited)")
    ap.add_argument("--refresh-procs-every", type=float, default=5.0, help="How often to refresh process discovery (seconds)")
    ap.add_argument("--stop-when-none", action="store_true", help="Exit when no matching processes remain")
    args = ap.parse_args()

    name_filters = args.names or []
    pid_whitelist = set(args.pids or [])

    header = [
        "timestamp", "run_id", "row_type",
        "system_cpu_percent", "system_mem_percent",
        "tracked_cpu_percent", "tracked_rss_mib", "tracked_vms_mib", "procs_tracked",
        "pid", "proc_name", "proc_cpu_percent", "proc_rss_mib", "proc_vms_mib"
    ]

    new_file = not (os.path.exists(args.outfile) and os.path.getsize(args.outfile) > 0)
    f = open(args.outfile, "a", newline="", encoding="utf-8")
    writer = csv.writer(f)
    if new_file:
        writer.writerow(header)

    procs = _find_procs(name_filters, pid_whitelist)
    if not procs:
        print("No matching processes found at start. The monitor will keep looking...")
    _prime_cpu(procs.values())
    last_refresh = time.monotonic()

    started = time.monotonic()
    try:
        while True:
            now = dt.datetime.now().isoformat(timespec="seconds")


            if (time.monotonic() - last_refresh) >= args.refresh_procs_every:
                current = _find_procs(name_filters, pid_whitelist)
                for pid, p in current.items():
                    if pid not in procs:
                        try:
                            p.cpu_percent(interval=None)
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            pass
                procs = {pid: p for pid, p in current.items() if p.is_running()}
                last_refresh = time.monotonic()

            time.sleep(max(0.0, args.interval))
            agg, details = _sample(procs, args.detailed)

            writer.writerow([
                now, args.run_id, "aggregate",
                agg["system_cpu"], agg["system_mem"],
                agg["tracked_cpu"], agg["tracked_rss"], agg["tracked_vms"], agg["n_procs"],
                "", "", "", "", ""
            ])

            if args.detailed:
                for d in details:
                    writer.writerow([
                        now, args.run_id, "process",
                        "", "", "", "", "", "",
                        d["pid"], d["name"], d["cpu"], d["rss"], d["vms"]
                    ])

            f.flush()

            if args.max_seconds and (time.monotonic() - started) >= args.max_seconds:
                print("Reached --max-seconds; exiting.")
                break
            if args.stop_when_none and len(procs) == 0 and (name_filters or pid_whitelist):
                print("No matching processes remain; exiting.")
                break

    except KeyboardInterrupt:
        print("Interrupted; exiting.")
    finally:
        f.close()

if __name__ == "__main__":
    main()
