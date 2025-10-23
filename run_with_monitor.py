#!/usr/bin/env python3
import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

def main():
    ap = argparse.ArgumentParser(description="Run ML-Agents command and auto-monitor CPU/RAM")
    ap.add_argument("--cmd", required=True, help="Training command to run (quote it)")
    ap.add_argument("--outfile", default="performance_log.csv", help="CSV file for resource monitor")
    ap.add_argument("--run-id", default="", help="Run id to pass to monitor")
    ap.add_argument("--interval", type=float, default=1.0, help="Sampling interval seconds")
    ap.add_argument("--detailed", action="store_true", help="Also log per-process rows")
    args = ap.parse_args()

    script_dir = Path(__file__).resolve().parent
    monitor_path = script_dir / "resource_monitor.py"
    if not monitor_path.exists():
        print(f"[!] resource_monitor.py not found next to this script: {monitor_path}")
        sys.exit(1)

    print(f"Launching trainer: {args.cmd}")
    trainer = subprocess.Popen(args.cmd, shell=True)

    py = sys.executable
    mon_cmd = [
        py, str(monitor_path),
        "--pids", str(trainer.pid),
        "--outfile", args.outfile,
        "--run-id", args.run_id,
        "--interval", str(args.interval),
        "--stop-when-none",
    ]
    if args.detailed:
        mon_cmd.append("--detailed")

    print("Launching monitor:", " ".join(shlex.quote(c) for c in mon_cmd))
    monitor = subprocess.Popen(mon_cmd)

    code = trainer.wait()
    print(f"Trainer exited with code {code}. Waiting for monitor to finish...")
    monitor.wait()
    print("Done. Logs written to:", args.outfile)

if __name__ == "__main__":
    main()
