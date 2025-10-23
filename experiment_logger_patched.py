#!/usr/bin/env python3
import os, argparse, csv, sys
from pathlib import Path

try:
    import pandas as pd
except Exception:
    pd = None
try:
    from tensorboard.backend.event_processing import event_accumulator
except Exception:
    event_accumulator = None

def summarise_perf(perf_csv, run_id):
    if not perf_csv or not os.path.exists(perf_csv) or pd is None:
        return {}
    try:
        df = pd.read_csv(perf_csv)
    except Exception as e:
        print(f"[warn] Could not read perf CSV: {e}")
        return {}
    if "run_id" in df.columns:
        df = df[df["run_id"] == run_id]
    if "row_type" in df.columns:
        df = df[df["row_type"] == "aggregate"]
    if df.empty:
        return {}
    out = {
        "cpu_mean_pct": float(df["tracked_cpu_percent"].mean()),
        "cpu_peak_pct": float(df["tracked_cpu_percent"].max()),
        "ram_mean_mib": float(df["tracked_rss_mib"].mean()),
        "ram_peak_mib": float(df["tracked_rss_mib"].max()),
        "sys_mem_mean_pct": float(df["system_mem_percent"].mean()),
        "samples": int(len(df)),
    }
    return out

def collect_tb_metrics(results_dir, run_id):
    if event_accumulator is None:
        return {}
    run_dir = os.path.join(results_dir, run_id)
    if not os.path.exists(run_dir):
        return {}
    event_path = None
    for root, _, files in os.walk(run_dir):
        for f in files:
            if f.startswith("events.out.tfevents"):
                event_path = os.path.join(root, f)
                break
        if event_path:
            break
    if not event_path:
        return {}
    try:
        ea = event_accumulator.EventAccumulator(event_path)
        ea.Reload()
    except Exception as e:
        print(f"[warn] TB load failed: {e}")
        return {}

    def last_scalar(tag):
        try:
            vals = ea.Scalars(tag)
            return vals[-1].value if vals else None
        except KeyError:
            return None

    data = {}
    for tag in [
        "Environment/Cumulative Reward",
        "Losses/Policy Loss",
        "Losses/Value Loss",
        "Policy/Learning Rate",
        "Policy/Entropy"
    ]:
        v = last_scalar(tag)
        if v is not None:
            key = tag.lower().replace("/", "_").replace(" ", "_")
            data[key] = v
    return data

def save_row(output_csv, row):
    # Append or create with header
    exists = os.path.exists(output_csv) and os.path.getsize(output_csv) > 0
    with open(output_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)

def main():
    ap = argparse.ArgumentParser(description="Summarize training + perf metrics to CSV")
    ap.add_argument("--run-id", required=True, help="Run identifier (used by perf CSV and results/<run_id>)")
    ap.add_argument("--perf-csv", default=None, help="Path to resource_monitor CSV")
    ap.add_argument("--results-dir", default="results", help="ML-Agents results directory")
    ap.add_argument("--out", default="training_summary.csv", help="Output CSV to append to")
    args = ap.parse_args()

    row = {"run_id": args.run_id}
    row.update(collect_tb_metrics(args.results_dir, args.run_id))
    row.update(summarise_perf(args.perf_csv, args.run_id))

    save_row(args.out, row)
    print(f"[ok] Wrote summary for {args.run_id} to {os.path.abspath(args.out)}")

if __name__ == "__main__":
    main()
