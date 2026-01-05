import os
import argparse
import psutil
import pandas as pd
import platform
from datetime import datetime
from tensorboard.backend.event_processing import event_accumulator
from tbparse import SummaryReader
from csv_validator import csv_is_complete_embedded

# Set your desired reward threshold here
REWARD_THRESHOLD = 1.2

def collectTrainingData(run_id, reward_threshold=REWARD_THRESHOLD):
    # Finds the tensorboard file for the training data
    base_dir = os.path.join("results", run_id)
    if not os.path.exists(base_dir):
        raise FileNotFoundError(f"Run folder not found: {base_dir}")

    event_path = None
    for root, _, files in os.walk(base_dir):
        for f in files:
            if f.startswith("events.out.tfevents"):
                event_path = os.path.join(root, f)
                break
        if event_path:
            break

    if not event_path:
        raise FileNotFoundError(f"No TensorBoard event file found in {base_dir}")

    ea = event_accumulator.EventAccumulator(event_path)
    ea.Reload()

    batch_size = None
    algorithm = None
    scene = None

    reader = SummaryReader(event_path)
    df = reader.text

    hyper_row = df[df['tag'].str.contains('Hyperparameters', case=False)]
    if not hyper_row.empty:
        text = hyper_row.iloc[0]['value']
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('batch_size:'):
                try:
                    batch_size = int(line.split(':')[1].strip())
                except:
                    pass
            if line.startswith('trainer_type:'):
                try:
                    algorithm = line.split(':')[1].strip()
                except:
                    pass

    # Get scene
    items = os.listdir(base_dir)
    directories = [item for item in items if os.path.isdir(os.path.join(base_dir, item))]
    if 'run_logs' in directories: directories.remove('run_logs')
    scene = directories[0] if len(directories) == 1 else 'Unknown'

    # Helper functions
    def get_last(tag):
        try:
            return ea.Scalars(tag)[-1].value
        except Exception:
            return None

    def get_last_step(tag):
        try:
            return ea.Scalars(tag)[-1].step
        except Exception:
            return None

    # Compute training time and steps per second
    try:
        events = ea.Scalars("Environment/Cumulative Reward")
        if len(events) >= 2:
            training_time_s = events[-1].wall_time - events[0].wall_time
            steps_per_second = events[-1].step / training_time_s if training_time_s > 0 else None
        else:
            training_time_s = 0
            steps_per_second = None
    except Exception:
        training_time_s = 0
        steps_per_second = None

    steps_to_threshold = None
    threshold_reached = False

    for e in events:
        if e.value >= reward_threshold:
            steps_to_threshold = e.step
            time_to_threshold_s = e.wall_time - events[0].wall_time
            threshold_reached = True
            break

    # Resource CSV
    resource_file = os.path.join(base_dir, f"{run_id}_resources.csv")
    avg_system_cpu = avg_cpu = avg_ram = max_ram = None

    # Gets operating system information
    operating_system=platform.platform(terse=False)

    #Reads the resource csv file
    if os.path.exists(resource_file):
        df_res = pd.read_csv(resource_file)
        if not df_res.empty and "row_type" in df_res.columns:
            df_res = df_res[df_res["row_type"] == "aggregate"]
            if "procs_tracked" in df_res.columns:
                df_res = df_res[df_res["procs_tracked"] > 0]
            if not df_res.empty:
                avg_system_cpu = df_res["system_cpu_percent"].mean()
                avg_cpu = df_res["tracked_cpu_percent"].mean()
                avg_ram = df_res["tracked_rss_mib"].mean()
                max_ram = df_res["tracked_rss_mib"].max()

    current_time = datetime.now()
    num_cores = os.cpu_count()
    total_ram_gb = psutil.virtual_memory().total / (1024**3)

    data = {
        "run_id": run_id,
        "scene": scene,
        "batch_size": batch_size,
        "algorithm": algorithm,
        "current_time": current_time,
        "mean_reward": get_last("Environment/Cumulative Reward"),
        "training_time_s": training_time_s,
        "total_steps": get_last_step("Environment/Cumulative Reward"),
        "steps_per_second": steps_per_second,
        "avg_system_cpu_percent": avg_system_cpu,
        "avg_tracked_cpu_percent": avg_cpu,
        "avg_ram_usage": avg_ram,
        "max_ram_usage": max_ram,
        "num_cores": num_cores,
        "total_ram": total_ram_gb,
        "policy_loss": get_last("Losses/Policy Loss"),
        "value_loss": get_last("Losses/Value Loss"),
        "learning_rate": get_last("Policy/Learning Rate"),
        "operating_system": platform.platform(terse=True),
        "cpu_model": platform.processor(),
        # --- New threshold fields ---
        "steps_to_threshold": steps_to_threshold,
        "threshold_reached": threshold_reached
    }

    return data


def createCSV(data, output_file="training_logs.csv"):
    df = pd.DataFrame([data])
    if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
        df.to_csv(output_file, mode="a", header=False, index=False)
    else:
        df.to_csv(output_file, index=False)
    print(f"Saved results to: {os.path.abspath(output_file)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True, help="Folder name under ./results/")
    parser.add_argument("--out", default="training_logs.csv", help="Output CSV path")
    args = parser.parse_args()

    data = collectTrainingData(args.run_id)
    createCSV(data, args.out)
    is_ok = csv_is_complete_embedded("training_logs.csv")
    if not is_ok:
        raise SystemError("CSV has not been created succesfully")
    print("CSV file created correctly")


if __name__ == "__main__":
    main()

