import os
import argparse
import psutil
import pandas as pd
import platform
from datetime import datetime
from tensorboard.backend.event_processing import event_accumulator
from tbparse import SummaryReader
from csv_validator import csv_is_complete_embedded

def collectTrainingData(run_id):
    #Finds the tensorboard file for the training data
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

    print("Available text tags:")
    print(df['tag'].tolist())

    hyper_row = df[df['tag'].str.contains('Hyperparameters', case=False)]

    if	not hyper_row.empty:
        text = hyper_row.iloc[0]['value']
        print("Found hyperparameters text:")
        print(text[:500])

        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('batch_size:'):
                try:
                    batch_size = int(line.split(':')[1].strip())
                    print(f"Found batch_size: {batch_size}")
                except:
                    pass
            if line.startswith('trainer_type:'):
                try:
                    algorithm = line.split(':')[1].strip()
                    print(f"Found algorithm: {algorithm}")
                except:
                    pass

    # Get the scene variable -----------------------------------------------------
    # The idea is to find the directory name in the base directory that is not called run_logs, this is always the name of the scene.
    items = os.listdir(base_dir)
    #this line filters on only directories.
    directories = [item for item in items if os.path.isdir(os.path.join(base_dir, item))]
    if 'run_logs' in directories:directories.remove('run_logs')
    if len(directories) == 1:
        scene = directories[0]
    else:
        scene = 'Unknown'

    #Gets the last output of the results data for a tag
    def get_last(tag):
        try:
            return ea.Scalars(tag)[-1].value
        except Exception:
            return None
    #same as get_last but for the number of steps
    def get_last_step(tag):
        try:
            return ea.Scalars(tag)[-1].step
        except Exception:
            return None

    #Estimates the training time and steps_per_second using event wall_time (more reliable than file times)
    try:
        events = ea.Scalars("Environment/Cumulative Reward")
        if len(events) >= 2:
            training_time_s = events[-1].wall_time - events[0].wall_time
            steps_per_second = (
                events[-1].step / training_time_s if training_time_s > 0 else None
            )
        else:
            training_time_s = 0
            steps_per_second = None
    except Exception:
        training_time_s = 0
        steps_per_second = None

    # Loads the resource csv file that the training_controller.py
    resource_file = os.path.join(base_dir, f"{run_id}_resources.csv")
    avg_system_cpu = avg_cpu = avg_ram = max_ram = None

    # Gets operating system information
    operating_system=platform.platform(terse=True)

    #Reads the resource csv file
    if os.path.exists(resource_file):
        df = pd.read_csv(resource_file)
        if not df.empty and "row_type" in df.columns:

            df = df[df["row_type"] == "aggregate"]
            if "procs_tracked" in df.columns:
                df = df[df["procs_tracked"] > 0]

            if not df.empty:
                avg_system_cpu = df["system_cpu_percent"].mean()
                avg_cpu = df["tracked_cpu_percent"].mean()
                avg_ram = df["tracked_rss_mib"].mean()
                max_ram = df["tracked_rss_mib"].max()
    current_time = datetime.now()
    num_cores = os.cpu_count()

    # Teads out the total amount of RAM
    total_ram = psutil.virtual_memory().total
    total_ram_gb = total_ram / (1024**3)
    # Creates the data set for the final CSV
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
        "operating_system":operating_system,
        "cpu_model":platform.processor()
    }


    return data

#Creates the final CSV file with all the data
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
    if(not is_ok):
        raise SystemError("CSV has not been created succesfully")
    else:
        print("CSV file created correctly")



if __name__ == "__main__":
    main()
