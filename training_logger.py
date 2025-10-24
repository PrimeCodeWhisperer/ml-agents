import os
import argparse
import pandas as pd
from datetime import datetime
from tensorboard.backend.event_processing import event_accumulator


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

    #Estimates the training time
    start_time = os.path.getctime(event_path)
    end_time = os.path.getmtime(event_path)
    training_time_s = end_time - start_time
    #calculates the amount of steps per second
    try:
        events = ea.Scalars("Environment/Cumulative Reward")
        steps_per_second = (
            (events[-1].step) / training_time_s if training_time_s > 0 else None
        )
    except Exception:
        steps_per_second = None

    # Loads the resource csv file that the training_controller.py 
    resource_file = os.path.join(base_dir, f"{run_id}_resources.csv")
    avg_system_cpu = avg_cpu = avg_ram = max_ram = None
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
    # Creates the data set for the final CSV
    data = {
        "run_id": run_id,
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
        "policy_loss": get_last("Losses/Policy Loss"),
        "value_loss": get_last("Losses/Value Loss"),
        "learning_rate": get_last("Policy/Learning Rate"),
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


if __name__ == "__main__":
    main()
