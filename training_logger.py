import os
import argparse
import pandas as pd
from tensorboard.backend.event_processing import event_accumulator


def collectTrainingData(run_id):
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
    
    try:
        events = ea.Scalars("Environment/Cumulative Reward")
        steps_per_second = (events[-1].step - events[0].step) / (events[-1].wall_time - events[0].wall_time)
    except Exception:
        steps_per_second = None
    # Approximates training time
    start_time = os.path.getctime(event_path)
    end_time = os.path.getmtime(event_path)
    training_time_s = end_time - start_time

    #Collects data from the results folder
    data = {
        "run_id": run_id,
        "mean_reward": get_last("Environment/Cumulative Reward"),
        "training_time_s": training_time_s,
        "total_steps": get_last_step("Environment/Cumulative Reward"),
        "steps_per_second": steps_per_second,
        "policy_loss": get_last("Losses/Policy Loss"),
        "value_loss": get_last("Losses/Value Loss"),
        "learning_rate": get_last("Policy/Learning Rate"),
        "entropy": get_last("Policy/Entropy"),
        
    }

    print(f"Collected data for {run_id}")
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
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    data = collectTrainingData(args.run_id)
    createCSV(data)


if __name__ == "__main__":
    main()
