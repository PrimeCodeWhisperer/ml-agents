import subprocess
import datetime
import os
import sys

def main():
    # === Configuration ===

    if len(sys.argv) != 2:
        print("2 arguments required", file=sys.stderr)
        sys.exit(1)

    
    unity_env_path = "Project/Build.app"  # path to your Unity build (.app), (this changes depending on your build)
    config_file = "config/ppo/3DBall.yaml"  # path to your ML-Agents config YAML
    base_port = 5004
    num_runs = int(sys.argv[1])


    for i in range(1, num_runs+1):
        # Create unique run ID
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        run_id = f"testrun-{timestamp}"

        # Build mlagents-learn command
        trainer_cmd = [
            "mlagents-learn",
            config_file,
            f"--run-id={run_id}",
            "--env", unity_env_path,
            "--no-graphics",
            "--force",
            "--base-port", str(base_port)
        ]

        print("\n Launching ML-Agents training...")

        # Run the trainer
        try:
            subprocess.run(trainer_cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"\n Training process failed with error code {e.returncode}")
            return

        print("\n Training finished successfully.")

        # === Optional: launch training logger after run ===
        logger_cmd = [
            "python", "training_logger.py",
            "--run-id", run_id
        ]

        print("\n Generating training logs...")
        try:
            subprocess.run(logger_cmd, check=True)
        except subprocess.CalledProcessError:
            print("Training logger failed")

if __name__ == "__main__":
    main()
