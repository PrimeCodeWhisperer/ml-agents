import subprocess
import datetime
import os
import sys
from upload_to_sheets import upload_results_to_sheets
from dotenv import load_dotenv


def main():
    # === Configuration ===
    load_dotenv()

    if len(sys.argv) != 2:
        print("2 arguments required", file=sys.stderr)
        sys.exit(1)


    unity_env_path = os.getenv('UNITY_ENV_PATH')  # path to your Unity build (.app), (this changes depending on your build)
    config_file = os.getenv('CONFIG_PATH')  # path to your ML-Agents config YAML
    base_port = 5004
    num_runs = int(sys.argv[1])

    # Google Sheets Configuration
    spreadsheet_name = "ML-Agents Training Results"
    output_csv = "training_logs.csv"


    for i in range(1, num_runs+1):
        print(f"\n{'='*60}")
        print(f"Starting Training Run {i}/{num_runs}")
        print(f"{'='*60}")

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

        print(f"\n[Step 1/{3}] Launching ML-Agents training...")
        print(f"Run ID: {run_id}")

        # Run the trainer
        try:
            subprocess.run(trainer_cmd, check=True)
            print("\n✓ Training finished successfully.")
        except subprocess.CalledProcessError as e:
            print(f"\n✗ Training process failed with error code {e.returncode}")
            print(f"Skipping to next run...")
            continue  # Skip to next iteration instead of returning

        # Launch training logger after run
        print(f"\n[Step 2/3] Generating training logs from TensorBoard data...")

        logger_cmd = [
            "python", "training_logger.py",
            "--run-id", run_id,
            "--out", output_csv
        ]

        try:
            subprocess.run(logger_cmd, check=True)
            print("✓ Training logs generated successfully.")
        except subprocess.CalledProcessError as e:
            print(f"✗ Training logger failed with error code {e.returncode}")
            print(f"Skipping Google Sheets upload for this run...")
            continue  # Skip to next iteration

        # === Upload to Google Sheets ===
        print(f"\n[Step 3/3] Uploading results to Google Sheets...")

        if os.path.exists(output_csv):
            try:
                # Read the last row from the CSV (the one just added)
                import pandas as pd
                df = pd.read_csv(output_csv)

                if not df.empty:
                    # Get the last row (most recent training data)
                    last_row = df.tail(1)

                    # Convert to list of lists for upload
                    # First time: include headers, subsequent times: data only
                    upload_success = upload_results_to_sheets(
                        csv_path=output_csv,
                        spreadsheet_name=spreadsheet_name,
                        run_id=run_id
                    )

                    if upload_success:
                        print(f"✓ Successfully uploaded results to Google Sheets")
                        print(f"  Spreadsheet: {spreadsheet_name}")
                        print(f"  Run ID: {run_id}")
                    else:
                        print(f"✗ Failed to upload to Google Sheets")
                        print(f"  Results saved locally in: {output_csv}")
                else:
                    print(f"⚠ CSV file is empty, skipping upload")

            except Exception as e:
                print(f"✗ Error during upload process: {e}")
                print(f"  Results saved locally in: {output_csv}")
        else:
            print(f"⚠ Warning: CSV file not found at {output_csv}")

        print(f"\n{'='*60}")
        print(f"Completed Run {i}/{num_runs}")
        print(f"{'='*60}\n")

    print(f"\n All {num_runs} training runs completed!")
    print(f" Results available in Google Sheets: {spreadsheet_name}")
    print(f" Local backup: {output_csv}")


if __name__ == "__main__":
    main()
