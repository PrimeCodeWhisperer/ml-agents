import gspread
from oauth2client.service_account import ServiceAccountCredentials
import pandas as pd
from datetime import datetime

def upload_results_to_sheets(csv_path, spreadsheet_name, run_id):
    """
    Upload the last row from CSV training results to Google Sheets

    Args:
        csv_path: Path to the CSV file with training results
        spreadsheet_name: Name of the Google Sheet
        run_id: Unique identifier for this training run

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Set up credentials
        scope = [
            "https://spreadsheets.google.com/feeds",
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive"
        ]

        credentials = ServiceAccountCredentials.from_json_keyfile_name(
            'credentials.json', scope
        )

        client = gspread.authorize(credentials)

        # Open the spreadsheet
        spreadsheet = client.open(spreadsheet_name)
        worksheet = spreadsheet.sheet1

        # Read CSV data
        df = pd.read_csv(csv_path)

        if df.empty:
            print("CSV is empty, nothing to upload")
            return False

        # Check if sheet is empty (needs headers)
        existing_data = worksheet.get_all_values()

        if len(existing_data) == 0:
            # Sheet is empty, add headers first
            headers = df.columns.tolist()
            worksheet.append_row(headers)
            print("Added headers to Google Sheet")

        # Get the last row from the CSV (most recent training data)
        last_row = df.tail(1)
        values_to_append = last_row.values.tolist()[0]

        # Convert any NaN or None values to empty string
        values_to_append = [str(v) if pd.notna(v) else "" for v in values_to_append]

        # Append the single row (much faster than batch operations for single rows)
        worksheet.append_row(values_to_append)

        return True

    except FileNotFoundError:
        print(f"Error: credentials.json not found")
        print("Make sure the file is in the same directory as this script")
        return False

    except gspread.exceptions.SpreadsheetNotFound:
        print(f"Error: Spreadsheet '{spreadsheet_name}' not found")
        print("Make sure:")
        print("  1. The spreadsheet name is correct")
        print("  2. You've shared the sheet with your service account email")
        return False

    except Exception as e:
        print(f"Error uploading to Google Sheets: {e}")
        return False    


if __name__ == "__main__":
    # Test the function
    upload_results_to_sheets(
        csv_path="training_logs.csv",
        spreadsheet_name="ML-Agents Training Results",
        run_id="test_run"
    )
