"""
Weather data ingestion script.

Fetches weather data for Malang from the Open-Meteo API and saves
timestamped CSV files to data/raw/ at the project root.
"""

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests


def fetch_weather_data() -> None:
    """Fetch weather data and save it as a timestamped raw CSV file."""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": -7.9839,
        "longitude": 112.6214,
        "hourly": [
            "temperature_2m",
            "relative_humidity_2m",
            "surface_pressure",
            "wind_speed_10m",
            "precipitation",
        ],
        "timezone": "Asia/Jakarta",
        "past_days": 2,
    }

    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{current_time}] Sending request to the Open-Meteo API...")

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()

        data = response.json()
        df = pd.DataFrame(data["hourly"])
        df["time"] = pd.to_datetime(df["time"])

        # Find the project root.
        project_root = Path(__file__).resolve().parents[1]

        # Create the raw data folder.
        output_dir = project_root / "data" / "raw"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Use a timestamp to avoid overwriting old files.
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        output_file = output_dir / f"weather_raw_{timestamp}.csv"

        df.to_csv(output_file, index=False)

        print("\n" + "=" * 50)
        print("DATA INGESTION COMPLETED SUCCESSFULLY")
        print("=" * 50)
        print(f"Output file : {output_file}")
        print(f"Total rows  : {len(df)}")
        print("\nPreview of the first five rows:")
        print(df.head())
        print("=" * 50 + "\n")

    except requests.exceptions.RequestException as exc:
        print(f"[ERROR] Connection or HTTP request failed: {exc}")
        sys.exit(1)

    except Exception as exc:
        print(f"[ERROR] Failed to retrieve or save weather data: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    fetch_weather_data()