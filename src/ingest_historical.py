"""Download hourly ERA5 weather data for Malang."""

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


API_URL = "https://archive-api.open-meteo.com/v1/archive"
VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "surface_pressure",
    "wind_speed_10m",
    "precipitation",
]
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def download(start, end, output):
    """Fetch monthly chunks and save a validated dataset."""
    if start > end:
        raise ValueError("Start date must not be after end date.")

    metadata_path = output.with_suffix(".metadata.json")
    if output.exists() or metadata_path.exists():
        raise FileExistsError("Output or metadata already exists.")

    base_params = {
        "latitude": -7.9839,
        "longitude": 112.6214,
        "hourly": ",".join(VARIABLES),
        "models": "era5",
        "timezone": "Asia/Jakarta",
        "temperature_unit": "celsius",
        "wind_speed_unit": "kmh",
        "precipitation_unit": "mm",
    }

    retry = Retry(
        total=3,
        backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )

    frames = []
    chunks = []
    expected_units = None
    cursor = start

    with requests.Session() as session:
        session.mount("https://", HTTPAdapter(max_retries=retry))

        while cursor <= end:
            chunk_end = min(cursor + pd.offsets.MonthEnd(0), end)
            params = {
                **base_params,
                "start_date": cursor.date().isoformat(),
                "end_date": chunk_end.date().isoformat(),
            }

            response = session.get(API_URL, params=params, timeout=90)
            response.raise_for_status()
            payload = response.json()

            if payload.get("timezone") != "Asia/Jakarta":
                raise ValueError("Unexpected response timezone.")

            units = payload["hourly_units"]
            if expected_units is None:
                expected_units = units
            elif units != expected_units:
                raise ValueError("Units differ between chunks.")

            frame = pd.DataFrame(payload["hourly"])
            frame = frame[["time", *VARIABLES]].copy()
            frames.append(frame)

            chunks.append({
                "start_date": params["start_date"],
                "end_date": params["end_date"],
                "latitude": payload.get("latitude"),
                "longitude": payload.get("longitude"),
                "elevation": payload.get("elevation"),
                "rows": len(frame),
            })
            print(
                f"{params['start_date']} to "
                f"{params['end_date']}: {len(frame)} rows",
                flush=True,
            )
            cursor = chunk_end + pd.Timedelta(days=1)

    data = pd.concat(frames, ignore_index=True)
    times = pd.DatetimeIndex(
        pd.to_datetime(data["time"], errors="raise")
    ).tz_localize("Asia/Jakarta")

    expected_times = pd.date_range(
        start=start.tz_localize("Asia/Jakarta"),
        end=(end + pd.Timedelta(hours=23)).tz_localize("Asia/Jakarta"),
        freq="h",
    )
    if not times.equals(expected_times):
        raise ValueError("Timestamps are incomplete, duplicated, or unordered.")

    for column in VARIABLES:
        data[column] = pd.to_numeric(data[column], errors="raise")

    values = data[VARIABLES]
    if values.isna().any().any():
        raise ValueError("Weather data contains missing values.")
    if values.isin([float("inf"), float("-inf")]).any().any():
        raise ValueError("Weather data contains infinite values.")

    # Store explicit timezone offsets in the CSV.
    data["time"] = times
    csv_bytes = data.to_csv(index=False).encode("utf-8")

    metadata = {
        "source": "Open-Meteo Historical Weather API",
        "data_type": "reanalysis",
        "endpoint": API_URL,
        "parameters": {
            **base_params,
            "start_date": start.date().isoformat(),
            "end_date": end.date().isoformat(),
        },
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "hourly_units": expected_units,
        "rows": len(data),
        "sha256": hashlib.sha256(csv_bytes).hexdigest(),
        "chunks": chunks,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("x", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, ensure_ascii=False)
        file.write("\n")
    with output.open("xb") as file:
        file.write(csv_bytes)

    print(f"\nSaved: {output}")
    print(f"Metadata: {metadata_path}")
    print(f"Total rows: {len(data)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    try:
        start = pd.Timestamp(
            datetime.strptime(args.start_date, "%Y-%m-%d")
        )
        end = pd.Timestamp(
            datetime.strptime(args.end_date, "%Y-%m-%d")
        )
        output = args.output
        if not output.is_absolute():
            output = PROJECT_ROOT / output
        download(start, end, output)
    except (OSError, ValueError, KeyError, requests.RequestException) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()