"""
Clean raw weather data and save it to data/processed/.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd


WEATHER_COLUMNS = [
    "temperature_2m",
    "relative_humidity_2m",
    "surface_pressure",
    "wind_speed_10m",
    "precipitation",
]


def preprocess_data(input_file: Path) -> Path:
    """Clean a raw weather CSV file."""
    if not input_file.is_file():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    df = pd.read_csv(input_file)

    # Check the required columns.
    required_columns = ["time", *WEATHER_COLUMNS]
    missing_columns = set(required_columns) - set(df.columns)

    if missing_columns:
        raise ValueError(f"Missing columns: {sorted(missing_columns)}")

    if df.empty:
        raise ValueError("The input file is empty.")

    df = df[required_columns].copy()
    input_rows = len(df)

    # Read the timestamps.
    df["time"] = pd.to_datetime(df["time"], errors="raise")

    if df["time"].isna().any():
        raise ValueError("The time column contains missing timestamps.")

    # Use Jakarta time.
    if df["time"].dt.tz is None:
        df["time"] = df["time"].dt.tz_localize("Asia/Jakarta")
    else:
        df["time"] = df["time"].dt.tz_convert("Asia/Jakarta")

    # Remove duplicate timestamps.
    df = df.drop_duplicates(subset="time", keep="last")
    duplicate_rows = input_rows - len(df)

    # Sort the data by time.
    df = df.sort_values("time").reset_index(drop=True)

    # Convert weather values to numbers.
    for column in WEATHER_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    # Treat infinite values as missing values.
    df[WEATHER_COLUMNS] = df[WEATHER_COLUMNS].replace(
        [float("inf"), float("-inf")],
        float("nan"),
    )

    # Check for columns with no valid values.
    empty_columns = [
        column
        for column in WEATHER_COLUMNS
        if df[column].isna().all()
    ]

    if empty_columns:
        raise ValueError(f"Columns have no valid values: {empty_columns}")

    missing_values = int(df[WEATHER_COLUMNS].isna().sum().sum())

    # Fill missing values using earlier rows.
    df[WEATHER_COLUMNS] = df[WEATHER_COLUMNS].ffill()

    # Remove rows that still have missing values.
    rows_before = len(df)
    df = df.dropna(subset=WEATHER_COLUMNS).copy()
    dropped_rows = rows_before - len(df)

    # Apply basic weather limits.
    df["temperature_2m"] = df["temperature_2m"].clip(-10, 50)
    df["relative_humidity_2m"] = df["relative_humidity_2m"].clip(0, 100)
    df["precipitation"] = df["precipitation"].clip(lower=0)
    df["wind_speed_10m"] = df["wind_speed_10m"].clip(lower=0)

    if df.empty:
        raise ValueError("No rows remain after cleaning.")

    # Find the project root.
    project_root = Path(__file__).resolve().parents[1]

    # Create the output folder.
    output_dir = project_root / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use the input filename for the output.
    output_name = f"weather_clean_{input_file.stem}.csv"
    if input_file.stem.startswith("weather_raw"):
        suffix = input_file.stem[len("weather_raw"):]
        output_name = f"weather_clean{suffix}.csv"

    output_file = output_dir / output_name

    # Save without overwriting an existing file.
    with output_file.open("x", encoding="utf-8", newline="") as file:
        df.to_csv(file, index=False)

    print("\n" + "=" * 50)
    print("PREPROCESSING COMPLETED SUCCESSFULLY")
    print("=" * 50)
    print(f"Input file           : {input_file}")
    print(f"Output file          : {output_file}")
    print(f"Input rows           : {input_rows}")
    print(f"Duplicate rows       : {duplicate_rows}")
    print(f"Missing values found : {missing_values}")
    print(f"Rows removed         : {dropped_rows}")
    print(f"Output rows          : {len(df)}")
    print(f"Start time           : {df['time'].min()}")
    print(f"End time             : {df['time'].max()}")
    print("\nPreview of the first five rows:")
    print(df.head())
    print("=" * 50 + "\n")

    return output_file


def main() -> None:
    """Read the input path and run preprocessing."""
    parser = argparse.ArgumentParser(
        description="Clean a raw weather CSV file."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to the raw CSV file.",
    )
    args = parser.parse_args()

    # Read relative paths from the project root.
    project_root = Path(__file__).resolve().parents[1]
    input_file = args.input

    if not input_file.is_absolute():
        input_file = project_root / input_file

    try:
        preprocess_data(input_file)
    except FileExistsError:
        print(
            "[ERROR] The output file already exists. "
            "Use a new raw file or rename the existing output file.",
            file=sys.stderr,
        )
        sys.exit(1)
    except (OSError, ValueError, KeyError) as exc:
        print(f"[ERROR] Preprocessing failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()