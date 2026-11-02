"""Append new weather timestamps to the cumulative dataset."""

import argparse
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "data/raw/weather_dataset.csv"

EXPECTED_COLUMNS = [
    "time",
    "temperature_2m",
    "relative_humidity_2m",
    "surface_pressure",
    "wind_speed_10m",
    "precipitation",
]


def read_dataset(path: Path) -> pd.DataFrame:
    """Read and validate a raw weather dataset."""
    df = pd.read_csv(path)

    if set(df.columns) != set(EXPECTED_COLUMNS):
        raise ValueError(f"Unexpected columns in {path}")

    df = df[EXPECTED_COLUMNS].copy()
    df["time"] = pd.to_datetime(df["time"], errors="raise")

    if df["time"].isna().any():
        raise ValueError(f"Missing timestamps in {path}")

    if df["time"].duplicated().any():
        raise ValueError(f"Duplicate timestamps in {path}")

    return df


def update_dataset(source: Path, target: Path) -> None:
    """Keep existing records and append previously unseen timestamps."""
    if source.resolve() == target.resolve():
        raise ValueError("Input and output paths must be different.")

    incoming = read_dataset(source)
    existing = (
        read_dataset(target)
        if target.exists()
        else incoming.iloc[:0].copy()
    )

    additional = incoming.loc[
        ~incoming["time"].isin(existing["time"])
    ]

    print(f"Input file: {source}")
    print(f"Dataset: {target}")
    print(f"Rows before: {len(existing)}")
    print(f"New timestamps: {len(additional)}")

    if additional.empty:
        print("No new timestamps found. Dataset remains unchanged.")
        return

    updated = (
        pd.concat([existing, additional], ignore_index=True)
        .sort_values("time")
        .reset_index(drop=True)
    )

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f"{target.name}.tmp")

    try:
        updated.to_csv(temporary, index=False)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)

    print(f"Rows after: {len(updated)}")
    print(
        f"Time range: {updated['time'].min()} "
        f"to {updated['time'].max()}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Timestamped CSV produced by ingest_data.py.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Cumulative dataset path.",
    )
    args = parser.parse_args()

    try:
        update_dataset(args.input, args.output)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        print(f"[ERROR] Dataset update failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()