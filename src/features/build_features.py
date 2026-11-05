"""Build hourly features and chronological splits from weather data."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


WEATHER_COLUMNS = [
    "temperature_2m",
    "relative_humidity_2m",
    "surface_pressure",
    "wind_speed_10m",
    "precipitation",
]
LAGS = [0, 1, 2, 3, 24]
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def build_features(input_path, output_path):
    """Build features using weather available through reference time t."""
    metadata_path = output_path.with_suffix(".metadata.json")
    if output_path.exists() or metadata_path.exists():
        raise FileExistsError("Output or metadata already exists.")

    source_bytes = input_path.read_bytes()
    raw = pd.read_csv(input_path)
    raw = raw[["time", *WEATHER_COLUMNS]].copy()
    times = pd.to_datetime(raw.pop("time"), utc=True)
    times = pd.DatetimeIndex(times).tz_convert("Asia/Jakarta")

    if times.hasnans or not times.is_unique:
        raise ValueError("Missing or duplicated timestamps.")
    if not times.is_monotonic_increasing:
        raise ValueError("Timestamps must be sorted.")
    if not times.to_series().diff().dropna().eq(
        pd.Timedelta(hours=1)
    ).all():
        raise ValueError("Input must have continuous hourly timestamps.")

    raw.index = times
    for column in WEATHER_COLUMNS:
        raw[column] = pd.to_numeric(raw[column], errors="raise")
    if not np.isfinite(raw.to_numpy()).all():
        raise ValueError("Weather values must be finite and complete.")

    features = pd.DataFrame(index=times)
    for lag in LAGS:
        for column in WEATHER_COLUMNS:
            features[f"{column}_lag_{lag}h"] = raw[column].shift(lag)

    features["target_temperature_2m"] = raw["temperature_2m"].shift(-1)
    features["target_precipitation"] = raw["precipitation"].shift(-1)
    features = features.dropna().copy()

    target_times = features.index + pd.Timedelta(hours=1)
    features.insert(0, "reference_time", features.index)
    features.insert(1, "target_time", target_times)

    features["target_hour_sin"] = np.sin(
        2 * np.pi * target_times.hour / 24
    )
    features["target_hour_cos"] = np.cos(
        2 * np.pi * target_times.hour / 24
    )
    features["target_month_sin"] = np.sin(
        2 * np.pi * (target_times.month - 1) / 12
    )
    features["target_month_cos"] = np.cos(
        2 * np.pi * (target_times.month - 1) / 12
    )

    train_end = pd.Timestamp("2025-01-01", tz="Asia/Jakarta")
    validation_end = pd.Timestamp("2026-01-01", tz="Asia/Jakarta")
    features["split"] = np.where(
        target_times < train_end,
        "train",
        np.where(target_times < validation_end, "validation", "test"),
    )

    if set(features["split"]) != {"train", "validation", "test"}:
        raise ValueError("All three splits must contain data.")

    summary = {}
    for split, group in features.groupby("split", sort=False):
        summary[split] = {
            "rows": len(group),
            "target_start": group["target_time"].min().isoformat(),
            "target_end": group["target_time"].max().isoformat(),
        }

    csv_bytes = features.to_csv(index=False).encode("utf-8")
    metadata = {
        "source_file": input_path.name,
        "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "output_sha256": hashlib.sha256(csv_bytes).hexdigest(),
        "timezone": "Asia/Jakarta",
        "horizon_hours": 1,
        "lags_hours": LAGS,
        "feature_count": 29,
        "targets": ["target_temperature_2m", "target_precipitation"],
        "split_basis": "target_time",
        "validation_start": train_end.isoformat(),
        "test_start": validation_end.isoformat(),
        "source_rows": len(raw),
        "output_rows": len(features),
        "splits": summary,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("x", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)
        file.write("\n")
    with output_path.open("xb") as file:
        file.write(csv_bytes)

    print(f"Saved: {output_path}")
    print(f"Metadata: {metadata_path}")
    print("Features: 29")
    print(f"Total rows: {len(features)}")
    for split, details in summary.items():
        print(
            f"{split}: {details['rows']} rows | "
            f"{details['target_start']} to {details['target_end']}"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    try:
        input_path = args.input
        output_path = args.output
        if not input_path.is_absolute():
            input_path = PROJECT_ROOT / input_path
        if not output_path.is_absolute():
            output_path = PROJECT_ROOT / output_path
        if input_path.resolve() == output_path.resolve():
            raise ValueError("Input and output must differ.")
        build_features(input_path, output_path)
    except (OSError, ValueError, KeyError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()