# MLOps - Weather Forecasting

**Sistem Prediksi Suhu dan Curah Hujan Berbasis Open-Meteo dengan Continuous Training**

A reproducible weather prediction project for Malang, Indonesia. The project is being developed toward periodic data updates, automated retraining, experiment tracking, and model versioning.

## Development Status

**Implemented:** API ingestion, cumulative dataset updates, preprocessing, and dataset versioning with DVC.

**Planned:** historical training data, feature engineering, MLflow experiments, Model Registry, inference, and Continuous Training.

All current processes run manually.

## Project Structure

```text
├── .devcontainer/      # Codespaces configuration
├── .dvc/               # DVC configuration and local cache
├── config/             # Pipeline configuration
├── data/               # Raw, processed, and external data
├── models/             # Model artifacts
├── notebooks/          # Exploration
├── src/                # Data processing and modeling
├── README.md
└── requirements.txt
```

## Getting Started

Run commands from the project root:

```bash
python -m pip install -r requirements.txt
```

### Collect Data

```bash
python src/ingest_data.py
```

Collects hourly temperature, humidity, surface pressure, wind speed, and precipitation. Each run saves a timestamped CSV in `data/raw/` and reports connection or HTTP errors.

The current configuration returns two previous days plus seven forecast days including today: **216 hourly rows**, with times in Asia/Jakarta.

The source is Open-Meteo Forecast API model output, not direct station observations.

### Update the Cumulative Dataset

```bash
python src/update_dataset.py --input data/raw/weather_raw_TIMESTAMP.csv
```

Replace `weather_raw_TIMESTAMP.csv` with the actual filename produced
by ingestion. The script appends only new timestamps; reusing the same
input leaves the dataset unchanged.

### Preprocess Data

```bash
python src/preprocess.py --input data/raw/weather_dataset.csv
```

Validates columns, handles timezone and missing values, removes duplicate timestamps, sorts records, and applies basic weather limits. Results are saved in `data/processed/` without modifying the raw input.

## Dataset Versioning

DVC tracks `data/raw/weather_dataset.csv`. Git tracks its `.dvc` pointer and configuration; raw CSV files are ignored by Git.

DVC was initialized once using `dvc init`. The initial dataset was copied from a timestamped ingestion snapshot and recorded using `dvc add`.

The initial version update was performed through a manual Pandas merge.
This process is now available in `src/update_dataset.py`, which preserves
existing records and appends only new timestamps. Existing forecast
values are not replaced.

| Version | Rows | Time range (Asia/Jakarta) | Hash prefix |
| --- | ---: | --- | --- |
| `data-v1` | 216 | 30 Oct–7 Nov 2026 | `4da59cf7` |
| `data-v2` | 240 | 30 Oct–8 Nov 2026 | `25b891fb` |

### Record an Updated Dataset

After merging new records into the cumulative CSV:

```bash
dvc status
dvc add data/raw/weather_dataset.csv
dvc diff --show-hash
git add data/raw/weather_dataset.csv.dvc
git commit -m "data: update cumulative weather dataset"
```

### Compare Versions

```bash
dvc diff --show-hash data-v1 data-v2
```

Recorded result:

```text
Modified:
    4da59cf7..25b891fb  data/raw/weather_dataset.csv

files summary: 1 modified
```

The merge separately confirmed **24 additional rows**. DVC compares file metadata and hashes rather than individual CSV rows.

### Reproducibility and Storage

Versioning identifies the exact dataset used for an experiment. Future MLflow runs will link the dataset identity and Git commit with training parameters, metrics, and model artifacts.

**No DVC remote is configured yet.** Data versions remain in the Codespace's local DVC cache. Preserve this cache: GitHub contains pointers, not dataset contents. Persistent storage and `dvc push` are needed before another environment can retrieve the data with `dvc pull`.

## Next Steps

Prepare historical training data, use chronological evaluation, track experiments with MLflow, and register selected models. Continuous Training will add scheduled data updates, retraining, performance monitoring, and evaluation before model promotion.

The current forecast dataset demonstrates versioning; it is not yet the final training dataset.

## References

- [Open-Meteo](https://open-meteo.com/en/docs)
- [DVC](https://dvc.org/doc)

## License

MIT License. See `LICENSE`.