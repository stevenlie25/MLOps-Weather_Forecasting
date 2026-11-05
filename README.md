# MLOps - Weather Forecasting

**Sistem Prediksi Suhu dan Curah Hujan Berbasis Open-Meteo dengan Continuous Training**

A reproducible weather prediction project for Malang, Indonesia. The project is being developed toward periodic data updates, automated retraining, experiment tracking, and model versioning.

## Development Status

**Implemented:** forecast ingestion, cumulative dataset updates, preprocessing, DVC versioning, historical ERA5 ingestion, feature engineering, MLflow experiment tracking, model selection, and held-out test evaluation.

**Planned:** registration of selected weather models, persistent DVC and MLflow storage, operational inference, and Continuous Training.

All current workflows run manually.

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

Versioning identifies the exact dataset used for an experiment. MLflow runs record the dataset identity and Git commit alongside training parameters, metrics, and model artifacts.

**No DVC remote is configured yet.** Data versions remain in the Codespace's local DVC cache. Preserve this cache: GitHub contains pointers, not dataset contents. Persistent storage and `dvc push` are needed before another environment can retrieve the data with `dvc pull`.

## Experiment Tracking and Model Evaluation

Historical ERA5 reanalysis for Malang covers 1 January 2023–30 October 2026 (33,576 hourly records). The DVC-versioned modeling dataset contains 33,551 rows and 29 features for predicting temperature and precipitation one hour ahead.

Data is split chronologically by target time: training 2023–2024, validation 2025, and test January–30 October 2026. Three Random Forest configurations were compared per target using validation MAE.

| Target | Trees / depth / min leaf | Validation MAE | Test MAE | Test RMSE |
| --- | --- | ---: | ---: | ---: |
| Temperature (°C) | 100 / 10 / 2 | 0.438655 | 0.442370 | 0.599829 |
| Precipitation (mm) | 200 / 15 / 5 | 0.293438 | 0.201436 | 0.666297 |

Both selected models outperformed persistence baselines on the test set. Selection details are stored in `config/selected_models.json`. MLflow records parameters, metrics, model artifacts, Git commits, and DVC dataset hashes.

```bash
python src/train.py --target temperature \
  --n-estimators 100 --max-depth 10 --min-samples-leaf 2

python src/evaluate.py

mlflow ui --backend-store-uri sqlite:///mlflow.db \
  --host 0.0.0.0 --port 5000
```

Run from the repository root with a clean Git working tree. Use private port forwarding in Codespaces. Training and evaluation require the local DVC datasets; evaluation also requires the selected MLflow artifacts.

Results describe historical ERA5 prediction, not operational forecasting against station observations. MLflow storage remains local. Registration of the selected weather models and Continuous Training remain planned.

## Next Steps

Register the selected models, configure persistent storage for DVC and MLflow, and prepare operational inference. Continuous Training will introduce scheduled data updates, retraining, monitoring, and evaluation before model promotion.

## References

- [Open-Meteo](https://open-meteo.com/en/docs)
- [DVC](https://dvc.org/doc)

## License

MIT License. See `LICENSE`.