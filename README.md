# MLOps - Weather Forecasting

**Sistem Prediksi Suhu dan Curah Hujan Berbasis Open-Meteo dengan Continuous Training**

A reproducible weather prediction project for Malang, Indonesia. The project is being developed toward periodic data updates, automated retraining, experiment tracking, and model versioning.

## Development Status

**Implemented:** forecast ingestion, cumulative dataset updates, preprocessing, DVC versioning, historical ERA5 ingestion, feature engineering, MLflow experiment tracking, model selection, held-out test evaluation, Model Registry versioning and stage transitions, DVC-tracked model metadata, and CSV inference through Production model URIs.

**Planned:** GitHub Actions automation for testing, training, performance threshold validation, and registration of passing candidates to Staging. Persistent DVC and MLflow storage is also planned.

All current workflows run manually.

## Project Structure

```text
├── .devcontainer/      # Codespaces configuration
├── .dvc/               # DVC configuration and local cache
├── config/             # Model selection configuration
├── data/               # Raw and processed datasets
├── examples/           # Prepared inference input
├── metadata/           # DVC-tracked model lineage
├── mlartifacts/        # Local MLflow artifacts; ignored by Git
├── models/             # Reserved model directory
├── notebooks/          # Exploration
├── reports/            # Experiment summaries and exported runs
├── src/                # Data processing, modeling, and inference
├── mlflow.db           # Local MLflow backend; ignored by Git
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

The current configuration returns two previous days plus seven forecast days including the collection day: **216 hourly rows**, with times in Asia/Jakarta.

The source is Open-Meteo Forecast API model output, not direct station observations.

### Update the Cumulative Dataset

```bash
python src/update_dataset.py --input data/raw/weather_raw_TIMESTAMP.csv
```

Replace `weather_raw_TIMESTAMP.csv` with the actual filename produced by ingestion. The script preserves existing records and appends only new timestamps; reusing the same input leaves the dataset unchanged.

Existing forecast values are not refreshed. This dataset contains forecast snapshots rather than a continuous series of actual observations.

### Preprocess Data

```bash
python src/preprocess.py --input data/raw/weather_dataset.csv
```

Validates columns, handles timezone and missing values, removes duplicate timestamps, sorts records, and applies basic weather limits. Results are saved in `data/processed/` without modifying the raw input.

## Dataset Versioning

DVC tracks the cumulative forecast dataset, historical ERA5 dataset, and modeling dataset. Git tracks their `.dvc` pointers rather than the dataset contents.

The original forecast dataset versions remain available:

| Version | Rows | Time range (Asia/Jakarta) | Hash prefix |
| --- | ---: | --- | --- |
| `data-v1` | 216 | 30 Oct–7 Nov 2026 | `4da59cf7` |
| `data-v2` | 240 | 30 Oct–8 Nov 2026 | `25b891fb` |

These tags identify forecast dataset versions, not Registry model versions. The forecast dataset remains separate from the historical ERA5 modeling data.

### Record an Updated Dataset

After updating the cumulative CSV:

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

MLflow runs record the dataset identity and training Git commit alongside parameters, metrics, and model artifacts.

**No DVC remote is configured yet.** Dataset and model metadata contents depend on the Codespace's local DVC storage. Preserve the cache and working files: GitHub contains pointers, not their contents.

Persistent storage and uploaded DVC assets are required before another environment can restore these files using `dvc pull`. The MLflow database and artifacts also remain local; cloning the repository alone does not restore the complete experiment environment.

## Experiment Tracking and Model Evaluation

Historical ERA5 reanalysis for Malang covers 1 January 2023–30 October 2026 (33,576 hourly records). The DVC-versioned modeling dataset contains 33,551 rows and 29 features for predicting temperature and precipitation one hour ahead.

Data is split chronologically by target time: training 2023–2024, validation 2025, and test January–30 October 2026. Three Random Forest configurations were initially compared per target using validation MAE.

The following results describe the original selected models, registered as v1:

| Target | Trees / depth / min leaf | Validation MAE | Test MAE | Test RMSE |
| --- | --- | ---: | ---: | ---: |
| Temperature (°C) | 100 / 10 / 2 | 0.438655 | 0.442370 | 0.599829 |
| Precipitation (mm) | 200 / 15 / 5 | 0.293438 | 0.201436 | 0.666297 |

Both original models outperformed persistence baselines on the test set. Test evaluation was performed without retraining and was not used to revise model selection.

Original selections are stored in `config/selected_models.json`, with summaries in `reports/experiments/`. The precipitation test metrics above apply to v1, not the later Production v2.

```bash
python src/train.py --target temperature \
  --n-estimators 100 --max-depth 10 --min-samples-leaf 2

python src/evaluate.py

mlflow ui --backend-store-uri sqlite:///mlflow.db \
  --host 0.0.0.0 --port 5000
```

Run from the repository root. Training requires a clean Git working tree and local DVC data; evaluation also requires the original selected MLflow artifacts. Use Private port forwarding in Codespaces.

Results describe historical ERA5 prediction, not operational forecasting against station observations.

## Model Registry and Inference

Two versions are registered for each target under `malang-weather-temperature-1h` and `malang-weather-precipitation-1h`.

Version 2 increases the number of trees from 100 to 150 for temperature and from 200 to 250 for precipitation. Other model settings remain unchanged.

| Target | v1 validation MAE | v2 validation MAE | Production version |
| --- | ---: | ---: | --- |
| Temperature (°C) | 0.438655 | 0.438838 | v1 |
| Precipitation (mm) | 0.293438 | 0.293313 | v2 |

Production versions are selected by the lowest validation MAE. Differences are small; newer versions are not automatically better. Test data was not used for this comparison.

Selected versions followed None → Staging → Production. Stages are deprecated in MLflow but are retained for this lifecycle demonstration.

### Model Metadata

`metadata/production_models.yaml` records active versions, inference URIs, run IDs, parameters, validation metrics, training commits, and dataset hashes. It is tracked through `metadata/production_models.yaml.dvc`.

Lineage follows **Registry version → MLflow run → training Git commit + DVC dataset hash**.

`config/selected_models.json` retains the original experiment selections, while the metadata YAML records the active Registry selections.

### Run Inference

```bash
python src/predict.py --target temperature --input examples/inference_input.csv
python src/predict.py --target precipitation --input examples/inference_input.csv
```

The script loads each model through `models:/<registry-name>/Production`, checks its version against the metadata, and validates required features, numeric values, and prediction outputs.

Input CSV files must contain 29 prepared features with complete, finite values. The included example contains three validation rows and successfully produces predictions for both targets. It is a historical smoke test, not live weather input.

Inference requires the local MLflow database, model artifacts, and metadata YAML. Production stage identifies the selected model version, not a deployed forecasting service.

## Next Steps

Make DVC datasets and MLflow storage accessible to the automation environment. Implement `.github/workflows/mlops-automation.yaml` with triggers on pushes to main or pull requests:

1. Verify code with pytest.
2. Train models using the current DVC-managed dataset.
3. Evaluate performance against predefined project thresholds.
4. Register passing candidates in MLflow Model Registry with stage Staging.

Demonstrate the pipeline through a code change and retain successful execution logs. Production promotion remains a separate decision.

Operational weather input integration, inference service deployment, scheduled ingestion, and monitoring remain future work.

## References

- [Open-Meteo Forecast API](https://open-meteo.com/en/docs)
- [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api)
- [MLflow Documentation](https://mlflow.org/docs/latest/)
- [DVC Documentation](https://dvc.org/doc)
- [scikit-learn Documentation](https://scikit-learn.org/stable/)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [pytest Documentation](https://docs.pytest.org/en/stable/)

## License

MIT License. See `LICENSE`.