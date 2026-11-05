# Experiment Summary

## Evaluation Setup

- Source: Open-Meteo Historical Weather API, ERA5 reanalysis for Malang.
- Historical period: 1 January 2023–30 October 2026.
- Dataset: 33,576 historical rows; 33,551 modeling rows; 29 features.
- Targets: temperature and precipitation one hour ahead.
- Chronological splits by target time: training 2023–2024 (17,519 rows),
  validation 2025 (8,760 rows), test January–30 October 2026 (7,272 rows).
- Modeling dataset DVC MD5: `02242626073bf45940b01fba97815038`.
- Training commit: `5942216b87efe04835d2fa8c9e07a831bb655192`.

All models used RandomForestRegressor with random_state=42 and n_jobs=2.
Selection used the lowest validation MAE per target, with RMSE as a
secondary metric. Test results did not influence selection.

## Validation Results

| Target | Trees | Max depth | Min samples leaf | MAE | RMSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Temperature | 100 | 10 | 2 | 0.438655 | 0.615012 |
| Temperature | 200 | 20 | 2 | 0.439841 | 0.615106 |
| Temperature | 200 | 15 | 5 | 0.439499 | 0.615490 |
| Precipitation | 100 | 10 | 2 | 0.299320 | 0.876072 |
| Precipitation | 200 | 20 | 2 | 0.302032 | 0.872170 |
| Precipitation | 200 | 15 | 5 | 0.293438 | 0.859629 |

Temperature metrics are in °C; precipitation metrics are in mm.

## Selected Models

- **Temperature:** 100 trees, max_depth=10, min_samples_leaf=2.
  Run: `2070440bb07a4bcf970f649a86db54a7`.
  This configuration achieved the lowest validation MAE and RMSE.
  Differences were small, and the selected model used fewer trees.
- **Precipitation:** 200 trees, max_depth=15, min_samples_leaf=5.
  Run: `4e2de5d9de9c4308b0eb426227ede66d`.
  This configuration achieved the lowest validation MAE and RMSE.

Both saved models were loaded through mlflow.pyfunc and reproduced
their logged validation MAE. Selection metadata is recorded in
`config/selected_models.json`.

## Held-Out Test Results

Selected models were evaluated without retraining. The persistence
baseline predicts the next hour using the current value.

| Target | Model MAE | Baseline MAE | Model RMSE | Baseline RMSE |
| --- | ---: | ---: | ---: | ---: |
| Temperature | 0.442370 | 0.787445 | 0.599829 | 1.054730 |
| Precipitation | 0.201436 | 0.217822 | 0.666297 | 0.863682 |

Evaluation runs:

- Temperature: `f3f86de3575c465c8568ad50380fc24e`.
- Precipitation: `dcead395a60c495a983b5f7e8e0c3d3a`.

## Reproducibility and Limitations

MLflow records parameters, metrics, model artifacts, dataset hashes,
and Git commits. The accompanying CSV exports contain three completed
training runs and one test evaluation run per target.

Results describe historical ERA5 prediction at one location, rather
than operational forecasting against station observations. Delayed
reanalysis availability requires a separate operational feature design.
Heavy-rain performance was not evaluated separately.

DVC has no remote configured. Dataset contents, the MLflow database,
and model artifacts remain local to the Codespace. CSV exports do not
replace backups of these resources.

Registration of the selected weather models and Continuous Training
remain planned. Stage transitions were verified separately with a
temporary compatibility model; MLflow stages are deprecated.