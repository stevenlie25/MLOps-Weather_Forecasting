# MLOps - Weather Forecasting

An end-to-end MLOps system for automated and standardized weather prediction and modeling.

---

## Project Overview
This project aims to build a reproducible Machine Learning pipeline for predicting key weather metrics (such as temperature and precipitation). The pipeline is designed around modern MLOps principles to support Continuous Training, automated testing, and seamless continuous integration.

---

## Project Structure
This repository follows the industry-standard *Cookiecutter Data Science* project layout:

```text
MLOps-Weather_Forecasting/
├── .devcontainer/      # GitHub Codespaces environment configuration
├── config/             # Pipeline & model parameter configurations
├── data/               # Data storage (raw, processed, external)
├── models/             # Saved model artifacts & experiments
├── notebooks/          # Jupyter Notebooks for EDA & early prototyping
├── src/                # Core source code (data processing, features, modeling)
├── .gitignore          # Git ignore configuration
├── LICENSE             # Project license (MIT)
├── README.md           # Primary project documentation
└── requirements.txt    # Python package dependencies
```

---

## Development Status

Implemented:
- Weather data ingestion from Open-Meteo for Malang, Indonesia.
- Timestamped raw CSV files for each ingestion run.
- Data preprocessing with column validation, timezone handling, duplicate removal, missing value handling, and basic weather limits.

Automatic scheduling, feature engineering, model training, and deployment are planned.

## Getting Started

Run all commands from the project root.

### Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### Collect Weather Data

```bash
python src/ingest_data.py
```

The script collects hourly temperature, humidity, surface pressure, wind speed, and precipitation data. Each run saves a new CSV file in `data/raw/`.

### Preprocess Weather Data

```bash
python src/preprocess.py --input data/raw/weather_raw_20261101_140955_288093.csv
```

Replace the input path with your raw CSV file.

The script validates columns, sets timestamps to Asia/Jakarta, removes duplicate timestamps, sorts rows, fills missing values using earlier data, and applies basic weather limits.

Clean files are saved in `data/processed/`. Raw files remain unchanged, and existing output files are not overwritten.

Both scripts currently run manually.
