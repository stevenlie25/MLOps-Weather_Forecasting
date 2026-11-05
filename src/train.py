"""Train one weather regression model and track it with MLflow."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed/weather_modeling_era5_1h.csv"
POINTER = DATA.with_suffix(".csv.dvc")
METADATA = DATA.with_suffix(".metadata.json")
TARGETS = {
    "temperature": ("target_temperature_2m", "temperature_2m_lag_0h"),
    "precipitation": ("target_precipitation", "precipitation_lag_0h"),
}


def git_output(*args):
    return subprocess.check_output(
        ["git", *args], cwd=ROOT, text=True
    ).strip()


def metrics(actual, predicted):
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target", required=True, choices=TARGETS
    )
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=10)
    parser.add_argument("--min-samples-leaf", type=int, default=2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    if min(
        args.n_estimators, args.max_depth, args.min_samples_leaf
    ) < 1:
        parser.error("Model parameters must be positive integers.")

    if git_output("status", "--porcelain", "--untracked-files=normal"):
        raise ValueError("Commit project changes before running training.")

    commit = git_output("rev-parse", "HEAD")
    metadata = json.loads(METADATA.read_text(encoding="utf-8"))
    pointer = yaml.safe_load(POINTER.read_text(encoding="utf-8"))
    dataset_bytes = DATA.read_bytes()
    dataset_md5 = hashlib.md5(dataset_bytes).hexdigest()
    dataset_sha256 = hashlib.sha256(dataset_bytes).hexdigest()

    if dataset_md5 != pointer["outs"][0]["md5"]:
        raise ValueError("Dataset does not match its DVC pointer.")
    if dataset_sha256 != metadata["output_sha256"]:
        raise ValueError("Dataset does not match feature metadata.")

    df = pd.read_csv(DATA)
    target_column, baseline_column = TARGETS[args.target]
    excluded = {
        "reference_time", "target_time", "split",
        "target_temperature_2m", "target_precipitation",
    }
    feature_columns = [
        column for column in df.columns if column not in excluded
    ]
    if len(feature_columns) != metadata["feature_count"]:
        raise ValueError("Unexpected feature count.")

    train = df.loc[df["split"] == "train"].copy()
    validation = df.loc[df["split"] == "validation"].copy()
    if train.empty or validation.empty:
        raise ValueError("Training and validation data must not be empty.")

    train_times = pd.to_datetime(train["target_time"], utc=True)
    validation_times = pd.to_datetime(validation["target_time"], utc=True)
    if train_times.max() >= validation_times.min():
        raise ValueError("Training must precede validation.")

    X_train = train[feature_columns]
    X_validation = validation[feature_columns]
    y_train = train[target_column]
    y_validation = validation[target_column]

    if not all(
        np.isfinite(values.to_numpy()).all()
        for values in [X_train, X_validation, y_train, y_validation]
    ):
        raise ValueError("Training or validation contains invalid values.")

    model_params = {
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "min_samples_leaf": args.min_samples_leaf,
        "random_state": args.random_state,
        "n_jobs": 2,
    }

    tracking_uri = f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.MlflowClient()
    experiment_name = f"weather-{args.target}-1h"
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = client.create_experiment(
            experiment_name,
            artifact_location=(ROOT / "mlartifacts").as_uri(),
        )
    else:
        experiment_id = experiment.experiment_id

    run_name = (
        f"rf-trees{args.n_estimators}"
        f"-depth{args.max_depth}-leaf{args.min_samples_leaf}"
    )
    with mlflow.start_run(
        experiment_id=experiment_id, run_name=run_name
    ) as run:
        mlflow.set_tags({
            "mlflow.source.git.commit": commit,
            "dataset.dvc_md5": dataset_md5,
            "dataset.sha256": dataset_sha256,
            "dataset.dvc_pointer": str(POINTER.relative_to(ROOT)),
            "data_source": "ERA5 reanalysis",
            "target": args.target,
            "evaluation_role": "validation_for_model_selection",
        })
        mlflow.log_params({
            **model_params,
            "model_type": "RandomForestRegressor",
            "horizon_hours": 1,
            "feature_count": len(feature_columns),
            "train_rows": len(train),
            "validation_rows": len(validation),
            "train_target_start": train_times.min().isoformat(),
            "train_target_end": train_times.max().isoformat(),
            "validation_target_start": validation_times.min().isoformat(),
            "validation_target_end": validation_times.max().isoformat(),
        })

        model = RandomForestRegressor(**model_params)
        model.fit(X_train, y_train)
        predictions = model.predict(X_validation)
        baseline = validation[baseline_column].to_numpy()

        scores = {}
        for prefix, predicted in [
            ("validation", predictions),
            ("baseline_validation", baseline),
        ]:
            scores.update({
                f"{prefix}_{name}": value
                for name, value in metrics(y_validation, predicted).items()
            })
        mlflow.log_metrics(scores)

        mlflow.log_artifact(str(POINTER), artifact_path="lineage")
        mlflow.log_artifact(str(METADATA), artifact_path="lineage")
        mlflow.log_artifact(str(ROOT / "requirements.txt"), "environment")
        mlflow.log_dict(
            {"features": feature_columns, "target": target_column},
            "feature_schema.json",
        )
        mlflow.log_text(
            subprocess.check_output(
                ["python", "-m", "pip", "freeze"], text=True
            ),
            "environment/pip_freeze.txt",
        )

        comparison = pd.DataFrame({
            "target_time": validation["target_time"].to_numpy(),
            "actual": y_validation.to_numpy(),
            "prediction": predictions,
            "persistence_baseline": baseline,
        })
        mlflow.log_text(
            comparison.to_csv(index=False),
            "evaluation/validation_predictions.csv",
        )
        mlflow.sklearn.log_model(
            sk_model=model,
            name="weather_model",
            serialization_format="skops",
            skops_trusted_types=["sklearn.tree._tree.Tree"],
            signature=infer_signature(
                X_train.head(5), model.predict(X_train.head(5))
            ),
            input_example=X_train.head(3),
        )

        print(f"Run ID: {run.info.run_id}")
        print(f"Target: {args.target}")
        print(f"Git commit: {commit}")
        print(f"DVC MD5: {dataset_md5}")
        for name, value in scores.items():
            print(f"{name}: {value:.6f}")


if __name__ == "__main__":
    main()