"""Evaluate selected weather models on the held-out test set."""

import hashlib
import json
import subprocess
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed/weather_modeling_era5_1h.csv"
SELECTION = ROOT / "config/selected_models.json"


def main():
    status = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True
    ).strip()
    if status:
        raise ValueError("Commit project changes before evaluation.")

    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    selection = json.loads(SELECTION.read_text(encoding="utf-8"))
    pointer_path = DATA.with_suffix(".csv.dvc")
    pointer = yaml.safe_load(pointer_path.read_text(encoding="utf-8"))
    digest = hashlib.md5(DATA.read_bytes()).hexdigest()

    if not (
        digest == selection["dataset_dvc_md5"]
        == pointer["outs"][0]["md5"]
    ):
        raise ValueError("Dataset identity does not match selection.")

    df = pd.read_csv(DATA)
    test = df.loc[df["split"] == "test"].copy()
    times = pd.to_datetime(test["target_time"], utc=True)
    start = pd.Timestamp("2026-01-01", tz="Asia/Jakarta")
    end = pd.Timestamp("2026-10-30 23:00", tz="Asia/Jakarta")

    if len(test) != 7272 or not times.between(start, end).all():
        raise ValueError("Unexpected test period or row count.")

    mlflow.set_tracking_uri(
        f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"
    )
    client = mlflow.MlflowClient()

    for target, selected in selection["models"].items():
        source_run = client.get_run(selected["run_id"])
        if source_run.info.status != "FINISHED":
            raise ValueError("Selected training run is not finished.")
        if source_run.data.tags["dataset.dvc_md5"] != digest:
            raise ValueError("Training run dataset does not match.")

        schema_path = mlflow.artifacts.download_artifacts(
            artifact_uri=(
                f"runs:/{selected['run_id']}/feature_schema.json"
            )
        )
        schema = json.loads(Path(schema_path).read_text())
        model = mlflow.pyfunc.load_model(selected["model_uri"])

        predictions = np.asarray(
            model.predict(test[schema["features"]])
        ).reshape(-1)
        actual = test[schema["target"]].to_numpy()
        weather_column = schema["target"].removeprefix("target_")
        baseline = test[f"{weather_column}_lag_0h"].to_numpy()

        if not all(
            np.isfinite(values).all()
            for values in [predictions, actual, baseline]
        ):
            raise ValueError("Test values must be finite.")

        scores = {}
        for prefix, predicted in [
            ("test", predictions),
            ("baseline_test", baseline),
        ]:
            scores[f"{prefix}_mae"] = float(
                mean_absolute_error(actual, predicted)
            )
            scores[f"{prefix}_rmse"] = float(
                np.sqrt(mean_squared_error(actual, predicted))
            )

        with mlflow.start_run(
            experiment_id=source_run.info.experiment_id,
            run_name=f"test-selected-{target}",
        ) as run:
            mlflow.set_tags({
                "evaluation_role": "held_out_test",
                "selected_training_run_id": selected["run_id"],
                "mlflow.source.git.commit": commit,
                "dataset.dvc_md5": digest,
                "target": target,
            })
            mlflow.log_params({
                "source_model_uri": selected["model_uri"],
                "selection_metric": selection["selection_metric"],
                "test_rows": len(test),
                "test_target_start": times.min().isoformat(),
                "test_target_end": times.max().isoformat(),
            })
            mlflow.log_metrics(scores)
            mlflow.log_artifact(str(SELECTION), "lineage")
            mlflow.log_artifact(str(pointer_path), "lineage")

            comparison = pd.DataFrame({
                "target_time": test["target_time"].to_numpy(),
                "actual": actual,
                "prediction": predictions,
                "persistence_baseline": baseline,
            })
            mlflow.log_text(
                comparison.to_csv(index=False),
                "evaluation/test_predictions.csv",
            )
            mlflow.log_dict(scores, "evaluation/test_metrics.json")

            print(f"\nTarget: {target}")
            print(f"Evaluation run ID: {run.info.run_id}")
            for name, value in scores.items():
                print(f"{name}: {value:.6f}")


if __name__ == "__main__":
    main()