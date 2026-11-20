"""Run inference using a Production model and a CSV of prepared features."""

import argparse
from pathlib import Path

import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target", required=True,
        choices=["temperature", "precipitation"],
    )
    parser.add_argument("--input", required=True, type=Path)
    args = parser.parse_args()

    db = ROOT / "mlflow.db"
    if not db.is_file():
        raise FileNotFoundError(db)

    metadata = yaml.safe_load(
        (ROOT / "metadata/production_models.yaml").read_text()
    )
    selected = metadata["models"][args.target]
    name = selected["registered_name"]

    mlflow.set_tracking_uri(f"sqlite:///{db.as_posix()}")
    client = mlflow.MlflowClient()
    production = [
        v for v in client.search_model_versions(
            filter_string=f"name = '{name}'"
        )
        if v.current_stage == "Production"
    ]
    if (
        len(production) != 1
        or str(production[0].version) != str(selected["active_version"])
    ):
        raise ValueError("Production version differs from metadata.")

    model = mlflow.pyfunc.load_model(selected["inference_uri"])
    schema = model.metadata.get_input_schema()
    columns = schema.input_names()

    input_path = args.input
    if not input_path.is_absolute():
        input_path = ROOT / input_path
    data = pd.read_csv(input_path)

    missing = sorted(set(columns) - set(data.columns))
    if missing:
        raise ValueError(f"Missing features: {missing}")
    inputs = data[columns].copy()
    if inputs.empty:
        raise ValueError("Input contains no rows.")

    for field in schema.inputs:
        values = pd.to_numeric(inputs[field.name], errors="raise")
        if not np.isfinite(values.to_numpy()).all():
            raise ValueError(f"Invalid values: {field.name}")
        if field.type.name in ("integer", "long"):
            if not values.eq(np.floor(values)).all():
                raise ValueError(f"Expected integers: {field.name}")
            dtype = "int32" if field.type.name == "integer" else "int64"
        else:
            dtype = "float64"
        inputs[field.name] = values.astype(dtype)

    predictions = np.asarray(model.predict(inputs)).reshape(-1)
    if len(predictions) != len(inputs) or not np.isfinite(predictions).all():
        raise ValueError("Invalid predictions.")

    unit = "degC" if args.target == "temperature" else "mm"
    print(f"URI: {selected['inference_uri']}")
    print(f"Version: {production[0].version} | Rows: {len(inputs)}")
    print(pd.DataFrame({f"prediction_{unit}": predictions}).to_string(index=False))


if __name__ == "__main__":
    main()