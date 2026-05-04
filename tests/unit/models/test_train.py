from pathlib import Path
import json

import numpy as np
import pandas as pd

import train


def test_split_train_validation_shapes() -> None:
    X = pd.DataFrame({"a": range(20), "b": range(20, 40)})
    y = np.array([0, 1] * 10)

    X_train, X_val, y_train, y_val = train.split_train_validation(X, y)

    assert len(X_train) + len(X_val) == len(X)
    assert len(y_train) + len(y_val) == len(y)


def test_tune_model_returns_pipeline(tmp_path: Path) -> None:
    X = pd.DataFrame({"a": range(30), "b": range(30, 60)})
    y = np.array([0, 1, 2] * 10)

    X_train, X_val, y_train, y_val = train.split_train_validation(X, y)

    spec = train.ModelSpec(
        name="logistic_regression",
        estimator=train.LogisticRegression(
            max_iter=200,
            class_weight=None,
            solver="saga",
            random_state=train.RANDOM_STATE,
        ),
        feature_set="linear",
        scale=True,
        param_grid=[{"model__C": 0.5}, {"model__C": 1.0}],
    )

    pipeline, best_params, metrics = train.tune_model(spec, X_train, y_train, X_val, y_val)

    assert "val_macro_f1" in metrics
    assert best_params
    assert pipeline is not None


def test_log_model_run_logs_artifacts_and_metrics(tmp_path: Path, monkeypatch) -> None:
    feature_path = tmp_path / "features.csv"
    feature_path.write_text("FTR,feature\n0,1\n", encoding="utf-8")

    output_dir = tmp_path / "results"
    output_dir.mkdir()
    (output_dir / "confusion.csv").write_text("0,1\n1,0\n", encoding="utf-8")
    (output_dir / "metrics.json").write_text(
        json.dumps({"metrics": {"accuracy": 0.5}, "best_params": {"model__C": 1.0}}),
        encoding="utf-8",
    )

    spec = train.ModelSpec(
        name="logistic_regression",
        estimator=train.LogisticRegression(),
        feature_set="linear",
        scale=True,
        param_grid=[{"model__C": 1.0}],
    )

    calls: list[tuple[str, object]] = []
    monkeypatch.setattr(train.mlflow, "log_params", lambda params: calls.append(("params", params)))
    monkeypatch.setattr(train.mlflow, "log_metrics", lambda metrics: calls.append(("metrics", metrics)))
    monkeypatch.setattr(train.mlflow, "log_artifact", lambda path, artifact_path=None: calls.append(("artifact", (path, artifact_path))))
    monkeypatch.setattr(train.mlflow, "log_artifacts", lambda path, artifact_path=None: calls.append(("artifacts", (path, artifact_path))))
    monkeypatch.setattr(train.mlflow.sklearn, "log_model", lambda pipeline, artifact_path=None: calls.append(("model", (pipeline, artifact_path))))

    train._log_model_run(
        spec,
        pipeline=object(),
        feature_path=feature_path,
        output_dir=output_dir,
        metrics={"model": spec.name, "accuracy": 0.5, "macro_f1": 0.4},
        best_params={"model__C": 1.0},
        val_metrics={"val_macro_f1": 0.45, "val_accuracy": 0.5},
    )

    assert ("params", {"model_name": "logistic_regression", "feature_set": "linear", "uses_scaling": True, "model__C": 1.0}) in calls
    assert ("metrics", {"accuracy": 0.5, "macro_f1": 0.4, "val_macro_f1": 0.45, "val_accuracy": 0.5}) in calls
    assert ("artifact", (str(feature_path), "inputs")) in calls
    assert ("artifacts", (str(output_dir), "evaluation")) in calls
    assert any(name == "model" for name, _ in calls)
