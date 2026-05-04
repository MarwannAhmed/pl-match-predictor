from pathlib import Path
import json
import json

import numpy as np
import pandas as pd
import pytest

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


# ── load_features ─────────────────────────────────────────────────────────────

def test_load_features_returns_non_target_columns(tmp_path: Path) -> None:
    df = pd.DataFrame({"feat1": [1, 2], "feat2": [3, 4], "FTR": [0, 1]})
    path = tmp_path / "features.csv"
    df.to_csv(path, index=False)

    features = train.load_features(path, "FTR")

    assert features == ["feat1", "feat2"]
    assert "FTR" not in features


def test_load_features_raises_for_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Missing feature set file"):
        train.load_features(tmp_path / "nonexistent.csv", "FTR")


# ── load_dataset ──────────────────────────────────────────────────────────────

def test_load_dataset_returns_correct_X_and_y(tmp_path: Path) -> None:
    df = pd.DataFrame({"feat1": [10, 20, 30], "other": [1, 2, 3], "FTR": [0, 1, 2]})
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)

    X, y = train.load_dataset(path, ["feat1"], "FTR")

    assert list(X.columns) == ["feat1"]
    assert list(y) == [0, 1, 2]
    assert len(X) == 3


# ── build_pipeline ────────────────────────────────────────────────────────────

def test_build_pipeline_with_scale_has_three_steps() -> None:
    from sklearn.linear_model import LogisticRegression as LR
    p = train.build_pipeline(LR(), scale=True)

    step_names = [name for name, _ in p.steps]
    assert step_names == ["imputer", "scaler", "model"]


def test_build_pipeline_without_scale_has_two_steps() -> None:
    from sklearn.linear_model import LogisticRegression as LR
    p = train.build_pipeline(LR(), scale=False)

    step_names = [name for name, _ in p.steps]
    assert step_names == ["imputer", "model"]


# ── evaluate_model ────────────────────────────────────────────────────────────

def test_evaluate_model_returns_metrics_and_writes_confusion_matrix(tmp_path: Path) -> None:
    rng = np.random.RandomState(0)
    X_train = pd.DataFrame({"a": rng.randn(30), "b": rng.randn(30)})
    y_train = np.array([0, 1, 2] * 10)
    X_test = pd.DataFrame({"a": rng.randn(9), "b": rng.randn(9)})
    y_test = np.array([0, 1, 2] * 3)

    pipeline = train.build_pipeline(
        train.LogisticRegression(max_iter=500, solver="saga", random_state=0),
        scale=True,
    )
    output_dir = tmp_path / "model_out"

    metrics = train.evaluate_model(
        "test_lr", pipeline, X_train, y_train, X_test, y_test, output_dir
    )

    assert metrics["model"] == "test_lr"
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["macro_f1"] <= 1.0
    assert 0.0 <= metrics["macro_precision"] <= 1.0
    assert 0.0 <= metrics["macro_recall"] <= 1.0
    assert (output_dir / "confusion.csv").exists()


def test_evaluate_model_confusion_matrix_shape(tmp_path: Path) -> None:
    rng = np.random.RandomState(1)
    X_train = pd.DataFrame({"a": rng.randn(30), "b": rng.randn(30)})
    y_train = np.array([0, 1, 2] * 10)
    X_test = pd.DataFrame({"a": rng.randn(9), "b": rng.randn(9)})
    y_test = np.array([0, 1, 2] * 3)

    pipeline = train.build_pipeline(
        train.LogisticRegression(max_iter=500, solver="saga", random_state=1),
        scale=True,
    )
    output_dir = tmp_path / "cm_out"
    train.evaluate_model("lr", pipeline, X_train, y_train, X_test, y_test, output_dir)

    cm_df = pd.read_csv(output_dir / "confusion.csv")
    assert cm_df.shape == (3, 3)


# ── write_model_summary ───────────────────────────────────────────────────────

def test_write_model_summary_creates_valid_json(tmp_path: Path) -> None:
    metrics = {"accuracy": 0.65, "macro_f1": 0.60, "macro_precision": 0.61, "macro_recall": 0.59}
    best_params = {"model__C": 1.0}

    train.write_model_summary(tmp_path, metrics, best_params)

    summary_path = tmp_path / "metrics.json"
    assert summary_path.exists()

    with summary_path.open() as f:
        data = json.load(f)

    assert data["metrics"]["accuracy"] == pytest.approx(0.65)
    assert data["best_params"]["model__C"] == pytest.approx(1.0)


def test_write_model_summary_creates_parent_dirs(tmp_path: Path) -> None:
    output_dir = tmp_path / "nested" / "dir"
    output_dir.mkdir(parents=True)

    train.write_model_summary(output_dir, {"accuracy": 0.5}, {})

    assert (output_dir / "metrics.json").exists()


# ── build_models ──────────────────────────────────────────────────────────────

def test_build_models_contains_all_expected_model_names() -> None:
    class_weights = {0: 1.0, 1: 1.2, 2: 0.9}
    models = train.build_models(class_weights)
    names = [m.name for m in models]

    for expected in [
        "home_win_baseline",
        "logistic_regression",
        "svm_rbf",
        "knn",
        "gaussian_nb",
        "random_forest",
        "gradient_boosting",
    ]:
        assert expected in names


def test_build_models_feature_set_assignments() -> None:
    class_weights = {0: 1.0, 1: 1.0, 2: 1.0}
    models = train.build_models(class_weights)
    model_map = {m.name: m for m in models}

    assert model_map["logistic_regression"].feature_set == "linear"
    assert model_map["svm_rbf"].feature_set == "linear"
    assert model_map["knn"].feature_set == "linear"
    assert model_map["gaussian_nb"].feature_set == "linear"
    assert model_map["random_forest"].feature_set == "trees"
    assert model_map["gradient_boosting"].feature_set == "trees"


def test_build_models_each_spec_has_nonempty_param_grid() -> None:
    class_weights = {0: 1.0, 1: 1.0, 2: 1.0}
    for spec in train.build_models(class_weights):
        assert len(spec.param_grid) >= 1, f"{spec.name} has empty param_grid"