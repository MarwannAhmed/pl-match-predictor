from pathlib import Path

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
