from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.model_selection import TimeSeriesSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.utils.class_weight import compute_class_weight


TARGET = "FTR"
TRAIN_PATH = Path("data/matches/preprocessed_train.csv")
TEST_PATH = Path("data/matches/preprocessed_test.csv")
FEATURES_LINEAR_PATH = Path("data/matches/features_linear.csv")
FEATURES_TREES_PATH = Path("data/matches/features_trees.csv")
RANDOM_STATE = 42
RESULTS_DIR = Path("results")


@dataclass
class ModelSpec:
    name: str
    estimator: object
    feature_set: str
    scale: bool
    param_grid: list[dict[str, object]]


def load_features(feature_path: Path, target: str) -> list[str]:
    if not feature_path.exists():
        raise FileNotFoundError(
            f"Missing feature set file: {feature_path}. Run the feature selection script first."
        )
    df = pd.read_csv(feature_path)
    return [col for col in df.columns if col != target]


def load_dataset(path: Path, features: list[str], target: str) -> tuple[pd.DataFrame, np.ndarray]:
    df = pd.read_csv(path)
    X = df[features].copy()
    y = df[target].values
    return X, y


def build_pipeline(estimator: object, scale: bool) -> Pipeline:
    steps: list[tuple[str, object]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", estimator))
    return Pipeline(steps)


def evaluate_model(
    name: str,
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    output_dir: Path,
) -> dict[str, float]:
    pipeline.fit(X_train, y_train)
    train_preds = pipeline.predict(X_train)
    preds = pipeline.predict(X_test)

    train_accuracy = accuracy_score(y_train, train_preds)
    accuracy = accuracy_score(y_test, preds)
    macro_f1 = f1_score(y_test, preds, average="macro")
    precision, recall, _, _ = precision_recall_fscore_support(
        y_test,
        preds,
        average="macro",
        zero_division=0,
    )

    cm = confusion_matrix(y_test, preds)
    output_dir.mkdir(parents=True, exist_ok=True)
    cm_df = pd.DataFrame(cm)
    cm_df.to_csv(output_dir / "confusion.csv", index=False)

    return {
        "model": name,
        "train_accuracy": train_accuracy,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "macro_precision": precision,
        "macro_recall": recall,
    }


def write_model_summary(
    output_dir: Path,
    metrics: dict[str, float],
    best_params: dict[str, object],
) -> None:
    summary = {
        "metrics": metrics,
        "best_params": best_params,
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, default=str)


def build_models(class_weights: dict[int, float]) -> list[ModelSpec]:
    return [
        ModelSpec(
            name="home_win_baseline",
            estimator=DummyClassifier(strategy="constant", constant=2),
            feature_set="linear",
            scale=False,
            param_grid=[
                {},
            ],
        ),
        ModelSpec(
            name="logistic_regression",
            estimator=LogisticRegression(
                max_iter=2000,
                class_weight=class_weights,
                solver="saga",
                random_state=RANDOM_STATE,
            ),
            feature_set="linear",
            scale=True,
            param_grid=[
                {"model__C": 0.1},
                {"model__C": 0.5},
                {"model__C": 1.0},
            ],
        ),
        ModelSpec(
            name="svm_rbf",
            estimator=SVC(
                kernel="rbf",
                gamma="scale",
                class_weight=class_weights,
                random_state=RANDOM_STATE,
            ),
            feature_set="linear",
            scale=True,
            param_grid=[
                {"model__C": 0.5},
                {"model__C": 1.0},
                {"model__C": 2.0},
            ],
        ),
        ModelSpec(
            name="knn",
            estimator=KNeighborsClassifier(
                weights="distance",
            ),
            feature_set="linear",
            scale=True,
            param_grid=[
                {"model__n_neighbors": 7},
                {"model__n_neighbors": 11},
                {"model__n_neighbors": 15},
            ],
        ),
        ModelSpec(
            name="gaussian_nb",
            estimator=GaussianNB(),
            feature_set="linear",
            scale=False,
            param_grid=[
                {},
            ],
        ),
        ModelSpec(
            name="random_forest",
            estimator=RandomForestClassifier(
                class_weight=class_weights,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            feature_set="trees",
            scale=False,
            param_grid=[
                {"model__n_estimators": 300, "model__max_depth": 8, "model__min_samples_leaf": 5},
                {"model__n_estimators": 500, "model__max_depth": 10, "model__min_samples_leaf": 3},
            ],
        ),
        ModelSpec(
            name="gradient_boosting",
            estimator=GradientBoostingClassifier(
                random_state=RANDOM_STATE,
            ),
            feature_set="trees",
            scale=False,
            param_grid=[
                {"model__n_estimators": 200, "model__learning_rate": 0.05, "model__max_depth": 3},
                {"model__n_estimators": 300, "model__learning_rate": 0.1, "model__max_depth": 3},
            ],
        ),
    ]


def split_train_validation(X: pd.DataFrame, y: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    tscv = TimeSeriesSplit(n_splits=5)
    splits = list(tscv.split(X))
    train_idx, val_idx = splits[-1]
    X_train = X.iloc[train_idx].copy()
    X_val = X.iloc[val_idx].copy()
    y_train = y[train_idx]
    y_val = y[val_idx]
    return X_train, X_val, y_train, y_val


def tune_model(
    spec: ModelSpec,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
) -> tuple[Pipeline, dict[str, object], dict[str, float]]:
    best_score = -np.inf
    best_params: dict[str, object] = {}
    best_pipeline: Pipeline | None = None
    best_metrics: dict[str, float] = {}

    for params in spec.param_grid:
        pipeline = build_pipeline(spec.estimator, spec.scale)
        pipeline.set_params(**params)
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_val)
        macro_f1 = f1_score(y_val, preds, average="macro")
        accuracy = accuracy_score(y_val, preds)
        if macro_f1 > best_score:
            best_score = macro_f1
            best_params = params
            best_pipeline = pipeline
            best_metrics = {"val_macro_f1": macro_f1, "val_accuracy": accuracy}

    if best_pipeline is None:
        raise RuntimeError(f"No valid params for model {spec.name}.")
    return best_pipeline, best_params, best_metrics


def main() -> None:
    if not TRAIN_PATH.exists() or not TEST_PATH.exists():
        raise FileNotFoundError("Missing preprocessed train/test data. Run preprocessing first.")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    linear_features = load_features(FEATURES_LINEAR_PATH, TARGET)
    tree_features = load_features(FEATURES_TREES_PATH, TARGET)

    X_train_linear, y_train = load_dataset(TRAIN_PATH, linear_features, TARGET)
    X_test_linear, y_test = load_dataset(TEST_PATH, linear_features, TARGET)

    X_train_tree, _ = load_dataset(TRAIN_PATH, tree_features, TARGET)
    X_test_tree, _ = load_dataset(TEST_PATH, tree_features, TARGET)

    X_train_linear_split, X_val_linear, y_train_split, y_val = split_train_validation(
        X_train_linear, y_train
    )
    X_train_tree_split, X_val_tree, _, _ = split_train_validation(X_train_tree, y_train)

    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    class_weights = dict(zip(classes, weights))

    results: list[dict[str, float]] = []
    for spec in build_models(class_weights):
        if spec.feature_set == "linear":
            X_tr, X_val = X_train_linear_split, X_val_linear
            X_te = X_test_linear
        else:
            X_tr, X_val = X_train_tree_split, X_val_tree
            X_te = X_test_tree

        tuned_pipeline, best_params, val_metrics = tune_model(
            spec,
            X_tr,
            y_train_split,
            X_val,
            y_val,
        )

        tuned_pipeline.fit(
            pd.concat([X_tr, X_val], axis=0),
            np.concatenate([y_train_split, y_val]),
        )
        model_dir = RESULTS_DIR / spec.name
        metrics = evaluate_model(
            spec.name,
            tuned_pipeline,
            pd.concat([X_tr, X_val], axis=0),
            np.concatenate([y_train_split, y_val]),
            X_te,
            y_test,
            model_dir,
        )
        metrics.update(val_metrics)
        metrics["best_params"] = best_params
        write_model_summary(model_dir, metrics, best_params)
        results.append(metrics)
        print(
            f"{spec.name}: accuracy={metrics['accuracy']:.4f} "
            f"macro_f1={metrics['macro_f1']:.4f} "
            f"macro_precision={metrics['macro_precision']:.4f} "
            f"macro_recall={metrics['macro_recall']:.4f}"
        )

    results_df = pd.DataFrame(results).sort_values("macro_f1", ascending=False)
    results_path = RESULTS_DIR / "model_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\nSaved model results to {results_path}")


if __name__ == "__main__":
    main()
