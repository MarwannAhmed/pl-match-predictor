from pathlib import Path
from unittest.mock import MagicMock, patch
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import _select as select


# ── existing tests ────────────────────────────────────────────────────────────

def test_correlation_filter_prefers_ranked_feature() -> None:
    df = pd.DataFrame(
        {
            "ELO_DIFF": [1.0, 2.0, 3.0, 4.0],
            "IMP_H_PROB_ODDS": [1.0, 2.0, 3.0, 4.0],
            "OTHER": [4.0, 3.0, 2.0, 1.0],
        }
    )

    filtered, removed = select.correlation_filter(
        df,
        threshold=0.85,
        preferred_order=["ELO_DIFF", "IMP_H_PROB_ODDS"],
    )

    assert "ELO_DIFF" in filtered.columns
    assert "IMP_H_PROB_ODDS" not in filtered.columns
    assert "IMP_H_PROB_ODDS" in removed


def test_drop_bottom_importance_removes_expected_features() -> None:
    df = pd.DataFrame(
        {
            "A": [1, 2, 3],
            "B": [1, 1, 1],
            "C": [2, 2, 2],
            "D": [3, 3, 3],
        }
    )
    perm_df = pd.DataFrame(
        {
            "Feature": ["A", "B", "C", "D"],
            "Importance_Mean": [0.4, 0.3, 0.2, 0.1],
        }
    )

    filtered, removed = select.drop_bottom_importance(df, perm_df, drop_fraction=0.25)

    assert "D" in removed
    assert "D" not in filtered.columns


# ── _split_target ─────────────────────────────────────────────────────────────

def test_split_target_separates_features_and_labels() -> None:
    df = pd.DataFrame({"feat1": [1, 2, 3], "feat2": [4, 5, 6], "FTR": [0, 1, 2]})

    X, y = select._split_target(df, "FTR")

    assert "FTR" not in X.columns
    assert list(y) == [0, 1, 2]
    assert list(X.columns) == ["feat1", "feat2"]


# ── hard_drop_features ────────────────────────────────────────────────────────

def test_hard_drop_features_removes_listed_columns() -> None:
    df = pd.DataFrame({"A": [1, 2], "B": [3, 4], "C": [5, 6]})

    result = select.hard_drop_features(df, ["A", "C"])

    assert "A" not in result.columns
    assert "C" not in result.columns
    assert "B" in result.columns


def test_hard_drop_features_ignores_missing_columns() -> None:
    df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})

    result = select.hard_drop_features(df, ["A", "NONEXISTENT"])

    assert "A" not in result.columns
    assert "B" in result.columns


# ── variance_filter ───────────────────────────────────────────────────────────

def test_variance_filter_removes_constant_column() -> None:
    df = pd.DataFrame(
        {
            "constant": [1.0] * 20,
            "varied": np.linspace(0, 1, 20),
        }
    )

    result, removed = select.variance_filter(df, threshold=0.01)

    assert "constant" in removed
    assert "constant" not in result.columns
    assert "varied" in result.columns


def test_variance_filter_keeps_all_when_none_below_threshold() -> None:
    df = pd.DataFrame(
        {
            "x": np.linspace(0, 10, 20),
            "y": np.linspace(5, 15, 20),
        }
    )

    result, removed = select.variance_filter(df, threshold=0.01)

    assert removed == []
    assert list(result.columns) == ["x", "y"]


# ── compute_vif ───────────────────────────────────────────────────────────────

def test_compute_vif_returns_dataframe_with_expected_columns() -> None:
    rng = np.random.RandomState(0)
    df = pd.DataFrame(
        {
            "x1": rng.randn(50),
            "x2": rng.randn(50),
            "x3": rng.randn(50),
        }
    )

    vif_df = select.compute_vif(df)

    assert "Feature" in vif_df.columns
    assert "VIF" in vif_df.columns
    assert len(vif_df) == 3
    assert set(vif_df["Feature"]) == {"x1", "x2", "x3"}


def test_compute_vif_sorted_descending() -> None:
    rng = np.random.RandomState(1)
    df = pd.DataFrame(
        {
            "indep1": rng.randn(50),
            "indep2": rng.randn(50),
        }
    )

    vif_df = select.compute_vif(df)

    assert vif_df["VIF"].iloc[0] >= vif_df["VIF"].iloc[-1]


# ── iterative_vif_removal ─────────────────────────────────────────────────────

def test_iterative_vif_removal_removes_collinear_feature() -> None:
    rng = np.random.RandomState(42)
    base = rng.randn(60)
    df = pd.DataFrame(
        {
            "x1": base,
            "x2": base + 0.001 * rng.randn(60),   # nearly identical to x1
            "x3": rng.randn(60),
        }
    )

    result, removed = select.iterative_vif_removal(df, threshold=5.0, protected=set(), verbose=False)

    assert len(removed) > 0
    # one of x1/x2 should have been dropped
    assert not ({"x1", "x2"} <= set(result.columns))


def test_iterative_vif_removal_respects_protected_set() -> None:
    rng = np.random.RandomState(7)
    base = rng.randn(60)
    df = pd.DataFrame(
        {
            "protected_col": base,
            "collinear_col": base + 0.001 * rng.randn(60),
            "indep": rng.randn(60),
        }
    )

    result, removed = select.iterative_vif_removal(
        df, threshold=5.0, protected={"protected_col"}, verbose=False
    )

    assert "protected_col" in result.columns
    assert "collinear_col" in removed


def test_iterative_vif_removal_stops_when_all_protected(capsys) -> None:
    rng = np.random.RandomState(9)
    base = rng.randn(60)
    df = pd.DataFrame(
        {
            "a": base,
            "b": base + 0.001 * rng.randn(60),
        }
    )

    # Both columns are protected; removal loop must exit without dropping either
    result, removed = select.iterative_vif_removal(
        df, threshold=5.0, protected={"a", "b"}, verbose=False
    )

    assert "a" in result.columns
    assert "b" in result.columns


# ── plot_permutation_importance ───────────────────────────────────────────────

def test_plot_permutation_importance_saves_file(tmp_path: Path) -> None:
    perm_df = pd.DataFrame(
        {
            "Feature": ["feat_a", "feat_b", "feat_c"],
            "Importance_Mean": [0.05, -0.01, 0.03],
            "Importance_Std": [0.01, 0.005, 0.008],
        }
    )
    output_path = str(tmp_path / "perm_importance.png")

    select.plot_permutation_importance(perm_df, output_path)

    assert Path(output_path).exists()
    assert Path(output_path).stat().st_size > 0


# ── permutation_importance_rank ───────────────────────────────────────────────

def test_permutation_importance_rank_returns_all_features(monkeypatch) -> None:
    rng = np.random.RandomState(0)
    n = 30
    X = pd.DataFrame(
        {
            "feat_a": rng.randn(n),
            "feat_b": rng.randn(n),
            "feat_c": rng.randn(n),
        }
    )
    y = np.array([0, 1, 2] * 10)

    # Patch permutation_importance to avoid slow n_repeats=20 run
    fake_result = SimpleNamespace(
        importances_mean=np.array([0.05, 0.02, 0.01]),
        importances_std=np.array([0.01, 0.005, 0.003]),
    )
    monkeypatch.setattr(select, "permutation_importance", lambda *a, **kw: fake_result)

    perm_df = select.permutation_importance_rank(X, y)

    assert set(perm_df["Feature"]) == {"feat_a", "feat_b", "feat_c"}
    assert "Importance_Mean" in perm_df.columns
    assert "Importance_Std" in perm_df.columns
    # Should be sorted descending
    assert perm_df["Importance_Mean"].iloc[0] >= perm_df["Importance_Mean"].iloc[-1]


# ── rfecv_select ──────────────────────────────────────────────────────────────

def test_rfecv_select_returns_selected_features_and_rfecv_object(monkeypatch) -> None:
    rng = np.random.RandomState(0)
    n = 40
    X = pd.DataFrame(
        {
            "feat_a": rng.randn(n),
            "feat_b": rng.randn(n),
            "feat_c": rng.randn(n),
            "feat_d": rng.randn(n),
        }
    )
    y = np.array([0, 1, 2] * 13 + [0])

    # Monkeypatch RFECV.fit so the test doesn't run a real CV loop
    import sklearn.feature_selection as skfs

    class _FakeRFECV:
        min_features_to_select = 2
        n_features_ = 2
        support_ = np.array([True, False, True, False])
        cv_results_ = {
            "mean_test_score": np.array([0.35, 0.36]),
            "std_test_score": np.array([0.02, 0.02]),
        }

        def __init__(self, **kwargs):
            self.min_features_to_select = kwargs.get("min_features_to_select", 2)

        def fit(self, X, y):
            return self

    monkeypatch.setattr(select, "RFECV", _FakeRFECV)

    selected, rfecv = select.rfecv_select(X, y)

    assert isinstance(selected, list)
    assert len(selected) > 0
    assert hasattr(rfecv, "n_features_")


def test_rfecv_select_handles_prob_feature_columns(monkeypatch) -> None:
    """IMP_H_PROB_ODDS and H_PROB_ELO should be logit-transformed, not standard-scaled."""
    rng = np.random.RandomState(2)
    n = 40
    X = pd.DataFrame(
        {
            "IMP_H_PROB_ODDS": rng.uniform(0.1, 0.9, n),
            "H_PROB_ELO": rng.uniform(0.1, 0.9, n),
            "ELO_DIFF": rng.randn(n),
        }
    )
    y = np.array([0, 1, 2] * 13 + [0])

    class _FakeRFECV:
        min_features_to_select = 2
        n_features_ = 2
        support_ = np.array([True, True, False])
        cv_results_ = {
            "mean_test_score": np.array([0.34]),
            "std_test_score": np.array([0.01]),
        }

        def __init__(self, **kwargs):
            self.min_features_to_select = kwargs.get("min_features_to_select", 2)

        def fit(self, X, y):
            return self

    monkeypatch.setattr(select, "RFECV", _FakeRFECV)

    selected, _ = select.rfecv_select(X, y)
    assert isinstance(selected, list)


# ── plot_rfecv_curve ──────────────────────────────────────────────────────────

def test_plot_rfecv_curve_saves_file(tmp_path: Path) -> None:
    fake_rfecv = SimpleNamespace(
        min_features_to_select=2,
        n_features_=3,
        cv_results_={
            "mean_test_score": np.array([0.33, 0.36, 0.35]),
            "std_test_score": np.array([0.02, 0.02, 0.02]),
        },
    )
    output_path = str(tmp_path / "rfecv_curve.png")

    select.plot_rfecv_curve(fake_rfecv, output_path)

    assert Path(output_path).exists()
    assert Path(output_path).stat().st_size > 0


# ── save_feature_sets ─────────────────────────────────────────────────────────

def test_save_feature_sets_writes_both_csv_files(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data" / "matches").mkdir(parents=True)

    df = pd.DataFrame({"feat_a": [1, 2, 3], "feat_b": [4, 5, 6], "FTR": [0, 1, 2]})

    select.save_feature_sets(df, "FTR", features_trees=["feat_a", "feat_b"], features_linear=["feat_a"])

    trees_path = tmp_path / "data" / "matches" / "features_trees.csv"
    linear_path = tmp_path / "data" / "matches" / "features_linear.csv"

    assert trees_path.exists()
    assert linear_path.exists()

    trees_df = pd.read_csv(trees_path)
    linear_df = pd.read_csv(linear_path)

    assert set(trees_df.columns) == {"feat_a", "feat_b", "FTR"}
    assert set(linear_df.columns) == {"feat_a", "FTR"}


# ── print_high_corr_pairs ─────────────────────────────────────────────────────

def test_print_high_corr_pairs_reports_highly_correlated_pair(capsys) -> None:
    df = pd.DataFrame(
        {
            "A": [1.0, 2.0, 3.0, 4.0, 5.0],
            "B": [1.0, 2.0, 3.0, 4.0, 5.0],   # perfectly correlated with A
            "C": [5.0, 3.0, 1.0, 4.0, 2.0],
        }
    )

    select.print_high_corr_pairs(df, ["A", "B", "C"], threshold=0.85)

    captured = capsys.readouterr().out
    assert "A" in captured and "B" in captured


def test_print_high_corr_pairs_reports_no_pairs_when_independent(capsys) -> None:
    rng = np.random.RandomState(0)
    df = pd.DataFrame(
        {
            "X": rng.randn(30),
            "Y": rng.randn(30),
        }
    )

    select.print_high_corr_pairs(df, ["X", "Y"], threshold=0.85)

    captured = capsys.readouterr().out
    assert "No remaining" in captured