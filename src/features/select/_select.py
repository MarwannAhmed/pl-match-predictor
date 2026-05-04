from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import shap
from xgboost import XGBClassifier

from scipy.special import logit
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV, VarianceThreshold
from sklearn.inspection import permutation_importance
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight


TARGET = "FTR"
RESULTS_DIR = Path("results/feature_selection")
HARD_DROP = [
    "DayOfWeek",
    "ELO_RATIO",
    "B365H",
    "B365D",
    "B365A",
    "FTHG",
    "FTAG",
    "HS",
    "AS",
    "HST",
    "AST",
    "HxG",
    "AxG",
    "ID",
    "Season",
]
PROTECTED_VIF = {"IMP_H_PROB_ODDS", "ELO_DIFF", "ELO_MARKET_DIFF", "xG_FORM_DIFF", "HPTS_FORM", "APTS_FORM"}
VARIANCE_THRESHOLD = 0.01
VIF_THRESHOLD = 10
TOP_N = 30
RANDOM_STATE = 42
CORRELATION_THRESHOLD = 0.85


def _split_target(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, np.ndarray]:
    y = df[target].values
    X = df.drop(columns=[target])
    return X, y


def hard_drop_features(X: pd.DataFrame, drop_cols: list[str]) -> pd.DataFrame:
    return X.drop(columns=drop_cols, errors="ignore")


def variance_filter(X: pd.DataFrame, threshold: float) -> tuple[pd.DataFrame, list[str]]:
    selector = VarianceThreshold(threshold=threshold)
    selector.fit(X)
    keep_mask = selector.get_support()
    removed = X.columns[~keep_mask].tolist()
    return X.loc[:, keep_mask], removed


def correlation_filter(
    X: pd.DataFrame,
    threshold: float,
    preferred_order: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    corr = X.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

    rank_map = {name: rank for rank, name in enumerate(preferred_order)}
    to_drop: set[str] = set()
    for col in upper.columns:
        for row in upper.index:
            value = upper.loc[row, col]
            if pd.notna(value) and value > threshold:
                col_rank = rank_map.get(col, len(rank_map))
                row_rank = rank_map.get(row, len(rank_map))
                if col_rank < row_rank:
                    to_drop.add(row)
                elif row_rank < col_rank:
                    to_drop.add(col)
                else:
                    to_drop.add(row)
    return X.drop(columns=sorted(to_drop), errors="ignore"), sorted(to_drop)


def compute_vif(X: pd.DataFrame) -> pd.DataFrame:
    df_clean = X.dropna(axis=1)
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(df_clean), columns=df_clean.columns)
    with np.errstate(divide="ignore", invalid="ignore"):
        vifs = [
            variance_inflation_factor(X_scaled.values, i)
            for i in range(X_scaled.shape[1])
        ]
    vif_data = pd.DataFrame({
        "Feature": X_scaled.columns,
        "VIF": vifs,
    })
    return vif_data.sort_values("VIF", ascending=False).reset_index(drop=True)


def iterative_vif_removal(
    X: pd.DataFrame,
    threshold: float,
    protected: set[str],
    verbose: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    df_work = X.copy().dropna(axis=1)
    removed = []
    while True:
        vif_result = compute_vif(df_work)
        max_vif = vif_result["VIF"].max()
        if max_vif <= threshold:
            break
        worst_feature = vif_result.loc[vif_result["VIF"].idxmax(), "Feature"]
        if worst_feature in protected:
            non_protected = vif_result[~vif_result["Feature"].isin(protected)]
            if non_protected.empty:
                break
            worst_feature = non_protected.iloc[0]["Feature"]
        if verbose:
            print(f"  Removing '{worst_feature}' (VIF={max_vif:.1f})")
        df_work = df_work.drop(columns=[worst_feature])
        removed.append(worst_feature)
    return df_work, removed


def permutation_importance_rank(X: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
    tscv = TimeSeriesSplit(n_splits=5)
    splits = list(tscv.split(X))
    train_idx, val_idx = splits[-1]

    imputer = SimpleImputer(strategy="median")
    X_tr = imputer.fit_transform(X.iloc[train_idx].values)
    X_val = imputer.transform(X.iloc[val_idx].values)
    y_tr = y[train_idx]
    y_val = y[val_idx]

    classes = np.unique(y_tr)
    class_weights = compute_class_weight("balanced", classes=classes, y=y_tr)
    cw_dict = dict(zip(classes, class_weights))

    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=10,
        class_weight=cw_dict,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    rf.fit(X_tr, y_tr)

    perm_result = permutation_importance(
        rf,
        X_val,
        y_val,
        n_repeats=20,
        random_state=RANDOM_STATE,
        scoring="accuracy",
        n_jobs=-1,
    )

    perm_df = pd.DataFrame({
        "Feature": X.columns,
        "Importance_Mean": perm_result.importances_mean,
        "Importance_Std": perm_result.importances_std,
    }).sort_values("Importance_Mean", ascending=False).reset_index(drop=True)
    return perm_df


def plot_permutation_importance(perm_df: pd.DataFrame, output_path: str) -> None:
    fig, ax = plt.subplots(figsize=(10, max(6, len(perm_df) * 0.28)))
    colors = ["#1f77b4" if v >= 0 else "#d62728" for v in perm_df["Importance_Mean"]]
    ax.barh(
        perm_df["Feature"][::-1],
        perm_df["Importance_Mean"][::-1],
        xerr=perm_df["Importance_Std"][::-1],
        color=colors[::-1],
        align="center",
        capsize=3,
    )
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Mean decrease in accuracy (permutation)")
    ax.set_title("Permutation feature importance - RandomForest")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def drop_bottom_importance(
    X: pd.DataFrame,
    perm_df: pd.DataFrame,
    drop_fraction: float,
) -> tuple[pd.DataFrame, list[str]]:
    drop_count = max(1, int(len(perm_df) * drop_fraction))
    bottom = perm_df.tail(drop_count)["Feature"].tolist()
    return X.drop(columns=bottom, errors="ignore"), bottom


def rfecv_select(X: pd.DataFrame, y: np.ndarray) -> tuple[list[str], RFECV]:
    imputer = SimpleImputer(strategy="median")
    X_imp = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    prob_features = [
        col for col in X_imp.columns
        if col in {"IMP_H_PROB_ODDS", "IMP_A_PROB_ODDS", "H_PROB_ELO"}
    ]
    other_features = [col for col in X_imp.columns if col not in prob_features]

    X_scaled = X_imp.copy()
    for col in prob_features:
        X_scaled[col] = logit(X_imp[col].clip(0.01, 0.99))

    scaler = StandardScaler()
    X_scaled[other_features] = scaler.fit_transform(X_imp[other_features])

    classes = np.unique(y)
    class_weights = compute_class_weight("balanced", classes=classes, y=y)
    cw_dict = dict(zip(classes, class_weights))

    lr = LogisticRegression(
        C=0.1,
        max_iter=1000,
        class_weight=cw_dict,
        solver="saga",
        random_state=RANDOM_STATE,
    )

    min_features = min(10, max(2, int(len(X_scaled.columns) * 0.5)))
    rfecv = RFECV(
        estimator=lr,
        step=1,
        cv=TimeSeriesSplit(n_splits=5),
        scoring="f1_macro",
        min_features_to_select=min_features,
        n_jobs=-1,
        verbose=1,
    )
    rfecv.fit(X_scaled.values, y)
    selected = X_scaled.columns[rfecv.support_].tolist()
    return selected, rfecv


def plot_rfecv_curve(rfecv: RFECV, output_path: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    n_features_range = range(
        rfecv.min_features_to_select,
        len(rfecv.cv_results_["mean_test_score"]) + rfecv.min_features_to_select,
    )
    ax.plot(
        n_features_range,
        rfecv.cv_results_["mean_test_score"],
        marker="o",
        markersize=4,
        linewidth=1.5,
    )
    ax.fill_between(
        n_features_range,
        rfecv.cv_results_["mean_test_score"] - rfecv.cv_results_["std_test_score"],
        rfecv.cv_results_["mean_test_score"] + rfecv.cv_results_["std_test_score"],
        alpha=0.2,
    )
    ax.axvline(
        rfecv.n_features_,
        color="red",
        linestyle="--",
        label=f"Optimal: {rfecv.n_features_}",
    )
    ax.set_xlabel("Number of features")
    ax.set_ylabel("CV macro F1")
    ax.set_title("RFECV: macro F1 vs number of features")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def save_feature_sets(
    df: pd.DataFrame,
    target: str,
    features_trees: list[str],
    features_linear: list[str],
) -> None:
    df_trees = df[features_trees + [target]]
    df_linear = df[features_linear + [target]]
    df_trees.to_csv("data/matches/features_trees.csv", index=False)
    df_linear.to_csv("data/matches/features_linear.csv", index=False)


def shap_summary_plot(
    X: pd.DataFrame,
    y: np.ndarray,
    output_path: Path,
) -> None:
    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="multi:softprob",
        num_class=len(np.unique(y)),
        random_state=RANDOM_STATE,
        n_jobs=-1,
        eval_metric="mlogloss",
    )
    model.fit(X, y)

    sample = X.sample(n=min(len(X), 2000), random_state=RANDOM_STATE)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, sample, show=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def print_high_corr_pairs(df: pd.DataFrame, features: list[str], threshold: float = 0.85) -> None:
    corr_matrix = df[features].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    pairs = [
        (col, row, upper.loc[row, col])
        for col in upper.columns
        for row in upper.index
        if pd.notna(upper.loc[row, col]) and upper.loc[row, col] > threshold
    ]
    if pairs:
        print(f"Remaining high-correlation pairs (|r| > {threshold}):")
        for c1, c2, corr in sorted(pairs, key=lambda x: -x[2]):
            print(f"  {c1} <-> {c2}: r={corr:.3f}")
    else:
        print(f"No remaining pairs with |r| > {threshold}.")


def main(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df = df.drop(columns="Day", errors="ignore")

    X, y = _split_target(df, TARGET)
    print(f"Starting features: {X.shape[1]}")

    print("\n-- Hard drops --")
    X = hard_drop_features(X, HARD_DROP)
    print(f"After hard drops: {X.shape[1]} features")
    print(f"  Dropped: {HARD_DROP}")

    print("\n-- Variance filter --")
    X, low_var_features = variance_filter(X, VARIANCE_THRESHOLD)
    if low_var_features:
        print(f"Near-zero variance features removed: {low_var_features}")
    else:
        print("No near-zero variance features found.")
    print(f"After variance filter: {X.shape[1]} features")

    print("\n-- Correlation filter --")
    interpretability_rank = [
        "ELO_DIFF",
        "IMP_H_PROB_ODDS",
        "HPTS_FORM",
        "APTS_FORM",
        "H_POS",
        "A_POS",
        "ELO_MARKET_DIFF",
        "xG_FORM_DIFF",
    ]
    X, corr_removed = correlation_filter(X, CORRELATION_THRESHOLD, interpretability_rank)
    print(f"Correlation filter removed {len(corr_removed)} features: {corr_removed}")
    print(f"After correlation filter: {X.shape[1]} features")

    print("\n-- VIF scores (top 20) --")
    vif_df = compute_vif(X)
    print(vif_df.head(20).to_string(index=False))
    high_vif = vif_df[vif_df["VIF"] > VIF_THRESHOLD]["Feature"].tolist()
    print(f"\nFeatures with VIF > {VIF_THRESHOLD}: {high_vif}")

    print("\n-- Iterative VIF removal --")
    X_vif, vif_removed = iterative_vif_removal(X, VIF_THRESHOLD, PROTECTED_VIF)
    print(f"VIF removal dropped {len(vif_removed)} features: {vif_removed}")
    print(f"After VIF removal: {X_vif.shape[1]} features")

    print("\n-- Permutation importance (RandomForest) --")
    perm_df = permutation_importance_rank(X_vif, y)
    print("\nTop 20 features by permutation importance:")
    print(perm_df.head(20).to_string(index=False))
    plot_permutation_importance(perm_df, str(RESULTS_DIR / "permutation_importance.png"))

    X_pi, pi_removed = drop_bottom_importance(X_vif, perm_df, drop_fraction=0.2)
    print(f"\nDropped bottom 20% features by permutation importance: {pi_removed}")
    print(f"After permutation importance filter: {X_pi.shape[1]} features")

    print("\n-- SHAP summary (XGBoost) --")
    shap_summary_plot(X_pi, y, RESULTS_DIR / "shap_summary.png")
    print(f"Saved SHAP summary plot to {RESULTS_DIR / 'shap_summary.png'}")

    print("\n-- RFECV (Logistic Regression + TimeSeriesSplit) --")
    rfecv_selected, rfecv = rfecv_select(X_pi, y)
    print(f"\nRFECV optimal feature count: {rfecv.n_features_}")
    print(f"Selected features:\n{rfecv_selected}")
    plot_rfecv_curve(rfecv, str(RESULTS_DIR / "rfecv_curve.png"))

    print("\n==========================================")
    print("FINAL FEATURE SETS")
    print("==========================================")
    tree_features = X_pi.columns.tolist()
    print(f"\nTree models ({len(tree_features)} features):")
    for feature in tree_features:
        importance = perm_df.loc[perm_df["Feature"] == feature, "Importance_Mean"].values
        imp_str = f"  (importance={importance[0]:.4f})" if len(importance) > 0 else ""
        print(f"  {feature}{imp_str}")

    print(f"\nLinear models ({len(rfecv_selected)} features):")
    for feature in rfecv_selected:
        print(f"  {feature}")

    save_feature_sets(df, TARGET, tree_features, rfecv_selected)
    print("\nSaved: features_trees.csv and features_linear.csv")

    print("\n-- Pairwise correlation check on final tree feature set --")
    print_high_corr_pairs(df, tree_features)

    return tree_features, rfecv_selected


if __name__ == "__main__":
    df = pd.read_csv("data/matches/preprocessed_train.csv")
    main(df)