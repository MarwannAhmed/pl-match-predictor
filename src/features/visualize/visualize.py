from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.decomposition import PCA


def _chunk_list(items: list[str], chunk_size: int) -> list[list[str]]:
    return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]


def _save_fig(pdf: PdfPages, fig: plt.Figure) -> None:
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _basic_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = pd.DataFrame({
        "non_null": df.notna().sum(),
        "nulls": df.isna().sum(),
        "unique": df.nunique(),
        "dtype": df.dtypes.astype(str),
    })
    return summary


def _numeric_columns(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(include="number").columns.tolist()


def _select_target(df: pd.DataFrame) -> str | None:
    return "FTR" if "FTR" in df.columns else None


def plot_overview(df: pd.DataFrame, pdf: PdfPages) -> None:
    summary = _basic_summary(df)
    fig = plt.figure(figsize=(11.0, 8.5))
    fig.suptitle("Dataset overview")

    ax_text = fig.add_subplot(2, 1, 1)
    ax_text.axis("off")
    text_lines = [
        f"Rows: {df.shape[0]}",
        f"Columns: {df.shape[1]}",
        f"Numeric columns: {len(_numeric_columns(df))}",
    ]
    ax_text.text(0.01, 0.95, "\n".join(text_lines), va="top", fontsize=11)

    ax_table = fig.add_subplot(2, 1, 2)
    ax_table.axis("off")
    summary_subset = summary.head(30)
    table = ax_table.table(
        cellText=summary_subset.values,
        colLabels=summary_subset.columns.tolist(),
        rowLabels=summary_subset.index.tolist(),
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.3)

    _save_fig(pdf, fig)


def plot_target_distribution(df: pd.DataFrame, pdf: PdfPages, target: str) -> None:
    counts = df[target].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    ax.bar(counts.index.astype(str), counts.values, color="#4C72B0")
    ax.set_title("Target distribution")
    ax.set_xlabel(target)
    ax.set_ylabel("Count")
    _save_fig(pdf, fig)


def plot_univariate(df: pd.DataFrame, pdf: PdfPages, features: list[str]) -> None:
    cols_per_page = 6
    rows = 3
    cols = 2
    for chunk in _chunk_list(features, cols_per_page):
        fig, axes = plt.subplots(rows, cols, figsize=(11.0, 8.5))
        fig.suptitle("Univariate distributions")
        for ax, col in zip(axes.ravel(), chunk, strict=False):
            ax.hist(df[col].dropna(), bins=30, color="#4C72B0", alpha=0.85)
            ax.set_title(col)
        for ax in axes.ravel()[len(chunk):]:
            ax.axis("off")
        _save_fig(pdf, fig)


def plot_bivariate_correlations(
    df: pd.DataFrame,
    pdf: PdfPages,
    target: str,
    features: list[str],
) -> list[str]:
    corr = df[features + [target]].corr()[target].drop(target)
    corr_sorted = corr.sort_values()

    fig, ax = plt.subplots(figsize=(10.0, 8.5))
    ax.barh(corr_sorted.index, corr_sorted.values, color="#55A868")
    ax.set_title("Feature correlation with target")
    ax.set_xlabel("Correlation")
    _save_fig(pdf, fig)

    return corr_sorted.abs().sort_values(ascending=False).index.tolist()


def plot_bivariate_boxplots(
    df: pd.DataFrame,
    pdf: PdfPages,
    target: str,
    features: list[str],
) -> None:
    cols_per_page = 6
    rows = 3
    cols = 2
    for chunk in _chunk_list(features, cols_per_page):
        fig, axes = plt.subplots(rows, cols, figsize=(11.0, 8.5))
        fig.suptitle("Feature distribution by target")
        for ax, col in zip(axes.ravel(), chunk, strict=False):
            data = [df.loc[df[target] == value, col].dropna() for value in sorted(df[target].dropna().unique())]
            ax.boxplot(data, tick_labels=sorted(df[target].dropna().unique()))
            ax.set_title(col)
            ax.set_xlabel(target)
        for ax in axes.ravel()[len(chunk):]:
            ax.axis("off")
        _save_fig(pdf, fig)


def plot_bivariate_scatter(
    df: pd.DataFrame,
    pdf: PdfPages,
    target: str | None,
    features: list[str],
) -> None:
    sample = df.sample(n=min(len(df), 4000), random_state=42)
    pairs = []
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            pairs.append((features[i], features[j]))

    cols = 2
    rows = 3
    per_page = rows * cols
    for chunk in _chunk_list(pairs, per_page):
        fig, axes = plt.subplots(rows, cols, figsize=(11.0, 8.5))
        fig.suptitle("Bivariate scatter plots")
        for ax, (x_col, y_col) in zip(axes.ravel(), chunk, strict=False):
            if target is None:
                ax.scatter(sample[x_col], sample[y_col], s=8, alpha=0.5, color="#4C72B0")
            else:
                for value in sorted(sample[target].dropna().unique()):
                    subset = sample[sample[target] == value]
                    ax.scatter(
                        subset[x_col],
                        subset[y_col],
                        s=8,
                        alpha=0.5,
                        label=str(value),
                    )
                ax.legend(title=target, fontsize=8)
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
        for ax in axes.ravel()[len(chunk):]:
            ax.axis("off")
        _save_fig(pdf, fig)


def plot_correlation_heatmap(df: pd.DataFrame, pdf: PdfPages, features: list[str]) -> None:
    corr = df[features].corr()
    fig, ax = plt.subplots(figsize=(11.0, 10.0))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_title("Correlation heatmap")
    ax.set_xticks(range(len(features)))
    ax.set_yticks(range(len(features)))
    ax.set_xticklabels(features, rotation=90, fontsize=7)
    ax.set_yticklabels(features, fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    _save_fig(pdf, fig)


def plot_pca_projection(
    df: pd.DataFrame,
    pdf: PdfPages,
    features: list[str],
    target: str | None,
) -> None:
    data = df[features].dropna()
    if data.empty:
        return
    pca = PCA(n_components=2, random_state=42)
    components = pca.fit_transform(data)

    fig, ax = plt.subplots(figsize=(8.0, 6.0))
    if target is None or target not in df.columns:
        ax.scatter(components[:, 0], components[:, 1], s=10, alpha=0.6, color="#4C72B0")
    else:
        target_values = df.loc[data.index, target]
        for value in sorted(target_values.dropna().unique()):
            mask = target_values == value
            ax.scatter(components[mask, 0], components[mask, 1], s=10, alpha=0.6, label=str(value))
        ax.legend(title=target, fontsize=8)
    ax.set_title("PCA projection")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    _save_fig(pdf, fig)


def main(
    df: pd.DataFrame,
    output_path: str | Path = "data/matches/visualizations.pdf",
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    numeric_cols = _numeric_columns(df)
    target = _select_target(df)
    features = [col for col in numeric_cols if col != target]

    with PdfPages(output_path) as pdf:
        plot_overview(df, pdf)
        if target:
            plot_target_distribution(df, pdf, target)

        plot_univariate(df, pdf, features)

        top_by_corr = []
        if target:
            top_by_corr = plot_bivariate_correlations(df, pdf, target, features)
            plot_bivariate_boxplots(df, pdf, target, top_by_corr[:12])

        scatter_features = top_by_corr[:6] if top_by_corr else features[:6]
        if len(scatter_features) >= 2:
            plot_bivariate_scatter(df, pdf, target, scatter_features)

        heatmap_features = top_by_corr[:20] if top_by_corr else features[:20]
        if len(heatmap_features) >= 2:
            plot_correlation_heatmap(df, pdf, heatmap_features)

        plot_pca_projection(df, pdf, features, target)

    return output_path


if __name__ == "__main__":
    data_path = Path("data/matches/preprocessed_train.csv")
    df = pd.read_csv(data_path)
    pdf_path = main(df)
    print(f"Saved visualizations to {pdf_path}")
