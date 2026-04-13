import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import ceil

# Dark plotting style
plt.style.use("dark_background")


def read_file(file):
    """
    Read the dataset from a CSV/text file into a pandas DataFrame.
    --------------------------------------------------------------
    INPUT:
        file: (str) Path to the dataset file.

    OUTPUT:
        df: (pd.DataFrame) Loaded dataset.
    """
    return pd.read_csv(file)


def missing_values(data):
    """
    Count the total number of missing values in the dataset.
    -------------------------------------------------------
    INPUT:
        data: (pd.DataFrame) Input dataset.

    OUTPUT:
        missing: (int) Total number of missing entries.
    """
    missing = data.isnull().sum().sum()
    return missing


def convert_data_numpy(dataframe):
    """
    Convert predictors and target into NumPy arrays.
    ------------------------------------------------
    Assumes the final column is the target variable.

    INPUT:
        dataframe: (pd.DataFrame) Input dataset.

    OUTPUT:
        X: (np.ndarray) Predictor matrix.
        y: (np.ndarray) Target vector.
    """
    X = np.array(dataframe[dataframe.columns[:-1]])
    y = np.array(dataframe[dataframe.columns[-1]])
    return X, y


def summarize_categorical_data(df):
    """
    Print a more useful summary for categorical data than default describe().
    ------------------------------------------------------------------------
    For each column, report:
        - data type
        - number of unique values
        - most frequent category
        - frequency of the most frequent category

    INPUT:
        df: (pd.DataFrame) Input dataset.
    """
    summary = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "unique_values": df.nunique(),
        "most_frequent": df.mode().iloc[0],
        "freq_of_mode": df.apply(lambda col: col.value_counts().iloc[0])
    })

    print("\nCATEGORICAL DATA SUMMARY")
    print(summary)


def plot_class_balance(target, target_name="Gay"):
    """
    Plot the class balance of the target variable.
    ----------------------------------------------
    INPUT:
        target: (array-like) Target labels.
        target_name: (str) Name of the target variable.
    """
    target_series = pd.Series(target)
    counts = target_series.value_counts().sort_index()

    plt.figure(figsize=(8, 6))
    plt.bar(counts.index.astype(str), counts.values)

    plt.xlabel(target_name)
    plt.ylabel("Count")
    plt.title(f"Class Balance of {target_name}")
    plt.grid(color="gray", axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()


def plot_feature_distributions(df, target_col):
    """
    Plot value counts for each predictor column.
    -------------------------------------------
    Since the features are categorical/ordinal, bar charts are appropriate.

    INPUT:
        df: (pd.DataFrame) Input dataset.
        target_col: (str) Name of the target column.
    """
    feature_cols = [col for col in df.columns if col != target_col]
    n_features = len(feature_cols)

    ncols = 2
    nrows = ceil(n_features / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows))
    axes = np.array(axes).reshape(-1)

    for i, col in enumerate(feature_cols):
        counts = df[col].value_counts().sort_index()
        axes[i].bar(counts.index.astype(str), counts.values)
        axes[i].set_title(f"{col} Distribution")
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("Count")
        axes[i].grid(color="gray", axis="y", linestyle="--", alpha=0.4)

    # Hide unused subplot axes if feature count is odd
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()


def plot_feature_vs_target(df, target_col):
    """
    Plot normalized stacked bar charts showing feature value vs target breakdown.
    ----------------------------------------------------------------------------
    For each feature value, this answers:
        "What fraction of cases are each target class?"

    This is especially useful for decision trees because it previews
    which features separate the target well.

    INPUT:
        df: (pd.DataFrame) Input dataset.
        target_col: (str) Name of the target column.
    """
    feature_cols = [col for col in df.columns if col != target_col]
    n_features = len(feature_cols)

    ncols = 2
    nrows = ceil(n_features / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows))
    axes = np.array(axes).reshape(-1)

    for i, col in enumerate(feature_cols):
        # Normalize across rows so each feature value sums to 1
        ctab = pd.crosstab(df[col], df[target_col], normalize="index")
        ctab.plot(kind="bar", stacked=True, ax=axes[i])

        axes[i].set_title(f"{col} vs {target_col}")
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("Proportion")
        axes[i].grid(color="gray", axis="y", linestyle="--", alpha=0.4)
        axes[i].legend(title=target_col)

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()


def plot_crosstab_heatmaps(df, target_col):
    """
    Plot heatmaps of feature-target crosstabs.
    -----------------------------------------
    This gives a compact view of raw counts for each feature value
    against each target class.

    INPUT:
        df: (pd.DataFrame) Input dataset.
        target_col: (str) Name of the target column.
    """
    feature_cols = [col for col in df.columns if col != target_col]
    n_features = len(feature_cols)

    ncols = 2
    nrows = ceil(n_features / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4 * nrows))
    axes = np.array(axes).reshape(-1)

    for i, col in enumerate(feature_cols):
        ctab = pd.crosstab(df[col], df[target_col])

        im = axes[i].imshow(ctab.values, aspect="auto")
        axes[i].set_title(f"Heatmap: {col} vs {target_col}")
        axes[i].set_xlabel(target_col)
        axes[i].set_ylabel(col)

        axes[i].set_xticks(range(len(ctab.columns)))
        axes[i].set_xticklabels(ctab.columns.astype(str))

        axes[i].set_yticks(range(len(ctab.index)))
        axes[i].set_yticklabels(ctab.index.astype(str))

        # Annotate each cell with the count
        for row in range(ctab.shape[0]):
            for col_idx in range(ctab.shape[1]):
                axes[i].text(
                    col_idx,
                    row,
                    str(ctab.iloc[row, col_idx]),
                    ha="center",
                    va="center"
                )

        fig.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()


def main():
    df = read_file("data/niblings.txt")
    target_col = "Gay"

    # Basic data info
    print(f"\n{df.shape[0]} rows, {df.shape[1]} columns\n")
    print("DATA INFO")
    df.info()

    # More useful categorical summary
    summarize_categorical_data(df)

    # Missing values
    missing = missing_values(df)
    print(f"\nMissing values: {missing}\n")

    # Convert data to numpy arrays
    X, y = convert_data_numpy(df)
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")

    # EDA plots
    plot_class_balance(y, target_name=target_col)
    plot_feature_distributions(df, target_col=target_col)
    plot_feature_vs_target(df, target_col=target_col)
    plot_crosstab_heatmaps(df, target_col=target_col)


if __name__ == "__main__":
    main()#!/usr/bin/env python

