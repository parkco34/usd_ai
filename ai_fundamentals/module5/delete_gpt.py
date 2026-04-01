#!/usr/bin/env python
# %% [markdown]
# # Assignment 5.2: Time Series Forecasting with Nixtla
#
# This notebook analyzes the classic AirPassengers time series using Nixtla's
# StatsForecast workflow. It includes:
#
# 1. Data loading and Nixtla long-format preparation
# 2. Visualization analysis
# 3. STL decomposition
# 4. AutoARIMA (SARIMA-style automatic seasonal ARIMA) and AutoETS modeling
# 5. One-step and three-step rolling forecast evaluation
# 6. Residual diagnostics
# 7. Interpretation and multivariate extension ideas

# %%
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pandas.plotting import lag_plot
from textwrap import dedent

from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, AutoETS

from statsmodels.tsa.seasonal import STL
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def read_file(path):
    """
    Reads file depending on file extension.
    --------------------------------------------
    INPUT:
        path: (str) Path (absolute/relative) to data file.

    OUTPUT:
        dframe: (pd.DataFrame) Loaded dataframe
    """
    if not isinstance(path, str):
        raise ValueError("Path provided is in wrong format!\nMust be a string")

    try:
        if path.endswith("csv") or path.endswith("txt"):
            dframe = pd.read_csv(path)

        elif path.endswith("dat"):
            dframe = pd.read_csv(path, sep=r"\s+")

        else:
            raise ValueError("Unsupported file extension")

    except Exception as err:
        raise RuntimeError(f"OOPZ!\n{err}")

    return dframe


def convert_to_datetimeindex(dataframe, column, remove=False):
    """
    Converts the index into a datetime index from the chosen column.
    ---------------------------------------------------------------
    INPUT:
        dataframe: (pd.DataFrame) Data
        column: (str) Column name to make datetime index
        remove: (bool; default=False) Whether to remove the original column

    OUTPUT:
        dframe: (pd.DataFrame) Dataframe with a datetime index
    """
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("dataframe must be a pandas DataFrame")

    if not isinstance(column, str):
        raise TypeError("column must be a string")

    if column not in dataframe.columns:
        raise KeyError(f"'{column}' not found in dataframe columns")

    df = dataframe.copy()
    df["dates"] = pd.to_datetime(df[column])
    df.set_index("dates", inplace=True)

    if remove:
        df.drop(column, axis=1, inplace=True)

    return df


def mae(y_true, y_pred):
    """Mean absolute error."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return np.mean(np.abs(y_true - y_pred))


def rmse(y_true, y_pred):
    """Root mean squared error."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def mape(y_true, y_pred):
    """Mean absolute percentage error."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def summarize_metrics(dataframe, actual_col="actual", pred_col="pred"):
    """
    Computes MAE, RMSE, and MAPE for a forecast dataframe.
    """
    return pd.Series(
        {
            "MAE": mae(dataframe[actual_col], dataframe[pred_col]),
            "RMSE": rmse(dataframe[actual_col], dataframe[pred_col]),
            "MAPE": mape(dataframe[actual_col], dataframe[pred_col]),
        }
    )


def rolling_origin_evaluation(dataframe, horizon, test_size=12, season_length=12, freq="MS"):
    """
    Rolling-origin evaluation using Nixtla models.

    ----------------------------------------------------
    INPUT:
        dataframe: (pd.DataFrame) Must contain ['unique_id', 'ds', 'y']
        horizon: (int) Forecast horizon
        test_size: (int) Number of final observations reserved for rolling evaluation
        season_length: (int) Seasonal cycle length
        freq: (str) Time frequency for StatsForecast

    OUTPUT:
        pred_df: (pd.DataFrame) Long-format actual vs predicted values
        metric_df: (pd.DataFrame) Summary metrics by model
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")

    df = dataframe.copy().reset_index(drop=True)

    start_test = len(df) - test_size
    last_start = len(df) - horizon

    rows = []

    for split_idx in range(start_test, last_start + 1):
        train_df = df.iloc[:split_idx].copy()
        valid_df = df.iloc[split_idx:split_idx + horizon].copy()

        models = [
            AutoARIMA(season_length=season_length),
            AutoETS(season_length=season_length),
        ]

        sf = StatsForecast(models=models, freq=freq, n_jobs=-1)
        fcst = sf.forecast(df=train_df, h=horizon)

        merged = valid_df[["ds", "y"]].merge(fcst, on="ds", how="left")

        for model_name in ["AutoARIMA", "AutoETS"]:
            block = merged[["ds", "y", model_name]].copy()
            block.rename(columns={"y": "actual", model_name: "pred"}, inplace=True)
            block["model"] = model_name
            block["cutoff"] = train_df["ds"].max()
            block["horizon"] = horizon
            rows.append(block)

    pred_df = pd.concat(rows, axis=0, ignore_index=True)

    metric_df = (
        pred_df.groupby("model")
        .apply(summarize_metrics)
        .reset_index()
        .sort_values("RMSE", ascending=True)
        .reset_index(drop=True)
    )

    return pred_df, metric_df


def choose_best_model(metric_df):
    """
    Choose best model by lowest RMSE, then MAE, then MAPE.
    """
    ordered = metric_df.sort_values(["RMSE", "MAE", "MAPE"], ascending=True).reset_index(drop=True)
    return ordered.loc[0, "model"]


# %% [markdown]
# ## 1. Load the AirPassengers data and convert to Nixtla long format

# %%
url = "https://datasets-nixtla.s3.amazonaws.com/air-passengers.csv"
df = pd.read_csv(url, parse_dates=["ds"])

# Keep exact Nixtla ordering
df = df[["unique_id", "ds", "y"]]

print(df.head())
print("\nShape:", df.shape)
print("\nDtypes:\n", df.dtypes)

# Optional datetime-indexed copy for visualization/statsmodels work
ts_df = convert_to_datetimeindex(df, "ds", remove=False)

# %% [markdown]
# ## 2. Initial visualization analysis

# %%
plt.figure(figsize=(12, 6))
plt.plot(df["ds"], df["y"], linewidth=2)
plt.title("AirPassengers Time Series")
plt.xlabel("Date")
plt.ylabel("Passengers")
plt.tight_layout()
plt.show()

# %%
plt.figure(figsize=(12, 6))
plt.scatter(df["ds"], df["y"], s=35)
plt.title("Scatterplot of AirPassengers Over Time")
plt.xlabel("Date")
plt.ylabel("Passengers")
plt.tight_layout()
plt.show()

# %%
# Seasonal plot: month-by-month trajectories for each year
seasonal_df = df.copy()
seasonal_df["year"] = seasonal_df["ds"].dt.year
seasonal_df["month_num"] = seasonal_df["ds"].dt.month
seasonal_df["month"] = seasonal_df["ds"].dt.strftime("%b")

pivot_seasonal = seasonal_df.pivot(index="month_num", columns="year", values="y")

plt.figure(figsize=(12, 6))
for year in pivot_seasonal.columns:
    plt.plot(
        pivot_seasonal.index,
        pivot_seasonal[year],
        marker="o",
        linewidth=1.5,
        label=str(year)
    )

plt.xticks(
    ticks=np.arange(1, 13),
    labels=["Jan", "Feb", "Mar", "Apr", "May", "Jun",
            "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
)
plt.title("Seasonal Plot: Monthly Passenger Counts by Year")
plt.xlabel("Month")
plt.ylabel("Passengers")
plt.legend(ncol=3, fontsize=8)
plt.tight_layout()
plt.show()

# %%
plt.figure(figsize=(6, 6))
lag_plot(df["y"], lag=1)
plt.title("Lag Plot (lag = 1)")
plt.tight_layout()
plt.show()

# %%
fig, axes = plt.subplots(2, 1, figsize=(12, 8))
plot_acf(df["y"], lags=24, ax=axes[0])
axes[0].set_title("ACF: AirPassengers")
plot_pacf(df["y"], lags=24, method="ywm", ax=axes[1])
axes[1].set_title("PACF: AirPassengers")
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Commentary
#
# The AirPassengers series shows a very strong upward trend over time, which means the process is
# not stationary in level. Passenger counts in the late 1950s are much higher than in the early 1950s,
# and the seasonal swings also become larger as the level rises. This suggests multiplicative-type
# seasonality: as the baseline increases, the size of the seasonal oscillation increases as well.
#
# The seasonal plot shows a highly repeatable yearly pattern. Passenger counts tend to rise into the
# summer months and remain relatively elevated near year-end, while the early months of the year
# are typically lower. That recurring structure is exactly why a seasonal period of
# $m = 12$ months is appropriate for both ETS and seasonal ARIMA-style modeling.
#
# The lag plot shows strong positive dependence, and the ACF remains large for many lags, which is
# consistent with persistent trend and seasonality. The ACF also shows spikes near multiples of 12,
# reinforcing annual seasonality. The PACF indicates that low-order autoregressive structure may help,
# but the series clearly needs a model family that can handle both trend and seasonal dependence.

# %% [markdown]
# ## 3. STL decomposition

# %%
stl = STL(ts_df["y"], period=12, robust=True)
stl_result = stl.fit()

fig = stl_result.plot()
fig.set_size_inches(12, 8)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Decomposition interpretation
#
# The STL decomposition separates the series into trend, seasonal, and remainder components.
# The trend component rises steadily across the sample, confirming long-run growth in airline demand.
# The seasonal component repeats on a 12-month cycle and is quite stable in timing, even though the
# raw series itself grows over time.
#
# The remainder component is relatively small compared with the overall signal, which means most of
# the variation is being explained by the combined trend and seasonal structure. Conceptually, this is
# a good sign for forecasting because it indicates the series contains strong systematic structure rather
# than being dominated by noise.

# %% [markdown]
# ## 4. Train / test split
#
# We reserve the final 12 months as a holdout period:
#
# $$
# \text{train} = \{1, \dots, T-12\}, \qquad
# \text{test} = \{T-11, \dots, T\}.
# $$

# %%
test_size = 12
season_length = 12
freq = "MS"

train_df = df.iloc[:-test_size].copy()
test_df = df.iloc[-test_size:].copy()

print("Train observations:", len(train_df))
print("Test observations :", len(test_df))
print("Train end date    :", train_df["ds"].max().date())
print("Test start date   :", test_df["ds"].min().date())

# %% [markdown]
# ## 5. Fit Nixtla models on training data and forecast 12 months ahead

# %%
models = [
    AutoARIMA(season_length=season_length),
    AutoETS(season_length=season_length),
]

sf = StatsForecast(models=models, freq=freq, n_jobs=-1)
holdout_fcst = sf.forecast(df=train_df, h=test_size)

comparison = test_df[["ds", "y"]].merge(holdout_fcst, on="ds", how="left")
comparison.head()

# %%
plt.figure(figsize=(12, 6))
plt.plot(train_df["ds"], train_df["y"], label="Train", linewidth=2)
plt.plot(test_df["ds"], test_df["y"], label="Test / Actual", linewidth=2)
plt.plot(comparison["ds"], comparison["AutoARIMA"], label="AutoARIMA Forecast", linestyle="--")
plt.plot(comparison["ds"], comparison["AutoETS"], label="AutoETS Forecast", linestyle="--")
plt.axvline(test_df["ds"].min(), color="gray", linestyle=":")
plt.title("Train/Test Forecast Comparison")
plt.xlabel("Date")
plt.ylabel("Passengers")
plt.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 6. One-step rolling forecast evaluation
#
# For one-step forecasting, each prediction targets
#
# $$
# \hat{y}_{t+1 \mid t}.
# $$
#
# We roll through the holdout region and re-fit at each cutoff.

# %%
pred_1, metrics_1 = rolling_origin_evaluation(
    dataframe=df,
    horizon=1,
    test_size=test_size,
    season_length=season_length,
    freq=freq
)

print("One-step forecast metrics")
print(metrics_1)

# %% [markdown]
# ## 7. Three-step rolling forecast evaluation
#
# For three-step forecasting, each model predicts
#
# $$
# \hat{y}_{t+1 \mid t}, \hat{y}_{t+2 \mid t}, \hat{y}_{t+3 \mid t}.
# $$
#
# We then aggregate all three-step rolling predictions across the holdout region.

# %%
pred_3, metrics_3 = rolling_origin_evaluation(
    dataframe=df,
    horizon=3,
    test_size=test_size,
    season_length=season_length,
    freq=freq
)

print("Three-step forecast metrics")
print(metrics_3)

# %% [markdown]
# ## 8. Final model choice from the evaluation metrics

# %%
best_1 = choose_best_model(metrics_1)
best_3 = choose_best_model(metrics_3)

print(f"Best one-step model : {best_1}")
print(f"Best three-step model: {best_3}")

# %%
print(
    dedent(
        f"""
        Interpretation guide:

        For the one-step-ahead problem, the preferred model is {best_1} because it has
        the lowest out-of-sample error under the ranking rule:
            1) RMSE
            2) MAE
            3) MAPE

        For the three-step-ahead problem, the preferred model is {best_3} by the same
        criterion. In practice, one-step performance reflects very short-run adaptation,
        while three-step performance tests whether the model keeps its structure when the
        forecast horizon extends further into the future.
        """
    )
)

# %% [markdown]
# ### Short written justification
#
# The final choice for the one-step forecast should go to the model with the smallest
# out-of-sample RMSE, since RMSE penalizes larger misses more strongly:
#
# $$
# \text{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}.
# $$
#
# I use MAE and MAPE as supporting metrics:
#
# $$
# \text{MAE} = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|,
# \qquad
# \text{MAPE} = \frac{100}{n}\sum_{i=1}^{n}\left|\frac{y_i-\hat{y}_i}{y_i}\right|.
# $$
#
# RMSE is the primary decision metric because larger forecast misses are especially undesirable in
# operational forecasting. MAE keeps the interpretation in original passenger units, while MAPE gives
# a scale-free percentage error that is easy to communicate.

# %% [markdown]
# ## 9. Residual diagnostics on the 12-step holdout forecast

# %%
for model_name in ["AutoARIMA", "AutoETS"]:
    comparison[f"{model_name}_resid"] = comparison["y"] - comparison[model_name]

    print(f"\nResidual summary for {model_name}")
    print(comparison[f"{model_name}_resid"].describe())

    lb_lag = min(6, len(comparison[f"{model_name}_resid"]) - 1)
    if lb_lag >= 1:
        lb_test = acorr_ljungbox(
            comparison[f"{model_name}_resid"],
            lags=[lb_lag],
            return_df=True
        )
        print("\nLjung-Box test")
        print(lb_test)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(comparison["ds"], comparison[f"{model_name}_resid"], marker="o")
    axes[0].axhline(0, color="black", linestyle="--")
    axes[0].set_title(f"{model_name} Residuals")
    axes[0].set_xlabel("Date")
    axes[0].set_ylabel("Residual")

    plot_acf(comparison[f"{model_name}_resid"], lags=min(6, len(comparison[f"{model_name}_resid"]) - 1), ax=axes[1])
    axes[1].set_title(f"{model_name} Residual ACF")

    plt.tight_layout()
    plt.show()

# %% [markdown]
# ### Residual interpretation
#
# A good forecast model should leave residuals that look approximately like white noise:
# no clear trend, no strong remaining autocorrelation, and no obvious change in variance.
# If the residual ACF still shows visible structure, then some time dependence remains unmodeled.
#
# The Ljung-Box test provides a formal check for remaining autocorrelation. If the p-value is large,
# that supports the claim that the residuals are approximately uncorrelated. If the p-value is small,
# it suggests leftover structure and therefore room for model improvement.

# %% [markdown]
# ## 10. Multivariate extension: two useful added features
#
# If I were extending this into a multivariate forecasting problem, the first feature I would add is a
# macro-demand indicator such as real disposable income, GDP growth, or an industrial production index.
# Airline demand is tied to the economic capacity of households and firms to travel. When the economy is
# stronger, both vacation travel and business travel tend to increase, so this kind of regressor could help
# explain part of the long-run growth now being absorbed only by the trend component.
#
# The second feature I would add is a calendar or event-effect regressor, such as a holiday / vacation-season
# indicator. The AirPassengers series clearly has recurring within-year peaks, especially around summer and
# year-end travel. Although ETS and SARIMA can capture seasonality statistically, an explicit calendar feature
# would let a multivariate model represent those demand surges more directly. In other words, instead of only
# saying "month 7 tends to be high," the model could say "travel demand rises during known vacation and holiday
# periods," which is closer to the real-world mechanism.

# %% [markdown]
# ## 11. Optional compact tables for the write-up

# %%
print("\nOne-step predictions (head)")
print(pred_1.head())

print("\nThree-step predictions (head)")
print(pred_3.head())

# %% [markdown]
# ## 12. AI use disclosure
#
# AI Use Disclosure:
# I used ChatGPT to help structure, refine, and debug the Python workflow for this assignment.
# I reviewed, edited, and interpreted the code and written responses myself before submission.

