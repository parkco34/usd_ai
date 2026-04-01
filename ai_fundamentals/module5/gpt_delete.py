#!/usr/bin/env python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use("dark_background")  # black theme

from pandas.plotting import lag_plot
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, AutoETS
from statsmodels.tsa.seasonal import STL
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

def read_file(path):
    """
    reads file depending on file extension.
    --------------------------------------------
    INPUT:
        path: (str) Path (absolute/relative) to data file.

    OUTPUT:
        X, y: (tuple of np.arrays) Features and target
    """
    # Input-valldation
    if not isinstance(path, str):
        raise ValueError("Path provided is in wrong format!\nMust be a string")

    try:
        # Case: .csv file or .txt file
        if path.endswith("csv") or path.endswith("txt"):
            dframe = pd.read_csv(path)

        # Case: .dat file
        elif path.endswith("dat"):
            dframe = pd.read_csv(path, sep=r"\s+")

    except Exception as err:
        print(f"OOPZ!\n{err}")

    return dframe

def convert_to_datetimeindex(dataframe, column, remove=False):
    """
    Converts the index into a datetime index from the column representing the dates, while removing the column afterwards.
    ----------------------------------------------------
    INPUT:
        dataframe: (pd.DataFrame) Data
        column: (str) Column name to make datetime index
        remove: (bool; default=False) Whether to remove unnecessaru column or not

    OUTPUT:
        dframe: (pd.DataFrame) Datframe with a datetime index
    """
    # Copy dataframe for safety
    df = dataframe.copy()

    # 1) Convert column to datetime objects
    df["dates"] = pd.to_datetime(df["ds"])

    # 2) Set datetime column as index
    df.set_index("dates", inplace=True)

    if remove:
        # Remove unnecessary column
        df.drop("ds", axis=1, inplace=True)

    return df


if __name__ == "__main__":
    dframe = read_file("https://datasets-nixtla.s3.amazonaws.com/air-passengers.csv")

    # Get datetimeindex via dates
    df = convert_to_datetimeindex(dframe, "ds")

    # ======= Line plot =========
    plt.figure(figsize=(12, 5))
    plt.plot(df.index, df["y"], color="dodgerblue", linewidth=2)
    plt.title("Air Passengers: 1949-1960")
    plt.xlabel("Date")
    plt.ylabel("Passengers")
    plt.tight_layout()
    plt.grid(color="gray")
    plt.show()

    # ========= Scatterplot ========
    plt.figure(figsize=(12, 5))
    plt.scatter(df.index, df["y"], c="lime")
    plt.title("Scatterplot Over Time")
    plt.xlabel("Dates")
    plt.ylabel("Passengers")
    plt.tight_layout()
    plt.grid(color="gray", linestyle="--")
    plt.show()

    # ====== Seasonal Plot =======
    # One line per year
    plt.figure(figsize=(12, 6))

    years = df.index.year.unique()

    for year in years:
        mask = df.index.year == year
        plt.plot(
            df.index.month[mask],
            df.loc[mask, "y"],
            marker="o",
            linewidth=1.5,
            label=str(year)
        )

    plt.title("Seasonal Plot: Air Passengers by Year")
    plt.xlabel("Month")
    plt.ylabel("Passengers")
    plt.xticks(
        ticks=np.arange(1, 13),
        labels=["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    )
    plt.legend(ncol=3, fontsize=8)
    plt.tight_layout()
    plt.grid(color="gray", linestyle=":")
    plt.show()

    # ===== Lag Plot =====
    plt.figure(figsize=(6, 6))
    lag_plot(df["y"])
    plt.title("Lag Plot")
    plt.tight_layout()
    plt.grid(color="gray", linestyle=":")
    plt.show()

    # ===== ACF / PACF =====
    fig, ax = plt.subplots(2, 1, figsize=(12, 8))

    plot_acf(df["y"], lags=24, ax=ax[0])
    ax[0].set_title("Autocorrelation Function (ACF)")

    plot_pacf(df["y"], lags=24, ax=ax[1], method="ywm")
    ax[1].set_title("Partial Autocorrelation Function (PACF)")

    plt.tight_layout()
    plt.show()

    # ===== STL Decomposition =====
    stl = STL(df["y"], period=12)
    result = stl.fit()

    fig = result.plot()
    fig.set_size_inches(12, 8)
    plt.tight_layout()
    plt.show()

    # ===== Train / Test Split =====
    # Last 12 months for test set
    train_df = dframe.iloc[:-12].copy()
    test_df = dframe.iloc[-12:].copy()

    # Nixtla wants columns: unique_id, ds, y
    train_df = train_df[["unique_id", "ds", "y"]]
    test_df = test_df[["unique_id", "ds", "y"]]

    # ===== Fit models =====
    models = [
        AutoETS(season_length=12),
        AutoARIMA(season_length=12)
    ]

    sf = StatsForecast(
        models=models,
        freq="M",
        n_jobs=-1
    )

    # ===== 12-step forecast =====
    forecasts_12 = sf.forecast(df=train_df, h=12)

    # Merge forecasts with actual test values
    results_12 = test_df.merge(forecasts_12, on=["unique_id", "ds"], how="left")

    # ===== Plot forecasts =====
    plt.figure(figsize=(12, 6))
    plt.plot(train_df["ds"], train_df["y"], label="Train", color="dodgerblue")
    plt.plot(test_df["ds"], test_df["y"], label="Actual", color="lime")
    plt.plot(results_12["ds"], results_12["AutoETS"], label="AutoETS Forecast", linestyle="--", color="orange")
    plt.plot(results_12["ds"], results_12["AutoARIMA"], label="AutoARIMA Forecast", linestyle="--", color="red")
    plt.axvline(test_df["ds"].iloc[0], color="gray", linestyle=":")
    plt.title("12-Step Forecast: AutoETS vs AutoARIMA")
    plt.xlabel("Date")
    plt.ylabel("Passengers")
    plt.legend()
    plt.tight_layout()
    plt.grid(color="gray", linestyle=":")
    plt.show()

    # ===== Metrics =====
    def mae(y_true, y_pred):
        return np.mean(np.abs(y_true - y_pred))

    def rmse(y_true, y_pred):
        return np.sqrt(np.mean((y_true - y_pred) ** 2))

    def mape(y_true, y_pred):
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    print("\n===== 12-Step Forecast Metrics =====")
    for model in ["AutoETS", "AutoARIMA"]:
        print(f"\n{model}")
        print("MAE :", round(mae(results_12["y"], results_12[model]), 3))
        print("RMSE:", round(rmse(results_12["y"], results_12[model]), 3))
        print("MAPE:", round(mape(results_12["y"], results_12[model]), 3))

    # ===== 1-step ahead forecast =====
    # Use all but last 1 observation as training
    train_1 = dframe.iloc[:-1].copy()
    test_1 = dframe.iloc[-1:].copy()

    sf1 = StatsForecast(
        models=[
            AutoETS(season_length=12),
            AutoARIMA(season_length=12)
        ],
        freq="M",
        n_jobs=-1
    )

    forecast_1 = sf1.forecast(df=train_1, h=1)
    results_1 = test_1.merge(forecast_1, on=["unique_id", "ds"], how="left")

    print("\n===== 1-Step Forecast Metrics =====")
    for model in ["AutoETS", "AutoARIMA"]:
        print(f"\n{model}")
        print("MAE :", round(mae(results_1["y"], results_1[model]), 3))
        print("RMSE:", round(rmse(results_1["y"], results_1[model]), 3))
        print("MAPE:", round(mape(results_1["y"], results_1[model]), 3))

    # ===== 3-step ahead forecast =====
    # Use all but last 3 observations as training
    train_3 = dframe.iloc[:-3].copy()
    test_3 = dframe.iloc[-3:].copy()

    sf3 = StatsForecast(
        models=[
            AutoETS(season_length=12),
            AutoARIMA(season_length=12)
        ],
        freq="M",
        n_jobs=-1
    )

    forecast_3 = sf3.forecast(df=train_3, h=3)
    results_3 = test_3.merge(forecast_3, on=["unique_id", "ds"], how="left")

    print("\n===== 3-Step Forecast Metrics =====")
    for model in ["AutoETS", "AutoARIMA"]:
        print(f"\n{model}")
        print("MAE :", round(mae(results_3["y"], results_3[model]), 3))
        print("RMSE:", round(rmse(results_3["y"], results_3[model]), 3))
        print("MAPE:", round(mape(results_3["y"], results_3[model]), 3))

    # ===== Residual checks for 12-step forecast =====
    results_12["ETS_resid"] = results_12["y"] - results_12["AutoETS"]
    results_12["ARIMA_resid"] = results_12["y"] - results_12["AutoARIMA"]

    plt.figure(figsize=(12, 5))
    plt.plot(results_12["ds"], results_12["ETS_resid"], marker="o", label="ETS Residuals")
    plt.plot(results_12["ds"], results_12["ARIMA_resid"], marker="o", label="ARIMA Residuals")
    plt.axhline(0, color="white", linestyle="--")
    plt.title("Residuals: Forecast Errors on Test Set")
    plt.xlabel("Date")
    plt.ylabel("Residual")
    plt.legend()
    plt.tight_layout()
    plt.grid(color="gray", linestyle=":")
    plt.show()

    fig, ax = plt.subplots(2, 1, figsize=(12, 8))

    plot_acf(results_12["ETS_resid"], lags=min(10, len(results_12["ETS_resid"]) - 1), ax=ax[0])
    ax[0].set_title("ACF of ETS Residuals")

    plot_acf(results_12["ARIMA_resid"], lags=min(10, len(results_12["ARIMA_resid"]) - 1), ax=ax[1])
    ax[1].set_title("ACF of ARIMA Residuals")

    plt.tight_layout()
    plt.show()

    # ===== Quick printed interpretation =====
    print("\n===== Interpretation Notes =====")
    print("1. The line plot and scatterplot should show upward trend.")
    print("2. The seasonal plot should show repeated yearly seasonality.")
    print("3. The lag plot should show strong positive dependence.")
    print("4. The ACF should decay slowly because the series is highly autocorrelated.")
    print("5. STL should separate trend, seasonality, and remainder.")
    print("6. Compare AutoETS and AutoARIMA using MAE, RMSE, and MAPE.")
    print("7. Better residuals should look more random and centered around zero.")

