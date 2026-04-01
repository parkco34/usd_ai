#!/usr/bin/env pytho
# Assignment 5.2 – Time Series Forecasting with Nixtla
# pip install statsforecast statsmodels matplotlib pandas numpy

import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import lag_plot
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, AutoETS
from statsmodels.tsa.seasonal import STL
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


# ── 1. LOAD DATA ──────────────────────────────────────────────────────────────
# Nixtla requires "long format": unique_id | ds | y

air_passengers = [
    112,118,132,129,121,135,148,148,136,119,104,118,
    115,126,141,135,125,149,170,170,158,133,114,140,
    145,150,178,163,172,178,199,199,184,162,146,166,
    171,180,193,181,183,218,230,242,209,191,172,194,
    196,196,236,235,229,243,264,272,237,211,180,201,
    204,188,235,227,234,264,302,293,259,229,203,229,
    242,233,267,269,270,315,364,347,312,274,237,278,
    284,277,317,313,318,374,413,405,355,306,271,306,
    315,301,356,348,355,422,465,467,404,347,305,336,
    340,318,362,348,363,435,491,505,404,359,310,337,
    360,342,406,396,420,472,548,559,463,407,362,405,
    417,391,419,461,472,535,622,606,508,461,390,432
]

df = pd.DataFrame({
    "unique_id": "air",
    "ds": pd.date_range("1949-01-01", periods=144, freq="MS"),
    "y":  air_passengers
})
series = pd.Series(df["y"].values, index=df["ds"])

breakpoint()
# ── 2. VISUALIZATION / EDA ────────────────────────────────────────────────────

# Line plot
plt.figure(figsize=(12,5))
plt.plot(df["ds"], df["y"]); plt.title("AirPassengers 1949–1960")
plt.xlabel("Date"); plt.ylabel("Passengers"); plt.tight_layout(); plt.show()

# Scatter over time
plt.figure(figsize=(12,5))
plt.scatter(df["ds"], df["y"], s=30); plt.title("Scatterplot Over Time")
plt.xlabel("Date"); plt.ylabel("Passengers"); plt.tight_layout(); plt.show()

# Seasonal plot – one line per year
tmp = df.copy()
tmp["year"]  = tmp["ds"].dt.year
tmp["month"] = tmp["ds"].dt.month_name().str[:3]
month_order  = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
pivot        = tmp.pivot(index="month", columns="year", values="y").reindex(month_order)

plt.figure(figsize=(12,6))
for yr in pivot.columns:
    plt.plot(pivot.index, pivot[yr], marker="o", linewidth=1, label=str(yr))
plt.title("Seasonal Plot by Year"); plt.xlabel("Month"); plt.ylabel("Passengers")
plt.legend(ncol=3, fontsize=8); plt.tight_layout(); plt.show()

# Lag plots
for lag in [1, 12]:
    plt.figure(figsize=(6,4))
    lag_plot(df["y"], lag=lag); plt.title(f"Lag Plot (lag={lag})")
    plt.tight_layout(); plt.show()

# ACF / PACF
fig, axes = plt.subplots(2,1, figsize=(12,7))
plot_acf( df["y"], lags=36, ax=axes[0], title="ACF – AirPassengers")
plot_pacf(df["y"], lags=24, ax=axes[1], title="PACF – AirPassengers", method="ywm")
plt.tight_layout(); plt.show()

print("""
VISUAL ANALYSIS
The series shows a clear upward trend and a regular annual seasonal cycle. The seasonal swings grow larger as
the overall level rises — a hallmark of multiplicative seasonality. Lag plots at k=1 and k=12 show strong
positive autocorrelation, especially at lag 12, confirming yearly periodicity. The ACF decays slowly across
many lags (non-stationary; needs differencing) and spikes again at multiples of 12, which directly motivates
seasonal_length=12 in both models. These patterns make AirPassengers an ideal candidate for SARIMA and ETS.
""")


# ── 3. STL DECOMPOSITION ─────────────────────────────────────────────────────
# STL splits the series into Trend + Seasonal + Residual.
# We use period=12 (monthly data, yearly cycle) and robust=True to down-weight outliers.

stl_result = STL(series, period=12, robust=True).fit()
fig = stl_result.plot(); fig.set_size_inches(12,8); plt.tight_layout(); plt.show()

print("""
DECOMPOSITION INTERPRETATION
The trend component rises steadily, confirming long-run passenger growth. The seasonal component repeats
a consistent yearly pattern — summer peaks, winter troughs — and its shape is stable across years. The
remainder is small relative to the other two components, meaning trend and seasonality explain most of the
variance. This clean structure is exactly what SARIMA and ETS exploit.
""")


# ── 4. MODELS ─────────────────────────────────────────────────────────────────
# AutoARIMA: auto-selects SARIMA(p,d,q)(P,D,Q)[12] order by AICc
# AutoETS  : auto-selects ETS(Error,Trend,Seasonal) family by AICc

models = [AutoARIMA(season_length=12), AutoETS(season_length=12)]


# ── 5. ACCURACY METRICS ───────────────────────────────────────────────────────

def mae(a, p):  return np.mean(np.abs(np.asarray(a) - np.asarray(p, float)))
def rmse(a, p): return np.sqrt(np.mean((np.asarray(a) - np.asarray(p, float))**2))
def mape(a, p): a,p = np.asarray(a,float), np.asarray(p,float); return np.mean(np.abs((a-p)/a))*100


# ── 6. ROLLING (EXPANDING-WINDOW) EVALUATION ─────────────────────────────────
# Train on all data up to time t, forecast h steps ahead, record error at t+h.
# Repeat, growing the training window by one month each iteration.
# This mirrors real deployment: you always train on everything you have so far.

def rolling_eval(data, horizon, min_train=120):
    results = []
    for end in range(min_train, len(data) - horizon + 1):
        train  = data.iloc[:end].copy()
        target = data.iloc[end:end+horizon]
        fcst   = StatsForecast(models=models, freq="MS", n_jobs=-1)\
                     .forecast(df=train, h=horizon).reset_index()
        t_row  = fcst.loc[fcst["ds"] == target.iloc[-1]["ds"]].iloc[0]
        results.append({"ds": target.iloc[-1]["ds"], "actual": target.iloc[-1]["y"],
                        "SARIMA": t_row["AutoARIMA"], "ETS": t_row["AutoETS"]})
    return pd.DataFrame(results)

eval_1 = rolling_eval(df, horizon=1)
eval_3 = rolling_eval(df, horizon=3)

def metric_table(ev, label):
    rows = []
    for name, col in [("SARIMA","SARIMA"),("ETS","ETS")]:
        rows.append({"Horizon":label,"Model":name,
                     "MAE":  mae( ev["actual"], ev[col]),
                     "RMSE": rmse(ev["actual"], ev[col]),
                     "MAPE": mape(ev["actual"], ev[col])})
    return pd.DataFrame(rows)

metrics = pd.concat([metric_table(eval_1,"1-step"), metric_table(eval_3,"3-step")])
print("\nFORECAST ACCURACY METRICS")
print(metrics.round(3).to_string(index=False))


# ── 7. NEXT-STEP AND NEXT-3-STEP FORECASTS ───────────────────────────────────

sf_full = StatsForecast(models=models, freq="MS", n_jobs=-1)

for h, label in [(1,"1-STEP"), (3,"3-STEP")]:
    fc = sf_full.forecast(df=df, h=h).reset_index()
    print(f"\nFORECAST — NEXT {label}")
    print(fc[["ds","AutoARIMA","AutoETS"]].round(2).to_string(index=False))


# ── 8. ROLLING FORECAST PLOTS ────────────────────────────────────────────────

for ev, label in [(eval_1,"1-Step"), (eval_3,"3-Step")]:
    plt.figure(figsize=(12,5))
    plt.plot(df["ds"], df["y"], label="Observed", linewidth=1.5)
    plt.plot(ev["ds"], ev["SARIMA"], "--", label=f"SARIMA {label}")
    plt.plot(ev["ds"], ev["ETS"],    "--", label=f"ETS {label}")
    plt.title(f"Rolling {label} Forecast Comparison")
    plt.xlabel("Date"); plt.ylabel("Passengers"); plt.legend(); plt.tight_layout(); plt.show()


# ── 9. MODEL CHOICE JUSTIFICATION ────────────────────────────────────────────

def best_model(metric_df):
    t = metric_df.copy()
    for m in ["MAE","RMSE","MAPE"]: t[f"r_{m}"] = t[m].rank()
    t["avg"] = t[["r_MAE","r_RMSE","r_MAPE"]].mean(axis=1)
    return t.sort_values(["avg","RMSE"]).iloc[0]["Model"]

best_1 = best_model(metrics[metrics["Horizon"]=="1-step"])
best_3 = best_model(metrics[metrics["Horizon"]=="3-step"])

for horizon, best, mdf in [("1-step", best_1, metrics[metrics["Horizon"]=="1-step"]),
                             ("3-step", best_3, metrics[metrics["Horizon"]=="3-step"])]:
    row = mdf.loc[mdf["Model"]==best].iloc[0]
    print(f"""
MODEL CHOICE — {horizon.upper()}
Preferred model: {best}
MAE={row['MAE']:.3f}  RMSE={row['RMSE']:.3f}  MAPE={row['MAPE']:.3f}%

{best} produced the lowest combined error across all three metrics at the {horizon} horizon. At shorter
horizons, the winning model is the one that best tracks the immediate trend and seasonal level. At longer
horizons, the advantage goes to whichever model most stably projects the underlying seasonal structure
several periods ahead. Both models are competitive on this dataset because AirPassengers has very clean
trend-plus-seasonality structure, but the metric rankings break the tie.
""")


# ── 10. MULTIVARIATE DISCUSSION ───────────────────────────────────────────────

print("""
MULTIVARIATE FORECAST DISCUSSION

Feature 1 — Macroeconomic activity (e.g. GDP growth or personal income)
Passenger demand rises and falls with the economy. Business and leisure travel both depend on income and
employment levels. A GDP growth variable would help the model explain demand shifts that go beyond the
historical seasonal pattern, particularly at economic turning points where pure time-series extrapolation
tends to fail.

Feature 2 — Fuel or fare price index
Travel demand is sensitive to price. Jet fuel costs flow through to ticket prices, and higher fares
suppress demand. Adding a pricing variable would help the model distinguish between calendar-driven
seasonality and economically-driven fluctuations, improving both accuracy and interpretability.
""")

