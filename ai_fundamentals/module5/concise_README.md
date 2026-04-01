# Assignment 5.2 — Time Series Forecasting with Nixtla
## Concise Conceptual Guide

---

## The Core Decomposition Idea

Every step in this assignment flows from one equation:

```
y_t = Trend_t  ×  Seasonal_t  ×  Remainder_t     (multiplicative)
y_t = Trend_t  +  Seasonal_t  +  Remainder_t     (additive)
```

**AirPassengers is multiplicative** — seasonal swings grow as the series grows.
This single observation should justify every modeling decision you write about.

---

## Nixtla's Required Format

```python
# DataFrame must have exactly these three columns, in this order:
df = df[['unique_id', 'ds', 'y']]
# unique_id = 'AirPassengers' (string label)
# ds        = datetime column
# y         = observation value (float)
```

---

## Step 1 — Visualization (20%)

Each plot answers one diagnostic question. **Your commentary must link observations to model choice** — that's what separates rubric tiers.

| Plot | Diagnostic Question | What to Look For |
|------|-------------------|-----------------|
| Line plot | What is the overall shape? | Trend, seasonality, growing variance |
| Seasonal plot | Is seasonality consistent year-over-year? | Parallel lines = additive; fanning = multiplicative |
| Lag-1 scatter | Is there autocorrelation? | Tight positive line = strong memory in series |
| ACF | At which lags is correlation significant? | Slow decay = trend; spikes at 12, 24 = annual seasonality |
| PACF | How many lags of *direct* influence? | Sharp cutoff at lag p = AR(p) process |

**ACF/PACF math (for your notes):**
```
rho_k = Cov(y_t, y_{t-k}) / Var(y_t)     <- ACF: total correlation at lag k
PACF_k = correlation after removing effect of lags 1...k-1
```

**Strong comment example:** *"The slow ACF decay indicates non-stationarity; the spike at lag 12 confirms annual seasonality — SARIMA needs d=1, D=1, m=12."*

---

## Step 2 — Decomposition (15%)

Use STL (`statsmodels.tsa.seasonal.STL`) with `period=12`.

After decomposing, check three things:
1. **Trend** — linear growth? exponential?
2. **Seasonal** — stable shape, or changing over time?
3. **Residuals** — should look like random noise with no visible pattern

If residuals still show structure, the model hasn't captured everything.

---

## Step 3 — Models & Evaluation (45%)

### Train-Test Split
Never shuffle time series data. The future cannot inform the past.
```
[------- Train --------][-- Test --]
                        ^ cutoff
```

### SARIMA — What the notation means
```
SARIMA(p, d, q)(P, D, Q)[m=12]
       |  |  |  |  |  |
       |  |  |  |  |  +-- Seasonal MA order
       |  |  |  |  +----- Seasonal differencing (removes seasonal trend)
       |  |  |  +-------- Seasonal AR order
       |  |  +----------- MA order (past errors)
       |  +-------------- Differencing order (removes trend)
       +----------------- AR order (past values)
```
Use `AutoARIMA(season_length=12)` — it searches these parameters automatically using AIC/BIC.

### ETS — What the notation means
```
ETS(Error, Trend, Seasonal)  <- each can be: None / Additive / Multiplicative
```
Exponential smoothing weights recent observations more heavily (like physical decay: w proportional to e^{-lambda*t}).
For AirPassengers, expect AutoETS to select **ETS(M,A,M)** — multiplicative error & seasonal.
Use `AutoETS(season_length=12)`.

### Two Forecast Horizons
Run the full pipeline **twice** — `h=1` and `h=3` are separate experiments.
Errors compound over horizon; the better model at h=1 may not win at h=3.

### Evaluation Metrics
```
MAE  = (1/n) * sum(|y_t - y_hat_t|)                    <- in original units
RMSE = sqrt[(1/n) * sum((y_t - y_hat_t)^2)]            <- penalizes large errors more
MAPE = (100/n) * sum(|(y_t - y_hat_t) / y_t|)          <- scale-free percentage
```

### Residual Diagnostics (Required for top rubric tier)
After fitting, check that residuals look like **white noise**:
- Plot residuals over time — no visible trend or pattern
- ACF of residuals — no significant spikes
- Ljung-Box test — p-value > 0.05 means no autocorrelation remains

---

## Step 4 — Multivariate Features (10%)

Name two external variables you'd add and justify *why they correlate with passenger volume*
and *whether they'd be available at forecast time*.

Strong candidates: **economic indicator (GDP/income)** and **fuel price index**.

---

## Notebook Structure Checklist

```
[ ] Load data -> Nixtla format
[ ] Line, seasonal, scatter, lag, ACF, PACF plots
[ ] Commentary paragraph(s)
[ ] STL decomposition + interpretation
[ ] Train-test split
[ ] h=1 forecast: AutoARIMA + AutoETS -> MAE, RMSE, MAPE
[ ] h=3 forecast: AutoARIMA + AutoETS -> MAE, RMSE, MAPE
[ ] Residual diagnostics on best model
[ ] Model justification
[ ] Multivariate feature discussion
```

---

## Lecture Code vs. Your Code: Key Differences

The lecture gives you the skeleton (Nixtla format, model init, `sf.forecast()` call).
You need to add: decomposition, dual horizons, quantitative metrics, residual checks,
and all written commentary.
