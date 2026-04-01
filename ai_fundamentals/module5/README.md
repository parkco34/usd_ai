# Assignment 5.2 — Time Series Forecasting with Nixtla
## Conceptual Study Guide & Step-by-Step Instructions

> **Philosophy of this guide:** Before writing a single line of code, understand *why* you're writing it.
> Every step below first explains the concept mathematically, then tells you what to implement.
> The lecture code (milk production dataset) is your structural template — you are swapping the
> dataset and extending the analysis significantly.

---

## Mental Model: What Is a Time Series?

A time series is a sequence of observations indexed by time:

```
y_t  where t = 1, 2, 3, ..., T
```

Unlike i.i.d. samples in classical statistics, **adjacent observations are correlated**.
This violates the independence assumption most ML models rely on — which is exactly why
time series has its own toolkit.

The fundamental decomposition assumption (used throughout this assignment) is:

```
y_t = Trend_t + Seasonal_t + Remainder_t        (additive)
y_t = Trend_t × Seasonal_t × Remainder_t        (multiplicative)
```

Choose **additive** when seasonal fluctuations are roughly constant in magnitude.
Choose **multiplicative** when seasonal fluctuations grow proportionally with the level
(i.e., bigger swings as the series grows). AirPassengers is a classic multiplicative case —
keep this in mind for every modeling decision you make.

---

## The Nixtla Long Format (Critical Setup)

Nixtla's `StatsForecast` library requires a specific "long format" DataFrame with exactly these columns:

| Column | Role | Type |
|--------|------|------|
| `unique_id` | Series identifier (for multiple series) | string |
| `ds` | Timestamp / date | datetime |
| `y` | Observation value | float |

**Why this format?** It scales to panel data (many series at once). Even with one series,
you must comply. This is your first implementation task.

The AirPassengers dataset lives here:
```
https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv
```

**Conceptual task:** Load it, rename columns to `ds` and `y`, parse the dates, and add
`unique_id = 'AirPassengers'`. Reorder to `[unique_id, ds, y]`.

---

## STEP 1 — Visualization Analysis (20% of grade)

This step is not just "make pretty plots." Each visualization answers a specific diagnostic question.
Work through them in this order, because each one builds intuition for the next.

### 1a. Line Plot (Time Plot)

**Question it answers:** What is the overall shape of the series?

Look for:
- **Trend** — does the mean drift upward or downward over time?
- **Seasonality** — are there regular, periodic patterns?
- **Variance non-stationarity** — do the oscillations get larger over time?

For AirPassengers, you should see all three. The growing amplitude of seasonal peaks
is the key observation that motivates the multiplicative model.

**Implementation hint:** Use `matplotlib`. Plot `ds` on x-axis, `y` on y-axis.

---

### 1b. Seasonal Plot

**Question it answers:** Is the seasonal pattern consistent across years?

A seasonal plot overlays each year as a separate line, with months on the x-axis.
This lets you directly compare January 1950 vs January 1955 vs January 1960, etc.

**Conceptual construction:**
- Extract `year` and `month` from `ds`
- Pivot so rows = months (1–12), columns = years
- Plot each column (year) as a separate line

If the lines are parallel (same shape, different heights), the pattern is additive.
If they fan outward (same shape but growing amplitude), it's multiplicative.

---

### 1c. Scatterplot (Lag-1 Scatter)

**Question it answers:** How strongly is today's value correlated with yesterday's?

Plot `y_t` (x-axis) vs `y_{t+1}` (y-axis). A tight positive linear relationship
indicates strong autocorrelation — the series has "memory."

For AirPassengers, you'll see a very tight positive relationship, which confirms
that past values are highly predictive of future values.

---

### 1d. Lag Plot (Multiple Lags)

**Question it answers:** At which lags does autocorrelation appear most strongly?

A lag plot at lag `k` plots `y_t` vs `y_{t-k}`.

```
Lag 1:   (y_1, y_2), (y_2, y_3), ...
Lag 12:  (y_1, y_13), (y_2, y_14), ...   ← monthly seasonality shows up here
```

For monthly data with annual seasonality, **lag 12 should show a strong positive
linear relationship** — observations 12 months apart are very similar.

**Pandas hint:** `pd.plotting.lag_plot()` handles this for single-lag views.
For multiple lags, use a subplot grid.

---

### 1e. ACF — Autocorrelation Function

**The math:**

The autocorrelation at lag `k` is:

```
ρ_k = Cov(y_t, y_{t-k}) / Var(y_t)
    = [ Σ(y_t - ȳ)(y_{t-k} - ȳ) ] / [ Σ(y_t - ȳ)² ]
```

This measures the **linear correlation** between the series and a lagged version of itself.
Values range from -1 to +1.

**What to look for in a seasonal series:**
- Slow decay of ACF at lags 1, 2, 3, ... → trend (non-stationarity)
- Spikes at lag 12, 24, 36, ... → annual seasonality (monthly data)
- Both together → you have a trending, seasonal series → need both differencing and seasonal differencing for ARIMA

**Implementation:** `statsmodels.graphics.tsaplots.plot_acf()`

---

### 1f. PACF — Partial Autocorrelation Function

**The math:**

The partial autocorrelation at lag `k` measures the correlation between `y_t` and `y_{t-k}`
**after removing the linear effect of the intermediate lags** 1, 2, ..., k-1.

Think of it as: "How much does lag 12 matter, *after* accounting for lags 1–11?"

**Why this matters for model selection:**
- In a pure AR(p) process: PACF cuts off sharply after lag p
- In a pure MA(q) process: ACF cuts off sharply after lag q
- These cutoff patterns guide ARIMA order selection

For AirPassengers, you'll likely see a slowly decaying ACF and a sharp PACF cutoff —
the hallmarks of an AR-dominated process on top of seasonality.

**Implementation:** `statsmodels.graphics.tsaplots.plot_pacf()`

---

### Commentary Requirements (2–3 paragraphs)

Your written commentary must:
1. Describe what patterns you see (trend, seasonality, variance growth)
2. Interpret the ACF/PACF in terms of what they imply about model structure
3. **Link patterns to model choice** — this is what separates "Meets Expectations"
   from "Approaches Expectations" in the rubric

Example of a weak comment: *"The ACF shows autocorrelation."*
Example of a strong comment: *"The slow decay of the ACF at all lags indicates the series
is non-stationary in the mean, suggesting first-order differencing is needed. The spike
at lag 12 in both ACF and PACF points to a seasonal AR(1) component — consistent with
SARIMA(p,d,q)(P,D,Q)[12] with P≥1."*

---

## STEP 2 — Time Series Decomposition (15% of grade)

### Classical Decomposition

Classical decomposition uses **centered moving averages** to estimate the trend.

For monthly data (m=12), the centered moving average at time t is:

```
T_t = (1/24) * y_{t-6} + (1/12) * Σ_{k=-5}^{5} y_{t+k} + (1/24) * y_{t+6}
```

The 1/24 weights at the endpoints ensure the sum of weights equals 1 for even-period
seasonality.

**Steps:**
1. Estimate trend `T_t` using centered moving average
2. De-trend: `y_t / T_t` (multiplicative) or `y_t - T_t` (additive)
3. Average de-trended values by month to get seasonal component `S_t`
4. Remainder: `R_t = y_t / (T_t × S_t)` or `y_t - T_t - S_t`

### STL Decomposition (Recommended)

STL (Seasonal-Trend decomposition using LOESS) is more robust — it uses locally
weighted regression (LOESS) to estimate trend and seasonal components.

**Key parameters:**
- `period` — the seasonal period (12 for monthly annual seasonality)
- `seasonal` — smoothing window for seasonal component (must be odd, ≥ 7)
- `trend` — smoothing window for trend (typically ≥ period + 1)

**Implementation:** `statsmodels.tsa.seasonal.STL`

### What to Interpret

After decomposition, examine each component:
- **Trend:** Is the growth linear? Exponential? Leveling off?
- **Seasonal:** Are the seasonal magnitudes stable? Growing? (For AirPassengers,
  STL in multiplicative mode or log-transformed additive will show stable seasonality)
- **Residuals (Remainder):** Should look like white noise — no patterns, no outliers.
  If you see structure in the residuals, the decomposition didn't capture everything.

**Key check:** Plot the residuals and visually inspect for autocorrelation (remaining
patterns would mean the model hasn't fully captured the series structure).

---

## STEP 3 — Model Building & Evaluation (20% + 25% of grade)

### Train-Test Split

This mirrors the lecture code's approach but is critical to understand conceptually.

```
Full series: [============================================]
              ← Train ──────────────────────→ ← Test →
                                              ^ cutoff
```

For AirPassengers (Jan 1949 – Dec 1960, 144 months):
- A common split is: train through mid-1960, test = last 12–24 months
- The test set size should match your forecast horizon

**Why not random split?** Because in time series, the **future cannot inform the past**.
Random shuffling would leak future information into training — a form of data leakage.

---

### Model A: SARIMA — Seasonal AutoRegressive Integrated Moving Average

**The general SARIMA(p,d,q)(P,D,Q)[m] model:**

```
Φ_P(B^m) φ_p(B) ∇^D_m ∇^d y_t = Θ_Q(B^m) θ_q(B) ε_t
```

Breaking this down from physics intuition — think of this as a **coupled oscillator equation**:

| Symbol | Meaning | Intuition |
|--------|---------|-----------|
| `p` | AR order | How many past values directly predict y_t |
| `d` | Differencing order | How many times to difference to remove trend |
| `q` | MA order | How many past *errors* influence y_t |
| `P` | Seasonal AR order | Same as p, but at seasonal lags |
| `D` | Seasonal differencing | Remove seasonal trend (∇_m = 1 - B^m) |
| `Q` | Seasonal MA order | Same as q, but at seasonal lags |
| `m` | Season length | 12 for monthly data |
| `B` | Backshift operator | B·y_t = y_{t-1} |

**Differencing (removing non-stationarity):**
- First difference: `∇y_t = y_t - y_{t-1}` → removes linear trend
- Seasonal difference: `∇_{12}y_t = y_t - y_{t-12}` → removes seasonal trend

For AirPassengers, the ACF slow decay tells you `d=1` (at minimum), and the seasonal
spike pattern tells you `D=1` as well.

**AutoARIMA in Nixtla** searches over (p,d,q)(P,D,Q) combinations automatically using
information criteria (AIC/BIC) to select the best model.

```
AIC = -2·log(L) + 2k
BIC = -2·log(L) + k·log(n)
```

Where `L` is the maximized likelihood and `k` is the number of parameters. Lower is better.
BIC penalizes complexity more harshly than AIC (prefer BIC when n is large).

---

### Model B: ETS — Error, Trend, Seasonality (Holt-Winters)

ETS models use **exponential smoothing** — a weighted average of past observations where
recent observations get exponentially higher weights.

**The three components:**

```
Level:     l_t = α·(y_t / s_{t-m}) + (1 - α)·(l_{t-1} + b_{t-1})    [multiplicative seasonal]
Trend:     b_t = β·(l_t - l_{t-1}) + (1 - β)·b_{t-1}
Seasonal:  s_t = γ·(y_t / l_t) + (1 - γ)·s_{t-m}
```

Parameters α, β, γ ∈ [0,1] are **smoothing parameters** — think of them as "memory decay rates":
- α close to 1 → model reacts quickly to changes (little memory)
- α close to 0 → model changes slowly (long memory)

This is analogous to **exponential decay** in physics: `f(t) = f_0·e^{-λt}`

**Forecast equation (Holt-Winters multiplicative):**

```
ŷ_{t+h|t} = (l_t + h·b_t) · s_{t+h-m(k+1)}
```

where `k = floor((h-1)/m)`.

**AutoETS in Nixtla** searches over all valid (Error, Trend, Seasonal) × (None/Additive/Multiplicative)
combinations and selects optimal α, β, γ by maximizing the likelihood.

For AirPassengers: expect AutoETS to select **ETS(M,A,M)** — multiplicative error,
additive trend, multiplicative seasonality. This is the classic Holt-Winters multiplicative model.

---

### Forecast Horizons

The assignment asks for **two forecast horizons**:

- **h=1 (one-step):** Forecast only the next time point. This is the easiest case —
  the model has maximum information from the past.

- **h=3 (three-step):** Forecast three periods into the future. Errors compound —
  uncertainty grows with horizon. This tests whether the model extrapolates well,
  not just fits.

**Implementation:** In `StatsForecast.forecast()`, set `h=1` or `h=3`.
Run the full pipeline separately (or use a loop) for each horizon.

---

### Evaluation Metrics

**Mean Absolute Error (MAE):**
```
MAE = (1/n) · Σ|y_t - ŷ_t|
```
Interpretable in original units. Not sensitive to outliers.

**Root Mean Squared Error (RMSE):**
```
RMSE = √[ (1/n) · Σ(y_t - ŷ_t)² ]
```
Penalizes large errors more than MAE (the squaring amplifies outliers).
Analogous to RMS in physics — the "energy" of the error.

**Mean Absolute Percentage Error (MAPE):**
```
MAPE = (100/n) · Σ|( y_t - ŷ_t ) / y_t|
```
Scale-independent (expressed as %). Allows comparison across datasets.
⚠️ **Warning:** MAPE is undefined when `y_t = 0` and biased when values are small.

**Model selection logic:**
- Compare SARIMA vs ETS on the **same test set** for each horizon separately
- A model that wins at h=1 may not win at h=3
- Also check **residual diagnostics** (see below)

---

### Residual Diagnostics (Critical for "Meets Expectations")

A well-fitting model should leave **white noise residuals**: `ε_t = y_t - ŷ_t`

White noise requires:
1. **Zero mean:** `E[ε_t] = 0`
2. **No autocorrelation:** `Corr(ε_t, ε_{t-k}) = 0  ∀ k ≠ 0`  → Check with ACF of residuals
3. **Constant variance (homoscedasticity):** `Var(ε_t) = σ²`  → Check with residual plot over time
4. **Normality (ideally):** `ε_t ~ N(0, σ²)`  → Check with histogram or Q-Q plot

**Ljung-Box test:** A formal test for residual autocorrelation.
```
H_0: Residuals are white noise (no autocorrelation)
H_1: Residuals exhibit autocorrelation
```
If p-value > 0.05: fail to reject H_0 — residuals look like white noise ✓
If p-value < 0.05: autocorrelation remains — model is misspecified ✗

---

## STEP 4 — Multivariate Feature Justification (10% of grade)

This is a **conceptual writing task** — no code required.

Imagine you wanted to add **exogenous variables** (external regressors) to your model.
These would turn SARIMA into SARIMAX (the X stands for eXogenous).

The SARIMAX model adds a regression component:
```
SARIMA terms + β₁·x₁_t + β₂·x₂_t + ... = y_t
```

**Two strong candidates for AirPassengers:**

1. **Economic indicator (e.g., GDP or disposable income index)**
   - Justification: Air travel demand is an economic activity — it correlates with
     the overall health of the economy. During recessions, fewer people fly.
     A GDP index captures the "ability to travel" signal beyond what the
     historical passenger count alone can model.

2. **Fuel price index (aviation kerosene price)**
   - Justification: Airline ticket prices are largely driven by fuel costs.
     Higher fuel costs → higher ticket prices → lower demand.
     This is a leading indicator — changes in fuel price often precede
     changes in passenger volume.

Your paragraphs should address:
- What the feature measures and why it correlates with air passenger volume
- Whether the feature would be available in real-time for forecasting
  (a feature you can't observe until after the fact is not useful for forecasting)
- How it would improve model performance (reduces MAPE by capturing variance
  the SARIMA/ETS residuals cannot explain)

---

## Code Architecture Checklist

Use this to structure your Jupyter notebook. Each cell block should be self-contained
with markdown explanations above and a docstring in each function.

```
[ ] 1. Imports and setup
[ ] 2. Load AirPassengers → format to Nixtla long format
[ ] 3. Line plot
[ ] 4. Seasonal plot
[ ] 5. Scatterplot / lag-1 scatter
[ ] 6. Lag plot (multiple lags)
[ ] 7. ACF plot
[ ] 8. PACF plot
[ ] 9. Commentary markdown cell (2-3 paragraphs)
[ ] 10. STL or classical decomposition + interpretation
[ ] 11. Train-test split
[ ] 12. StatsForecast initialization with AutoARIMA + AutoETS (season_length=12)
[ ] 13. h=1 forecast → compute MAE, RMSE, MAPE for both models
[ ] 14. h=3 forecast → compute MAE, RMSE, MAPE for both models
[ ] 15. Residual diagnostics for winning model (ACF of residuals, Ljung-Box)
[ ] 16. Final model justification markdown cell
[ ] 17. Multivariate feature discussion (2-3 paragraphs)
```

---

## Key Conceptual Differences from the Lecture Code

| Aspect | Lecture (Milk) | Your Assignment (AirPassengers) |
|--------|---------------|--------------------------------|
| Dataset | Monthly milk production | Monthly airline passengers |
| Models | SeasonalNaive + AutoETS + AutoARIMA | AutoETS + AutoARIMA (no Naive) |
| Forecast horizons | h=12 only | h=1 AND h=3 (separately) |
| Evaluation | Visual only | Quantitative: MAE/RMSE/MAPE |
| Residuals | Not checked | Required diagnostic |
| Decomposition | Not done | Required (STL or Classical) |
| Commentary | Not written | Required (2–3 paragraphs each section) |

The lecture code gives you the **skeleton** (data loading, Nixtla format, model init,
forecast call, plot structure). Everything else you need to build.

---

## Debugging Tips (Think Like a Programmer)

Before running any block, ask yourself:
1. **What shape should this DataFrame be?** Print `df.shape` and `df.head()` after every transformation.
2. **Are my dates parsed correctly?** Print `df['ds'].dtype` — it must be `datetime64`, not `object`.
3. **Is `unique_id` present?** Nixtla will throw cryptic errors if this column is missing or misnamed.
4. **Does `season_length` match my data frequency?** Monthly data → `season_length=12`. This is a parameter (handed to you by the problem), not a decision variable.
5. **Are train and test non-overlapping?** Print `train_df['ds'].max()` and `test_df['ds'].min()`.

---

## Docstring Template for Your Functions

When you write helper functions (e.g., `compute_metrics(actual, predicted)`), use this structure:

```python
def compute_metrics(actual, predicted):
    """
    Compute MAE, RMSE, and MAPE between actual and predicted time series values.

    Mathematical definitions:
        MAE  = (1/n) * sum(|y_t - ŷ_t|)
        RMSE = sqrt((1/n) * sum((y_t - ŷ_t)^2))
        MAPE = (100/n) * sum(|y_t - ŷ_t| / |y_t|)

    Note: MAPE is undefined if any actual value is zero.

    Parameters
    ----------
    actual : array-like of shape (n,)
        True observed values y_t
    predicted : array-like of shape (n,)
        Model forecast values ŷ_t

    Returns
    -------
    dict with keys 'MAE', 'RMSE', 'MAPE'
    """
    # Your implementation goes here
```

---

*Good luck. The AirPassengers dataset is a classic for a reason — it exhibits
every property (trend, multiplicative seasonality, growing variance) that makes
time series analysis both challenging and rewarding.*
