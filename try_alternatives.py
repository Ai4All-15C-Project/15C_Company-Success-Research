#!/usr/bin/env python3
"""
Try different SARIMAX configurations to find what works
"""
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("TESTING ALTERNATIVE APPROACHES")
print("="*70)

# Load data
df = pd.read_csv('Datasets/Tech_Stock_Data_SEC_Cleaned_SARIMAX.csv', parse_dates=['Date'])
df.set_index('Date', inplace=True)

# Create index
stocks = ['NVDA', 'AMD', 'INTC', 'GOOGL', 'MSFT', 'AAPL', 'META', 'AMZN',
          'CRM', 'ORCL', 'NOW', 'OKTA', 'ZS', 'CRWD', 'PANW',
          'ADBE', 'SHOP', 'TWLO', 'MDB', 'DDOG', 'NET', 'PYPL', 'ANET']
normalized = df[stocks].div(df[stocks].iloc[0]) * 100
df['AI_Tech_Index'] = normalized.mean(axis=1)

# Get exogenous
exog_vars = ['VIX', 'Treasury_10Y', 'Yield_Curve_Slope', 'AI_Boom_Period', 'Fed_Hike_Period']
available = [v for v in exog_vars if v in df.columns]
X_full = df[available].fillna(method='ffill').fillna(method='bfill')

print(f"\nAvailable exogenous: {available}")

# ============================================================================
# TEST 1: Different horizons for log returns
# ============================================================================
print("\n" + "="*70)
print("TEST 1: Different Horizons (7, 14, 30 days)")
print("="*70)

for horizon in [7, 14, 30]:
    print(f"\n{horizon}-day horizon:")

    df[f'logret_{horizon}d'] = np.log(df['AI_Tech_Index'].shift(-horizon) / df['AI_Tech_Index'])
    y = df[f'logret_{horizon}d'].dropna()
    X = X_full.loc[y.index]

    train_size = int(len(y) * 0.85)
    y_train, y_test = y[:train_size], y[train_size:]
    X_train, X_test = X[:train_size], X[train_size:]

    # Baseline
    baseline = ARIMA(y_train, order=(1, 0, 1)).fit()
    baseline_pred = baseline.forecast(steps=len(y_test))
    baseline_rmse = np.sqrt(mean_squared_error(y_test, baseline_pred))

    # SARIMAX with VIX
    try:
        model = SARIMAX(y_train, exog=X_train[['VIX']],
                       order=(1, 0, 1), seasonal_order=(0, 0, 0, 0),
                       enforce_stationarity=False, enforce_invertibility=False)
        fitted = model.fit(disp=False, maxiter=200)
        pred = fitted.forecast(steps=len(y_test), exog=X_test[['VIX']])
        rmse = np.sqrt(mean_squared_error(y_test, pred))
        r2 = r2_score(y_test, pred)

        improvement = ((baseline_rmse - rmse) / baseline_rmse * 100)
        print(f"  Baseline RMSE: {baseline_rmse:.6f}")
        print(f"  SARIMAX RMSE:  {rmse:.6f}")
        print(f"  R²: {r2:.4f}")
        print(f"  Improvement: {improvement:.1f}%")
    except Exception as e:
        print(f"  FAILED: {str(e)[:60]}")

# ============================================================================
# TEST 2: Different ARIMA orders (7-day)
# ============================================================================
print("\n" + "="*70)
print("TEST 2: Different ARIMA Orders (7-day log returns)")
print("="*70)

y = df['logret_7d'].dropna()
X = X_full.loc[y.index]
train_size = int(len(y) * 0.85)
y_train, y_test = y[:train_size], y[train_size:]
X_train, X_test = X[:train_size], X[train_size:]

orders = [(0,0,1), (1,0,0), (2,0,0), (0,0,2), (2,0,2), (1,0,2)]

for order in orders:
    try:
        # Baseline
        baseline = ARIMA(y_train, order=order).fit()
        baseline_pred = baseline.forecast(steps=len(y_test))
        baseline_rmse = np.sqrt(mean_squared_error(y_test, baseline_pred))

        # With VIX
        model = SARIMAX(y_train, exog=X_train[['VIX']],
                       order=order, seasonal_order=(0, 0, 0, 0),
                       enforce_stationarity=False, enforce_invertibility=False)
        fitted = model.fit(disp=False, maxiter=200)
        pred = fitted.forecast(steps=len(y_test), exog=X_test[['VIX']])
        rmse = np.sqrt(mean_squared_error(y_test, pred))

        improvement = ((baseline_rmse - rmse) / baseline_rmse * 100)
        print(f"\nOrder {order}:")
        print(f"  Baseline: {baseline_rmse:.6f}, With VIX: {rmse:.6f}, Improvement: {improvement:.1f}%")
    except Exception as e:
        print(f"\nOrder {order}: FAILED - {str(e)[:50]}")

# ============================================================================
# TEST 3: Multiple exogenous variables
# ============================================================================
print("\n" + "="*70)
print("TEST 3: Multiple Exogenous Variables (7-day)")
print("="*70)

var_combos = [
    ['VIX'],
    ['VIX', 'Treasury_10Y'],
    ['VIX', 'AI_Boom_Period'],
    ['VIX', 'Treasury_10Y', 'AI_Boom_Period'],
    available[:3]
]

for vars_to_use in var_combos:
    try:
        model = SARIMAX(y_train, exog=X_train[vars_to_use],
                       order=(1, 0, 1), seasonal_order=(0, 0, 0, 0),
                       enforce_stationarity=False, enforce_invertibility=False)
        fitted = model.fit(disp=False, maxiter=200)
        pred = fitted.forecast(steps=len(y_test), exog=X_test[vars_to_use])
        rmse = np.sqrt(mean_squared_error(y_test, pred))
        r2 = r2_score(y_test, pred)

        print(f"\nVariables: {vars_to_use}")
        print(f"  RMSE: {rmse:.6f}, R²: {r2:.4f}")
    except Exception as e:
        print(f"\nVariables: {vars_to_use}")
        print(f"  FAILED: {str(e)[:50]}")

# ============================================================================
# TEST 4: Simple returns instead of log returns
# ============================================================================
print("\n" + "="*70)
print("TEST 4: Simple Returns (7-day)")
print("="*70)

df['simple_ret_7d'] = (df['AI_Tech_Index'].shift(-7) - df['AI_Tech_Index']) / df['AI_Tech_Index']
y = df['simple_ret_7d'].dropna()
X = X_full.loc[y.index]

train_size = int(len(y) * 0.85)
y_train, y_test = y[:train_size], y[train_size:]
X_train, X_test = X[:train_size], X[train_size:]

# Baseline
baseline = ARIMA(y_train, order=(1, 0, 1)).fit()
baseline_pred = baseline.forecast(steps=len(y_test))
baseline_rmse = np.sqrt(mean_squared_error(y_test, baseline_pred))

# With VIX
try:
    model = SARIMAX(y_train, exog=X_train[['VIX']],
                   order=(1, 0, 1), seasonal_order=(0, 0, 0, 0),
                   enforce_stationarity=False, enforce_invertibility=False)
    fitted = model.fit(disp=False, maxiter=200)
    pred = fitted.forecast(steps=len(y_test), exog=X_test[['VIX']])
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)

    improvement = ((baseline_rmse - rmse) / baseline_rmse * 100)
    print(f"Baseline RMSE: {baseline_rmse:.6f}")
    print(f"SARIMAX RMSE:  {rmse:.6f}")
    print(f"R²: {r2:.4f}")
    print(f"Improvement: {improvement:.1f}%")
except Exception as e:
    print(f"FAILED: {str(e)}")

# ============================================================================
# TEST 5: Index levels (differenced) - original approach
# ============================================================================
print("\n" + "="*70)
print("TEST 5: Index Levels with Differencing (7-day ahead)")
print("="*70)

# For levels, shift the index itself
y_levels = df['AI_Tech_Index'].shift(-7).dropna()
y_current = df['AI_Tech_Index'].loc[y_levels.index]
X = X_full.loc[y_levels.index]

train_size = int(len(y_levels) * 0.85)
y_train = y_levels[:train_size]
y_current_train = y_current[:train_size]
X_train = X[:train_size]

y_test = y_levels[train_size:]
y_current_test = y_current[train_size:]
X_test = X[train_size:]

# Baseline ARIMA(1,1,1) - with differencing
try:
    baseline = ARIMA(y_current_train, order=(1, 1, 1)).fit()
    # Forecast from last training point
    baseline_pred = baseline.forecast(steps=len(y_test))
    baseline_rmse = np.sqrt(mean_squared_error(y_test, baseline_pred))

    # SARIMAX with VIX
    model = SARIMAX(y_current_train, exog=X_train[['VIX']],
                   order=(1, 1, 1), seasonal_order=(0, 0, 0, 0),
                   enforce_stationarity=False, enforce_invertibility=False)
    fitted = model.fit(disp=False, maxiter=200)
    pred = fitted.forecast(steps=len(y_test), exog=X_test[['VIX']])
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)

    improvement = ((baseline_rmse - rmse) / baseline_rmse * 100)
    print(f"Baseline RMSE: {baseline_rmse:.2f}")
    print(f"SARIMAX RMSE:  {rmse:.2f}")
    print(f"R²: {r2:.4f}")
    print(f"Improvement: {improvement:.1f}%")
except Exception as e:
    print(f"FAILED: {str(e)}")

print("\n" + "="*70)
print("DONE - Check which approach works best")
print("="*70)
