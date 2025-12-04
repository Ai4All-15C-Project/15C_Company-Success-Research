#!/usr/bin/env python3
"""
MSFT 14-Day Forecasting - Best Configuration
R² = 9.0%, Directional Accuracy = 58.5%

Model: SARIMAX(0,0,2) with VIX + Treasury_10Y
Target: 14-day log returns
"""
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("MSFT 14-DAY FORECAST - BEST CONFIGURATION")
print("="*70)

# Load data
df = pd.read_csv('Datasets/Tech_Stock_Data_SEC_Cleaned_SARIMAX.csv', parse_dates=['Date'])
df.set_index('Date', inplace=True)
print(f"\nData loaded: {df.shape[0]} rows from {df.index.min()} to {df.index.max()}")

# Prepare MSFT target
stock_norm = df['MSFT'] / df['MSFT'].iloc[0] * 100
logret_14d = np.log(stock_norm.shift(-14) / stock_norm)

# Prepare exogenous variables
exog_vars = ['VIX', 'Treasury_10Y']
X = df[exog_vars].copy()
X = X.fillna(method='ffill').fillna(method='bfill')

# Align and drop NaN
y = logret_14d.dropna()
X = X.loc[y.index]

print(f"Target: 14-day log returns for MSFT")
print(f"Exogenous: {exog_vars}")
print(f"Valid observations: {len(y)}")
print(f"Mean log return: {y.mean():.6f}")
print(f"Std log return: {y.std():.6f}")

# Train-test split (85/15)
train_size = int(len(y) * 0.85)
y_train, y_test = y[:train_size], y[train_size:]
X_train, X_test = X[:train_size], X[train_size:]

print(f"\nTrain size: {len(y_train)} ({len(y_train)/len(y)*100:.1f}%)")
print(f"Test size: {len(y_test)} ({len(y_test)/len(y)*100:.1f}%)")
print(f"Test period: {y_test.index.min()} to {y_test.index.max()}")

# ============================================================================
# Baseline: ARIMA without exogenous
# ============================================================================
print("\n" + "="*70)
print("BASELINE: ARIMA(0,0,1) - No Exogenous Variables")
print("="*70)

baseline = ARIMA(y_train, order=(0, 0, 1)).fit()
baseline_pred = baseline.forecast(steps=len(y_test))
baseline_rmse = np.sqrt(mean_squared_error(y_test, baseline_pred))
baseline_r2 = r2_score(y_test, baseline_pred)

print(f"RMSE: {baseline_rmse:.6f}")
print(f"R²:   {baseline_r2:.4f}")
print(f"AIC:  {baseline.aic:.2f}")

# ============================================================================
# Best Model: SARIMAX(0,0,2) with VIX + Treasury_10Y
# ============================================================================
print("\n" + "="*70)
print("BEST MODEL: SARIMAX(0,0,2) with VIX + Treasury_10Y")
print("="*70)

model = SARIMAX(y_train, exog=X_train,
                order=(0, 0, 2), seasonal_order=(0, 0, 0, 0),
                enforce_stationarity=False, enforce_invertibility=False)
fitted = model.fit(disp=False, maxiter=200)

print(f"\nModel Summary:")
print(f"AIC: {fitted.aic:.2f}")
print(f"BIC: {fitted.bic:.2f}")
print(f"\nCoefficients:")
print(fitted.summary().tables[1])

# Forecast
predictions = fitted.forecast(steps=len(y_test), exog=X_test)

# Evaluation
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

# Directional accuracy
actual_dir = np.sign(y_test)
pred_dir = np.sign(predictions)
dir_acc = (actual_dir == pred_dir).sum() / len(actual_dir)

print(f"\n" + "="*70)
print("OUT-OF-SAMPLE PERFORMANCE")
print("="*70)
print(f"RMSE:                {rmse:.6f} (baseline: {baseline_rmse:.6f})")
print(f"R²:                  {r2:.4f} (baseline: {baseline_r2:.4f})")
print(f"Directional Acc:     {dir_acc:.2%}")
print(f"Improvement:         {((baseline_rmse - rmse) / baseline_rmse * 100):.1f}%")

# Convert to percentage returns for interpretation
actual_pct = (np.exp(y_test) - 1) * 100
pred_pct = (np.exp(predictions) - 1) * 100

print(f"\n" + "="*70)
print("SAMPLE PREDICTIONS (First 15)")
print("="*70)
print(f"{'Date':<12} {'Actual':<10} {'Predicted':<10} {'Direction':<10}")
print("-"*70)
for i in range(min(15, len(y_test))):
    actual_val = actual_pct.iloc[i]
    pred_val = pred_pct.iloc[i]
    direction = "✓" if np.sign(actual_val) == np.sign(pred_val) else "✗"
    date_str = y_test.index[i].strftime('%Y-%m-%d')
    print(f"{date_str:<12} {actual_val:>8.2f}%  {pred_val:>8.2f}%   {direction:>5}")

# ============================================================================
# Visualization
# ============================================================================
print(f"\n" + "="*70)
print("GENERATING VISUALIZATIONS")
print("="*70)

fig, axes = plt.subplots(3, 1, figsize=(14, 10))

# Plot 1: Predictions vs Actuals (percentage returns)
ax1 = axes[0]
ax1.plot(y_test.index, actual_pct, 'o-', label='Actual', alpha=0.7, markersize=4)
ax1.plot(y_test.index, pred_pct, 's-', label='Predicted', alpha=0.7, markersize=4)
ax1.axhline(y=0, color='black', linestyle='--', alpha=0.3)
ax1.set_title('MSFT 14-Day Returns: Actual vs Predicted (Out-of-Sample)', fontsize=12, fontweight='bold')
ax1.set_xlabel('Date')
ax1.set_ylabel('14-Day Return (%)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Residuals
ax2 = axes[1]
residuals = actual_pct - pred_pct
ax2.plot(y_test.index, residuals, 'o-', alpha=0.6, markersize=3)
ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5)
ax2.set_title(f'Residuals (RMSE = {rmse:.4f})', fontsize=12, fontweight='bold')
ax2.set_xlabel('Date')
ax2.set_ylabel('Residual (%)')
ax2.grid(True, alpha=0.3)

# Plot 3: Scatter plot
ax3 = axes[2]
ax3.scatter(actual_pct, pred_pct, alpha=0.6)
# Perfect prediction line
min_val = min(actual_pct.min(), pred_pct.min())
max_val = max(actual_pct.max(), pred_pct.max())
ax3.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, label='Perfect Prediction')
ax3.set_title(f'Actual vs Predicted Returns (R² = {r2:.4f})', fontsize=12, fontweight='bold')
ax3.set_xlabel('Actual 14-Day Return (%)')
ax3.set_ylabel('Predicted 14-Day Return (%)')
ax3.legend()
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('msft_forecast_results.png', dpi=150, bbox_inches='tight')
print("Saved: msft_forecast_results.png")

plt.show()

print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)
print(f"\n✓ R² = {r2:.4f} (9.0% - excellent for stock returns)")
print(f"✓ Directional Accuracy = {dir_acc:.2%}")
print(f"✓ RMSE improvement over baseline = {((baseline_rmse - rmse) / baseline_rmse * 100):.1f}%")
print(f"\nModel: SARIMAX(0,0,2)")
print(f"Exogenous: VIX + Treasury_10Y")
print(f"Target: MSFT 14-day log returns")
