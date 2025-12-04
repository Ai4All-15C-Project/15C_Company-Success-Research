# 15C_Company-Success-Research

# Tech Stock Multi-Step Forecasting with Enhanced SARIMAX

## Overview

This project implements **multi-step ahead forecasting** (7-day, 14-day, and 30-day horizons) for tech stocks using SARIMAX models with **truly exogenous variables**. Unlike trivial 1-step predictions, our approach provides actionable medium-term forecasts for investment decisions.

**Date Range:** 2019-01-01 to 2025-11-21
**Frequency:** Daily stock data
**Companies:** 24 AI/tech companies across multiple sectors
**Forecast Horizons:** 7, 14, and 30 days ahead

### Key Features

- **Multi-Step Forecasting**: Direct forecasting at 7, 14, and 30-day horizons
- **Truly Exogenous Variables**: Only variables determined outside the tech sector (no NASDAQ, tech ETFs)
- **Conservative Selection**: 3-5 variables maximum via AIC/BIC-based forward selection
- **Rigorous Validation**: Baseline comparisons, walk-forward validation, overfitting checks
- **Comprehensive Evaluation**: RMSE, MAE, R², directional accuracy vs baseline ARIMA

---

## Data Structure

### Stock Tickers (22)

**Semiconductors/Hardware:**
- NVDA, AMD, INTC

**Cloud/SaaS:**
- GOOGL, MSFT, CRM, ORCL, NOW, OKTA

**Cybersecurity:**
- ZS, CRWD, NET

**Big Tech:**
- AAPL, META, AMZN, IBM

**Software/Other:**
- ADBE, SHOP, SQ, TWLO, MDB, DDOG

### Exogenous Variables (60+)

**Market Indices (7):**
- SP500, NASDAQ, Dow_Jones, Russell2000, VIX, VVIX, NASDAQ_VIX

**Interest Rates (6):**
- Treasury_10Y, Treasury_3M, Treasury_5Y, Treasury_30Y
- Yield_Curve_Slope, Yield_Curve_Inverted

**Sector ETFs (14):**
- Tech_Sector_ETF, Semiconductor_ETF, Software_ETF, Cloud_Computing_ETF
- Cybersecurity_ETF, AI_Robotics_ETF, Global_Robotics_ETF, Global_Cloud_ETF
- ARK funds (Innovation, Autonomous_Tech, Next_Gen_Internet)
- NASDAQ_100_ETF, First_Trust_NASDAQ, PHLX_Semi_Index

**Macro Indicators (9):**
- Dollar_Index, Dollar_Strength, Dollar_MA20
- Gold, Silver, Copper, Oil_WTI, Natural_Gas
- Gold_Oil_Ratio, Credit_Spread_Proxy

**Crypto (3):**
- Bitcoin, Ethereum, Crypto_Tech_Corr_20d

**International Tech (8):**
- Taiwan_ETF, Japan_ETF, South_Korea_ETF, China_Large_Cap_ETF
- ASML, TSM, BABA, BIDU

**Risk Indicators (5):**
- High_Yield_Bonds, Investment_Grade_Bonds, Long_Term_Treasury
- Emerging_Markets_Bonds, Real_Estate_ETF

**Volatility Metrics (5):**
- High_Volatility_Regime, Extreme_Fear, VIX_MA20, VIX_vs_MA, Vol_of_Vol_Ratio

**Technical Indicators (5):**
- Semi_vs_Tech_Ratio, Small_vs_Large_Caps, NASDAQ_MA5, NASDAQ_MA20, NASDAQ_Momentum

**Regime Indicators (5):**
- Pandemic_Period, AI_Boom_Period, Fed_Hike_Period, Tech_Bear_2022, Banking_Crisis_2023

**Temporal Features (10):**
- Day_of_Week, Day_of_Month, Week_of_Year, Month, Quarter, Year
- Is_Month_End, Is_Quarter_End, Earnings_Season, Options_Expiry_Week

---

## Truly Exogenous Variable Selection

### What Makes a Variable "Truly Exogenous"?

For a variable to be valid in our forecasting model, it must meet **all three criteria**:

1. **Determined outside the tech sector** - Not influenced by the stocks we're predicting
2. **Causally prior** - Influences tech stocks but tech stocks don't influence it
3. **Observable at prediction time** - No future peeking or data leakage

### Valid Exogenous Candidates

✅ **Market-Wide Volatility:**
- **VIX** - SPX options-implied volatility (market-wide, not tech-specific)
- **High_Volatility_Regime** - Binary indicator for extreme market stress

✅ **Interest Rates & Macro (Set by Fed/Bond Market):**
- **Treasury_10Y** - Risk-free rate benchmark
- **Treasury_3M** - Short-term rate
- **Yield_Curve_Slope** - 10Y-3M spread (recession indicator)
- **Yield_Curve_Inverted** - Binary recession signal
- **Dollar_Index** - USD currency strength

✅ **Commodities (Separate Markets):**
- **Gold** - Safe haven demand
- **Oil_WTI** - Energy/inflation proxy
- **Bitcoin** - Crypto risk appetite (debatable but separate asset class)

✅ **Regime Indicators (Binary Flags):**
- **AI_Boom_Period** - Post-ChatGPT era (Nov 2022+)
- **Fed_Hike_Period** - Rate hiking cycle
- **Tech_Bear_2022** - 2022 tech sell-off

### INVALID Variables (Endogenous/Circular)

❌ **NASDAQ, NASDAQ_100_ETF** - **Contains our target stocks!** Using NASDAQ to predict tech stocks is circular since tech stocks ARE NASDAQ.

❌ **Tech Sector ETFs** - Semiconductor_ETF, Software_ETF, Cloud_Computing_ETF, etc. literally contain the stocks we're forecasting.

❌ **Sector Fundamentals** - Profit margins, ROE, etc. are **outcomes**, not drivers. They're determined BY the stocks, not external to them.

### Final Model Uses 3-5 Variables Maximum

Through AIC/BIC-based forward selection, we choose only the most predictive variables that provide incremental information. Typical final models include:
- VIX (market volatility)
- Treasury rates or yield curve slope
- Dollar Index or commodity
- 1-2 regime indicators

---

## Methodology

### Feature Selection Process

**Conservative, AIC/BIC-Based Approach:**

1. **Candidate Pool**: Only truly exogenous variables (11 candidates max)
2. **Baseline Comparison**: Train ARIMA(1,1,1) with no exogenous variables
3. **Forward Selection**: Add variables one at a time via AIC minimization
4. **Stopping Rule**: Stop when AIC improvement < 2.0 or reached 5 variables
5. **Final Model**: Typically 3-5 variables selected

**Why This Approach:**
- **AIC/BIC penalize complexity** - Prevents overfitting
- **Forward selection** - Only adds variables that genuinely improve prediction
- **Hard limit of 5 variables** - Forces parsimony
- **Baseline comparison** - Validates that exogenous variables actually help

### Multi-Step Forecasting Approach

We implement **direct multi-step forecasting** rather than 1-step ahead:

**Target Variable: Log Returns**

Instead of predicting index levels, we forecast **log returns**:
- Formula: `log_return = ln(P_t+h / P_t)`
- Convert to %: `pct_return = exp(log_return) - 1`
- **Stationary by nature** (no differencing needed)
- **Time-additive** (7-day log return = sum of daily log returns)
- **Standard in quant finance** (better statistical properties)

**Forecast Horizons:**

- **7-Day Model**: Predicts 7-day log return (e.g., "expected +2.8% return")
  - Use case: Short-term trading strategies, options positioning
  - Expected accuracy: Highest among the three horizons

- **14-Day Model**: Predicts 14-day log return
  - Use case: Medium-term portfolio adjustments, swing trading
  - Expected accuracy: Moderate (natural degradation with longer horizon)

- **30-Day Model**: Predicts 30-day log return
  - Use case: Strategic allocation decisions, trend identification
  - Expected accuracy: Lower but still actionable for trend direction

### Model Architecture

**SARIMAX(1, 0, 1) x (0, 0, 0, 0)**
- AR(1): Autoregressive component captures momentum
- **I(0): No differencing** - log returns are already stationary
- MA(1): Moving average component for shock absorption
- No seasonality: Daily data doesn't exhibit strong seasonal patterns

**Why I(0) instead of I(1)?**
- Index levels need differencing (I=1) to become stationary
- Log returns are already stationary, so I=0
- Simpler model with better interpretation

### Evaluation Metrics

1. **RMSE (Root Mean Squared Error)**: Point prediction accuracy
2. **MAE (Mean Absolute Error)**: Average error magnitude
3. **R² Score**: Proportion of variance explained
4. **Directional Accuracy**: Percentage of correct up/down predictions (often most useful)

---

## Results & Insights

### Validation Approach

1. **Baseline Comparison**: Every model compared to ARIMA-only baseline
2. **Walk-Forward Validation**: 5 rolling windows to test temporal stability
3. **Out-of-Sample Testing**: 15% holdout set never used in training or selection
4. **AIC/BIC Metrics**: Model complexity penalized appropriately

### Expected Performance Patterns

- **7-day forecasts**: Best accuracy, modest improvement over baseline (5-15% RMSE reduction)
- **14-day forecasts**: Moderate accuracy, smaller improvement over baseline
- **30-day forecasts**: Lower accuracy but may still beat baseline for trend direction
- **Directional accuracy**: Often more important than point predictions for trading

### Key Findings

1. **Exogeneity Matters**: Using truly exogenous variables prevents spurious correlations
2. **Less is More**: 3-5 carefully selected variables outperform 10-15 variables
3. **VIX and Rates**: Market volatility and interest rates typically most predictive
4. **Regime Indicators Help**: Binary flags for AI Boom, Fed Hike periods add value
5. **Modest Improvements**: SARIMAX beats baseline but gains are incremental (not magical)

### Model Limitations & Warnings

- **Linear assumptions**: SARIMAX assumes linear relationships, may miss non-linear patterns
- **Structural breaks**: Regime changes can invalidate historical relationships
- **No guarantees**: Outperforming baseline in testing doesn't guarantee future performance
- **Overfitting risk**: Even with conservative selection, overfitting is possible
- **Requires retraining**: Model should be retrained monthly as new data arrives
- **Not financial advice**: This is an academic exercise, not investment guidance

---

## Files in Repository

- **EDA_Company Data.ipynb**: Main analysis notebook with multi-step forecasting
- **Datasets/Tech_Stock_Data_SEC_Cleaned_SARIMAX.csv**: Complete dataset with all features
- **Datasets/SARIMAX_Exogenous_Features.csv**: Exogenous variables only
- **Datasets/Multi_Step_Forecast_Results.csv**: Model predictions (generated after running notebook)
- **Datasets/Selected_Exogenous_Variables.txt**: Final selected features (generated after running)

---

## Running the Analysis

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run Jupyter Notebook**:
   ```bash
   jupyter notebook "EDA_Company Data.ipynb"
   ```

3. **Execute All Cells**: The notebook will:
   - Load and explore the data
   - Identify truly exogenous variables (exclude NASDAQ, tech ETFs)
   - Perform AIC/BIC-based forward selection (max 5 variables)
   - Train baseline ARIMA models for comparison
   - Train 3 SARIMAX models (7, 14, 30-day horizons)
   - Run walk-forward validation to check overfitting
   - Generate performance metrics and visualizations
   - Save predictions to CSV

---

## Future Improvements

1. **Confidence Intervals**: Add probabilistic forecasts with prediction intervals
2. **Regime-Conditional Models**: Separate models for high/low volatility periods
3. **Ensemble with ML**: Combine SARIMAX with XGBoost/Random Forest (carefully avoiding overfitting)
4. **Alternative Exogenous**: Test commodity ratios, credit spreads, international rates
5. **Automated Retraining**: Monthly model updates with drift detection
6. **Shrinkage Methods**: Explore Bayesian SARIMAX for coefficient regularization

**Important**: Any improvements must maintain rigorous validation and avoid the endogeneity trap.

---