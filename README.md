# AI Tech Sector Forecasting Research

## Project Overview

This project analyzes and forecasts the AI/Tech sector using a custom-built **AI Tech Index** representing 24 major technology companies. The analysis includes comprehensive EDA, time series modeling with SARIMAX, and evaluation of forecasting limitations.

**Date Range:** 2019-01-01 to 2025-11-21  
**Frequency:** Daily (weekends removed)  
**Target Variable:** AI Tech Index (normalized equal-weighted average of 24 companies)

---

## Key Findings

### 1. Model Performance
- **Rolling 1-Step-Ahead R²:** 0.9726 (97.26%)
- **Long-Horizon Forecast R²:** -1.95 (model fails for multi-step forecasts)
- **Key Insight:** High R² for 1-step forecasts is largely due to autocorrelation, not predictive power

### 2. Regime Shift Detection
The model identified a **regime shift in 2025** where the AI tech sector reached all-time highs beyond the training data range. This is a fundamental limitation of statistical time series models - they cannot extrapolate to new market regimes.

### 3. Exogeneity Problem Solved
We explicitly **avoided using SP500/NASDAQ** as predictors because our target contains the same stocks (NVDA, AMD, MSFT, GOOGL, AAPL, etc.). Using them would be circular reasoning.

---

## Data Structure

### AI Tech Index Components (24 Companies)

| Sector | Companies |
|--------|-----------|
| **AI Hardware** | NVDA, AMD, INTC |
| **Cloud/AI Services** | GOOGL, MSFT, CRM, ORCL, NOW |
| **Cybersecurity** | OKTA, ZS, CRWD, NET, PANW |
| **Big Tech** | AAPL, META, AMZN, IBM |
| **Software/SaaS** | ADBE, SHOP, TWLO, MDB, DDOG, PYPL, ANET |

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

**SEC Fundamentals (Sector-Level):**
- Sector_Profit_Margin, Sector_ROE, Sector_Revenue_Growth, Sector_Asset_Turnover

**Regime Indicators (5):**
- Pandemic_Period, AI_Boom_Period, Fed_Hike_Period, Tech_Bear_2022, Banking_Crisis_2023

---

## EDA Notebook Analysis

### Part 1: Exploratory Data Analysis

1. **Univariate Analysis** - Distribution and summary statistics of all variables
2. **Bivariate Analysis** - Stock performance comparisons by sector
3. **Multivariate Analysis** - Correlation heatmaps and regime analysis
4. **Time Series Analysis** - Stationarity testing (ADF), ACF/PACF plots
5. **SEC Fundamentals** - Sector-level financial metrics analysis

### Part 2: SARIMAX Model Development

1. **Data Preparation** - Target and exogenous variable selection
2. **Multicollinearity Check** - VIF analysis (removed features with VIF > 10)
3. **Train-Test Split** - 90% training, 10% testing
4. **Model Fit** - SARIMAX(1,1,1) and SARIMAX(2,1,2) comparison
5. **Model Diagnostics** - Residual analysis, Ljung-Box test, Jarque-Bera test
6. **Anomaly Detection** - Isolation Forest on residuals

### Part 3: Forecast Evaluation

1. **Long-Horizon Forecast** - Multi-step ahead predictions (failed due to regime shift)
2. **Rolling 1-Step-Ahead** - Re-fit model daily (R² = 0.97)
3. **Regime Shift Analysis** - Identified extrapolation problem in 2025 data

---

## Key Methodological Decisions

### Why We Use Truly Exogenous Variables

**Problem:** Our AI Tech Index contains stocks like NVDA, MSFT, GOOGL, AAPL, etc. These same stocks are major components of the S&P 500 and NASDAQ indices.

**Solution:** We use only **truly exogenous** variables:
- **VIX** - Market volatility/fear (sentiment, not prices)
- **Treasury_10Y** - Interest rate environment (monetary policy)
- **Yield_Curve_Slope** - Economic outlook indicator
- **SEC Fundamentals** - Company financial health metrics

### Why Rolling Forecasts Work Better

| Approach | R² | Why |
|----------|-----|-----|
| Long-Horizon | -1.95 | Model cannot extrapolate beyond training data range |
| Rolling 1-Step | 0.97 | Model re-fits daily with new information |

**Caveat:** The high R² for 1-step forecasts is largely due to autocorrelation (today ≈ yesterday). A naive baseline would also perform well.

---

## Files

| File | Description |
|------|-------------|
| `EDA_Company Data.ipynb` | Main analysis notebook with EDA and SARIMAX modeling |
| `Datasets/Tech_Stock_Data_SEC_Cleaned_SARIMAX.csv` | Full cleaned dataset with all features |
| `Datasets/SARIMAX_Exogenous_Features.csv` | Selected exogenous variables for modeling |
| `Datasets/Tech_Stock_Data_with_SEC_Fundamentals.csv` | Dataset with SEC fundamental data |
| `clean_sec_fundamentals.py` | Script to clean SEC data |
| `gen_company_data.py` | Script to generate company data |
| `requirements.txt` | Python dependencies |

---

## Installation

```bash
pip install -r requirements.txt
```

### Requirements
- pandas
- numpy
- matplotlib
- seaborn
- statsmodels
- scikit-learn

---

## Future Work

1. **Naive Baseline Comparison** - Compare SARIMAX to simple persistence model
2. **Multi-Step Forecasting** - Evaluate 5, 10, 30-day ahead predictions
3. **Return Prediction** - Forecast returns instead of levels (lower autocorrelation)
4. **Alternative Models** - Prophet, LSTM, XGBoost for comparison
5. **Regime-Switching Models** - Handle different market conditions
6. **Directional Accuracy** - Measure ability to predict up/down movements

---

## Conclusions

1. **SARIMAX is effective for 1-step-ahead forecasting** but the high R² is partially due to autocorrelation in price data.

2. **Long-horizon forecasting fails** when the market enters a new regime (2025 AI boom) beyond the training data range.

3. **Truly exogenous variables** (VIX, interest rates, SEC fundamentals) provide legitimate predictive signal without data leakage.

4. **Rolling forecasts are essential** for real-world deployment - the model must be re-fit as new data arrives.

5. **Statistical models have fundamental limitations** for financial forecasting - they cannot predict regime changes or black swan events.