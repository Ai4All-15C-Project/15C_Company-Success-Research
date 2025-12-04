# 15C_Company-Success-Research

# Tech Stock Multi-Step Forecasting with Enhanced SARIMAX

## Overview

This project implements **multi-step ahead forecasting** (7-day, 14-day, and 30-day horizons) for tech stocks using SARIMAX models with rich exogenous variables. Unlike trivial 1-step predictions, our approach provides actionable medium-term forecasts for investment decisions.

**Date Range:** 2019-01-01 to 2025-11-21
**Frequency:** Daily stock data
**Companies:** 24 AI/tech companies across multiple sectors
**Forecast Horizons:** 7, 14, and 30 days ahead

### Key Features

- **Multi-Step Forecasting**: Direct forecasting at 7, 14, and 30-day horizons
- **Enhanced Exogenous Variables**: Tech-specific indices, crypto correlations, sector fundamentals, regime indicators
- **Advanced Feature Selection**: VIF-based multicollinearity removal, correlation analysis
- **Comprehensive Evaluation**: RMSE, MAE, R², and directional accuracy metrics

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

## Enhanced Exogenous Variable Categories

Our approach uses a sophisticated feature selection process that prioritizes tech-specific indicators over broad market metrics:

### Tech-Specific Indices (High Priority)
- **NASDAQ, NASDAQ_100_ETF** - Core tech market benchmarks
- **Semiconductor_ETF, Software_ETF** - Sector-specific momentum
- **Cloud_Computing_ETF, Cybersecurity_ETF** - High-growth tech subsectors
- **AI_Robotics_ETF, First_Trust_NASDAQ** - AI and innovation exposure

### Crypto Indicators (Tech Correlation Signal)
- **Bitcoin, Ethereum** - Strong correlation with tech risk appetite, especially during AI boom

### Sector Fundamentals (Health Indicators)
- **Sector_Profit_Margin** - Profitability trends
- **Sector_ROE** - Return on equity
- **Sector_Revenue_Growth** - Growth momentum
- **Sector_Asset_Turnover** - Operational efficiency
- **Sector_Profitable_Pct** - Percentage of profitable companies

### Volatility & Risk Measures
- **VIX, NASDAQ_VIX** - Market fear gauges
- **Vol_of_Vol_Ratio** - Meta-volatility (uncertainty about uncertainty)
- **High_Volatility_Regime** - Binary regime indicator

### Macro & Interest Rates
- **Treasury_10Y** - Risk-free rate
- **Yield_Curve_Slope** - Leading recession indicator
- **Yield_Curve_Inverted** - Recession predictor
- **Dollar_Index** - Currency strength

### Regime Indicators (Structural Breaks)
- **AI_Boom_Period** - Post-ChatGPT era (Nov 2022+)
- **Fed_Hike_Period** - Rate hiking cycle
- **Tech_Bear_2022** - 2022 tech sell-off period

### Technical Ratios
- **Semi_vs_Tech_Ratio** - Semiconductor strength vs broad tech
- **Small_vs_Large_Caps** - Market breadth
- **Credit_Spread_Proxy** - Credit market stress

---

## Methodology

### Feature Selection Process

1. **Initial Candidate Pool**: 30+ exogenous variables selected for tech-relevance
2. **Correlation Analysis**: Ranked by correlation with AI Tech Index
3. **VIF Filtering**: Iteratively remove variables with VIF > 10 to eliminate multicollinearity
4. **Final Model**: 10-15 variables per model (varies by horizon)

### Multi-Step Forecasting Approach

We implement **direct multi-step forecasting** rather than 1-step ahead:

- **7-Day Model**: Predicts stock index value 7 days into the future
  - Use case: Short-term trading strategies, options positioning
  - Expected accuracy: Highest among the three horizons

- **14-Day Model**: Predicts stock index value 14 days ahead
  - Use case: Medium-term portfolio adjustments, swing trading
  - Expected accuracy: Moderate (natural degradation with longer horizon)

- **30-Day Model**: Predicts stock index value 30 days ahead
  - Use case: Strategic allocation decisions, trend identification
  - Expected accuracy: Lower but still actionable for trend direction

### Model Architecture

**SARIMAX(1, 1, 1) x (0, 0, 0, 0)**
- AR(1): Autoregressive component captures momentum
- I(1): First-order differencing for stationarity
- MA(1): Moving average component for shock absorption
- No seasonality: Daily data doesn't exhibit strong seasonal patterns

### Evaluation Metrics

1. **RMSE (Root Mean Squared Error)**: Point prediction accuracy
2. **MAE (Mean Absolute Error)**: Average error magnitude
3. **R² Score**: Proportion of variance explained
4. **Directional Accuracy**: Percentage of correct up/down predictions (often most useful)

---

## Results & Insights

### Expected Performance Patterns

- **7-day forecasts** typically achieve R² > 0.85 with directional accuracy > 60%
- **14-day forecasts** show moderate degradation, R² around 0.70-0.80
- **30-day forecasts** have lower R² but still capture major trend shifts

### Key Findings

1. **Tech-Specific Indices Outperform**: NASDAQ and sector ETFs more predictive than S&P 500
2. **Crypto-Tech Correlation**: Bitcoin/Ethereum strong leading indicators during AI boom
3. **Fundamentals Matter**: Sector profit margins and ROE provide medium-term signals
4. **Regime Shifts Critical**: AI Boom and Fed Hike periods capture structural market changes

### Model Limitations

- SARIMAX assumes linear relationships (may miss non-linear patterns)
- Extreme events (black swans) not well-captured by historical data
- Parameter stability: Variable relationships change over time (recommend periodic retraining)

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
   - Perform feature selection via VIF analysis
   - Train 3 SARIMAX models (7, 14, 30-day horizons)
   - Generate performance metrics and visualizations
   - Save predictions to CSV

---

## Future Improvements

1. **Ensemble Methods**: Combine SARIMAX with XGBoost/LSTM for non-linear patterns
2. **Rolling Window Retraining**: Weekly/monthly model updates for changing conditions
3. **Probabilistic Forecasts**: Quantile regression for confidence intervals
4. **Regime-Conditional Models**: Separate models for bull/bear/sideways markets
5. **Real-Time Integration**: News sentiment, earnings surprises, Fed announcements

---