# 15C_Company-Success-Research-

# Tech Stock Data - SARIMAX Forecasting

## Overview

This dataset contains 22 tech company stocks and 60+ exogenous variables for time series forecasting using SARIMAX.

**Date Range:** 2019-01-01 to 2025-11-21  
**Frequency:** Daily (includes weekends - will need cleaning)  
**Companies:** All public before 2019 (5+ years of history)

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

## Key Variables for SARIMAX

### High Priority (Leading Indicators)

1. **Yield_Curve_Slope** - Leading recession indicator
2. **Vol_of_Vol_Ratio** - Market uncertainty measure
3. **Semi_vs_Tech_Ratio** - AI hardware strength relative to broader tech
4. **Crypto_Tech_Corr_20d** - Risk sentiment correlation
5. **Small_vs_Large_Caps** - Market breadth indicator
6. **Gold_Oil_Ratio** - Macro sentiment (fear vs growth)

### Medium Priority

7. VIX - Volatility index
8. Dollar_Strength - USD relative strength
9. Credit_Spread_Proxy - Credit market stress
10. AI_Boom_Period - Post-ChatGPT regime
11. Fed_Hike_Period - Rate sensitivity
12. NASDAQ_Momentum - Short-term tech momentum

---