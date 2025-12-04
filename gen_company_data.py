#!/usr/bin/env python3
"""
AI/Tech Stock Data Collector - SARIMAX Ready with SEC Fundamentals
Collects stock data, exogenous variables, and SEC financial data for time series forecasting
"""

import pandas as pd
import numpy as np
import yfinance as yf
import requests
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configuration
START_DATE = '2019-01-01'
END_DATE = '2025-11-21'
USER_EMAIL = "mohammadeltawilcs@gmail.com"  # Required for SEC API

# US Tech companies only (have SEC filings)
ai_tickers = [
    'NVDA', 'AMD', 'INTC', 'GOOGL', 'MSFT', 'CRM', 'ORCL',
    'OKTA', 'ZS', 'CRWD', 'DDOG', 'NET',
    'AAPL', 'META', 'AMZN', 'IBM', 'ADBE', 'NOW', 'SHOP', 'TWLO', 'MDB',
    'PYPL', 'ANET', 'PANW'
]

# CIK mapping for SEC API (US companies only)
COMPANY_CIKS = {
    'NVDA': '0001045810', 'AMD': '0000002488', 'INTC': '0000050863',
    'GOOGL': '0001652044', 'MSFT': '0000789019', 'CRM': '0001108524',
    'ORCL': '0001341439', 'NOW': '0001373715', 'OKTA': '0001660134',
    'ZS': '0001713683', 'CRWD': '0001535527', 'NET': '0001477333',
    'PANW': '0001327567', 'AAPL': '0000320193', 'META': '0001326801',
    'AMZN': '0001018724', 'IBM': '0000051143', 'ADBE': '0000796343',
    'SHOP': '0001594805', 'TWLO': '0001447669', 'MDB': '0001441816',
    'DDOG': '0001561550', 'PYPL': '0001633917', 'ANET': '0001596532',
}

# Exogenous variables
market_indices = ['^GSPC', '^IXIC', '^DJI', '^RUT', '^VIX', '^VVIX', '^VXN']
rates_tickers = ['^TNX', '^IRX', '^FVX', '^TYX']
sector_etfs = ['XLK', 'SMH', 'SOXX', 'IGV', 'SKYY', 'HACK', 'BOTZ', 'ROBO', 
               'CLOU', 'ARKK', 'ARKQ', 'ARKW', 'QQQ', 'QTEC']
macro_tickers = ['DX-Y.NYB', 'GC=F', 'SI=F', 'HG=F', 'CL=F', 'NG=F']
crypto_tickers = ['BTC-USD', 'ETH-USD']
# Removed non-US companies: ASML, TSM, BABA, BIDU (no SEC filings)
intl_etfs = ['EWT', 'EWJ', 'EWY', 'FXI']  # Keep ETFs only
risk_tickers = ['HYG', 'LQD', 'TLT', 'EMB', 'VNQ']

all_tickers = list(set(ai_tickers + market_indices + rates_tickers + sector_etfs + 
                       macro_tickers + crypto_tickers + intl_etfs + risk_tickers))

# ============================================================
# SEC API FUNCTIONS
# ============================================================
def fetch_sec_metric(cik, metric_name, user_email):
    """Fetch a specific financial metric from SEC API."""
    headers = {
        'User-Agent': f'AI4ALL Research Project {user_email}',
        'Accept-Encoding': 'gzip, deflate',
        'Host': 'data.sec.gov'
    }
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    
    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if 'us-gaap' in data['facts'] and metric_name in data['facts']['us-gaap']:
            metric_data = data['facts']['us-gaap'][metric_name]
            if 'USD' in metric_data['units']:
                df_metric = pd.DataFrame(metric_data['units']['USD'])
                df_metric = df_metric[df_metric['form'].isin(['10-Q', '10-K'])]
                df_metric['end'] = pd.to_datetime(df_metric['end'])
                df_metric = df_metric.sort_values('end').drop_duplicates(subset=['end'], keep='last')
                return df_metric.set_index('end')['val']
        return None
    except:
        return None

def fetch_all_sec_data(company_ciks, metrics, user_email):
    """Fetch SEC data for all companies."""
    print(f"\nFetching SEC fundamental data for {len(company_ciks)} companies...")
    sec_data = {}
    
    for ticker, cik in company_ciks.items():
        sec_data[ticker] = {}
        for metric in metrics:
            ts = fetch_sec_metric(cik, metric, user_email)
            if ts is not None and len(ts) > 0:
                sec_data[ticker][metric] = ts
        
        available = [m for m in metrics if m in sec_data[ticker]]
        if available:
            print(f"  ✓ {ticker}: {len(available)} metrics")
        time.sleep(0.15)  # Rate limiting
    
    return sec_data

def aggregate_sec_fundamentals(sec_data, metrics):
    """
    Aggregate SEC metrics into meaningful sector-level indicators.
    These serve as exogenous variables for a general (non-company-specific) model.
    """
    fundamental_df = pd.DataFrame()
    
    # Collect all company data per metric
    metric_data = {}
    for metric in metrics:
        dfs = []
        for ticker, data in sec_data.items():
            if metric in data and data[metric] is not None:
                ts = data[metric].copy()
                ts.name = ticker
                dfs.append(ts)
        
        if dfs:
            combined = pd.concat(dfs, axis=1)
            quarterly = combined.resample('Q').last()
            metric_data[metric] = quarterly
    
    # Create sector-level indicators
    if 'Assets' in metric_data:
        fundamental_df['Sector_Total_Assets'] = metric_data['Assets'].sum(axis=1)
        fundamental_df['Sector_Avg_Assets'] = metric_data['Assets'].mean(axis=1)
        fundamental_df['Sector_Assets_Growth'] = fundamental_df['Sector_Total_Assets'].pct_change() * 100
    
    if 'Revenues' in metric_data:
        fundamental_df['Sector_Total_Revenue'] = metric_data['Revenues'].sum(axis=1)
        fundamental_df['Sector_Avg_Revenue'] = metric_data['Revenues'].mean(axis=1)
        fundamental_df['Sector_Revenue_Growth'] = fundamental_df['Sector_Total_Revenue'].pct_change() * 100
    
    if 'NetIncomeLoss' in metric_data:
        fundamental_df['Sector_Total_NetIncome'] = metric_data['NetIncomeLoss'].sum(axis=1)
        fundamental_df['Sector_Avg_NetIncome'] = metric_data['NetIncomeLoss'].mean(axis=1)
        fundamental_df['Sector_NetIncome_Growth'] = fundamental_df['Sector_Total_NetIncome'].pct_change() * 100
    
    if 'StockholdersEquity' in metric_data:
        fundamental_df['Sector_Total_Equity'] = metric_data['StockholdersEquity'].sum(axis=1)
        fundamental_df['Sector_Avg_Equity'] = metric_data['StockholdersEquity'].mean(axis=1)
    
    # Calculate meaningful financial ratios (sector-wide)
    if 'Revenues' in metric_data and 'NetIncomeLoss' in metric_data:
        total_rev = metric_data['Revenues'].sum(axis=1)
        total_income = metric_data['NetIncomeLoss'].sum(axis=1)
        fundamental_df['Sector_Profit_Margin'] = (total_income / total_rev) * 100
    
    if 'NetIncomeLoss' in metric_data and 'StockholdersEquity' in metric_data:
        total_income = metric_data['NetIncomeLoss'].sum(axis=1)
        total_equity = metric_data['StockholdersEquity'].sum(axis=1)
        fundamental_df['Sector_ROE'] = (total_income / total_equity) * 100
    
    if 'Revenues' in metric_data and 'Assets' in metric_data:
        total_rev = metric_data['Revenues'].sum(axis=1)
        total_assets = metric_data['Assets'].sum(axis=1)
        fundamental_df['Sector_Asset_Turnover'] = total_rev / total_assets
    
    # Count of profitable companies (health indicator)
    if 'NetIncomeLoss' in metric_data:
        profitable_count = (metric_data['NetIncomeLoss'] > 0).sum(axis=1)
        total_companies = metric_data['NetIncomeLoss'].notna().sum(axis=1)
        fundamental_df['Sector_Profitable_Pct'] = (profitable_count / total_companies) * 100
    
    return fundamental_df

print(f"Downloading {len(all_tickers)} tickers ({START_DATE} to {END_DATE})...")

# Batch download
batch_size = 20
batches = [all_tickers[i:i+batch_size] for i in range(0, len(all_tickers), batch_size)]
df_data = pd.DataFrame()
successful_tickers = []

for i, batch in enumerate(batches):
    try:
        data = yf.download(
            tickers=batch if len(batch) > 1 else batch[0],
            start=START_DATE, end=END_DATE,
            group_by='ticker' if len(batch) > 1 else None,
            auto_adjust=True, progress=False, threads=True
        )
        
        if not data.empty:
            if df_data.empty:
                df_data = pd.DataFrame(index=data.index)
            
            if len(batch) == 1:
                ticker = batch[0]
                if 'Close' in data.columns:
                    df_data[ticker] = data['Close']
                    successful_tickers.append(ticker)
            else:
                for ticker in batch:
                    try:
                        if len(data.columns.levels) > 1:
                            if ticker in data.columns.get_level_values(0):
                                if 'Close' in data[ticker].columns:
                                    df_data[ticker] = data[ticker]['Close']
                                else:
                                    df_data[ticker] = data[ticker].iloc[:, 3]
                                successful_tickers.append(ticker)
                        else:
                            if 'Close' in data.columns:
                                df_data[ticker] = data['Close']
                                successful_tickers.append(ticker)
                    except:
                        pass
    except:
        # Retry individually
        for ticker in batch:
            try:
                ticker_data = yf.download(ticker, start=START_DATE, end=END_DATE, 
                                        progress=False, auto_adjust=True)
                if not ticker_data.empty:
                    if df_data.empty:
                        df_data = pd.DataFrame(index=ticker_data.index)
                    df_data[ticker] = ticker_data['Close'] if 'Close' in ticker_data.columns else ticker_data.iloc[:, 0]
                    successful_tickers.append(ticker)
            except:
                pass

if df_data.empty:
    raise Exception("No data downloaded!")

print(f"Downloaded: {len(successful_tickers)}/{len(all_tickers)} tickers")

# Rename tickers
ticker_rename = {
    '^GSPC': 'SP500', '^IXIC': 'NASDAQ', '^DJI': 'Dow_Jones', '^RUT': 'Russell2000',
    '^VIX': 'VIX', '^VVIX': 'VVIX', '^VXN': 'NASDAQ_VIX',
    '^TNX': 'Treasury_10Y', '^IRX': 'Treasury_3M', '^FVX': 'Treasury_5Y', '^TYX': 'Treasury_30Y',
    'DX-Y.NYB': 'Dollar_Index', 'GC=F': 'Gold', 'SI=F': 'Silver', 'HG=F': 'Copper',
    'CL=F': 'Oil_WTI', 'NG=F': 'Natural_Gas', 'BTC-USD': 'Bitcoin', 'ETH-USD': 'Ethereum',
    'XLK': 'Tech_Sector_ETF', 'SMH': 'Semiconductor_ETF', 'SOXX': 'PHLX_Semi_Index',
    'IGV': 'Software_ETF', 'SKYY': 'Cloud_Computing_ETF', 'HACK': 'Cybersecurity_ETF',
    'BOTZ': 'AI_Robotics_ETF', 'ROBO': 'Global_Robotics_ETF', 'CLOU': 'Global_Cloud_ETF',
    'ARKK': 'ARK_Innovation', 'ARKQ': 'ARK_Autonomous_Tech', 'ARKW': 'ARK_Next_Gen_Internet',
    'QQQ': 'NASDAQ_100_ETF', 'QTEC': 'First_Trust_NASDAQ',
    'HYG': 'High_Yield_Bonds', 'LQD': 'Investment_Grade_Bonds', 'TLT': 'Long_Term_Treasury',
    'EMB': 'Emerging_Markets_Bonds', 'VNQ': 'Real_Estate_ETF',
    'EWT': 'Taiwan_ETF', 'EWJ': 'Japan_ETF', 'EWY': 'South_Korea_ETF', 'FXI': 'China_Large_Cap_ETF'
}
df_data.rename(columns=ticker_rename, inplace=True)

# Engineer features
print("Engineering features...")

if 'Treasury_10Y' in df_data.columns and 'Treasury_3M' in df_data.columns:
    df_data['Yield_Curve_Slope'] = df_data['Treasury_10Y'] - df_data['Treasury_3M']
    df_data['Yield_Curve_Inverted'] = (df_data['Yield_Curve_Slope'] < 0).astype(int)

if 'VIX' in df_data.columns:
    df_data['High_Volatility_Regime'] = (df_data['VIX'] > 20).astype(int)
    df_data['Extreme_Fear'] = (df_data['VIX'] > 30).astype(int)
    df_data['VIX_MA20'] = df_data['VIX'].rolling(20).mean()
    df_data['VIX_vs_MA'] = df_data['VIX'] / df_data['VIX_MA20']

if 'VVIX' in df_data.columns and 'VIX' in df_data.columns:
    df_data['Vol_of_Vol_Ratio'] = df_data['VVIX'] / df_data['VIX']

if 'Semiconductor_ETF' in df_data.columns and 'Tech_Sector_ETF' in df_data.columns:
    df_data['Semi_vs_Tech_Ratio'] = df_data['Semiconductor_ETF'] / df_data['Tech_Sector_ETF']

if 'Russell2000' in df_data.columns and 'SP500' in df_data.columns:
    df_data['Small_vs_Large_Caps'] = df_data['Russell2000'] / df_data['SP500']

if 'NASDAQ' in df_data.columns:
    df_data['NASDAQ_MA5'] = df_data['NASDAQ'].rolling(5).mean()
    df_data['NASDAQ_MA20'] = df_data['NASDAQ'].rolling(20).mean()
    df_data['NASDAQ_Momentum'] = df_data['NASDAQ_MA5'] / df_data['NASDAQ_MA20']

if 'Bitcoin' in df_data.columns and 'NASDAQ' in df_data.columns:
    df_data['Crypto_Tech_Corr_20d'] = df_data['Bitcoin'].rolling(20).corr(df_data['NASDAQ'])

if 'Dollar_Index' in df_data.columns:
    df_data['Dollar_MA20'] = df_data['Dollar_Index'].rolling(20).mean()
    df_data['Dollar_Strength'] = df_data['Dollar_Index'] / df_data['Dollar_MA20']

if 'Gold' in df_data.columns and 'Oil_WTI' in df_data.columns:
    df_data['Gold_Oil_Ratio'] = df_data['Gold'] / df_data['Oil_WTI']

if 'High_Yield_Bonds' in df_data.columns and 'Long_Term_Treasury' in df_data.columns:
    df_data['Credit_Spread_Proxy'] = df_data['High_Yield_Bonds'] / df_data['Long_Term_Treasury']

# Add temporal features
df_data = df_data.reset_index()
df_data.rename(columns={'index': 'Date'}, inplace=True)
df_data['Date'] = pd.to_datetime(df_data['Date'])

df_data['Day_of_Week'] = df_data['Date'].dt.dayofweek
df_data['Day_of_Month'] = df_data['Date'].dt.day
df_data['Week_of_Year'] = df_data['Date'].dt.isocalendar().week
df_data['Month'] = df_data['Date'].dt.month
df_data['Quarter'] = df_data['Date'].dt.quarter
df_data['Year'] = df_data['Date'].dt.year
df_data['Is_Month_End'] = df_data['Date'].dt.is_month_end.astype(int)
df_data['Is_Quarter_End'] = df_data['Date'].dt.is_quarter_end.astype(int)

# Market regime indicators
df_data['Pandemic_Period'] = ((df_data['Date'] >= '2020-03-01') & (df_data['Date'] <= '2021-06-30')).astype(int)
df_data['AI_Boom_Period'] = (df_data['Date'] >= '2022-11-30').astype(int)
df_data['Fed_Hike_Period'] = ((df_data['Date'] >= '2022-03-01') & (df_data['Date'] <= '2023-07-31')).astype(int)
df_data['Tech_Bear_2022'] = ((df_data['Date'] >= '2022-01-01') & (df_data['Date'] <= '2022-10-31')).astype(int)
df_data['Banking_Crisis_2023'] = ((df_data['Date'] >= '2023-03-01') & (df_data['Date'] <= '2023-05-31')).astype(int)
df_data['Earnings_Season'] = df_data['Month'].isin([1, 4, 7, 10]).astype(int)
df_data['Options_Expiry_Week'] = ((df_data['Day_of_Month'] >= 15) & (df_data['Day_of_Month'] <= 21) & (df_data['Day_of_Week'] == 4)).astype(int)

df_data['Date'] = df_data['Date'].dt.strftime('%Y-%m-%d')

# Organize columns
ai_cols = [col for col in ai_tickers if col in df_data.columns]
market_cols = ['SP500', 'NASDAQ', 'Dow_Jones', 'Russell2000', 'VIX', 'VVIX', 'NASDAQ_VIX']
rate_cols = ['Treasury_10Y', 'Treasury_3M', 'Treasury_5Y', 'Treasury_30Y', 'Yield_Curve_Slope', 'Yield_Curve_Inverted']
etf_cols = [col for col in df_data.columns if 'ETF' in col or 'ARK' in col or 'QQQ' in col]
macro_cols = ['Dollar_Index', 'Dollar_Strength', 'Gold', 'Silver', 'Copper', 'Oil_WTI', 'Natural_Gas', 'Gold_Oil_Ratio']
crypto_cols = ['Bitcoin', 'Ethereum', 'Crypto_Tech_Corr_20d']
regime_cols = ['Pandemic_Period', 'AI_Boom_Period', 'Fed_Hike_Period', 'Tech_Bear_2022', 'Banking_Crisis_2023']
time_cols = ['Day_of_Week', 'Day_of_Month', 'Week_of_Year', 'Month', 'Quarter', 'Year', 'Is_Month_End', 'Is_Quarter_End', 'Earnings_Season', 'Options_Expiry_Week']

final_order = ['Date'] + ai_cols
for col_group in [market_cols, rate_cols, etf_cols, macro_cols, crypto_cols, regime_cols, time_cols]:
    final_order.extend([col for col in col_group if col in df_data.columns])
remaining_cols = [col for col in df_data.columns if col not in final_order]
final_order.extend(remaining_cols)

df_final = df_data[[col for col in final_order if col in df_data.columns]]

# Save stock data
output_file = 'Datasets/Tech_Stock_Data_2019-2025.csv'
df_final.to_csv(output_file, index=False)

print(f"\n✅ Stock data complete: {df_final.shape[0]:,} rows × {df_final.shape[1]:,} columns")
print(f"   Date range: {df_final['Date'].iloc[0]} to {df_final['Date'].iloc[-1]}")
print(f"   Saved to: {output_file}")

# ============================================================
# FETCH SEC FUNDAMENTAL DATA
# ============================================================
SEC_METRICS = ['Assets', 'Revenues', 'NetIncomeLoss', 'StockholdersEquity']

sec_data = fetch_all_sec_data(COMPANY_CIKS, SEC_METRICS, USER_EMAIL)
sec_fundamentals = aggregate_sec_fundamentals(sec_data, SEC_METRICS)

print(f"\n✓ SEC fundamentals aggregated: {sec_fundamentals.shape}")

# Merge SEC data with stock data (forward-fill quarterly to daily)
df_final_with_sec = df_final.copy()
df_final_with_sec['Date'] = pd.to_datetime(df_final_with_sec['Date'])
df_final_with_sec.set_index('Date', inplace=True)

for col in sec_fundamentals.columns:
    daily_series = sec_fundamentals[col].reindex(df_final_with_sec.index, method='ffill')
    df_final_with_sec[col] = daily_series

df_final_with_sec.reset_index(inplace=True)
df_final_with_sec['Date'] = df_final_with_sec['Date'].dt.strftime('%Y-%m-%d')

# Save enhanced dataset with SEC fundamentals
output_file_sec = 'Datasets/Tech_Stock_Data_with_SEC_Fundamentals.csv'
df_final_with_sec.to_csv(output_file_sec, index=False)

print(f"\n✅ SEC-enhanced data complete: {df_final_with_sec.shape[0]:,} rows × {df_final_with_sec.shape[1]:,} columns")
print(f"   Saved to: {output_file_sec}")
print(f"\n   SEC columns added:")
for col in sec_fundamentals.columns:
    print(f"     - {col}")

print("\n" + "="*60)
print("DONE! Two files created:")
print(f"  1. {output_file} (stock + market data)")
print(f"  2. {output_file_sec} (with SEC fundamentals)")
print("="*60)
