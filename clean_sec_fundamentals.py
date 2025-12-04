"""
Clean SEC Fundamentals CSV for SARIMAX Model
=============================================
This script cleans the Tech_Stock_Data_with_SEC_Fundamentals.csv file to make it 
suitable for time series analysis with SARIMAX models.

Key cleaning operations:
1. Parse dates and set as index
2. Remove weekend/non-trading days (optional)
3. Handle missing values appropriately
4. Forward-fill SEC fundamentals (quarterly data)
5. Interpolate or fill stock prices and indicators
6. Remove columns with too many missing values
7. Handle infinite values
8. Ensure proper data types
9. Create lagged features if needed
10. Output a clean, model-ready dataset
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


def load_data(filepath: str) -> pd.DataFrame:
    """Load the raw CSV file."""
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    print(f"Raw data shape: {df.shape}")
    return df


def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Parse Date column and set as index."""
    print("\nParsing dates...")
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date')
    df = df.sort_index()
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    return df


def remove_weekends(df: pd.DataFrame, keep_weekends: bool = False) -> pd.DataFrame:
    """Remove weekend rows (non-trading days) if specified."""
    if keep_weekends:
        print("\nKeeping weekend data...")
        return df
    
    print("\nRemoving weekend/non-trading days...")
    initial_rows = len(df)
    # Day of week: 0=Monday, 6=Sunday
    df = df[df.index.dayofweek < 5]
    removed = initial_rows - len(df)
    print(f"Removed {removed} weekend rows. Remaining: {len(df)} rows")
    return df


def analyze_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze and report missing values."""
    print("\nAnalyzing missing values...")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing Count': missing,
        'Missing %': missing_pct
    }).sort_values('Missing %', ascending=False)
    
    print("\nTop 20 columns with missing values:")
    print(missing_df[missing_df['Missing Count'] > 0].head(20).to_string())
    return missing_df


def drop_high_missing_columns(df: pd.DataFrame, threshold: float = 50.0) -> pd.DataFrame:
    """Drop columns with missing values above threshold percentage."""
    print(f"\nDropping columns with >{threshold}% missing values...")
    missing_pct = (df.isnull().sum() / len(df)) * 100
    cols_to_drop = missing_pct[missing_pct > threshold].index.tolist()
    
    if cols_to_drop:
        print(f"Dropping {len(cols_to_drop)} columns: {cols_to_drop[:10]}..." if len(cols_to_drop) > 10 else f"Dropping columns: {cols_to_drop}")
        df = df.drop(columns=cols_to_drop)
    else:
        print("No columns to drop.")
    
    return df


def identify_column_types(df: pd.DataFrame) -> dict:
    """Identify different types of columns for appropriate handling."""
    column_types = {
        'stock_prices': [],  # Individual stock tickers
        'indices': [],       # Market indices
        'etfs': [],          # ETF prices
        'volatility': [],    # VIX and volatility measures
        'treasuries': [],    # Treasury yields
        'commodities': [],   # Gold, Silver, Oil, etc.
        'crypto': [],        # Bitcoin, Ethereum
        'sec_fundamentals': [],  # SEC data columns
        'technical': [],     # Technical indicators
        'calendar': [],      # Day, week, month features
        'binary': [],        # Binary/boolean columns
        'other': []
    }
    
    stock_tickers = ['NVDA', 'AMD', 'INTC', 'GOOGL', 'MSFT', 'CRM', 'ORCL', 'OKTA', 
                     'ZS', 'CRWD', 'DDOG', 'NET', 'AAPL', 'META', 'AMZN', 'IBM', 
                     'ADBE', 'NOW', 'SHOP', 'TWLO', 'MDB', 'PYPL', 'ANET', 'PANW']
    
    indices = ['SP500', 'NASDAQ', 'Dow_Jones', 'Russell2000', 'PHLX_Semi_Index']
    
    etf_keywords = ['ETF', 'ARK_', 'Taiwan_ETF', 'China_Large_Cap', 'Japan_ETF', 
                    'South_Korea_ETF', 'Cloud_', 'Tech_Sector', 'Software_', 
                    'Semiconductor_', 'Real_Estate_', 'Cybersecurity_']
    
    volatility_keywords = ['VIX', 'VVIX', 'Vol', 'Volatility']
    
    treasury_keywords = ['Treasury_', 'Yield_Curve']
    
    commodities = ['Gold', 'Silver', 'Copper', 'Oil_WTI', 'Natural_Gas', 'Dollar_Index', 'Dollar_']
    
    crypto = ['Bitcoin', 'Ethereum', 'Crypto_']
    
    sec_keywords = ['Sector_', 'Assets', 'Revenue', 'NetIncome', 'Equity', 
                    'Profit_Margin', 'ROE', 'Asset_Turnover', 'Profitable_Pct']
    
    calendar = ['Day_of_Week', 'Day_of_Month', 'Week_of_Year', 'Month', 'Quarter', 
                'Year', 'Is_Month_End', 'Is_Quarter_End', 'Earnings_Season', 'Options_Expiry']
    
    binary_keywords = ['Is_', 'Period', 'Regime', 'Fear', 'Inverted']
    
    technical = ['MA', 'Momentum', 'Ratio', 'Spread', 'vs_']
    
    for col in df.columns:
        if col in stock_tickers:
            column_types['stock_prices'].append(col)
        elif col in indices:
            column_types['indices'].append(col)
        elif any(kw in col for kw in etf_keywords):
            column_types['etfs'].append(col)
        elif any(kw in col for kw in volatility_keywords):
            column_types['volatility'].append(col)
        elif any(kw in col for kw in treasury_keywords):
            column_types['treasuries'].append(col)
        elif any(kw in col for kw in commodities):
            column_types['commodities'].append(col)
        elif any(kw in col for kw in crypto):
            column_types['crypto'].append(col)
        elif any(kw in col for kw in sec_keywords):
            column_types['sec_fundamentals'].append(col)
        elif col in calendar:
            column_types['calendar'].append(col)
        elif any(kw in col for kw in binary_keywords):
            column_types['binary'].append(col)
        elif any(kw in col for kw in technical):
            column_types['technical'].append(col)
        else:
            column_types['other'].append(col)
    
    print("\nColumn type identification:")
    for ctype, cols in column_types.items():
        if cols:
            print(f"  {ctype}: {len(cols)} columns")
    
    return column_types


def handle_missing_values(df: pd.DataFrame, column_types: dict) -> pd.DataFrame:
    """Handle missing values based on column types."""
    print("\nHandling missing values by column type...")
    
    # Stock prices, indices, ETFs: forward fill then backward fill
    price_cols = (column_types['stock_prices'] + column_types['indices'] + 
                  column_types['etfs'] + column_types['commodities'] + 
                  column_types['crypto'])
    price_cols = [c for c in price_cols if c in df.columns]
    if price_cols:
        df[price_cols] = df[price_cols].ffill().bfill()
        print(f"  Forward/backward filled {len(price_cols)} price columns")
    
    # Treasury yields: forward fill
    treasury_cols = [c for c in column_types['treasuries'] if c in df.columns]
    if treasury_cols:
        df[treasury_cols] = df[treasury_cols].ffill().bfill()
        print(f"  Forward/backward filled {len(treasury_cols)} treasury columns")
    
    # Volatility: forward fill
    vol_cols = [c for c in column_types['volatility'] if c in df.columns]
    if vol_cols:
        df[vol_cols] = df[vol_cols].ffill().bfill()
        print(f"  Forward/backward filled {len(vol_cols)} volatility columns")
    
    # SEC Fundamentals: forward fill (quarterly data)
    sec_cols = [c for c in column_types['sec_fundamentals'] if c in df.columns]
    if sec_cols:
        df[sec_cols] = df[sec_cols].ffill()
        print(f"  Forward filled {len(sec_cols)} SEC fundamental columns")
    
    # Technical indicators: forward fill then interpolate
    tech_cols = [c for c in column_types['technical'] if c in df.columns]
    if tech_cols:
        df[tech_cols] = df[tech_cols].ffill().bfill()
        print(f"  Forward/backward filled {len(tech_cols)} technical columns")
    
    # Binary columns: fill with 0
    binary_cols = [c for c in column_types['binary'] if c in df.columns]
    if binary_cols:
        df[binary_cols] = df[binary_cols].fillna(0)
        print(f"  Filled {len(binary_cols)} binary columns with 0")
    
    # Calendar columns: should be derived from date, fill with appropriate values
    calendar_cols = [c for c in column_types['calendar'] if c in df.columns]
    if calendar_cols:
        df[calendar_cols] = df[calendar_cols].ffill().bfill()
        print(f"  Forward/backward filled {len(calendar_cols)} calendar columns")
    
    # Other columns: forward fill
    other_cols = [c for c in column_types['other'] if c in df.columns]
    if other_cols:
        df[other_cols] = df[other_cols].ffill().bfill()
        print(f"  Forward/backward filled {len(other_cols)} other columns")
    
    return df


def handle_infinite_values(df: pd.DataFrame) -> pd.DataFrame:
    """Replace infinite values with NaN, then fill."""
    print("\nHandling infinite values...")
    inf_count = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
    if inf_count > 0:
        print(f"  Found {inf_count} infinite values, replacing with NaN")
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.ffill().bfill()
    else:
        print("  No infinite values found")
    return df


def ensure_data_types(df: pd.DataFrame, column_types: dict) -> pd.DataFrame:
    """Ensure proper data types for all columns."""
    print("\nEnsuring proper data types...")
    
    # Binary columns should be int
    binary_cols = [c for c in column_types['binary'] if c in df.columns]
    for col in binary_cols:
        df[col] = df[col].astype(int)
    
    # Calendar columns should be int
    calendar_cols = [c for c in column_types['calendar'] if c in df.columns]
    for col in calendar_cols:
        if col in df.columns:
            df[col] = df[col].astype(int)
    
    print(f"  Converted {len(binary_cols)} binary columns to int")
    print(f"  Converted {len(calendar_cols)} calendar columns to int")
    
    return df


def create_target_variable(df: pd.DataFrame, forecast_horizon: int = 1) -> pd.DataFrame:
    """Create target variables for ALL stocks (future returns and direction).
    
    This allows the SARIMAX model to be applied to any of the 24 stocks.
    """
    print(f"\nCreating target variables for all stocks (horizon={forecast_horizon})...")
    
    # All 24 stock tickers
    all_stocks = ['NVDA', 'AMD', 'INTC', 'GOOGL', 'MSFT', 'CRM', 'ORCL', 'NOW',
                  'OKTA', 'ZS', 'CRWD', 'NET', 'PANW', 'AAPL', 'META', 'AMZN', 
                  'IBM', 'ADBE', 'SHOP', 'TWLO', 'MDB', 'DDOG', 'PYPL', 'ANET']
    
    created_count = 0
    for stock in all_stocks:
        if stock in df.columns:
            # Create future returns
            df[f'{stock}_future_return'] = df[stock].pct_change(forecast_horizon).shift(-forecast_horizon)
            # Create direction (up/down)
            df[f'{stock}_future_direction'] = (df[f'{stock}_future_return'] > 0).astype(int)
            created_count += 1
    
    print(f"  Created future_return and future_direction for {created_count} stocks")
    
    return df


def add_lagged_features(df: pd.DataFrame, columns: list, lags: list = [1, 5, 20]) -> pd.DataFrame:
    """Add lagged features for specified columns."""
    print(f"\nAdding lagged features for {len(columns)} columns with lags {lags}...")
    
    for col in columns:
        if col in df.columns:
            for lag in lags:
                df[f'{col}_lag{lag}'] = df[col].shift(lag)
    
    return df


def add_rolling_features(df: pd.DataFrame, columns: list, windows: list = [5, 20]) -> pd.DataFrame:
    """Add rolling mean and std features."""
    print(f"\nAdding rolling features for {len(columns)} columns with windows {windows}...")
    
    for col in columns:
        if col in df.columns:
            for window in windows:
                df[f'{col}_ma{window}'] = df[col].rolling(window=window).mean()
                df[f'{col}_std{window}'] = df[col].rolling(window=window).std()
    
    return df


def select_features_for_sarimax(df: pd.DataFrame, column_types: dict, 
                                 include_sec: bool = True,
                                 include_technical: bool = True,
                                 include_macro: bool = True) -> pd.DataFrame:
    """Select relevant features for SARIMAX modeling."""
    print("\nSelecting features for SARIMAX model...")
    
    selected_cols = []
    
    # Always include main indices
    selected_cols.extend([c for c in column_types['indices'] if c in df.columns])
    
    # Include volatility measures
    if include_macro:
        selected_cols.extend([c for c in column_types['volatility'] if c in df.columns])
        selected_cols.extend([c for c in column_types['treasuries'] if c in df.columns])
    
    # Include SEC fundamentals
    if include_sec:
        selected_cols.extend([c for c in column_types['sec_fundamentals'] if c in df.columns])
    
    # Include technical indicators
    if include_technical:
        selected_cols.extend([c for c in column_types['technical'] if c in df.columns])
    
    # Include some key binary indicators
    key_binary = ['Pandemic_Period', 'AI_Boom_Period', 'Fed_Hike_Period', 
                  'Tech_Bear_2022', 'Banking_Crisis_2023', 'High_Volatility_Regime',
                  'Yield_Curve_Inverted', 'Earnings_Season']
    selected_cols.extend([c for c in key_binary if c in df.columns])
    
    # Remove duplicates while preserving order
    selected_cols = list(dict.fromkeys(selected_cols))
    
    print(f"  Selected {len(selected_cols)} exogenous features")
    
    return df[selected_cols]


def final_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Final cleaning: remove any remaining NaN rows at edges."""
    print("\nFinal cleaning...")
    initial_rows = len(df)
    df = df.dropna()
    removed = initial_rows - len(df)
    print(f"  Removed {removed} rows with remaining NaN values")
    print(f"  Final dataset shape: {df.shape}")
    return df


def save_cleaned_data(df: pd.DataFrame, output_path: str):
    """Save the cleaned dataframe to CSV."""
    print(f"\nSaving cleaned data to {output_path}...")
    df.to_csv(output_path)
    print(f"  Saved {len(df)} rows and {len(df.columns)} columns")


def generate_data_report(df: pd.DataFrame, output_path: str):
    """Generate a summary report of the cleaned data."""
    report_path = output_path.replace('.csv', '_report.txt')
    
    with open(report_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("CLEANED DATA REPORT\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"Date Range: {df.index.min()} to {df.index.max()}\n")
        f.write(f"Total Rows: {len(df)}\n")
        f.write(f"Total Columns: {len(df.columns)}\n\n")
        
        f.write("Column Summary:\n")
        f.write("-" * 40 + "\n")
        for col in df.columns:
            f.write(f"{col}: {df[col].dtype}, min={df[col].min():.4f}, max={df[col].max():.4f}\n")
        
        f.write("\n" + "=" * 60 + "\n")
        f.write("Missing Values After Cleaning:\n")
        f.write("-" * 40 + "\n")
        missing = df.isnull().sum()
        for col, count in missing.items():
            if count > 0:
                f.write(f"{col}: {count} missing\n")
        if missing.sum() == 0:
            f.write("No missing values!\n")
    
    print(f"\nReport saved to {report_path}")


def main():
    """Main function to run the cleaning pipeline."""
    print("=" * 60)
    print("SEC FUNDAMENTALS DATA CLEANING PIPELINE")
    print("=" * 60)
    
    # Configuration
    input_file = Path("Datasets/Tech_Stock_Data_with_SEC_Fundamentals.csv")
    output_file = Path("Datasets/Tech_Stock_Data_SEC_Cleaned_SARIMAX.csv")
    
    # Additional output for just exogenous features
    exog_output_file = Path("Datasets/SARIMAX_Exogenous_Features.csv")
    
    # Cleaning parameters
    KEEP_WEEKENDS = False
    MISSING_THRESHOLD = 50.0  # Drop columns with >50% missing
    ADD_LAGS = True
    ADD_ROLLING = True
    LAG_COLUMNS = ['SP500', 'NASDAQ', 'VIX']  # Key columns for lagging
    
    # Step 1: Load data
    df = load_data(str(input_file))
    
    # Step 2: Parse dates
    df = parse_dates(df)
    
    # Step 3: Remove weekends (optional)
    df = remove_weekends(df, keep_weekends=KEEP_WEEKENDS)
    
    # Step 4: Analyze missing values
    analyze_missing_values(df)
    
    # Step 5: Identify column types
    column_types = identify_column_types(df)
    
    # Step 6: Drop high missing columns
    df = drop_high_missing_columns(df, threshold=MISSING_THRESHOLD)
    
    # Step 7: Handle missing values
    # Re-identify after dropping columns
    column_types = identify_column_types(df)
    df = handle_missing_values(df, column_types)
    
    # Step 8: Handle infinite values
    df = handle_infinite_values(df)
    
    # Step 9: Ensure data types
    df = ensure_data_types(df, column_types)
    
    # Step 10: Create target variables for ALL stocks
    df = create_target_variable(df)
    
    # Step 11: Add lagged features (optional)
    if ADD_LAGS:
        df = add_lagged_features(df, LAG_COLUMNS)
    
    # Step 12: Add rolling features (optional)
    if ADD_ROLLING:
        df = add_rolling_features(df, LAG_COLUMNS)
    
    # Step 13: Final clean
    df_full = final_clean(df.copy())
    
    # Step 14: Save full cleaned dataset
    save_cleaned_data(df_full, str(output_file))
    
    # Step 15: Create and save exogenous features subset
    df_exog = select_features_for_sarimax(df_full, column_types)
    save_cleaned_data(df_exog, str(exog_output_file))
    
    # Step 16: Generate report
    generate_data_report(df_full, str(output_file))
    
    print("\n" + "=" * 60)
    print("CLEANING COMPLETE!")
    print("=" * 60)
    print(f"\nOutput files:")
    print(f"  1. Full cleaned dataset: {output_file}")
    print(f"  2. Exogenous features only: {exog_output_file}")
    print(f"  3. Data report: {output_file.with_suffix('')}_report.txt")
    
    return df_full


if __name__ == "__main__":
    df_cleaned = main()
