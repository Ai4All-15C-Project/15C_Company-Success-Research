"""
AI Tech Stock Analysis Dashboard
================================
Streamlit UI for exploring tech stock data and SARIMAX forecasting analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="AI Tech Stock Analysis",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    }
    .stMetric {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# Cache data loading
@st.cache_data
def load_data():
    """Load and prepare the dataset."""
    df = pd.read_csv('Datasets/Tech_Stock_Data_SEC_Cleaned_SARIMAX.csv', parse_dates=['Date'])
    df.set_index('Date', inplace=True)
    return df

@st.cache_data
def get_stock_categories():
    """Define stock categories."""
    return {
        'Big Tech': ['AAPL', 'MSFT', 'GOOGL', 'META', 'AMZN'],
        'AI & Hardware': ['NVDA', 'AMD', 'INTC'],
        'Cybersecurity': ['CRWD', 'PANW', 'ZS', 'OKTA'],
        'Cloud & Software': ['CRM', 'ORCL', 'NOW', 'ADBE', 'DDOG', 'NET'],
        'E-commerce & Fintech': ['SHOP', 'PYPL', 'TWLO', 'MDB']
    }

@st.cache_data
def calculate_ai_tech_index(df):
    """Calculate the AI Tech Index (average of AI/Hardware stocks)."""
    ai_hardware = ['NVDA', 'AMD', 'INTC']
    available = [col for col in ai_hardware if col in df.columns]
    if available:
        return df[available].mean(axis=1)
    return None

def create_stock_chart(df, stocks, title):
    """Create an interactive stock price chart."""
    fig = go.Figure()
    for stock in stocks:
        if stock in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df[stock],
                name=stock, mode='lines',
                hovertemplate=f'{stock}: $%{{y:.2f}}<extra></extra>'
            ))
    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title='Price ($)',
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        height=500
    )
    return fig

def create_correlation_heatmap(df, columns):
    """Create a correlation heatmap."""
    corr_matrix = df[columns].corr()
    fig = px.imshow(
        corr_matrix,
        labels=dict(color="Correlation"),
        x=columns, y=columns,
        color_continuous_scale='RdBu_r',
        zmin=-1, zmax=1
    )
    fig.update_layout(height=600)
    return fig

def run_rolling_forecast(df, target_col, exog_vars, train_size=0.8, order=(1,1,1), seasonal_order=(1,0,1,5)):
    """Run rolling 1-step-ahead forecast."""
    y = df[target_col].dropna()
    
    # Prepare exogenous variables
    exog = df[exog_vars].loc[y.index] if exog_vars else None
    
    split_idx = int(len(y) * train_size)
    
    rolling_predictions = []
    rolling_actuals = []
    rolling_dates = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i in range(split_idx, len(y)):
        try:
            # Get training data up to this point
            y_train = y.iloc[:i]
            exog_train = exog.iloc[:i] if exog is not None else None
            exog_test = exog.iloc[[i]] if exog is not None else None
            
            # Fit model
            model = SARIMAX(y_train, exog=exog_train, order=order, seasonal_order=seasonal_order,
                          enforce_stationarity=False, enforce_invertibility=False)
            results = model.fit(disp=False, maxiter=50)
            
            # Predict next step
            pred = results.forecast(steps=1, exog=exog_test)
            
            rolling_predictions.append(pred.values[0])
            rolling_actuals.append(y.iloc[i])
            rolling_dates.append(y.index[i])
            
            # Update progress
            progress = (i - split_idx + 1) / (len(y) - split_idx)
            progress_bar.progress(progress)
            status_text.text(f'Forecasting: {i - split_idx + 1}/{len(y) - split_idx} steps')
            
        except Exception as e:
            continue
    
    progress_bar.empty()
    status_text.empty()
    
    return pd.Series(rolling_predictions, index=rolling_dates), pd.Series(rolling_actuals, index=rolling_dates)

def calculate_naive_baseline(actuals):
    """Calculate naive baseline (yesterday's value as prediction)."""
    return actuals.shift(1).dropna(), actuals.iloc[1:]

# Main app
def main():
    st.title("📈 AI Tech Stock Analysis Dashboard")
    st.markdown("---")
    
    # Load data
    try:
        df = load_data()
        stock_categories = get_stock_categories()
        all_stocks = [s for stocks in stock_categories.values() for s in stocks]
        available_stocks = [s for s in all_stocks if s in df.columns]
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.info("Please ensure the dataset file exists at 'Datasets/Tech_Stock_Data_SEC_Cleaned_SARIMAX.csv'")
        return
    
    # Sidebar
    st.sidebar.title("🔧 Controls")
    
    page = st.sidebar.selectbox(
        "Select Analysis",
        ["📊 Overview", "📈 Stock Explorer", "🔗 Correlations", "🤖 Forecast Model", "📉 Market Regimes", "📋 SEC Fundamentals"]
    )
    
    # Overview Page
    if page == "📊 Overview":
        st.header("Market Overview")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        # Calculate AI Tech Index
        ai_index = calculate_ai_tech_index(df)
        
        with col1:
            if ai_index is not None:
                current_val = ai_index.iloc[-1]
                prev_val = ai_index.iloc[-2]
                change = ((current_val - prev_val) / prev_val) * 100
                st.metric("AI Tech Index", f"${current_val:.2f}", f"{change:+.2f}%")
        
        with col2:
            if 'SP500' in df.columns:
                current = df['SP500'].iloc[-1]
                prev = df['SP500'].iloc[-2]
                change = ((current - prev) / prev) * 100
                st.metric("S&P 500", f"{current:.2f}", f"{change:+.2f}%")
        
        with col3:
            if 'VIX' in df.columns:
                current = df['VIX'].iloc[-1]
                st.metric("VIX (Volatility)", f"{current:.2f}")
        
        with col4:
            if 'NVDA' in df.columns:
                current = df['NVDA'].iloc[-1]
                prev = df['NVDA'].iloc[-2]
                change = ((current - prev) / prev) * 100
                st.metric("NVIDIA", f"${current:.2f}", f"{change:+.2f}%")
        
        st.markdown("---")
        
        # AI Tech Index Chart
        st.subheader("AI Tech Index Over Time")
        if ai_index is not None:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=ai_index.index, y=ai_index.values, 
                                     mode='lines', name='AI Tech Index',
                                     line=dict(color='#1f77b4', width=2)))
            
            # Add regime highlights
            if 'Pandemic_Period' in df.columns:
                pandemic = df[df['Pandemic_Period'] == 1]
                if len(pandemic) > 0:
                    fig.add_vrect(x0=pandemic.index.min(), x1=pandemic.index.max(),
                                 fillcolor="red", opacity=0.1, line_width=0,
                                 annotation_text="Pandemic", annotation_position="top left")
            
            if 'AI_Boom_Period' in df.columns:
                ai_boom = df[df['AI_Boom_Period'] == 1]
                if len(ai_boom) > 0:
                    fig.add_vrect(x0=ai_boom.index.min(), x1=ai_boom.index.max(),
                                 fillcolor="green", opacity=0.1, line_width=0,
                                 annotation_text="AI Boom", annotation_position="top left")
            
            fig.update_layout(height=400, xaxis_title='Date', yaxis_title='Index Value')
            st.plotly_chart(fig, use_container_width=True)
        
        # Sector Performance
        st.subheader("Sector Performance")
        col1, col2 = st.columns(2)
        
        with col1:
            # Calculate sector returns
            sector_returns = {}
            for sector, stocks in stock_categories.items():
                available = [s for s in stocks if s in df.columns]
                if available:
                    start_prices = df[available].iloc[0].mean()
                    end_prices = df[available].iloc[-1].mean()
                    sector_returns[sector] = ((end_prices - start_prices) / start_prices) * 100
            
            if sector_returns:
                returns_df = pd.DataFrame({
                    'Sector': list(sector_returns.keys()),
                    'Return (%)': list(sector_returns.values())
                }).sort_values('Return (%)', ascending=True)
                
                fig = px.bar(returns_df, x='Return (%)', y='Sector', orientation='h',
                            color='Return (%)', color_continuous_scale='RdYlGn')
                fig.update_layout(height=300, title='Total Return by Sector')
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Data summary
            st.markdown("**Dataset Summary**")
            st.write(f"📅 Date Range: {df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}")
            st.write(f"📊 Total Trading Days: {len(df):,}")
            st.write(f"📈 Stocks Tracked: {len(available_stocks)}")
            st.write(f"🔢 Total Features: {len(df.columns)}")
    
    # Stock Explorer Page
    elif page == "📈 Stock Explorer":
        st.header("Stock Explorer")
        
        # Stock selection
        col1, col2 = st.columns([1, 3])
        
        with col1:
            sector = st.selectbox("Select Sector", list(stock_categories.keys()))
            selected_stocks = st.multiselect(
                "Select Stocks",
                [s for s in stock_categories[sector] if s in df.columns],
                default=[s for s in stock_categories[sector] if s in df.columns][:3]
            )
            
            date_range = st.date_input(
                "Date Range",
                value=[df.index.min().date(), df.index.max().date()],
                min_value=df.index.min().date(),
                max_value=df.index.max().date()
            )
            
            normalize = st.checkbox("Normalize Prices (Start = 100)", value=False)
        
        with col2:
            if selected_stocks and len(date_range) == 2:
                plot_df = df.loc[str(date_range[0]):str(date_range[1]), selected_stocks]
                
                if normalize:
                    plot_df = (plot_df / plot_df.iloc[0]) * 100
                
                fig = create_stock_chart(plot_df.reset_index(), selected_stocks, f"{sector} Stocks")
                fig.update_traces(x=plot_df.index)
                st.plotly_chart(fig, use_container_width=True)
                
                # Statistics
                st.subheader("Stock Statistics")
                stats_df = pd.DataFrame({
                    'Stock': selected_stocks,
                    'Current Price': [df[s].iloc[-1] for s in selected_stocks],
                    'Total Return (%)': [((df[s].iloc[-1] - df[s].iloc[0]) / df[s].iloc[0]) * 100 for s in selected_stocks],
                    'Volatility (Std)': [df[s].std() for s in selected_stocks],
                    'Min': [df[s].min() for s in selected_stocks],
                    'Max': [df[s].max() for s in selected_stocks]
                })
                st.dataframe(stats_df.style.format({
                    'Current Price': '${:.2f}',
                    'Total Return (%)': '{:+.2f}%',
                    'Volatility (Std)': '{:.2f}',
                    'Min': '${:.2f}',
                    'Max': '${:.2f}'
                }), use_container_width=True)
    
    # Correlations Page
    elif page == "🔗 Correlations":
        st.header("Correlation Analysis")
        
        tab1, tab2 = st.tabs(["Stock Correlations", "Market Indicators"])
        
        with tab1:
            st.subheader("Stock Price Correlations")
            corr_stocks = st.multiselect(
                "Select stocks for correlation matrix",
                available_stocks,
                default=available_stocks[:8]
            )
            
            if len(corr_stocks) >= 2:
                fig = create_correlation_heatmap(df, corr_stocks)
                st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.subheader("Market Indicator Correlations")
            market_cols = ['SP500', 'NASDAQ', 'VIX', 'Treasury_10Y', 'Dollar_Index', 'Gold', 'Oil_WTI', 'Bitcoin']
            available_market = [c for c in market_cols if c in df.columns]
            
            if available_market:
                fig = create_correlation_heatmap(df, available_market)
                st.plotly_chart(fig, use_container_width=True)
    
    # Forecast Model Page
    elif page == "🤖 Forecast Model":
        st.header("SARIMAX Forecast Model")
        
        st.markdown("""
        This section demonstrates rolling 1-step-ahead forecasting using SARIMAX models.
        
        **⚠️ Important Note on R² Interpretation:**
        - High R² (>0.95) in 1-step forecasts can be misleading
        - Stock prices are highly autocorrelated (today ≈ yesterday)
        - We compare against a **naive baseline** to show true predictive value
        """)
        
        st.markdown("---")
        
        # Calculate AI Tech Index for forecasting
        ai_index = calculate_ai_tech_index(df)
        
        if ai_index is not None:
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.subheader("Model Configuration")
                
                train_size = st.slider("Training Data %", 50, 90, 80) / 100
                
                use_exog = st.checkbox("Use Exogenous Variables", value=False)
                
                exog_vars = []
                if use_exog:
                    available_exog = ['SP500', 'VIX', 'NASDAQ', 'Treasury_10Y']
                    available_exog = [v for v in available_exog if v in df.columns]
                    exog_vars = st.multiselect("Select Exogenous Variables", available_exog, default=available_exog[:2])
                
                run_forecast = st.button("🚀 Run Forecast", type="primary")
            
            with col2:
                if run_forecast:
                    st.subheader("Running Rolling Forecast...")
                    
                    # Prepare data
                    forecast_df = df.copy()
                    forecast_df['AI_Tech_Index'] = ai_index
                    forecast_df = forecast_df.dropna(subset=['AI_Tech_Index'] + exog_vars)
                    
                    # Run rolling forecast
                    predictions, actuals = run_rolling_forecast(
                        forecast_df, 'AI_Tech_Index', exog_vars if use_exog else [],
                        train_size=train_size
                    )
                    
                    if len(predictions) > 0:
                        # Calculate metrics
                        r2 = r2_score(actuals, predictions)
                        rmse = np.sqrt(mean_squared_error(actuals, predictions))
                        mae = mean_absolute_error(actuals, predictions)
                        
                        # Naive baseline
                        naive_pred, naive_actual = calculate_naive_baseline(actuals)
                        r2_naive = r2_score(naive_actual, naive_pred)
                        rmse_naive = np.sqrt(mean_squared_error(naive_actual, naive_pred))
                        
                        # Display metrics
                        st.subheader("Model Performance")
                        
                        met1, met2, met3 = st.columns(3)
                        with met1:
                            st.metric("R² Score", f"{r2:.4f}", f"{r2 - r2_naive:+.4f} vs Naive")
                        with met2:
                            st.metric("RMSE", f"{rmse:.2f}", f"{rmse - rmse_naive:+.2f} vs Naive")
                        with met3:
                            st.metric("MAE", f"{mae:.2f}")
                        
                        # Warning about high R²
                        if r2 > 0.95:
                            st.warning("""
                            **⚠️ High R² Warning:** The R² of {:.2%} appears very high, but this is common for 
                            1-step-ahead forecasts due to autocorrelation. The naive baseline (predict yesterday's value) 
                            achieves R² = {:.2%}. The model improvement is only **{:.4f}**.
                            """.format(r2, r2_naive, r2 - r2_naive))
                        
                        # Plot
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=actuals.index, y=actuals.values,
                                               name='Actual', mode='lines', line=dict(color='black')))
                        fig.add_trace(go.Scatter(x=predictions.index, y=predictions.values,
                                               name='SARIMAX Prediction', mode='lines', 
                                               line=dict(color='green', dash='dash')))
                        fig.add_trace(go.Scatter(x=naive_actual.index, y=naive_pred.values,
                                               name='Naive Baseline', mode='lines',
                                               line=dict(color='red', dash='dot', width=1)))
                        
                        fig.update_layout(
                            title='Rolling 1-Step-Ahead Forecast vs Naive Baseline',
                            xaxis_title='Date',
                            yaxis_title='AI Tech Index',
                            height=500,
                            legend=dict(orientation='h', yanchor='bottom', y=1.02)
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Residual analysis
                        st.subheader("Residual Analysis")
                        residuals = actuals - predictions
                        
                        fig_res = make_subplots(rows=1, cols=2, subplot_titles=['Residual Distribution', 'Residuals Over Time'])
                        
                        fig_res.add_trace(go.Histogram(x=residuals.values, nbinsx=30, name='Residuals'),
                                         row=1, col=1)
                        fig_res.add_trace(go.Scatter(x=residuals.index, y=residuals.values, 
                                                    mode='lines', name='Residuals'),
                                         row=1, col=2)
                        fig_res.add_hline(y=0, line_dash='dash', line_color='red', row=1, col=2)
                        
                        fig_res.update_layout(height=350, showlegend=False)
                        st.plotly_chart(fig_res, use_container_width=True)
                else:
                    st.info("👆 Configure parameters and click 'Run Forecast' to generate predictions")
    
    # Market Regimes Page
    elif page == "📉 Market Regimes":
        st.header("Market Regime Analysis")
        
        regime_cols = ['Pandemic_Period', 'AI_Boom_Period', 'Fed_Hike_Period', 'Tech_Bear_2022', 'Banking_Crisis_2023']
        available_regimes = [c for c in regime_cols if c in df.columns]
        
        if available_regimes:
            # Regime timeline
            st.subheader("Market Regime Timeline")
            
            ai_index = calculate_ai_tech_index(df)
            if ai_index is not None:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=ai_index.index, y=ai_index.values,
                                        mode='lines', name='AI Tech Index', line=dict(color='blue')))
                
                colors = {'Pandemic_Period': 'red', 'AI_Boom_Period': 'green', 
                         'Fed_Hike_Period': 'orange', 'Tech_Bear_2022': 'purple',
                         'Banking_Crisis_2023': 'brown'}
                
                for regime in available_regimes:
                    regime_data = df[df[regime] == 1]
                    if len(regime_data) > 0:
                        fig.add_vrect(
                            x0=regime_data.index.min(), x1=regime_data.index.max(),
                            fillcolor=colors.get(regime, 'gray'), opacity=0.2,
                            line_width=0, annotation_text=regime.replace('_', ' '),
                            annotation_position="top left"
                        )
                
                fig.update_layout(height=500, title='AI Tech Index with Market Regimes')
                st.plotly_chart(fig, use_container_width=True)
            
            # Regime statistics
            st.subheader("Performance by Regime")
            
            regime_stats = []
            for regime in available_regimes:
                regime_data = df[df[regime] == 1]
                if len(regime_data) > 0 and 'SP500' in df.columns:
                    sp_returns = regime_data['SP500'].pct_change().dropna()
                    regime_stats.append({
                        'Regime': regime.replace('_', ' '),
                        'Days': len(regime_data),
                        'Avg Daily Return (%)': sp_returns.mean() * 100,
                        'Volatility (%)': sp_returns.std() * 100,
                        'Max Drawdown (%)': ((regime_data['SP500'] / regime_data['SP500'].cummax()) - 1).min() * 100
                    })
            
            if regime_stats:
                stats_df = pd.DataFrame(regime_stats)
                st.dataframe(stats_df.style.format({
                    'Avg Daily Return (%)': '{:+.4f}%',
                    'Volatility (%)': '{:.4f}%',
                    'Max Drawdown (%)': '{:.2f}%'
                }), use_container_width=True)
    
    # SEC Fundamentals Page
    elif page == "📋 SEC Fundamentals":
        st.header("SEC Fundamentals Analysis")
        
        sec_cols = ['Sector_Total_Revenue', 'Sector_Total_NetIncome', 'Sector_Profit_Margin', 
                   'Sector_ROE', 'Sector_Total_Assets', 'Sector_Total_Equity']
        available_sec = [c for c in sec_cols if c in df.columns]
        
        if available_sec:
            st.subheader("Sector Fundamental Metrics Over Time")
            
            selected_metric = st.selectbox("Select Metric", available_sec)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df.index, y=df[selected_metric],
                                    mode='lines', name=selected_metric))
            fig.update_layout(
                title=f'{selected_metric.replace("_", " ")} Over Time',
                xaxis_title='Date',
                yaxis_title=selected_metric.replace('_', ' '),
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Correlation with stock prices
            st.subheader("Fundamental vs Stock Price Correlations")
            
            if available_sec and available_stocks:
                correlations = {}
                for sec_col in available_sec:
                    stock_corrs = {}
                    for stock in available_stocks[:5]:  # Top 5 stocks
                        corr = df[sec_col].corr(df[stock])
                        stock_corrs[stock] = corr
                    correlations[sec_col] = stock_corrs
                
                corr_df = pd.DataFrame(correlations).T
                fig = px.imshow(corr_df, color_continuous_scale='RdBu_r', zmin=-1, zmax=1,
                               labels=dict(color='Correlation'))
                fig.update_layout(height=400, title='SEC Fundamentals vs Top Stock Correlations')
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("SEC fundamental data columns not found in the dataset.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
        <p>AI Tech Stock Analysis Dashboard | Data Source: Yahoo Finance & SEC EDGAR</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
