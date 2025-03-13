import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import time
from tickers import TICKER_CATEGORIES

# Set page config
st.set_page_config(
    page_title="MCSO Ticker Scanner",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS
st.markdown("""
<style>
    .bullish {
        color: #00FF00 !important;
        font-weight: bold;
    }
    .bearish {
        color: #888888 !important;
    }
    .stProgress > div > div > div > div {
        background-color: #0068c9;
    }
    .stDataFrame {
        width: 100%;
    }
    .title-container {
        background-color: #1a1a1a;
        padding: 20px;
        border-radius: 5px;
        margin-bottom: 20px;
        text-align: center;
    }
    .category-header {
        background-color: #2a2a2a;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# App title
st.markdown("""
<div class="title-container">
    <h1>📊 MCSO Ticker Scanner</h1>
    <p>Scans tickers based on the Monthly Cycle Swing Oscillator (MCSO) indicator</p>
</div>
""", unsafe_allow_html=True)

@st.cache_data(ttl=3600)  # Cache data for 1 hour
def calculate_mcso(ticker_symbol, period="1mo", interval="1d"):
    """
    Calculate MCSO (Monthly Cycle Swing Oscillator) for a given ticker.
    MCSO = ((close - month_low) / (month_high - month_low)) * 100
    """
    try:
        # Index tickers (starting with ^) need a longer period to ensure enough data
        actual_period = "3mo" if ticker_symbol.startswith('^') else period
        
        # Get historical data
        data = yf.download(ticker_symbol, period=actual_period, interval=interval, progress=False)
        
        # Check if data is empty or too small
        if data.empty or len(data) < 5:  # Need at least a few days of data
            return None, None, None, None
        
        # For index tickers, use more recent data matching the original requested period
        if ticker_symbol.startswith('^') and period == "1mo":
            # Keep approximately one month of trading days
            data = data.tail(22)  # ~22 trading days in a month
        
        # Calculate monthly high and low (using last 20 bars as in the script)
        month_high = data['High'].rolling(window=20).max().iloc[-1]
        month_low = data['Low'].rolling(window=20).min().iloc[-1]
        close = data['Close'].iloc[-1]
        
        # Convert to float to ensure scalar values
        try:
            month_high = float(month_high)
            month_low = float(month_low)
            close = float(close)
        except (TypeError, ValueError):
            # If conversion fails, it's likely we have invalid data
            return None, None, None, None
        
        # Calculate MCSO - check numeric equality properly for floats
        if abs(month_high - month_low) < 1e-6:  # Avoid division by zero
            return 0, close, month_low, month_high
        
        mcso = ((close - month_low) / (month_high - month_low)) * 100
        return mcso, close, month_low, month_high
    
    except Exception as e:
        st.error(f"Error calculating MCSO for {ticker_symbol}: {e}")
        return None, None, None, None

# Add new function to display a consolidated "All Tickers" table
def display_all_tickers_table(results_df, mcso_threshold):
    """Display a consolidated table with all tickers."""
    st.markdown("""
    <div class="category-header">
        <h3>All Tickers (Consolidated View)</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Create a styled dataframe for all tickers
    styled_df = results_df[['Category', 'Ticker', 'Name', 'MCSO', 'Current', 'Month Low', 'Month High', 'Status']].copy()
    
    # Sort by MCSO (high to low)
    styled_df = styled_df.sort_values(by='MCSO', ascending=False)
    
    # Format the display
    styled_df['MCSO'] = styled_df['MCSO'].round(2)
    styled_df['Current'] = styled_df['Current'].round(2)
    styled_df['Month Low'] = styled_df['Month Low'].round(2)
    styled_df['Month High'] = styled_df['Month High'].round(2)
    
    # Apply row styling based on MCSO value
    def style_rows(row):
        if row['MCSO'] >= mcso_threshold:
            return ['background-color: rgba(0, 128, 0, 0.1)'] * len(row)
        return ['background-color: transparent'] * len(row)
    
    styled_df = styled_df.style.apply(style_rows, axis=1)
    
    # Format the numeric columns
    styled_df = styled_df.format({
        'MCSO': '{:.2f}',
        'Current': '{:.2f}',
        'Month Low': '{:.2f}',
        'Month High': '{:.2f}'
    })
    
    st.dataframe(styled_df, use_container_width=True)

def scan_tickers(categories, min_mcso=50, progress_bar=None):
    """
    Scan tickers from selected categories and return those with calculated MCSO.
    """
    results = []
    
    # Get total ticker count for progress tracking
    ticker_dict = {}
    total_tickers = 0
    for category in categories:
        ticker_dict[category] = TICKER_CATEGORIES[category]
        total_tickers += len(TICKER_CATEGORIES[category])
    
    # Scan tickers
    processed = 0
    for category in categories:
        for ticker, name in ticker_dict[category].items():
            # Update progress
            processed += 1
            if progress_bar is not None:
                progress_bar.progress(processed / total_tickers, 
                                     text=f"Processing {ticker} ({processed}/{total_tickers})")
            
            # Calculate MCSO
            mcso, close, month_low, month_high = calculate_mcso(ticker)
            
            if mcso is not None:
                status = "BULLISH" if mcso >= min_mcso else "BEARISH"
                results.append({
                    'Category': category,
                    'Ticker': ticker,
                    'Name': name,
                    'MCSO': mcso,
                    'Current': close,
                    'Month Low': month_low,
                    'Month High': month_high,
                    'Status': status
                })
    
    # Convert to DataFrame
    if results:
        df = pd.DataFrame(results)
        return df
    else:
        return pd.DataFrame(columns=['Category', 'Ticker', 'Name', 'MCSO', 'Current', 'Month Low', 'Month High', 'Status'])

def display_mcso_chart(data):
    """Display a histogram of MCSO values"""
    if len(data) > 0:
        fig = go.Figure()
        
        # Add histogram trace for bullish
        bullish_data = data[data['MCSO'] >= 50]['MCSO']
        if len(bullish_data) > 0:
            fig.add_trace(go.Histogram(
                x=bullish_data,
                nbinsx=10,
                name="Bullish",
                marker_color='rgba(0, 255, 0, 0.6)'
            ))
        
        # Add histogram trace for bearish
        bearish_data = data[data['MCSO'] < 50]['MCSO']
        if len(bearish_data) > 0:
            fig.add_trace(go.Histogram(
                x=bearish_data,
                nbinsx=10,
                name="Bearish",
                marker_color='rgba(150, 150, 150, 0.6)'
            ))
        
        # Update layout
        fig.update_layout(
            title="MCSO Distribution",
            xaxis_title="MCSO Value",
            yaxis_title="Count",
            barmode='overlay',
            bargap=0.1,
            height=400,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Add vertical line at the MCSO threshold
        fig.add_shape(
            type="line",
            x0=50, y0=0,
            x1=50, y1=1,
            yref="paper",
            line=dict(color="white", width=2, dash="dash"),
        )
        
        # Add annotation for the line
        fig.add_annotation(
            x=50, y=1,
            yref="paper",
            text="MCSO = 50",
            showarrow=True,
            arrowhead=1,
            ax=0, ay=-40
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No data available for chart.")

# Sidebar
st.sidebar.title("Scan Options")

# Category selection
all_categories = list(TICKER_CATEGORIES.keys())
default_categories = ["INDICES"]  # Default to indices
selected_categories = st.sidebar.multiselect(
    "Select ticker categories to scan:",
    all_categories,
    default=default_categories
)

# MCSO threshold
mcso_threshold = st.sidebar.slider(
    "MCSO Threshold (50+ is Bullish):",
    0, 100, 50, 5
)

# Period selection
period_options = {
    "1 Month": "1mo",
    "3 Months": "3mo", 
    "6 Months": "6mo",
    "1 Year": "1y"
}
selected_period = st.sidebar.selectbox(
    "Select data period:",
    list(period_options.keys())
)

# Sort options
sort_options = {
    "MCSO (High to Low)": ("MCSO", False),
    "MCSO (Low to High)": ("MCSO", True),
    "Alphabetical (A-Z)": ("Name", True),
    "Alphabetical (Z-A)": ("Name", False),
    "Category": ("Category", True)
}
sort_by = st.sidebar.selectbox(
    "Sort results by:",
    list(sort_options.keys()),
    index=0
)

# Button to run the scan
if st.sidebar.button("Run Scan", type="primary"):
    if not selected_categories:
        st.warning("Please select at least one category to scan.")
    else:
        # Show progress bar
        progress_bar = st.progress(0, text="Starting scan...")
        
        # Run the scan
        results_df = scan_tickers(
            selected_categories, 
            mcso_threshold,
            progress_bar
        )
        
        # Sort results
        sort_col, sort_asc = sort_options[sort_by]
        results_df = results_df.sort_values(by=sort_col, ascending=sort_asc)
        
        # Clear progress bar
        progress_bar.empty()
        
        # Display results
        if len(results_df) > 0:
            # Display summary statistics
            bullish_count = len(results_df[results_df['MCSO'] >= mcso_threshold])
            bearish_count = len(results_df[results_df['MCSO'] < mcso_threshold])
            total_count = len(results_df)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Tickers", total_count)
            with col2:
                st.metric("Bullish (MCSO ≥ 50)", bullish_count)
            with col3:
                st.metric("Bearish (MCSO < 50)", bearish_count)
            
                        ## Display MCSO distribution chart
            display_mcso_chart(results_df)
            
            # Display the consolidated "All Tickers" table
            display_all_tickers_table(results_df, mcso_threshold)
            
            # Display results by category
            st.subheader("Scan Results")
            
            # Display results by category
            st.subheader("Scan Results")
            
            # Add download button for CSV
            csv = results_df.to_csv(index=False)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            st.download_button(
                label="Download Results as CSV",
                data=csv,
                file_name=f"mcso_scan_{timestamp}.csv",
                mime="text/csv",
            )
            
            # Display tables by category
            for category in sorted(results_df['Category'].unique()):
                cat_df = results_df[results_df['Category'] == category]
                
                st.markdown(f"""
                <div class="category-header">
                    <h3>{category} ({len(cat_df)} tickers)</h3>
                </div>
                """, unsafe_allow_html=True)
                
                # Create a styled dataframe
                styled_df = cat_df[['Ticker', 'Name', 'MCSO', 'Current', 'Month Low', 'Month High', 'Status']].copy()
                
                # Format the display
                styled_df['MCSO'] = styled_df['MCSO'].round(2)
                styled_df['Current'] = styled_df['Current'].round(2)
                styled_df['Month Low'] = styled_df['Month Low'].round(2)
                styled_df['Month High'] = styled_df['Month High'].round(2)
                
                # Apply row styling based on MCSO value
                def style_rows(row):
                    if row['MCSO'] >= mcso_threshold:
                        return ['background-color: rgba(0, 128, 0, 0.1)'] * len(row)
                    return ['background-color: transparent'] * len(row)
                
                styled_df = styled_df.style.apply(style_rows, axis=1)
                
                # Format the numeric columns
                styled_df = styled_df.format({
                    'MCSO': '{:.2f}',
                    'Current': '{:.2f}',
                    'Month Low': '{:.2f}',
                    'Month High': '{:.2f}'
                })
                
                st.dataframe(styled_df, use_container_width=True)
        else:
            st.warning("No tickers found matching the criteria.")

# About section
with st.sidebar.expander("About MCSO Indicator"):
    st.markdown("""
    ### Monthly Cycle Swing Oscillator (MCSO)
    
    The MCSO measures a security's position within its monthly price range.
    
    **Formula:**  
    ```
    MCSO = ((close - month_low) / (month_high - month_low)) * 100
    ```
    
    **Interpretation:**
    - **MCSO ≥ 50**: Price is closer to the monthly high (Bullish)
    - **MCSO < 50**: Price is closer to the monthly low (Bearish)
    
    **Trading Strategy:**
    - Consider buying when MCSO crosses above 50
    - Consider selling when MCSO crosses below 50
    - Higher MCSO values indicate stronger bullish momentum
    """)

# Default content when app starts
if not st.session_state.get('scan_run', False):
    st.info("👈 Select categories and click 'Run Scan' to analyze tickers")
    
    # Show quick start guide
    with st.expander("Quick Start Guide"):
        st.markdown("""
        ### How to use this app:
        
        1. **Select ticker categories** in the sidebar (start with INDICES for an overview)
        2. **Set the MCSO threshold** (default is 50)
        3. **Click 'Run Scan'** to analyze the selected tickers
        4. **View the results** sorted by category
        5. **Download the CSV** for further analysis
        
        ### Interpreting results:
        
        - **Green background** indicates bullish tickers (MCSO ≥ threshold)
        - **No background** indicates bearish tickers (MCSO < threshold)
        - Higher MCSO values suggest stronger bullish momentum
        
        ### Tips:
        
        - Start by scanning INDICES to get a market overview
        - Then focus on specific sectors showing strength
        - The histogram shows the distribution of MCSO values
        """)
