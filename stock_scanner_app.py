import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import plotly.graph_objects as go

# Set page config
st.set_page_config(
    page_title="Stock Market Scanner",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Symbol explanation
SYMBOL_EXPLANATION = """
🚀🚀 = Both Daily and Weekly Bullish
🕣🕣 = Daily Bearish, Weekly Bullish (Clock)
⚠️⚠️ = Daily Bullish, Weekly Bearish (Caution)
💀💀 = Both Daily and Weekly Bearish
✔️ = EMAs are aligned (7 EMA > 11 EMA > 21 EMA) on Daily Timeframe
❌ = EMAs are NOT aligned on Daily Timeframe
"""

# Hardcoded tickers with proper Yahoo Finance symbols and their display names
INDICES = {
    # Major US Indices
    "^GSPC": "S&P 500",
    "^IXIC": "NASDAQ Composite",
    "^DJI": "Dow Jones Industrial",
    "^RUT": "Russell 2000",
    # European Indices
    "^FTSE": "UK FTSE 100",
    "^GDAXI": "German DAX",
    "^FCHI": "French CAC 40",
    "^STOXX50E": "Euro STOXX 50",
    "^IBEX": "Spanish IBEX 35",
    "^AEX": "Netherlands AEX",
    # Asia Pacific Indices
    "^AXJO": "Australian ASX 200",
    "^N225": "Japanese Nikkei 225",
    "^HSI": "Hong Kong Hang Seng",
    "000001.SS": "Shanghai Composite",
}

FOREX = {
    # Major pairs
    "EURUSD=X": "EUR/USD",
    "GBPUSD=X": "GBP/USD",
    "USDJPY=X": "USD/JPY",
    "AUDUSD=X": "AUD/USD",
    "NZDUSD=X": "NZD/USD",
    # Cross pairs
    "EURGBP=X": "EUR/GBP",
    "GBPJPY=X": "GBP/JPY",
    "EURJPY=X": "EUR/JPY",
    "GBPAUD=X": "GBP/AUD",
    "GBPCAD=X": "GBP/CAD",
    "GBPCHF=X": "GBP/CHF",
    "CHFJPY=X": "CHF/JPY",
}

# Commodities
COMMODITIES = {
    "GC=F": "Gold",
    "SI=F": "Silver", 
    "HG=F": "Copper",
    "NG=F": "Natural Gas",
    "BZ=F": "Brent Crude Oil",
    "CL=F": "WTI Crude Oil"
}

# US Treasuries
TREASURIES = {
    "^IRX": "US 3M Treasury",
    "^FVX": "US 5Y Treasury",
    "^TNX": "US 10Y Treasury",
    "^TYX": "US 30Y Treasury"
}

# All categories in a dictionary
TICKER_CATEGORIES = {
    "INDICES": INDICES,
    "FOREX": FOREX,
    "COMMODITIES": COMMODITIES,
    "TREASURIES": TREASURIES
}

@st.cache_data(ttl=600)
def fetch_stock_data(ticker, period="6mo", interval="1d"):
    """
    Fetch stock data for a given ticker
    """
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period, interval=interval)
        return hist
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")
        return pd.DataFrame()

def calculate_rsi(data, window=14):
    """
    Calculate RSI (Relative Strength Index) using the standard method
    """
    if data.empty or len(data) < window*2:
        return pd.Series([np.nan] * len(data))
    
    # Calculate price changes
    delta = data['Close'].diff()
    
    # Separate gains and losses
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    # First average gain and loss
    first_avg_gain = gain.iloc[1:window+1].mean()
    first_avg_loss = loss.iloc[1:window+1].mean()
    
    # Initialize lists with NaNs for the first window periods
    avg_gains = [np.nan] * window
    avg_losses = [np.nan] * window
    
    # Set the first average gain and loss
    avg_gains.append(first_avg_gain)
    avg_losses.append(first_avg_loss)
    
    # Calculate subsequent values
    for i in range(window+1, len(delta)):
        avg_gain = (avg_gains[-1] * (window-1) + gain.iloc[i]) / window
        avg_loss = (avg_losses[-1] * (window-1) + loss.iloc[i]) / window
        avg_gains.append(avg_gain)
        avg_losses.append(avg_loss)
    
    # Convert to Series
    avg_gain_series = pd.Series(avg_gains, index=data.index)
    avg_loss_series = pd.Series(avg_losses, index=data.index)
    
    # Calculate RS and RSI
    rs = avg_gain_series / avg_loss_series
    rsi = 100 - (100 / (1 + rs))
    
    # Handle division by zero (when avg_loss is 0)
    rsi = rsi.replace([np.inf, -np.inf], 100)
    
    return rsi

def calculate_ema(data, spans=[7, 11, 21]):
    """
    Calculate EMAs for given spans
    """
    emas = {}
    for span in spans:
        emas[f'EMA_{span}'] = data['Close'].ewm(span=span, adjust=False).mean()
    return emas

def check_ema_alignment(emas):
    """
    Check if EMAs are aligned (7 EMA > 11 EMA > 21 EMA)
    """
    if all(pd.isna(ema.iloc[-1]) for ema in emas.values()):
        return False
    
    return emas['EMA_7'].iloc[-1] > emas['EMA_11'].iloc[-1] > emas['EMA_21'].iloc[-1]

def calculate_bullish_score(daily_rsi, weekly_rsi, ema_aligned):
    """
    Calculate a bullish score to sort results (higher is more bullish)
    """
    score = 0
    
    # RSI components (0-200 points)
    score += daily_rsi  # 0-100 points
    score += weekly_rsi  # 0-100 points
    
    # EMA alignment bonus (50 points)
    if ema_aligned:
        score += 50
    
    return score

def scan_ticker(ticker, display_name):
    """
    Scan a ticker and return analysis based on criteria
    """
    try:
        # Fetch data
        daily_data = fetch_stock_data(ticker, period="3mo", interval="1d")
        weekly_data = fetch_stock_data(ticker, period="1y", interval="1wk")
        
        if daily_data.empty or weekly_data.empty or len(daily_data) < 30 or len(weekly_data) < 14:
            return {"display_name": display_name, "error": "Insufficient data", "score": -1000}
        
        # Calculate indicators
        daily_rsi = calculate_rsi(daily_data)
        weekly_rsi = calculate_rsi(weekly_data)
        emas = calculate_ema(daily_data)
        
        # Get latest values
        latest_daily_rsi = daily_rsi.iloc[-1]
        latest_weekly_rsi = weekly_rsi.iloc[-1]
        ema_aligned = check_ema_alignment(emas)
        
        # Determine conditions
        daily_bullish = latest_daily_rsi > 50
        weekly_bullish = latest_weekly_rsi > 50
        
        # Prepare status strings
        daily_status = "Bullish" if daily_bullish else "Bearish"
        weekly_status = "Bullish" if weekly_bullish else "Bearish"
        ema_status = "✔️" if ema_aligned else "❌"
        
        # Determine emoji
        if daily_bullish and weekly_bullish:
            emoji = "🚀🚀"
        elif not daily_bullish and weekly_bullish:
            emoji = "🕣🕣"
        elif daily_bullish and not weekly_bullish:
            emoji = "⚠️⚠️"
        else:
            emoji = "💀💀"
            
        # Calculate bullish score for sorting
        bullish_score = calculate_bullish_score(latest_daily_rsi, latest_weekly_rsi, ema_aligned)
        
        # Current price
        current_price = daily_data['Close'].iloc[-1]
        
        # Return result as dictionary for easier sorting
        return {
            "display_name": display_name,
            "ticker": ticker,
            "emoji": emoji,
            "daily_status": daily_status,
            "weekly_status": weekly_status,
            "ema_status": ema_status,
            "daily_rsi": latest_daily_rsi,
            "weekly_rsi": latest_weekly_rsi,
            "price": current_price,
            "score": bullish_score,
            "daily_data": daily_data,
            "weekly_data": weekly_data,
            "emas": emas,
            "error": None
        }
    
    except Exception as e:
        return {
            "display_name": display_name,
            "ticker": ticker,
            "error": str(e),
            "score": -1000
        }

def create_chart(result):
    """
    Create an interactive chart for a given ticker
    """
    if result.get("error") or "daily_data" not in result:
        return None
    
    # Create price chart
    fig = go.Figure()
    
    # Add candlestick chart
    fig.add_trace(go.Candlestick(
        x=result["daily_data"].index,
        open=result["daily_data"]['Open'],
        high=result["daily_data"]['High'],
        low=result["daily_data"]['Low'],
        close=result["daily_data"]['Close'],
        name="Price"
    ))
    
    # Add EMAs
    emas = result["emas"]
    for span in [7, 11, 21]:
        fig.add_trace(go.Scatter(
            x=emas[f'EMA_{span}'].index,
            y=emas[f'EMA_{span}'],
            mode='lines',
            line=dict(width=2),
            name=f'EMA {span}'
        ))
    
    # Update layout
    fig.update_layout(
        title=f"{result['display_name']} - Price and EMAs",
        xaxis_title="Date",
        yaxis_title="Price",
        height=500,
        template="plotly_dark",
        xaxis_rangeslider_visible=False
    )
    
    return fig

def main():
    # Sidebar configuration
    st.sidebar.title("Stock Market Scanner")
    st.sidebar.write("Configure your scanner settings below:")
    
    # Category selection
    selected_categories = st.sidebar.multiselect(
        "Select Categories to Scan",
        options=list(TICKER_CATEGORIES.keys()),
        default=list(TICKER_CATEGORIES.keys())
    )
    
    # Scan interval
    refresh_interval = st.sidebar.slider(
        "Auto-refresh Interval (minutes)",
        min_value=1,
        max_value=60,
        value=5,
        step=1
    )
    
    # Display setting
    show_charts = st.sidebar.checkbox("Show Charts for Top Performers", value=False)
    
    # About section
    st.sidebar.markdown("---")
    st.sidebar.info(
        "This app scans stock and forex markets for trading signals based on RSI and EMA indicators. "
        "It automatically refreshes at your selected interval."
    )
    
    # Main page content
    st.title("📊 Market Scanner Dashboard")
    
    # Display legend
    with st.expander("📝 Signal Legend", expanded=False):
        st.markdown(SYMBOL_EXPLANATION)
    
    # Current time
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.subheader(f"Checking Trading Signals at {current_time}")
    
    # Progress bar for scanning
    if selected_categories:
        # Count total tickers to scan
        total_tickers = sum(len(TICKER_CATEGORIES[cat]) for cat in selected_categories)
        
        with st.spinner(f"Scanning {total_tickers} tickers..."):
            # Create a progress bar
            progress_bar = st.progress(0)
            
            # Collect all results
            all_results = []
            tickers_scanned = 0
            
            # Scan each selected category
            for category in selected_categories:
                if category in TICKER_CATEGORIES:
                    category_tickers = TICKER_CATEGORIES[category]
                    
                    # Scan each ticker in the category
                    for ticker, name in category_tickers.items():
                        result = scan_ticker(ticker, name)
                        result["category"] = category
                        all_results.append(result)
                        
                        # Update progress
                        tickers_scanned += 1
                        progress_bar.progress(tickers_scanned / total_tickers)
            
            # Remove progress bar when done
            progress_bar.empty()
        
        # Sort results by bullish score (most bullish first)
        all_results.sort(key=lambda x: x.get("score", -1000), reverse=True)
        
        # Create DataFrame for display
        valid_results = [r for r in all_results if not r.get("error")]
        
        if valid_results:
            # Create a dataframe for display
            display_data = []
            for r in valid_results:
                display_data.append({
                    "Signal": r["emoji"],
                    "Instrument": r["display_name"],
                    "Daily": r["daily_status"],
                    "Weekly": r["weekly_status"],
                    "EMA": r["ema_status"],
                    "Price": f"{r['price']:.4f}",
                    "Daily RSI": f"{r['daily_rsi']:.2f}",
                    "Weekly RSI": f"{r['weekly_rsi']:.2f}"
                })
            
            df = pd.DataFrame(display_data)
            
            # Display the results table
            st.dataframe(df, use_container_width=True, height=400)
            
            # Show charts for top performers if requested
            if show_charts and valid_results:
                st.subheader("Charts for Top Performers")
                
                # Show top 3 (or fewer if not enough)
                top_n = min(3, len(valid_results))
                
                # Create columns for charts
                cols = st.columns(top_n)
                
                # Display each chart in its own column
                for i in range(top_n):
                    with cols[i]:
                        chart = create_chart(valid_results[i])
                        if chart:
                            st.plotly_chart(chart, use_container_width=True)
        else:
            st.warning("No valid results found. Check your internet connection or try different tickers.")
        
        # Show errors if any
        errors = [r for r in all_results if r.get("error")]
        if errors:
            with st.expander("Show Errors", expanded=False):
                for e in errors:
                    st.error(f"{e['display_name']}: {e['error']}")
    
    else:
        st.warning("Please select at least one category to scan.")
    
    # Set up auto-refresh
    st.markdown("---")
    refresh_text = st.empty()
    refresh_text.text(f"Next refresh in {refresh_interval} minutes (at {(datetime.now() + timedelta(minutes=refresh_interval)).strftime('%H:%M:%S')})")
    
    # Schedule the next refresh
    if "last_refresh" not in st.session_state:
        st.session_state.last_refresh = datetime.now()
    
    # Check if it's time to refresh
    if datetime.now() >= st.session_state.last_refresh + timedelta(minutes=refresh_interval):
        st.session_state.last_refresh = datetime.now()
        st.experimental_rerun()

if __name__ == "__main__":
    main()
