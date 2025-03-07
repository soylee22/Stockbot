import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set page config
st.set_page_config(
    page_title="Market Signal Scanner",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    /* Main container */
    .main {
        background-color: #0e1117;
        color: #ffffff;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #ffffff;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Signal legend */
    .legend-container {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 20px;
        border-left: 5px solid #4CAF50;
    }
    
    /* Category headers */
    .category-header {
        background: linear-gradient(90deg, #1e2233 0%, #0e1117 100%);
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
        font-weight: bold;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Metrics */
    .css-1wivap2 {
        background-color: #1e2233;
        border-radius: 8px;
        padding: 10px;
    }
    
    /* Signal indicators */
    .signal-emoji {
        font-size: 24px;
        margin-right: 10px;
    }
    
    /* Table styling */
    .dataframe-container {
        border-radius: 10px;
        overflow: hidden;
        border: 1px solid #2d3546;
    }
    
    /* Status indicators */
    .status-bullish {
        color: #4CAF50;
        font-weight: bold;
    }
    .status-bearish {
        color: #F44336;
        font-weight: bold;
    }
    
    /* Hover effects */
    .hover-effect:hover {
        transform: scale(1.02);
        transition: transform 0.2s;
    }
    
    /* Refresh timer */
    .refresh-timer {
        text-align: center;
        padding: 10px;
        background-color: #1e2233;
        border-radius: 8px;
        margin-top: 20px;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #1a1c25;
    }
    
    /* Card layout */
    .card {
        background-color: #1e2233;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
    }
    
    /* Guide */
    .guide-container {
        background-color: #f8f9fa;
        color: #212529;
        border-radius: 10px;
        padding: 15px;
        margin-top: 15px;
    }
</style>
""", unsafe_allow_html=True)

# Symbol explanation
SYMBOL_EXPLANATION = {
    "🚀🚀": "Both Daily and Weekly Bullish",
    "🕣🕣": "Daily Bearish, Weekly Bullish (Clock)",
    "⚠️⚠️": "Daily Bullish, Weekly Bearish (Caution)",
    "💀💀": "Both Daily and Weekly Bearish",
    "✅": "EMAs aligned (7 EMA > 11 EMA > 21 EMA) on Daily Timeframe",
    "❌": "EMAs NOT aligned on Daily Timeframe"
}

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

def calculate_bullish_score(daily_rsi, weekly_rsi, ema_aligned, pct_change):
    """
    Calculate a bullish score to sort results (higher is more bullish)
    Takes into account RSI values, EMA alignment and recent price momentum
    """
    score = 0
    
    # RSI components (weighted)
    score += daily_rsi * 1.5  # 0-150 points, with higher weight on daily
    score += weekly_rsi        # 0-100 points
    
    # EMA alignment bonus (50 points)
    if ema_aligned:
        score += 50
    
    # Recent momentum (price change)
    score += pct_change * 10  # Price momentum factor
    
    # Penalize overbought conditions slightly
    if daily_rsi > 80:
        score -= (daily_rsi - 80) * 2
    
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
        ema_status = "✅" if ema_aligned else "❌"
        
        # Determine emoji
        if daily_bullish and weekly_bullish:
            emoji = "🚀🚀"
        elif not daily_bullish and weekly_bullish:
            emoji = "🕣🕣"
        elif daily_bullish and not weekly_bullish:
            emoji = "⚠️⚠️"
        else:
            emoji = "💀💀"
        
        # Current price
        current_price = daily_data['Close'].iloc[-1]
        
        # Calculate % change today
        if len(daily_data) > 1:
            prev_close = daily_data['Close'].iloc[-2]
            pct_change = ((current_price - prev_close) / prev_close) * 100
        else:
            pct_change = 0
            
        # Calculate bullish score for sorting
        bullish_score = calculate_bullish_score(latest_daily_rsi, latest_weekly_rsi, ema_aligned, pct_change)
        
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
            "pct_change": pct_change,
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
    
    # Create figure with secondary y-axis
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.03, 
                        row_heights=[0.7, 0.3],
                        subplot_titles=(f"{result['display_name']} - Price Chart", "RSI"))
    
    # Add candlestick chart
    fig.add_trace(go.Candlestick(
        x=result["daily_data"].index,
        open=result["daily_data"]['Open'],
        high=result["daily_data"]['High'],
        low=result["daily_data"]['Low'],
        close=result["daily_data"]['Close'],
        name="Price",
        increasing_line_color='#26A69A', 
        decreasing_line_color='#EF5350'
    ), row=1, col=1)
    
    # Add EMAs
    emas = result["emas"]
    colors = ['#1E88E5', '#FFC107', '#7CB342']  # Blue, Amber, Green
    for i, span in enumerate([7, 11, 21]):
        fig.add_trace(go.Scatter(
            x=emas[f'EMA_{span}'].index,
            y=emas[f'EMA_{span}'],
            mode='lines',
            line=dict(width=2, color=colors[i]),
            name=f'EMA {span}'
        ), row=1, col=1)
    
    # Add RSI
    daily_rsi_series = calculate_rsi(result["daily_data"])
    fig.add_trace(go.Scatter(
        x=daily_rsi_series.index, 
        y=daily_rsi_series,
        line=dict(color='#BA68C8', width=2),
        name='RSI (14)'
    ), row=2, col=1)
    
    # Add RSI horizontal lines at 70 and 30
    fig.add_shape(type="line", x0=daily_rsi_series.index[0], x1=daily_rsi_series.index[-1], 
                 y0=70, y1=70, line=dict(color="red", width=1, dash="dash"), row=2, col=1)
    fig.add_shape(type="line", x0=daily_rsi_series.index[0], x1=daily_rsi_series.index[-1], 
                 y0=30, y1=30, line=dict(color="green", width=1, dash="dash"), row=2, col=1)
    # Add a center line at 50
    fig.add_shape(type="line", x0=daily_rsi_series.index[0], x1=daily_rsi_series.index[-1], 
                 y0=50, y1=50, line=dict(color="gray", width=1, dash="dash"), row=2, col=1)
    
    # Update layout
    fig.update_layout(
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        height=600,
        paper_bgcolor='#0e1117',
        plot_bgcolor='#0e1117',
        margin=dict(l=10, r=10, t=60, b=10),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Y-axis ranges
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="RSI", range=[0, 100], row=2, col=1)
    
    # Style updates
    fig.update_xaxes(
        gridcolor="#1f2630",
        showgrid=True
    )
    fig.update_yaxes(
        gridcolor="#1f2630",
        showgrid=True
    )
    
    return fig

def display_signal_legend():
    """Display the signal legend attractively"""
    st.markdown("""
    <div class="legend-container" style="background-color: #f8f9fa; color: #212529; border-left: 5px solid #4CAF50;">
        <h3 style="margin-top:0; color: #212529;">📊 Signal Legend</h3>
        <table style="width:100%; color: #212529;">
            <tr>
                <td style="padding:10px;width:60px;text-align:center;"><span style="font-size:24px;">🚀🚀</span></td>
                <td style="color: #212529;">Both Daily and Weekly Bullish</td>
            </tr>
            <tr>
                <td style="padding:10px;width:60px;text-align:center;"><span style="font-size:24px;">🕣🕣</span></td>
                <td style="color: #212529;">Daily Bearish, Weekly Bullish (Clock)</td>
            </tr>
            <tr>
                <td style="padding:10px;width:60px;text-align:center;"><span style="font-size:24px;">⚠️⚠️</span></td>
                <td style="color: #212529;">Daily Bullish, Weekly Bearish (Caution)</td>
            </tr>
            <tr>
                <td style="padding:10px;width:60px;text-align:center;"><span style="font-size:24px;">💀💀</span></td>
                <td style="color: #212529;">Both Daily and Weekly Bearish</td>
            </tr>
            <tr>
                <td style="padding:10px;width:60px;text-align:center;"><span style="font-size:24px;">✅</span></td>
                <td style="color: #212529;">EMAs aligned (7 EMA > 11 EMA > 21 EMA) on Daily Timeframe</td>
            </tr>
            <tr>
                <td style="padding:10px;width:60px;text-align:center;"><span style="font-size:24px;">❌</span></td>
                <td style="color: #212529;">EMAs NOT aligned on Daily Timeframe</td>
            </tr>
        </table>
    </div>
    """, unsafe_allow_html=True)

def display_rsi_guide():
    """Display the RSI color guide"""
    st.markdown("""
    <div class="guide-container">
        <h4 style="margin-top:0; color: #212529;">RSI Color Guide</h4>
        <ul style="list-style-type: none; padding-left: 10px; color: #212529;">
            <li><span style="color: #F44336; font-weight: bold;">Red (>70)</span>: Overbought - potential reversal</li>
            <li><span style="color: #4CAF50; font-weight: bold;">Green (<30)</span>: Oversold - potential reversal</li>
            <li><span style="color: #AED581;">Light Green (>50)</span>: Bullish momentum</li>
            <li><span style="color: #EF9A9A;">Light Red (<50)</span>: Bearish momentum</li>
        </ul>
        <p style="margin-bottom:0; color: #212529;">The <b>Change %</b> column shows the daily price change percentage. Green indicates positive change, red indicates negative.</p>
    </div>
    """, unsafe_allow_html=True)

def format_dataframe(df):
    """
    Apply conditional formatting to the dataframe
    """
    # Make a copy to ensure we don't modify the original
    df = df.copy()
    
    # Convert Change % to numeric (removing % sign if present)
    if 'Change %' in df.columns:
        df['Change %'] = pd.to_numeric(df['Change %'].str.replace('%', ''), errors='coerce')
    
    # Convert RSI columns to numeric
    if 'Daily RSI' in df.columns:
        df['Daily RSI'] = pd.to_numeric(df['Daily RSI'], errors='coerce')
    if 'Weekly RSI' in df.columns:
        df['Weekly RSI'] = pd.to_numeric(df['Weekly RSI'], errors='coerce')
    
    # Format the Daily and Weekly status columns
    df_styled = df.style.map(
        lambda x: 'color: #4CAF50; font-weight: bold' if x == 'Bullish' else 'color: #F44336; font-weight: bold',
        subset=['Daily', 'Weekly']
    )
    
    # Format the price change column
    df_styled = df_styled.map(
        lambda x: f'color: {"#4CAF50" if x > 0 else "#F44336"}; font-weight: bold',
        subset=['Change %']
    )
    
    # Format the RSI columns
    def highlight_rsi(val):
        try:
            val_num = float(val)
            if val_num > 70:
                return 'color: #F44336; font-weight: bold'  # Overbought
            elif val_num < 30:
                return 'color: #4CAF50; font-weight: bold'  # Oversold
            elif val_num > 50:
                return 'color: #AED581'  # Bullish
            else:
                return 'color: #EF9A9A'  # Bearish
        except (ValueError, TypeError):
            return ''
    
    df_styled = df_styled.map(highlight_rsi, subset=['Daily RSI', 'Weekly RSI'])
    
    return df_styled

def main():
    # Sidebar configuration
    with st.sidebar:
        st.image("https://i.imgur.com/8NtMsXM.png", width=80)  # A placeholder logo
        st.title("Market Scanner")
        st.markdown("---")
        
        st.subheader("📊 Scan Settings")
        
        # Category selection
        selected_categories = st.multiselect(
            "Select Markets to Scan",
            options=list(TICKER_CATEGORIES.keys()),
            default=list(TICKER_CATEGORIES.keys())
        )
        
        # Scan interval
        refresh_interval = st.slider(
            "Auto-refresh Interval (minutes)",
            min_value=1,
            max_value=60,
            value=5,
            step=1
        )
        
        # Display setting
        show_charts = st.checkbox("Show Charts for Top Performers", value=True)
        
        st.markdown("---")
        st.subheader("📈 Market Overview")
        
        # Create placeholder for market overview metrics
        market_metrics = st.empty()
        
        # About section
        st.markdown("---")
        st.info(
            "This app scans financial markets for trading signals based on RSI and EMA indicators."
        )
        
        # Add version info
        st.markdown("<div style='text-align:center; font-size:0.8em; color:#666;'>v1.0.0</div>", unsafe_allow_html=True)
    
    # Main page content
    st.title("📊 Market Signal Scanner")
    
    # Current time with cleaner display
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.markdown(f"<div style='font-size:1.2em; margin-bottom:20px;'>⏱️ Last Updated: <b>{current_time}</b></div>", unsafe_allow_html=True)
    
    # Display legends
    col1, col2 = st.columns(2)
    with col1:
        display_signal_legend()
    with col2:
        display_rsi_guide()
    
    # Create placeholder for results
    results_placeholder = st.empty()
    
    # Create placeholder for top charts
    charts_placeholder = st.empty()
    
    # Progress bar for scanning
    if selected_categories:
        # Count total tickers to scan
        total_tickers = sum(len(TICKER_CATEGORIES[cat]) for cat in selected_categories)
        
        with st.spinner(f"Scanning {total_tickers} markets..."):
            # Create a progress bar
            progress_bar = st.progress(0)
            
            # Collect all results
            all_results = []
            category_results = {cat: [] for cat in selected_categories}
            tickers_scanned = 0
            
            # Scan each selected category
            for category in selected_categories:
                if category in TICKER_CATEGORIES:
                    category_tickers = TICKER_CATEGORIES[category]
                    
                    # Scan each ticker in the category
                    for ticker, name in category_tickers.items():
                        result = scan_ticker(ticker, name)
                        result["category"] = category  # Add category info
                        all_results.append(result)
                        category_results[category].append(result)
                        
                        # Update progress
                        tickers_scanned += 1
                        progress_bar.progress(tickers_scanned / total_tickers)
            
            # Remove progress bar when done
            progress_bar.empty()
            
            # Calculate market metrics for sidebar
            valid_results = [r for r in all_results if not r.get("error")]
            if valid_results:
                bullish_count = sum(1 for r in valid_results if r["daily_status"] == "Bullish" and r["weekly_status"] == "Bullish")
                bearish_count = sum(1 for r in valid_results if r["daily_status"] == "Bearish" and r["weekly_status"] == "Bearish")
                mixed_count = len(valid_results) - bullish_count - bearish_count
                
                # Display metrics in sidebar
                with market_metrics.container():
                    metrics_cols = st.columns(3)
                    metrics_cols[0].metric("Bullish", f"{bullish_count}", f"{bullish_count/len(valid_results)*100:.1f}%")
                    metrics_cols[1].metric("Bearish", f"{bearish_count}", f"{bearish_count/len(valid_results)*100:.1f}%")
                    metrics_cols[2].metric("Mixed", f"{mixed_count}", f"{mixed_count/len(valid_results)*100:.1f}%")
            
            # Sort all results by bullish score (most bullish first)
            all_results.sort(key=lambda x: x.get("score", -1000), reverse=True)
            
            # Sort category results
            for cat in category_results:
                category_results[cat] = [r for r in category_results[cat] if not r.get("error")]
                category_results[cat].sort(key=lambda x: x.get("score", -1000), reverse=True)
        
        # Display the results in the main area
        with results_placeholder.container():
            # Format the data into a pretty table
            if valid_results:
                # Create tabs for All and each category
                tab_names = ["All Markets"] + selected_categories
                tabs = st.tabs(tab_names)
                
                # All Markets tab
                with tabs[0]:
                    display_data = []
                    for r in all_results:
                        if not r.get("error"):
                            display_data.append({
                                "Signal": r["emoji"],
                                "Market": r["display_name"],
                                "Daily": r["daily_status"],
                                "Weekly": r["weekly_status"],
                                "EMA": r["ema_status"],
                                "Price": f"{r['price']:.4f}",
                                "Change %": f"{r['pct_change']:.2f}",
                                "Daily RSI": f"{r['daily_rsi']:.1f}",
                                "Weekly RSI": f"{r['weekly_rsi']:.1f}"
                            })
                    
                    if display_data:
                        df = pd.DataFrame(display_data)
                        st.markdown('<div class="dataframe-container">', unsafe_allow_html=True)
                        st.dataframe(
                            format_dataframe(df),
                            use_container_width=True,
                            height=500,
                            hide_index=True
                        )
                        st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        st.warning("No data available for selected markets.")
                
                # Category tabs
                for i, category in enumerate(selected_categories, 1):
                    with tabs[i]:
                        cat_display_data = []
                        for r in category_results[category]:
                            if not r.get("error"):
                                cat_display_data.append({
                                    "Signal": r["emoji"],
                                    "Market": r["display_name"],
                                    "Daily": r["daily_status"],
                                    "Weekly": r["weekly_status"],
                                    "EMA": r["ema_status"],
                                    "Price": f"{r['price']:.4f}",
                                    "Change %": f"{r['pct_change']:.2f}",
                                    "Daily RSI": f"{r['daily_rsi']:.1f}",
                                    "Weekly RSI": f"{r['weekly_rsi']:.1f}"
                                })
                        
                        if cat_display_data:
                            cat_df = pd.DataFrame(cat_display_data)
                            st.markdown('<div class="dataframe-container">', unsafe_allow_html=True)
                            st.dataframe(
                                format_dataframe(cat_df),
                                use_container_width=True,
                                height=400,
                                hide_index=True
                            )
                            st.markdown('</div>', unsafe_allow_html=True)
                        else:
                            st.warning(f"No data available for {category}.")
            else:
                st.warning("No valid results found. Check your internet connection or try different markets.")
        
        # Show charts for top performers if requested
        if show_charts and valid_results:
            with charts_placeholder.container():
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.subheader("📈 Top Performers")
                
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
                            
                            # Add key metrics below chart
                            metric_cols = st.columns(3)
                            metric_cols[0].metric(
                                "Daily RSI", 
                                f"{valid_results[i]['daily_rsi']:.1f}",
                                delta=f"{valid_results[i]['daily_rsi'] - 50:.1f} from 50"
                            )
                            metric_cols[1].metric(
                                "Weekly RSI", 
                                f"{valid_results[i]['weekly_rsi']:.1f}",
                                delta=f"{valid_results[i]['weekly_rsi'] - 50:.1f} from 50"
                            )
                            metric_cols[2].metric(
                                "Today", 
                                f"{valid_results[i]['price']:.4f}",
                                delta=f"{valid_results[i]['pct_change']:.2f}%"
                            )
                st.markdown('</div>', unsafe_allow_html=True)
        
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
    countdown = st.empty()
    
    # Add the refresh timer
    refresh_time = datetime.now() + timedelta(minutes=refresh_interval)
    countdown.markdown(f"""
    <div class="refresh-timer">
        <span>Next refresh at {refresh_time.strftime('%H:%M:%S')}</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Schedule the next refresh
    if "last_refresh" not in st.session_state:
        st.session_state.last_refresh = datetime.now()
    
    # Check if it's time to refresh
    if datetime.now() >= st.session_state.last_refresh + timedelta(minutes=refresh_interval):
        st.session_state.last_refresh = datetime.now()
        st.rerun()

if __name__ == "__main__":
    main()
