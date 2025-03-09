import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import time
import base64
from io import BytesIO
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys

# Import ticker categories from tickers.py
from tickers import TICKER_CATEGORIES, SYMBOL_EXPLANATION

# Set page configuration
st.set_page_config(
    page_title="Market Technical Scanner",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS for styling
st.markdown("""
<style>
    .bullish { color: #0ECB81; font-weight: bold; }
    .bearish { color: #F6465D; font-weight: bold; }
    .cautious { color: #F0B90B; font-weight: bold; }
    .neutral { color: #8A8A8A; font-weight: bold; }
    .error { color: #FF6B6B; font-style: italic; }
    .small-font { font-size: 0.8em; }
    .symbol { font-size: 1.2em; margin-right: 5px; }
    .ticker-card {
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 10px;
        margin: 5px 0;
        transition: all 0.3s;
    }
    .ticker-card:hover {
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .summary-box {
        background-color: #f8f9fa;
        border-radius: 5px;
        padding: 15px;
        margin: 10px 0;
    }
    .indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 5px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=3600)  # Cache data for 1 hour
def fetch_data(ticker, period="3mo", interval="1d"):
    """Fetch historical data for a ticker"""
    try:
        data = yf.download(ticker, period=period, interval=interval, progress=False)
        if data.empty:
            return None
        return data
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")
        return None

def calculate_rsi(data, window=14):
    """Calculate RSI for the given data"""
    if data is None or len(data) < window:
        return None, None
    
    # Calculate price changes
    delta = data['Close'].diff()
    
    # Separate gains and losses
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    # Calculate average gain and loss over the window
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    
    # Calculate RS and RSI
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    latest_rsi = rsi.iloc[-1]
    return rsi, latest_rsi

def calculate_macd(data):
    """Calculate MACD and detect crosses"""
    if data is None:
        return None, None, None
        
    # Need at least 35 days for the MACD (26 for longest EMA + 9 for signal line)
    if len(data) < 35:
        st.warning(f"Only {len(data)} days of data available, need at least 35 for MACD calculation")
        return None, None, None
    
    # Calculate EMAs
    ema12 = data['Close'].ewm(span=12, adjust=False).mean()
    ema26 = data['Close'].ewm(span=26, adjust=False).mean()
    
    # Calculate MACD line and signal line
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    
    # Check for crosses in the recent data
    last_n_days = min(10, len(data))
    for i in range(1, last_n_days):
        # Check for golden cross (MACD crosses above signal line)
        if macd_line.iloc[-i] > signal_line.iloc[-i] and macd_line.iloc[-(i+1)] <= signal_line.iloc[-(i+1)]:
            return macd_line, signal_line, "GOLDEN CROSS"
        
        # Check for death cross (MACD crosses below signal line)
        if macd_line.iloc[-i] < signal_line.iloc[-i] and macd_line.iloc[-(i+1)] >= signal_line.iloc[-(i+1)]:
            return macd_line, signal_line, "DEATH CROSS"
    
    # No cross detected
    return macd_line, signal_line, "NO STRONG SIGNAL"

def determine_market_sentiment(rsi, macd_signal):
    """Determine market sentiment based on RSI and MACD signals"""
    if macd_signal == "GOLDEN CROSS" and rsi >= 50:
        return "BULLISH", "🚀🚀"
    elif macd_signal == "GOLDEN CROSS" and rsi < 50:
        return "CAUTIOUS (Bullish MACD, Weak RSI)", "🕣🕣"
    elif macd_signal == "DEATH CROSS" and rsi < 50:
        return "BEARISH", "💀💀"
    elif macd_signal == "DEATH CROSS" and rsi >= 50:
        return "CAUTIOUS (Bearish MACD, Strong RSI)", "⚠️⚠️"
    else:
        if rsi >= 70:
            return "NEUTRAL (Possibly Overbought)", "❓"
        elif rsi <= 30:
            return "NEUTRAL (Possibly Oversold)", "❓"
        else:
            return "NEUTRAL", "❓"

def check_ema_alignment(data):
    """Check if EMAs are aligned (7 > 11 > 21) on daily timeframe"""
    if data is None or len(data) < 21:
        return False, None
    
    # Calculate EMAs
    ema7 = data['Close'].ewm(span=7, adjust=False).mean()
    ema11 = data['Close'].ewm(span=11, adjust=False).mean()
    ema21 = data['Close'].ewm(span=21, adjust=False).mean()
    
    # Check if aligned (7 > 11 > 21)
    if ema7.iloc[-1] > ema11.iloc[-1] > ema21.iloc[-1]:
        return True, (ema7, ema11, ema21)
    else:
        return False, (ema7, ema11, ema21)

def analyze_ticker(ticker, name):
    """Analyze a ticker and provide market signals"""
    try:
        # Fetch data
        with st.spinner(f"Analyzing {ticker}..."):
            data = fetch_data(ticker)
            if data is None or len(data) == 0:
                return {
                    "ticker": ticker,
                    "name": name,
                    "rsi": None,
                    "macd_signal": "No data",
                    "sentiment": "No data",
                    "symbol": "❓",
                    "ema_aligned": False,
                    "error": "No data available",
                    "data": None,
                    "charts": None
                }
            
            # Calculate RSI
            rsi_series, latest_rsi = calculate_rsi(data)
            if latest_rsi is None:
                return {
                    "ticker": ticker,
                    "name": name,
                    "rsi": None,
                    "macd_signal": "Insufficient data",
                    "sentiment": "No data",
                    "symbol": "❓",
                    "ema_aligned": False,
                    "error": "Insufficient data for RSI calculation",
                    "data": data,
                    "charts": None
                }
            
            # Calculate MACD and detect cross
            macd_line, signal_line, macd_signal = calculate_macd(data)
            if macd_signal is None:
                return {
                    "ticker": ticker,
                    "name": name,
                    "rsi": latest_rsi,
                    "macd_signal": "Insufficient data",
                    "sentiment": "No data",
                    "symbol": "❓",
                    "ema_aligned": False,
                    "error": "Insufficient data for MACD calculation",
                    "data": data,
                    "charts": None
                }
            
            # Check EMA alignment
            ema_aligned, ema_lines = check_ema_alignment(data)
            
            # Determine market sentiment
            sentiment, symbol = determine_market_sentiment(latest_rsi, macd_signal)
            
            # Add EMA symbol
            ema_symbol = "✅" if ema_aligned else "❌"
            
            # Create charts data
            charts = {
                "price": data,
                "rsi": rsi_series,
                "macd_line": macd_line,
                "signal_line": signal_line,
                "ema_lines": ema_lines
            }
            
            return {
                "ticker": ticker,
                "name": name,
                "rsi": latest_rsi,
                "macd_signal": macd_signal,
                "sentiment": sentiment,
                "symbol": symbol,
                "ema_aligned": ema_aligned,
                "ema_symbol": ema_symbol,
                "error": None,
                "data": data,
                "charts": charts
            }
    
    except Exception as e:
        return {
            "ticker": ticker,
            "name": name,
            "rsi": None,
            "macd_signal": "Error",
            "sentiment": "Error",
            "symbol": "❓",
            "ema_aligned": False,
            "ema_symbol": "❓",
            "error": str(e),
            "data": None,
            "charts": None
        }

def get_sentiment_class(sentiment):
    """Get CSS class for sentiment"""
    if "BULLISH" in sentiment:
        return "bullish"
    elif "BEARISH" in sentiment:
        return "bearish"
    elif "CAUTIOUS" in sentiment:
        return "cautious"
    else:
        return "neutral"

def display_ticker_card(result):
    """Display a ticker result card"""
    ticker = result["ticker"]
    name = result["name"]
    rsi = result["rsi"]
    macd_signal = result["macd_signal"]
    sentiment = result["sentiment"]
    symbol = result["symbol"]
    ema_symbol = result.get("ema_symbol", "❓")
    error = result["error"]
    
    # Create a bordered card for the ticker
    col1, col2 = st.columns([3, 1])
    
    with col1:
        if error:
            st.markdown(f"""
            <div class="ticker-card">
                <h4>{ticker} - {name}</h4>
                <p class="error">{error}</p>
            </div>
            """, unsafe_allow_html=True)
            return
        
        # Format RSI value
        rsi_str = f"{rsi:.2f}" if rsi is not None else "N/A"
        
        # Get sentiment class
        sentiment_class = get_sentiment_class(sentiment)
        
        st.markdown(f"""
        <div class="ticker-card">
            <h4>{ticker} - {name}</h4>
            <p>
                <span class="symbol">{symbol}</span>
                <span class="symbol">{ema_symbol}</span>
                <span class="{sentiment_class}">{sentiment}</span>
            </p>
            <p class="small-font">RSI: {rsi_str} | MACD: {macd_signal}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        if st.button("View Charts", key=f"btn_{ticker}"):
            st.session_state.selected_ticker = ticker
            st.session_state.selected_result = result

def get_sentiment_category(sentiment):
    """Extract the main sentiment category from the detailed sentiment"""
    if "BULLISH" in sentiment:
        return "BULLISH"
    elif "BEARISH" in sentiment:
        return "BEARISH"
    elif "CAUTIOUS" in sentiment:
        return "CAUTIOUS"
    else:
        return "NEUTRAL"

def display_market_summary(results):
    """Display summary of market sentiment"""
    # Count sentiments
    sentiment_counts = {}
    for r in results:
        if r["sentiment"] != "No data" and r["sentiment"] != "Error":
            category = get_sentiment_category(r["sentiment"])
            sentiment_counts[category] = sentiment_counts.get(category, 0) + 1
    
    total_valid = sum(sentiment_counts.values())
    
    if total_valid == 0:
        st.warning("No valid sentiment data available for summary")
        return
    
    # Calculate percentages
    sentiment_percentages = {k: (v / total_valid * 100) for k, v in sentiment_counts.items()}
    
    # Overall market tendency
    if sentiment_percentages.get("BULLISH", 0) > 50:
        overall = "BULLISH"
        overall_class = "bullish"
    elif sentiment_percentages.get("BEARISH", 0) > 50:
        overall = "BEARISH"
        overall_class = "bearish"
    elif sentiment_percentages.get("BULLISH", 0) + sentiment_percentages.get("CAUTIOUS", 0) > 60:
        overall = "CAUTIOUSLY BULLISH"
        overall_class = "cautious"
    elif sentiment_percentages.get("BEARISH", 0) + sentiment_percentages.get("CAUTIOUS", 0) > 60:
        overall = "CAUTIOUSLY BEARISH"
        overall_class = "cautious"
    else:
        overall = "NEUTRAL/MIXED"
        overall_class = "neutral"
    
    # Create summary box
    st.markdown(f"""
    <div class="summary-box">
        <h3>Market Sentiment Summary</h3>
        <p>Total tickers analyzed: {total_valid}</p>
        <p>Overall market tendency: <span class="{overall_class}">{overall}</span></p>
        <div style="display: flex; justify-content: space-between; max-width: 500px;">
            <div>
                <div class="indicator" style="background-color: #0ECB81;"></div>
                BULLISH: {sentiment_counts.get('BULLISH', 0)} ({sentiment_percentages.get('BULLISH', 0):.1f}%)
            </div>
            <div>
                <div class="indicator" style="background-color: #F0B90B;"></div>
                CAUTIOUS: {sentiment_counts.get('CAUTIOUS', 0)} ({sentiment_percentages.get('CAUTIOUS', 0):.1f}%)
            </div>
            <div>
                <div class="indicator" style="background-color: #F6465D;"></div>
                BEARISH: {sentiment_counts.get('BEARISH', 0)} ({sentiment_percentages.get('BEARISH', 0):.1f}%)
            </div>
            <div>
                <div class="indicator" style="background-color: #8A8A8A;"></div>
                NEUTRAL: {sentiment_counts.get('NEUTRAL', 0)} ({sentiment_percentages.get('NEUTRAL', 0):.1f}%)
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Create pie chart
    labels = ["BULLISH", "CAUTIOUS", "BEARISH", "NEUTRAL"]
    values = [sentiment_counts.get(label, 0) for label in labels]
    colors = ['#0ECB81', '#F0B90B', '#F6465D', '#8A8A8A']
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=.3,
        marker=dict(colors=colors)
    )])
    
    fig.update_layout(
        title="Sentiment Distribution",
        height=400,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    
    st.plotly_chart(fig, use_container_width=True)

def display_charts(result):
    """Display interactive charts for a ticker"""
    if not result or not result["charts"] or result["error"]:
        st.error("No chart data available")
        return
    
    charts = result["charts"]
    price_data = charts["price"]
    rsi_data = charts["rsi"]
    macd_line = charts["macd_line"]
    signal_line = charts["signal_line"]
    ema_lines = charts["ema_lines"]
    
    # Create figure with subplots
    fig = make_subplots(
        rows=3, 
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.5, 0.25, 0.25],
        subplot_titles=("Price with EMAs", "RSI", "MACD")
    )
    
    # Add price candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=price_data.index,
            open=price_data['Open'],
            high=price_data['High'],
            low=price_data['Low'],
            close=price_data['Close'],
            name="Price"
        ),
        row=1, col=1
    )
    
    # Add EMAs to price chart
    if ema_lines:
        ema7, ema11, ema21 = ema_lines
        fig.add_trace(
            go.Scatter(x=price_data.index, y=ema7, name="EMA 7", line=dict(color="purple", width=1)),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=price_data.index, y=ema11, name="EMA 11", line=dict(color="blue", width=1)),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=price_data.index, y=ema21, name="EMA 21", line=dict(color="green", width=1)),
            row=1, col=1
        )
    
    # Add RSI
    if rsi_data is not None:
        fig.add_trace(
            go.Scatter(x=price_data.index, y=rsi_data, name="RSI", line=dict(color="orange", width=1)),
            row=2, col=1
        )
        
        # Add RSI reference lines
        fig.add_shape(
            type="line", line=dict(dash="dash", width=1, color="red"),
            x0=price_data.index[0], x1=price_data.index[-1], y0=70, y1=70,
            row=2, col=1
        )
        fig.add_shape(
            type="line", line=dict(dash="dash", width=1, color="green"),
            x0=price_data.index[0], x1=price_data.index[-1], y0=30, y1=30,
            row=2, col=1
        )
        fig.add_shape(
            type="line", line=dict(dash="dot", width=0.5, color="gray"),
            x0=price_data.index[0], x1=price_data.index[-1], y0=50, y1=50,
            row=2, col=1
        )
    
    # Add MACD
    if macd_line is not None and signal_line is not None:
        fig.add_trace(
            go.Scatter(x=price_data.index, y=macd_line, name="MACD", line=dict(color="blue", width=1)),
            row=3, col=1
        )
        fig.add_trace(
            go.Scatter(x=price_data.index, y=signal_line, name="Signal", line=dict(color="red", width=1)),
            row=3, col=1
        )
        
        # Add MACD histogram
        histogram = macd_line - signal_line
        colors = ['green' if val >= 0 else 'red' for val in histogram]
        fig.add_trace(
            go.Bar(x=price_data.index, y=histogram, name="Histogram", marker_color=colors),
            row=3, col=1
        )
    
    # Update layout
    fig.update_layout(
        title=f"{result['ticker']} - {result['name']} Analysis",
        height=800,
        xaxis_rangeslider_visible=False,
        yaxis2=dict(range=[0, 100]),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=50, t=80, b=50),
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Display technical summary
    st.subheader("Technical Analysis Summary")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        sentiment_class = get_sentiment_class(result["sentiment"])
        st.markdown(f"""
        <div style="padding: 10px; border: 1px solid #ddd; border-radius: 5px;">
            <h4>Market Sentiment</h4>
            <p>
                <span class="symbol">{result["symbol"]}</span>
                <span class="{sentiment_class}">{result["sentiment"]}</span>
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        rsi_level = "OVERBOUGHT" if result["rsi"] > 70 else "OVERSOLD" if result["rsi"] < 30 else "NEUTRAL"
        rsi_class = "bearish" if rsi_level == "OVERBOUGHT" else "bullish" if rsi_level == "OVERSOLD" else "neutral"
        
        st.markdown(f"""
        <div style="padding: 10px; border: 1px solid #ddd; border-radius: 5px;">
            <h4>RSI Analysis</h4>
            <p>Current RSI: {result["rsi"]:.2f}</p>
            <p>Level: <span class="{rsi_class}">{rsi_level}</span></p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        macd_class = "bullish" if result["macd_signal"] == "GOLDEN CROSS" else "bearish" if result["macd_signal"] == "DEATH CROSS" else "neutral"
        
        st.markdown(f"""
        <div style="padding: 10px; border: 1px solid #ddd; border-radius: 5px;">
            <h4>MACD Signal</h4>
            <p class="{macd_class}">{result["macd_signal"]}</p>
            <p>EMA Alignment: {result["ema_symbol"]}</p>
        </div>
        """, unsafe_allow_html=True)

def export_to_csv(results):
    """Create and return a CSV download link"""
    # Convert results to DataFrame
    df = pd.DataFrame([
        {
            "Ticker": r["ticker"],
            "Name": r["name"],
            "RSI": r["rsi"],
            "MACD Signal": r["macd_signal"],
            "Sentiment": r["sentiment"],
            "Symbol": r["symbol"],
            "EMA Aligned": "Yes" if r.get("ema_aligned", False) else "No"
        }
        for r in results if not r.get("error")
    ])
    
    # Convert to CSV
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"market_analysis_{timestamp}.csv"
    
    # Create download link
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download CSV File</a>'
    return href

def display_symbol_explanation():
    """Display explanation of symbols used"""
    st.markdown("### Symbol Explanation")
    
    for symbol, explanation in SYMBOL_EXPLANATION.items():
        st.markdown(f"**{symbol}**: {explanation}")

def scan_tickers_st(categories=None, specific_tickers=None):
    """Scan selected tickers and display results in Streamlit"""
    results = []
    
    # If specific_tickers is provided, only scan those
    if specific_tickers:
        progress_bar = st.progress(0)
        total_tickers = len(specific_tickers)
        
        for i, ticker in enumerate(specific_tickers):
            ticker_found = False
            for cat, ticker_dict in TICKER_CATEGORIES.items():
                if ticker in ticker_dict:
                    result = analyze_ticker(ticker, ticker_dict[ticker])
                    results.append(result)
                    ticker_found = True
                    break
            
            if not ticker_found:
                st.warning(f"Ticker {ticker} not found in any category")
            
            # Update progress
            progress_bar.progress((i + 1) / total_tickers)
        
        return results
    
    # If categories is provided, only scan those categories
    if categories:
        categories_to_scan = {cat: TICKER_CATEGORIES.get(cat, {}) for cat in categories if cat in TICKER_CATEGORIES}
    else:
        categories_to_scan = TICKER_CATEGORIES
    
    # Track overall progress
    total_tickers = sum(len(tickers) for tickers in categories_to_scan.values())
    progress_bar = st.progress(0)
    processed = 0
    
    for category, tickers in categories_to_scan.items():
        st.subheader(category)
        category_results = []
        
        for ticker, name in tickers.items():
            result = analyze_ticker(ticker, name)
            results.append(result)
            category_results.append(result)
            
            # Update progress
            processed += 1
            progress_bar.progress(processed / total_tickers)
        
    return results

def main():
    st.title("Market Technical Scanner")
    
    # Initialize session state
    if 'results' not in st.session_state:
        st.session_state.results = []
    if 'selected_ticker' not in st.session_state:
        st.session_state.selected_ticker = None
    if 'selected_result' not in st.session_state:
        st.session_state.selected_result = None
    if 'last_scan_time' not in st.session_state:
        st.session_state.last_scan_time = None
    
    # Sidebar
    st.sidebar.title("Scan Settings")
    
    # Display scan options
    scan_option = st.sidebar.radio(
        "Scan Options", 
        ("All Categories", "Select Categories", "Specific Tickers"),
        index=1  # Default to Select Categories
    )
    
    # Set up scan parameters based on selection
    categories = None
    specific_tickers = None
    
    if scan_option == "Select Categories":
        # Allow selection of categories
        available_categories = list(TICKER_CATEGORIES.keys())
        categories = st.sidebar.multiselect(
            "Select categories to scan",
            available_categories,
            default=["INDICES", "US STOCKS"]  # Default selections
        )
    elif scan_option == "Specific Tickers":
        # Allow specific tickers
        ticker_input = st.sidebar.text_area("Enter specific tickers (comma-separated)")
        if ticker_input:
            specific_tickers = [t.strip().upper() for t in ticker_input.split(',')]
            st.sidebar.write(f"Selected tickers: {', '.join(specific_tickers)}")
    
    # Scan interval
    interval_hours = st.sidebar.slider(
        "Auto-refresh interval (hours)", 
        min_value=1, 
        max_value=24, 
        value=6,
        step=1
    )
    
    # Display time until next auto-refresh
    if st.session_state.last_scan_time:
        next_scan_time = st.session_state.last_scan_time + timedelta(hours=interval_hours)
        time_remaining = next_scan_time - datetime.now()
        
        if time_remaining.total_seconds() > 0:
            hours = int(time_remaining.total_seconds() // 3600)
            minutes = int((time_remaining.total_seconds() % 3600) // 60)
            st.sidebar.info(f"Next auto-refresh in: {hours:02d}:{minutes:02d}")
    
    # Manual refresh button
    refresh_pressed = st.sidebar.button("Refresh Now", use_container_width=True)
    
    # Symbol explanation in sidebar
    with st.sidebar.expander("Symbol Explanation"):
        display_symbol_explanation()
    
    # Check if auto-refresh is needed
    auto_refresh = False
    if st.session_state.last_scan_time:
        time_since_last = datetime.now() - st.session_state.last_scan_time
        if time_since_last.total_seconds() >= interval_hours * 3600:
            auto_refresh = True
    
    # Main content
    tabs = st.tabs(["Dashboard", "Detailed Analysis"])
    
    with tabs[0]:  # Dashboard tab
        st.subheader("Market Scanner Dashboard")
        
        # Perform scan if this is the first load, manual refresh requested, or auto-refresh needed
        if refresh_pressed or auto_refresh or not st.session_state.results:
            with st.spinner("Scanning tickers..."):
                st.session_state.results = scan_tickers_st(categories, specific_tickers)
                st.session_state.last_scan_time = datetime.now()
                st.success(f"Scan completed at {st.session_state.last_scan_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Display scan time
        if st.session_state.last_scan_time:
            st.info(f"Last scan: {st.session_state.last_scan_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Display market summary
        if st.session_state.results:
            display_market_summary(st.session_state.results)
            
            # Provide CSV export option
            st.markdown(export_to_csv(st.session_state.results), unsafe_allow_html=True)
            
            # Display results by category
            for category in TICKER_CATEGORIES.keys():
                # Filter results for this category
                category_results = [r for r in st.session_state.results 
                                 if any(r["ticker"] == t for t in TICKER_CATEGORIES[category].keys())]
                
                if category_results:
                    with st.expander(f"{category} ({len(category_results)} tickers)", expanded=category=="INDICES"):
                        for result in category_results:
                            display_ticker_card(result)
    
    with tabs[1]:  # Detailed Analysis tab
        st.subheader("Detailed Technical Analysis")
        
        # Check if a ticker is selected for detailed view
        if st.session_state.selected_ticker and st.session_state.selected_result:
            ticker = st.session_state.selected_ticker
            result = st.session_state.selected_result
            
            # Add a back button
            if st.button("← Back to Dashboard"):
                st.session_state.selected_ticker = None
                st.session_state.selected_result = None
                st.experimental_rerun()
            
            # Display detailed charts and analysis
            display_charts(result)
        else:
            st.info("Select a ticker from the Dashboard to view detailed analysis")

if __name__ == "__main__":
    main()
