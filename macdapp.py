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
import pandas_ta as ta # Using pandas_ta for easier indicator calculation

# Import ticker categories from tickers.py
from tickers import TICKER_CATEGORIES, SYMBOL_EXPLANATION # Keep using your categories

# Set page configuration
st.set_page_config(
    page_title="I Chart Daily Strategy Scanner",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS for styling (simplified for new focus)
st.markdown("""
<style>
    /* Score-based highlighting */
    .strong-long { color: #00A100; font-weight: bold; }
    .potential-long { color: #0ECB81; font-weight: bold; }
    .strong-short { color: #D20000; font-weight: bold; }
    .potential-short { color: #F6465D; font-weight: bold; }
    .neutral-conflicting { color: #8A8A8A; font-weight: bold; }
    .error { color: #FF6B6B; font-style: italic; }
    .small-font { font-size: 0.8em; }
    .indicator {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-right: 5px;
    }
    .ok { background-color: #0ECB81; }
    .nok { background-color: #F6465D; }
    .neutral { background-color: #F0B90B; }
    .dataframe-container { border-radius: 5px; overflow: hidden; }
    .dataframe tr:hover { background-color: rgba(0,0,0,0.05) !important; cursor: pointer; }
</style>
""", unsafe_allow_html=True)

# --- Configuration for Strategy ---
# Moving Averages (adjust periods as needed)
EMA_SHORT = 11
EMA_LONG = 21
EMA_LONGER = 50 # Optional longer MA for context
# RSI
RSI_WINDOW = 14
RSI_MID = 50
# MACD
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
# Scoring Thresholds
STRONG_SETUP_THRESHOLD = 5 # How many rules must be met for a "Strong" setup
POTENTIAL_SETUP_THRESHOLD = 4 # How many rules for a "Potential" setup

@st.cache_data(ttl=3600)  # Cache data for 1 hour
def fetch_all_data(ticker):
    """Fetch historical data for a ticker (Daily & Weekly)"""
    try:
        ticker_obj = yf.Ticker(ticker)
        # Fetch enough data for calculations (e.g., 1 year daily, 5 years weekly)
        daily_data = ticker_obj.history(period="1y", interval="1d")
        weekly_data = ticker_obj.history(period="5y", interval="1wk")

        if daily_data.empty or len(daily_data) < EMA_LONGER or \
           weekly_data.empty or len(weekly_data) < EMA_LONGER:
            st.warning(f"Insufficient data for {ticker}")
            return None, None
        return daily_data, weekly_data
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")
        return None, None

def calculate_indicators(data, timeframe_suffix=""):
    """Calculate required indicators for a given dataframe (daily or weekly)"""
    if data is None or data.empty:
        return {}

    indicators = {}
    try:
        # EMAs
        data.ta.ema(length=EMA_SHORT, append=True)
        data.ta.ema(length=EMA_LONG, append=True)
        data.ta.ema(length=EMA_LONGER, append=True)
        indicators[f'EMA_{EMA_SHORT}'] = data[f'EMA_{EMA_SHORT}']
        indicators[f'EMA_{EMA_LONG}'] = data[f'EMA_{EMA_LONG}']
        indicators[f'EMA_{EMA_LONGER}'] = data[f'EMA_{EMA_LONGER}']

        # RSI
        data.ta.rsi(length=RSI_WINDOW, append=True)
        indicators[f'RSI_{RSI_WINDOW}'] = data[f'RSI_{RSI_WINDOW}']
        indicators[f'RSI_latest'] = data[f'RSI_{RSI_WINDOW}'].iloc[-1]

        # MACD
        data.ta.macd(fast=MACD_FAST, slow=MACD_SLOW, signal=MACD_SIGNAL, append=True)
        indicators[f'MACD_line'] = data[f'MACD_{MACD_FAST}_{MACD_SLOW}_{MACD_SIGNAL}']
        indicators[f'MACD_signal'] = data[f'MACDs_{MACD_FAST}_{MACD_SLOW}_{MACD_SIGNAL}']
        indicators[f'MACD_hist'] = data[f'MACDh_{MACD_FAST}_{MACD_SLOW}_{MACD_SIGNAL}']

        # Determine MACD Status (Cross or Current State)
        macd_line = indicators[f'MACD_line']
        signal_line = indicators[f'MACD_signal']
        if len(macd_line) > 1:
            # Check for cross in the last period
            crossed_above = (macd_line.iloc[-2] < signal_line.iloc[-2]) and (macd_line.iloc[-1] > signal_line.iloc[-1])
            crossed_below = (macd_line.iloc[-2] > signal_line.iloc[-2]) and (macd_line.iloc[-1] < signal_line.iloc[-1])
            if crossed_above:
                indicators['MACD_status'] = "Golden Cross"
            elif crossed_below:
                indicators['MACD_status'] = "Death Cross"
            # If no cross, check current state
            elif macd_line.iloc[-1] > signal_line.iloc[-1]:
                indicators['MACD_status'] = "Bullish"
            else:
                indicators['MACD_status'] = "Bearish"
        else:
             indicators['MACD_status'] = "Bullish" if macd_line.iloc[-1] > signal_line.iloc[-1] else "Bearish"

        # Price vs MAs
        indicators['Price'] = data['Close']
        indicators['Price_latest'] = data['Close'].iloc[-1]
        indicators['Price_vs_EMA_Short'] = indicators['Price_latest'] > indicators[f'EMA_{EMA_SHORT}'].iloc[-1]
        indicators['Price_vs_EMA_Long'] = indicators['Price_latest'] > indicators[f'EMA_{EMA_LONG}'].iloc[-1]
        indicators['Price_vs_EMA_Longer'] = indicators['Price_latest'] > indicators[f'EMA_{EMA_LONGER}'].iloc[-1]

    except Exception as e:
        st.warning(f"Indicator calculation error for {timeframe_suffix}: {e}")
        return {} # Return empty if error

    return indicators

def evaluate_playbook_setup(weekly_indicators, daily_indicators):
    """Evaluate Long and Short setup scores based on playbook rules"""
    long_score = 0
    short_score = 0
    rules_met_long = []
    rules_met_short = []

    # --- Check Long Playbook Rules ---
    # 1. Market Conditions (Weekly)
    if weekly_indicators.get('RSI_latest', np.nan) > RSI_MID:
        long_score += 1; rules_met_long.append("W_RSI > 50")
    if weekly_indicators.get('MACD_status') in ["Golden Cross", "Bullish"]:
        long_score += 1; rules_met_long.append("W_MACD Bull")
    if weekly_indicators.get('Price_vs_EMA_Long'): # Price > Weekly EMA_LONG
        long_score += 1; rules_met_long.append("W_Price > EMA_L")

    # 2. Entry Criteria (Daily)
    if daily_indicators.get('RSI_latest', np.nan) > RSI_MID:
        long_score += 1; rules_met_long.append("D_RSI > 50")
    if daily_indicators.get('MACD_status') in ["Golden Cross", "Bullish"]:
        long_score += 1; rules_met_long.append("D_MACD Bull")
    # Creative Price Structure Check: Is price above *both* short and long EMAs on daily?
    if daily_indicators.get('Price_vs_EMA_Short') and daily_indicators.get('Price_vs_EMA_Long'):
        long_score += 1; rules_met_long.append("D_Price > EMAs")

    # --- Check Short Playbook Rules ---
    # 1. Market Conditions (Weekly)
    if weekly_indicators.get('RSI_latest', np.nan) < RSI_MID:
        short_score += 1; rules_met_short.append("W_RSI < 50")
    if weekly_indicators.get('MACD_status') in ["Death Cross", "Bearish"]:
        short_score += 1; rules_met_short.append("W_MACD Bear")
    if not weekly_indicators.get('Price_vs_EMA_Long', True): # Price < Weekly EMA_LONG
        short_score += 1; rules_met_short.append("W_Price < EMA_L")

    # 2. Entry Criteria (Daily)
    if daily_indicators.get('RSI_latest', np.nan) < RSI_MID:
        short_score += 1; rules_met_short.append("D_RSI < 50")
    if daily_indicators.get('MACD_status') in ["Death Cross", "Bearish"]:
        short_score += 1; rules_met_short.append("D_MACD Bear")
    # Creative Price Structure Check: Is price below *both* short and long EMAs on daily?
    if not daily_indicators.get('Price_vs_EMA_Short', True) and not daily_indicators.get('Price_vs_EMA_Long', True):
        short_score += 1; rules_met_short.append("D_Price < EMAs")

    # Determine Setup Description
    setup_description = "Neutral / Conflicting"
    setup_class = "neutral-conflicting"
    if long_score >= STRONG_SETUP_THRESHOLD:
        setup_description = "Strong Long Setup"
        setup_class = "strong-long"
    elif long_score >= POTENTIAL_SETUP_THRESHOLD and short_score < POTENTIAL_SETUP_THRESHOLD:
         setup_description = "Potential Long Setup"
         setup_class = "potential-long"
    elif short_score >= STRONG_SETUP_THRESHOLD:
        setup_description = "Strong Short Setup"
        setup_class = "strong-short"
    elif short_score >= POTENTIAL_SETUP_THRESHOLD and long_score < POTENTIAL_SETUP_THRESHOLD:
        setup_description = "Potential Short Setup"
        setup_class = "potential-short"

    return {
        "long_score": long_score,
        "short_score": short_score,
        "setup_description": setup_description,
        "setup_class": setup_class,
        "rules_met_long": ", ".join(rules_met_long),
        "rules_met_short": ", ".join(rules_met_short)
    }

def analyze_ticker_strategy(ticker, name):
    """Analyze ticker based on the I Chart Daily Strategy"""
    daily_data, weekly_data = fetch_all_data(ticker)

    if daily_data is None or weekly_data is None:
        return {"ticker": ticker, "name": name, "error": "Data Fetch Failed"}

    # Calculate indicators for both timeframes
    daily_indicators = calculate_indicators(daily_data, "Daily")
    weekly_indicators = calculate_indicators(weekly_data, "Weekly")

    if not daily_indicators or not weekly_indicators:
         return {"ticker": ticker, "name": name, "error": "Indicator Calc Failed"}

    # Evaluate setup based on playbook
    setup_evaluation = evaluate_playbook_setup(weekly_indicators, daily_indicators)

    return {
        "ticker": ticker,
        "name": name,
        "daily_data": daily_data,
        "weekly_data": weekly_data,
        "daily_indicators": daily_indicators,
        "weekly_indicators": weekly_indicators,
        "setup_evaluation": setup_evaluation,
        "error": None
    }

def create_scanner_dataframe(results):
    """Create DataFrame summarizing the scanner results"""
    data = []
    for r in results:
        if r.get("error"):
            # Optionally include errors or skip
            # data.append({"Ticker": r["ticker"], "Name": r["name"], "Setup": r["error"]})
            continue

        wi = r["weekly_indicators"]
        di = r["daily_indicators"]
        se = r["setup_evaluation"]

        # Helper function to format indicators with icons
        def format_indicator(value, threshold=None, condition='greater'):
            if value is None or np.isnan(value): return "N/A"
            if threshold is not None:
                if condition == 'greater' and value > threshold:
                    icon = '<div class="indicator ok"></div>'
                elif condition == 'less' and value < threshold:
                    icon = '<div class="indicator ok"></div>' # OK for shorts
                else:
                    icon = '<div class="indicator nok"></div>'
            else: # Boolean checks like Price vs MA
                 icon = '<div class="indicator ok"></div>' if value else '<div class="indicator nok"></div>'

            if isinstance(value, (float, np.floating)):
                return f"{icon}{value:.1f}"
            elif isinstance(value, bool):
                 return icon
            else: # String status like MACD
                 status_class = "ok" if "Bull" in value or "Golden" in value else "nok" if "Bear" in value or "Death" in value else "neutral"
                 icon = f'<div class="indicator {status_class}"></div>'
                 return f"{icon}{value}"

        data.append({
            "Name": r["name"],
            "Ticker": r["ticker"],
            "Setup": f"<span class='{se['setup_class']}'>{se['setup_description']}</span>",
            "L Score": se['long_score'],
            "S Score": se['short_score'],
            "W RSI": format_indicator(wi.get('RSI_latest'), RSI_MID, 'greater'),
            "W MACD": format_indicator(wi.get('MACD_status')),
            "W Prc>EMA": format_indicator(wi.get('Price_vs_EMA_Long')),
            "D RSI": format_indicator(di.get('RSI_latest'), RSI_MID, 'greater'),
            "D MACD": format_indicator(di.get('MACD_status')),
            "D Prc>EMAs": format_indicator(di.get('Price_vs_EMA_Short') and di.get('Price_vs_EMA_Long')),
            "_rules_long": se.get('rules_met_long', ''), # For tooltips or detailed view later
            "_rules_short": se.get('rules_met_short', ''),
             "_index": results.index(r) # Store original index
        })

    df = pd.DataFrame(data)
    # Make Setup column HTML renderable
    df['Setup'] = df['Setup'].astype(str)
    return df

def display_scanner_charts(result):
    """Display Daily and Weekly charts with strategy indicators"""
    if not result or result.get("error"):
        st.error("No chart data available or error encountered.")
        return

    st.title(f"{result['name']} ({result['ticker']})")
    se = result["setup_evaluation"]
    st.subheader(f"Setup Evaluation: <span class='{se['setup_class']}'>{se['setup_description']}</span> (Long: {se['long_score']}, Short: {se['short_score']})", unsafe_allow_html=True)
    # Optionally display rules met:
    # st.caption(f"Long Rules Met: {se.get('rules_met_long','')}")
    # st.caption(f"Short Rules Met: {se.get('rules_met_short','')}")


    daily_data = result["daily_data"]
    weekly_data = result["weekly_data"]
    di = result["daily_indicators"]
    wi = result["weekly_indicators"]

    # --- Create Daily Chart ---
    fig_daily = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                              row_heights=[0.6, 0.2, 0.2],
                              subplot_titles=("Daily Price & EMAs", "Daily RSI", "Daily MACD"))

    # Daily Price
    fig_daily.add_trace(go.Candlestick(x=daily_data.index, open=daily_data['Open'], high=daily_data['High'], low=daily_data['Low'], close=daily_data['Close'], name="Price"), row=1, col=1)
    fig_daily.add_trace(go.Scatter(x=daily_data.index, y=di.get(f'EMA_{EMA_SHORT}'), name=f"EMA {EMA_SHORT}", line=dict(color='cyan', width=1)), row=1, col=1)
    fig_daily.add_trace(go.Scatter(x=daily_data.index, y=di.get(f'EMA_{EMA_LONG}'), name=f"EMA {EMA_LONG}", line=dict(color='magenta', width=1)), row=1, col=1)
    if f'EMA_{EMA_LONGER}' in di:
       fig_daily.add_trace(go.Scatter(x=daily_data.index, y=di.get(f'EMA_{EMA_LONGER}'), name=f"EMA {EMA_LONGER}", line=dict(color='yellow', width=1, dash='dot')), row=1, col=1)

    # Daily RSI
    fig_daily.add_trace(go.Scatter(x=daily_data.index, y=di.get(f'RSI_{RSI_WINDOW}'), name="RSI", line=dict(color='orange')), row=2, col=1)
    fig_daily.add_hline(y=RSI_MID, line_dash="dash", line_color="grey", row=2, col=1)
    fig_daily.add_hline(y=70, line_dash="dot", line_color="red", row=2, col=1)
    fig_daily.add_hline(y=30, line_dash="dot", line_color="green", row=2, col=1)

    # Daily MACD
    colors = ['green' if val >= 0 else 'red' for val in di.get(f'MACD_hist', [])]
    fig_daily.add_trace(go.Bar(x=daily_data.index, y=di.get(f'MACD_hist'), name='MACD Hist', marker_color=colors), row=3, col=1)
    fig_daily.add_trace(go.Scatter(x=daily_data.index, y=di.get(f'MACD_line'), name="MACD", line=dict(color="blue")), row=3, col=1)
    fig_daily.add_trace(go.Scatter(x=daily_data.index, y=di.get(f'MACD_signal'), name="Signal", line=dict(color="red")), row=3, col=1)

    fig_daily.update_layout(title="Daily Timeframe Analysis", height=700, xaxis_rangeslider_visible=False, showlegend=True)
    fig_daily.update_yaxes(range=[0, 100], row=2, col=1) # RSI Range

    # --- Create Weekly Chart ---
    fig_weekly = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                              row_heights=[0.6, 0.2, 0.2],
                              subplot_titles=("Weekly Price & EMAs", "Weekly RSI", "Weekly MACD"))

    # Weekly Price
    fig_weekly.add_trace(go.Candlestick(x=weekly_data.index, open=weekly_data['Open'], high=weekly_data['High'], low=weekly_data['Low'], close=weekly_data['Close'], name="Price"), row=1, col=1)
    fig_weekly.add_trace(go.Scatter(x=weekly_data.index, y=wi.get(f'EMA_{EMA_SHORT}'), name=f"EMA {EMA_SHORT}", line=dict(color='cyan', width=1)), row=1, col=1)
    fig_weekly.add_trace(go.Scatter(x=weekly_data.index, y=wi.get(f'EMA_{EMA_LONG}'), name=f"EMA {EMA_LONG}", line=dict(color='magenta', width=1)), row=1, col=1)
    if f'EMA_{EMA_LONGER}' in wi:
       fig_weekly.add_trace(go.Scatter(x=weekly_data.index, y=wi.get(f'EMA_{EMA_LONGER}'), name=f"EMA {EMA_LONGER}", line=dict(color='yellow', width=1, dash='dot')), row=1, col=1)

    # Weekly RSI
    fig_weekly.add_trace(go.Scatter(x=weekly_data.index, y=wi.get(f'RSI_{RSI_WINDOW}'), name="RSI", line=dict(color='orange')), row=2, col=1)
    fig_weekly.add_hline(y=RSI_MID, line_dash="dash", line_color="grey", row=2, col=1)
    fig_weekly.add_hline(y=70, line_dash="dot", line_color="red", row=2, col=1)
    fig_weekly.add_hline(y=30, line_dash="dot", line_color="green", row=2, col=1)

    # Weekly MACD
    colors_w = ['green' if val >= 0 else 'red' for val in wi.get(f'MACD_hist', [])]
    fig_weekly.add_trace(go.Bar(x=weekly_data.index, y=wi.get(f'MACD_hist'), name='MACD Hist', marker_color=colors_w), row=3, col=1)
    fig_weekly.add_trace(go.Scatter(x=weekly_data.index, y=wi.get(f'MACD_line'), name="MACD", line=dict(color="blue")), row=3, col=1)
    fig_weekly.add_trace(go.Scatter(x=weekly_data.index, y=wi.get(f'MACD_signal'), name="Signal", line=dict(color="red")), row=3, col=1)

    fig_weekly.update_layout(title="Weekly Timeframe Analysis", height=700, xaxis_rangeslider_visible=False, showlegend=True)
    fig_weekly.update_yaxes(range=[0, 100], row=2, col=1) # RSI Range

    # Display charts in tabs
    daily_tab, weekly_tab = st.tabs(["Daily Chart", "Weekly Chart"])
    with daily_tab:
        st.plotly_chart(fig_daily, use_container_width=True)
    with weekly_tab:
        st.plotly_chart(fig_weekly, use_container_width=True)


# --- Streamlit App Main Logic ---

def scan_tickers_strategy_st(categories=None, specific_tickers=None):
    """Scan selected tickers using the strategy and display progress"""
    results = []
    tickers_to_scan = {}

    if specific_tickers:
        for ticker in specific_tickers:
            found = False
            for cat_tickers in TICKER_CATEGORIES.values():
                if ticker in cat_tickers:
                    tickers_to_scan[ticker] = cat_tickers[ticker]
                    found = True
                    break
            if not found:
                 st.warning(f"Ticker {ticker} not found in categories, attempting direct scan.")
                 tickers_to_scan[ticker] = ticker # Use ticker symbol as name
    elif categories:
        for cat in categories:
            tickers_to_scan.update(TICKER_CATEGORIES.get(cat, {}))
    else: # Scan all
        for cat_tickers in TICKER_CATEGORIES.values():
            tickers_to_scan.update(cat_tickers)

    total_tickers = len(tickers_to_scan)
    if total_tickers == 0:
        st.warning("No tickers selected or found for scanning.")
        return []

    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, (ticker, name) in enumerate(tickers_to_scan.items()):
        status_text.text(f"Analyzing {i+1}/{total_tickers}: {name} ({ticker})...")
        result = analyze_ticker_strategy(ticker, name)
        results.append(result)
        progress_bar.progress((i + 1) / total_tickers)

    status_text.text(f"Scan Complete: {total_tickers} tickers analyzed.")
    return results

def display_strategy_summary(results):
    """Display summary based on setup scores"""
    valid_results = [r for r in results if not r.get("error")]
    if not valid_results:
        st.warning("No valid results for summary.")
        return

    counts = {"Strong Long": 0, "Potential Long": 0, "Strong Short": 0, "Potential Short": 0, "Neutral / Conflicting": 0}
    for r in valid_results:
        desc = r["setup_evaluation"]["setup_description"]
        if desc in counts:
            counts[desc] += 1

    total = len(valid_results)
    percentages = {k: (v / total * 100) for k, v in counts.items()}

    st.markdown(f"""
    <div class="summary-box">
        <h3>Strategy Setup Summary</h3>
        <p>Total tickers analyzed: {total}</p>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); grid-gap: 10px;">
            <div><span class="strong-long">Strong Long:</span> {counts['Strong Long']} ({percentages['Strong Long']:.1f}%)</div>
            <div><span class="potential-long">Potential Long:</span> {counts['Potential Long']} ({percentages['Potential Long']:.1f}%)</div>
            <div><span class="strong-short">Strong Short:</span> {counts['Strong Short']} ({percentages['Strong Short']:.1f}%)</div>
            <div><span class="potential-short">Potential Short:</span> {counts['Potential Short']} ({percentages['Potential Short']:.1f}%)</div>
            <div><span class="neutral-conflicting">Neutral/Conflicting:</span> {counts['Neutral / Conflicting']} ({percentages['Neutral / Conflicting']:.1f}%)</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Pie chart
    labels = list(counts.keys())
    values = list(counts.values())
    colors = {'Strong Long': '#00A100', 'Potential Long': '#0ECB81',
              'Strong Short': '#D20000', 'Potential Short': '#F6465D',
              'Neutral / Conflicting': '#8A8A8A'}
    pie_colors = [colors.get(label, '#CCCCCC') for label in labels]

    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3, marker=dict(colors=pie_colors))])
    fig.update_layout(title="Setup Distribution", height=400, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig, use_container_width=True)

# --- Main App ---
def main():
    st.title("📊 I Chart Daily Strategy Scanner")

    if 'results' not in st.session_state: st.session_state.results = []
    if 'selected_ticker_idx' not in st.session_state: st.session_state.selected_ticker_idx = None
    if 'last_scan_time' not in st.session_state: st.session_state.last_scan_time = None

    # Sidebar for scan settings
    st.sidebar.title("Scan Settings")
    scan_option = st.sidebar.radio("Scan Options", ("All Categories", "Select Categories", "Specific Tickers"), index=1)
    categories, specific_tickers = None, None
    if scan_option == "Select Categories":
        available_categories = list(TICKER_CATEGORIES.keys())
        categories = st.sidebar.multiselect("Select categories", available_categories, default=["INDICES", "COMMODITIES", "FOREX"])
    elif scan_option == "Specific Tickers":
        ticker_input = st.sidebar.text_area("Enter tickers (comma-separated)")
        if ticker_input: specific_tickers = [t.strip().upper() for t in ticker_input.split(',')]

    # Auto-refresh (optional - keep simple for now)
    refresh_pressed = st.sidebar.button("Scan Now", use_container_width=True, type="primary")

    # --- Main Area Tabs ---
    tab1, tab2 = st.tabs(["📊 Scanner Dashboard", "📈 Detailed Analysis"])

    with tab1:
        if refresh_pressed or not st.session_state.results:
            st.session_state.results = scan_tickers_strategy_st(categories, specific_tickers)
            st.session_state.last_scan_time = datetime.now()

        if st.session_state.last_scan_time:
            st.success(f"Last scan: {st.session_state.last_scan_time.strftime('%Y-%m-%d %H:%M:%S')}")

        if st.session_state.results:
            # Display Summary
            display_strategy_summary(st.session_state.results)

            # Display Results Table
            st.subheader("Scan Results")
            df = create_scanner_dataframe(st.session_state.results)

            if not df.empty:
                 # Sort options focusing on scores
                sort_options = {
                    "Strongest Long First": {"column": "L Score", "ascending": False},
                    "Strongest Short First": {"column": "S Score", "ascending": False},
                    "Alphabetical (A-Z)": {"column": "Name", "ascending": True},
                }
                sort_choice = st.selectbox("Sort by:", options=list(sort_options.keys()), index=0)
                sort_config = sort_options[sort_choice]
                df_sorted = df.sort_values(by=sort_config["column"], ascending=sort_config["ascending"]).reset_index(drop=True)

                # Columns to display - Keep it concise
                display_columns = ["Name", "Ticker", "Setup", "L Score", "S Score", "W RSI", "W MACD", "D RSI", "D MACD"]
                df_display = df_sorted[display_columns]

                st.markdown('<div class="dataframe-container">', unsafe_allow_html=True)
                # Use st.write to render HTML, or st.dataframe with tweaks if needed
                st.write(df_display.to_html(escape=False, index=False), unsafe_allow_html=True)
                # st.dataframe(df_display, hide_index=True, use_container_width=True) # Alternative if HTML is tricky
                st.markdown('</div>', unsafe_allow_html=True)

                st.markdown("---")
                # Selection for Detailed View
                st.write("Select an instrument for detailed analysis:")
                selected_name = st.selectbox("Instrument Name", options=df_sorted["Name"].tolist(), index=0, key="detail_select")

                if st.button("View Detailed Analysis"):
                    # Find the original index from the potentially sorted dataframe
                    selected_row = df_sorted[df_sorted["Name"] == selected_name].iloc[0]
                    original_index = int(selected_row["_index"]) # Get back the original index
                    st.session_state.selected_ticker_idx = original_index
                    # Trigger rerun to switch tab implicitly or add logic to switch tabs
                    st.info(f"Switch to 'Detailed Analysis' tab to view {selected_name}")
                    # st.experimental_rerun() # Optional: force rerun if tab switching needs help

            else:
                st.warning("No valid results found matching criteria.")
        else:
             st.info("Click 'Scan Now' to begin.")

    with tab2:
        st.subheader("Detailed Strategy Analysis")
        if st.session_state.selected_ticker_idx is not None and st.session_state.results:
            selected_result = st.session_state.results[st.session_state.selected_ticker_idx]

            # Add a way to clear selection or go back
            if st.button("← Clear Selection / Back to Dashboard"):
                 st.session_state.selected_ticker_idx = None
                 st.experimental_rerun() # Rerun to clear the view

            display_scanner_charts(selected_result)
        else:
            st.info("Select an instrument from the Dashboard table and click 'View Detailed Analysis' to see charts here.")


if __name__ == "__main__":
    main()
