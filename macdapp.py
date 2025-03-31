import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pandas_ta as ta # Essential for indicator calculation
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import ticker categories (keep using your tickers.py)
from tickers import TICKER_CATEGORIES

# --- Strategy Configuration ---
# Timeframes
TF_CONDITIONS = '1wk' # Timeframe for Market Conditions (Weekly)
TF_ENTRY = '1d'       # Timeframe for Entry Signals (Daily)
# Data Periods (adjust if needed for indicator accuracy)
PERIOD_CONDITIONS = "5y"
PERIOD_ENTRY = "1y"
# Moving Averages (Using EMA as per video examples)
EMA_SHORT = 11
EMA_LONG = 21
EMA_CONTEXT = 50 # Optional longer MA for weekly context
# RSI
RSI_WINDOW = 14
RSI_MID = 50
# MACD
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
# Setup Threshold (How many ENTRY rules must be met *after* conditions are met)
MIN_ENTRY_RULES_MET = 2 # e.g., Need at least 2 out of 3 Daily rules

# --- Page Config ---
st.set_page_config(
    page_title="I Chart Daily Setup Scanner",
    page_icon="🎯",
    layout="wide",
)

# --- Custom CSS ---
st.markdown("""
<style>
    .setup-long { background-color: #d4edda; color: #155724; padding: 2px 5px; border-radius: 3px; font-weight: bold; }
    .setup-short { background-color: #f8d7da; color: #721c24; padding: 2px 5px; border-radius: 3px; font-weight: bold; }
    .rule-met { color: green; }
    .rule-not-met { color: red; }
    .dataframe-container { border-radius: 5px; overflow: hidden; }
    .dataframe tr:hover { background-color: rgba(0,0,0,0.05) !important; cursor: pointer; }
</style>
""", unsafe_allow_html=True)

# --- Helper Functions ---

@st.cache_data(ttl=1800) # Cache for 30 minutes
def fetch_strategy_data(ticker):
    """Fetches Weekly and Daily data required for the strategy"""
    try:
        ticker_obj = yf.Ticker(ticker)
        data_conditions = ticker_obj.history(period=PERIOD_CONDITIONS, interval=TF_CONDITIONS)
        data_entry = ticker_obj.history(period=PERIOD_ENTRY, interval=TF_ENTRY)

        # Basic validation
        min_len_cond = max(EMA_LONG, MACD_SLOW) + 5 # Need enough data for longest indicator
        min_len_entry = max(EMA_LONG, MACD_SLOW) + 5
        if data_conditions.empty or len(data_conditions) < min_len_cond or \
           data_entry.empty or len(data_entry) < min_len_entry:
            # st.warning(f"Insufficient data for {ticker}")
            return None, None
        return data_conditions, data_entry
    except Exception as e:
        # st.error(f"Error fetching data for {ticker}: {e}")
        return None, None

def calculate_strategy_indicators(data):
    """Calculates all necessary indicators using pandas_ta"""
    if data is None or data.empty:
        return None
    try:
        # EMAs
        data.ta.ema(length=EMA_SHORT, append=True, col_names=(f"EMA_{EMA_SHORT}",))
        data.ta.ema(length=EMA_LONG, append=True, col_names=(f"EMA_{EMA_LONG}",))
        # data.ta.ema(length=EMA_CONTEXT, append=True, col_names=(f"EMA_{EMA_CONTEXT}",)) # Optional

        # RSI
        data.ta.rsi(length=RSI_WINDOW, append=True, col_names=(f"RSI_{RSI_WINDOW}",))

        # MACD
        data.ta.macd(fast=MACD_FAST, slow=MACD_SLOW, signal=MACD_SIGNAL, append=True,
                    col_names=(f"MACD_{MACD_FAST}_{MACD_SLOW}_{MACD_SIGNAL}",
                               f"MACDh_{MACD_FAST}_{MACD_SLOW}_{MACD_SIGNAL}", # Histogram
                               f"MACDs_{MACD_FAST}_{MACD_SLOW}_{MACD_SIGNAL}")) # Signal

        # Calculate latest values / states
        indicators = {}
        indicators['Close'] = data['Close'].iloc[-1]
        indicators[f'EMA_{EMA_SHORT}'] = data[f'EMA_{EMA_SHORT}'].iloc[-1]
        indicators[f'EMA_{EMA_LONG}'] = data[f'EMA_{EMA_LONG}'].iloc[-1]
        # indicators[f'EMA_{EMA_CONTEXT}'] = data[f'EMA_{EMA_CONTEXT}'].iloc[-1]
        indicators[f'RSI_{RSI_WINDOW}'] = data[f'RSI_{RSI_WINDOW}'].iloc[-1]
        indicators[f'MACD_Line'] = data[f"MACD_{MACD_FAST}_{MACD_SLOW}_{MACD_SIGNAL}"].iloc[-1]
        indicators[f'MACD_Signal'] = data[f"MACDs_{MACD_FAST}_{MACD_SLOW}_{MACD_SIGNAL}"].iloc[-1]
        indicators[f'MACD_Hist'] = data[f"MACDh_{MACD_FAST}_{MACD_SLOW}_{MACD_SIGNAL}"].iloc[-1]

        # --- Derived Boolean States for Easier Rule Checking ---
        # RSI State
        indicators['RSI_Bullish'] = indicators[f'RSI_{RSI_WINDOW}'] > RSI_MID
        indicators['RSI_Bearish'] = indicators[f'RSI_{RSI_WINDOW}'] < RSI_MID

        # MACD State (Simplified: Line vs Signal)
        indicators['MACD_Bullish'] = indicators['MACD_Line'] > indicators['MACD_Signal']
        indicators['MACD_Bearish'] = indicators['MACD_Line'] < indicators['MACD_Signal']
        # Optional: Check for recent cross (more complex, omitted for simplicity now)

        # Price vs EMAs State
        indicators['Price_Above_EMA_Short'] = indicators['Close'] > indicators[f'EMA_{EMA_SHORT}']
        indicators['Price_Above_EMA_Long'] = indicators['Close'] > indicators[f'EMA_{EMA_LONG}']
        indicators['Price_Below_EMA_Short'] = indicators['Close'] < indicators[f'EMA_{EMA_SHORT}']
        indicators['Price_Below_EMA_Long'] = indicators['Close'] < indicators[f'EMA_{EMA_LONG}']

        return indicators, data # Return indicators dict and dataframe with calculated columns
    except Exception as e:
        # st.warning(f"Indicator calculation error: {e}")
        return None, None

def check_strategy_setup(conditions_indicators, entry_indicators):
    """Checks if the indicators meet the Long or Short setup criteria"""
    if not conditions_indicators or not entry_indicators:
        return "Error", 0, [], "Error", 0, []

    setup_long = "None"
    score_long = 0
    rules_met_long_details = []

    setup_short = "None"
    score_short = 0
    rules_met_short_details = []

    # --- Evaluate LONG Setup ---
    # 1. Conditions Check (Weekly)
    cond_r_ok = conditions_indicators.get('RSI_Bullish', False)
    cond_m_ok = conditions_indicators.get('MACD_Bullish', False)
    cond_p_ok = conditions_indicators.get('Price_Above_EMA_Long', False) # Price above longer EMA on weekly
    conditions_long_met = cond_r_ok and cond_m_ok and cond_p_ok # ALL Weekly conditions must be met

    if conditions_long_met:
        score_long = 3 # Base score for conditions met
        rules_met_long_details = ["W:RSI>50", "W:MACD Bull", "W:Prc>EMA_L"]

        # 2. Entry Check (Daily) - Only if conditions met
        entry_rules_met_count = 0
        if entry_indicators.get('RSI_Bullish', False):
            entry_rules_met_count += 1; rules_met_long_details.append("D:RSI>50")
        if entry_indicators.get('MACD_Bullish', False):
            entry_rules_met_count += 1; rules_met_long_details.append("D:MACD Bull")
        if entry_indicators.get('Price_Above_EMA_Short', False) and entry_indicators.get('Price_Above_EMA_Long', False):
            entry_rules_met_count += 1; rules_met_long_details.append("D:Prc>EMAs")

        if entry_rules_met_count >= MIN_ENTRY_RULES_MET:
            setup_long = "Potential Long"
            score_long += entry_rules_met_count
        else:
             # Conditions met, but entry not triggered yet
             setup_long = "Watch Long" # Indicate conditions are good, entry pending
             score_long = score_long # Keep score reflecting conditions met


    # --- Evaluate SHORT Setup ---
    # 1. Conditions Check (Weekly)
    cond_r_ok_s = conditions_indicators.get('RSI_Bearish', False)
    cond_m_ok_s = conditions_indicators.get('MACD_Bearish', False)
    cond_p_ok_s = conditions_indicators.get('Price_Below_EMA_Long', False) # Price below longer EMA on weekly
    conditions_short_met = cond_r_ok_s and cond_m_ok_s and cond_p_ok_s # ALL Weekly conditions must be met

    if conditions_short_met:
        score_short = 3 # Base score for conditions met
        rules_met_short_details = ["W:RSI<50", "W:MACD Bear", "W:Prc<EMA_L"]

        # 2. Entry Check (Daily) - Only if conditions met
        entry_rules_met_count_s = 0
        if entry_indicators.get('RSI_Bearish', False):
            entry_rules_met_count_s += 1; rules_met_short_details.append("D:RSI<50")
        if entry_indicators.get('MACD_Bearish', False):
            entry_rules_met_count_s += 1; rules_met_short_details.append("D:MACD Bear")
        if entry_indicators.get('Price_Below_EMA_Short', False) and entry_indicators.get('Price_Below_EMA_Long', False):
            entry_rules_met_count_s += 1; rules_met_short_details.append("D:Prc<EMAs")

        if entry_rules_met_count_s >= MIN_ENTRY_RULES_MET:
            setup_short = "Potential Short"
            score_short += entry_rules_met_count_s
        else:
             # Conditions met, but entry not triggered yet
             setup_short = "Watch Short"
             score_short = score_short # Keep score reflecting conditions met

    # Determine final setup type (prioritize potential setups)
    final_setup = "None"
    final_score = 0
    final_rules = []
    if setup_long == "Potential Long":
        final_setup = "Potential Long"
        final_score = score_long
        final_rules = rules_met_long_details
    elif setup_short == "Potential Short":
        final_setup = "Potential Short"
        final_score = score_short
        final_rules = rules_met_short_details
    elif setup_long == "Watch Long":
        final_setup = "Watch Long"
        final_score = score_long
        final_rules = rules_met_long_details
    elif setup_short == "Watch Short":
        final_setup = "Watch Short"
        final_score = score_short
        final_rules = rules_met_short_details


    return final_setup, final_score, final_rules


def scan_tickers(tickers_dict):
    """Scans a dictionary of tickers {symbol: name} for strategy setups"""
    results = []
    total_tickers = len(tickers_dict)
    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, (ticker, name) in enumerate(tickers_dict.items()):
        status_text.text(f"Scanning {i+1}/{total_tickers}: {name} ({ticker})...")
        data_conditions, data_entry = fetch_strategy_data(ticker)

        if data_conditions is None or data_entry is None:
            results.append({"ticker": ticker, "name": name, "Setup": "Data Error", "Score": 0, "Rules Met": [], "error": True})
            progress_bar.progress((i + 1) / total_tickers)
            continue

        conditions_indicators, data_conditions_with_indicators = calculate_strategy_indicators(data_conditions)
        entry_indicators, data_entry_with_indicators = calculate_strategy_indicators(data_entry)

        if conditions_indicators is None or entry_indicators is None:
            results.append({"ticker": ticker, "name": name, "Setup": "Calc Error", "Score": 0, "Rules Met": [], "error": True})
            progress_bar.progress((i + 1) / total_tickers)
            continue

        setup_type, setup_score, rules_met = check_strategy_setup(conditions_indicators, entry_indicators)

        results.append({
            "ticker": ticker,
            "name": name,
            "Setup": setup_type,
            "Score": setup_score,
            "Rules Met": ", ".join(rules_met),
            "error": False,
            # Store data for detailed view
            "_data_conditions": data_conditions_with_indicators,
            "_data_entry": data_entry_with_indicators,
            "_indicators_conditions": conditions_indicators,
            "_indicators_entry": entry_indicators,
        })
        progress_bar.progress((i + 1) / total_tickers)

    status_text.text(f"Scan Complete: {total_tickers} tickers analyzed.")
    return results

def display_results_table(results_list):
    """Displays the filtered scan results in a table"""
    if not results_list:
        st.warning("No results to display.")
        return None

    df_data = []
    for i, r in enumerate(results_list):
        if r['Setup'] == "Error" or r['Setup'] == "None": continue # Skip errors and non-setups for now

        setup_class = "setup-long" if "Long" in r['Setup'] else "setup-short" if "Short" in r['Setup'] else ""
        setup_html = f"<span class='{setup_class}'>{r['Setup']}</span>"

        # Format key indicators for quick view
        ci = r.get('_indicators_conditions', {})
        ei = r.get('_indicators_entry', {})
        wrsi_val = ci.get(f'RSI_{RSI_WINDOW}', float('nan'))
        drsi_val = ei.get(f'RSI_{RSI_WINDOW}', float('nan'))
        wmacd_bull = ci.get('MACD_Bullish', None)
        dmacd_bull = ei.get('MACD_Bullish', None)

        wrsi_str = f"{wrsi_val:.1f}" if not np.isnan(wrsi_val) else "N/A"
        drsi_str = f"{drsi_val:.1f}" if not np.isnan(drsi_val) else "N/A"
        wmacd_str = "Bull" if wmacd_bull else "Bear" if wmacd_bull is not None else "N/A"
        dmacd_str = "Bull" if dmacd_bull else "Bear" if dmacd_bull is not None else "N/A"


        df_data.append({
            "Name": r["name"],
            "Ticker": r["ticker"],
            "Setup": setup_html,
            "Score": r["Score"],
            "W:RSI": wrsi_str,
            "W:MACD": wmacd_str,
            "D:RSI": drsi_str,
            "D:MACD": dmacd_str,
            # "Rules Met": r["Rules Met"], # Keep table cleaner, show in detail
             "_original_index": i # Link back to the full results list
        })

    if not df_data:
        st.info("No potential Long or Short setups found based on current criteria.")
        return None

    df_display = pd.DataFrame(df_data)

    # Sort by Score descending by default
    df_display = df_display.sort_values(by="Score", ascending=False).reset_index(drop=True)

    st.markdown('<div class="dataframe-container">', unsafe_allow_html=True)
    # Display using st.write with HTML for styling
    st.write(df_display.to_html(escape=False, index=False), unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    return df_display # Return the displayed dataframe for selection purposes

def display_detailed_charts(result):
    """Displays Weekly and Daily charts for a selected result"""
    st.header(f"Detailed Analysis: {result['name']} ({result['ticker']})")
    st.subheader(f"Detected Setup: <span class='{ 'setup-long' if 'Long' in result['Setup'] else 'setup-short' if 'Short' in result['Setup'] else '' }'>{result['Setup']}</span> (Score: {result['Score']})", unsafe_allow_html=True)
    st.caption(f"Rules Met: {result['Rules Met']}")
    st.markdown("---")

    data_conditions = result.get('_data_conditions')
    data_entry = result.get('_data_entry')

    if data_conditions is None or data_entry is None:
        st.error("Chart data not available.")
        return

    # --- Create Weekly Chart (Conditions Timeframe) ---
    fig_w = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                          row_heights=[0.6, 0.2, 0.2],
                          subplot_titles=(f"{TF_CONDITIONS} Price & EMAs", f"{TF_CONDITIONS} RSI ({RSI_WINDOW})", f"{TF_CONDITIONS} MACD ({MACD_FAST},{MACD_SLOW},{MACD_SIGNAL})"))
    # Price & EMAs
    fig_w.add_trace(go.Candlestick(x=data_conditions.index, open=data_conditions['Open'], high=data_conditions['High'], low=data_conditions['Low'], close=data_conditions['Close'], name="Price"), row=1, col=1)
    fig_w.add_trace(go.Scatter(x=data_conditions.index, y=data_conditions.get(f'EMA_{EMA_SHORT}'), name=f"EMA {EMA_SHORT}", line=dict(color='cyan', width=1)), row=1, col=1)
    fig_w.add_trace(go.Scatter(x=data_conditions.index, y=data_conditions.get(f'EMA_{EMA_LONG}'), name=f"EMA {EMA_LONG}", line=dict(color='magenta', width=1)), row=1, col=1)
    # RSI
    fig_w.add_trace(go.Scatter(x=data_conditions.index, y=data_conditions.get(f'RSI_{RSI_WINDOW}'), name="RSI", line=dict(color='orange')), row=2, col=1)
    fig_w.add_hline(y=RSI_MID, line_dash="dash", line_color="grey", row=2, col=1)
    # MACD
    colors_w = ['green' if val >= 0 else 'red' for val in data_conditions.get(f"MACDh_{MACD_FAST}_{MACD_SLOW}_{MACD_SIGNAL}", [])]
    fig_w.add_trace(go.Bar(x=data_conditions.index, y=data_conditions.get(f"MACDh_{MACD_FAST}_{MACD_SLOW}_{MACD_SIGNAL}"), name='Hist', marker_color=colors_w), row=3, col=1)
    fig_w.add_trace(go.Scatter(x=data_conditions.index, y=data_conditions.get(f"MACD_{MACD_FAST}_{MACD_SLOW}_{MACD_SIGNAL}"), name="MACD", line=dict(color="blue")), row=3, col=1)
    fig_w.add_trace(go.Scatter(x=data_conditions.index, y=data_conditions.get(f"MACDs_{MACD_FAST}_{MACD_SLOW}_{MACD_SIGNAL}"), name="Signal", line=dict(color="red")), row=3, col=1)
    fig_w.update_layout(title=f"Weekly ({TF_CONDITIONS}) Chart - Market Conditions", height=600, xaxis_rangeslider_visible=False, showlegend=False)
    fig_w.update_yaxes(range=[0, 100], row=2, col=1)
    st.plotly_chart(fig_w, use_container_width=True)

    st.markdown("---")

    # --- Create Daily Chart (Entry Timeframe) ---
    fig_d = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                          row_heights=[0.6, 0.2, 0.2],
                          subplot_titles=(f"{TF_ENTRY} Price & EMAs", f"{TF_ENTRY} RSI ({RSI_WINDOW})", f"{TF_ENTRY} MACD ({MACD_FAST},{MACD_SLOW},{MACD_SIGNAL})"))
    # Price & EMAs
    fig_d.add_trace(go.Candlestick(x=data_entry.index, open=data_entry['Open'], high=data_entry['High'], low=data_entry['Low'], close=data_entry['Close'], name="Price"), row=1, col=1)
    fig_d.add_trace(go.Scatter(x=data_entry.index, y=data_entry.get(f'EMA_{EMA_SHORT}'), name=f"EMA {EMA_SHORT}", line=dict(color='cyan', width=1)), row=1, col=1)
    fig_d.add_trace(go.Scatter(x=data_entry.index, y=data_entry.get(f'EMA_{EMA_LONG}'), name=f"EMA {EMA_LONG}", line=dict(color='magenta', width=1)), row=1, col=1)
    # RSI
    fig_d.add_trace(go.Scatter(x=data_entry.index, y=data_entry.get(f'RSI_{RSI_WINDOW}'), name="RSI", line=dict(color='orange')), row=2, col=1)
    fig_d.add_hline(y=RSI_MID, line_dash="dash", line_color="grey", row=2, col=1)
    # MACD
    colors_d = ['green' if val >= 0 else 'red' for val in data_entry.get(f"MACDh_{MACD_FAST}_{MACD_SLOW}_{MACD_SIGNAL}", [])]
    fig_d.add_trace(go.Bar(x=data_entry.index, y=data_entry.get(f"MACDh_{MACD_FAST}_{MACD_SLOW}_{MACD_SIGNAL}"), name='Hist', marker_color=colors_d), row=3, col=1)
    fig_d.add_trace(go.Scatter(x=data_entry.index, y=data_entry.get(f"MACD_{MACD_FAST}_{MACD_SLOW}_{MACD_SIGNAL}"), name="MACD", line=dict(color="blue")), row=3, col=1)
    fig_d.add_trace(go.Scatter(x=data_entry.index, y=data_entry.get(f"MACDs_{MACD_FAST}_{MACD_SLOW}_{MACD_SIGNAL}"), name="Signal", line=dict(color="red")), row=3, col=1)
    fig_d.update_layout(title=f"Daily ({TF_ENTRY}) Chart - Entry Criteria", height=600, xaxis_rangeslider_visible=False, showlegend=False)
    fig_d.update_yaxes(range=[0, 100], row=2, col=1)
    st.plotly_chart(fig_d, use_container_width=True)


# --- Main App Flow ---
def main():
    st.title("🎯 I Chart Daily Strategy Setup Scanner")
    st.caption("Scans for potential Long/Short setups based on Weekly conditions and Daily entry criteria.")

    # Initialize session state
    if 'scan_results' not in st.session_state: st.session_state.scan_results = []
    if 'selected_instrument_index' not in st.session_state: st.session_state.selected_instrument_index = None

    # Sidebar for Ticker Selection
    st.sidebar.title("Scan Settings")
    scan_option = st.sidebar.radio("Select Tickers To Scan:", ("All Categories", "Select Categories", "Specific Tickers"), index=1, key="scan_option")

    tickers_to_scan = {}
    if scan_option == "Select Categories":
        available_categories = list(TICKER_CATEGORIES.keys())
        selected_categories = st.sidebar.multiselect("Categories:", available_categories, default=["INDICES", "COMMODITIES", "FOREX"], key="sel_cats")
        if selected_categories:
            for cat in selected_categories:
                tickers_to_scan.update(TICKER_CATEGORIES.get(cat, {}))
        else:
            st.sidebar.warning("Please select at least one category.")
    elif scan_option == "Specific Tickers":
        ticker_input = st.sidebar.text_area("Enter tickers (comma-separated):", key="spec_ticks")
        if ticker_input:
            specific_tickers_list = [t.strip().upper() for t in ticker_input.split(',')]
            # Try to find names from categories, default to ticker symbol if not found
            for ticker in specific_tickers_list:
                found = False
                for cat_tickers in TICKER_CATEGORIES.values():
                    if ticker in cat_tickers:
                        tickers_to_scan[ticker] = cat_tickers[ticker]
                        found = True
                        break
                if not found:
                    tickers_to_scan[ticker] = ticker # Use symbol as name
        else:
            st.sidebar.warning("Please enter at least one ticker.")
    else: # All Categories
        for cat_tickers in TICKER_CATEGORIES.values():
            tickers_to_scan.update(cat_tickers)

    # Scan Button
    if st.sidebar.button("▶️ Run Scan", use_container_width=True, type="primary", disabled=(len(tickers_to_scan) == 0)):
        st.session_state.scan_results = scan_tickers(tickers_to_scan)
        st.session_state.selected_instrument_index = None # Reset detail view on new scan

    st.sidebar.markdown("---")
    st.sidebar.caption(f"Strategy uses {TF_CONDITIONS} for conditions and {TF_ENTRY} for entries.")

    # Main Area Tabs
    tab_dashboard, tab_details = st.tabs(["🔎 Scan Results", "📈 Detailed Charts"])

    with tab_dashboard:
        st.header("Scan Results Dashboard")
        if not st.session_state.scan_results:
            st.info("Click 'Run Scan' in the sidebar to start.")
        else:
            filtered_results = [r for r in st.session_state.scan_results if r['Setup'] not in ["Error", "None", "Calc Error"]]

            if not filtered_results:
                 st.success("Scan complete. No active Long/Short/Watch setups found matching the criteria.")
            else:
                st.success(f"Scan complete. Found {len(filtered_results)} potential setups or watchlist candidates.")
                # Display the table
                displayed_df = display_results_table(filtered_results) # Use filtered results for table

                # Selection for detailed view
                if displayed_df is not None and not displayed_df.empty:
                    st.markdown("---")
                    st.write("Select an instrument from the table above for detailed charts:")
                    # Use the names from the *displayed* dataframe for selection
                    selected_name = st.selectbox(
                        "Instrument Name:",
                        options=displayed_df["Name"].tolist(),
                        index=0,
                        key="detail_select_dashboard"
                    )
                    if st.button("Show Detailed Charts"):
                        # Find the selected row in the displayed (and potentially sorted) dataframe
                        selected_row_df = displayed_df[displayed_df["Name"] == selected_name].iloc[0]
                        # Get the original index stored during dataframe creation
                        original_idx = int(selected_row_df["_original_index"])
                        # Find the corresponding full result in the *original* scan_results list
                        full_result = next((res for i, res in enumerate(st.session_state.scan_results) if i == original_idx), None)

                        if full_result:
                            st.session_state.selected_instrument_index = original_idx # Store the index from the *original* list
                            st.info(f"Switch to the 'Detailed Charts' tab to view {selected_name}.")
                             # Use st.experimental_rerun() if automatic tab switching is desired/needed
                             # st.experimental_rerun()
                        else:
                            st.error("Could not find the selected instrument's full data.")


    with tab_details:
        st.header("Detailed Instrument Analysis")
        if st.session_state.selected_instrument_index is not None:
             # Check if index is valid
             if 0 <= st.session_state.selected_instrument_index < len(st.session_state.scan_results):
                 selected_result_data = st.session_state.scan_results[st.session_state.selected_instrument_index]
                 # Add a button to go back or clear selection
                 if st.button("← Back to Scan Results / Clear Selection"):
                     st.session_state.selected_instrument_index = None
                     st.experimental_rerun() # Rerun to clear the detail view
                 else:
                    display_detailed_charts(selected_result_data)
             else:
                 st.warning("Selected instrument index is out of bounds. Please re-select from the dashboard.")
                 st.session_state.selected_instrument_index = None # Reset invalid index
        else:
            st.info("Select an instrument from the 'Scan Results' tab and click 'Show Detailed Charts'.")


if __name__ == "__main__":
    main()
