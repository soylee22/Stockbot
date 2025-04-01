import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pandas_ta as ta # Essential for indicator calculation
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

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
# Keep the CSS for setup styling
st.markdown("""
<style>
    .setup-long { background-color: #d4edda; color: #155724; padding: 2px 5px; border-radius: 3px; font-weight: bold; }
    .setup-short { background-color: #f8d7da; color: #721c24; padding: 2px 5px; border-radius: 3px; font-weight: bold; }
    .setup-watch { background-color: #fff3cd; color: #856404; padding: 2px 5px; border-radius: 3px; font-weight: bold; }
    .dataframe-container { border-radius: 5px; overflow-x: auto; } /* Allow horizontal scroll */
    .dataframe tr:hover { background-color: rgba(0,0,0,0.05) !important; cursor: pointer; }
    .dataframe td[align="right"], .dataframe th[align="right"] { text-align: right !important; }
</style>
""", unsafe_allow_html=True)

# --- Helper Functions ---

@st.cache_data(ttl=1800) # Cache for 30 minutes
def fetch_strategy_data(ticker):
    try:
        ticker_obj = yf.Ticker(ticker)
        data_conditions = ticker_obj.history(period=PERIOD_CONDITIONS, interval=TF_CONDITIONS)
        data_entry = ticker_obj.history(period=PERIOD_ENTRY, interval=TF_ENTRY)
        min_len_cond = max(EMA_LONG, MACD_SLOW) + 5
        min_len_entry = max(EMA_LONG, MACD_SLOW) + 5
        if data_conditions.empty or len(data_conditions) < min_len_cond or \
           data_entry.empty or len(data_entry) < min_len_entry:
            return None, None
        return data_conditions, data_entry
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {str(e)}")
        return None, None


def calculate_strategy_indicators(data):
    if data is None or data.empty: return None, None
    try:
        # Create a copy of the data to avoid SettingWithCopyWarning
        data_copy = data.copy()
        
        data_copy.ta.ema(length=EMA_SHORT, append=True, col_names=(f"EMA_{EMA_SHORT}",))
        data_copy.ta.ema(length=EMA_LONG, append=True, col_names=(f"EMA_{EMA_LONG}",))
        data_copy.ta.rsi(length=RSI_WINDOW, append=True, col_names=(f"RSI_{RSI_WINDOW}",))
        data_copy.ta.macd(fast=MACD_FAST, slow=MACD_SLOW, signal=MACD_SIGNAL, append=True,
                    col_names=(f"MACD_{MACD_FAST}_{MACD_SLOW}_{MACD_SIGNAL}",
                               f"MACDh_{MACD_FAST}_{MACD_SLOW}_{MACD_SIGNAL}",
                               f"MACDs_{MACD_FAST}_{MACD_SLOW}_{MACD_SIGNAL}"))
        indicators = {}
        indicators['Close'] = data_copy['Close'].iloc[-1]
        indicators[f'EMA_{EMA_SHORT}'] = data_copy[f'EMA_{EMA_SHORT}'].iloc[-1]
        indicators[f'EMA_{EMA_LONG}'] = data_copy[f'EMA_{EMA_LONG}'].iloc[-1]
        indicators[f'RSI_{RSI_WINDOW}'] = data_copy[f'RSI_{RSI_WINDOW}'].iloc[-1]
        indicators[f'MACD_Line'] = data_copy[f"MACD_{MACD_FAST}_{MACD_SLOW}_{MACD_SIGNAL}"].iloc[-1]
        indicators[f'MACD_Signal'] = data_copy[f"MACDs_{MACD_FAST}_{MACD_SLOW}_{MACD_SIGNAL}"].iloc[-1]
        indicators[f'MACD_Hist'] = data_copy[f"MACDh_{MACD_FAST}_{MACD_SLOW}_{MACD_SIGNAL}"].iloc[-1]
        # --- Derived Boolean States ---
        indicators['RSI_Bullish'] = indicators[f'RSI_{RSI_WINDOW}'] > RSI_MID
        indicators['RSI_Bearish'] = indicators[f'RSI_{RSI_WINDOW}'] < RSI_MID
        indicators['MACD_Bullish'] = indicators['MACD_Line'] > indicators['MACD_Signal']
        indicators['MACD_Bearish'] = indicators['MACD_Line'] < indicators['MACD_Signal']
        indicators['Price_Above_EMA_Short'] = indicators['Close'] > indicators[f'EMA_{EMA_SHORT}']
        indicators['Price_Above_EMA_Long'] = indicators['Close'] > indicators[f'EMA_{EMA_LONG}']
        indicators['Price_Below_EMA_Short'] = indicators['Close'] < indicators[f'EMA_{EMA_SHORT}']
        indicators['Price_Below_EMA_Long'] = indicators['Close'] < indicators[f'EMA_{EMA_LONG}']
        # Combined daily price check helper
        indicators['Daily_Price_Structure_Long'] = indicators['Price_Above_EMA_Short'] and indicators['Price_Above_EMA_Long']
        indicators['Daily_Price_Structure_Short'] = indicators['Price_Below_EMA_Short'] and indicators['Price_Below_EMA_Long']
        return indicators, data_copy
    except Exception as e:
        st.error(f"Error calculating indicators: {str(e)}")
        return None, None


def check_strategy_setup(conditions_indicators, entry_indicators):
    if not conditions_indicators or not entry_indicators: return "Error", 0, []
    setup_long_status = "None"; positive_score = 0; rules_met_long_details = []
    setup_short_status = "None"; negative_score_magnitude = 0; rules_met_short_details = []
    # LONG Checks
    cond_r_ok_l = conditions_indicators.get('RSI_Bullish', False)
    cond_m_ok_l = conditions_indicators.get('MACD_Bullish', False)
    cond_p_ok_l = conditions_indicators.get('Price_Above_EMA_Long', False)
    conditions_long_met = cond_r_ok_l and cond_m_ok_l and cond_p_ok_l
    if conditions_long_met:
        positive_score = 3; rules_met_long_details = ["W:RSI>50", "W:MACD Bull", "W:Prc>EMA_L"]
        entry_rules_met_count_l = 0
        if entry_indicators.get('RSI_Bullish', False): entry_rules_met_count_l += 1; rules_met_long_details.append("D:RSI>50")
        if entry_indicators.get('MACD_Bullish', False): entry_rules_met_count_l += 1; rules_met_long_details.append("D:MACD Bull")
        if entry_indicators.get('Daily_Price_Structure_Long', False): entry_rules_met_count_l += 1; rules_met_long_details.append("D:Prc>EMAs")
        if entry_rules_met_count_l >= MIN_ENTRY_RULES_MET: setup_long_status = "Potential Long"; positive_score += entry_rules_met_count_l
        else: setup_long_status = "Watch Long"
    # SHORT Checks
    cond_r_ok_s = conditions_indicators.get('RSI_Bearish', False)
    cond_m_ok_s = conditions_indicators.get('MACD_Bearish', False)
    cond_p_ok_s = conditions_indicators.get('Price_Below_EMA_Long', False)
    conditions_short_met = cond_r_ok_s and cond_m_ok_s and cond_p_ok_s
    if conditions_short_met:
        negative_score_magnitude = 3; rules_met_short_details = ["W:RSI<50", "W:MACD Bear", "W:Prc<EMA_L"]
        entry_rules_met_count_s = 0
        if entry_indicators.get('RSI_Bearish', False): entry_rules_met_count_s += 1; rules_met_short_details.append("D:RSI<50")
        if entry_indicators.get('MACD_Bearish', False): entry_rules_met_count_s += 1; rules_met_short_details.append("D:MACD Bear")
        if entry_indicators.get('Daily_Price_Structure_Short', False): entry_rules_met_count_s += 1; rules_met_short_details.append("D:Prc<EMAs")
        if entry_rules_met_count_s >= MIN_ENTRY_RULES_MET: setup_short_status = "Potential Short"; negative_score_magnitude += entry_rules_met_count_s
        else: setup_short_status = "Watch Short"
    # Determine Final
    final_setup = "None"; final_score = 0; final_rules = []
    if setup_long_status == "Potential Long" and setup_short_status != "Potential Short": final_setup = setup_long_status; final_score = positive_score; final_rules = rules_met_long_details
    elif setup_short_status == "Potential Short" and setup_long_status != "Potential Long": final_setup = setup_short_status; final_score = -negative_score_magnitude; final_rules = rules_met_short_details
    elif setup_long_status == "Watch Long" and setup_short_status == "None": final_setup = setup_long_status; final_score = positive_score; final_rules = rules_met_long_details
    elif setup_short_status == "Watch Short" and setup_long_status == "None": final_setup = setup_short_status; final_score = -negative_score_magnitude; final_rules = rules_met_short_details
    elif setup_long_status == "Potential Long" and setup_short_status == "Potential Short": final_setup = "Conflicting"; final_score = 0; final_rules = ["Conflicting"]
    else: final_setup = "None"; final_score = 0; final_rules = []
    return final_setup, final_score, final_rules


def scan_tickers(tickers_dict, max_tickers=25):
    """Scan tickers with a maximum limit for performance"""
    # Limit the number of tickers to scan
    if len(tickers_dict) > max_tickers:
        st.warning(f"Limiting scan to {max_tickers} tickers for performance. Use specific categories or tickers for more focused results.")
        limited_tickers = dict(list(tickers_dict.items())[:max_tickers])
    else:
        limited_tickers = tickers_dict
    
    results = []
    total_tickers = len(limited_tickers)
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Update progress less frequently to reduce UI overhead
    update_frequency = max(1, min(5, total_tickers // 10))
    
    try:
        for i, (ticker, name) in enumerate(limited_tickers.items()):
            # Only update UI at specific intervals
            if i % update_frequency == 0 or i == total_tickers - 1:
                status_text.text(f"Scanning {i+1}/{total_tickers}: {name} ({ticker})...")
                progress_bar.progress((i + 1) / total_tickers)
            
            try:
                data_conditions, data_entry = fetch_strategy_data(ticker)
                if data_conditions is None or data_entry is None:
                    results.append({"ticker": ticker, "name": name, "Setup": "Data Error", "Score": 0, "Rules Met": [], "error": True})
                    continue
                    
                conditions_indicators, data_conditions_with_indicators = calculate_strategy_indicators(data_conditions)
                entry_indicators, data_entry_with_indicators = calculate_strategy_indicators(data_entry)
                if conditions_indicators is None or entry_indicators is None:
                    results.append({"ticker": ticker, "name": name, "Setup": "Calc Error", "Score": 0, "Rules Met": [], "error": True})
                    continue
                    
                setup_type, setup_score, rules_met = check_strategy_setup(conditions_indicators, entry_indicators)
                results.append({
                    "ticker": ticker, "name": name, "Setup": setup_type, "Score": setup_score,
                    "Rules Met": ", ".join(rules_met), "error": False,
                    "_data_conditions": data_conditions_with_indicators, "_data_entry": data_entry_with_indicators,
                    "_indicators_conditions": conditions_indicators, "_indicators_entry": entry_indicators,
                })
                
                # Small delay to prevent API rate limits and reduce resource usage
                time.sleep(0.1)
                
            except Exception as e:
                results.append({"ticker": ticker, "name": name, "Setup": "Error", "Score": 0, "Rules Met": [f"Error: {str(e)}"], "error": True})
    
    except Exception as e:
        st.error(f"Error during scanning: {str(e)}")
    finally:
        status_text.text(f"Scan Complete: {len(results)} tickers analyzed.")
        
    return results


def display_results_table(results_list):
    """Displays the filtered scan results with basic info, using HTML for Setup styling."""
    if not results_list:
        st.warning("No results to display.")
        return None

    # Filter out errors and 'None' setups
    filtered_results = [r for r in results_list if r['Setup'] not in ["Error", "None", "Calc Error", "Data Error", "Conflicting"]]
    
    if not filtered_results:
        st.info("No potential Long/Short/Watch setups found matching the criteria.")
        return None

    # Use Streamlit's native dataframe for better performance
    df_data = []
    
    for i, r in enumerate(filtered_results):
        # Define setup class based on the setup type
        setup_class = ""
        if "Long" in r['Setup']: setup_class = "setup-long"
        elif "Short" in r['Setup']: setup_class = "setup-short"
        if "Watch" in r['Setup']: setup_class += " setup-watch" # Combine if needed

        df_data.append({
            "Name": r["name"],
            "Setup": r["Setup"], 
            "Score": r["Score"],
            "_original_index": results_list.index(r) # Link back to the full results list
        })

    df_display = pd.DataFrame(df_data)
    
    # Default sort: High positive scores first (best Longs), then low negative scores (best Shorts)
    df_display = df_display.sort_values(by="Score", ascending=False).reset_index(drop=True)

    # Use Streamlit's native dataframe display
    st.dataframe(
        df_display.drop(columns=['_original_index']),
        use_container_width=True,
        column_config={
            "Name": st.column_config.TextColumn(
                "Name",
                width="medium"
            ),
            "Setup": st.column_config.TextColumn(
                "Setup",
                width="medium"
            ),
            "Score": st.column_config.NumberColumn(
                "Score",
                format="%d",
                width="small"
            )
        },
        hide_index=True
    )

    return df_display


def display_detailed_charts(result):
    st.header(f"Detailed Analysis: {result['name']} ({result['ticker']})")
    setup_class = ""
    if "Long" in result['Setup']: setup_class = "setup-long"
    elif "Short" in result['Setup']: setup_class = "setup-short"
    if "Watch" in result['Setup']: setup_class += " setup-watch"
    st.subheader(f"Detected Setup: <span class='{setup_class.strip()}'>{result['Setup']}</span> (Score: {result['Score']})", unsafe_allow_html=True)
    st.caption(f"Rules Met: {result['Rules Met']}")
    st.markdown("---")
    
    data_conditions = result.get('_data_conditions')
    data_entry = result.get('_data_entry')
    
    if data_conditions is None or data_entry is None:
        st.error("Chart data not available.")
        return
    
    # Limit data points for better performance
    data_conditions = data_conditions.iloc[-52:] if len(data_conditions) > 52 else data_conditions
    data_entry = data_entry.iloc[-120:] if len(data_entry) > 120 else data_entry
    
    # Weekly Chart with spinner for feedback
    with st.spinner("Generating weekly chart..."):
        fig_w = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03, 
                            row_heights=[0.6, 0.2, 0.2], 
                            subplot_titles=(f"{TF_CONDITIONS} Price & EMAs", f"{TF_CONDITIONS} RSI ({RSI_WINDOW})", f"{TF_CONDITIONS} MACD"))
        
        fig_w.add_trace(go.Candlestick(x=data_conditions.index, 
                                      open=data_conditions['Open'], 
                                      high=data_conditions['High'], 
                                      low=data_conditions['Low'], 
                                      close=data_conditions['Close'], 
                                      name="Price"), 
                       row=1, col=1)
        
        fig_w.add_trace(go.Scatter(x=data_conditions.index, 
                                  y=data_conditions.get(f'EMA_{EMA_SHORT}'), 
                                  name=f"EMA {EMA_SHORT}", 
                                  line=dict(color='cyan', width=1)), 
                       row=1, col=1)
        
        fig_w.add_trace(go.Scatter(x=data_conditions.index, 
                                  y=data_conditions.get(f'EMA_{EMA_LONG}'), 
                                  name=f"EMA {EMA_LONG}", 
                                  line=dict(color='magenta', width=1)), 
                       row=1, col=1)
        
        fig_w.add_trace(go.Scatter(x=data_conditions.index, 
                                  y=data_conditions.get(f'RSI_{RSI_WINDOW}'), 
                                  name="RSI", 
                                  line=dict(color='orange')), 
                       row=2, col=1)
        
        fig_w.add_hline(y=RSI_MID, line_dash="dash", line_color="grey", row=2, col=1)
        
        macd_hist = data_conditions.get(f"MACDh_{MACD_FAST}_{MACD_SLOW}_{MACD_SIGNAL}", [])
        colors_w = ['green' if val >= 0 else 'red' for val in macd_hist]
        
        fig_w.add_trace(go.Bar(x=data_conditions.index, 
                              y=macd_hist, 
                              name='Hist', 
                              marker_color=colors_w), 
                       row=3, col=1)
        
        fig_w.add_trace(go.Scatter(x=data_conditions.index, 
                                  y=data_conditions.get(f"MACD_{MACD_FAST}_{MACD_SLOW}_{MACD_SIGNAL}"), 
                                  name="MACD", 
                                  line=dict(color="blue")), 
                       row=3, col=1)
        
        fig_w.add_trace(go.Scatter(x=data_conditions.index, 
                                  y=data_conditions.get(f"MACDs_{MACD_FAST}_{MACD_SLOW}_{MACD_SIGNAL}"), 
                                  name="Signal", 
                                  line=dict(color="red")), 
                       row=3, col=1)
        
        fig_w.update_layout(
            title=f"Weekly ({TF_CONDITIONS}) Chart - Market Conditions", 
            height=600, 
            xaxis_rangeslider_visible=False,
            showlegend=False
        )
        
        fig_w.update_yaxes(range=[0, 100], row=2, col=1)
        
        st.plotly_chart(fig_w, use_container_width=True)
    
    st.markdown("---")
    
    # Daily Chart with spinner for feedback
    with st.spinner("Generating daily chart..."):
        fig_d = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03, 
                            row_heights=[0.6, 0.2, 0.2], 
                            subplot_titles=(f"{TF_ENTRY} Price & EMAs", f"{TF_ENTRY} RSI ({RSI_WINDOW})", f"{TF_ENTRY} MACD"))
        
        fig_d.add_trace(go.Candlestick(x=data_entry.index, 
                                      open=data_entry['Open'], 
                                      high=data_entry['High'], 
                                      low=data_entry['Low'], 
                                      close=data_entry['Close'], 
                                      name="Price"), 
                       row=1, col=1)
        
        fig_d.add_trace(go.Scatter(x=data_entry.index, 
                                  y=data_entry.get(f'EMA_{EMA_SHORT}'), 
                                  name=f"EMA {EMA_SHORT}", 
                                  line=dict(color='cyan', width=1)), 
                       row=1, col=1)
        
        fig_d.add_trace(go.Scatter(x=data_entry.index, 
                                  y=data_entry.get(f'EMA_{EMA_LONG}'), 
                                  name=f"EMA {EMA_LONG}", 
                                  line=dict(color='magenta', width=1)), 
                       row=1, col=1)
        
        fig_d.add_trace(go.Scatter(x=data_entry.index, 
                                  y=data_entry.get(f'RSI_{RSI_WINDOW}'), 
                                  name="RSI", 
                                  line=dict(color='orange')), 
                       row=2, col=1)
        
        fig_d.add_hline(y=RSI_MID, line_dash="dash", line_color="grey", row=2, col=1)
        
        macd_hist_d = data_entry.get(f"MACDh_{MACD_FAST}_{MACD_SLOW}_{MACD_SIGNAL}", [])
        colors_d = ['green' if val >= 0 else 'red' for val in macd_hist_d]
        
        fig_d.add_trace(go.Bar(x=data_entry.index, 
                              y=macd_hist_d, 
                              name='Hist', 
                              marker_color=colors_d), 
                       row=3, col=1)
        
        fig_d.add_trace(go.Scatter(x=data_entry.index, 
                                  y=data_entry.get(f"MACD_{MACD_FAST}_{MACD_SLOW}_{MACD_SIGNAL}"), 
                                  name="MACD", 
                                  line=dict(color="blue")), 
                       row=3, col=1)
        
        fig_d.add_trace(go.Scatter(x=data_entry.index, 
                                  y=data_entry.get(f"MACDs_{MACD_FAST}_{MACD_SLOW}_{MACD_SIGNAL}"), 
                                  name="Signal", 
                                  line=dict(color="red")), 
                       row=3, col=1)
        
        fig_d.update_layout(
            title=f"Daily ({TF_ENTRY}) Chart - Entry Criteria", 
            height=600, 
            xaxis_rangeslider_visible=False,
            showlegend=False
        )
        
        fig_d.update_yaxes(range=[0, 100], row=2, col=1)
        
        st.plotly_chart(fig_d, use_container_width=True)


# --- Main App Flow ---
def main():
    st.title("🎯 I Chart Daily Strategy Setup Scanner")
    with st.expander("📖 Strategy Criteria Explained"):
        st.markdown(f"""
        This scanner identifies potential trading setups based on the "I Chart Daily" multi-timeframe momentum strategy.

        **Core Idea:** Use the Weekly timeframe (`{TF_CONDITIONS}`) to establish the dominant market trend (bias) and the Daily timeframe (`{TF_ENTRY}`) to find confirmation/entry triggers aligned with that trend.

        **LONG Setup Criteria:**
        1.  **Weekly Conditions (ALL must be met):** `W:RSI>50`, `W:MACD Bull`, `W:P>E21` (Price > EMA {EMA_LONG})
        2.  **Daily Entry Confirmation (at least {MIN_ENTRY_RULES_MET} must be met):** `D:RSI>50`, `D:MACD Bull`, `D:P>Es` (Price > EMA {EMA_SHORT} & {EMA_LONG})

        **SHORT Setup Criteria:**
        1.  **Weekly Conditions (ALL must be met):** `W:RSI<50`, `W:MACD Bear`, `W:P<E21` (Price < EMA {EMA_LONG})
        2.  **Daily Entry Confirmation (at least {MIN_ENTRY_RULES_MET} must be met):** `D:RSI<50`, `D:MACD Bear`, `D:P<Es` (Price < EMA {EMA_SHORT} & {EMA_LONG})

        **Setup Types & Scoring:**
        *   `Potential Long/Short`: All Weekly conditions met + ≥{MIN_ENTRY_RULES_MET} Daily rules met. (Score +4 to +6 / -4 to -6).
        *   `Watch Long/Short`: All Weekly conditions met, but <{MIN_ENTRY_RULES_MET} Daily rules met. (Score +3 / -3).
        *   `None/Error`: Criteria not met or data issues (Score 0).
        """)
        
    # Initialize session state variables
    if 'scan_results' not in st.session_state:
        st.session_state.scan_results = []
    if 'selected_instrument_index' not in st.session_state:
        st.session_state.selected_instrument_index = None

    # Sidebar controls
    st.sidebar.title("Scan Settings")
    scan_option = st.sidebar.radio(
        "Select Tickers To Scan:",
        ("All Categories", "Select Categories", "Specific Tickers"),
        index=1,
        key="scan_option"
    )
    
    tickers_to_scan = {}
    max_tickers = 25  # Default maximum for performance
    
    if scan_option == "Select Categories":
        available_categories = list(TICKER_CATEGORIES.keys())
        selected_categories = st.sidebar.multiselect(
            "Categories:",
            available_categories,
            default=["INDICES", "COMMODITIES", "FOREX"] if available_categories else [],
            key="sel_cats"
        )
        
        if selected_categories:
            for cat in selected_categories:
                tickers_to_scan.update(TICKER_CATEGORIES.get(cat, {}))
        else:
            st.sidebar.warning("Please select at least one category.")
    
    elif scan_option == "Specific Tickers":
        ticker_input = st.sidebar.text_area("Enter tickers (comma-separated):", key="spec_ticks")
        
        if ticker_input:
            specific_tickers_list = [t.strip().upper() for t in ticker_input.split(',')]
            for ticker in specific_tickers_list:
                found = False
                for cat_tickers in TICKER_CATEGORIES.values():
                    if ticker in cat_tickers:
                        tickers_to_scan[ticker] = cat_tickers[ticker]
                        found = True
                        break
                if not found:
                    tickers_to_scan[ticker] = ticker
        else:
            st.sidebar.warning("Please enter at least one ticker.")
    
    else:  # All Categories - but with warnings and limits
        st.sidebar.warning("Scanning all tickers may take a long time. Consider selecting specific categories instead.")
        sample_categories = list(TICKER_CATEGORIES.keys())[:2]  # Just take first 2 categories for performance
        for cat in sample_categories:
            tickers_to_scan.update(TICKER_CATEGORIES.get(cat, {}))
        st.sidebar.info(f"For performance, limiting to {len(tickers_to_scan)} tickers from {', '.join(sample_categories)} categories.")
    
    # Custom max ticker input
    custom_max = st.sidebar.number_input("Maximum tickers to scan:", min_value=1, max_value=100, value=max_tickers)
    max_tickers = custom_max if custom_max else max_tickers
    
    # Scan button
    if st.sidebar.button("▶️ Run Scan", use_container_width=True, type="primary", disabled=(len(tickers_to_scan) == 0)):
        with st.spinner(f"Scanning up to {max_tickers} tickers..."):
            st.session_state.scan_results = scan_tickers(tickers_to_scan, max_tickers)
            st.session_state.selected_instrument_index = None
    
    st.sidebar.markdown("---")
    st.sidebar.caption(f"Strategy uses {TF_CONDITIONS} conditions / {TF_ENTRY} entries.")

    # Tabs
    tab_dashboard, tab_details = st.tabs(["🔎 Scan Results", "📈 Detailed Charts"])
    
    with tab_dashboard:
        st.header("Scan Results Dashboard")
        
        if not st.session_state.scan_results:
            st.info("Click 'Run Scan' in the sidebar to start.")
        else:
            displayable_results = [r for r in st.session_state.scan_results 
                                  if r['Setup'] not in ["Error", "None", "Calc Error", "Data Error", "Conflicting"]]
            
            if not displayable_results:
                st.success("Scan complete. No active Long/Short/Watch setups found.")
            else:
                st.success(f"Scan complete. Found {len(displayable_results)} potential setups or watchlist candidates.")
                
                # Use the optimized display function
                displayed_df = display_results_table(st.session_state.scan_results)
                
                if displayed_df is not None and not displayed_df.empty:
                    st.markdown("---")
                    st.write("Select an instrument from the table above for detailed charts:")
                    
                    # Use selectbox for instrument selection
                    selected_name = st.selectbox(
                        "Instrument Name:",
                        options=displayed_df["Name"].tolist(),
                        index=0,
                        key="detail_select_dashboard"
                    )
                    
                    if st.button("Show Detailed Charts"):
                        try:
                            selected_row_df = displayed_df[displayed_df["Name"] == selected_name].iloc[0]
                            original_idx = int(selected_row_df["_original_index"])
                            
                            if 0 <= original_idx < len(st.session_state.scan_results):
                                st.session_state.selected_instrument_index = original_idx
                                st.info("Switch to 'Detailed Charts' tab to view analysis.")
                            else:
                                st.error("Error linking selection.")
                        except Exception as e:
                            st.error(f"Error selecting instrument: {str(e)}")
    
    with tab_details:
        st.header("Detailed Instrument Analysis")
        
        if st.session_state.selected_instrument_index is not None:
            try:
                if 0 <= st.session_state.selected_instrument_index < len(st.session_state.scan_results):
                    selected_result_data = st.session_state.scan_results[st.session_state.selected_instrument_index]
                    
                    if st.button("← Back / Clear Selection"):
                        st.session_state.selected_instrument_index = None
                        st.rerun()
                    else:
                        display_detailed_charts(selected_result_data)
                else:
                    st.warning("Invalid selection index.")
                    st.session_state.selected_instrument_index = None
            except Exception as e:
                st.error(f"Error displaying charts: {str(e)}")
                st.session_state.selected_instrument_index = None
        else:
            st.info("Select an instrument from 'Scan Results' and click 'Show Detailed Charts'.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
