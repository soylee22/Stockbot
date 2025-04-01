import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pandas_ta as ta # Essential for indicator calculation
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
st.markdown("""
<style>
    /* Enhanced setup styling with stronger colors */
    .setup-long { background-color: #28a745; color: white; padding: 4px 8px; border-radius: 4px; font-weight: bold; }
    .setup-short { background-color: #dc3545; color: white; padding: 4px 8px; border-radius: 4px; font-weight: bold; }
    .setup-watch { background-color: #ffc107; color: black; padding: 4px 8px; border-radius: 4px; font-weight: bold; }
    .setup-none { background-color: #6c757d; color: white; padding: 4px 8px; border-radius: 4px; font-weight: bold; }
    
    /* Cell colors for metrics */
    .bullish { background-color: rgba(40, 167, 69, 0.2); }
    .bearish { background-color: rgba(220, 53, 69, 0.2); }
    .neutral { background-color: rgba(108, 117, 125, 0.1); }
    
    .stDataFrame [data-testid="stDataFrameResizable"] {
        max-height: 600px;
        overflow-y: auto;
    }
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
        
        # Add RSI and MACD actual values for display
        indicators['RSI_Value'] = round(indicators[f'RSI_{RSI_WINDOW}'], 1)
        indicators['MACD_Value'] = round(indicators['MACD_Line'], 3)
        
        return indicators, data_copy
    except Exception as e:
        st.error(f"Error calculating indicators: {str(e)}")
        return None, None


def check_strategy_setup(conditions_indicators, entry_indicators):
    if not conditions_indicators or not entry_indicators: return "Error", 0, [], {}
    setup_long_status = "None"; positive_score = 0; rules_met_long_details = []
    setup_short_status = "None"; negative_score_magnitude = 0; rules_met_short_details = []
    
    # Collect all metrics for display
    all_metrics = {}
    
    # LONG Checks - Weekly
    cond_r_ok_l = conditions_indicators.get('RSI_Bullish', False)
    cond_m_ok_l = conditions_indicators.get('MACD_Bullish', False)
    cond_p_ok_l = conditions_indicators.get('Price_Above_EMA_Long', False)
    conditions_long_met = cond_r_ok_l and cond_m_ok_l and cond_p_ok_l
    
    # Store metrics from weekly timeframe
    all_metrics['W_RSI'] = {
        'value': conditions_indicators.get('RSI_Value', 0),
        'signal': 'bullish' if cond_r_ok_l else 'bearish',
        'desc': 'RSI > 50' if cond_r_ok_l else 'RSI < 50'
    }
    
    all_metrics['W_MACD'] = {
        'value': conditions_indicators.get('MACD_Value', 0),
        'signal': 'bullish' if cond_m_ok_l else 'bearish',
        'desc': 'MACD Bull' if cond_m_ok_l else 'MACD Bear'
    }
    
    all_metrics['W_Price_EMA'] = {
        'value': f"{conditions_indicators.get('Close', 0):.2f} vs {conditions_indicators.get(f'EMA_{EMA_LONG}', 0):.2f}",
        'signal': 'bullish' if cond_p_ok_l else 'bearish',
        'desc': f'Price > EMA{EMA_LONG}' if cond_p_ok_l else f'Price < EMA{EMA_LONG}'
    }
    
    if conditions_long_met:
        positive_score = 3; rules_met_long_details = ["W:RSI>50", "W:MACD Bull", "W:Prc>EMA_L"]
        entry_rules_met_count_l = 0
        
        # Daily Metrics for Long
        if entry_indicators.get('RSI_Bullish', False): 
            entry_rules_met_count_l += 1
            rules_met_long_details.append("D:RSI>50")
        
        if entry_indicators.get('MACD_Bullish', False): 
            entry_rules_met_count_l += 1
            rules_met_long_details.append("D:MACD Bull")
        
        if entry_indicators.get('Daily_Price_Structure_Long', False): 
            entry_rules_met_count_l += 1
            rules_met_long_details.append("D:Prc>EMAs")
        
        if entry_rules_met_count_l >= MIN_ENTRY_RULES_MET: 
            setup_long_status = "Potential Long"
            positive_score += entry_rules_met_count_l
        else: 
            setup_long_status = "Watch Long"
    
    # SHORT Checks - Weekly
    cond_r_ok_s = conditions_indicators.get('RSI_Bearish', False)
    cond_m_ok_s = conditions_indicators.get('MACD_Bearish', False)
    cond_p_ok_s = conditions_indicators.get('Price_Below_EMA_Long', False)
    conditions_short_met = cond_r_ok_s and cond_m_ok_s and cond_p_ok_s
    
    if conditions_short_met:
        negative_score_magnitude = 3; rules_met_short_details = ["W:RSI<50", "W:MACD Bear", "W:Prc<EMA_L"]
        entry_rules_met_count_s = 0
        
        # Daily Metrics for Short
        if entry_indicators.get('RSI_Bearish', False): 
            entry_rules_met_count_s += 1
            rules_met_short_details.append("D:RSI<50")
        
        if entry_indicators.get('MACD_Bearish', False): 
            entry_rules_met_count_s += 1
            rules_met_short_details.append("D:MACD Bear")
        
        if entry_indicators.get('Daily_Price_Structure_Short', False): 
            entry_rules_met_count_s += 1
            rules_met_short_details.append("D:Prc<EMAs")
        
        if entry_rules_met_count_s >= MIN_ENTRY_RULES_MET: 
            setup_short_status = "Potential Short"
            negative_score_magnitude += entry_rules_met_count_s
        else: 
            setup_short_status = "Watch Short"
    
    # Store daily metrics
    all_metrics['D_RSI'] = {
        'value': entry_indicators.get('RSI_Value', 0),
        'signal': 'bullish' if entry_indicators.get('RSI_Bullish', False) else 'bearish',
        'desc': 'RSI > 50' if entry_indicators.get('RSI_Bullish', False) else 'RSI < 50'
    }
    
    all_metrics['D_MACD'] = {
        'value': entry_indicators.get('MACD_Value', 0),
        'signal': 'bullish' if entry_indicators.get('MACD_Bullish', False) else 'bearish',
        'desc': 'MACD Bull' if entry_indicators.get('MACD_Bullish', False) else 'MACD Bear'
    }
    
    # Daily Price Structure
    price_structure = 'neutral'
    price_desc = 'Mixed'
    
    if entry_indicators.get('Daily_Price_Structure_Long', False):
        price_structure = 'bullish'
        price_desc = f'Price > EMA{EMA_SHORT} & EMA{EMA_LONG}'
    elif entry_indicators.get('Daily_Price_Structure_Short', False):
        price_structure = 'bearish'
        price_desc = f'Price < EMA{EMA_SHORT} & EMA{EMA_LONG}'
    
    all_metrics['D_Price_EMA'] = {
        'value': f"{entry_indicators.get('Close', 0):.2f} vs {entry_indicators.get(f'EMA_{EMA_SHORT}', 0):.2f}/{entry_indicators.get(f'EMA_{EMA_LONG}', 0):.2f}",
        'signal': price_structure,
        'desc': price_desc
    }
    
    # Determine Final
    final_setup = "None"; final_score = 0; final_rules = []
    if setup_long_status == "Potential Long" and setup_short_status != "Potential Short": 
        final_setup = setup_long_status
        final_score = positive_score
        final_rules = rules_met_long_details
    elif setup_short_status == "Potential Short" and setup_long_status != "Potential Long": 
        final_setup = setup_short_status
        final_score = -negative_score_magnitude
        final_rules = rules_met_short_details
    elif setup_long_status == "Watch Long" and setup_short_status == "None": 
        final_setup = setup_long_status
        final_score = positive_score
        final_rules = rules_met_long_details
    elif setup_short_status == "Watch Short" and setup_long_status == "None": 
        final_setup = setup_short_status
        final_score = -negative_score_magnitude
        final_rules = rules_met_short_details
    elif setup_long_status == "Potential Long" and setup_short_status == "Potential Short": 
        final_setup = "Conflicting"
        final_score = 0
        final_rules = ["Conflicting"]
    else: 
        final_setup = "None"
        final_score = 0
        final_rules = []
    
    return final_setup, final_score, final_rules, all_metrics


def scan_tickers(tickers_dict, max_tickers=40):
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
                    results.append({
                        "ticker": ticker, 
                        "name": name, 
                        "Setup": "Data Error", 
                        "Score": 0, 
                        "Rules Met": [], 
                        "error": True,
                        "metrics": {}
                    })
                    continue
                    
                conditions_indicators, data_conditions_with_indicators = calculate_strategy_indicators(data_conditions)
                entry_indicators, data_entry_with_indicators = calculate_strategy_indicators(data_entry)
                if conditions_indicators is None or entry_indicators is None:
                    results.append({
                        "ticker": ticker, 
                        "name": name, 
                        "Setup": "Calc Error", 
                        "Score": 0, 
                        "Rules Met": [], 
                        "error": True,
                        "metrics": {}
                    })
                    continue
                    
                setup_type, setup_score, rules_met, all_metrics = check_strategy_setup(conditions_indicators, entry_indicators)
                
                # Calculate price and date for display
                current_price = entry_indicators.get('Close', 0) if entry_indicators else 0
                last_date = data_entry.index[-1].strftime('%Y-%m-%d') if data_entry is not None and not data_entry.empty else "N/A"
                
                results.append({
                    "ticker": ticker, 
                    "name": name, 
                    "Setup": setup_type, 
                    "Score": setup_score,
                    "Price": round(current_price, 2),
                    "Last Date": last_date,
                    "Rules Met": ", ".join(rules_met), 
                    "error": False,
                    "metrics": all_metrics
                })
                
                # Small delay to prevent API rate limits and reduce resource usage
                time.sleep(0.1)
                
            except Exception as e:
                results.append({
                    "ticker": ticker, 
                    "name": name, 
                    "Setup": "Error", 
                    "Score": 0, 
                    "Rules Met": [f"Error: {str(e)}"], 
                    "error": True,
                    "metrics": {}
                })
    
    except Exception as e:
        st.error(f"Error during scanning: {str(e)}")
    finally:
        status_text.text(f"Scan Complete: {len(results)} tickers analyzed.")
        
    return results


def format_cell(value, signal_type):
    """Format a table cell with appropriate styling based on signal type"""
    if signal_type == 'bullish':
        return f'<span class="bullish">{value}</span>'
    elif signal_type == 'bearish':
        return f'<span class="bearish">{value}</span>'
    else:
        return f'<span class="neutral">{value}</span>'


def display_results_table(results_list):
    """Displays the scan results with metrics columns"""
    if not results_list:
        st.warning("No results to display.")
        return None

    # Include even if setup is "None" to show all metrics
    filtered_results = [r for r in results_list if not r['error']]
    
    if not filtered_results:
        st.info("No valid results found. Try scanning different tickers.")
        return None

    # Build dataframe with all metrics
    df_data = []
    
    for r in filtered_results:
        # Skip 'Error', 'Data Error', 'Calc Error' entries
        if r['Setup'] in ["Error", "Data Error", "Calc Error"]:
            continue
        
        # Define setup class based on the setup type
        setup_class = "setup-none"
        if "Long" in r['Setup']: setup_class = "setup-long"
        elif "Short" in r['Setup']: setup_class = "setup-short"
        elif "Watch" in r['Setup']: setup_class = "setup-watch"
        
        # Format setup with HTML for styling
        setup_html = f'<span class="{setup_class}">{r["Setup"]}</span>'
        
        # Get metrics from result
        metrics = r.get('metrics', {})
        
        # Create a row with all metrics
        row_data = {
            "Name": r["name"],
            "Ticker": r["ticker"],
            "Price": r["Price"],
            "Last Date": r["Last Date"],
            "Setup": setup_html,
            "Score": r["Score"],
            "W_RSI": format_cell(metrics.get('W_RSI', {}).get('value', 'N/A'), metrics.get('W_RSI', {}).get('signal', 'neutral')),
            "W_MACD": format_cell(metrics.get('W_MACD', {}).get('value', 'N/A'), metrics.get('W_MACD', {}).get('signal', 'neutral')),
            "W_Price_EMA": format_cell(metrics.get('W_Price_EMA', {}).get('value', 'N/A'), metrics.get('W_Price_EMA', {}).get('signal', 'neutral')),
            "D_RSI": format_cell(metrics.get('D_RSI', {}).get('value', 'N/A'), metrics.get('D_RSI', {}).get('signal', 'neutral')),
            "D_MACD": format_cell(metrics.get('D_MACD', {}).get('value', 'N/A'), metrics.get('D_MACD', {}).get('signal', 'neutral')),
            "D_Price_EMA": format_cell(metrics.get('D_Price_EMA', {}).get('value', 'N/A'), metrics.get('D_Price_EMA', {}).get('signal', 'neutral')),
        }
        
        df_data.append(row_data)

    if not df_data:
        st.info("No valid results to display after filtering.")
        return None

    # Create DataFrame
    df_display = pd.DataFrame(df_data)
    
    # Default sort: High positive scores first (best Longs), then low negative scores (best Shorts)
    df_display = df_display.sort_values(by="Score", ascending=False).reset_index(drop=True)
    
    # Add filter for setup types
    setup_filter = st.multiselect(
        "Filter by Setup Type:",
        ["Potential Long", "Watch Long", "Potential Short", "Watch Short", "None", "Conflicting"],
        default=["Potential Long", "Watch Long", "Potential Short", "Watch Short"]
    )
    
    if setup_filter:
        # Extract setup type from HTML for filtering
        setup_series = df_display["Setup"].str.extract(r'>([^<]+)<')
        filtered_df = df_display[setup_series[0].isin(setup_filter)]
    else:
        filtered_df = df_display
        
    if filtered_df.empty:
        st.info("No results match the selected filter criteria.")
        return None
        
    # Display with rich HTML formatting
    st.markdown('<div class="stDataFrame">', unsafe_allow_html=True)
    st.write(
        filtered_df.to_html(
            escape=False,  # Allow HTML in cells
            index=False,
            classes="dataframe",
            border=0
        ),
        unsafe_allow_html=True
    )
    st.markdown('</div>', unsafe_allow_html=True)

    return filtered_df


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
        
        **Columns Legend:**
        * **W_RSI**: Weekly RSI value (Bullish >50, Bearish <50)
        * **W_MACD**: Weekly MACD value (Bullish when MACD > Signal, Bearish when MACD < Signal)
        * **W_Price_EMA**: Weekly Price relative to EMA{EMA_LONG}
        * **D_RSI**: Daily RSI value (Bullish >50, Bearish <50)
        * **D_MACD**: Daily MACD value (Bullish when MACD > Signal, Bearish when MACD < Signal)
        * **D_Price_EMA**: Daily Price relative to both EMAs
        
        **Cell Colors:**
        * <span class="bullish">Green</span>: Bullish signal
        * <span class="bearish">Red</span>: Bearish signal
        * <span class="neutral">Gray</span>: Neutral signal
        """)
        
    # Initialize session state variables
    if 'scan_results' not in st.session_state:
        st.session_state.scan_results = []

    # Sidebar controls
    st.sidebar.title("Scan Settings")
    scan_option = st.sidebar.radio(
        "Select Tickers To Scan:",
        ("All Categories", "Select Categories", "Specific Tickers"),
        index=1,
        key="scan_option"
    )
    
    tickers_to_scan = {}
    max_tickers = 40  # Default maximum
    
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
    
    else:  # All Categories
        for cat in TICKER_CATEGORIES.keys():
            tickers_to_scan.update(TICKER_CATEGORIES.get(cat, {}))
    
    # Scan button
    if st.sidebar.button("▶️ Run Scan", use_container_width=True, type="primary", disabled=(len(tickers_to_scan) == 0)):
        with st.spinner(f"Scanning tickers (max {max_tickers})..."):
            st.session_state.scan_results = scan_tickers(tickers_to_scan, max_tickers)
    
    st.sidebar.markdown("---")
    st.sidebar.caption(f"Strategy uses {TF_CONDITIONS} conditions / {TF_ENTRY} entries.")

    # Main Results Display
    st.header("Scan Results Dashboard")
    
    if not st.session_state.scan_results:
        st.info("Click 'Run Scan' in the sidebar to start.")
    else:
        valid_results = [r for r in st.session_state.scan_results if not r.get('error', True)]
        
        if not valid_results:
            st.warning("Scan complete, but no valid results were found. Try different tickers.")
        else:
            active_setups = [r for r in valid_results if r['Setup'] not in ["None", "Conflicting"]]
            
            if active_setups:
                st.success(f"Scan complete. Found {len(active_setups)} potential setups out of {len(valid_results)} valid instruments.")
            else:
                st.info(f"Scan complete. No active setups found among {len(valid_results)} valid instruments.")
            
            # Display results table with all metrics
            display_results_table(st.session_state.scan_results)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
