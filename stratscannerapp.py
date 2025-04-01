# Add these additional imports alongside your existing imports
# Put these after the existing imports at the top of your file
import os
from datetime import datetime, timedelta
import json
import plotly.express as px
import plotly.graph_objects as go

# Add these functions for tracking historical scores

def get_history_directory():
    """Get or create the directory for storing history data"""
    # Create a directory in the current working directory
    history_dir = os.path.join(os.getcwd(), "score_history")
    if not os.path.exists(history_dir):
        os.makedirs(history_dir)
    return history_dir

def save_scan_results_history(results_list):
    """Save today's scan results to history for future comparison"""
    if not results_list:
        return False
        
    # Get valid results only
    valid_results = [r for r in results_list if not r.get('error', True)]
    if not valid_results:
        return False
    
    # Format for storage
    today = datetime.now().strftime("%Y-%m-%d")
    history_data = {
        "date": today,
        "tickers": {}
    }
    
    for result in valid_results:
        ticker = result['ticker']
        history_data["tickers"][ticker] = {
            "name": result['name'],
            "setup": result['Setup'],
            "score": result['Score'],
            "price": result.get('Price', 0)
        }
    
    # Save to file
    history_dir = get_history_directory()
    history_file = os.path.join(history_dir, f"scores_{today}.json")
    
    with open(history_file, 'w') as f:
        json.dump(history_data, f)
    
    return True

def load_historical_data(days_back=7):
    """Load historical scan data for specified number of days"""
    history_dir = get_history_directory()
    
    # Calculate date range
    end_date = datetime.now().date()
    date_list = [end_date - timedelta(days=i) for i in range(days_back)]
    date_list.reverse()  # Oldest to newest
    
    # Collect historical data
    history_data = {}
    
    for single_date in date_list:
        date_str = single_date.strftime("%Y-%m-%d")
        file_path = os.path.join(history_dir, f"scores_{date_str}.json")
        
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    day_data = json.load(f)
                history_data[date_str] = day_data
            except Exception as e:
                st.error(f"Error loading data for {date_str}: {str(e)}")
    
    return history_data

def prepare_history_dataframe(history_data):
    """Convert history data to DataFrame for display"""
    if not history_data:
        return None
    
    # Prepare data for DataFrame
    rows = []
    
    # Get all unique tickers across all dates
    all_tickers = set()
    for date_data in history_data.values():
        for ticker in date_data.get("tickers", {}).keys():
            all_tickers.add(ticker)
    
    # Get sorted dates
    dates = sorted(history_data.keys())
    
    # Create rows with ticker info across all dates
    for ticker in sorted(all_tickers):
        row = {"ticker": ticker}
        
        # Get name from most recent data available
        ticker_name = None
        for date in reversed(dates):
            if ticker in history_data[date].get("tickers", {}):
                ticker_name = history_data[date]["tickers"][ticker].get("name", ticker)
                break
        
        row["name"] = ticker_name or ticker
        
        # Add scores for each date
        for date in dates:
            date_key = f"score_{date}"
            setup_key = f"setup_{date}"
            price_key = f"price_{date}"
            
            if ticker in history_data[date].get("tickers", {}):
                ticker_data = history_data[date]["tickers"][ticker]
                row[date_key] = ticker_data.get("score", None)
                row[setup_key] = ticker_data.get("setup", "N/A")
                row[price_key] = ticker_data.get("price", 0)
            else:
                row[date_key] = None
                row[setup_key] = "N/A"
                row[price_key] = None
        
        rows.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(rows)
    return df

def display_history_dashboard():
    """Display dashboard showing score changes over time"""
    st.header("📊 Setup Score History Dashboard")
    
    # Controls for the dashboard
    col1, col2 = st.columns(2)
    
    with col1:
        days_back = st.slider("Days to look back:", min_value=2, max_value=30, value=7, 
                              help="Number of days of historical data to display")
    
    # Load historical data
    history_data = load_historical_data(days_back)
    
    if not history_data:
        st.warning("No historical data found. Run scans on multiple days to build history.")
        st.info("Each time you run a scan, the results will be saved for historical comparison.")
        return
    
    # Prepare data for display
    history_df = prepare_history_dataframe(history_data)
    
    if history_df is None or history_df.empty:
        st.warning("Could not prepare history data for display.")
        return
    
    # Get dates for column selection
    dates = sorted([col.replace("score_", "") for col in history_df.columns if col.startswith("score_")])
    
    # Allow filtering by setup type
    with col2:
        setup_filter = st.multiselect(
            "Filter by setup type:",
            ["Potential Long", "Watch Long", "Caution Long", 
             "Potential Short", "Watch Short", "Caution Short", "None"],
            default=["Potential Long", "Watch Long", "Potential Short", "Watch Short"],
            help="Show only tickers with these setup types on the most recent date"
        )
    
    if setup_filter and dates:
        latest_date = dates[-1]
        latest_setup_col = f"setup_{latest_date}"
        
        if latest_setup_col in history_df.columns:
            history_df = history_df[history_df[latest_setup_col].isin(setup_filter)]
    
    if history_df.empty:
        st.info("No data matches your filter criteria.")
        return
    
    # Display options
    view_mode = st.radio(
        "View mode:",
        ["Score Table", "Score Heatmap", "Price Changes", "Score Chart"],
        horizontal=True
    )
    
    # Select limited tickers to display in chart view
    if view_mode == "Score Chart":
        selected_tickers = st.multiselect(
            "Select tickers to chart (max 10):",
            list(zip(history_df['ticker'], history_df['name'])),
            format_func=lambda x: f"{x[0]} - {x[1]}",
            default=list(zip(history_df['ticker'], history_df['name']))[:min(5, len(history_df))]
        )
        selected_ticker_symbols = [t[0] for t in selected_tickers]
        history_df_filtered = history_df[history_df['ticker'].isin(selected_ticker_symbols)]
    else:
        history_df_filtered = history_df
    
    # Display based on selected view mode
    if view_mode == "Score Table":
        display_score_table(history_df_filtered, dates)
    elif view_mode == "Score Heatmap":
        display_score_heatmap(history_df_filtered, dates)
    elif view_mode == "Price Changes":
        display_price_changes(history_df_filtered, dates)
    elif view_mode == "Score Chart":
        display_score_chart(history_df_filtered, dates)

def display_score_table(df, dates):
    """Display the score history as a table with trend indicators"""
    # Create a copy for display
    display_df = df.copy()
    
    # Rename and select columns for display
    columns_to_display = ["ticker", "name"]
    
    for date in dates:
        # Add current day score
        score_col = f"score_{date}"
        setup_col = f"setup_{date}"
        display_col = date
        columns_to_display.append(display_col)
        
        # Format the scores with the setup type
        display_df[display_col] = display_df.apply(
            lambda row: f"{row[score_col]} ({row[setup_col]})" if pd.notnull(row[score_col]) else "N/A",
            axis=1
        )
    
    # Add trend column if we have more than one date
    if len(dates) > 1:
        display_df["Trend"] = display_df.apply(
            lambda row: calculate_trend(row, dates),
            axis=1
        )
        columns_to_display.append("Trend")
    
    # Display the table
    st.subheader("Score History Table")
    st.dataframe(display_df[columns_to_display], use_container_width=True)

def calculate_trend(row, dates):
    """Calculate and format trend indicator based on score changes"""
    scores = []
    for date in dates:
        score_col = f"score_{date}"
        if pd.notnull(row[score_col]):
            scores.append(row[score_col])
    
    if len(scores) < 2:
        return "—"
    
    # Calculate trend based on first and last scores
    first_score = scores[0]
    last_score = scores[-1]
    
    if abs(last_score - first_score) < 0.5:  # Almost no change
        return "→"
    elif last_score > first_score:
        if last_score > 0:  # Improving long setup
            return "↑↑"
        else:  # Less negative short setup
            return "↑"
    else:  # last_score < first_score
        if last_score < 0:  # Strengthening short setup
            return "↓↓"
        else:  # Weakening long setup
            return "↓"

def display_score_heatmap(df, dates):
    """Display scores as a heatmap for visual comparison"""
    # Create a dataframe just for the heatmap
    heatmap_data = []
    
    for _, row in df.iterrows():
        ticker_name = f"{row['ticker']} ({row['name']})"
        
        for date in dates:
            score_col = f"score_{date}"
            setup_col = f"setup_{date}"
            
            if pd.notnull(row[score_col]):
                heatmap_data.append({
                    "Ticker": ticker_name,
                    "Date": date,
                    "Score": row[score_col],
                    "Setup": row[setup_col]
                })
    
    if not heatmap_data:
        st.warning("No data available for heatmap.")
        return
    
    heatmap_df = pd.DataFrame(heatmap_data)
    
    # Create heatmap
    st.subheader("Score History Heatmap")
    
    fig = px.imshow(
        heatmap_df.pivot(index="Ticker", columns="Date", values="Score"),
        color_continuous_scale=["red", "white", "green"],
        color_continuous_midpoint=0,
        aspect="auto",
        labels=dict(color="Score")
    )
    
    fig.update_layout(
        height=max(400, min(800, len(df) * 25)),
        margin=dict(l=10, r=10, t=30, b=10),
    )
    
    st.plotly_chart(fig, use_container_width=True)

def display_price_changes(df, dates):
    """Display price changes over the period"""
    if len(dates) < 2:
        st.warning("Need at least two dates to calculate price changes.")
        return
    
    # Create a copy for display
    price_df = df.copy()
    
    # Add columns for first and last prices
    first_date = dates[0]
    last_date = dates[-1]
    first_price_col = f"price_{first_date}"
    last_price_col = f"price_{last_date}"
    
    # Calculate price changes
    price_df["First Price"] = price_df[first_price_col]
    price_df["Last Price"] = price_df[last_price_col]
    price_df["Change ($)"] = price_df["Last Price"] - price_df["First Price"]
    price_df["Change (%)"] = (price_df["Change ($)"] / price_df["First Price"] * 100).round(2)
    
    # Format for display
    display_df = price_df[["ticker", "name", "First Price", "Last Price", "Change ($)", "Change (%)"]]
    display_df = display_df.dropna(subset=["First Price", "Last Price"])
    
    if display_df.empty:
        st.warning("No price data available for the selected dates.")
        return
    
    # Sort by percentage change
    display_df = display_df.sort_values("Change (%)", ascending=False)
    
    # Display the table
    st.subheader(f"Price Changes ({first_date} to {last_date})")
    st.dataframe(display_df, use_container_width=True)

def display_score_chart(df, dates):
    """Display a line chart of score changes over time"""
    if df.empty:
        st.warning("No data selected for charting.")
        return
    
    # Prepare data for plotting
    chart_data = []
    
    for _, row in df.iterrows():
        ticker_name = f"{row['ticker']} ({row['name']})"
        
        for date in dates:
            score_col = f"score_{date}"
            
            if pd.notnull(row[score_col]):
                chart_data.append({
                    "Ticker": ticker_name,
                    "Date": date,
                    "Score": row[score_col]
                })
    
    if not chart_data:
        st.warning("No score data available for charting.")
        return
    
    chart_df = pd.DataFrame(chart_data)
    
    # Create line chart
    st.subheader("Score History Chart")
    
    fig = px.line(
        chart_df,
        x="Date",
        y="Score",
        color="Ticker",
        markers=True,
        title="Setup Score History"
    )
    
    # Add horizontal line at zero to separate long/short setups
    fig.add_shape(
        type="line",
        x0=dates[0],
        y0=0,
        x1=dates[-1],
        y1=0,
        line=dict(color="gray", width=1, dash="dash")
    )
    
    # Improve visual appearance
    fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=10, r=10, t=50, b=10),
        yaxis=dict(zeroline=False)
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Modify your existing main() function to include the dashboard functionality
# Replace your current main() function with this updated version:

def main():
    st.title("🎯 Strict Strategy Scanner")
    
    # Add tabs to separate main functions
    tab1, tab2 = st.tabs(["Scanner", "History Dashboard"])
    
    with tab1:
        # Original scanner code starts here
        with st.expander("📖 Trading Strategy Implementation"):
            st.markdown(f"""
            ### Trading Strategy Implementation
            
            This scanner strictly implements your detailed trading rules, especially focusing on the critical requirement of RSI relative to its MA.
            
            #### For LONG Setups:
            
            **Higher Timeframe Conditions (Weekly):**
            - Weekly RSI **MUST** be > 50 **AND** above its Moving Average *(this is a strict rule)*
            - Weekly MACD must be in Golden Cross OR trending above signal line
            - Weekly Price must be above key Moving Average band (EMAs {EMA_SHORT}/{EMA_LONG})
            - Monthly RSI is checked for major contradictions
            
            **Lower Timeframe Entry Criteria (Daily):**
            - Daily RSI must cross above 50 AND above its Moving Average
            - Daily MACD must show Golden Cross OR bullish hook
            - Price must find support at key MAs or trade above MAs
            
            #### For SHORT Setups:
            
            **Higher Timeframe Conditions (Weekly):**
            - Weekly RSI **MUST** be < 50 **AND** below its Moving Average *(this is a strict rule)*
            - Weekly MACD must be in Death Cross OR trending below signal line
            - Weekly Price must be below key Moving Average band (EMAs {EMA_SHORT}/{EMA_LONG})
            - Monthly RSI is checked for major contradictions
            
            **Lower Timeframe Entry Criteria (Daily):**
            - Daily RSI must cross below 50 AND below its Moving Average
            - Daily MACD must show Death Cross OR bearish hook
            - Price must be rejected at key MAs or trade below MAs
            
            #### Column Explanations:
            - **Weekly RSI**: RSI value and its relation to the 9-period moving average
            - **Weekly MACD**: MACD line relative to signal line and zero
            - **Weekly Price**: Price relation to EMAs ({EMA_SHORT}/{EMA_LONG}/{EMA_CONTEXT})
            - **Daily RSI**: Daily RSI value and its relation to its MA
            - **Daily MACD**: Daily MACD line, signal and crosses
            - **Daily Price**: Daily price relation to EMAs
            - **Monthly Trend**: Monthly RSI context
            
            #### Color Legend:
            - <span class="bullish-strong">Dark Green</span>: Strongly bullish signal
            - <span class="bullish">Light Green</span>: Moderately bullish signal
            - <span class="bearish-strong">Dark Red</span>: Strongly bearish signal
            - <span class="bearish">Light Red</span>: Moderately bearish signal
            - <span class="warning">Yellow</span>: Warning signal or condition not fully met
            - <span class="neutral">Gray</span>: Neutral signal
            
            #### Setup Types:
            - <span class="setup-long">Potential Long</span>: All mandatory HTF conditions met + ≥2 Daily rules met, strong conviction
            - <span class="setup-watch-long">Watch Long</span>: All mandatory HTF conditions met but waiting for more Daily confirmations
            - <span class="setup-caution">Caution Long</span>: Valid Long setup but with Monthly context warning
            - <span class="setup-short">Potential Short</span>: All mandatory HTF conditions met + ≥2 Daily rules met, strong conviction
            - <span class="setup-watch-short">Watch Short</span>: All mandatory HTF conditions met but waiting for more Daily confirmations
            - <span class="setup-caution">Caution Short</span>: Valid Short setup but with Monthly context warning
            """)
            
        # Initialize session state variables
        if 'scan_results' not in st.session_state:
            st.session_state.scan_results = []
        if 'selected_ticker' not in st.session_state:
            st.session_state.selected_ticker = None

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
                st.session_state.selected_ticker = None
                
                # Save results for historical tracking
                save_scan_results_history(st.session_state.scan_results)
        
        st.sidebar.markdown("---")
        st.sidebar.caption(f"Technical Parameters: RSI({RSI_WINDOW}), RSI MA({RSI_MA_PERIOD}), EMAs: {EMA_SHORT}/{EMA_LONG}/{EMA_CONTEXT}")

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
                filtered_df = display_results_table(st.session_state.scan_results)
                
                if filtered_df is not None and not filtered_df.empty:
                    # Allow user to select an instrument for detailed rule analysis
                    st.subheader("Detailed Rule Analysis")
                    # Use hidden _ticker column for selection but display names to user
                    instrument_options = [(row['_ticker'], row['Name']) for _, row in filtered_df.iterrows()]
                    instrument_dict = {t: n for t, n in instrument_options}
                    
                    selected_ticker = st.selectbox(
                        "Select an instrument for detailed rule analysis:",
                        options=list(instrument_dict.keys()),
                        format_func=lambda x: f"{instrument_dict[x]}"
                    )
                    
                    if st.button("Show Rule Details"):
                        st.session_state.selected_ticker = selected_ticker
                    
                    if st.session_state.selected_ticker:
                        # Find the selected ticker in results
                        for result in st.session_state.scan_results:
                            if result['ticker'] == st.session_state.selected_ticker:
                                display_rules_detail(
                                    result['ticker'], 
                                    result['name'], 
                                    result.get('rule_details', {})
                                )
                                break
    
    with tab2:
        # Historical dashboard tab
        display_history_dashboard()

# Keep your original main function call at the bottom of the file:
if __name__ == "__main__":
    try:
        main()  # Keep your original main function
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
