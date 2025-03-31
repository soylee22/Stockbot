def display_results_table(results_list):
    """Displays the filtered scan results, including detailed criteria checks."""
    if not results_list:
        st.warning("No results to display.")
        return None

    df_data = []

    # Helper to format rule status using Y/N
    def format_rule_status(met):
        # Use simple characters 'Y'/'N' and CSS classes
        return '<span class="criterion-met">Y</span>' if met else '<span class="criterion-not-met">N</span>'

    for i, r in enumerate(results_list):
         # Filter out errors and 'None' setups
        if r['Setup'] in ["Error", "None", "Calc Error", "Conflicting"]: continue

        # Define setup class based on the setup type
        setup_class = ""
        if "Long" in r['Setup']: setup_class = "setup-long"
        elif "Short" in r['Setup']: setup_class = "setup-short"
        if "Watch" in r['Setup']: setup_class += " setup-watch"
        setup_html = f"<span class='{setup_class.strip()}'>{r['Setup']}</span>"

        # Get indicator dictionaries
        ci = r.get('_indicators_conditions', {})
        ei = r.get('_indicators_entry', {})

        # Get boolean status for each rule
        wr_gt50 = ci.get('RSI_Bullish', False)
        wm_bull = ci.get('MACD_Bullish', False)
        wp_gt_eL = ci.get('Price_Above_EMA_Long', False)
        dr_gt50 = ei.get('RSI_Bullish', False)
        dm_bull = ei.get('MACD_Bullish', False)
        dp_gt_Es = ei.get('Daily_Price_Structure_Long', False)
        wr_lt50 = ci.get('RSI_Bearish', False)
        wm_bear = ci.get('MACD_Bearish', False)
        wp_lt_eL = ci.get('Price_Below_EMA_Long', False)
        dr_lt50 = ei.get('RSI_Bearish', False)
        dm_bear = ei.get('MACD_Bearish', False)
        dp_lt_Es = ei.get('Daily_Price_Structure_Short', False)


        df_data.append({
            "Name": r["name"],
            "Setup": setup_html,
            "Score": r["Score"],
            # --- Criteria Columns using Y/N ---
            "W:R>50": format_rule_status(wr_gt50),
            "W:M Bull": format_rule_status(wm_bull),
            "W:P>E21": format_rule_status(wp_gt_eL),
            "W:R<50": format_rule_status(wr_lt50),
            "W:M Bear": format_rule_status(wm_bear),
            "W:P<E21": format_rule_status(wp_lt_eL),
            "D:R>50": format_rule_status(dr_gt50),
            "D:M Bull": format_rule_status(dm_bull),
            "D:P>Es": format_rule_status(dp_gt_Es),
            "D:R<50": format_rule_status(dr_lt50),
            "D:M Bear": format_rule_status(dm_bear),
            "D:P<Es": format_rule_status(dp_lt_Es),
             "_original_index": i
        })

    if not df_data:
        st.info("No potential Long/Short/Watch setups found based on current criteria.")
        return None

    df_display = pd.DataFrame(df_data)
    df_display = df_display.sort_values(by="Score", ascending=False).reset_index(drop=True)

    column_order = [
        "Name", "Setup", "Score",
        "W:R>50", "W:M Bull", "W:P>E21", "W:R<50", "W:M Bear", "W:P<E21",
        "D:R>50", "D:M Bull", "D:P>Es", "D:R<50", "D:M Bear", "D:P<Es",
        "_original_index"
    ]
    df_display = df_display[[col for col in column_order if col in df_display.columns]]

    # --- Display Logic ---
    st.markdown('<div class="dataframe-container">', unsafe_allow_html=True)
    st.write(
        df_display.drop(columns=['_original_index']).to_html(
            escape=False, index=False, justify='center',
            classes="dataframe", border=0
        ),
        unsafe_allow_html=True
    )
    st.markdown('</div>', unsafe_allow_html=True)

    return df_display
