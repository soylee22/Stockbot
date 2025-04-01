# Create a row with all metrics - without Ticker, with friendly column names
        row_data = {
            "Name": r["name"],
            "_ticker": r["ticker"],  # Hidden column for filtering/selection
            "Price": r["Price"],
            "Last Update": r["Last Date"],
            "Setup": setup_html,
            "Score": r["Score"],
            "Weekly RSI": format_cell(
                metrics.get('W_RSI', {}).get('value', 'N/A'), 
                metrics.get('W_RSI', {}).get('signal', 'neutral')
            ),
            "Weekly MACD": format_cell(
                metrics.get('W_MACD', {}).get('value', 'N/A'), 
                metrics.get('W_MACD', {}).get('signal', 'neutral')
            ),
            "Weekly Price": format_cell(
                metrics.get('W_Price', {}).get('value', 'N/A'), 
                metrics.get('W_Price', {}).get('signal', 'neutral')
            ),
            "Daily RSI": format_cell(
                metrics.get('D_RSI', {}).get('value', 'N/A'), 
                metrics.get('D_RSI', {}).get('signal', 'neutral')
            ),
            "Daily MACD": format_cell(
                metrics.get('D_MACD', {}).get('value', 'N/A'), 
                metrics.get('D_MACD', {}).get('signal', 'neutral')
            ),
            "Daily Price": format_cell(
                metrics.get('D_Price', {}).get('value', 'N/A'), 
                metrics.get('D_Price', {}).get('signal', 'neutral')
            ),
            "Monthly Trend": format_cell(
                metrics.get('M_Trend', {}).get('value', 'N/A'), 
                metrics.get('M_Trend', {}).get('signal', 'neutral')
            ),
        }
