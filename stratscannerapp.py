import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pandas_ta as ta
import time

# Import ticker categories (keep using your tickers.py)
from tickers import TICKER_CATEGORIES

# --- Strategy Configuration ---
# Timeframes
TF_CONDITIONS = '1wk'  # Timeframe for Market Conditions (Weekly)
TF_ENTRY = '1d'        # Timeframe for Entry Signals (Daily)
TF_MONTHLY = '1mo'     # Monthly timeframe for additional context

# Data Periods (adjusted for RSI MA calculation)
PERIOD_CONDITIONS = "5y"
PERIOD_ENTRY = "1y"
PERIOD_MONTHLY = "10y"

# Moving Averages
EMA_SHORT = 11
EMA_LONG = 21
EMA_CONTEXT = 50  # Longer-term MA for additional context

# RSI
RSI_WINDOW = 14
RSI_MID = 50
RSI_MA_PERIOD = 9  # For RSI Moving Average

# MACD
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

# Setup Threshold (How many ENTRY rules must be met *after* conditions are met)
MIN_ENTRY_RULES_MET = 2  # e.g., Need at least 2 out of 3 Daily rules

# --- Page Config ---
st.set_page_config(
    page_title="Trading Strategy Scanner",
    page_icon="🎯",
    layout="wide",
)

# --- Custom CSS ---
st.markdown("""
<style>
    /* Enhanced setup styling with stronger colors */
    .setup-ideal-long { background-color: #006400; color: white; padding: 4px 8px; border-radius: 4px; font-weight: bold; }
    .setup-long { background-color: #28a745; color: white; padding: 4px 8px; border-radius: 4px; font-weight: bold; }
    .setup-caution-long { background-color: #5cb85c; color: white; padding: 4px 8px; border-radius: 4px; font-weight: bold; }
    .setup-ideal-short { background-color: #8b0000; color: white; padding: 4px 8px; border-radius: 4px; font-weight: bold; }
    .setup-short { background-color: #dc3545; color: white; padding: 4px 8px; border-radius: 4px; font-weight: bold; }
    .setup-caution-short { background-color: #d9534f; color: white; padding: 4px 8px; border-radius: 4px; font-weight: bold; }
    .setup-watch { background-color: #ffc107; color: black; padding: 4px 8px; border-radius: 4px; font-weight: bold; }
    .setup-none { background-color: #6c757d; color: white; padding: 4px 8px; border-radius: 4px; font-weight: bold; }
    .setup-conflict { background-color: #9932cc; color: white; padding: 4px 8px; border-radius: 4px; font-weight: bold; }
    
    /* Cell colors for metrics */
    .must-bullish { background-color: rgba(0, 100, 0, 0.4); }
    .prefer-bullish { background-color: rgba(40, 167, 69, 0.4); }
    .caution-bullish { background-color: rgba(255, 193, 7, 0.4); color: black; }
    .must-bearish { background-color: rgba(139, 0, 0, 0.4); }
    .prefer-bearish { background-color: rgba(220, 53, 69, 0.4); }
    .caution-bearish { background-color: rgba(255, 193, 7, 0.4); color: black; }
    .neutral { background-color: rgba(108, 117, 125, 0.1); }
    .red-flag { background-color: rgba(255, 0, 0, 0.2); }
    
    .stDataFrame [data-testid="stDataFrameResizable"] {
        max-height: 800px;
        overflow-y: auto;
    }
    
    /* Table styling */
    .dataframe {
        border-collapse: collapse;
        margin: 25px 0;
        font-size: 0.9em;
        font-family: sans-serif;
        min-width: 400px;
        box-shadow: 0 0 20px rgba(0, 0, 0, 0.15);
        width: 100%;
    }
    .dataframe thead tr {
        background-color: #009879;
        color: #ffffff;
        text-align: left;
    }
    .dataframe th,
    .dataframe td {
        padding: 12px 15px;
    }
    .dataframe tbody tr {
        border-bottom: 1px solid #dddddd;
    }
    .dataframe tbody tr:nth-of-type(even) {
        background-color: #f3f3f3;
    }
    .dataframe tbody tr:last-of-type {
        border-bottom: 2px solid #009879;
    }
    
    /* Metric explanation styles */
    .metric-explain {
        margin-top: 20px;
        padding: 10px;
        background-color: #f8f9fa;
        border-radius: 4px;
    }
    .metric-row {
        margin-bottom: 8px;
    }
    .metric-label {
        font-weight: bold;
        display: inline-block;
        width: 120px;
    }
    .must-condition {
        color: #dc3545;
        font-weight: bold;
    }
    .prefer-condition {
        color: #fd7e14;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# --- Helper Functions ---

@st.cache_data(ttl=1800)  # Cache for 30 minutes
def fetch_strategy_data(ticker):
    try:
        ticker_obj = yf.Ticker(ticker)
        data_conditions = ticker_obj.history(period=PERIOD_CONDITIONS, interval=TF_CONDITIONS)
        data_entry = ticker_obj.history(period=PERIOD_ENTRY, interval=TF_ENTRY)
        data_monthly = ticker_obj.history(period=PERIOD_MONTHLY, interval=TF_MONTHLY)
        
        min_len_cond = max(EMA_LONG, MACD_SLOW, RSI_WINDOW + RSI_MA_PERIOD) + 5
        min_len_entry = max(EMA_LONG, MACD_SLOW, RSI_WINDOW + RSI_MA_PERIOD) + 5
        
        if data_conditions.empty or len(data_conditions) < min_len_cond or \
           data_entry.empty or len(data_entry) < min_len_entry:
            return None, None, None
            
        return data_conditions, data_entry, data_monthly
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {str(e)}")
        return None, None, None


def calculate_strategy_indicators(data, timeframe="weekly"):
    if data is None or data.empty: 
        return None, None
        
    try:
        # Create a copy of the data to avoid SettingWithCopyWarning
        data_copy = data.copy()
        
        # Calculate primary EMAs
        data_copy.ta.ema(length=EMA_SHORT, append=True, col_names=(f"EMA_{EMA_SHORT}",))
        data_copy.ta.ema(length=EMA_LONG, append=True, col_names=(f"EMA_{EMA_LONG}",))
        
        # Calculate context EMA
        data_copy.ta.ema(length=EMA_CONTEXT, append=True, col_names=(f"EMA_{EMA_CONTEXT}",))
        
        # Calculate RSI and its Moving Average
        data_copy.ta.rsi(length=RSI_WINDOW, append=True, col_names=(f"RSI_{RSI_WINDOW}",))
        data_copy[f"RSI_{RSI_WINDOW}_MA_{RSI_MA_PERIOD}"] = data_copy[f"RSI_{RSI_WINDOW}"].rolling(window=RSI_MA_PERIOD).mean()
        
        # Calculate MACD
        data_copy.ta.macd(fast=MACD_FAST, slow=MACD_SLOW, signal=MACD_SIGNAL, append=True,
                         col_names=(f"MACD_{MACD_FAST}_{MACD_SLOW}_{MACD_SIGNAL}",
                                   f"MACDh_{MACD_FAST}_{MACD_SLOW}_{MACD_SIGNAL}",
                                   f"MACDs_{MACD_FAST}_{MACD_SLOW}_{MACD_SIGNAL}"))
        
        # Extract latest values
        indicators = {}
        indicators['Close'] = data_copy['Close'].iloc[-1]
        indicators[f'EMA_{EMA_SHORT}'] = data_copy[f'EMA_{EMA_SHORT}'].iloc[-1]
        indicators[f'EMA_{EMA_LONG}'] = data_copy[f'EMA_{EMA_LONG}'].iloc[-1]
        indicators[f'EMA_{EMA_CONTEXT}'] = data_copy[f'EMA_{EMA_CONTEXT}'].iloc[-1]
        indicators[f'RSI_{RSI_WINDOW}'] = data_copy[f'RSI_{RSI_WINDOW}'].iloc[-1]
        indicators[f'RSI_{RSI_WINDOW}_MA'] = data_copy[f'RSI_{RSI_WINDOW}_MA_{RSI_MA_PERIOD}'].iloc[-1]
        indicators[f'MACD_Line'] = data_copy[f"MACD_{MACD_FAST}_{MACD_SLOW}_{MACD_SIGNAL}"].iloc[-1]
        indicators[f'MACD_Signal'] = data_copy[f"MACDs_{MACD_FAST}_{MACD_SLOW}_{MACD_SIGNAL}"].iloc[-1]
        indicators[f'MACD_Hist'] = data_copy[f"MACDh_{MACD_FAST}_{MACD_SLOW}_{MACD_SIGNAL}"].iloc[-1]
        
        # Calculate MACD Cross Status (using last 3 bars)
        if len(data_copy) >= 3:
            macd_line_recent = data_copy[f"MACD_{MACD_FAST}_{MACD_SLOW}_{MACD_SIGNAL}"].iloc[-3:].values
            macd_signal_recent = data_copy[f"MACDs_{MACD_FAST}_{MACD_SLOW}_{MACD_SIGNAL}"].iloc[-3:].values
            
            # Check for recent Golden Cross (MACD line crosses above Signal)
            indicators['MACD_Recent_Golden_Cross'] = False
            for i in range(1, len(macd_line_recent)):
                if macd_line_recent[i-1] < macd_signal_recent[i-1] and macd_line_recent[i] > macd_signal_recent[i]:
                    indicators['MACD_Recent_Golden_Cross'] = True
                    break
                
            # Check for recent Death Cross (MACD line crosses below Signal)
            indicators['MACD_Recent_Death_Cross'] = False
            for i in range(1, len(macd_line_recent)):
                if macd_line_recent[i-1] > macd_signal_recent[i-1] and macd_line_recent[i] < macd_signal_recent[i]:
                    indicators['MACD_Recent_Death_Cross'] = True
                    break
        else:
            indicators['MACD_Recent_Golden_Cross'] = False
            indicators['MACD_Recent_Death_Cross'] = False
            
        # Check for MACD hook (changing direction without cross)
        if len(data_copy) >= 5:
            macd_hist_recent = data_copy[f"MACDh_{MACD_FAST}_{MACD_SLOW}_{MACD_SIGNAL}"].iloc[-5:].values
            
            # Bullish hook (histogram getting less negative)
            bullish_hook = False
            if macd_hist_recent[-1] < 0:
                # Look for consistent increase in histogram values (getting less negative)
                hist_changes = np.diff(macd_hist_recent[-3:])
                bullish_hook = all(change > 0 for change in hist_changes)
                
            indicators['MACD_Bullish_Hook'] = bullish_hook
            
            # Bearish hook (histogram getting less positive)
            bearish_hook = False
            if macd_hist_recent[-1] > 0:
                # Look for consistent decrease in histogram values (getting less positive)
                hist_changes = np.diff(macd_hist_recent[-3:])
                bearish_hook = all(change < 0 for change in hist_changes)
                
            indicators['MACD_Bearish_Hook'] = bearish_hook
        else:
            indicators['MACD_Bullish_Hook'] = False
            indicators['MACD_Bearish_Hook'] = False
        
        # --- Derived Boolean States Based on STRICT rules ---
        
        # RSI States - STRICT implementation (MUST vs PREFER)
        indicators['RSI_Value'] = round(indicators[f'RSI_{RSI_WINDOW}'], 1)
        indicators['RSI_MA_Value'] = round(indicators[f'RSI_{RSI_WINDOW}_MA'], 1)
        
        # Long - RSI MUST be > 50 (mandatory)
        indicators['RSI_Above_50'] = indicators[f'RSI_{RSI_WINDOW}'] > RSI_MID
        # Long - RSI preferably above MA (preferred but not mandatory)
        indicators['RSI_Above_MA'] = indicators[f'RSI_{RSI_WINDOW}'] > indicators[f'RSI_{RSI_WINDOW}_MA']
        
        # Short - RSI MUST be < 50 (mandatory)
        indicators['RSI_Below_50'] = indicators[f'RSI_{RSI_WINDOW}'] < RSI_MID
        # Short - RSI preferably below MA (preferred but not mandatory)
        indicators['RSI_Below_MA'] = indicators[f'RSI_{RSI_WINDOW}'] < indicators[f'RSI_{RSI_WINDOW}_MA']
        
        # RSI Status for Long
        if indicators['RSI_Above_50'] and indicators['RSI_Above_MA']:
            indicators['RSI_Long_Status'] = 'ideal'  # Both conditions met
        elif indicators['RSI_Above_50'] and not indicators['RSI_Above_MA']:
            indicators['RSI_Long_Status'] = 'caution'  # MUST met, PREFER not met - RED FLAG
        else:
            indicators['RSI_Long_Status'] = 'fail'  # MUST condition not met
            
        # RSI Status for Short
        if indicators['RSI_Below_50'] and indicators['RSI_Below_MA']:
            indicators['RSI_Short_Status'] = 'ideal'  # Both conditions met
        elif indicators['RSI_Below_50'] and not indicators['RSI_Below_MA']:
            indicators['RSI_Short_Status'] = 'caution'  # MUST met, PREFER not met - RED FLAG
        else:
            indicators['RSI_Short_Status'] = 'fail'  # MUST condition not met
        
        # MACD States - STRICT implementation
        # MACD relative to signal line (MUST condition)
        indicators['MACD_Above_Signal'] = indicators['MACD_Line'] > indicators['MACD_Signal']
        indicators['MACD_Below_Signal'] = indicators['MACD_Line'] < indicators['MACD_Signal']
        
        # MACD relative to zero (PREFER condition)
        indicators['MACD_Above_Zero'] = indicators['MACD_Line'] > 0
        indicators['MACD_Below_Zero'] = indicators['MACD_Line'] < 0
        
        # Recent crosses are also valid for MUST condition
        indicators['MACD_Bullish'] = indicators['MACD_Above_Signal'] or indicators['MACD_Recent_Golden_Cross']
        indicators['MACD_Bearish'] = indicators['MACD_Below_Signal'] or indicators['MACD_Recent_Death_Cross']
        
        # MACD Status for Long
        if indicators['MACD_Bullish'] and indicators['MACD_Above_Zero']:
            indicators['MACD_Long_Status'] = 'ideal'  # Both conditions met
        elif indicators['MACD_Bullish'] and not indicators['MACD_Above_Zero']:
            indicators['MACD_Long_Status'] = 'caution'  # MUST met, PREFER not met
        else:
            indicators['MACD_Long_Status'] = 'fail'  # MUST condition not met
            
        # MACD Status for Short
        if indicators['MACD_Bearish'] and indicators['MACD_Below_Zero']:
            indicators['MACD_Short_Status'] = 'ideal'  # Both conditions met
        elif indicators['MACD_Bearish'] and not indicators['MACD_Below_Zero']:
            indicators['MACD_Short_Status'] = 'caution'  # MUST met, PREFER not met
        else:
            indicators['MACD_Short_Status'] = 'fail'  # MUST condition not met
        
        # Price Structure - STRICT implementation
        # Price relative to EMAs
        indicators['Price_Above_EMA_Short'] = indicators['Close'] > indicators[f'EMA_{EMA_SHORT}']
        indicators['Price_Above_EMA_Long'] = indicators['Close'] > indicators[f'EMA_{EMA_LONG}']
        indicators['Price_Above_EMA_Context'] = indicators['Close'] > indicators[f'EMA_{EMA_CONTEXT}']
        
        indicators['Price_Below_EMA_Short'] = indicators['Close'] < indicators[f'EMA_{EMA_SHORT}']
        indicators['Price_Below_EMA_Long'] = indicators['Close'] < indicators[f'EMA_{EMA_LONG}']
        indicators['Price_Below_EMA_Context'] = indicators['Close'] < indicators[f'EMA_{EMA_CONTEXT}']
        
        # EMA band relationship (cloud)
        indicators['EMA_Band_Bullish'] = indicators[f'EMA_{EMA_SHORT}'] > indicators[f'EMA_{EMA_LONG}']
        indicators['EMA_Band_Bearish'] = indicators[f'EMA_{EMA_SHORT}'] < indicators[f'EMA_{EMA_LONG}']
        
        # For Long: 
        # MUST: Price above both 11 & 21 EMAs
        # PREFER: Also above 50 EMA
        indicators['Price_Above_MA_Band'] = indicators['Price_Above_EMA_Short'] and indicators['Price_Above_EMA_Long']
        
        # For Short:
        # MUST: Price below both 11 & 21 EMAs
        # PREFER: Also below 50 EMA
        indicators['Price_Below_MA_Band'] = indicators['Price_Below_EMA_Short'] and indicators['Price_Below_EMA_Long']
        
        # Price Status for Long
        if indicators['Price_Above_MA_Band'] and indicators['Price_Above_EMA_Context']:
            indicators['Price_Long_Status'] = 'ideal'  # Both conditions met
        elif indicators['Price_Above_MA_Band'] and not indicators['Price_Above_EMA_Context']:
            indicators['Price_Long_Status'] = 'caution'  # MUST met, PREFER not met
        else:
            indicators['Price_Long_Status'] = 'fail'  # MUST condition not met
            
        # Price Status for Short
        if indicators['Price_Below_MA_Band'] and indicators['Price_Below_EMA_Context']:
            indicators['Price_Short_Status'] = 'ideal'  # Both conditions met
        elif indicators['Price_Below_MA_Band'] and not indicators['Price_Below_EMA_Context']:
            indicators['Price_Short_Status'] = 'caution'  # MUST met, PREFER not met
        else:
            indicators['Price_Short_Status'] = 'fail'  # MUST condition not met
        
        # Additional indicators for Daily timeframe
        if timeframe == "daily":
            # Check for RSI crossing above/below 50 and MA (recent)
            if len(data_copy) >= 5:
                rsi_values = data_copy[f'RSI_{RSI_WINDOW}'].iloc[-5:].values
                rsi_ma_values = data_copy[f'RSI_{RSI_WINDOW}_MA_{RSI_MA_PERIOD}'].iloc[-5:].values
                
                # Check for RSI crossing above 50 recently
                rsi_crossed_above_50 = False
                for i in range(1, len(rsi_values)):
                    if rsi_values[i-1] < 50 and rsi_values[i] > 50:
                        rsi_crossed_above_50 = True
                        break
                
                # Check for RSI crossing below 50 recently
                rsi_crossed_below_50 = False
                for i in range(1, len(rsi_values)):
                    if rsi_values[i-1] > 50 and rsi_values[i] < 50:
                        rsi_crossed_below_50 = True
                        break
                
                # Check for RSI crossing above MA recently
                rsi_crossed_above_ma = False
                for i in range(1, len(rsi_values)):
                    if rsi_values[i-1] < rsi_ma_values[i-1] and rsi_values[i] > rsi_ma_values[i]:
                        rsi_crossed_above_ma = True
                        break
                
                # Check for RSI crossing below MA recently
                rsi_crossed_below_ma = False
                for i in range(1, len(rsi_values)):
                    if rsi_values[i-1] > rsi_ma_values[i-1] and rsi_values[i] < rsi_ma_values[i]:
                        rsi_crossed_below_ma = True
                        break
                
                indicators['RSI_Recent_Cross_Above_50'] = rsi_crossed_above_50
                indicators['RSI_Recent_Cross_Below_50'] = rsi_crossed_below_50
                indicators['RSI_Recent_Cross_Above_MA'] = rsi_crossed_above_ma
                indicators['RSI_Recent_Cross_Below_MA'] = rsi_crossed_below_ma
                
                # Check for RSI crossing both 50 and MA
                indicators['RSI_Bullish_Cross_Complete'] = rsi_crossed_above_50 and rsi_crossed_above_ma
                indicators['RSI_Bearish_Cross_Complete'] = rsi_crossed_below_50 and rsi_crossed_below_ma
            else:
                indicators['RSI_Recent_Cross_Above_50'] = False
                indicators['RSI_Recent_Cross_Below_50'] = False
                indicators['RSI_Recent_Cross_Above_MA'] = False
                indicators['RSI_Recent_Cross_Below_MA'] = False
                indicators['RSI_Bullish_Cross_Complete'] = False
                indicators['RSI_Bearish_Cross_Complete'] = False
            
            # Check for pullbacks/rejections at EMAs
            if len(data_copy) >= 10:
                recent_lows = data_copy['Low'].iloc[-10:].values
                recent_highs = data_copy['High'].iloc[-10:].values
                recent_ema_short = data_copy[f'EMA_{EMA_SHORT}'].iloc[-10:].values
                recent_ema_long = data_copy[f'EMA_{EMA_LONG}'].iloc[-10:].values
                
                # For long: check if price recently pulled back to EMAs and bounced
                pullback_to_emas = False
                for i in range(len(recent_lows)):
                    # Check if Low came close to either EMA
                    near_ema_short = abs(recent_lows[i] - recent_ema_short[i]) / recent_ema_short[i] < 0.01
                    near_ema_long = abs(recent_lows[i] - recent_ema_long[i]) / recent_ema_long[i] < 0.01
                    
                    # And then price moved higher (bounce)
                    if (near_ema_short or near_ema_long) and i < len(recent_lows) - 1:
                        if data_copy['Close'].iloc[-1] > recent_lows[i]:
                            pullback_to_emas = True
                            break
                
                # For short: check if price recently rallied to EMAs and got rejected
                rejection_at_emas = False
                for i in range(len(recent_highs)):
                    # Check if High came close to either EMA
                    near_ema_short = abs(recent_highs[i] - recent_ema_short[i]) / recent_ema_short[i] < 0.01
                    near_ema_long = abs(recent_highs[i] - recent_ema_long[i]) / recent_ema_long[i] < 0.01
                    
                    # And then price moved lower (rejection)
                    if (near_ema_short or near_ema_long) and i < len(recent_highs) - 1:
                        if data_copy['Close'].iloc[-1] < recent_highs[i]:
                            rejection_at_emas = True
                            break
                
                indicators['Recent_Pullback_To_EMA'] = pullback_to_emas
                indicators['Recent_Rejection_At_EMA'] = rejection_at_emas
            else:
                indicators['Recent_Pullback_To_EMA'] = False
                indicators['Recent_Rejection_At_EMA'] = False
        
        return indicators, data_copy
    except Exception as e:
        st.error(f"Error calculating indicators for {timeframe}: {str(e)}")
        return None, None


def check_monthly_contradictions(monthly_indicators):
    """Check if monthly timeframe has major contradictions to weekly signals"""
    if not monthly_indicators:
        return False, False, "No monthly data"
    
    # Check contradictions for LONG setups
    monthly_contradicts_long = False
    long_contradiction_reason = ""
    
    # Check for major RSI contradiction (deeply below 50)
    if monthly_indicators.get('RSI_Below_50', False) and monthly_indicators.get('RSI_Value', 50) < 40:
        monthly_contradicts_long = True
        long_contradiction_reason = f"Monthly RSI deeply bearish ({monthly_indicators.get('RSI_Value', 0):.1f})"
    
    # Check for major MACD contradiction (strong bearish signal)
    if monthly_indicators.get('MACD_Short_Status', '') == 'ideal':
        if not monthly_contradicts_long:  # Only update if not already contradicted
            monthly_contradicts_long = True
            long_contradiction_reason = "Monthly MACD strongly bearish"
    
    # Check contradictions for SHORT setups
    monthly_contradicts_short = False
    short_contradiction_reason = ""
    
    # Check for major RSI contradiction (strongly above 50)
    if monthly_indicators.get('RSI_Above_50', False) and monthly_indicators.get('RSI_Value', 50) > 60:
        monthly_contradicts_short = True
        short_contradiction_reason = f"Monthly RSI strongly bullish ({monthly_indicators.get('RSI_Value', 0):.1f})"
    
    # Check for major MACD contradiction (strong bullish signal)
    if monthly_indicators.get('MACD_Long_Status', '') == 'ideal':
        if not monthly_contradicts_short:  # Only update if not already contradicted
            monthly_contradicts_short = True
            short_contradiction_reason = "Monthly MACD strongly bullish"
    
    return monthly_contradicts_long, monthly_contradicts_short, long_contradiction_reason, short_contradiction_reason


def check_strategy_setup(weekly_indicators, daily_indicators, monthly_indicators=None):
    """Strictly implements trading rules with MUST vs PREFER conditions explicitly checked"""
    if not weekly_indicators or not daily_indicators: 
        return "Error", 0, [], {}
    
    # Initialize
    long_setup_type = "None"
    short_setup_type = "None"
    long_score = 0
    short_score = 0
    long_rules_met = []
    short_rules_met = []
    long_red_flags = []
    short_red_flags = []
    
    # Collect all metrics for display
    all_metrics = {}
    
    # --- Process Weekly (HTF) Metrics ---
    # Store metrics from weekly timeframe for display
    all_metrics['W_RSI'] = {
        'value': f"{weekly_indicators.get('RSI_Value', 0):.1f} vs MA: {weekly_indicators.get('RSI_MA_Value', 0):.1f}",
        'signal': 'must-bullish' if weekly_indicators.get('RSI_Long_Status', '') == 'ideal' else
                 'caution-bullish' if weekly_indicators.get('RSI_Long_Status', '') == 'caution' else
                 'must-bearish' if weekly_indicators.get('RSI_Short_Status', '') == 'ideal' else
                 'caution-bearish' if weekly_indicators.get('RSI_Short_Status', '') == 'caution' else 'neutral',
        'desc': 'RSI>50 & >MA ✓' if weekly_indicators.get('RSI_Long_Status', '') == 'ideal' else
               'RSI>50 but <MA ⚠️' if weekly_indicators.get('RSI_Long_Status', '') == 'caution' else
               'RSI<50 & <MA ✓' if weekly_indicators.get('RSI_Short_Status', '') == 'ideal' else
               'RSI<50 but >MA ⚠️' if weekly_indicators.get('RSI_Short_Status', '') == 'caution' else 'Neutral',
        'long_status': weekly_indicators.get('RSI_Long_Status', 'fail'),
        'short_status': weekly_indicators.get('RSI_Short_Status', 'fail')
    }
    
    all_metrics['W_MACD'] = {
        'value': f"{weekly_indicators.get('MACD_Line', 0):.3f} vs {weekly_indicators.get('MACD_Signal', 0):.3f}",
        'signal': 'must-bullish' if weekly_indicators.get('MACD_Long_Status', '') == 'ideal' else
                 'caution-bullish' if weekly_indicators.get('MACD_Long_Status', '') == 'caution' else
                 'must-bearish' if weekly_indicators.get('MACD_Short_Status', '') == 'ideal' else
                 'caution-bearish' if weekly_indicators.get('MACD_Short_Status', '') == 'caution' else 'neutral',
        'desc': 'MACD>Sig & >0 ✓' if weekly_indicators.get('MACD_Long_Status', '') == 'ideal' else
               'MACD>Sig but <0 ⚠️' if weekly_indicators.get('MACD_Long_Status', '') == 'caution' else
               'MACD<Sig & <0 ✓' if weekly_indicators.get('MACD_Short_Status', '') == 'ideal' else
               'MACD<Sig but >0 ⚠️' if weekly_indicators.get('MACD_Short_Status', '') == 'caution' else 'Neutral',
        'long_status': weekly_indicators.get('MACD_Long_Status', 'fail'),
        'short_status': weekly_indicators.get('MACD_Short_Status', 'fail')
    }
    
    all_metrics['W_Price'] = {
        'value': f"{weekly_indicators.get('Close', 0):.2f} vs EMAs",
        'signal': 'must-bullish' if weekly_indicators.get('Price_Long_Status', '') == 'ideal' else
                 'caution-bullish' if weekly_indicators.get('Price_Long_Status', '') == 'caution' else
                 'must-bearish' if weekly_indicators.get('Price_Short_Status', '') == 'ideal' else
                 'caution-bearish' if weekly_indicators.get('Price_Short_Status', '') == 'caution' else 'neutral',
        'desc': 'P>EMAs & P>EMA50 ✓' if weekly_indicators.get('Price_Long_Status', '') == 'ideal' else
               'P>EMAs but <EMA50 ⚠️' if weekly_indicators.get('Price_Long_Status', '') == 'caution' else
               'P<EMAs & P<EMA50 ✓' if weekly_indicators.get('Price_Short_Status', '') == 'ideal' else
               'P<EMAs but >EMA50 ⚠️' if weekly_indicators.get('Price_Short_Status', '') == 'caution' else 'Neutral',
        'long_status': weekly_indicators.get('Price_Long_Status', 'fail'),
        'short_status': weekly_indicators.get('Price_Short_Status', 'fail')
    }
    
    # --- Process Daily (LTF) Metrics ---
    # For LTF RSI, specifically check for cross above/below 50 AND MA
    rsi_ltf_long_status = 'fail'
    if daily_indicators.get('RSI_Bullish_Cross_Complete', False):
        rsi_ltf_long_status = 'ideal'  # Recently crossed above both 50 and MA
    elif daily_indicators.get('RSI_Above_50', False) and daily_indicators.get('RSI_Above_MA', False):
        rsi_ltf_long_status = 'caution'  # Above both but didn't just cross
    
    rsi_ltf_short_status = 'fail'
    if daily_indicators.get('RSI_Bearish_Cross_Complete', False):
        rsi_ltf_short_status = 'ideal'  # Recently crossed below both 50 and MA
    elif daily_indicators.get('RSI_Below_50', False) and daily_indicators.get('RSI_Below_MA', False):
        rsi_ltf_short_status = 'caution'  # Below both but didn't just cross
    
    all_metrics['D_RSI'] = {
        'value': f"{daily_indicators.get('RSI_Value', 0):.1f} vs MA: {daily_indicators.get('RSI_MA_Value', 0):.1f}",
        'signal': 'must-bullish' if rsi_ltf_long_status == 'ideal' else
                 'caution-bullish' if rsi_ltf_long_status == 'caution' else
                 'must-bearish' if rsi_ltf_short_status == 'ideal' else
                 'caution-bearish' if rsi_ltf_short_status == 'caution' else 'neutral',
        'desc': 'RSI crossed >50 & >MA ✓' if daily_indicators.get('RSI_Bullish_Cross_Complete', False) else
               'RSI>50 & >MA ⚠️' if daily_indicators.get('RSI_Above_50', False) and daily_indicators.get('RSI_Above_MA', False) else
               'RSI crossed <50 & <MA ✓' if daily_indicators.get('RSI_Bearish_Cross_Complete', False) else
               'RSI<50 & <MA ⚠️' if daily_indicators.get('RSI_Below_50', False) and daily_indicators.get('RSI_Below_MA', False) else 'Neutral',
        'long_status': rsi_ltf_long_status,
        'short_status': rsi_ltf_short_status
    }
    
    # For LTF MACD, check for Golden/Death Cross OR clear hook
    macd_ltf_long_status = 'fail'
    if daily_indicators.get('MACD_Recent_Golden_Cross', False):
        macd_ltf_long_status = 'ideal'  # Recent Golden Cross
    elif daily_indicators.get('MACD_Bullish_Hook', False) and daily_indicators.get('MACD_Above_Zero', False):
        macd_ltf_long_status = 'ideal'  # Hook upwards above zero
    elif daily_indicators.get('MACD_Bullish_Hook', False):
        macd_ltf_long_status = 'caution'  # Hook upwards but below zero
    elif daily_indicators.get('MACD_Above_Signal', False) and daily_indicators.get('MACD_Above_Zero', False):
        macd_ltf_long_status = 'caution'  # Above signal and zero but not a recent cross or hook
    
    macd_ltf_short_status = 'fail'
    if daily_indicators.get('MACD_Recent_Death_Cross', False):
        macd_ltf_short_status = 'ideal'  # Recent Death Cross
    elif daily_indicators.get('MACD_Bearish_Hook', False) and daily_indicators.get('MACD_Below_Zero', False):
        macd_ltf_short_status = 'ideal'  # Hook downwards below zero
    elif daily_indicators.get('MACD_Bearish_Hook', False):
        macd_ltf_short_status = 'caution'  # Hook downwards but above zero
    elif daily_indicators.get('MACD_Below_Signal', False) and daily_indicators.get('MACD_Below_Zero', False):
        macd_ltf_short_status = 'caution'  # Below signal and zero but not a recent cross or hook
    
    all_metrics['D_MACD'] = {
        'value': f"{daily_indicators.get('MACD_Line', 0):.3f} vs {daily_indicators.get('MACD_Signal', 0):.3f}",
        'signal': 'must-bullish' if macd_ltf_long_status == 'ideal' else
                 'caution-bullish' if macd_ltf_long_status == 'caution' else
                 'must-bearish' if macd_ltf_short_status == 'ideal' else
                 'caution-bearish' if macd_ltf_short_status == 'caution' else 'neutral',
        'desc': 'MACD Golden Cross/Hook ✓' if macd_ltf_long_status == 'ideal' else
               'MACD>Sig & >0 ⚠️' if macd_ltf_long_status == 'caution' else
               'MACD Death Cross/Hook ✓' if macd_ltf_short_status == 'ideal' else
               'MACD<Sig & <0 ⚠️' if macd_ltf_short_status == 'caution' else 'Neutral',
        'long_status': macd_ltf_long_status,
        'short_status': macd_ltf_short_status
    }
    
    # For LTF Price Action, check for pullback to support or rejection at resistance
    price_ltf_long_status = 'fail'
    if daily_indicators.get('Recent_Pullback_To_EMA', False) and daily_indicators.get('Price_Above_MA_Band', False):
        price_ltf_long_status = 'ideal'  # Pullback to MA and bounce, now above MAs
    elif daily_indicators.get('Price_Above_MA_Band', False) and daily_indicators.get('Price_Above_EMA_Context', False):
        price_ltf_long_status = 'caution'  # Above MAs but no recent pullback
    elif daily_indicators.get('Price_Above_MA_Band', False):
        price_ltf_long_status = 'caution'  # Above MA band but not above longer-term MA
    
    price_ltf_short_status = 'fail'
    if daily_indicators.get('Recent_Rejection_At_EMA', False) and daily_indicators.get('Price_Below_MA_Band', False):
        price_ltf_short_status = 'ideal'  # Rejection at MA, now below MAs
    elif daily_indicators.get('Price_Below_MA_Band', False) and daily_indicators.get('Price_Below_EMA_Context', False):
        price_ltf_short_status = 'caution'  # Below MAs but no recent rejection
    elif daily_indicators.get('Price_Below_MA_Band', False):
        price_ltf_short_status = 'caution'  # Below MA band but not below longer-term MA
    
    all_metrics['D_Price'] = {
        'value': f"{daily_indicators.get('Close', 0):.2f} vs EMAs",
        'signal': 'must-bullish' if price_ltf_long_status == 'ideal' else
                 'caution-bullish' if price_ltf_long_status == 'caution' else
                 'must-bearish' if price_ltf_short_status == 'ideal' else
                 'caution-bearish' if price_ltf_short_status == 'caution' else 'neutral',
        'desc': 'Pullback Support at EMAs ✓' if price_ltf_long_status == 'ideal' else
               'Price>EMAs without pullback ⚠️' if price_ltf_long_status == 'caution' else
               'Rejection at EMAs ✓' if price_ltf_short_status == 'ideal' else
               'Price<EMAs without rejection ⚠️' if price_ltf_short_status == 'caution' else 'Neutral',
        'long_status': price_ltf_long_status,
        'short_status': price_ltf_short_status
    }
    
    # --- Process Monthly context ---
    monthly_contradicts_long, monthly_contradicts_short, long_contradiction_reason, short_contradiction_reason = False, False, "", ""
    if monthly_indicators:
        monthly_contradicts_long, monthly_contradicts_short, long_contradiction_reason, short_contradiction_reason = check_monthly_contradictions(monthly_indicators)
        
        all_metrics['M_Trend'] = {
            'value': f"RSI: {monthly_indicators.get('RSI_Value', 0):.1f}, MACD: {monthly_indicators.get('MACD_Line', 0):.3f}",
            'signal': 'red-flag' if monthly_contradicts_long or monthly_contradicts_short else 
                     'must-bullish' if monthly_indicators.get('RSI_Above_50', False) else
                     'must-bearish' if monthly_indicators.get('RSI_Below_50', False) else 'neutral',
            'desc': long_contradiction_reason if monthly_contradicts_long else 
                   short_contradiction_reason if monthly_contradicts_short else
                   'No major contradictions',
            'contradicts_long': monthly_contradicts_long,
            'contradicts_short': monthly_contradicts_short
        }
    else:
        all_metrics['M_Trend'] = {
            'value': 'N/A',
            'signal': 'neutral',
            'desc': 'Not Available',
            'contradicts_long': False,
            'contradicts_short': False
        }
    
    # --- Check LONG Setup Conditions (Group 1 - Higher Timeframe) ---
    
    # Rule 1.1: Weekly RSI MUST be > 50 AND preferably above MA
    weekly_rsi_long_check = weekly_indicators.get('RSI_Long_Status', 'fail')
    
    # Rule 1.2: Weekly MACD MUST be bullish AND preferably above zero
    weekly_macd_long_check = weekly_indicators.get('MACD_Long_Status', 'fail')
    
    # Rule 1.3: Weekly Price MUST be above MA band AND ideally above longer-term MA
    weekly_price_long_check = weekly_indicators.get('Price_Long_Status', 'fail')
    
    # Check if ALL MUST conditions are met (at minimum level)
    long_htf_conditions_met = (
        weekly_rsi_long_check != 'fail' and  # RSI > 50 (at minimum)
        weekly_macd_long_check != 'fail' and  # MACD bullish (at minimum)
        weekly_price_long_check != 'fail'  # Price above MA band (at minimum)
    )
    
    if long_htf_conditions_met:
        # Start with base score
        long_score = 3
        
        # Add basic rules met
        long_rules_met.append("W:RSI>50")
        long_rules_met.append("W:MACD Bullish")
        long_rules_met.append("W:Price>EMAs")
        
        # Check for red flags in weekly (MUST met but PREFER not met)
        if weekly_rsi_long_check == 'caution':
            long_red_flags.append("W:RSI>50 but <MA ⚠️")
        if weekly_macd_long_check == 'caution':
            long_red_flags.append("W:MACD>Signal but <0 ⚠️")
        if weekly_price_long_check == 'caution':
            long_red_flags.append("W:Price>EMAs but <EMA50 ⚠️")
        
        # Check monthly for contradictions
        if monthly_contradicts_long:
            long_red_flags.append(f"M:{long_contradiction_reason} ⚠️")
        
        # --- Check Daily (LTF) Entry Signals for Long ---
        ltf_entry_ideal_count = 0
        ltf_entry_caution_count = 0
        
        # Rule 2.1: LTF RSI Confirmation
        if all_metrics['D_RSI']['long_status'] == 'ideal':
            ltf_entry_ideal_count += 1
            long_rules_met.append("D:RSI crossed >50 & >MA")
        elif all_metrics['D_RSI']['long_status'] == 'caution':
            ltf_entry_caution_count += 1
            long_rules_met.append("D:RSI>50 & >MA (no cross)")
            long_red_flags.append("D:RSI no recent cross ⚠️")
        
        # Rule 2.2: LTF MACD Confirmation
        if all_metrics['D_MACD']['long_status'] == 'ideal':
            ltf_entry_ideal_count += 1
            long_rules_met.append("D:MACD Golden Cross/Hook")
        elif all_metrics['D_MACD']['long_status'] == 'caution':
            ltf_entry_caution_count += 1
            long_rules_met.append("D:MACD>Signal (no cross)")
            long_red_flags.append("D:MACD no recent cross ⚠️")
        
        # Rule 2.3: LTF Price Action
        if all_metrics['D_Price']['long_status'] == 'ideal':
            ltf_entry_ideal_count += 1
            long_rules_met.append("D:Pullback Support at EMAs")
        elif all_metrics['D_Price']['long_status'] == 'caution':
            ltf_entry_caution_count += 1
            long_rules_met.append("D:Price>EMAs (no pullback)")
            long_red_flags.append("D:No recent pullback ⚠️")
        
        # Score LTF signals
        total_ltf_rules_met = ltf_entry_ideal_count + ltf_entry_caution_count
        long_score += ltf_entry_ideal_count + (ltf_entry_caution_count * 0.5)  # Full points for ideal, half for caution
        
        # Determine Long Setup Quality based on:
        # 1. HTF conditions quality (any cautions?)
        # 2. LTF entry signals (enough met? how many ideal vs caution?)
        # 3. Monthly contradictions
        
        if total_ltf_rules_met >= MIN_ENTRY_RULES_MET:
            htf_all_ideal = (weekly_rsi_long_check == 'ideal' and 
                            weekly_macd_long_check == 'ideal' and 
                            weekly_price_long_check == 'ideal')
            
            ltf_mostly_ideal = ltf_entry_ideal_count >= MIN_ENTRY_RULES_MET
            
            if htf_all_ideal and ltf_mostly_ideal and not monthly_contradicts_long:
                long_setup_type = "Ideal Long"  # Perfect setup
            elif monthly_contradicts_long:
                long_setup_type = "Caution Long"  # Monthly contradiction
            elif len(long_red_flags) > 2:
                long_setup_type = "Caution Long"  # Too many red flags
            else:
                long_setup_type = "Potential Long"  # Good setup with some cautions
        else:
            long_setup_type = "Watch Long"  # HTF conditions met but not enough LTF signals
    
    # --- Check SHORT Setup Conditions (Group 1 - Higher Timeframe) ---
    
    # Rule 1.1: Weekly RSI MUST be < 50 AND preferably below MA
    weekly_rsi_short_check = weekly_indicators.get('RSI_Short_Status', 'fail')
    
    # Rule 1.2: Weekly MACD MUST be bearish AND preferably below zero
    weekly_macd_short_check = weekly_indicators.get('MACD_Short_Status', 'fail')
    
    # Rule 1.3: Weekly Price MUST be below MA band AND ideally below longer-term MA
    weekly_price_short_check = weekly_indicators.get('Price_Short_Status', 'fail')
    
    # Check if ALL MUST conditions are met (at minimum level)
    short_htf_conditions_met = (
        weekly_rsi_short_check != 'fail' and  # RSI < 50 (at minimum)
        weekly_macd_short_check != 'fail' and  # MACD bearish (at minimum)
        weekly_price_short_check != 'fail'  # Price below MA band (at minimum)
    )
    
    if short_htf_conditions_met:
        # Start with base score
        short_score = 3
        
        # Add basic rules met
        short_rules_met.append("W:RSI<50")
        short_rules_met.append("W:MACD Bearish")
        short_rules_met.append("W:Price<EMAs")
        
        # Check for red flags in weekly (MUST met but PREFER not met)
        if weekly_rsi_short_check == 'caution':
            short_red_flags.append("W:RSI<50 but >MA ⚠️")
        if weekly_macd_short_check == 'caution':
            short_red_flags.append("W:MACD<Signal but >0 ⚠️")
        if weekly_price_short_check == 'caution':
            short_red_flags.append("W:Price<EMAs but >EMA50 ⚠️")
        
        # Check monthly for contradictions
        if monthly_contradicts_short:
            short_red_flags.append(f"M:{short_contradiction_reason} ⚠️")
        
        # --- Check Daily (LTF) Entry Signals for Short ---
        ltf_entry_ideal_count = 0
        ltf_entry_caution_count = 0
        
        # Rule 2.1: LTF RSI Confirmation
        if all_metrics['D_RSI']['short_status'] == 'ideal':
            ltf_entry_ideal_count += 1
            short_rules_met.append("D:RSI crossed <50 & <MA")
        elif all_metrics['D_RSI']['short_status'] == 'caution':
            ltf_entry_caution_count += 1
            short_rules_met.append("D:RSI<50 & <MA (no cross)")
            short_red_flags.append("D:RSI no recent cross ⚠️")
        
        # Rule 2.2: LTF MACD Confirmation
        if all_metrics['D_MACD']['short_status'] == 'ideal':
            ltf_entry_ideal_count += 1
            short_rules_met.append("D:MACD Death Cross/Hook")
        elif all_metrics['D_MACD']['short_status'] == 'caution':
            ltf_entry_caution_count += 1
            short_rules_met.append("D:MACD<Signal (no cross)")
            short_red_flags.append("D:MACD no recent cross ⚠️")
        
        # Rule 2.3: LTF Price Action
        if all_metrics['D_Price']['short_status'] == 'ideal':
            ltf_entry_ideal_count += 1
            short_rules_met.append("D:Rejection at EMAs")
        elif all_metrics['D_Price']['short_status'] == 'caution':
            ltf_entry_caution_count += 1
            short_rules_met.append("D:Price<EMAs (no rejection)")
            short_red_flags.append("D:No recent rejection ⚠️")
        
        # Score LTF signals
        total_ltf_rules_met = ltf_entry_ideal_count + ltf_entry_caution_count
        short_score += ltf_entry_ideal_count + (ltf_entry_caution_count * 0.5)  # Full points for ideal, half for caution
        
        # Determine Short Setup Quality
        if total_ltf_rules_met >= MIN_ENTRY_RULES_MET:
            htf_all_ideal = (weekly_rsi_short_check == 'ideal' and 
                            weekly_macd_short_check == 'ideal' and 
                            weekly_price_short_check == 'ideal')
            
            ltf_mostly_ideal = ltf_entry_ideal_count >= MIN_ENTRY_RULES_MET
            
            if htf_all_ideal and ltf_mostly_ideal and not monthly_contradicts_short:
                short_setup_type = "Ideal Short"  # Perfect setup
            elif monthly_contradicts_short:
                short_setup_type = "Caution Short"  # Monthly contradiction
            elif len(short_red_flags) > 2:
                short_setup_type = "Caution Short"  # Too many red flags
            else:
                short_setup_type = "Potential Short"  # Good setup with some cautions
        else:
            short_setup_type = "Watch Short"  # HTF conditions met but not enough LTF signals
    
    # --- Determine Final Setup Type ---
    final_setup = "None"
    final_score = 0
    final_rules = []
    red_flags = []
    
    # No conflict between Long and Short (one is None)
    if long_setup_type != "None" and short_setup_type == "None":
        final_setup = long_setup_type
        final_score = long_score
        final_rules = long_rules_met
        red_flags = long_red_flags
    elif short_setup_type != "None" and long_setup_type == "None":
        final_setup = short_setup_type
        final_score = -short_score  # Negative score for shorts
        final_rules = short_rules_met
        red_flags = short_red_flags
    
    # Handle conflict cases
    elif long_setup_type != "None" and short_setup_type != "None":
        # Both long and short setups detected - this is a conflict
        final_setup = "Conflict"
        final_score = 0
        final_rules = ["Conflicting Signals"]
        red_flags = ["Both Long and Short conditions met simultaneously"]
    else:
        final_setup = "None"
        final_score = 0
        final_rules = []
        red_flags = []
    
    return final_setup, final_score, final_rules, red_flags, all_metrics


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
                data_conditions, data_entry, data_monthly = fetch_strategy_data(ticker)
                if data_conditions is None or data_entry is None:
                    results.append({
                        "ticker": ticker, 
                        "name": name, 
                        "Setup": "Data Error", 
                        "Score": 0, 
                        "Rules Met": [], 
                        "Red Flags": [],
                        "error": True,
                        "metrics": {}
                    })
                    continue
                    
                weekly_indicators, _ = calculate_strategy_indicators(data_conditions, "weekly")
                daily_indicators, _ = calculate_strategy_indicators(data_entry, "daily")
                monthly_indicators = None
                if data_monthly is not None and not data_monthly.empty:
                    monthly_indicators, _ = calculate_strategy_indicators(data_monthly, "monthly")
                
                if weekly_indicators is None or daily_indicators is None:
                    results.append({
                        "ticker": ticker, 
                        "name": name, 
                        "Setup": "Calc Error", 
                        "Score": 0, 
                        "Rules Met": [], 
                        "Red Flags": [],
                        "error": True,
                        "metrics": {}
                    })
                    continue
                    
                setup_type, setup_score, rules_met, red_flags, all_metrics = check_strategy_setup(
                    weekly_indicators, daily_indicators, monthly_indicators
                )
                
                # Calculate price and date for display
                current_price = daily_indicators.get('Close', 0) if daily_indicators else 0
                last_date = data_entry.index[-1].strftime('%Y-%m-%d') if data_entry is not None and not data_entry.empty else "N/A"
                
                results.append({
                    "ticker": ticker, 
                    "name": name, 
                    "Setup": setup_type, 
                    "Score": setup_score,
                    "Price": round(current_price, 2),
                    "Last Date": last_date,
                    "Rules Met": rules_met, 
                    "Red Flags": red_flags,
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
                    "Red Flags": [],
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
    if signal_type == 'must-bullish':
        return f'<span class="must-bullish">{value}</span>'
    elif signal_type == 'prefer-bullish':
        return f'<span class="prefer-bullish">{value}</span>'
    elif signal_type == 'caution-bullish':
        return f'<span class="caution-bullish">{value}</span>'
    elif signal_type == 'must-bearish':
        return f'<span class="must-bearish">{value}</span>'
    elif signal_type == 'prefer-bearish':
        return f'<span class="prefer-bearish">{value}</span>'
    elif signal_type == 'caution-bearish':
        return f'<span class="caution-bearish">{value}</span>'
    elif signal_type == 'red-flag':
        return f'<span class="red-flag">{value}</span>'
    else:
        return f'<span class="neutral">{value}</span>'


def display_results_table(results_list):
    """Displays the scan results with metrics columns, highlighting MUST vs PREFER conditions"""
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
        if "Ideal Long" in r['Setup']: setup_class = "setup-ideal-long"
        elif "Potential Long" in r['Setup']: setup_class = "setup-long"
        elif "Caution Long" in r['Setup']: setup_class = "setup-caution-long"
        elif "Watch Long" in r['Setup']: setup_class = "setup-watch"
        elif "Ideal Short" in r['Setup']: setup_class = "setup-ideal-short"
        elif "Potential Short" in r['Setup']: setup_class = "setup-short"
        elif "Caution Short" in r['Setup']: setup_class = "setup-caution-short"
        elif "Watch Short" in r['Setup']: setup_class = "setup-watch"
        elif "Conflict" in r['Setup']: setup_class = "setup-conflict"
        
        # Format setup with HTML for styling
        setup_html = f'<span class="{setup_class}">{r["Setup"]}</span>'
        
        # Format red flags
        red_flags_html = "<br>".join(r.get("Red Flags", []))
        
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
            "Red Flags": red_flags_html,
            "W_RSI": format_cell(
                metrics.get('W_RSI', {}).get('value', 'N/A'), 
                metrics.get('W_RSI', {}).get('signal', 'neutral')
            ),
            "W_MACD": format_cell(
                metrics.get('W_MACD', {}).get('value', 'N/A'), 
                metrics.get('W_MACD', {}).get('signal', 'neutral')
            ),
            "W_Price": format_cell(
                metrics.get('W_Price', {}).get('value', 'N/A'), 
                metrics.get('W_Price', {}).get('signal', 'neutral')
            ),
            "D_RSI": format_cell(
                metrics.get('D_RSI', {}).get('value', 'N/A'), 
                metrics.get('D_RSI', {}).get('signal', 'neutral')
            ),
            "D_MACD": format_cell(
                metrics.get('D_MACD', {}).get('value', 'N/A'), 
                metrics.get('D_MACD', {}).get('signal', 'neutral')
            ),
            "D_Price": format_cell(
                metrics.get('D_Price', {}).get('value', 'N/A'), 
                metrics.get('D_Price', {}).get('signal', 'neutral')
            ),
            "M_Trend": format_cell(
                metrics.get('M_Trend', {}).get('value', 'N/A'), 
                metrics.get('M_Trend', {}).get('signal', 'neutral')
            ),
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
        ["Ideal Long", "Potential Long", "Caution Long", "Watch Long", 
         "Ideal Short", "Potential Short", "Caution Short", "Watch Short", 
         "None", "Conflict"],
        default=["Ideal Long", "Potential Long", "Caution Long", "Watch Long", 
                "Ideal Short", "Potential Short", "Caution Short", "Watch Short"]
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
    st.title("🎯 Trading Strategy Scanner")
    
    with st.expander("📖 Understanding the Scanner Rules & Color Coding"):
        st.markdown(f"""
        ### How This Scanner Strictly Follows Your Trading Rules
        
        This scanner implements your trading rules with careful attention to the distinction between **MUST** conditions and **PREFER/IDEAL** conditions.
        
        #### For LONG Setups:
        
        **Higher Timeframe MUST Conditions:**
        - Weekly RSI <span class="must-condition">MUST</span> be > 50 (AND preferably above its MA)
        - Weekly MACD <span class="must-condition">MUST</span> be bullish (AND preferably above zero)
        - Weekly Price <span class="must-condition">MUST</span> be above MA band (AND ideally above longer-term MA)
        
        **Lower Timeframe Entry Criteria (need at least {MIN_ENTRY_RULES_MET}):**
        - Daily RSI <span class="must-condition">MUST</span> cross decisively above 50 AND above its MA
        - Daily MACD <span class="must-condition">MUST</span> execute a Golden Cross OR show a hook upwards 
        - Price <span class="must-condition">MUST</span> pull back to and find support at EMAs OR break resistance
        
        #### For SHORT Setups:
        
        **Higher Timeframe MUST Conditions:**
        - Weekly RSI <span class="must-condition">MUST</span> be < 50 (AND preferably below its MA)
        - Weekly MACD <span class="must-condition">MUST</span> be bearish (AND preferably below zero)
        - Weekly Price <span class="must-condition">MUST</span> be below MA band (AND ideally below longer-term MA)
        
        **Lower Timeframe Entry Criteria (need at least {MIN_ENTRY_RULES_MET}):**
        - Daily RSI <span class="must-condition">MUST</span> cross decisively below 50 AND below its MA
        - Daily MACD <span class="must-condition">MUST</span> execute a Death Cross OR show a hook downwards
        - Price <span class="must-condition">MUST</span> rally to and get rejected by EMAs OR break support
        
        #### Color Legend for Metrics:
        - <span class="must-bullish">Dark Green</span>: Ideal bullish condition (MUST + PREFER both met)
        - <span class="caution-bullish">Yellow-Green</span>: Caution bullish (MUST met, PREFER not met) - RED FLAG
        - <span class="must-bearish">Dark Red</span>: Ideal bearish condition (MUST + PREFER both met)
        - <span class="caution-bearish">Yellow-Red</span>: Caution bearish (MUST met, PREFER not met) - RED FLAG
        - <span class="red-flag">Red</span>: Major contradiction or warning
        - <span class="neutral">Gray</span>: Neutral or condition not met
        
        #### Setup Types:
        - <span class="setup-ideal-long">Ideal Long</span>: Perfect long setup with all conditions met
        - <span class="setup-long">Potential Long</span>: Good long setup with some cautions
        - <span class="setup-caution-long">Caution Long</span>: Long setup with multiple red flags
        - <span class="setup-watch">Watch Long</span>: HTF conditions met but LTF not confirmed
        - <span class="setup-ideal-short">Ideal Short</span>: Perfect short setup with all conditions met
        - <span class="setup-short">Potential Short</span>: Good short setup with some cautions
        - <span class="setup-caution-short">Caution Short</span>: Short setup with multiple red flags
        - <span class="setup-watch">Watch Short</span>: HTF conditions met but LTF not confirmed
        - <span class="setup-conflict">Conflict</span>: Contradictory signals
        
        #### RED FLAGS:
        Special attention is given to cases where MUST conditions are met but PREFER conditions are not. For example:
        - RSI > 50 but below its MA (not strongly bullish)
        - MACD > Signal but below zero (not strongly bullish)
        - Monthly timeframe contradictions
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
    st.sidebar.caption(f"Strategy uses {TF_CONDITIONS} and {TF_ENTRY} timeframes, with {TF_MONTHLY} context.")
    
    # Metric explanations
    st.sidebar.markdown("#### Key Indicators Explained:")
    st.sidebar.markdown("""
    - **W_RSI**: Weekly RSI vs its MA
    - **W_MACD**: Weekly MACD vs Signal & Zero
    - **W_Price**: Weekly Price vs EMAs
    - **D_RSI**: Daily RSI cross/position
    - **D_MACD**: Daily MACD cross/hook
    - **D_Price**: Daily pullback/rejection
    - **M_Trend**: Monthly context
    
    **Color code**:
    - Dark Green/Red: Ideal condition
    - Yellow: Caution/RED FLAG
    - Gray: Neutral/Not Met
    """)

    # Main Results Display
    st.header("Scan Results Dashboard")
    
    if not st.session_state.scan_results:
        st.info("Click 'Run Scan' in the sidebar to start.")
    else:
        valid_results = [r for r in st.session_state.scan_results if not r.get('error', True)]
        
        if not valid_results:
            st.warning("Scan complete, but no valid results were found. Try different tickers.")
        else:
            ideal_setups = [r for r in valid_results if r['Setup'].startswith("Ideal")]
            potential_setups = [r for r in valid_results if r['Setup'].startswith("Potential")]
            caution_setups = [r for r in valid_results if r['Setup'].startswith("Caution")]
            watch_setups = [r for r in valid_results if r['Setup'].startswith("Watch")]
            
            setup_counts = {
                "Ideal": len(ideal_setups),
                "Potential": len(potential_setups),
                "Caution": len(caution_setups),
                "Watch": len(watch_setups)
            }
            
            if sum(setup_counts.values()) > 0:
                st.success(f"Scan complete. Found {sum(setup_counts.values())} setups: {setup_counts['Ideal']} Ideal, {setup_counts['Potential']} Potential, {setup_counts['Caution']} Caution, {setup_counts['Watch']} Watch")
            else:
                st.info(f"Scan complete. No active setups found among {len(valid_results)} valid instruments.")
            
            # Display results table with all metrics
            display_results_table(st.session_state.scan_results)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
