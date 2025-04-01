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

# Data Periods
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

# --- Page Config ---
st.set_page_config(
    page_title="Strict Strategy Scanner",
    page_icon="🎯",
    layout="wide",
)

# --- Custom CSS ---
st.markdown("""
<style>
    /* Enhanced setup styling with stronger colors */
    .setup-long { background-color: #28a745; color: white; padding: 4px 8px; border-radius: 4px; font-weight: bold; }
    .setup-short { background-color: #dc3545; color: white; padding: 4px 8px; border-radius: 4px; font-weight: bold; }
    .setup-watch-long { background-color: #5cb85c; color: white; padding: 4px 8px; border-radius: 4px; font-weight: bold; }
    .setup-watch-short { background-color: #d9534f; color: white; padding: 4px 8px; border-radius: 4px; font-weight: bold; }
    .setup-caution { background-color: #ffc107; color: black; padding: 4px 8px; border-radius: 4px; font-weight: bold; }
    .setup-none { background-color: #6c757d; color: white; padding: 4px 8px; border-radius: 4px; font-weight: bold; }
    
    /* Cell colors for metrics */
    .bullish-strong { background-color: rgba(40, 167, 69, 0.4); }
    .bullish { background-color: rgba(40, 167, 69, 0.2); }
    .bearish-strong { background-color: rgba(220, 53, 69, 0.4); }
    .bearish { background-color: rgba(220, 53, 69, 0.2); }
    .neutral { background-color: rgba(108, 117, 125, 0.1); }
    .warning { background-color: rgba(255, 193, 7, 0.3); border: 1px solid #ffc107; }
    
    /* Table styling */
    .dataframe {
        border-collapse: collapse;
        margin: 25px 0;
        font-size: 0.9em;
        font-family: sans-serif;
        width: 100%;
        box-shadow: 0 0 20px rgba(0, 0, 0, 0.15);
    }
    .dataframe thead tr {
        background-color: #009879;
        color: #ffffff;
        text-align: left;
    }
    .dataframe th,
    .dataframe td {
        padding: 12px 15px;
        white-space: nowrap;
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
    
    /* Tooltip style */
    .tooltip {
        position: relative;
        display: inline-block;
        cursor: help;
    }
    .tooltip .tooltiptext {
        visibility: hidden;
        width: 200px;
        background-color: #555;
        color: #fff;
        text-align: center;
        border-radius: 6px;
        padding: 5px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -100px;
        opacity: 0;
        transition: opacity 0.3s;
    }
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
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
        
        min_len_cond = max(EMA_LONG, MACD_SLOW, RSI_WINDOW + RSI_MA_PERIOD) + 10
        min_len_entry = max(EMA_LONG, MACD_SLOW, RSI_WINDOW + RSI_MA_PERIOD) + 10
        
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
        
        # Get more historical data for cross detection
        if len(data_copy) >= 10:
            recent_indices = range(max(0, len(data_copy)-10), len(data_copy))
            
            # Get recent RSI and its MA for cross detection
            recent_rsi = data_copy[f"RSI_{RSI_WINDOW}"].iloc[recent_indices].values
            recent_rsi_ma = data_copy[f"RSI_{RSI_WINDOW}_MA_{RSI_MA_PERIOD}"].iloc[recent_indices].values
            
            # Check for RSI crossing above its MA
            rsi_cross_above_ma = False
            for i in range(1, len(recent_rsi)):
                if recent_rsi[i-1] <= recent_rsi_ma[i-1] and recent_rsi[i] > recent_rsi_ma[i]:
                    rsi_cross_above_ma = True
                    break
            indicators['RSI_Cross_Above_MA'] = rsi_cross_above_ma
            
            # Check for RSI crossing below its MA
            rsi_cross_below_ma = False
            for i in range(1, len(recent_rsi)):
                if recent_rsi[i-1] >= recent_rsi_ma[i-1] and recent_rsi[i] < recent_rsi_ma[i]:
                    rsi_cross_below_ma = True
                    break
            indicators['RSI_Cross_Below_MA'] = rsi_cross_below_ma
            
            # Check for RSI crossing above 50
            rsi_cross_above_50 = False
            for i in range(1, len(recent_rsi)):
                if recent_rsi[i-1] <= 50 and recent_rsi[i] > 50:
                    rsi_cross_above_50 = True
                    break
            indicators['RSI_Cross_Above_50'] = rsi_cross_above_50
            
            # Check for RSI crossing below 50
            rsi_cross_below_50 = False
            for i in range(1, len(recent_rsi)):
                if recent_rsi[i-1] >= 50 and recent_rsi[i] < 50:
                    rsi_cross_below_50 = True
                    break
            indicators['RSI_Cross_Below_50'] = rsi_cross_below_50
            
            # Get recent MACD and Signal for cross detection
            recent_macd = data_copy[f"MACD_{MACD_FAST}_{MACD_SLOW}_{MACD_SIGNAL}"].iloc[recent_indices].values
            recent_signal = data_copy[f"MACDs_{MACD_FAST}_{MACD_SLOW}_{MACD_SIGNAL}"].iloc[recent_indices].values
            
            # Check for MACD Golden Cross (MACD crosses above Signal)
            macd_golden_cross = False
            for i in range(1, len(recent_macd)):
                if recent_macd[i-1] <= recent_signal[i-1] and recent_macd[i] > recent_signal[i]:
                    macd_golden_cross = True
                    break
            indicators['MACD_Golden_Cross'] = macd_golden_cross
            
            # Check for MACD Death Cross (MACD crosses below Signal)
            macd_death_cross = False
            for i in range(1, len(recent_macd)):
                if recent_macd[i-1] >= recent_signal[i-1] and recent_macd[i] < recent_signal[i]:
                    macd_death_cross = True
                    break
            indicators['MACD_Death_Cross'] = macd_death_cross
            
            # Check for MACD hook (changing direction without cross)
            recent_hist = data_copy[f"MACDh_{MACD_FAST}_{MACD_SLOW}_{MACD_SIGNAL}"].iloc[recent_indices].values
            
            # Bullish hook (histogram getting less negative or more positive)
            macd_bullish_hook = False
            if len(recent_hist) >= 3:
                # Check for two consecutive increases in histogram
                if (recent_hist[-3] < recent_hist[-2] < recent_hist[-1]) and recent_hist[-1] < 0:
                    macd_bullish_hook = True
            indicators['MACD_Bullish_Hook'] = macd_bullish_hook
            
            # Bearish hook (histogram getting less positive or more negative)
            macd_bearish_hook = False
            if len(recent_hist) >= 3:
                # Check for two consecutive decreases in histogram
                if (recent_hist[-3] > recent_hist[-2] > recent_hist[-1]) and recent_hist[-1] > 0:
                    macd_bearish_hook = True
            indicators['MACD_Bearish_Hook'] = macd_bearish_hook
            
            # Check for price pullback to MA and finding support (for daily only)
            if timeframe == "daily":
                recent_lows = data_copy['Low'].iloc[recent_indices].values
                recent_close = data_copy['Close'].iloc[recent_indices].values
                recent_ema_short = data_copy[f'EMA_{EMA_SHORT}'].iloc[recent_indices].values
                recent_ema_long = data_copy[f'EMA_{EMA_LONG}'].iloc[recent_indices].values
                
                # Detect pullback to support at EMAs
                pullback_to_ema_support = False
                for i in range(1, len(recent_indices)-1):
                    # Low touches or breaches EMA but Close is above
                    if ((recent_lows[i] <= recent_ema_short[i] and recent_close[i] > recent_ema_short[i]) or
                        (recent_lows[i] <= recent_ema_long[i] and recent_close[i] > recent_ema_long[i])):
                        if recent_close[i+1] > recent_close[i]:  # Next day closes higher (found support)
                            pullback_to_ema_support = True
                            break
                indicators['Pullback_To_EMA_Support'] = pullback_to_ema_support
                
                # Detect rally to resistance at EMAs for shorts
                rally_to_ema_resistance = False
                recent_high = data_copy['High'].iloc[recent_indices].values
                for i in range(1, len(recent_indices)-1):
                    # High touches or breaches EMA but Close is below
                    if ((recent_high[i] >= recent_ema_short[i] and recent_close[i] < recent_ema_short[i]) or
                        (recent_high[i] >= recent_ema_long[i] and recent_close[i] < recent_ema_long[i])):
                        if recent_close[i+1] < recent_close[i]:  # Next day closes lower (rejected at resistance)
                            rally_to_ema_resistance = True
                            break
                indicators['Rally_To_EMA_Resistance'] = rally_to_ema_resistance
        else:
            # Default values if not enough data points
            indicators['RSI_Cross_Above_MA'] = False
            indicators['RSI_Cross_Below_MA'] = False
            indicators['RSI_Cross_Above_50'] = False
            indicators['RSI_Cross_Below_50'] = False
            indicators['MACD_Golden_Cross'] = False
            indicators['MACD_Death_Cross'] = False
            indicators['MACD_Bullish_Hook'] = False
            indicators['MACD_Bearish_Hook'] = False
            indicators['Pullback_To_EMA_Support'] = False
            indicators['Rally_To_EMA_Resistance'] = False
        
        # --- Basic derived boolean states ---
        indicators['RSI_Value'] = round(indicators[f'RSI_{RSI_WINDOW}'], 1)
        indicators['RSI_MA_Value'] = round(indicators[f'RSI_{RSI_WINDOW}_MA'], 1)
        indicators['RSI_Above_50'] = indicators[f'RSI_{RSI_WINDOW}'] > RSI_MID
        indicators['RSI_Below_50'] = indicators[f'RSI_{RSI_WINDOW}'] < RSI_MID
        indicators['RSI_Above_MA'] = indicators[f'RSI_{RSI_WINDOW}'] > indicators[f'RSI_{RSI_WINDOW}_MA']
        indicators['RSI_Below_MA'] = indicators[f'RSI_{RSI_WINDOW}'] < indicators[f'RSI_{RSI_WINDOW}_MA']
        
        # MACD States
        indicators['MACD_Above_Signal'] = indicators['MACD_Line'] > indicators['MACD_Signal']
        indicators['MACD_Below_Signal'] = indicators['MACD_Line'] < indicators['MACD_Signal']
        indicators['MACD_Above_Zero'] = indicators['MACD_Line'] > 0
        indicators['MACD_Below_Zero'] = indicators['MACD_Line'] < 0
        
        # Price Structure
        indicators['Price_Above_EMA_Short'] = indicators['Close'] > indicators[f'EMA_{EMA_SHORT}']
        indicators['Price_Above_EMA_Long'] = indicators['Close'] > indicators[f'EMA_{EMA_LONG}']
        indicators['Price_Above_EMA_Context'] = indicators['Close'] > indicators[f'EMA_{EMA_CONTEXT}']
        indicators['Price_Below_EMA_Short'] = indicators['Close'] < indicators[f'EMA_{EMA_SHORT}']
        indicators['Price_Below_EMA_Long'] = indicators['Close'] < indicators[f'EMA_{EMA_LONG}']
        indicators['Price_Below_EMA_Context'] = indicators['Close'] < indicators[f'EMA_{EMA_CONTEXT}']
        
        # EMA relationships (cloud)
        indicators['EMA_Band_Bullish'] = indicators[f'EMA_{EMA_SHORT}'] > indicators[f'EMA_{EMA_LONG}']
        indicators['EMA_Band_Bearish'] = indicators[f'EMA_{EMA_SHORT}'] < indicators[f'EMA_{EMA_LONG}']
        
        return indicators, data_copy
    except Exception as e:
        st.error(f"Error calculating indicators for {timeframe}: {str(e)}")
        return None, None


def check_strategy_setup(weekly_indicators, daily_indicators, monthly_indicators=None):
    if not weekly_indicators or not daily_indicators: 
        return "Error", 0, [], {}, {}
    
    setup_type = "None"
    score = 0
    rules_met = []
    rule_details = {}  # Detailed rule checking results
    metrics = {}  # For display
    
    # --- Process metrics for display ---
    
    # Weekly RSI
    w_rsi_signal = 'neutral'
    w_rsi_desc = 'Neutral'
    if weekly_indicators['RSI_Above_50'] and weekly_indicators['RSI_Above_MA']:
        w_rsi_signal = 'bullish-strong'
        w_rsi_desc = 'RSI>50 & >MA (✓)'
    elif weekly_indicators['RSI_Above_50'] and not weekly_indicators['RSI_Above_MA']:
        w_rsi_signal = 'warning'
        w_rsi_desc = 'RSI>50 but <MA (⚠️)'
    elif weekly_indicators['RSI_Below_50'] and weekly_indicators['RSI_Below_MA']:
        w_rsi_signal = 'bearish-strong'
        w_rsi_desc = 'RSI<50 & <MA (✓)'
    elif weekly_indicators['RSI_Below_50'] and not weekly_indicators['RSI_Below_MA']:
        w_rsi_signal = 'warning'
        w_rsi_desc = 'RSI<50 but >MA (⚠️)'
    
    metrics['W_RSI'] = {
        'value': f"{weekly_indicators['RSI_Value']} vs MA: {weekly_indicators['RSI_MA_Value']}",
        'signal': w_rsi_signal,
        'desc': w_rsi_desc
    }
    
    # Weekly MACD
    w_macd_signal = 'neutral'
    w_macd_desc = 'Neutral'
    
    if weekly_indicators['MACD_Above_Signal']:
        if weekly_indicators['MACD_Golden_Cross'] or weekly_indicators['MACD_Above_Zero']:
            w_macd_signal = 'bullish-strong'
            w_macd_desc = 'MACD>Signal & >0 or Cross (✓)'
        else:
            w_macd_signal = 'bullish'
            w_macd_desc = 'MACD>Signal but <0 (✓)'
    elif weekly_indicators['MACD_Below_Signal']:
        if weekly_indicators['MACD_Death_Cross'] or weekly_indicators['MACD_Below_Zero']:
            w_macd_signal = 'bearish-strong'
            w_macd_desc = 'MACD<Signal & <0 or Cross (✓)'
        else:
            w_macd_signal = 'bearish'
            w_macd_desc = 'MACD<Signal but >0 (✓)'
    
    metrics['W_MACD'] = {
        'value': f"{weekly_indicators['MACD_Line']:.3f} vs {weekly_indicators['MACD_Signal']:.3f} ({'+' if weekly_indicators['MACD_Line'] > 0 else ''}{weekly_indicators['MACD_Line']:.3f})",
        'signal': w_macd_signal,
        'desc': w_macd_desc
    }
    
    # Weekly Price Structure
    w_price_signal = 'neutral'
    w_price_desc = 'Mixed'
    
    if weekly_indicators['Price_Above_EMA_Short'] and weekly_indicators['Price_Above_EMA_Long']:
        if weekly_indicators['Price_Above_EMA_Context']:
            w_price_signal = 'bullish-strong'
            w_price_desc = f"Price > EMA{EMA_SHORT}/{EMA_LONG}/{EMA_CONTEXT} (✓)"
        else:
            w_price_signal = 'bullish'
            w_price_desc = f"Price > EMA{EMA_SHORT}/{EMA_LONG} (✓)"
    elif weekly_indicators['Price_Below_EMA_Short'] and weekly_indicators['Price_Below_EMA_Long']:
        if weekly_indicators['Price_Below_EMA_Context']:
            w_price_signal = 'bearish-strong'
            w_price_desc = f"Price < EMA{EMA_SHORT}/{EMA_LONG}/{EMA_CONTEXT} (✓)"
        else:
            w_price_signal = 'bearish'
            w_price_desc = f"Price < EMA{EMA_SHORT}/{EMA_LONG} (✓)"
    
    metrics['W_Price'] = {
        'value': f"{weekly_indicators['Close']:.2f} vs {weekly_indicators[f'EMA_{EMA_SHORT}']:.2f}/{weekly_indicators[f'EMA_{EMA_LONG}']:.2f}/{weekly_indicators[f'EMA_{EMA_CONTEXT}']:.2f}",
        'signal': w_price_signal,
        'desc': w_price_desc
    }
    
    # Daily RSI
    d_rsi_signal = 'neutral'
    d_rsi_desc = 'Neutral'
    
    if daily_indicators['RSI_Above_50'] and daily_indicators['RSI_Above_MA']:
        if daily_indicators['RSI_Cross_Above_50'] or daily_indicators['RSI_Cross_Above_MA']:
            d_rsi_signal = 'bullish-strong'
            d_rsi_desc = 'RSI>50 & >MA with recent cross (✓)'
        else:
            d_rsi_signal = 'bullish'
            d_rsi_desc = 'RSI>50 & >MA (✓)'
    elif daily_indicators['RSI_Below_50'] and daily_indicators['RSI_Below_MA']:
        if daily_indicators['RSI_Cross_Below_50'] or daily_indicators['RSI_Cross_Below_MA']:
            d_rsi_signal = 'bearish-strong'
            d_rsi_desc = 'RSI<50 & <MA with recent cross (✓)'
        else:
            d_rsi_signal = 'bearish'
            d_rsi_desc = 'RSI<50 & <MA (✓)'
    elif daily_indicators['RSI_Above_50'] and not daily_indicators['RSI_Above_MA']:
        d_rsi_signal = 'warning'
        d_rsi_desc = 'RSI>50 but <MA (⚠️)'
    elif daily_indicators['RSI_Below_50'] and not daily_indicators['RSI_Below_MA']:
        d_rsi_signal = 'warning'
        d_rsi_desc = 'RSI<50 but >MA (⚠️)'
    
    metrics['D_RSI'] = {
        'value': f"{daily_indicators['RSI_Value']} vs MA: {daily_indicators['RSI_MA_Value']}",
        'signal': d_rsi_signal,
        'desc': d_rsi_desc
    }
    
    # Daily MACD
    d_macd_signal = 'neutral'
    d_macd_desc = 'Neutral'
    
    if daily_indicators['MACD_Golden_Cross']:
        d_macd_signal = 'bullish-strong'
        d_macd_desc = 'Recent MACD Golden Cross (✓)'
    elif daily_indicators['MACD_Death_Cross']:
        d_macd_signal = 'bearish-strong'
        d_macd_desc = 'Recent MACD Death Cross (✓)'
    elif daily_indicators['MACD_Above_Signal']:
        if daily_indicators['MACD_Above_Zero'] or daily_indicators['MACD_Bullish_Hook']:
            d_macd_signal = 'bullish-strong'
            d_macd_desc = 'MACD>Signal & >0 or Hook Up (✓)'
        else:
            d_macd_signal = 'bullish'
            d_macd_desc = 'MACD>Signal but <0 (✓)'
    elif daily_indicators['MACD_Below_Signal']:
        if daily_indicators['MACD_Below_Zero'] or daily_indicators['MACD_Bearish_Hook']:
            d_macd_signal = 'bearish-strong'
            d_macd_desc = 'MACD<Signal & <0 or Hook Down (✓)'
        else:
            d_macd_signal = 'bearish'
            d_macd_desc = 'MACD<Signal but >0 (✓)'
    
    metrics['D_MACD'] = {
        'value': f"{daily_indicators['MACD_Line']:.3f} vs {daily_indicators['MACD_Signal']:.3f} ({'+' if daily_indicators['MACD_Line'] > 0 else ''}{daily_indicators['MACD_Line']:.3f})",
        'signal': d_macd_signal,
        'desc': d_macd_desc
    }
    
    # Daily Price Structure
    d_price_signal = 'neutral'
    d_price_desc = 'Mixed'
    
    if daily_indicators['Price_Above_EMA_Short'] and daily_indicators['Price_Above_EMA_Long']:
        if daily_indicators['Pullback_To_EMA_Support']:
            d_price_signal = 'bullish-strong'
            d_price_desc = f"Price>EMAs with recent pullback support (✓✓)"
        else:
            d_price_signal = 'bullish'
            d_price_desc = f"Price>EMAs (✓)"
    elif daily_indicators['Price_Below_EMA_Short'] and daily_indicators['Price_Below_EMA_Long']:
        if daily_indicators['Rally_To_EMA_Resistance']:
            d_price_signal = 'bearish-strong'
            d_price_desc = f"Price<EMAs with recent rally rejection (✓✓)"
        else:
            d_price_signal = 'bearish'
            d_price_desc = f"Price<EMAs (✓)"
    
    metrics['D_Price'] = {
        'value': f"{daily_indicators['Close']:.2f} vs {daily_indicators[f'EMA_{EMA_SHORT}']:.2f}/{daily_indicators[f'EMA_{EMA_LONG}']:.2f}",
        'signal': d_price_signal,
        'desc': d_price_desc
    }
    
    # Monthly context
    m_signal = 'neutral'
    m_desc = 'N/A'
    
    if monthly_indicators:
        if monthly_indicators['RSI_Above_50']:
            if monthly_indicators['RSI_Value'] > 60:
                m_signal = 'bullish-strong'
                m_desc = f"Monthly RSI: {monthly_indicators['RSI_Value']} (Strong Bullish)"
            else:
                m_signal = 'bullish'
                m_desc = f"Monthly RSI: {monthly_indicators['RSI_Value']} (Bullish)"
        elif monthly_indicators['RSI_Below_50']:
            if monthly_indicators['RSI_Value'] < 40:
                m_signal = 'bearish-strong'
                m_desc = f"Monthly RSI: {monthly_indicators['RSI_Value']} (Strong Bearish)"
            else:
                m_signal = 'bearish'
                m_desc = f"Monthly RSI: {monthly_indicators['RSI_Value']} (Bearish)"
    
    metrics['M_Trend'] = {
        'value': monthly_indicators['RSI_Value'] if monthly_indicators else 'N/A',
        'signal': m_signal,
        'desc': m_desc
    }
    
    # --- Check LONG Setup Conditions ---
    
    # Rule 1.1 - Weekly RSI Trend
    rule_details['W_RSI_Long'] = {
        'name': 'Weekly RSI > 50 AND preferably > MA',
        'status': weekly_indicators['RSI_Above_50'] and weekly_indicators['RSI_Above_MA'],
        'details': f"RSI: {weekly_indicators['RSI_Value']:.1f}, MA: {weekly_indicators['RSI_MA_Value']:.1f}",
        'critical': True  # This rule is mandatory
    }
    
    # Rule 1.2 - Weekly MACD Trend
    w_macd_long_ok = (weekly_indicators['MACD_Golden_Cross'] or 
                      (weekly_indicators['MACD_Above_Signal'] and 
                       (weekly_indicators['MACD_Above_Zero'] or weekly_indicators['MACD_Bullish_Hook'])))
                       
    rule_details['W_MACD_Long'] = {
        'name': 'Weekly MACD Golden Cross OR > Signal',
        'status': w_macd_long_ok,
        'details': f"MACD: {weekly_indicators['MACD_Line']:.3f}, Signal: {weekly_indicators['MACD_Signal']:.3f}",
        'critical': True  # This rule is mandatory
    }
    
    # Rule 1.3 - Weekly Price Structure
    w_price_long_ok = weekly_indicators['Price_Above_EMA_Short'] and weekly_indicators['Price_Above_EMA_Long']
    w_price_long_context_ok = weekly_indicators['Price_Above_EMA_Context']
    
    rule_details['W_Price_Long'] = {
        'name': f"Weekly Price > EMA{EMA_SHORT}/{EMA_LONG} (ideally > EMA{EMA_CONTEXT})",
        'status': w_price_long_ok,
        'details': f"Price: {weekly_indicators['Close']:.2f}, EMAs: {weekly_indicators[f'EMA_{EMA_SHORT}']:.2f}/{weekly_indicators[f'EMA_{EMA_LONG}']:.2f}/{weekly_indicators[f'EMA_{EMA_CONTEXT}']:.2f}",
        'critical': True,  # This rule is mandatory
        'context_ok': w_price_long_context_ok  # Additional positive factor
    }
    
    # Rule 1.4 - Monthly Check
    monthly_contradicts_long = False
    if monthly_indicators:
        monthly_contradicts_long = monthly_indicators['RSI_Below_50'] and monthly_indicators['RSI_Value'] < 40
    
    rule_details['M_Check_Long'] = {
        'name': 'Monthly RSI Check',
        'status': not monthly_contradicts_long,
        'details': f"Monthly RSI: {monthly_indicators['RSI_Value'] if monthly_indicators else 'N/A'}",
        'critical': False  # Optional rule
    }
    
    # Check if all HTF conditions are met for LONG
    long_htf_conditions_met = (
        rule_details['W_RSI_Long']['status'] and
        rule_details['W_MACD_Long']['status'] and
        rule_details['W_Price_Long']['status']
    )
    
    # Only check LTF conditions if HTF conditions are met
    long_ltf_rules_met = 0
    long_ltf_total_possible = 3  # Number of LTF rules that can be met
    
    if long_htf_conditions_met:
        # Rule 2.1 - LTF RSI Confirmation
        rule_details['D_RSI_Long'] = {
            'name': 'Daily RSI Cross > 50 AND > MA',
            'status': daily_indicators['RSI_Above_50'] and daily_indicators['RSI_Above_MA'],
            'strong': daily_indicators['RSI_Cross_Above_50'] or daily_indicators['RSI_Cross_Above_MA'],
            'details': f"RSI: {daily_indicators['RSI_Value']:.1f}, MA: {daily_indicators['RSI_MA_Value']:.1f}",
            'critical': False
        }
        
        if rule_details['D_RSI_Long']['status']:
            long_ltf_rules_met += 1
        
        # Rule 2.2 - LTF MACD Confirmation
        d_macd_long_ok = (daily_indicators['MACD_Golden_Cross'] or 
                         (daily_indicators['MACD_Above_Signal'] and 
                          (daily_indicators['MACD_Above_Zero'] or daily_indicators['MACD_Bullish_Hook'])))
                          
        rule_details['D_MACD_Long'] = {
            'name': 'Daily MACD Golden Cross OR Bullish',
            'status': d_macd_long_ok,
            'strong': daily_indicators['MACD_Golden_Cross'] or daily_indicators['MACD_Bullish_Hook'],
            'details': f"MACD: {daily_indicators['MACD_Line']:.3f}, Signal: {daily_indicators['MACD_Signal']:.3f}",
            'critical': False
        }
        
        if rule_details['D_MACD_Long']['status']:
            long_ltf_rules_met += 1
        
        # Rule 2.3 - LTF Price Action
        d_price_long_ok = daily_indicators['Price_Above_EMA_Short'] and daily_indicators['Price_Above_EMA_Long']
        d_price_pullback_ok = daily_indicators['Pullback_To_EMA_Support']
        
        rule_details['D_Price_Long'] = {
            'name': f"Daily Price > EMA{EMA_SHORT}/{EMA_LONG} (ideally with pullback)",
            'status': d_price_long_ok,
            'strong': d_price_pullback_ok,
            'details': f"Price: {daily_indicators['Close']:.2f}, EMAs: {daily_indicators[f'EMA_{EMA_SHORT}']:.2f}/{daily_indicators[f'EMA_{EMA_LONG}']:.2f}",
            'critical': False
        }
        
        if rule_details['D_Price_Long']['status']:
            long_ltf_rules_met += 1
        
        # Add Strong HTF rules met to the count as bonus points
        long_htf_bonus = 0
        if rule_details['W_Price_Long'].get('context_ok', False):
            long_htf_bonus += 1
            
        # Determine Long Setup quality
        if long_ltf_rules_met >= 2:  # Need at least 2 LTF rules for a potential setup
            if monthly_contradicts_long:
                setup_type = "Caution Long"
                score = long_ltf_rules_met + 3 - 1  # Penalty for Monthly contradiction
            else:
                setup_type = "Potential Long"
                score = long_ltf_rules_met + 3 + long_htf_bonus
            
            # Compile rules met
            rules_met.append("W:RSI>50 & >MA")
            rules_met.append("W:MACD Bullish")
            rules_met.append("W:Price>EMAs")
            
            if rule_details['W_Price_Long'].get('context_ok', False):
                rules_met.append("W:Price>EMA50 (Bonus)")
                
            if rule_details['D_RSI_Long']['status']:
                if rule_details['D_RSI_Long'].get('strong', False):
                    rules_met.append("D:RSI Cross >50 & >MA")
                else:
                    rules_met.append("D:RSI>50 & >MA")
                    
            if rule_details['D_MACD_Long']['status']:
                if rule_details['D_MACD_Long'].get('strong', False):
                    rules_met.append("D:MACD Golden Cross/Hook")
                else:
                    rules_met.append("D:MACD Bullish")
                    
            if rule_details['D_Price_Long']['status']:
                if rule_details['D_Price_Long'].get('strong', False):
                    rules_met.append("D:Pullback Support at EMAs")
                else:
                    rules_met.append("D:Price>EMAs")
                    
            if monthly_contradicts_long:
                rules_met.append("M:WARNING-RSI<40")
        elif long_ltf_rules_met > 0:  # At least one LTF rule met
            setup_type = "Watch Long"
            score = long_ltf_rules_met + 3  # Less weight for Watch setups
            
            # Compile basic rules met
            rules_met.append("W:RSI>50 & >MA")
            rules_met.append("W:MACD Bullish")
            rules_met.append("W:Price>EMAs")
            
            if rule_details['D_RSI_Long']['status']:
                rules_met.append("D:RSI>50 & >MA")
            if rule_details['D_MACD_Long']['status']:
                rules_met.append("D:MACD Bullish")
            if rule_details['D_Price_Long']['status']:
                rules_met.append("D:Price>EMAs")
    
    # --- Check SHORT Setup Conditions ---
    
    # Rule 1.1 - Weekly RSI Trend (Short)
    rule_details['W_RSI_Short'] = {
        'name': 'Weekly RSI < 50 AND preferably < MA',
        'status': weekly_indicators['RSI_Below_50'] and weekly_indicators['RSI_Below_MA'],
        'details': f"RSI: {weekly_indicators['RSI_Value']:.1f}, MA: {weekly_indicators['RSI_MA_Value']:.1f}",
        'critical': True  # This rule is mandatory
    }
    
    # Rule 1.2 - Weekly MACD Trend (Short)
    w_macd_short_ok = (weekly_indicators['MACD_Death_Cross'] or 
                       (weekly_indicators['MACD_Below_Signal'] and 
                        (weekly_indicators['MACD_Below_Zero'] or weekly_indicators['MACD_Bearish_Hook'])))
                       
    rule_details['W_MACD_Short'] = {
        'name': 'Weekly MACD Death Cross OR < Signal',
        'status': w_macd_short_ok,
        'details': f"MACD: {weekly_indicators['MACD_Line']:.3f}, Signal: {weekly_indicators['MACD_Signal']:.3f}",
        'critical': True  # This rule is mandatory
    }
    
    # Rule 1.3 - Weekly Price Structure (Short)
    w_price_short_ok = weekly_indicators['Price_Below_EMA_Short'] and weekly_indicators['Price_Below_EMA_Long']
    w_price_short_context_ok = weekly_indicators['Price_Below_EMA_Context']
    
    rule_details['W_Price_Short'] = {
        'name': f"Weekly Price < EMA{EMA_SHORT}/{EMA_LONG} (ideally < EMA{EMA_CONTEXT})",
        'status': w_price_short_ok,
        'details': f"Price: {weekly_indicators['Close']:.2f}, EMAs: {weekly_indicators[f'EMA_{EMA_SHORT}']:.2f}/{weekly_indicators[f'EMA_{EMA_LONG}']:.2f}/{weekly_indicators[f'EMA_{EMA_CONTEXT}']:.2f}",
        'critical': True,  # This rule is mandatory
        'context_ok': w_price_short_context_ok  # Additional positive factor
    }
    
    # Rule 1.4 - Monthly Check (Short)
    monthly_contradicts_short = False
    if monthly_indicators:
        monthly_contradicts_short = monthly_indicators['RSI_Above_50'] and monthly_indicators['RSI_Value'] > 60
    
    rule_details['M_Check_Short'] = {
        'name': 'Monthly RSI Check',
        'status': not monthly_contradicts_short,
        'details': f"Monthly RSI: {monthly_indicators['RSI_Value'] if monthly_indicators else 'N/A'}",
        'critical': False  # Optional rule
    }
    
    # Check if all HTF conditions are met for SHORT
    short_htf_conditions_met = (
        rule_details['W_RSI_Short']['status'] and
        rule_details['W_MACD_Short']['status'] and
        rule_details['W_Price_Short']['status']
    )
    
    # Only check LTF conditions if HTF conditions are met
    short_ltf_rules_met = 0
    short_ltf_total_possible = 3  # Number of LTF rules that can be met
    
    if short_htf_conditions_met:
        # Rule 2.1 - LTF RSI Confirmation (Short)
        rule_details['D_RSI_Short'] = {
            'name': 'Daily RSI < 50 AND < MA',
            'status': daily_indicators['RSI_Below_50'] and daily_indicators['RSI_Below_MA'],
            'strong': daily_indicators['RSI_Cross_Below_50'] or daily_indicators['RSI_Cross_Below_MA'],
            'details': f"RSI: {daily_indicators['RSI_Value']:.1f}, MA: {daily_indicators['RSI_MA_Value']:.1f}",
            'critical': False
        }
        
        if rule_details['D_RSI_Short']['status']:
            short_ltf_rules_met += 1
        
        # Rule 2.2 - LTF MACD Confirmation (Short)
        d_macd_short_ok = (daily_indicators['MACD_Death_Cross'] or 
                          (daily_indicators['MACD_Below_Signal'] and 
                           (daily_indicators['MACD_Below_Zero'] or daily_indicators['MACD_Bearish_Hook'])))
                           
        rule_details['D_MACD_Short'] = {
            'name': 'Daily MACD Death Cross OR Bearish',
            'status': d_macd_short_ok,
            'strong': daily_indicators['MACD_Death_Cross'] or daily_indicators['MACD_Bearish_Hook'],
            'details': f"MACD: {daily_indicators['MACD_Line']:.3f}, Signal: {daily_indicators['MACD_Signal']:.3f}",
            'critical': False
        }
        
        if rule_details['D_MACD_Short']['status']:
            short_ltf_rules_met += 1
        
        # Rule 2.3 - LTF Price Action (Short)
        d_price_short_ok = daily_indicators['Price_Below_EMA_Short'] and daily_indicators['Price_Below_EMA_Long']
        d_price_rally_ok = daily_indicators['Rally_To_EMA_Resistance']
        
        rule_details['D_Price_Short'] = {
            'name': f"Daily Price < EMA{EMA_SHORT}/{EMA_LONG} (ideally with rally rejection)",
            'status': d_price_short_ok,
            'strong': d_price_rally_ok,
            'details': f"Price: {daily_indicators['Close']:.2f}, EMAs: {daily_indicators[f'EMA_{EMA_SHORT}']:.2f}/{daily_indicators[f'EMA_{EMA_LONG}']:.2f}",
            'critical': False
        }
        
        if rule_details['D_Price_Short']['status']:
            short_ltf_rules_met += 1
        
        # Add Strong HTF rules met to the count as bonus points
        short_htf_bonus = 0
        if rule_details['W_Price_Short'].get('context_ok', False):
            short_htf_bonus += 1
            
        # Determine Short Setup quality
        if short_ltf_rules_met >= 2:  # Need at least 2 LTF rules for a potential setup
            if monthly_contradicts_short:
                setup_type = "Caution Short"
                score = -(short_ltf_rules_met + 3 - 1)  # Negative score for shorts with penalty
            else:
                setup_type = "Potential Short"
                score = -(short_ltf_rules_met + 3 + short_htf_bonus)  # Negative score for shorts
            
            # Compile rules met
            rules_met.append("W:RSI<50 & <MA")
            rules_met.append("W:MACD Bearish")
            rules_met.append("W:Price<EMAs")
            
            if rule_details['W_Price_Short'].get('context_ok', False):
                rules_met.append("W:Price<EMA50 (Bonus)")
                
            if rule_details['D_RSI_Short']['status']:
                if rule_details['D_RSI_Short'].get('strong', False):
                    rules_met.append("D:RSI Cross <50 & <MA")
                else:
                    rules_met.append("D:RSI<50 & <MA")
                    
            if rule_details['D_MACD_Short']['status']:
                if rule_details['D_MACD_Short'].get('strong', False):
                    rules_met.append("D:MACD Death Cross/Hook")
                else:
                    rules_met.append("D:MACD Bearish")
                    
            if rule_details['D_Price_Short']['status']:
                if rule_details['D_Price_Short'].get('strong', False):
                    rules_met.append("D:Rally Rejection at EMAs")
                else:
                    rules_met.append("D:Price<EMAs")
                    
            if monthly_contradicts_short:
                rules_met.append("M:WARNING-RSI>60")
        elif short_ltf_rules_met > 0:  # At least one LTF rule met
            setup_type = "Watch Short"
            score = -(short_ltf_rules_met + 3)  # Less weight for Watch setups
            
            # Compile basic rules met
            rules_met.append("W:RSI<50 & <MA")
            rules_met.append("W:MACD Bearish")
            rules_met.append("W:Price<EMAs")
            
            if rule_details['D_RSI_Short']['status']:
                rules_met.append("D:RSI<50 & <MA")
            if rule_details['D_MACD_Short']['status']:
                rules_met.append("D:MACD Bearish")
            if rule_details['D_Price_Short']['status']:
                rules_met.append("D:Price<EMAs")
    
    # Handle potential conflict if both long and short HTF conditions are somehow met
    if long_htf_conditions_met and short_htf_conditions_met:
        setup_type = "Conflicting"
        score = 0
        rules_met = ["Conflicting Signals"]
    
    return setup_type, score, rules_met, metrics, rule_details


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
                        "error": True,
                        "metrics": {},
                        "rule_details": {}
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
                        "error": True,
                        "metrics": {},
                        "rule_details": {}
                    })
                    continue
                    
                setup_type, setup_score, rules_met, all_metrics, rule_details = check_strategy_setup(
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
                    "Rules Met": ", ".join(rules_met), 
                    "error": False,
                    "metrics": all_metrics,
                    "rule_details": rule_details
                })
                
                # Small delay to prevent API rate limits
                time.sleep(0.1)
                
            except Exception as e:
                results.append({
                    "ticker": ticker, 
                    "name": name, 
                    "Setup": "Error", 
                    "Score": 0, 
                    "Rules Met": [f"Error: {str(e)}"], 
                    "error": True,
                    "metrics": {},
                    "rule_details": {}
                })
    
    except Exception as e:
        st.error(f"Error during scanning: {str(e)}")
    finally:
        status_text.text(f"Scan Complete: {len(results)} tickers analyzed.")
        
    return results


def format_cell(value, signal_type):
    """Format a table cell with appropriate styling based on signal type"""
    if signal_type == 'bullish-strong':
        return f'<span class="bullish-strong">{value}</span>'
    elif signal_type == 'bullish':
        return f'<span class="bullish">{value}</span>'
    elif signal_type == 'bearish-strong':
        return f'<span class="bearish-strong">{value}</span>'
    elif signal_type == 'bearish':
        return f'<span class="bearish">{value}</span>'
    elif signal_type == 'warning':
        return f'<span class="warning">{value}</span>'
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
        if "Potential Long" in r['Setup']: setup_class = "setup-long"
        elif "Watch Long" in r['Setup']: setup_class = "setup-watch-long"
        elif "Potential Short" in r['Setup']: setup_class = "setup-short"
        elif "Watch Short" in r['Setup']: setup_class = "setup-watch-short"
        elif "Caution" in r['Setup']: setup_class = "setup-caution"
        
        # Format setup with HTML for styling
        setup_html = f'<span class="{setup_class}">{r["Setup"]}</span>'
        
        # Get metrics from result
        metrics = r.get('metrics', {})
        
        # Create a row with all metrics - with ticker included for CSV export
        row_data = {
            "Name": r["name"],
            "Ticker": r["ticker"],  # Include actual ticker for CSV export
            "_ticker": r["ticker"],  # Hidden column for filtering/selection
            "Price": r["Price"],
            "Last Update": r["Last Date"],
            "Setup": setup_html,
            "Setup_plain": r["Setup"],  # Plain text version for CSV export
            "Score": r["Score"],
            "Weekly RSI": metrics.get('W_RSI', {}).get('value', 'N/A'),
            "Weekly MACD": metrics.get('W_MACD', {}).get('value', 'N/A'),
            "Weekly Price": metrics.get('W_Price', {}).get('value', 'N/A'),
            "Daily RSI": metrics.get('D_RSI', {}).get('value', 'N/A'),
            "Daily MACD": metrics.get('D_MACD', {}).get('value', 'N/A'),
            "Daily Price": metrics.get('D_Price', {}).get('value', 'N/A'),
            "Monthly Trend": metrics.get('M_Trend', {}).get('value', 'N/A'),
            "Rules Met": r["Rules Met"],
        }
        
        # Add HTML formatted versions for display
        row_data["Weekly RSI_html"] = format_cell(
            metrics.get('W_RSI', {}).get('value', 'N/A'), 
            metrics.get('W_RSI', {}).get('signal', 'neutral')
        )
        row_data["Weekly MACD_html"] = format_cell(
            metrics.get('W_MACD', {}).get('value', 'N/A'), 
            metrics.get('W_MACD', {}).get('signal', 'neutral')
        )
        row_data["Weekly Price_html"] = format_cell(
            metrics.get('W_Price', {}).get('value', 'N/A'), 
            metrics.get('W_Price', {}).get('signal', 'neutral')
        )
        row_data["Daily RSI_html"] = format_cell(
            metrics.get('D_RSI', {}).get('value', 'N/A'), 
            metrics.get('D_RSI', {}).get('signal', 'neutral')
        )
        row_data["Daily MACD_html"] = format_cell(
            metrics.get('D_MACD', {}).get('value', 'N/A'), 
            metrics.get('D_MACD', {}).get('signal', 'neutral')
        )
        row_data["Daily Price_html"] = format_cell(
            metrics.get('D_Price', {}).get('value', 'N/A'), 
            metrics.get('D_Price', {}).get('signal', 'neutral')
        )
        row_data["Monthly Trend_html"] = format_cell(
            metrics.get('M_Trend', {}).get('value', 'N/A'), 
            metrics.get('M_Trend', {}).get('signal', 'neutral')
        )
        
        df_data.append(row_data)

    if not df_data:
        st.info("No valid results to display after filtering.")
        return None

    # Create DataFrame
    df_display = pd.DataFrame(df_data)
    
    # Add filter for setup types with checkboxes for better UX
    st.subheader("Filter Results")
    cols = st.columns(4)
    with cols[0]:
        show_potential_long = st.checkbox("Potential Long", value=True)
    with cols[1]:
        show_watch_long = st.checkbox("Watch Long", value=True)
    with cols[0]:
        show_caution_long = st.checkbox("Caution Long", value=True)
    with cols[2]:
        show_potential_short = st.checkbox("Potential Short", value=True)
    with cols[3]:
        show_watch_short = st.checkbox("Watch Short", value=True)
    with cols[2]:
        show_caution_short = st.checkbox("Caution Short", value=True)
    with cols[3]:
        show_none = st.checkbox("None", value=False)
    
    # Prepare filter based on selections
    setup_filter = []
    if show_potential_long: setup_filter.append("Potential Long")
    if show_watch_long: setup_filter.append("Watch Long")
    if show_caution_long: setup_filter.append("Caution Long")
    if show_potential_short: setup_filter.append("Potential Short")
    if show_watch_short: setup_filter.append("Watch Short")
    if show_caution_short: setup_filter.append("Caution Short")
    if show_none: setup_filter.append("None")
    
    if setup_filter:
        # Extract setup type from HTML for filtering
        setup_series = df_display["Setup"].str.extract(r'>([^<]+)<')
        filtered_df = df_display[setup_series[0].isin(setup_filter)]
    else:
        filtered_df = df_display
        
    if filtered_df.empty:
        st.info("No results match the selected filter criteria.")
        return None
    
    # Sort options 
    sort_option = st.radio(
        "Sort by:",
        ["Score (Descending)", "Score (Ascending)", "Name"],
        horizontal=True
    )
    
    if sort_option == "Score (Descending)":
        filtered_df = filtered_df.sort_values(by="Score", ascending=False)
    elif sort_option == "Score (Ascending)":
        filtered_df = filtered_df.sort_values(by="Score", ascending=True)
    elif sort_option == "Name":
        filtered_df = filtered_df.sort_values(by="Name")
    
    filtered_df = filtered_df.reset_index(drop=True)
    
    # Add download button for CSV export
    current_date = datetime.now().strftime('%Y-%m-%d')
    csv_filename = f"strategy_scanner_results_{current_date}.csv"
    
    # Create a CSV export version of the dataframe (without HTML formatting)
    export_columns = [
        "Name", "Ticker", "Price", "Last Update", "Setup_plain", "Score", 
        "Weekly RSI", "Weekly MACD", "Weekly Price", 
        "Daily RSI", "Daily MACD", "Daily Price", "Monthly Trend",
        "Rules Met"
    ]
    export_df = filtered_df[export_columns].copy()
    export_df.rename(columns={"Setup_plain": "Setup"}, inplace=True)
    
    # Add download button
    st.download_button(
        label="📥 Download Results as CSV",
        data=export_df.to_csv(index=False),
        file_name=csv_filename,
        mime="text/csv",
    )
    
    # Create display dataframe with HTML formatted columns
    display_columns = [
        "Name", "Price", "Last Update", "Setup", "Score", 
        "Weekly RSI_html", "Weekly MACD_html", "Weekly Price_html", 
        "Daily RSI_html", "Daily MACD_html", "Daily Price_html", "Monthly Trend_html"
    ]
    
    # Rename HTML columns for display
    display_df = filtered_df[display_columns].copy()
    display_df.columns = [col.replace('_html', '') for col in display_columns]
    
    # Display table with tooltips
    st.markdown('<div class="stDataFrame">', unsafe_allow_html=True)
    st.write(
        display_df.to_html(
            escape=False,  # Allow HTML in cells
            index=False,
            classes="dataframe",
            border=0
        ),
        unsafe_allow_html=True
    )
    st.markdown('</div>', unsafe_allow_html=True)

    return filtered_df


def display_rules_detail(ticker, name, rule_details):
    """Display detailed rule checking results for a specific instrument"""
    st.subheader(f"Rule Details for {name}")
    
    # Separate long and short rules
    long_rules = {k: v for k, v in rule_details.items() if k.endswith('Long')}
    short_rules = {k: v for k, v in rule_details.items() if k.endswith('Short')}
    
    # Create two columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### LONG Rules")
        for key, rule in long_rules.items():
            if rule['status']:
                st.markdown(f"✅ **{rule['name']}**")
            else:
                if rule.get('critical', False):
                    st.markdown(f"❌ **{rule['name']}** (CRITICAL)")
                else:
                    st.markdown(f"❌ **{rule['name']}**")
            st.markdown(f"   *{rule['details']}*")
    
    with col2:
        st.markdown("### SHORT Rules")
        for key, rule in short_rules.items():
            if rule['status']:
                st.markdown(f"✅ **{rule['name']}**")
            else:
                if rule.get('critical', False):
                    st.markdown(f"❌ **{rule['name']}** (CRITICAL)")
                else:
                    st.markdown(f"❌ **{rule['name']}**")
            st.markdown(f"   *{rule['details']}*")


# --- Main App Flow ---
def main():
    st.title("🎯 Strict Strategy Scanner")
    
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
    
    st.sidebar.markdown("---")
    st.sidebar.caption(f"Technical Parameters: RSI({RSI_WINDOW}), RSI MA({RSI_MA_PERIOD}), EMAs: {EMA_SHORT}/{EMA_LONG}/{EMA_CONTEXT}")

    # Main tabs
    tab_options = ["Scan Results", "Rule Analysis"]
    tabs = st.tabs(tab_options)
    
    # Scan Results Tab
    with tabs[0]:
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
                    # Save filtered tickers for other tabs to use
                    st.session_state.filtered_tickers = {
                        row['_ticker']: row['Name'] for _, row in filtered_df.iterrows()
                    }
    
    # Rule Analysis Tab
    with tabs[1]:
        st.header("Detailed Rule Analysis")
        
        if not st.session_state.scan_results or not hasattr(st.session_state, 'filtered_tickers'):
            st.info("First run a scan to view rule analysis.")
        else:
            if not st.session_state.filtered_tickers:
                st.warning("No valid tickers available. Try scanning different tickers.")
            else:
                # Select a ticker for rule analysis
                rule_ticker = st.selectbox(
                    "Select an instrument for detailed rule analysis:",
                    options=list(st.session_state.filtered_tickers.keys()),
                    format_func=lambda x: f"{st.session_state.filtered_tickers[x]} ({x})",
                    key="rule_ticker_select"
                )
                
                if rule_ticker:
                    # Find the selected ticker in results
                    for result in st.session_state.scan_results:
                        if result['ticker'] == rule_ticker:
                            display_rules_detail(
                                result['ticker'], 
                                result['name'], 
                                result.get('rule_details', {})
                            )
                            break


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
