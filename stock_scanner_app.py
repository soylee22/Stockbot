import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set page config - favicon needs to be in the same folder as your script
st.set_page_config(
    page_title="Slater Stockbot",
    page_icon="favicon.ico",  # Just use the filename if in same directory
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    /* Main Theme Settings */
    :root {
        --primary-color: #2E5BFF;
        --primary-light: #E9EFFF;
        --secondary-color: #2EC5FF;
        --dark-blue: #0A2463;
        --success-color: #00C48C;
        --warning-color: #FFB74D;
        --danger-color: #FF5252;
        --bg-color: #F8F9FC;
        --card-bg: #FFFFFF;
        --text-color: #1F2937;
        --text-light: #6B7280;
        --border-color: #E5E7EB;
        --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
        --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
        --transition-fast: all 0.2s ease;
        --font-main: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Base styles */
    .main {
        background-color: var(--bg-color);
        color: var(--text-color);
        font-family: var(--font-main);
    }

    .stApp {
        background-color: var(--bg-color) !important;
    }
    
    /* Typography */
    h1, h2, h3, h4, h5, h6 {
        color: var(--dark-blue);
        font-family: var(--font-main);
        font-weight: 600;
    }
    
    h1 {
        font-size: 1.875rem;
        letter-spacing: -0.025em;
    }
    
    h2 {
        font-size: 1.5rem;
        letter-spacing: -0.025em;
    }
    
    h3 {
        font-size: 1.25rem;
    }
    
    /* Overwriting default Streamlit elements */
    .stButton>button {
        background-color: var(--primary-color);
        color: white;
        border-radius: 6px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 500;
        transition: var(--transition-fast);
        box-shadow: var(--shadow-sm);
        margin-bottom: 0.5rem;
    }
    
    .stButton>button:hover {
        background-color: var(--dark-blue);
        box-shadow: var(--shadow-md);
        transform: translateY(-1px);
    }
    
    .stTextInput>div>div>input {
        border-radius: 6px;
    }
    
    .stSlider>div>div>div {
        background-color: var(--primary-color);
    }
    
    /* Cards */
    .card {
        background-color: var(--card-bg);
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: var(--shadow-md);
        border: 1px solid var(--border-color);
        transition: var(--transition-fast);
    }
    
    .card:hover {
        box-shadow: var(--shadow-lg);
        transform: translateY(-2px);
    }
    
    .card-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1rem;
        border-bottom: 1px solid var(--border-color);
        padding-bottom: 0.75rem;
    }
    
    .card-title {
        font-size: 1.25rem;
        font-weight: 600;
        color: var(--dark-blue);
        margin: 0;
    }
    
    /* Data Tables */
    .dataframe-container {
        border-radius: 8px;
        overflow: hidden;
        border: 1px solid var(--border-color);
        box-shadow: var(--shadow-sm);
    }
    
    .dataframe {
        border-collapse: separate;
        border-spacing: 0;
        width: 100%;
    }
    
    .dataframe th {
        background-color: var(--primary-light);
        color: var(--dark-blue);
        font-weight: 600;
        text-align: left;
        padding: 0.75rem 1rem;
        border-bottom: 1px solid var(--border-color);
    }
    
    .dataframe td {
        padding: 0.75rem 1rem;
        border-bottom: 1px solid var(--border-color);
        transition: var(--transition-fast);
    }
    
    .dataframe tr:last-child td {
        border-bottom: none;
    }
    
    .dataframe tr:hover td {
        background-color: rgba(46, 91, 255, 0.05);
    }
    
    /* Sidebar */
    .css-1d391kg, .css-163ttbj, [data-testid="stSidebar"] {
        background-color: var(--card-bg);
        border-right: 1px solid var(--border-color);
    }
    
    /* Metrics */
    .metric-card {
        background-color: var(--card-bg);
        border-radius: 8px;
        padding: 1rem;
        box-shadow: var(--shadow-sm);
        border: 1px solid var(--border-color);
        text-align: center;
        transition: var(--transition-fast);
    }
    
    .metric-card:hover {
        box-shadow: var(--shadow-md);
        transform: translateY(-2px);
    }
    
    .metric-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: var(--primary-color);
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-size: 0.875rem;
        color: var(--text-light);
        margin: 0;
    }
    
    /* Signals legend */
    .legend-container {
        background-color: var(--card-bg);
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: var(--shadow-md);
        border: 1px solid var(--border-color);
        border-left: 4px solid var(--primary-color);
        transition: var(--transition-fast);
    }
    
    .legend-container:hover {
        box-shadow: var(--shadow-lg);
    }
    
    .legend-title {
        font-size: 1.125rem;
        font-weight: 600;
        color: var(--dark-blue);
        margin-top: 0;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
    }
    
    .legend-title svg {
        margin-right: 0.5rem;
    }
    
    /* Guide container */
    .guide-container {
        background-color: var(--card-bg);
        border-radius: 10px;
        padding: 1.5rem;
        margin-top: 1rem;
        box-shadow: var(--shadow-md);
        border: 1px solid var(--border-color);
        border-left: 4px solid var(--warning-color);
        transition: var(--transition-fast);
    }
    
    .guide-container:hover {
        box-shadow: var(--shadow-lg);
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 3rem;
        white-space: pre-wrap;
        background-color: var(--card-bg);
        border-radius: 6px 6px 0 0;
        gap: 0.5rem;
        padding: 0 1rem;
        border: 1px solid var(--border-color);
        border-bottom: none;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: var(--primary-light);
        border-top: 3px solid var(--primary-color);
    }
    
    /* Status indicators */
    .status-bullish {
        color: var(--success-color);
        font-weight: 600;
    }
    
    .status-bearish {
        color: var(--danger-color);
        font-weight: 600;
    }
    
    /* Refresh timer */
    .refresh-timer {
        text-align: center;
        padding: 0.75rem;
        background-color: var(--card-bg);
        border-radius: 8px;
        margin-top: 1.5rem;
        border: 1px solid var(--border-color);
        box-shadow: var(--shadow-sm);
    }
    
    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .animate-fade-in {
        animation: fadeIn 0.5s ease forwards;
    }
    
    /* Custom selection buttons */
    .selection-button {
        background-color: var(--card-bg);
        border: 1px solid var(--border-color);
        border-radius: 6px;
        padding: 0.5rem 0.75rem;
        font-size: 0.875rem;
        font-weight: 500;
        color: var(--text-color);
        cursor: pointer;
        transition: var(--transition-fast);
        text-align: center;
        box-shadow: var(--shadow-sm);
    }
    
    .selection-button:hover {
        background-color: var(--primary-light);
        border-color: var(--primary-color);
        box-shadow: var(--shadow-md);
    }
    
    .selection-button.active {
        background-color: var(--primary-color);
        color: white;
        border-color: var(--primary-color);
    }
    
    /* Multi-select styling */
    div[data-baseweb="select"] {
        border-radius: 6px;
    }
    
    /* Badge/pill styling for categories */
    .category-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 500;
        background-color: var(--primary-light);
        color: var(--primary-color);
        margin-right: 0.5rem;
        margin-bottom: 0.5rem;
        transition: var(--transition-fast);
    }
    
    .category-badge:hover {
        background-color: var(--primary-color);
        color: white;
    }
    
    /* Dashboard header */
    .dashboard-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1.5rem;
        padding-bottom: 1rem;
        border-bottom: 1px solid var(--border-color);
    }
    
    .dashboard-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: var(--dark-blue);
        margin: 0;
        display: flex;
        align-items: center;
    }
    
    .dashboard-title svg {
        margin-right: 0.75rem;
    }
    
    .last-updated {
        font-size: 0.875rem;
        color: var(--text-light);
    }
    
    /* Top performer cards */
    .performer-card {
        background-color: var(--card-bg);
        border-radius: 10px;
        overflow: hidden;
        transition: var(--transition-fast);
        box-shadow: var(--shadow-md);
        height: 100%;
    }
    
    .performer-card:hover {
        box-shadow: var(--shadow-lg);
        transform: translateY(-3px);
    }
    
    .performer-header {
        padding: 1rem;
        border-bottom: 1px solid var(--border-color);
        background-color: var(--primary-light);
    }
    
    .performer-title {
        font-weight: 600;
        color: var(--dark-blue);
        margin: 0;
        font-size: 1rem;
    }
    
    .performer-content {
        padding: 1rem;
    }
    
    .performer-metrics {
        display: flex;
        gap: 1rem;
        margin-top: 0.5rem;
    }
    
    .performer-metric {
        flex: 1;
        text-align: center;
    }
    
    .performer-metric-value {
        font-size: 1.25rem;
        font-weight: 600;
        color: var(--primary-color);
    }
    
    .performer-metric-label {
        font-size: 0.75rem;
        color: var(--text-light);
    }
    
    /* Progress bar styling */
    .stProgress > div > div > div {
        background-color: var(--primary-color);
    }
</style>
""", unsafe_allow_html=True)

# Symbol explanation
SYMBOL_EXPLANATION = {
    "🚀🚀": "Both Daily and Weekly Bullish",
    "🕣🕣": "Daily Bearish, Weekly Bullish (Clock)",
    "⚠️⚠️": "Daily Bullish, Weekly Bearish (Caution)",
    "💀💀": "Both Daily and Weekly Bearish",
    "✅": "EMAs aligned (7 EMA > 11 EMA > 21 EMA) on Daily Timeframe",
    "❌": "EMAs NOT aligned on Daily Timeframe"
}

# Expanded FTSE 100 stocks
FTSE_STOCKS = {
    # Original listings
    "AAL.L": "Anglo American",
    "ABF.L": "Associated British Foods",
    "AZN.L": "AstraZeneca",
    "BA.L": "BAE Systems",
    "BARC.L": "Barclays",
    "BATS.L": "British American Tobacco",
    "BP.L": "BP",
    "BRBY.L": "Burberry Group",
    "BT.A.L": "BT Group",
    "CPG.L": "Compass Group",
    "DGE.L": "Diageo",
    "GLEN.L": "Glencore",
    "GSK.L": "GSK",
    "HSBA.L": "HSBC Holdings",
    "LGEN.L": "Legal & General Group",
    "LLOY.L": "Lloyds Banking Group",
    "NG.L": "National Grid",
    "PRU.L": "Prudential",
    "REL.L": "RELX Group",
    "RIO.L": "Rio Tinto",
    "RR.L": "Rolls-Royce Holdings",
    "SHEL.L": "Shell",
    "STAN.L": "Standard Chartered",
    "TSCO.L": "Tesco",
    "ULVR.L": "Unilever",
    "VOD.L": "Vodafone Group",
    
    # Additional FTSE 100 stocks
    "ADM.L": "Admiral Group",
    "AGR.L": "Assura",
    "AHT.L": "Ashtead Group",
    "ANTO.L": "Antofagasta",
    "AUTO.L": "Auto Trader Group",
    "AV.L": "Aviva",
    "AVV.L": "AVEVA Group",
    "AVST.L": "Avast",
    "BKG.L": "Berkeley Group Holdings",
    "BNC.L": "Bunzl",
    "BNZL.L": "Bunzl",
    "BVIC.L": "Britvic",
    "CCH.L": "Coca-Cola HBC",
    "CNA.L": "Centrica",
    "CRDA.L": "Croda International",
    "CRH.L": "CRH plc",
    "DCC.L": "DCC plc",
    "ENT.L": "Entain",
    "EXPN.L": "Experian",
    "FCIT.L": "Foreign & Colonial Investment Trust",
    "FERG.L": "Ferguson plc",
    "FLTR.L": "Flutter Entertainment",
    "FRAS.L": "Frasers Group",
    "FRES.L": "Fresnillo plc",
    "GFS.L": "G4S",
    "HLMA.L": "Halma",
    "HLN.L": "Haleon",
    "HMSO.L": "Hammerson",
    "HWDN.L": "Howden Joinery Group",
    "IAG.L": "International Airlines Group",
    "ICP.L": "Intermediate Capital Group",
    "IHG.L": "InterContinental Hotels Group",
    "III.L": "3i Group",
    "IMB.L": "Imperial Brands",
    "INF.L": "Informa",
    "ITRK.L": "Intertek Group",
    "ITV.L": "ITV plc",
    "JD.L": "JD Sports Fashion",
    "KGF.L": "Kingfisher plc",
    "LAND.L": "Land Securities",
    "LCID.L": "Lucid Group",
    "LSEG.L": "London Stock Exchange Group",
    "MKS.L": "Marks & Spencer",
    "MNDI.L": "Mondi",
    "MRO.L": "Melrose Industries",
    "NWG.L": "NatWest Group",
    "NXT.L": "Next plc",
    "OCDO.L": "Ocado Group",
    "PSON.L": "Pearson plc",
    "PSN.L": "Persimmon plc",
    "PHNX.L": "Phoenix Group",
    "RKT.L": "Reckitt Benckiser",
    "RTO.L": "Rentokil Initial",
    "SBRY.L": "Sainsbury's",
    "SDR.L": "Schroders",
    "SGE.L": "Sage Group",
    "SGRO.L": "SEGRO",
    "SMT.L": "Scottish Mortgage Investment Trust",
    "SMIN.L": "Smiths Group",
    "SPX.L": "Spirax-Sarco Engineering",
    "SSE.L": "SSE plc",
    "STJ.L": "St. James's Place plc",
    "SVT.L": "Severn Trent",
    "TCAP.L": "TP ICAP Group",
    "TW.L": "Taylor Wimpey",
    "UU.L": "United Utilities",
    "WPP.L": "WPP plc",
    "WTB.L": "Whitbread",
    
    # Already included defense stock
    # "BA.L": "BAE Systems",
    # "RR.L": "Rolls-Royce Holdings",
}

# Expanded US stocks with Defense stocks integrated
US_STOCKS = {
    # Original listings
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "GOOGL": "Alphabet (Google)",
    "AMZN": "Amazon",
    "META": "Meta Platforms",
    "TSLA": "Tesla",
    "NVDA": "NVIDIA",
    "JPM": "JPMorgan Chase",
    "BAC": "Bank of America",
    "WMT": "Walmart",
    "PG": "Procter & Gamble",
    "JNJ": "Johnson & Johnson",
    "UNH": "UnitedHealth Group",
    "HD": "Home Depot",
    "MA": "Mastercard",
    "V": "Visa",
    "DIS": "Walt Disney",
    "NFLX": "Netflix",
    "INTC": "Intel",
    "AMD": "Advanced Micro Devices",
    "PYPL": "PayPal",
    "NKE": "Nike",
    "COST": "Costco",
    "SBUX": "Starbucks",
    "TXN": "Texas Instruments",
    
    # Additional US stocks - top tech stocks
    "ADBE": "Adobe",
    "AVGO": "Broadcom",
    "CRM": "Salesforce",
    "CSCO": "Cisco Systems",
    "ORCL": "Oracle",
    "IBM": "IBM",
    "QCOM": "Qualcomm",
    "MU": "Micron Technology",
    "AMAT": "Applied Materials",
    "TSM": "Taiwan Semiconductor",
    "ASML": "ASML Holding",
    "LRCX": "Lam Research",
    "KLAC": "KLA Corporation",
    "ADSK": "Autodesk",
    "ACN": "Accenture",
    "NOW": "ServiceNow",
    "SNOW": "Snowflake",
    "SHOP": "Shopify",
    "SQ": "Block (Square)",
    "NET": "Cloudflare",
    "ZM": "Zoom Video",
    "CRWD": "CrowdStrike",
    "OKTA": "Okta",
    "ZS": "Zscaler",
    "DDOG": "Datadog",
    "TEAM": "Atlassian",
    "TTD": "The Trade Desk",
    "PINS": "Pinterest",
    "SNAP": "Snap",
    "UBER": "Uber",
    "LYFT": "Lyft",
    "DASH": "DoorDash",
    "ABNB": "Airbnb",
    "SPOT": "Spotify",
    "RBLX": "Roblox",
    "U": "Unity Software",
    "PLTR": "Palantir",
    
    # Additional US stocks - other sectors
    "XOM": "Exxon Mobil",
    "CVX": "Chevron",
    "PFE": "Pfizer",
    "MRK": "Merck",
    "ABBV": "AbbVie",
    "BMY": "Bristol Myers Squibb",
    "LLY": "Eli Lilly",
    "AMGN": "Amgen",
    "TMO": "Thermo Fisher Scientific",
    "DHR": "Danaher",
    "ABT": "Abbott Laboratories",
    "MDT": "Medtronic",
    "CVS": "CVS Health",
    "UNP": "Union Pacific",
    "KO": "Coca-Cola",
    "PEP": "PepsiCo",
    "MDLZ": "Mondelez International",
    "MCD": "McDonald's",
    # "SBUX": "Starbucks", # Already listed above
    "CMG": "Chipotle",
    "TGT": "Target",
    "LOW": "Lowe's",
    "GS": "Goldman Sachs",
    "MS": "Morgan Stanley",
    "C": "Citigroup",
    "WFC": "Wells Fargo",
    "AXP": "American Express",
    "BLK": "BlackRock",
    "SCHW": "Charles Schwab",
    "CB": "Chubb",
    "MET": "MetLife",
    "COP": "ConocoPhillips",
    "EOG": "EOG Resources",
    "SLB": "Schlumberger",
    "SPGI": "S&P Global",
    "MMM": "3M",
    "CAT": "Caterpillar",
    "DE": "Deere & Company",
    "BA": "Boeing",
    "LMT": "Lockheed Martin",
    "GE": "General Electric",
    "HON": "Honeywell",
    "UPS": "United Parcel Service",
    "FDX": "FedEx",
    "RTX": "Raytheon Technologies",
    "NEE": "NextEra Energy",
    "DUK": "Duke Energy",
    "SO": "Southern Company",
    "AEP": "American Electric Power",
    "D": "Dominion Energy",
    "LIN": "Linde",
    "APD": "Air Products",
    "SHW": "Sherwin-Williams",
    
    # Additional US defense stocks and ADRs
    "NOC": "Northrop Grumman",
    "GD": "General Dynamics",
    "LHX": "L3Harris Technologies",
    "HII": "Huntington Ingalls Industries",
    "TDG": "TransDigm Group",
    "TXT": "Textron",
    "SPR": "Spirit AeroSystems",
    "FINMY": "Leonardo (ADR)",
    "SAABY": "Saab (ADR)"
    
    # Defense stocks already included in the original list:
    # "LMT": "Lockheed Martin",
    # "RTX": "Raytheon Technologies",
    # "BA": "Boeing",
}

# Expanded Euro Stoxx stocks with Defense stocks integrated
EURO_STOCKS = {
    # Original listings
    "AIR.PA": "Airbus",
    "ALV.DE": "Allianz",
    "ADS.DE": "Adidas",
    "ASML.AS": "ASML Holding",
    "BAS.DE": "BASF",
    "BAYN.DE": "Bayer",
    "BMW.DE": "BMW",
    "BNP.PA": "BNP Paribas",
    "CS.PA": "AXA",
    "DAI.DE": "Daimler",
    "DB1.DE": "Deutsche Börse",
    "DPW.DE": "Deutsche Post",
    "DTE.DE": "Deutsche Telekom",
    "ENEL.MI": "Enel",
    "ENGI.PA": "ENGIE",
    "ENI.MI": "Eni",
    "IBE.MC": "Iberdrola",
    "IFX.DE": "Infineon",
    "ISP.MI": "Intesa Sanpaolo",
    "KER.PA": "Kering",
    "MC.PA": "LVMH",
    "MUV2.DE": "Munich Re",
    "NOKIA.HE": "Nokia",
    "OR.PA": "L'Oréal",
    "SAF.PA": "Safran",
    "SAN.MC": "Banco Santander",
    "SAP.DE": "SAP",
    "SIE.DE": "Siemens",
    "SU.PA": "Schneider Electric",
    "VNA.DE": "Vonovia",
    
    # Additional European stocks
    # French stocks (.PA)
    "AI.PA": "Air Liquide",
    "CA.PA": "Carrefour",
    "BN.PA": "Danone",
    "DSY.PA": "Dassault Systèmes",
    "EL.PA": "EssilorLuxottica",
    "HO.PA": "Thales",
    "ML.PA": "Michelin",
    "ORA.PA": "Orange",
    "RI.PA": "Pernod Ricard",
    "RMS.PA": "Hermès",
    "RNO.PA": "Renault",
    "SGO.PA": "Saint-Gobain",
    "STM.PA": "STMicroelectronics",
    "UG.PA": "Peugeot",
    "VIE.PA": "Veolia Environnement",
    "VIV.PA": "Vivendi",
    
    # German stocks (.DE)
    "1COV.DE": "Covestro",
    "ARL.DE": "Aareal Bank",
    "CBK.DE": "Commerzbank",
    "CON.DE": "Continental",
    "DB.DE": "Deutsche Bank",
    "DHER.DE": "Delivery Hero",
    "DHL.DE": "Deutsche Post",
    # "DPW.DE": "Deutsche Post", # Already listed above
    "DTG.DE": "Daimler Truck",
    "ENR.DE": "Siemens Energy",
    "EOAN.DE": "E.ON",
    "EVK.DE": "Evonik Industries",
    "FME.DE": "Fresenius Medical Care",
    "FRA.DE": "Fraport",
    "FRE.DE": "Fresenius",
    "HEI.DE": "HeidelbergCement",
    "HEN3.DE": "Henkel",
    "HFG.DE": "HelloFresh",
    "HOT.DE": "Hochtief",
    "LHA.DE": "Lufthansa",
    "LIN.DE": "Linde",
    "MBG.DE": "Mercedes-Benz Group",
    "MRK.DE": "Merck",
    "NTOG.DE": "Nordex",
    "PAH3.DE": "Porsche",
    "PUM.DE": "Puma",
    "RWE.DE": "RWE",
    "SDF.DE": "K+S",
    "SHL.DE": "Siemens Healthineers",
    "SRT3.DE": "Sartorius",
    "TKA.DE": "ThyssenKrupp",
    "VOW3.DE": "Volkswagen",
    "ZAL.DE": "Zalando",
    
    # Italian stocks (.MI)
    "ATL.MI": "Atlantia",
    "EGPW.MI": "Enel Green Power",
    "EXO.MI": "Exor",
    "FCA.MI": "Fiat Chrysler Automobiles",
    "G.MI": "Assicurazioni Generali",
    # "ISP.MI": "Intesa Sanpaolo", # Already listed above
    "LDO.MI": "Leonardo",
    "MB.MI": "Mediobanca",
    "MS.MI": "Mediaset",
    "RACE.MI": "Ferrari",
    "SPM.MI": "Saipem",
    "SRG.MI": "Snam",
    "STM.MI": "STMicroelectronics",
    "TIT.MI": "Telecom Italia",
    "TRN.MI": "Terna",
    "UBI.MI": "UBI Banca",
    "UCG.MI": "UniCredit",
    
    # Spanish stocks (.MC)
    "ACS.MC": "ACS",
    "AMS.MC": "Amadeus IT Group",
    "ANA.MC": "Acciona",
    "BBVA.MC": "BBVA",
    "BKT.MC": "Bankinter",
    "CABK.MC": "CaixaBank",
    "ELE.MC": "Endesa",
    "FER.MC": "Ferrovial",
    "GRF.MC": "Grifols",
    "IAG.MC": "International Airlines Group",
    "IDR.MC": "Indra Sistemas",
    "ITX.MC": "Inditex",
    "MAP.MC": "MAPFRE",
    "MEL.MC": "Meliá Hotels",
    "REP.MC": "Repsol",
    "TEF.MC": "Telefónica",
    
    # Dutch stocks (.AS)
    "ABN.AS": "ABN AMRO",
    "AD.AS": "Ahold Delhaize",
    "AKZA.AS": "Akzo Nobel",
    "DSM.AS": "DSM",
    "HEIA.AS": "Heineken",
    "IMCD.AS": "IMCD",
    "INGA.AS": "ING Group",
    "KPN.AS": "KPN",
    "MT.AS": "ArcelorMittal",
    "NN.AS": "NN Group",
    "PHIA.AS": "Philips",
    "RAND.AS": "Randstad",
    "REN.AS": "RELX",
    "UNA.AS": "Unilever",
    "URW.AS": "Unibail-Rodamco-Westfield",
    "WKL.AS": "Wolters Kluwer",
    
    # Swiss stocks (not in Euro but important European stocks)
    "ABBN.SW": "ABB Ltd",
    "ADEN.SW": "Adecco Group",
    "CFR.SW": "Richemont",
    "CSGN.SW": "Credit Suisse",
    "GIVN.SW": "Givaudan",
    "HOLN.SW": "Holcim",
    "LONN.SW": "Lonza Group",
    "NESN.SW": "Nestlé",
    "NOVN.SW": "Novartis",
    "ROG.SW": "Roche",
    "SGSN.SW": "SGS",
    "SIKA.SW": "Sika",
    "UHR.SW": "Swatch Group",
    "UBSG.SW": "UBS Group",
    "ZURN.SW": "Zurich Insurance",
    
    # Additional defense stocks (European)
    "RHM.DE": "Rheinmetall",
    "HAG.DE": "Hensoldt",
    "MTX.DE": "MTU Aero Engines",
    "R3NK.DE": "RENK Group",
    "NTH.DE": "Northrop Grumman (German listing)",
    "AM.PA": "Dassault Aviation",
    "EXA.PA": "Exail Technologies",
    "THEON.AS": "Theon International",
    "SDV1.DE": "Saab (German listing)",
    "SAAB-B.ST": "Saab (Stockholm listing)",
    "FACC.VI": "FACC",
    "KOZ.DE": "Kongsberg Gruppen (German listing)",
    "KOG.OL": "Kongsberg Gruppen (Oslo listing)",
    "LDOF.MI": "Leonardo",
    "FNC.MI": "Fincantieri",
    "MGNT.MI": "Magnaghi Aeronautica"
    
    # Already included defense stocks:
    # "AIR.PA": "Airbus",
    # "SAF.PA": "Safran",
    # "HO.PA": "Thales",
    # "TKA.DE": "ThyssenKrupp",
    # "LDO.MI": "Leonardo",
}

# Asian stocks category remains unchanged
ASIAN_STOCKS = {
    # Japanese stocks
    "7203.T": "Toyota Motor",
    "9984.T": "SoftBank Group",
    "6758.T": "Sony Group",
    "6861.T": "Keyence",
    "6501.T": "Hitachi",
    "9433.T": "KDDI",
    "4063.T": "Shin-Etsu Chemical",
    "8306.T": "Mitsubishi UFJ Financial",
    "7267.T": "Honda Motor",
    "9432.T": "Nippon Telegraph & Telephone",
    "6367.T": "Daikin Industries",
    "6098.T": "Recruit Holdings",
    "7974.T": "Nintendo",
    "4543.T": "Terumo",
    "7751.T": "Canon",
    
    # Hong Kong stocks
    "0700.HK": "Tencent Holdings",
    "9988.HK": "Alibaba Group",
    "0941.HK": "China Mobile",
    "1398.HK": "ICBC",
    "3690.HK": "Meituan",
    "0883.HK": "CNOOC",
    "0005.HK": "HSBC Holdings",
    "0939.HK": "China Construction Bank",
    "2318.HK": "Ping An Insurance",
    "0388.HK": "Hong Kong Exchanges",
    "1211.HK": "BYD Company",
    "0003.HK": "Hong Kong and China Gas",
    "0027.HK": "Galaxy Entertainment",
    "1177.HK": "Sino Biopharmaceutical",
    "0016.HK": "Sun Hung Kai Properties",
    
    # Singapore stocks
    "D05.SI": "DBS Group",
    "O39.SI": "OCBC Bank",
    "U11.SI": "United Overseas Bank",
    "Z74.SI": "Singapore Telecommunications",
    "C52.SI": "ComfortDelGro",
    "C38U.SI": "CapitaLand Integrated Commercial Trust",
    "A17U.SI": "Ascendas REIT",
    "C09.SI": "City Developments",
    "F34.SI": "Wilmar International",
    "U96.SI": "Sembcorp Industries",
    "S58.SI": "SATS",
    "BS6.SI": "Genting Singapore",
    "C31.SI": "CapitaLand",
    "H78.SI": "Hongkong Land",
    "J36.SI": "Jardine Matheson",
    
    # Chinese A-shares
    "601318.SS": "Ping An Insurance",
    "601857.SS": "PetroChina",
    "601288.SS": "Agricultural Bank of China",
    "601988.SS": "Bank of China",
    "601628.SS": "China Life Insurance",
    "600519.SS": "Kweichow Moutai",
    "600036.SS": "China Merchants Bank",
    "601166.SS": "Industrial Bank",
    "600276.SS": "Jiangsu Hengrui Medicine",
    "600887.SS": "Inner Mongolia Yili Industrial",
    "601888.SS": "China Tourism Group Duty Free",
    "600030.SS": "CITIC Securities",
    "601816.SS": "China Construction Bank",
    "600000.SS": "Shanghai Pudong Development Bank",
    "601088.SS": "China Shenhua Energy"
}

# Updated Market Indices with Swiss and Singapore indices
INDICES = {
    # Major US Indices
    "^GSPC": "S&P 500",
    "^IXIC": "NASDAQ Composite",
    "^DJI": "Dow Jones Industrial",
    "^RUT": "Russell 2000",
    # European Indices
    "^FTSE": "UK FTSE 100",
    "^GDAXI": "German DAX",
    "^FCHI": "French CAC 40",
    "^STOXX50E": "Euro STOXX 50",
    "^IBEX": "Spanish IBEX 35",
    "^AEX": "Netherlands AEX",
    # Added Swiss Market Index (SMI)
    "^SSMI": "Swiss SMI 20",
    # Asia Pacific Indices
    "^AXJO": "Australian ASX 200",
    "^N225": "Japanese Nikkei 225",
    "^HSI": "Hong Kong Hang Seng",
    "000001.SS": "Shanghai Composite",
    # Added Singapore Straits Times Index (STI)
    "^STI": "Singapore STI 30",
}

# Add all required dictionaries in the correct order before they're referenced

# Commodities dictionary definition
COMMODITIES = {
    "GC=F": "Gold",
    "SI=F": "Silver", 
    "HG=F": "Copper",
    "NG=F": "Natural Gas",
    "BZ=F": "Brent Crude Oil",
    "CL=F": "WTI Crude Oil",
    # Added Platinum and Palladium
    "PL=F": "Platinum",
    "PA=F": "Palladium"
}

# FOREX dictionary definition
FOREX = {
    # Major pairs
    "EURUSD=X": "EUR/USD",
    "GBPUSD=X": "GBP/USD",
    "USDJPY=X": "USD/JPY",
    "AUDUSD=X": "AUD/USD",
    "NZDUSD=X": "NZD/USD",
    # Cross pairs
    "EURGBP=X": "EUR/GBP",
    "GBPJPY=X": "GBP/JPY",
    "EURJPY=X": "EUR/JPY",
    "GBPAUD=X": "GBP/AUD",
    "GBPCAD=X": "GBP/CAD",
    "GBPCHF=X": "GBP/CHF",
    "CHFJPY=X": "CHF/JPY",
}

# US Treasury ETFs and bond funds
TREASURIES = {
    "SHY": "iShares 1-3 Year Treasury Bond ETF",
    "IEI": "iShares 3-7 Year Treasury Bond ETF",
    "IEF": "iShares 7-10 Year Treasury Bond ETF",
    "TLH": "iShares 10-20 Year Treasury Bond ETF",
    "TLT": "iShares 20+ Year Treasury Bond ETF",
    "GOVT": "iShares U.S. Treasury Bond ETF",
    "VGSH": "Vanguard Short-Term Treasury ETF",
    "VGIT": "Vanguard Intermediate-Term Treasury ETF",
    "VGLT": "Vanguard Long-Term Treasury ETF",
    "BND": "Vanguard Total Bond Market ETF",
    "AGG": "iShares Core U.S. Aggregate Bond ETF",
    "MBB": "iShares MBS ETF"
}

# Make sure TICKER_CATEGORIES uses these defined dictionaries
TICKER_CATEGORIES = {
    "INDICES": INDICES,
    "FOREX": FOREX,
    "COMMODITIES": COMMODITIES,
    "TREASURIES": TREASURIES,
    "FTSE STOCKS": FTSE_STOCKS,
    "US STOCKS": US_STOCKS,
    "EURO STOCKS": EURO_STOCKS,
    "ASIAN STOCKS": ASIAN_STOCKS
}

@st.cache_data(ttl=600)
def fetch_stock_data(ticker, period="6mo", interval="1d"):
    """
    Fetch stock data for a given ticker
    """
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period, interval=interval)
        return hist
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")
        return pd.DataFrame()

def calculate_rsi(data, window=14):
    """
    Calculate RSI (Relative Strength Index) using the standard method
    """
    if data.empty or len(data) < window*2:
        return pd.Series([np.nan] * len(data))
    
    # Calculate price changes
    delta = data['Close'].diff()
    
    # Separate gains and losses
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    # First average gain and loss
    first_avg_gain = gain.iloc[1:window+1].mean()
    first_avg_loss = loss.iloc[1:window+1].mean()
    
    # Initialize lists with NaNs for the first window periods
    avg_gains = [np.nan] * window
    avg_losses = [np.nan] * window
    
    # Set the first average gain and loss
    avg_gains.append(first_avg_gain)
    avg_losses.append(first_avg_loss)
    
    # Calculate subsequent values
    for i in range(window+1, len(delta)):
        avg_gain = (avg_gains[-1] * (window-1) + gain.iloc[i]) / window
        avg_loss = (avg_losses[-1] * (window-1) + loss.iloc[i]) / window
        avg_gains.append(avg_gain)
        avg_losses.append(avg_loss)
    
    # Convert to Series
    avg_gain_series = pd.Series(avg_gains, index=data.index)
    avg_loss_series = pd.Series(avg_losses, index=data.index)
    
    # Calculate RS and RSI
    rs = avg_gain_series / avg_loss_series
    rsi = 100 - (100 / (1 + rs))
    
    # Handle division by zero (when avg_loss is 0)
    rsi = rsi.replace([np.inf, -np.inf], 100)
    
    return rsi

def calculate_ema(data, spans=[7, 11, 21]):
    """
    Calculate EMAs for given spans
    """
    emas = {}
    for span in spans:
        emas[f'EMA_{span}'] = data['Close'].ewm(span=span, adjust=False).mean()
    return emas

def check_ema_alignment(emas):
    """
    Check if EMAs are aligned (7 EMA > 11 EMA > 21 EMA)
    """
    if all(pd.isna(ema.iloc[-1]) for ema in emas.values()):
        return False
    
    return emas['EMA_7'].iloc[-1] > emas['EMA_11'].iloc[-1] > emas['EMA_21'].iloc[-1]

def calculate_bullish_score(daily_rsi, weekly_rsi, ema_aligned, pct_change):
    """
    Calculate a bullish score to sort results (higher is more bullish)
    Pure RSI-based scoring to find the true top performers
    """
    # Very straightforward - just use the RSI values directly
    # Daily RSI is weighted 2x as it's more current
    score = daily_rsi * 2 + weekly_rsi
    
    return score

def scan_ticker(ticker, display_name):
    """
    Scan a ticker and return analysis based on criteria
    """
    try:
        # Fetch data
        daily_data = fetch_stock_data(ticker, period="3mo", interval="1d")
        weekly_data = fetch_stock_data(ticker, period="1y", interval="1wk")
        
        if daily_data.empty or weekly_data.empty or len(daily_data) < 30 or len(weekly_data) < 14:
            return {"display_name": display_name, "error": "Insufficient data", "score": -1000}
        
        # Calculate indicators
        daily_rsi = calculate_rsi(daily_data)
        weekly_rsi = calculate_rsi(weekly_data)
        emas = calculate_ema(daily_data)
        
        # Get latest values
        latest_daily_rsi = daily_rsi.iloc[-1]
        latest_weekly_rsi = weekly_rsi.iloc[-1]
        ema_aligned = check_ema_alignment(emas)
        
        # Determine conditions
        daily_bullish = latest_daily_rsi > 50
        weekly_bullish = latest_weekly_rsi > 50
        
        # Prepare status strings
        daily_status = "Bullish" if daily_bullish else "Bearish"
        weekly_status = "Bullish" if weekly_bullish else "Bearish"
        ema_status = "✅" if ema_aligned else "❌"
        
        # Determine emoji
        if daily_bullish and weekly_bullish:
            emoji = "🚀🚀"
        elif not daily_bullish and weekly_bullish:
            emoji = "🕣🕣"
        elif daily_bullish and not weekly_bullish:
            emoji = "⚠️⚠️"
        else:
            emoji = "💀💀"
        
        # Current price
        current_price = daily_data['Close'].iloc[-1]
        
        # Calculate % change today
        if len(daily_data) > 1:
            prev_close = daily_data['Close'].iloc[-2]
            pct_change = ((current_price - prev_close) / prev_close) * 100
        else:
            pct_change = 0
            
        # Calculate bullish score for sorting
        bullish_score = calculate_bullish_score(latest_daily_rsi, latest_weekly_rsi, ema_aligned, pct_change)
        
        # Return result as dictionary for easier sorting
        return {
            "display_name": display_name,
            "ticker": ticker,
            "emoji": emoji,
            "daily_status": daily_status,
            "weekly_status": weekly_status,
            "ema_status": ema_status,
            "daily_rsi": latest_daily_rsi,
            "weekly_rsi": latest_weekly_rsi,
            "price": current_price,
            "pct_change": pct_change,
            "score": bullish_score,
            "daily_data": daily_data,
            "weekly_data": weekly_data,
            "emas": emas,
            "error": None
        }
    
    except Exception as e:
        return {
            "display_name": display_name,
            "ticker": ticker,
            "error": str(e),
            "score": -1000
        }

def create_chart(result):
    """
    Create an interactive chart for a given ticker
    """
    if result.get("error") or "daily_data" not in result:
        return None
    
    # Create figure with secondary y-axis
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.03, 
                        row_heights=[0.7, 0.3],
                        subplot_titles=(f"{result['display_name']} - Price Chart", "RSI"))
    
    # Add candlestick chart
    fig.add_trace(go.Candlestick(
        x=result["daily_data"].index,
        open=result["daily_data"]['Open'],
        high=result["daily_data"]['High'],
        low=result["daily_data"]['Low'],
        close=result["daily_data"]['Close'],
        name="Price",
        increasing_line_color='#26A69A', 
        decreasing_line_color='#EF5350'
    ), row=1, col=1)
    
    # Add EMAs
    emas = result["emas"]
    colors = ['#1E88E5', '#FFC107', '#7CB342']  # Blue, Amber, Green
    for i, span in enumerate([7, 11, 21]):
        fig.add_trace(go.Scatter(
            x=emas[f'EMA_{span}'].index,
            y=emas[f'EMA_{span}'],
            mode='lines',
            line=dict(width=2, color=colors[i]),
            name=f'EMA {span}'
        ), row=1, col=1)
    
    # Add RSI
    daily_rsi_series = calculate_rsi(result["daily_data"])
    fig.add_trace(go.Scatter(
        x=daily_rsi_series.index, 
        y=daily_rsi_series,
        line=dict(color='#BA68C8', width=2),
        name='RSI (14)'
    ), row=2, col=1)
    
    # Add RSI horizontal lines at 70 and 30
    fig.add_shape(type="line", x0=daily_rsi_series.index[0], x1=daily_rsi_series.index[-1], 
                 y0=70, y1=70, line=dict(color="red", width=1, dash="dash"), row=2, col=1)
    fig.add_shape(type="line", x0=daily_rsi_series.index[0], x1=daily_rsi_series.index[-1], 
                 y0=30, y1=30, line=dict(color="green", width=1, dash="dash"), row=2, col=1)
    # Add a center line at 50
    fig.add_shape(type="line", x0=daily_rsi_series.index[0], x1=daily_rsi_series.index[-1], 
                 y0=50, y1=50, line=dict(color="gray", width=1, dash="dash"), row=2, col=1)
    
    # Update layout
    fig.update_layout(
        xaxis_rangeslider_visible=False,
        template="plotly_white",
        height=600,
        paper_bgcolor='white',
        plot_bgcolor='#f8f9fa',
        margin=dict(l=10, r=10, t=60, b=10),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        font=dict(
            family="Arial, sans-serif",
            size=12,
            color="#212529"
        )
    )
    
    # Y-axis ranges
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="RSI", range=[0, 100], row=2, col=1)
    
    # Style updates
    fig.update_xaxes(
        gridcolor="#e0e0e0",
        showgrid=True
    )
    fig.update_yaxes(
        gridcolor="#e0e0e0",
        showgrid=True
    )
    
    return fig

def display_signal_legend():
    """Display the signal legend attractively"""
    st.markdown("""
    <div class="legend-container animate-fade-in">
        <h3 class="legend-title">
            <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M3 3v18h18"/>
                <path d="M18.7 8l-5.1 5.2-2.8-2.7L7 14.3"/>
            </svg>
            Signal Indicators
        </h3>
        <table style="width:100%; color: #1F2937;">
            <tr>
                <td style="padding:10px;width:60px;text-align:center;"><span style="font-size:24px;">🚀🚀</span></td>
                <td style="color: #1F2937; font-weight: 500;">Both Daily and Weekly Bullish</td>
            </tr>
            <tr>
                <td style="padding:10px;width:60px;text-align:center;"><span style="font-size:24px;">🕣🕣</span></td>
                <td style="color: #1F2937; font-weight: 500;">Daily Bearish, Weekly Bullish (Clock)</td>
            </tr>
            <tr>
                <td style="padding:10px;width:60px;text-align:center;"><span style="font-size:24px;">⚠️⚠️</span></td>
                <td style="color: #1F2937; font-weight: 500;">Daily Bullish, Weekly Bearish (Caution)</td>
            </tr>
            <tr>
                <td style="padding:10px;width:60px;text-align:center;"><span style="font-size:24px;">💀💀</span></td>
                <td style="color: #1F2937; font-weight: 500;">Both Daily and Weekly Bearish</td>
            </tr>
            <tr>
                <td style="padding:10px;width:60px;text-align:center;"><span style="font-size:24px;">✅</span></td>
                <td style="color: #1F2937; font-weight: 500;">EMAs aligned (7 EMA > 11 EMA > 21 EMA) on Daily Timeframe</td>
            </tr>
            <tr>
                <td style="padding:10px;width:60px;text-align:center;"><span style="font-size:24px;">❌</span></td>
                <td style="color: #1F2937; font-weight: 500;">EMAs NOT aligned on Daily Timeframe</td>
            </tr>
        </table>
    </div>
    """, unsafe_allow_html=True)

def display_rsi_guide():
    """Display the RSI color guide"""
    st.markdown("""
    <div class="guide-container animate-fade-in">
        <h3 class="legend-title">
            <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <circle cx="12" cy="12" r="10"/>
                <line x1="12" y1="16" x2="12" y2="12"/>
                <line x1="12" y1="8" x2="12" y2="8"/>
            </svg>
            RSI Indicator Guide
        </h3>
        <div style="display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 15px;">
            <div class="category-badge" style="background-color: #e6f4ea; color: #00B050;">Strong Bullish</div>
            <div class="category-badge" style="background-color: #f1f8e9; color: #92D050;">Bullish</div>
            <div class="category-badge" style="background-color: #feeaeb; color: #FF6666;">Bearish</div>
            <div class="category-badge" style="background-color: #ffdada; color: #FF0000;">Strong Bearish</div>
        </div>
        <ul style="list-style-type: none; padding-left: 10px; color: #1F2937;">
            <li style="margin-bottom: 8px; display: flex; align-items: center;">
                <span style="width: 16px; height: 16px; display: inline-block; background-color: #00B050; border-radius: 4px; margin-right: 10px;"></span>
                <strong>Strong Green (>70)</strong>: Strong bullish momentum
            </li>
            <li style="margin-bottom: 8px; display: flex; align-items: center;">
                <span style="width: 16px; height: 16px; display: inline-block; background-color: #92D050; border-radius: 4px; margin-right: 10px;"></span>
                <strong>Light Green (>50)</strong>: Bullish momentum
            </li>
            <li style="margin-bottom: 8px; display: flex; align-items: center;">
                <span style="width: 16px; height: 16px; display: inline-block; background-color: #FF6666; border-radius: 4px; margin-right: 10px;"></span>
                <strong>Light Red (<50)</strong>: Bearish momentum
            </li>
            <li style="margin-bottom: 8px; display: flex; align-items: center;">
                <span style="width: 16px; height: 16px; display: inline-block; background-color: #FF0000; border-radius: 4px; margin-right: 10px;"></span>
                <strong>Strong Red (<30)</strong>: Strong bearish momentum
            </li>
        </ul>
        <p style="margin-bottom:0; color: #1F2937; padding-top: 10px; border-top: 1px solid #E5E7EB;">
            The <b>Change %</b> column shows the daily price change percentage. Green indicates positive change, red indicates negative.
        </p>
    </div>
    """, unsafe_allow_html=True)

def format_dataframe(df):
    """
    Apply conditional formatting to the dataframe
    """
    # Make a copy to ensure we don't modify the original
    df = df.copy()
    
    # Convert Change % to numeric (removing % sign if present) and round to 2 decimal places
    if 'Change %' in df.columns:
        df['Change %'] = pd.to_numeric(df['Change %'].str.replace('%', ''), errors='coerce')
        df['Change %'] = df['Change %'].round(2).astype(str)
    
    # Convert RSI columns to numeric
    if 'Daily RSI' in df.columns:
        df['Daily RSI'] = pd.to_numeric(df['Daily RSI'], errors='coerce')
    if 'Weekly RSI' in df.columns:
        df['Weekly RSI'] = pd.to_numeric(df['Weekly RSI'], errors='coerce')
    
    # Format the Daily and Weekly status columns
    df_styled = df.style.map(
        lambda x: 'color: #4CAF50; font-weight: bold' if x == 'Bullish' else 'color: #F44336; font-weight: bold',
        subset=['Daily', 'Weekly']
    )
    
    # Format the price change column
    df_styled = df_styled.map(
        lambda x: f'color: {"#4CAF50" if float(x) > 0 else "#F44336"}; font-weight: bold' if x != 'nan' else '',
        subset=['Change %']
    )
    
    # Format the RSI columns
    def highlight_rsi(val):
        try:
            val_num = float(val)
            if val_num > 70:
                return 'color: #00B050; font-weight: bold'  # Strong bullish (deep green)
            elif val_num < 30:
                return 'color: #FF0000; font-weight: bold'  # Strong bearish (deep red)
            elif val_num > 50:
                return 'color: #92D050'  # Moderate bullish (light green)
            else:
                return 'color: #FF6666'  # Moderate bearish (light red)
        except (ValueError, TypeError):
            return ''
    
    df_styled = df_styled.map(highlight_rsi, subset=['Daily RSI', 'Weekly RSI'])
    
    return df_styled

def main():
    # Sidebar configuration
    with st.sidebar:
        # Logo at the top
        try:
            # Try to open and display the logo - using st.image which is more reliable
            st.image("837934968543099023.png", width=180)
        except:
            # Fallback if image isn't found
            st.markdown("""
            <div style="text-align: center; padding: 1rem 0; background-color: #f0f2f6; border-radius: 10px;">
                <h2 style="color: #2E5BFF;">Slater Stockbot</h2>
            </div>
            """, unsafe_allow_html=True)
        
        st.title("Slater Stockbot")
        
        st.markdown("<div style='height: 1px; background-color: #E5E7EB; margin: 0.5rem 0 1.5rem;'></div>", unsafe_allow_html=True)
        
        st.markdown("""
        <div style="display: flex; align-items: center; margin-bottom: 1rem;">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" style="margin-right: 10px;">
                <path d="M12 8V12L15 15" stroke="#2E5BFF" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                <circle cx="12" cy="12" r="9" stroke="#2E5BFF" stroke-width="2"/>
            </svg>
            <h2 style="margin: 0; font-size: 1.25rem;">Scan Settings</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Quick selection buttons
        st.markdown("""
        <p style="font-size: 0.875rem; color: #6B7280; margin-bottom: 0.5rem;">Quick Select</p>
        """, unsafe_allow_html=True)
        
        selection_cols = st.columns(3)
        
        # Get all categories and stock categories
        all_categories = list(TICKER_CATEGORIES.keys())
        stock_categories = [cat for cat in all_categories if "STOCKS" in cat]
        
        # Default to all categories if none selected yet
        if "selected_categories" not in st.session_state:
            st.session_state.selected_categories = all_categories
            
        # Custom button styling
        button_style = """
        <style>
        div[data-testid="column"] button {
            width: 100%;
            border: none;
            box-shadow: none;
        }
        
        div[data-testid="column"]:nth-child(1) button {
            background-color: #2E5BFF;
        }
        
        div[data-testid="column"]:nth-child(2) button {
            background-color: #2EC5FF;
        }
        
        div[data-testid="column"]:nth-child(3) button {
            background-color: #9E9E9E;
        }
        </style>
        """
        st.markdown(button_style, unsafe_allow_html=True)
        
        # Select All button
        if selection_cols[0].button("Select All", key="select_all"):
            st.session_state.selected_categories = all_categories
        
        # Select Stocks Only button
        if selection_cols[1].button("Stocks Only", key="stocks_only"):
            st.session_state.selected_categories = stock_categories
        
        # Clear All button
        if selection_cols[2].button("Clear All", key="clear_all"):
            st.session_state.selected_categories = []
        
        # Category selection with session state
        st.markdown("<p style='font-size: 0.875rem; color: #6B7280; margin: 1rem 0 0.5rem;'>Market Categories</p>", unsafe_allow_html=True)
        
        selected_categories = st.multiselect(
            "",  # Empty label since we're using the custom label above
            options=all_categories,
            default=st.session_state.selected_categories
        )
        
        # Update session state
        st.session_state.selected_categories = selected_categories
        
        # Scan interval
        st.markdown("<p style='font-size: 0.875rem; color: #6B7280; margin: 1rem 0 0.5rem;'>Refresh Interval (minutes)</p>", unsafe_allow_html=True)
        refresh_interval = st.slider(
            "",  # Empty label since we're using the custom label above
            min_value=1,
            max_value=60,
            value=5,
            step=1
        )
        
        # Display setting
        st.markdown("<p style='font-size: 0.875rem; color: #6B7280; margin: 1rem 0 0.5rem;'>Display Options</p>", unsafe_allow_html=True)
        show_charts = st.checkbox("Show Charts for Top Performers", value=True)
        
        st.markdown("<div style='height: 1px; background-color: #E5E7EB; margin: 1.5rem 0;'></div>", unsafe_allow_html=True)
        
        st.markdown("""
        <div style="display: flex; align-items: center; margin-bottom: 1rem;">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" style="margin-right: 10px;">
                <path d="M10 3H3V10H10V3Z" stroke="#2E5BFF" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                <path d="M21 3H14V10H21V3Z" stroke="#2E5BFF" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                <path d="M21 14H14V21H21V14Z" stroke="#2E5BFF" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                <path d="M10 14H3V21H10V14Z" stroke="#2E5BFF" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
            </svg>
            <h2 style="margin: 0; font-size: 1.25rem;">Market Overview</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Create placeholder for market overview metrics
        market_metrics = st.empty()
        
        # About section
        st.markdown("<div style='height: 1px; background-color: #E5E7EB; margin: 1.5rem 0;'></div>", unsafe_allow_html=True)
        
        st.markdown("""
        <div style="background-color: rgba(46, 91, 255, 0.08); border-radius: 8px; padding: 1rem; margin-top: 1rem;">
            <p style="margin: 0; font-size: 0.875rem; color: #2E5BFF;">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" style="vertical-align: middle; margin-right: 5px;">
                    <circle cx="12" cy="12" r="10" stroke="#2E5BFF" stroke-width="2"/>
                    <line x1="12" y1="8" x2="12" y2="16" stroke="#2E5BFF" stroke-width="2"/>
                    <line x1="12" y1="8" x2="12" y2="8" stroke="#2E5BFF" stroke-width="2"/>
                </svg>
                TradePulse scans markets for trading signals using RSI and EMA indicators.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Version info
        st.markdown("""
        <div style="text-align: center; font-size: 0.75rem; color: #9CA3AF; margin-top: 1rem;">
            Slater Stockbot v1.0.0
        </div>
        """, unsafe_allow_html=True)
    
    # Main page content
    # Dashboard header
    st.markdown("""
    <div class="dashboard-header animate-fade-in">
        <h1 class="dashboard-title">
            <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <rect x="2" y="3" width="20" height="14" rx="2" ry="2"></rect>
                <line x1="8" y1="21" x2="16" y2="21"></line>
                <line x1="12" y1="17" x2="12" y2="21"></line>
            </svg>
            Slater Stockbot Dashboard
        </h1>
        <div class="last-updated">
            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; margin-right: 5px;">
                <circle cx="12" cy="12" r="10"></circle>
                <polyline points="12 6 12 12 16 14"></polyline>
            </svg>
            Last Updated: <span id="current-time">{}</span>
        </div>
    </div>
    """.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")), unsafe_allow_html=True)
    
    # Display legends in tabs
    st.markdown("<div class='animate-fade-in'>", unsafe_allow_html=True)
    legends_tabs = st.tabs(["Signal Guide", "RSI Guide"])
    
    with legends_tabs[0]:
        display_signal_legend()
    
    with legends_tabs[1]:
        display_rsi_guide()
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Create placeholder for results
    results_placeholder = st.empty()
    
    # Create placeholder for top charts
    charts_placeholder = st.empty()
    
    # Progress bar for scanning
    if selected_categories:
        # Count total tickers to scan
        total_tickers = sum(len(TICKER_CATEGORIES[cat]) for cat in selected_categories)
        
        scan_message = st.markdown(f"""
        <div style="display: flex; align-items: center; margin-bottom: 1rem; color: #2E5BFF;">
            <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="margin-right: 8px;">
                <circle cx="12" cy="12" r="10"></circle>
                <polyline points="12 6 12 12 16 14"></polyline>
            </svg>
            <span style="font-weight: 500;">Scanning {total_tickers} markets...</span>
        </div>
        """, unsafe_allow_html=True)
        
        with st.spinner(''):
            # Create a progress bar
            progress_bar = st.progress(0)
            
            # Collect all results
            all_results = []
            category_results = {cat: [] for cat in selected_categories}
            tickers_scanned = 0
            
            # Scan each selected category
            for category in selected_categories:
                if category in TICKER_CATEGORIES:
                    category_tickers = TICKER_CATEGORIES[category]
                    
                    # Scan each ticker in the category
                    for ticker, name in category_tickers.items():
                        result = scan_ticker(ticker, name)
                        result["category"] = category  # Add category info
                        all_results.append(result)
                        category_results[category].append(result)
                        
                        # Update progress
                        tickers_scanned += 1
                        progress_bar.progress(tickers_scanned / total_tickers)
            
            # Remove progress elements when done
            progress_bar.empty()
            scan_message.empty()
            
            # Calculate market metrics for sidebar
            valid_results = [r for r in all_results if not r.get("error")]
            if valid_results:
                bullish_count = sum(1 for r in valid_results if r["daily_status"] == "Bullish" and r["weekly_status"] == "Bullish")
                bearish_count = sum(1 for r in valid_results if r["daily_status"] == "Bearish" and r["weekly_status"] == "Bearish")
                mixed_count = len(valid_results) - bullish_count - bearish_count
                
                # Display metrics in sidebar
                with market_metrics.container():
                    st.markdown("""
                    <div style="display: flex; justify-content: space-between; gap: 10px; margin-bottom: 1rem;">
                    """, unsafe_allow_html=True)
                    
                    metrics_cols = st.columns(3)
                    
                    bull_percent = f"{bullish_count/len(valid_results)*100:.1f}%"
                    bear_percent = f"{bearish_count/len(valid_results)*100:.1f}%"
                    mixed_percent = f"{mixed_count/len(valid_results)*100:.1f}%"
                    
                    metrics_cols[0].markdown(f"""
                    <div class="metric-card" style="border-left: 3px solid #00C48C;">
                        <p class="metric-label">Bullish</p>
                        <p class="metric-value" style="color: #00C48C;">{bullish_count}</p>
                        <p style="font-size: 0.75rem; color: #6B7280; margin: 0;">{bull_percent}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    metrics_cols[1].markdown(f"""
                    <div class="metric-card" style="border-left: 3px solid #FF5252;">
                        <p class="metric-label">Bearish</p>
                        <p class="metric-value" style="color: #FF5252;">{bearish_count}</p>
                        <p style="font-size: 0.75rem; color: #6B7280; margin: 0;">{bear_percent}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    metrics_cols[2].markdown(f"""
                    <div class="metric-card" style="border-left: 3px solid #FFB74D;">
                        <p class="metric-label">Mixed</p>
                        <p class="metric-value" style="color: #FFB74D;">{mixed_count}</p>
                        <p style="font-size: 0.75rem; color: #6B7280; margin: 0;">{mixed_percent}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("""</div>""", unsafe_allow_html=True)
            
            # Sort all results by bullish score (most bullish first)
            all_results.sort(key=lambda x: x.get("score", -1000), reverse=True)
            
            # Sort category results
            for cat in category_results:
                category_results[cat] = [r for r in category_results[cat] if not r.get("error")]
                category_results[cat].sort(key=lambda x: x.get("score", -1000), reverse=True)
        
        # Display the results in the main area
        with results_placeholder.container():
            st.markdown("<div class='animate-fade-in'>", unsafe_allow_html=True)
            # Format the data into a pretty table
            if valid_results:
                # Create tabs for All, Categories, and Signal Categories
                tab_names = ["All Markets"] + selected_categories + ["Signal Categories"]
                tabs = st.tabs(tab_names)
                
                # All Markets tab
                with tabs[0]:
                    display_data = []
                    for r in all_results:
                        if not r.get("error"):
                            display_data.append({
                                "Signal": r["emoji"],
                                "Market": r["display_name"],
                                "Daily": r["daily_status"],
                                "Weekly": r["weekly_status"],
                                "EMA": r["ema_status"],
                                "Price": f"{r['price']:.4f}",
                                "Change %": f"{r['pct_change']:.2f}",
                                "Daily RSI": f"{r['daily_rsi']:.0f}",
                                "Weekly RSI": f"{r['weekly_rsi']:.0f}"
                            })
                    
                    if display_data:
                        df = pd.DataFrame(display_data)
                        st.markdown('<div class="dataframe-container">', unsafe_allow_html=True)
                        st.dataframe(
                            format_dataframe(df),
                            use_container_width=True,
                            height=500,
                            hide_index=True
                        )
                        st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        st.info("No data available for selected markets.")
                
                # Category tabs
                for i, category in enumerate(selected_categories, 1):
                    with tabs[i]:
                        cat_display_data = []
                        for r in category_results[category]:
                            if not r.get("error"):
                                cat_display_data.append({
                                    "Signal": r["emoji"],
                                    "Market": r["display_name"],
                                    "Daily": r["daily_status"],
                                    "Weekly": r["weekly_status"],
                                    "EMA": r["ema_status"],
                                    "Price": f"{r['price']:.4f}",
                                    "Change %": f"{r['pct_change']:.2f}",
                                    "Daily RSI": f"{r['daily_rsi']:.0f}",
                                    "Weekly RSI": f"{r['weekly_rsi']:.0f}"
                                })
                        
                        if cat_display_data:
                            cat_df = pd.DataFrame(cat_display_data)
                            st.markdown('<div class="dataframe-container">', unsafe_allow_html=True)
                            st.dataframe(
                                format_dataframe(cat_df),
                                use_container_width=True,
                                height=400,
                                hide_index=True
                            )
                            st.markdown('</div>', unsafe_allow_html=True)
                        else:
                            st.info(f"No data available for {category}.")
                
                # Signal Categories tab
                with tabs[-1]:  # The last tab (Signal Categories)
                    signal_subtabs = st.tabs(["🚀 Bulls", "🕣 Waiting", "⚠️ Caution", "💀 Bears"])
                    
                    # Group results by signal type
                    rocket_results = [r for r in valid_results if r["emoji"] == "🚀🚀"]
                    clock_results = [r for r in valid_results if r["emoji"] == "🕣🕣"]
                    warning_results = [r for r in valid_results if r["emoji"] == "⚠️⚠️"]
                    death_results = [r for r in valid_results if r["emoji"] == "💀💀"]
                    
                    # Bulls subtab (rocket emoji)
                    with signal_subtabs[0]:
                        if rocket_results:
                            rocket_data = []
                            for r in rocket_results:
                                rocket_data.append({
                                    "Signal": r["emoji"],
                                    "Market": r["display_name"],
                                    "Daily": r["daily_status"],
                                    "Weekly": r["weekly_status"],
                                    "EMA": r["ema_status"],
                                    "Price": f"{r['price']:.4f}",
                                    "Change %": f"{r['pct_change']:.2f}",
                                    "Daily RSI": f"{r['daily_rsi']:.0f}",
                                    "Weekly RSI": f"{r['weekly_rsi']:.0f}"
                                })
                            
                            rocket_df = pd.DataFrame(rocket_data)
                            st.markdown('<div class="dataframe-container">', unsafe_allow_html=True)
                            st.dataframe(
                                format_dataframe(rocket_df),
                                use_container_width=True,
                                height=400,
                                hide_index=True
                            )
                            st.markdown('</div>', unsafe_allow_html=True)
                            st.markdown(f"<p style='text-align: right; color: #6B7280; font-size: 0.875rem;'>{len(rocket_results)} markets found</p>", unsafe_allow_html=True)
                        else:
                            st.info("No markets with 🚀🚀 signals found.")
                    
                    # Waiting subtab (clock emoji)
                    with signal_subtabs[1]:
                        if clock_results:
                            clock_data = []
                            for r in clock_results:
                                clock_data.append({
                                    "Signal": r["emoji"],
                                    "Market": r["display_name"],
                                    "Daily": r["daily_status"],
                                    "Weekly": r["weekly_status"],
                                    "EMA": r["ema_status"],
                                    "Price": f"{r['price']:.4f}",
                                    "Change %": f"{r['pct_change']:.2f}",
                                    "Daily RSI": f"{r['daily_rsi']:.0f}",
                                    "Weekly RSI": f"{r['weekly_rsi']:.0f}"
                                })
                            
                            clock_df = pd.DataFrame(clock_data)
                            st.markdown('<div class="dataframe-container">', unsafe_allow_html=True)
                            st.dataframe(
                                format_dataframe(clock_df),
                                use_container_width=True,
                                height=400,
                                hide_index=True
                            )
                            st.markdown('</div>', unsafe_allow_html=True)
                            st.markdown(f"<p style='text-align: right; color: #6B7280; font-size: 0.875rem;'>{len(clock_results)} markets found</p>", unsafe_allow_html=True)
                        else:
                            st.info("No markets with 🕣🕣 signals found.")
                    
                    # Caution subtab (warning emoji)
                    with signal_subtabs[2]:
                        if warning_results:
                            warning_data = []
                            for r in warning_results:
                                warning_data.append({
                                    "Signal": r["emoji"],
                                    "Market": r["display_name"],
                                    "Daily": r["daily_status"],
                                    "Weekly": r["weekly_status"],
                                    "EMA": r["ema_status"],
                                    "Price": f"{r['price']:.4f}",
                                    "Change %": f"{r['pct_change']:.2f}",
                                    "Daily RSI": f"{r['daily_rsi']:.0f}",
                                    "Weekly RSI": f"{r['weekly_rsi']:.0f}"
                                })
                            
                            warning_df = pd.DataFrame(warning_data)
                            st.markdown('<div class="dataframe-container">', unsafe_allow_html=True)
                            st.dataframe(
                                format_dataframe(warning_df),
                                use_container_width=True,
                                height=400,
                                hide_index=True
                            )
                            st.markdown('</div>', unsafe_allow_html=True)
                            st.markdown(f"<p style='text-align: right; color: #6B7280; font-size: 0.875rem;'>{len(warning_results)} markets found</p>", unsafe_allow_html=True)
                        else:
                            st.info("No markets with ⚠️⚠️ signals found.")
                    
                    # Bears subtab (death emoji)
                    with signal_subtabs[3]:
                        if death_results:
                            death_data = []
                            for r in death_results:
                                death_data.append({
                                    "Signal": r["emoji"],
                                    "Market": r["display_name"],
                                    "Daily": r["daily_status"],
                                    "Weekly": r["weekly_status"],
                                    "EMA": r["ema_status"],
                                    "Price": f"{r['price']:.4f}",
                                    "Change %": f"{r['pct_change']:.2f}",
                                    "Daily RSI": f"{r['daily_rsi']:.0f}",
                                    "Weekly RSI": f"{r['weekly_rsi']:.0f}"
                                })
                            
                            death_df = pd.DataFrame(death_data)
                            st.markdown('<div class="dataframe-container">', unsafe_allow_html=True)
                            st.dataframe(
                                format_dataframe(death_df),
                                use_container_width=True,
                                height=400,
                                hide_index=True
                            )
                            st.markdown('</div>', unsafe_allow_html=True)
                            st.markdown(f"<p style='text-align: right; color: #6B7280; font-size: 0.875rem;'>{len(death_results)} markets found</p>", unsafe_allow_html=True)
                        else:
                            st.info("No markets with 💀💀 signals found.")
            else:
                st.warning("No valid results found. Check your internet connection or try different markets.")
            st.markdown("</div>", unsafe_allow_html=True)
        # Show charts for top performers if requested
        if show_charts and valid_results:
            with charts_placeholder.container():
                st.markdown("<div class='animate-fade-in'>", unsafe_allow_html=True)
                # Create tabs for bulls and bears
                bull_bear_tabs = st.tabs(["Top Bulls", "Top Bears"])
                
                # Top Bulls Tab
                with bull_bear_tabs[0]:
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.subheader("📈 Top Bulls")
                    
                    # Filter and sort by highest RSI for bulls
                    bullish_results = [r for r in valid_results 
                                      if r["daily_status"] == "Bullish" and r["weekly_status"] == "Bullish"]
                    bullish_results.sort(key=lambda x: x.get("score", -1000), reverse=True)
                    
                    # Show top 3 (or fewer if not enough)
                    top_n = min(3, len(bullish_results))
                    
                    if top_n > 0:
                        # Create columns for charts
                        cols = st.columns(top_n)
                        
                        # Display each chart in its own column
                        for i in range(top_n):
                            with cols[i]:
                                chart = create_chart(bullish_results[i])
                                if chart:
                                    st.plotly_chart(chart, use_container_width=True)
                                    
                                    # Add key metrics below chart in a card
                                    st.markdown(f"""
                                    <div style="display: flex; justify-content: space-between; gap: 10px; margin-top: 10px;">
                                        <div style="flex: 1; background-color: #e6f4ea; border-radius: 8px; padding: 10px; text-align: center;">
                                            <p style="font-size: 0.75rem; color: #1F2937; margin: 0;">Daily RSI</p>
                                            <p style="font-size: 1.25rem; font-weight: 600; color: #00C48C; margin: 5px 0;">{bullish_results[i]['daily_rsi']:.0f}</p>
                                            <p style="font-size: 0.7rem; color: #6B7280; margin: 0;">+{bullish_results[i]['daily_rsi'] - 50:.0f} from 50</p>
                                        </div>
                                        <div style="flex: 1; background-color: #e6f4ea; border-radius: 8px; padding: 10px; text-align: center;">
                                            <p style="font-size: 0.75rem; color: #1F2937; margin: 0;">Weekly RSI</p>
                                            <p style="font-size: 1.25rem; font-weight: 600; color: #00C48C; margin: 5px 0;">{bullish_results[i]['weekly_rsi']:.0f}</p>
                                            <p style="font-size: 0.7rem; color: #6B7280; margin: 0;">+{bullish_results[i]['weekly_rsi'] - 50:.0f} from 50</p>
                                        </div>
                                        <div style="flex: 1; background-color: {'#e6f4ea' if bullish_results[i]['pct_change'] > 0 else '#feeaeb'}; border-radius: 8px; padding: 10px; text-align: center;">
                                            <p style="font-size: 0.75rem; color: #1F2937; margin: 0;">Change</p>
                                            <p style="font-size: 1.25rem; font-weight: 600; color: {'#00C48C' if bullish_results[i]['pct_change'] > 0 else '#FF5252'}; margin: 5px 0;">{bullish_results[i]['pct_change']:.2f}%</p>
                                            <p style="font-size: 0.7rem; color: #6B7280; margin: 0;">Today</p>
                                        </div>
                                    </div>
                                    """, unsafe_allow_html=True)
                    else:
                        st.info("No strong bullish instruments found.")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Top Bears Tab
                with bull_bear_tabs[1]:
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.subheader("📉 Top Bears")
                    
                    # Filter and sort by lowest RSI for bears
                    bearish_results = [r for r in valid_results 
                                      if r["daily_status"] == "Bearish" and r["weekly_status"] == "Bearish"]
                    bearish_results.sort(key=lambda x: x.get("score", 1000))
                    
                    # Show top 3 (or fewer if not enough)
                    top_n = min(3, len(bearish_results))
                    
                    if top_n > 0:
                        # Create columns for charts
                        cols = st.columns(top_n)
                        
                        # Display each chart in its own column
                        for i in range(top_n):
                            with cols[i]:
                                chart = create_chart(bearish_results[i])
                                if chart:
                                    st.plotly_chart(chart, use_container_width=True)
                                    
                                    # Add key metrics below chart
                                    st.markdown(f"""
                                    <div style="display: flex; justify-content: space-between; gap: 10px; margin-top: 10px;">
                                        <div style="flex: 1; background-color: #feeaeb; border-radius: 8px; padding: 10px; text-align: center;">
                                            <p style="font-size: 0.75rem; color: #1F2937; margin: 0;">Daily RSI</p>
                                            <p style="font-size: 1.25rem; font-weight: 600; color: #FF5252; margin: 5px 0;">{bearish_results[i]['daily_rsi']:.0f}</p>
                                            <p style="font-size: 0.7rem; color: #6B7280; margin: 0;">{bearish_results[i]['daily_rsi'] - 50:.0f} from 50</p>
                                        </div>
                                        <div style="flex: 1; background-color: #feeaeb; border-radius: 8px; padding: 10px; text-align: center;">
                                            <p style="font-size: 0.75rem; color: #1F2937; margin: 0;">Weekly RSI</p>
                                            <p style="font-size: 1.25rem; font-weight: 600; color: #FF5252; margin: 5px 0;">{bearish_results[i]['weekly_rsi']:.0f}</p>
                                            <p style="font-size: 0.7rem; color: #6B7280; margin: 0;">{bearish_results[i]['weekly_rsi'] - 50:.0f} from 50</p>
                                        </div>
                                        <div style="flex: 1; background-color: {'#e6f4ea' if bearish_results[i]['pct_change'] > 0 else '#feeaeb'}; border-radius: 8px; padding: 10px; text-align: center;">
                                            <p style="font-size: 0.75rem; color: #1F2937; margin: 0;">Change</p>
                                            <p style="font-size: 1.25rem; font-weight: 600; color: {'#00C48C' if bearish_results[i]['pct_change'] > 0 else '#FF5252'}; margin: 5px 0;">{bearish_results[i]['pct_change']:.2f}%</p>
                                            <p style="font-size: 0.7rem; color: #6B7280; margin: 0;">Today</p>
                                        </div>
                                    </div>
                                    """, unsafe_allow_html=True)
                    else:
                        st.info("No strong bearish instruments found.")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
        
        # Show errors if any
        errors = [r for r in all_results if r.get("error")]
        if errors:
            with st.expander("Show Errors", expanded=False):
                for e in errors:
                    st.error(f"{e['display_name']}: {e['error']}")
    
    else:
        st.warning("Please select at least one category to scan.")
    
    # Set up auto-refresh
    st.markdown("---")
    countdown = st.empty()
    
    # Add the refresh timer with animation
    refresh_time = datetime.now() + timedelta(minutes=refresh_interval)
    countdown.markdown(f"""
    <div class="refresh-timer">
        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; margin-right: 5px;">
            <circle cx="12" cy="12" r="10"></circle>
            <polyline points="12 6 12 12 16 14"></polyline>
        </svg>
        <span>Next refresh at {refresh_time.strftime('%H:%M:%S')}</span>
    </div>
    
    <script>
        // Update current time every second
        setInterval(function() {{
            const now = new Date();
            const timeStr = now.toLocaleTimeString();
            document.getElementById('current-time').textContent = now.toISOString().split('T')[0] + ' ' + timeStr;
        }}, 1000);
    </script>
    """, unsafe_allow_html=True)
    
    # Schedule the next refresh
    if "last_refresh" not in st.session_state:
        st.session_state.last_refresh = datetime.now()
    
    # Check if it's time to refresh
    if datetime.now() >= st.session_state.last_refresh + timedelta(minutes=refresh_interval):
        st.session_state.last_refresh = datetime.now()
        st.rerun()

if __name__ == "__main__":
    main()
