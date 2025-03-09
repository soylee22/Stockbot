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
