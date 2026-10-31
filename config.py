"""
Zephyr – Global configuration
Intelligent Stock Market Forecasting & News Sentiment Decision Support System
"""

from pathlib import Path

# Paths
ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "data"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
NOTEBOOKS_DIR = ROOT_DIR / "notebooks"

ARTIFACTS_DIR.mkdir(exist_ok=True)

# Project metadata
PROJECT_NAME = "Zephyr"
PROJECT_TITLE = "Zephyr — Intelligent Stock Market Forecasting & News Sentiment Decision Support System"
VERSION = "1.0.0"
DISCLAIMER = (
    "Educational project only. Not financial advice. "
    "Past performance is not indicative of future results. "
    "Ticker coverage, intraday history and quote delays can change."
)

# 65 High-Profile Company Universe
TICKERS = {
    "United States": [
        # Mega-cap Tech
        "AAPL", "MSFT", "NVDA", "GOOGL", "GOOG", "AMZN", "META", "TSLA",
        "AVGO", "ORCL", "CRM", "ADBE", "AMD", "INTC", "QCOM", "TXN",
        "NOW", "INTU", "PANW", "SNOW", "PLTR", "UBER", "SHOP",

        # Top Finance & Payments
        "JPM", "BAC", "GS", "MS", "V", "MA", "AXP", "BLK", "SPGI", "MCO",

        # Elite Healthcare / Biotech
        "LLY", "UNH", "JNJ", "ABBV", "MRK", "TMO", "DHR", "ISRG", "VRTX", "REGN",

        # Consumer & Others
        "COST", "NKE", "SBUX", "MCD", "DIS", "NFLX", "BKNG"
    ],
    "United Kingdom": [
        "AZN.L", "SHEL.L", "HSBA.L", "ULVR.L", "BP.L", "GSK.L", "RIO.L", "LSEG.L"
    ],
    "Europe": [
        "ASML.AS", "SAP.DE", "SIE.DE", "MC.PA", "OR.PA", "AIR.PA",
        "NESN.SW", "ROG.SW", "NOVO-B.CO", "UBSG.SW", "BMW.DE", "ALV.DE"
    ],
    "Australia": [
        "CBA.AX", "BHP.AX", "CSL.AX", "WBC.AX", "NAB.AX", "ANZ.AX"
    ],
}

ALL_TICKERS = [t for region in TICKERS.values() for t in region]

TICKER_TO_REGION = {
    ticker: region
    for region, tickers in TICKERS.items()
    for ticker in tickers
}

REGIONS = list(TICKERS.keys())

# Data & refresh settings
REFRESH_OPTIONS = {
    "Every 1 minute": 60,
    "Every 5 minutes": 300,
}
DEFAULT_REFRESH = "Every 5 minutes"

CACHE_TTL_SECONDS = {
    "Every 1 minute": 45,
    "Every 5 minutes": 240,
}

DEFAULT_HISTORY_PERIOD = "2y"
INTRADAY_INTERVALS = ["1m", "5m"]
MAX_INTRADAY_DAYS = 7

# Forecasting
DEFAULT_HORIZON = 1
HORIZON_OPTIONS = [1, 3, 5]

TARGET_TYPE = "close"

ML_MODELS = [
    "RandomForest",
    "ExtraTrees",
    "XGBoost",
    "LightGBM",
    "HistGradientBoosting",
]

DL_MODELS = [
    "LSTM",
    "GRU",
]

BASELINE_MODELS = [
    "NaiveLastClose",
    "NaiveLastReturn",
]

ALL_FORECAST_MODELS = ML_MODELS + DL_MODELS + BASELINE_MODELS

DEFAULT_ML_MODEL = "XGBoost"
DEFAULT_DL_MODEL = "LSTM"
DEFAULT_BASELINE = "NaiveLastClose"

MA_WINDOWS = [5, 10, 20, 50]
VOLATILITY_WINDOW = 20
MOMENTUM_WINDOW = 10

# Evaluation metrics
METRICS = ["MAE", "RMSE", "DirectionalAccuracy"]

# UI / Streamlit
PAGE_TITLE = PROJECT_TITLE
PAGE_ICON = "📈"
LAYOUT = "wide"

# News tab has been removed
TAB_NAMES = [
    "1. Overview",
    "2. Raw Data",
    "3. Time Series + ML",
    "4. Neural Networks",
    "5. Performance Metrics",
    "6. AI Forecast",
]

# Random seed
RANDOM_SEED = 42