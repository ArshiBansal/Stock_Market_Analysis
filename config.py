"""
Zephyr – Global configuration
Intelligent Stock Market Forecasting
"""

from pathlib import Path

# Paths
ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "data"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
MODEL_ARTIFACTS_DIR = ARTIFACTS_DIR / "models"
EXPERIMENTS_DIR = ARTIFACTS_DIR / "experiments"

def ensure_directories() -> None:
    """Create required project directories (idempotent)."""
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    MODEL_ARTIFACTS_DIR.mkdir(exist_ok=True)
    EXPERIMENTS_DIR.mkdir(exist_ok=True)


# Project metadata
PROJECT_NAME = "Zephyr"
PROJECT_TITLE = "Zephyr — Intelligent Stock Market Forecasting"
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

# Forecasting
DEFAULT_HORIZON = 1
HORIZON_OPTIONS = [1, 3, 5]

# Risk metrics
RISK_FREE_RATE = 0.02
TRADING_DAYS = 252
CONFIDENCE_LEVEL = 0.95

ML_MODELS = [
    "RandomForest",
    "ExtraTrees",
    "XGBoost",
    "LightGBM",
    "HistGradientBoosting",
    "Prophet",
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

# Restructured dashboard: 6 tabs
TAB_NAMES = [
    "1. Overview",
    "2. Models & Evaluation",
    "3. AI Forecast",
    "4. Portfolio",
    "5. Research",
    "6. Scenario Lab",
]

# Sub-navigation within merged tabs
MODEL_EVAL_SECTIONS = ["Classical ML", "Neural Networks", "Evaluation"]
RESEARCH_SECTIONS = ["Explainability", "Hyperparameter Tuning", "Backtest", "Experiments"]
SCENARIO_SECTIONS = ["Monte Carlo", "Target / Stop", "Volatility Regime", "Shock What-if"]

# ----------------------------------------------------------------------
# Credibility & rigor: prediction intervals
# ----------------------------------------------------------------------
FORECAST_INTERVAL_LEVELS = [0.5, 0.8, 0.9, 0.95]
DEFAULT_INTERVAL_LEVEL = 0.8

# ----------------------------------------------------------------------
# Credibility & rigor: explainability
# ----------------------------------------------------------------------
EXPLAIN_METHODS = ["permutation", "shap"]
SHAP_SAMPLE_SIZE = 100
PERMUTATION_REPEATS = 10

# ----------------------------------------------------------------------
# Credibility & rigor: hyperparameter tuning
# ----------------------------------------------------------------------
TUNING_N_ITER = 20
TUNING_CV_SPLITS = 5

PARAM_GRIDS = {
    "RandomForest": {
        "model__n_estimators": [100, 200, 300],
        "model__max_depth": [4, 6, 8, 10],
        "model__min_samples_leaf": [2, 5, 10],
    },
    "ExtraTrees": {
        "model__n_estimators": [100, 200, 300],
        "model__max_depth": [4, 6, 8, 10],
        "model__min_samples_leaf": [2, 5, 10],
    },
    "HistGradientBoosting": {
        "model__max_iter": [100, 200, 300],
        "model__max_depth": [4, 6, 8],
        "model__learning_rate": [0.03, 0.05, 0.1],
    },
    "XGBoost": {
        "model__n_estimators": [200, 300, 400],
        "model__max_depth": [3, 4, 6],
        "model__learning_rate": [0.03, 0.05, 0.1],
        "model__subsample": [0.7, 0.8, 1.0],
    },
    "LightGBM": {
        "model__n_estimators": [200, 300, 400],
        "model__max_depth": [3, 4, 6],
        "model__learning_rate": [0.03, 0.05, 0.1],
        "model__subsample": [0.7, 0.8, 1.0],
    },
}

# ----------------------------------------------------------------------
# Credibility & rigor: backtesting
# ----------------------------------------------------------------------
BACKTEST_DEFAULTS = dict(
    initial_capital=10_000.0,
    position_size=1.0,
    transaction_cost_bps=1.0,
    slippage_bps=0.5,
    stop_loss=None,
)
DEFAULT_SIGNAL_THRESHOLD = 0.0

# ----------------------------------------------------------------------
# Scenario Lab
# ----------------------------------------------------------------------
MC_PATHS = 2000
MC_HORIZON = 21
MC_METHODS = ["bootstrap", "gbm"]
VOL_REGIME_WINDOW = 20
VOL_REGIME_QUANTILES = (0.33, 0.67)
SCENARIO_SAMPLE_PATHS = 12

# Random seed
RANDOM_SEED = 42