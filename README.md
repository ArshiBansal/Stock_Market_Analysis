# Zephyr — AI Stock Forecasting System

<div align="center">
  <img
    src="https://readme-typing-svg.demolab.com?font=Poppins:ital,wght@1,700&weight=700&size=28&duration=4000&pause=1200&color=00CC88&center=true&vCenter=true&width=950&height=160&multiline=true&repeat=false&lines=Project+crashed.+Motivation+crashed+harder.;Coffee+spilled.+Code+broke.;Out+of+chaos,+a+forecasting+beast+emerged.;So+naturally,+we+built+something+that+might+survive+live+markets."
    alt="Course Project Chaos Origin"
  />
</div>

<br>

<div align="center">
  <b><i>A dual-mode AI forecasting pipeline that leaves the notebook and faces real markets.</i></b><br><br>
  Historical research on Kaggle-style data. Live yfinance prices + news sentiment for accountability.
</div>

<br>

<div align="center">
  <a href="https://stocksnews.streamlit.app/" target="_blank" rel="noopener noreferrer">
    <img
      src="https://img.shields.io/badge/Live%20Forecast-Open%20Dashboard-brightgreen?style=for-the-badge&logo=streamlit&logoColor=white&color=00CC88"
      alt="Live Forecast"
    />
  </a>
  <br><br>
  <p>
    Watch models compete against reality → <a href="https://stocksnews.streamlit.app/" target="_blank"><b>Streamlit App (Zephyr)</b></a>
  </p>
  <p style="font-size: 0.9rem; opacity: 0.85;">
    No login. Live prices. 8 models. News + VADER sentiment. Zero excuses.
  </p>
</div>

---

## About the Project

**Zephyr** is a complete dual-mode stock forecasting system built to answer one question:

> **What actually survives when models leave notebooks and face live markets?**

It combines:

- **Offline research mode** (Jupyter notebook) — full EDA, feature engineering, classical ML, unsupervised learning, and neural nets on historical major-tech stock data (2019–2024)
- **Live forecasting mode** (Streamlit app named Zephyr) — real-time prices via yfinance, on-demand training of 8 models, backtesting, ensemble forecasts with buy/sell signals, and live news + sentiment analysis

The goal is not perfect predictions. The goal is transparent evaluation under real volatility, regime shifts, and market noise — with downloadable receipts.

---

## What Problem Does It Solve?

Most stock-prediction projects stop at impressive offline metrics and pretty charts. They suffer from:

- Results that look great on historical data but collapse on live data
- Cherry-picked windows and metrics
- No continuous backtesting or rolling evaluation
- Almost zero visibility into how models react to sudden market moves
- No public accountability

Zephyr addresses these by:

1. Forcing every model to run on the **same** data and evaluation protocol
2. Supporting both historical research and live inference
3. Providing consistent backtesting metrics (MAE, RMSE, MAPE, R², Direction Accuracy)
4. Making model performance and forecasts downloadable
5. Adding live news + VADER sentiment for market context
6. Treating model failure as useful information rather than something to hide

It turns forecasting from a one-time notebook experiment into an ongoing, testable system.

---

## Architecture — Two Modes

### Mode 1 — Research & Experimentation (`Project.ipynb`)
- Historical major-tech stock dataset (AAPL, MSFT, GOOGL, AMZN, TSLA, etc.)
- Data cleaning, feature engineering (returns, moving averages, volatility, lags)
- Inter-ticker comparison and correlation analysis
- Classical models: Linear / Polynomial Regression, Random Forest
- Unsupervised: K-Means clustering, Association Rules, PCA visualization
- Neural network regression (PyTorch)
- Performance tracking and saved metrics

Built for **reproducibility and deep analysis**.

### Mode 2 — Live Forecasting Engine (`streamlit.py` / Zephyr)
- Real-time OHLC data via **yfinance** (5-year history + live 1-min price)
- Stock groups: Magnificent 7, Top Tech Giants, AI Chip & Cloud Leaders
- On-demand training of 8 models with backtesting
- Ensemble forecast + SMA crossover buy/sell signals
- Interactive tabs: Overview, Raw Data, Time Series + ML, Neural Networks, Performance Metrics, AI Forecast, News & Sentiment
- Downloadable historical data and forecasts
- Live news scraping + VADER sentiment from Investing.com
- Optional email subscription (Google Sheets backend)

Built for **real data and real consequences**.

---

## Models Included

| Family              | Models                                      | Role |
|---------------------|---------------------------------------------|------|
| **Time Series**     | Prophet                                     | Seasonality-aware classical baseline |
| **Machine Learning**| LightGBM, Random Forest, XGBoost            | Tree-based lag pipelines |
| **Deep Learning**   | LSTM, GRU                                   | Sequence models with sliding windows |
| **Unsupervised**    | K-Means, Autoencoder                        | Regime clustering & anomaly-style reconstruction |
| **Research Notebook**| Linear/Polynomial Regression, RF, K-Means, Association Rules, PCA, PyTorch NN | Broader classical + unsupervised exploration |

No model is trusted by default. All must earn credibility on the same evaluation framework — or get exposed by live data.

---

## Key Features

- Dual execution modes: offline research + live forecasting
- 8 models competing on identical live data
- Rolling backtests with 5 consistent metrics (MAE, RMSE, MAPE, R², Direction Accuracy)
- Ensemble forecast weighted by inverse MAE
- SMA crossover buy/sell signals on the ensemble
- Interactive Streamlit dashboard with 7 tabs
- Live price ticker (auto-refreshes ~30s)
- News + VADER sentiment analysis grouped by Market Movers / Company Updates / Analyst & Macro
- Downloadable CSVs for history and forecasts
- Clear cache / retrain buttons for controlled experiments

If a model fails, it fails publicly.  
If it succeeds, it must keep proving it.

---

## Tech Stack

| Category              | Tools |
|-----------------------|-------|
| Language              | Python |
| Data & Numerics       | Pandas, NumPy |
| Classical ML          | Scikit-learn, LightGBM, XGBoost |
| Time Series           | Prophet |
| Deep Learning         | TensorFlow / Keras (LSTM, GRU, Autoencoder), PyTorch (research notebook) |
| Visualization         | Plotly, Matplotlib, Seaborn, Plotly Express |
| Live App              | Streamlit |
| Live Market Data      | yfinance |
| News & Sentiment      | requests, BeautifulSoup, VADER |
| Optional Backend      | gspread + Google Service Account (email subscribe) |

The stack prioritizes reliability and clarity over novelty.

---

## Data

- **Offline / Research mode**: `major-tech-stock-2019-2024.csv` (historical OHLCV for major tech tickers)
- **Live mode**: Real-time and historical OHLC via yfinance (5-year lookback + 1-minute live price)

No external databases required for core forecasting.  
News is scraped on demand. Subscription uses an optional Google Sheet.

---

## How to Run

### Live App (Zephyr)
```bash
pip install -r requirements.txt
streamlit run streamlit.py
```

1. Select stock group and ticker in the sidebar
2. Choose forecast horizon and lookback window
3. Train models (Time Series + ML tab and Neural Networks tab)
4. View performance metrics and generate ensemble forecast
5. Explore news & sentiment
6. Download predictions or historical data

### Research Notebook
```bash
jupyter notebook Project.ipynb
```
Run cells sequentially for full EDA → feature engineering → modeling → evaluation pipeline.

---

## Project Structure (Core Files)

```
├── streamlit.py          # Live Zephyr dashboard
├── Project.ipynb         # Offline research & experimentation
├── major-tech-stock-2019-2024.csv   # Historical dataset (research)
└── requirements.txt      # Dependencies
```

---

## Planned Improvements

- Automated regime-switching model selection
- Probabilistic forecasts with confidence intervals
- Model explainability views (feature importance / SHAP)
- Event-aware forecasting (earnings, macro releases)
- Longer-horizon evaluation under sustained volatility
- More robust news source fallbacks

Markets change. The system is designed to evolve with them.

---

## Disclaimer

**This is not financial advice.**  
Never was. Never will be.

The system exists to evaluate forecasting methods under realistic conditions — not to generate trading signals or guarantee returns. Models can and will be wrong. When they are, the system is doing its job by exposing the failure.

Buy/sell signals shown in the app are simple technical overlays on the ensemble forecast and should not be treated as recommendations.

---

**Built to measure what survives contact with live markets.**
