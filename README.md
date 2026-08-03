<div align="center">
  <img
    src="https://readme-typing-svg.demolab.com?font=Poppins&weight=700&size=28&duration=3500&pause=1000&color=00CC88&center=true&vCenter=true&width=900&height=140&multiline=true&repeat=false&lines=From+notebook+to+live+markets.;Eight+models.+One+goal.;Measure+what+survives+real+volatility."
    alt="Zephyr"
  />
</div>

<br>

<div align="center">
  <b><i>A dual-mode stock forecasting system — offline research meets live market evaluation.</i></b><br><br>
  Historical analysis on major-tech stocks (2019–2024) · Live yfinance prices + Investing.com news + VADER sentiment
</div>

<br>

<div align="center">
  <a href="https://stocksnews.streamlit.app/" target="_blank" rel="noopener noreferrer">
    <img
      src="https://img.shields.io/badge/Live%20Dashboard-Zephyr-brightgreen?style=for-the-badge&logo=streamlit&logoColor=white&color=00CC88"
      alt="Live Zephyr Dashboard"
    />
  </a>
  &nbsp;
  <img
    src="https://img.shields.io/badge/Models-8-blue?style=for-the-badge&color=0A66C2"
    alt="8 Models"
  />
  &nbsp;
  <img
    src="https://img.shields.io/badge/Sentiment-VADER-purple?style=for-the-badge&color=7C3AED"
    alt="VADER Sentiment"
  />
  <br><br>
  <p>
    Open the live app → 
    <a href="https://stocksnews.streamlit.app/" target="_blank"><b>Zephyr Dashboard</b></a>
  </p>
  <p style="font-size: 0.9rem; opacity: 0.85;">
    No login · Live prices · 8 models · News + sentiment · Downloadable forecasts
  </p>
</div>

---

<div align="center">
  <img
    src="https://readme-typing-svg.demolab.com?font=Poppins&weight=700&size=22&duration=3000&pause=1200&color=00CC88&center=true&vCenter=true&width=600&height=50&lines=About+the+Project"
    alt="About the Project"
  />
</div>

**Zephyr** is a dual-mode stock forecasting system built to answer a simple question:

> **Which models still hold up when they leave the notebook and face live market data?**

It consists of two complementary parts:

1. **Research notebook** (`project2.ipynb`)  
   Full offline pipeline on historical major-tech stocks: data cleaning, feature engineering, cross-stock comparison, classical time-series models, supervised & unsupervised ML, and neural networks.

2. **Live dashboard** (`streamlit.py`)  
   Real-time prices via yfinance, on-demand training of 8 models, consistent backtesting, ensemble forecasts with buy/sell signals, and live news with VADER sentiment analysis.

The focus is transparent evaluation under real conditions — not perfect predictions.

---

<div align="center">
  <img
    src="https://readme-typing-svg.demolab.com?font=Poppins&weight=700&size=22&duration=3000&pause=1200&color=00CC88&center=true&vCenter=true&width=700&height=50&lines=What+Problem+Does+It+Address%3F"
    alt="What Problem Does It Address?"
  />
</div>

Most stock-prediction projects end with strong offline metrics and stop there. Common issues include:

- Results that look good on historical data but degrade on live data
- Inconsistent evaluation windows and metrics
- No easy way to retrain or compare models under the same conditions
- Little visibility into how models react to sudden market moves
- No public, interactive accountability

Zephyr counters this by:

- Running every model on the same data and the same five metrics (MAE, RMSE, MAPE, R², Direction Accuracy)
- Supporting both deep offline research and live inference
- Making forecasts and historical data downloadable
- Adding live news + sentiment for market context
- Treating model underperformance as useful information rather than something to hide

---

<div align="center">
  <img
    src="https://readme-typing-svg.demolab.com?font=Poppins&weight=700&size=22&duration=3000&pause=1200&color=00CC88&center=true&vCenter=true&width=700&height=50&lines=Architecture+—+Two+Modes"
    alt="Architecture — Two Modes"
  />
</div>

### Mode 1 — Research Notebook (`project2.ipynb`)

**Data**  
- Source: `major-tech-stock-2019-2024.csv`  
- Tickers: AAPL, AMZN, GOOGL, MSFT, TSLA  
- Cleaned range: 2019-05-10 → 2023-12-29 (~1,169 trading days per ticker after feature engineering)

**Pipeline**
- Cleaning (duplicates, missing values, forward/back fill per ticker)
- Feature engineering:  
  Daily Return, MA7 / MA30 / MA90, 30-day volatility, Lag Close (1/2/3/5), Momentum 5d/10d, Price Position 20d, Volume MA20 & Ratio, High-Low Range, Typical Price, Overnight Gap, Target Direction
- Cross-stock comparison: cumulative return, annualized return & volatility, Sharpe proxy, max drawdown, correlation matrix
- Classical time series: ADF stationarity test, Holt-Winters, Auto-ARIMA, Prophet
- Supervised ML: Linear / Ridge / Lasso, Polynomial Regression, Random Forest, KNN, LightGBM, XGBoost
- Unsupervised: K-Means, Agglomerative Clustering, Gaussian Mixture, Isolation Forest, PCA, UMAP
- Neural networks: MLPRegressor (scikit-learn), Keras Sequential, Wide & Deep (Keras Functional API)

Built for reproducibility and detailed analysis.

### Mode 2 — Live Dashboard (`streamlit.py` / Zephyr)

**Data**  
- Real-time + 5-year OHLC via yfinance  
- Live 1-minute price (refreshes ~every 30 seconds)  
- Stock groups:  
  - Magnificent 7 → AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA  
  - Top Tech Giants → AMD, NFLX, ADBE, ORCL, CRM, INTC, PYPL, AVGO, QCOM, TXN  
  - AI Chip & Cloud Leaders → TSM, ASML, SNOW, CRWD, PLTR, ARM

**Models trained on demand**

| Tab                  | Models                                      |
|----------------------|---------------------------------------------|
| Time Series + ML     | LightGBM, Prophet, Random Forest, XGBoost, K-Means, Autoencoder |
| Neural Networks      | LSTM, GRU                                   |

**Evaluation & Forecast**
- Backtest metrics: MAE, RMSE, MAPE (%), R², Direction Accuracy (%)
- Ensemble forecast (Prophet + LSTM) weighted by inverse MAE
- Buy / Sell signals from 5-day vs 10-day SMA crossover on the ensemble
- Downloadable historical data and forecast CSVs

**Additional features**
- 7 interactive tabs: Overview, Raw Data, Time Series + ML, Neural Networks, Performance Metrics, AI Forecast, News & Sentiment
- Live news scrape from Investing.com + VADER sentiment  
  Grouped into: Market Movers · Company Updates · Analyst & Macro
- Optional email subscription stored in Google Sheets

Built for real data and transparent comparison.

---

<div align="center">
  <img
    src="https://readme-typing-svg.demolab.com?font=Poppins&weight=700&size=22&duration=3000&pause=1200&color=00CC88&center=true&vCenter=true&width=600&height=50&lines=Models+at+a+Glance"
    alt="Models at a Glance"
  />
</div>

| Family              | Models                                      | Purpose                                   |
|---------------------|---------------------------------------------|-------------------------------------------|
| Time Series         | Prophet                                     | Seasonality-aware baseline                |
| Tree-based ML       | LightGBM, Random Forest, XGBoost            | Lag-feature regression                    |
| Deep Learning       | LSTM, GRU                                   | Sequence models with sliding windows      |
| Unsupervised        | K-Means, Autoencoder                        | Clustering / reconstruction               |
| Research Notebook   | Linear/Ridge/Lasso, Polynomial, RF, KNN, LightGBM, XGBoost, K-Means, GMM, PCA, MLP, Keras NN, Wide & Deep | Broader classical + unsupervised + NN exploration |

All live models are evaluated on the same backtest window and the same five metrics.

---

<div align="center">
  <img
    src="https://readme-typing-svg.demolab.com?font=Poppins&weight=700&size=22&duration=3000&pause=1200&color=00CC88&center=true&vCenter=true&width=500&height=50&lines=Key+Features"
    alt="Key Features"
  />
</div>

- Dual modes: offline research + live forecasting
- 8 models trained and backtested on identical live data
- Consistent metrics across all models
- Inverse-MAE weighted ensemble forecast
- SMA crossover buy/sell signals on the ensemble
- Interactive Streamlit interface with 7 tabs
- Live price ticker (~30 s refresh)
- News + VADER sentiment analysis
- Downloadable history and forecast CSVs
- Clear-cache and retrain controls

---

<div align="center">
  <img
    src="https://readme-typing-svg.demolab.com?font=Poppins&weight=700&size=22&duration=3000&pause=1200&color=00CC88&center=true&vCenter=true&width=450&height=50&lines=Tech+Stack"
    alt="Tech Stack"
  />
</div>

| Category            | Tools                                                                 |
|---------------------|-----------------------------------------------------------------------|
| Language            | Python                                                                |
| Data & Numerics     | Pandas, NumPy                                                         |
| Classical ML        | Scikit-learn, LightGBM, XGBoost                                       |
| Time Series         | Prophet, statsmodels, pmdarima                                        |
| Deep Learning       | TensorFlow / Keras (LSTM, GRU, Autoencoder, Wide & Deep)              |
| Visualization       | Plotly, Matplotlib, Seaborn                                           |
| Live App            | Streamlit                                                             |
| Market Data         | yfinance                                                              |
| News & Sentiment    | requests, BeautifulSoup, VADER                                        |
| Optional Backend    | gspread + Google Service Account (email subscribe)                    |

---

<div align="center">
  <img
    src="https://readme-typing-svg.demolab.com?font=Poppins&weight=700&size=22&duration=3000&pause=1200&color=00CC88&center=true&vCenter=true&width=500&height=50&lines=Data+Sources"
    alt="Data Sources"
  />
</div>

| Mode          | Source                                      | Details                                      |
|---------------|---------------------------------------------|----------------------------------------------|
| Research      | `major-tech-stock-2019-2024.csv`            | AAPL, AMZN, GOOGL, MSFT, TSLA · 2019–2023   |
| Live          | yfinance                                    | 5-year history + 1-minute live price         |
| News          | Investing.com (scraped)                     | Title + snippet → VADER sentiment            |

No external database is required for core forecasting.

---

<div align="center">
  <img
    src="https://readme-typing-svg.demolab.com?font=Poppins&weight=700&size=22&duration=3000&pause=1200&color=00CC88&center=true&vCenter=true&width=450&height=50&lines=How+to+Run"
    alt="How to Run"
  />
</div>

### Live Dashboard (Zephyr)

```bash
pip install -r requirements.txt
streamlit run streamlit.py
```

1. Select stock group and ticker in the sidebar  
2. Set forecast days and lookback window  
3. Train models in the **Time Series + ML** and **Neural Networks** tabs  
4. Review metrics and generate the ensemble forecast  
5. Explore news & sentiment  
6. Download historical data or forecasts  

### Research Notebook

```bash
jupyter notebook project2.ipynb
```

Run cells in order: data loading → cleaning → feature engineering → comparison → classical models → supervised / unsupervised ML → neural networks.

---

<div align="center">
  <img
    src="https://readme-typing-svg.demolab.com?font=Poppins&weight=700&size=22&duration=3000&pause=1200&color=00CC88&center=true&vCenter=true&width=600&height=50&lines=Project+Structure"
    alt="Project Structure"
  />
</div>

```
├── streamlit.py                      # Live Zephyr dashboard
├── project2.ipynb                    # Offline research pipeline
├── major-tech-stock-2019-2024.csv    # Historical research dataset
└── requirements.txt                  # Dependencies
```

---

<div align="center">
  <img
    src="https://readme-typing-svg.demolab.com?font=Poppins&weight=700&size=22&duration=3000&pause=1200&color=00CC88&center=true&vCenter=true&width=650&height=50&lines=Planned+Improvements"
    alt="Planned Improvements"
  />
</div>

- Regime-aware model selection  
- Probabilistic forecasts with confidence intervals  
- Feature importance / SHAP views  
- Event-aware forecasting (earnings, macro releases)  
- Longer-horizon evaluation under sustained volatility  
- More robust multi-source news fallbacks  

---

<div align="center">
  <img
    src="https://readme-typing-svg.demolab.com?font=Poppins&weight=700&size=22&duration=3000&pause=1200&color=00CC88&center=true&vCenter=true&width=400&height=50&lines=Disclaimer"
    alt="Disclaimer"
  />
</div>

**This is not financial advice.**

The system is designed to evaluate forecasting methods under realistic market conditions. It does not generate trading recommendations or guarantee returns. Models can and will be wrong — when they are, the system is doing its job by exposing the failure.

Buy/sell signals shown in the app are simple technical overlays on the ensemble forecast and should not be treated as investment advice.

---

<div align="center">
  <img
    src="https://readme-typing-svg.demolab.com?font=Poppins&weight=700&size=20&duration=4000&pause=1500&color=00CC88&center=true&vCenter=true&width=800&height=60&lines=Built+to+measure+what+survives+contact+with+live+markets."
    alt="Tagline"
  />
</div>
