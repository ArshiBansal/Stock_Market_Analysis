<div align="center">

<img src="assets/zephyr-banner.svg" alt="Zephyr - a record of how the market lab has changed" width="100%">

### The project has moved rooms.

*This changelog compares the current modular Zephyr dashboard with the earlier Streamlit application and research notebook supplied as references.*

</div>

<div align="center">

**BEFORE**　→　**REWORKED (v2)**　→　**CURRENT (v3)**

</div>

<h3 align="center" id="v3-release-notes">v3 — Expanded forecasting, risk, persistence, and portfolio views</h3>

| Area | v2 | v3 (current) |
|---|---|---|
| Model set | Baselines, Random Forest, Extra Trees, HistGradientBoosting, XGBoost, LightGBM, LSTM, GRU | All of the above plus **Prophet** for direct horizon-return regression |
| Risk reporting | MAE, RMSE, directional accuracy | Adds **Sharpe ratio, maximum drawdown, and value-at-risk** |
| Model handling | Models retrained on every forecast click | **Model persistence** — trained models saved to `artifacts/models/` and reused across sessions |
| Cross-tab efficiency | Data re-fetched and re-engineered per tab | **Session-state caching** — supervised datasets cached per (ticker, horizon, window) |
| Evaluation scope | Single-ticker chronological holdout and walk-forward | Adds a **backtesting engine** with position sizing, transaction costs, slippage, and stop-loss |
| Single-ticker focus | One ticker per screen | Adds a **multi-ticker Portfolio** view: correlation matrix, normalized performance, equal-weight risk summary |
| Fundamental context | Latest quote only | Adds **fundamental data** (market cap, P/E, dividend yield, beta, 52-week range) per ticker |

> **Version note:** the project configuration (`config.py`) declares `VERSION = "1.0.0"`. "v3" here refers to the current feature set described in this changelog, not a published Git release tag. Bump `VERSION` and add a release tag when you ship a formal release.

<h3 align="center" id="v4-release-notes">v4 — Tab restructure, credibility & rigor, and the Scenario Lab</h3>

| Area | v3 | v4 (current) |
|---|---|---|
| Dashboard tabs | 7 tabs: Overview, Raw Data, Time Series + ML, Neural Networks, Performance Metrics, AI Forecast, Portfolio | **6 tabs:** Overview (absorbs Raw Data), **Models & Evaluation** (merges Time Series+ML, Neural Networks, Performance), AI Forecast, Portfolio, **Research**, **Scenario Lab** |
| Uncertainty | Point forecast only | **Split-conformal prediction intervals** with a calibrated band on the AI Forecast chart and empirical coverage |
| Backtest | Engine existed but was not exposed in any UI | **Backtest tab in Research** — model signals → equity curve, win rate, Sharpe, drawdown, VaR, with costs/slippage/stop-loss and a non-overlapping stride mode |
| Explainability | None | **Feature importance** via permutation importance (always) or **SHAP** (optional), with global and latest-observation local contributions |
| Tuning | Experiment defaults only | **Hyperparameter tuning** with `RandomizedSearchCV` over time-series CV folds |
| Reproducibility | Transient in-memory results | **Experiment registry** — append-only JSONL at `artifacts/experiments/`, browsable and comparable in the Research hub |
| Scenarios | None | **Scenario Lab** — Monte Carlo fan chart, first-passage target/stop probability, volatility-regime map, and interactive shock what-if |
| Dependencies | — | Adds optional `shap` for explainability |

<details>
<summary><strong>Choose a chapter</strong></summary>

- [The shape of the change](#the-shape-of-the-change)
- [What changed across the project](#what-changed-across-the-project)
- [Dashboard and model comparison](#dashboard-and-model-comparison)
- [Notebook research compared with the current app](#notebook-research-compared-with-the-current-app)
- [Features no longer in the active dashboard](#features-no-longer-in-the-active-dashboard)
- [What is not claimed](#what-is-not-claimed)
- [Related guides](#related-guides)

</details>

<h2 align="center" id="the-shape-of-the-change">The shape of the change</h2>

The earlier app was a **single, all-in-one Streamlit script**, with broad interactive forecasting, market-analysis, news, and subscription features. The supplied notebook was a wide-ranging, multi-ticker research workspace. The current repository organizes the dashboard into UI, data, model, NLP, and utility modules, and makes its main forecasting workflow more explicitly about daily observations, feature-based horizon returns, and chronological evaluation.

That change is principally one of **structure and scope**. This comparison does not establish that the current models are more accurate, profitable, or reliable.

<h2 align="center" id="what-changed-across-the-project">What changed across the project</h2>

| Area | Earlier references | Current repository |
|---|---|---|
| Application shape | A large `streamlit.py` file contained page setup, fetching, analysis, model code, charts, news, and subscription handling. | `app.py` composes seven `ui/` views; data, features, models, configuration, and metrics have separate packages. |
| Main market-data workflow | The dashboard loaded five years of daily history and also requested a one-minute live-price series with a 25-second cache. | Dashboard screens fetch daily history for a selected date range, defaulting to about two years; quotes and history are fetched through `yfinance`. |
| Ticker selection | Three technology-focused groups covering 23 listed symbols. | A configured, curated set across the United States, United Kingdom, Europe, and Australia. |
| Feature design | The notebook worked with a multi-ticker CSV and engineered grouped features such as moving averages, lagged closes, and direction labels. The app used close-price lags for selected models. | Reusable per-ticker features include returns and lags, moving averages, volatility, momentum, candle shape, and volume ratios; the target is a selected-horizon close-to-close return. |
| Model organization | The app bundled several different model/task types into a “Train All 6” action; the notebook explored additional time-series, supervised, unsupervised, and neural-network methods. | Model factories and training utilities separate baseline, classical ML, and sequence-model experiments. |
| Evaluation | The old app reported MAE, RMSE, MAPE, R², and a direction score; the notebook included model metrics and analytical/statistical tests. | Shared comparison screens report MAE, RMSE, and return-direction accuracy on chronological holdouts; the Performance Metrics screen also offers walk-forward validation. |
| Documentation and tests | The supplied app and notebook were the main reference artifacts for the earlier workflow. | The repository includes focused data, feature, and metric tests plus README, architecture, methodology, and limitations guides. |

<h2 align="center" id="dashboard-and-model-comparison">Dashboard and model comparison</h2>

### The earlier app had seven tabs

The supplied `streamlit.py` provided Overview, Raw Data, Time Series + ML, Neural Networks, Performance Metrics, AI Forecast, and News & Sentiment. Its Overview included price, volume, market-cap and P/E context, moving averages, volatility, return distribution, and cumulative-return charts.

### The current app has seven focused tabs

| Current tab | How it differs from the earlier app |
|---|---|
| **Overview** | Focuses on the latest available quote, daily change, volume, day range, exchange/region, and data age. |
| **Raw Data** | Adds explicit quality reporting, selectable missing-value handling, and a cleaned-history CSV download. |
| **Time Series + ML** | Uses engineered return features and compares selected baselines and classical regressors on a chronological split. |
| **Neural Networks** | Provides configurable LSTM/GRU sequence length and epochs, training/validation curves, and return predictions. |
| **Performance Metrics** | Uses MAE, RMSE, and directional accuracy for comparable horizon-specific return experiments, with optional expanding-window walk-forward validation; this differs from the earlier app's five-metric price-oriented scorecard. |
| **AI Forecast** | Trains the selected supported model for a direct 1-, 3-, or 5-trading-day cumulative return estimate and displays holdout metrics. Trained classical ML models are persisted to `artifacts/models/` and reused across sessions. |
| **Portfolio** | Multi-ticker view: correlation matrix, normalized equal-weight performance, Sharpe ratio, max drawdown, VaR, and per-ticker fundamentals. |

### Model names are not a benchmark result

The earlier Streamlit app's “Train All 6 Models” action grouped LightGBM, Prophet, Random Forest, XGBoost, K-Means clustering, and an autoencoder experiment. The supplied notebook separately explored classical time-series methods, regressors, boosting, clustering, anomaly detection, dimensionality reduction, mixture models, and neural networks.

The current supervised dashboard instead offers last-close/last-return baselines, Random Forest, Extra Trees, HistGradientBoosting, XGBoost, LightGBM, Prophet, and separate LSTM/GRU experiments. These model lists are **not directly comparable**: target definitions, features, samples, forecast horizons, and evaluation procedures changed. No numerical claim that one snapshot outperforms the other is made here.

The `NaiveLastClose` baseline represents a zero-return prediction, and `NaiveLastReturn` uses each row's latest known one-day return. The shared horizon selection now changes the supervised return target and evaluation across forecast screens.

<h2 align="center" id="notebook-research-compared-with-the-current-app">Notebook research compared with the current app</h2>

The supplied notebook was an 81-cell research notebook and expected a local CSV file. Its work included:

- Multi-ticker cleaning and grouped feature engineering.
- Cross-stock comparisons, return/volatility summaries, and ticker correlations.
- ADF stationarity checks and SARIMA/Prophet experiments.
- Linear, Ridge, Lasso, Random Forest, LightGBM, and XGBoost regression experiments.
- K-Means, Isolation Forest, PCA, and Gaussian-mixture explorations.
- MLP, Keras neural-network, and wide-and-deep experiments.

The current dashboard is primarily a **single selected ticker at a time** and fetches its observations at runtime instead of depending on that notebook CSV. Some themes remain—feature engineering, chronological evaluation, tree models, and neural networks—but the current user interface does not reproduce the notebook's full cross-ticker, statistical, clustering, anomaly, or dimensionality-reduction analyses.

<h2 align="center" id="features-no-longer-in-the-active-dashboard">Features no longer in the active dashboard</h2>

The following features existed in the supplied earlier application but are **not available as equivalent active dashboard flows** in the current repository:

| Earlier capability | Current status |
|---|---|
| One-minute live-price ticker with short cache refresh | Replaced by latest-available quote/history retrieval and a data-age indicator; the current Overview is not a one-minute live feed. |
| Selectable 10–180 business-day forecast and lookback controls | Replaced by 1-, 3-, and 5-trading-day direct cumulative-return estimates; these are not recursively generated daily paths. |
| Multi-model future-price ensemble weighted by inverse MAE | Not present in the current AI Forecast screen. |
| Forecast-based BUY/SELL markers from moving-average crossovers | Not present. Current project output is educational model analysis, not a trading signal service. |
| Prophet-based future path in the Streamlit forecast tab | **Added in v3** — Prophet is available as a classical ML model across the forecast and comparison screens. It estimates the selected cumulative horizon return rather than a recursively generated price path. |
| Dedicated Investing.com news scraping and categorized VADER news cards | The current repository has separate `nlp/` helpers for Yahoo Finance headlines and VADER/FinBERT, but no News & Sentiment tab and no direct equivalent of the earlier scraper presentation. |
| Email newsletter subscription backed by Google Sheets | Not present in the current dashboard. |
| Notebook's CSV-backed cross-stock exploratory analyses | Not presented as current dashboard screens; the supplied source CSV is not part of the current project's runtime workflow. |

<h2 align="center" id="what-is-not-claimed">What is not claimed</h2>

- This file does not reconstruct a Git-based release timeline: the supplied earlier artifacts do not identify release dates or version numbers.
- It does not claim that removed features were all intentionally deprecated or that every experiment in the notebook produced a validated result.
- It does not compare historical model scores across snapshots. Their datasets, target types, splits, horizons, and model implementations are not a controlled match.
- It does not claim future returns, profitable strategies, or predictive improvements.

<h2 align="center" id="related-guides">Related guides</h2>

<div align="center">

[README](README.md)　·　[Architecture](ARCHITECTURE.md)　·　[Methodology](METHODOLOGY.md)　·　[Limitations](LIMITATIONS.md)

<sub>Compared from the earlier user-provided app and notebook against the current repository snapshot.</sub>

</div>