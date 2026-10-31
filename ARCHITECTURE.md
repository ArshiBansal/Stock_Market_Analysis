<div align="center">

<img src="assets/architecture-banner.svg" alt="Zephyr architecture: Streamlit presentation, Python analysis, and Yahoo Finance services" width="100%">

### A guided walk through the system behind the charts.

*Start at the dashboard. Follow a market-data request through the analysis modules, and see where models and external services join the story.*

</div>

> **Before the tour:** this is a map of the current code, not a proposed redesign. Market observations and predictions remain in application memory; the project does **not** configure a database or model-serving backend.

<div align="center">

**THE SCREENS**　<span style="color:#77B8DF">■ PRESENTATION</span>　→　**THE WORKROOM**　<span style="color:#B6A1E5">■ ANALYSIS</span>　→　**THE OUTSIDE WORLD**　<span style="color:#F19A9F">■ SERVICES</span>

</div>

<details>
<summary><strong>Choose a path through the map</strong></summary>

- [01 · The big picture](#application-architecture)
- [Technology stack](#technology-stack-summary)
- [Data storage and external services](#data-storage-amp-external-services)
- [Architectural choices](#key-architectural-decisions)
- [02 · A closer look at the components](#component-relationships)
- [Component inventory](#component-inventory)
- [Related guides](#related-guides)

</details>

<h2 align="center" id="application-architecture">01 · The big picture</h2>

Think of Zephyr as a small research room. A browser opens the screens; those screens request data and call the Python analysis modules; results return to the dashboard as tables, charts, and metrics. The top-level diagram follows that journey.

<!-- mermaid-checked: no \n, no em-dash/en-dash, no {} in labels, subgraphs are id["label"], arrows are -->|"label"|, all subgraphs closed by end, ids unique -->
```mermaid
flowchart TD
    subgraph User["User"]
        Browser["Web browser"]
    end
    subgraph Presentation["Presentation layer"]
        App["Streamlit application"]
        Sidebar["Shared sidebar filters"]
        Tabs["Six dashboard tabs"]
    end
    subgraph Analysis["Analysis layer"]
        Fetch["Market data fetching"]
        Clean["Cleaning and quality checks"]
        Features["Technical feature engineering"]
        Training["Training and evaluation"]
        Baselines["Baseline regressors"]
        Classical["Classical machine learning"]
        Sequences["TensorFlow LSTM and GRU"]
    end
    subgraph Supporting["Supporting modules"]
        News["News retrieval helpers"]
        Sentiment["VADER and FinBERT sentiment"]
    end
    subgraph Services["External services"]
        Yahoo["Yahoo Finance through yfinance"]
        ModelHub["Hugging Face model hub"]
    end

    Browser -->|"views dashboard"| App
    App -->|"renders"| Sidebar
    App -->|"renders"| Tabs
    Sidebar -->|"selected filters"| Tabs
    Tabs -->|"requests quotes or history"| Fetch
    Fetch -->|"downloads quotes and OHLCV"| Yahoo
    Yahoo -->|"market observations"| Fetch
    Fetch -->|"OHLCV frames"| Clean
    Clean -->|"validated observations"| Features
    Features -->|"feature matrix and target"| Training
    Training -->|"selects reference model"| Baselines
    Training -->|"fits tabular models"| Classical
    Training -->|"fits sequence models"| Sequences
    Baselines -->|"predictions"| Training
    Classical -->|"predictions"| Training
    Sequences -->|"predictions"| Training
    Training -->|"scores and predictions"| Tabs
    News -->|"requests ticker headlines"| Yahoo
    News -->|"sends headlines for scoring"| Sentiment
    Sentiment -.->|"loads pretrained weights when needed"| ModelHub
```

<h3 align="center" id="technology-stack-summary">The tools in the room</h3>

| Layer | Technology | Version | Purpose |
|---|---|---|---|
| Presentation | Python | Not pinned | Application runtime and module implementation |
| Presentation | Streamlit | `>=1.32.0` | Interactive dashboard, sidebar controls, and tab navigation |
| Visualization | Plotly | `>=5.18.0` | Interactive price, comparison, and training charts |
| Data handling | pandas | `>=2.1.0` | OHLCV frames, feature tables, and data transformations |
| Data handling | NumPy | `>=1.26.0` | Numeric operations and model input arrays |
| Market data | yfinance | `>=0.2.36` | Yahoo Finance quotes, daily history, and supporting news retrieval |
| Feature analysis | ta | `>=0.11.0` | Technical-analysis dependency; feature code also computes rolling and lagged measures directly |
| Classical machine learning | scikit-learn | `>=1.4.0` | Preprocessing, regressors, chronological evaluation, and metrics |
| Classical machine learning | XGBoost | `>=2.0.0` | Optional gradient-boosted tree regressor |
| Classical machine learning | LightGBM | `>=4.3.0` | Optional gradient-boosted tree regressor |
| Deep learning | TensorFlow | `>=2.15.0` | LSTM and GRU sequence models |
| Sentiment analysis | vaderSentiment | `>=3.3.2` | Lightweight headline sentiment |
| Sentiment analysis | transformers and PyTorch | `>=4.38.0`; `>=2.1.0` | Optional FinBERT pipeline and model inference |

Versions shown are minimum constraints from `requirements.txt`, not a lockfile or guarantee of compatibility on every operating system.

<h3 align="center" id="data-storage-amp-external-services">Where the data goes</h3>

Yahoo Finance is accessed at runtime through `yfinance` for quotes, historical OHLCV observations, and supporting news retrieval. FinBERT can obtain pretrained model files from the Hugging Face model hub when sentiment analysis is invoked. **There is no database, message broker, or durable market-data store:** fetched data and model results are handled in application memory. The configured `artifacts` directory is created by the application, but the current dashboard does not use it as a persistent model registry.

<h3 align="center" id="key-architectural-decisions">Why the system is arranged this way</h3>

- **Modular Streamlit presentation:** `app.py` composes the shared sidebar and six focused tab renderers in `ui/`.
- **Separated analysis responsibilities:** fetching, cleaning, feature engineering, model factories, and evaluation live in distinct Python packages.
- **Time-aware evaluation:** supervised targets are shifted to the next observation and comparisons use chronological splits rather than shuffled samples.

<h2 align="center" id="component-relationships">02 · A closer look at the components</h2>

<div align="center">

*Now zoom in from the rooms to the connections: which dashboard view calls which reusable module?*

</div>

<!-- mermaid-checked: no \n, no em-dash/en-dash, no {} in labels, subgraphs are id["label"], arrows are -->|"label"|, all subgraphs closed by end, ids unique -->
```mermaid
flowchart LR
    subgraph cPresentation["Presentation"]
        cApp["app.py"]
        cSidebar["ui.sidebar"]
        cOverview["ui.overview"]
        cRaw["ui.raw_data"]
        cTime["ui.time_series_ml"]
        cNeural["ui.neural_networks"]
        cPerformance["ui.performance"]
        cForecast["ui.ai_forecast"]
    end
    subgraph cAnalysis["Analysis and models"]
        cFetch["data.fetch"]
        cPreprocess["data.preprocess"]
        cFeatures["data.features"]
        cTrainEval["models.train_eval"]
        cBaselines["models.baselines"]
        cML["models.ml_models"]
        cDL["models.dl_models"]
        cMetrics["utils.metrics"]
    end
    subgraph cSupport["Supporting modules"]
        cNews["nlp.news_fetch"]
        cSentiment["nlp.sentiment"]
        cConfig["config.py"]
    end

    cApp -->|"renders shared controls"| cSidebar
    cApp -->|"renders dashboard views"| cOverview
    cApp -->|"renders dashboard views"| cRaw
    cApp -->|"renders dashboard views"| cTime
    cApp -->|"renders dashboard views"| cNeural
    cApp -->|"renders dashboard views"| cPerformance
    cApp -->|"renders dashboard views"| cForecast
    cSidebar -->|"passes ticker and date filters"| cOverview
    cSidebar -->|"passes ticker and date filters"| cRaw
    cSidebar -->|"passes ticker model and dates"| cTime
    cSidebar -->|"passes ticker and dates"| cNeural
    cSidebar -->|"passes ticker and dates"| cPerformance
    cSidebar -->|"passes forecast selections"| cForecast
    cOverview -->|"fetches latest quote"| cFetch
    cRaw -->|"loads history and cleans it"| cFetch
    cRaw -->|"checks quality and missing values"| cPreprocess
    cTime -->|"loads historical observations"| cFetch
    cNeural -->|"loads historical observations"| cFetch
    cPerformance -->|"loads historical observations"| cFetch
    cForecast -->|"loads quote and history"| cFetch
    cTime -->|"prepares supervised features"| cPreprocess
    cNeural -->|"prepares supervised features"| cPreprocess
    cPerformance -->|"prepares supervised features"| cPreprocess
    cForecast -->|"prepares supervised features"| cPreprocess
    cPreprocess -->|"clean OHLCV frames"| cFeatures
    cFeatures -->|"feature matrix and shifted target"| cTrainEval
    cTime -->|"runs model comparisons"| cTrainEval
    cPerformance -->|"runs comparison and validation"| cTrainEval
    cNeural -->|"builds and trains sequences"| cDL
    cNeural -->|"evaluates predictions"| cMetrics
    cForecast -->|"creates selected ML model"| cML
    cForecast -->|"creates selected baseline"| cBaselines
    cForecast -->|"scores holdout predictions"| cMetrics
    cTrainEval -->|"runs reference models"| cBaselines
    cTrainEval -->|"fits tabular estimators"| cML
    cTrainEval -->|"trains sequence estimators"| cDL
    cTrainEval -->|"computes evaluation scores"| cMetrics
    cNews -->|"adds headline sentiment"| cSentiment
    cApp -.->|"loads project settings"| cConfig
    cFetch -.->|"uses ticker and history settings"| cConfig
    cML -.->|"uses random seed"| cConfig
    cDL -.->|"uses random seed"| cConfig
```

<h3 align="center" id="component-inventory">The component field guide</h3>

| Component | Layer | Type | Responsibility |
|---|---|---|---|
| `app.py` | Presentation | Streamlit entry point | Configures the page and renders the shared sidebar and six dashboard views |
| `ui.sidebar` | Presentation | Shared UI module | Collects region, ticker, date range, model, horizon, and refresh selections |
| `ui.overview` | Presentation | Dashboard view | Displays the latest quote, daily movement, and observation freshness |
| `ui.raw_data` | Presentation | Dashboard view | Displays OHLCV history, quality information, cleaning controls, and CSV download |
| `ui.time_series_ml` | Presentation | Dashboard view | Charts history and starts classical ML and baseline comparisons |
| `ui.neural_networks` | Presentation | Dashboard view | Prepares sequences, trains LSTM or GRU, and displays evaluation charts |
| `ui.performance` | Presentation | Dashboard view | Compares model metrics and optionally runs walk-forward validation |
| `ui.ai_forecast` | Presentation | Dashboard view | Generates a next-period estimate and shows holdout performance |
| `data.fetch` | Analysis | Data integration module | Retrieves quotes, daily OHLCV data, and records ticker fetch failures |
| `data.preprocess` | Analysis | Data preparation module | Sorts and cleans price observations, handles missing values, and reports quality |
| `data.features` | Analysis | Feature engineering module | Builds technical and lagged features and shifted supervised targets |
| `models.train_eval` | Analysis | Training and evaluation module | Performs chronological splits, model comparisons, metrics, and walk-forward validation |
| `models.baselines` | Analysis | Estimators | Provides last-close, last-return, and moving-average reference models |
| `models.ml_models` | Analysis | Model factory | Builds scikit-learn, XGBoost, and LightGBM regressors |
| `models.dl_models` | Analysis | Deep-learning module | Builds and trains LSTM and GRU sequence models |
| `utils.metrics` | Analysis | Metric helpers | Calculates MAE, RMSE, and directional accuracy |
| `nlp.news_fetch` | Supporting | News helper | Retrieves and deduplicates ticker headlines; not wired into the current dashboard |
| `nlp.sentiment` | Supporting | Sentiment helper | Scores headlines with VADER or optional FinBERT; not wired into the current dashboard |
| `config.py` | Supporting | Configuration module | Defines project metadata, ticker universe, refresh choices, feature windows, and model options |

<div align="center">

> “Good architecture makes the route from observation to result easier to follow.”

<sub>For how the experiments work, read the [methodology](METHODOLOGY.md). For what those results cannot promise, see [limitations](LIMITATIONS.md).</sub>

<br>

[Back to the project README](README.md)

</div>