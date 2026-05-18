[README.md](https://github.com/user-attachments/files/27977249/README.md)
# Polymarket Weather XGBoost Template

A reusable template for building, running, and backtesting XGBoost models on Polymarket daily weather markets.

The project is designed around a city configuration .yaml file, so the same scripts can be reused for different markets by changing station IDs, coordinates, timezone, forecast hours, model file, MAE, and output suffix.

> \*\*Status:\*\* research/backtesting template. This repository does not execute live trades automatically.

\---

## What this project does

This template supports the full workflow for a Polymarket weather model:

1. Collect historical weather observations and forecast-derived features.
2. Train an XGBoost model for daily maximum temperature prediction.
3. Generate daily model predictions for a city.
4. Download historical Polymarket orderbook snapshots.
5. Resolve historical outcomes using oracle/weather-station data.
6. Backtest a probability-based betting strategy with EV filters and Kelly-style sizing.

The current scripts are focused on daily **highest temperature in `<city>`** markets, but the structure can be adapted to other weather markets.

\---

## Repository structure

```text
.
├── README.md
├── milan.yaml
├── XGBOOST\_template.ipynb
├── Model\_run\_template.py
├── run\_model\_historical\_template.py
├── backtesting\_data\_extraction\_template.py
├── backtest\_template.py
├── Wundergrounds\_scraper.py
├── mass\_scraper.py
└── mass\_scraper\_claude\_version.py
```

### Main files

|File|Purpose|
|-|-|
|`milan.yaml`|Example city configuration file. Change this or create a new YAML file for each city.|
|`XGBOOST\_template.ipynb`|Notebook template for feature engineering, model training, validation, and exporting the XGBoost model.|
|`Model\_run\_template.py`|Generates today’s feature row and, if a model file is available, produces today’s prediction and probability board.|
|`run\_model\_historical\_template.py`|Runs the trained XGBoost model historically for all dates found in an orderbook CSV.|
|`backtesting\_data\_extraction\_template.py`|Downloads historical Polymarket orderbook snapshots through the Predexon API.|
|`backtest\_template.py`|Replays the strategy on historical predictions and orderbook snapshots.|
|`Wundergrounds\_scraper.py`|Scrapes station-level historical daily high temperature data from the https://www.wunderground.com/.|
|`mass\_scraper.py`|Scrapes a date range of station-level oracle data.|
|`mass\_scraper\_claude\_version.py`|Improved/resumable oracle scraper that can scrape exactly the dates present in an orderbook CSV.|

\---

## Core idea

The model predicts the daily maximum temperature for a target city. The prediction is converted into a probability distribution over integer °C temperature brackets. The backtester then compares the model probability with market prices and simulates trades only when the expected value is inside a configured range.

At a high level:

```text
weather observations + forecast features
              ↓
        XGBoost model
              ↓
 predicted daily high temperature
              ↓
 probability board over Polymarket brackets
              ↓
 orderbook comparison / EV filter / Kelly sizing
              ↓
 historical backtest results
```

\---

## Requirements

Recommended Python version:

```text
Python 3.10+
```

Main Python packages:

```bash
pip install pandas numpy scipy scikit-learn xgboost requests pyyaml tqdm pytz matplotlib
pip install herbie-data xarray cfgrib
pip install selenium beautifulsoup4 webdriver-manager
```

\---

## API keys

Historical Polymarket orderbook extraction uses the Predexon API.

\---

## City configuration

Each city should have its own YAML config file. Example:

```yaml
city:
  name: Milan
  output\_suffix: milan
  polymarket\_slug: milan

location:
  timezone: Europe/Rome
  latitude: 45.63060
  longitude: 8.72811

iem:
  station: LIMC

forecast\_hours:
  gfs\_surface: \[12, 15, 18]
  gfs\_850hpa: \[12, 15]
  ecmwf\_surface: \[12, 15, 18]

model:
  file: milan\_MAE\_081\_oracle.json
  mae: 0.81

target\_window:
  start: "12:00"
  end: "20:00"

temp\_range: \[0, 46]
```

Important notes:

* `timezone` should match the city/market local timezone.
* `latitude` and `longitude` should correspond to the weather station or target city location.
* The code automatically converts negative western longitudes to the 0–360 convention for GFS.
* ECMWF uses signed longitude, so western longitudes stay negative.
* `iem.station` should be the station code used by Iowa Environmental Mesonet.
* `model.mae` is used to convert the point forecast into a probability distribution.
* `target\_window` should match the time window used during model training and market settlement.

\---

## Workflow

### 1\. Train an XGBoost model

Use the notebook:

```text
XGBOOST\_template.ipynb
```

The notebook is intended for:

* collecting historical station observations,
* mining GFS / ECMWF forecast features,
* engineering weather features,
* training an XGBoost regression model,
* evaluating MAE,
* exporting the model as a JSON file.

Example model output:

```text
milan\_MAE\_081\_oracle.json
```

\---

### 2\. Generate today’s prediction

```bash
python Model\_run\_template.py --config milan.yaml
```

Outputs:

```text
today\_features\_milan.csv
today\_probabilities\_milan.csv
```

\---

### 3\. Download historical Polymarket orderbooks

First run a one-day probe to check that the market slug and API access work:

```bash
python backtesting\_data\_extraction\_template.py --config milan.yaml --probe-only
```

Optionally probe a specific date:

```bash
python backtesting\_data\_extraction\_template.py --config milan.yaml --probe-only --date 2026-04-19
```

Then download historical orderbook snapshots:

```bash
python backtesting\_data\_extraction\_template.py --config milan.yaml --lookback 60
```

Default output:

```text
milan\_temperature\_orderbooks.csv
```

The script samples orderbook snapshots every 10 minutes between 10:00 and 17:00 local time by default.

\---

### 4\. Generate historical model predictions

Once the orderbook CSV exists, generate model predictions for the same dates:

```bash
python run\_model\_historical\_template.py --config milan.yaml --resume
```

Outputs:

```text
historical\_predictions\_milan.csv
historical\_predictions\_failures\_milan.csv
```

The `--resume` flag skips dates that are already present in the output CSV.

\---

### 5\. Collect oracle / settlement temperature data

The backtester can use station-level historical daily high temperatures as the authoritative resolution source.

Recommended script:

```bash
python mass\_scraper\_claude\_version.py
```

Before running it, edit the station code and orderbook CSV path inside the script:

```python
target\_station = "LIMC"
orderbook\_csv = "milan\_temperature\_orderbooks.csv"
```

Expected output:

```text
LIMC\_FINAL\_ORACLE\_DATA.csv
```

Use responsible scraping practices, respect website terms of service, and apply conservative request rates.

\---

### 6\. Run the backtest

```bash
python backtest\_template.py --config milan.yaml --wunderground LIMC\_FINAL\_ORACLE\_DATA.csv
```

Default inputs:

```text
milan\_temperature\_orderbooks.csv
historical\_predictions\_milan.csv
LIMC\_FINAL\_ORACLE\_DATA.csv
```

Outputs:

```text
backtest\_trades\_milan.csv
backtest\_daily\_milan.csv
backtest\_summary\_milan.txt
```

The summary includes:

* number of trading days,
* total trades,
* win rate,
* ending bankroll,
* ROI on bankroll,
* ROI on capital used,
* average EV,
* average Kelly fraction,
* daily Sharpe-style metric.

\---

## Backtesting logic

The backtester:

1. Loads historical model predictions.
2. Converts each predicted high into a bracket probability distribution.
3. Reads historical Polymarket orderbook snapshots.
4. Scans the configured limit-order window.
5. Computes expected value for YES and NO sides.
6. Places simulated trades only if EV is within the configured range.
7. Sizes positions using fractional Kelly.
8. Resolves trades using oracle data, orderbook inference, or observed max temperature fallback.

Key strategy parameters are currently defined near the top of `backtest\_template.py`:

```python
LIMIT\_WINDOW\_START = "10:00"
LIMIT\_WINDOW\_END   = "13:00"
MIN\_MODEL\_PROB     = 0.5
MAX\_MODEL\_PROB     = 0.9
MIN\_EV             = 0.1
MAX\_EV             = 0.2
BANKROLL\_START     = 1000.0
KELLY\_FRACTION     = 0.1
UNCERTAINTY\_MULTIPLIER = 1.12
ALPHA = 0.9
```

You should treat these as experimental parameters, not universal settings. Modify accordingly

\---

## Output files

|Output file|Created by|Description|
|-|-|-|
|`today\_features\_<city>.csv`|`Model\_run\_template.py`|Single feature row for today.|
|`today\_probabilities\_<city>.csv`|`Model\_run\_template.py`|Probability board for today.|
|`<city>\_temperature\_orderbooks.csv`|`backtesting\_data\_extraction\_template.py`|Historical orderbook snapshots.|
|`historical\_predictions\_<city>.csv`|`run\_model\_historical\_template.py`|Historical model predictions for orderbook dates.|
|`<station>\_FINAL\_ORACLE\_DATA.csv`|`mass\_scraper\_claude\_version.py`|Station-level historical high temperatures.|
|`backtest\_trades\_<city>.csv`|`backtest\_template.py`|One row per simulated trade.|
|`backtest\_daily\_<city>.csv`|`backtest\_template.py`|Daily aggregated PnL.|
|`backtest\_summary\_<city>.txt`|`backtest\_template.py`|Human-readable backtest summary.|





\---

## Limitations

This project is experimental. Important limitations include:

* Backtest results may be unstable with small sample sizes.
* Weather forecast APIs and data availability can change.
* Market liquidity, spreads, partial fills, and latency are simplified in the backtest.
* A low MAE does not automatically imply a profitable trading strategy.
* Positive historical ROI does not guarantee future performance.

\---

## Disclaimer

This repository is for educational and research purposes only. It is not financial advice, trading advice, or a recommendation to trade prediction markets. Prediction markets involve risk, and you can lose money. Always validate your data sources, assumptions, model calibration, and execution logic before using any strategy with real capital.



## AI STATEMENT

This project was developed with assistance from AI tools, including Claude Opus 4.7 and ChatGPT 5.5, for feature engineering, code generation, debugging, and documentation support. All outputs were reviewed, tested, and validated by the author.



