Polymarket Weather XGBoost Template
A reusable template for building, running, and backtesting XGBoost models on Polymarket daily weather markets.
The project is designed around a city configuration .yaml file, so the same scripts can be reused for different markets by changing station IDs, coordinates, timezone, forecast hours, model file, MAE, and output suffix.
> **Status:** research/backtesting template. This repository does not execute live trades automatically.
---
What this project does
This template supports the full workflow for a Polymarket weather model:
Collect historical weather observations and forecast-derived features.
Train an XGBoost model for daily maximum temperature prediction.
Generate daily model predictions for a city.
Download historical Polymarket orderbook snapshots.
Resolve historical outcomes using oracle/weather-station data.
Backtest a probability-based betting strategy with EV filters and Kelly-style sizing.
The current scripts are focused on daily highest temperature in `<city>` markets, but the structure can be adapted to other weather markets.
---
Repository structure
```text
.
├── README.md
├── milan.yaml
├── XGBOOST_template.ipynb
├── Model_run_template.py
├── run_model_historical_template.py
├── backtesting_data_extraction_template.py
├── backtest_template.py
├── Wundergrounds_scraper.py
├── mass_scraper.py
└── mass_scraper_claude_version.py
```
Main files
File	Purpose
`milan.yaml`	Example city configuration file. Change this or create a new YAML file for each city.
`XGBOOST_template.ipynb`	Notebook template for feature engineering, model training, validation, and exporting the XGBoost model.
`Model_run_template.py`	Generates today’s feature row and, if a model file is available, produces today’s prediction and probability board.
`run_model_historical_template.py`	Runs the trained XGBoost model historically for all dates found in an orderbook CSV.
`backtesting_data_extraction_template.py`	Downloads historical Polymarket orderbook snapshots through the Predexon API.
`backtest_template.py`	Replays the strategy on historical predictions and orderbook snapshots.
`Wundergrounds_scraper.py`	Scrapes station-level historical daily high temperature data from the https://www.wunderground.com/.
`mass_scraper.py`	Scrapes a date range of station-level oracle data.
`mass_scraper_claude_version.py`	Improved/resumable oracle scraper that can scrape exactly the dates present in an orderbook CSV.
---
Core idea
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
---
Requirements
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
---
API keys
Historical Polymarket orderbook extraction uses the Predexon API.
---
City configuration
Each city should have its own YAML config file. Example:
```yaml
city:
  name: Milan
  output_suffix: milan
  polymarket_slug: milan

location:
  timezone: Europe/Rome
  latitude: 45.63060
  longitude: 8.72811

iem:
  station: LIMC

forecast_hours:
  gfs_surface: [12, 15, 18]
  gfs_850hpa: [12, 15]
  ecmwf_surface: [12, 15, 18]

model:
  file: milan_MAE_081_oracle.json
  mae: 0.81

target_window:
  start: "12:00"
  end: "20:00"

temp_range: [0, 46]
```
Important notes:
`timezone` should match the city/market local timezone.
`latitude` and `longitude` should correspond to the weather station or target city location.
The code automatically converts negative western longitudes to the 0–360 convention for GFS.
ECMWF uses signed longitude, so western longitudes stay negative.
`iem.station` should be the station code used by Iowa Environmental Mesonet.
`model.mae` is used to convert the point forecast into a probability distribution.
`target_window` should match the time window used during model training and market settlement.
---
Workflow
1. Train an XGBoost model
Use the notebook:
```text
XGBOOST_template.ipynb
```
The notebook is intended for:
collecting historical station observations,
mining GFS / ECMWF forecast features,
engineering weather features,
training an XGBoost regression model,
evaluating MAE,
exporting the model as a JSON file.
Example model output:
```text
milan_MAE_081_oracle.json
```
---
2. Generate today’s prediction
```bash
python Model_run_template.py --config milan.yaml
```
Outputs:
```text
today_features_milan.csv
today_probabilities_milan.csv
```
---
3. Download historical Polymarket orderbooks
First run a one-day probe to check that the market slug and API access work:
```bash
python backtesting_data_extraction_template.py --config milan.yaml --probe-only
```
Optionally probe a specific date:
```bash
python backtesting_data_extraction_template.py --config milan.yaml --probe-only --date 2026-04-19
```
Then download historical orderbook snapshots:
```bash
python backtesting_data_extraction_template.py --config milan.yaml --lookback 60
```
Default output:
```text
milan_temperature_orderbooks.csv
```
The script samples orderbook snapshots every 10 minutes between 10:00 and 17:00 local time by default.
---
4. Generate historical model predictions
Once the orderbook CSV exists, generate model predictions for the same dates:
```bash
python run_model_historical_template.py --config milan.yaml --resume
```
Outputs:
```text
historical_predictions_milan.csv
historical_predictions_failures_milan.csv
```
The `--resume` flag skips dates that are already present in the output CSV.
---
5. Collect oracle / settlement temperature data
The backtester can use station-level historical daily high temperatures as the authoritative resolution source.
Recommended script:
```bash
python mass_scraper_claude_version.py
```
Before running it, edit the station code and orderbook CSV path inside the script:
```python
target_station = "LIMC"
orderbook_csv = "milan_temperature_orderbooks.csv"
```
Expected output:
```text
LIMC_FINAL_ORACLE_DATA.csv
```
Use responsible scraping practices, respect website terms of service, and apply conservative request rates.
---
6. Run the backtest
```bash
python backtest_template.py --config milan.yaml --wunderground LIMC_FINAL_ORACLE_DATA.csv
```
Default inputs:
```text
milan_temperature_orderbooks.csv
historical_predictions_milan.csv
LIMC_FINAL_ORACLE_DATA.csv
```
Outputs:
```text
backtest_trades_milan.csv
backtest_daily_milan.csv
backtest_summary_milan.txt
```
The summary includes:
number of trading days,
total trades,
win rate,
ending bankroll,
ROI on bankroll,
ROI on capital used,
average EV,
average Kelly fraction,
daily Sharpe-style metric.
---
Backtesting logic
The backtester:
Loads historical model predictions.
Converts each predicted high into a bracket probability distribution.
Reads historical Polymarket orderbook snapshots.
Scans the configured limit-order window.
Computes expected value for YES and NO sides.
Places simulated trades only if EV is within the configured range.
Sizes positions using fractional Kelly.
Resolves trades using oracle data, orderbook inference, or observed max temperature fallback.
Key strategy parameters are currently defined near the top of `backtest_template.py`:
```python
LIMIT_WINDOW_START = "10:00"
LIMIT_WINDOW_END   = "13:00"
MIN_MODEL_PROB     = 0.5
MAX_MODEL_PROB     = 0.9
MIN_EV             = 0.1
MAX_EV             = 0.2
BANKROLL_START     = 1000.0
KELLY_FRACTION     = 0.1
UNCERTAINTY_MULTIPLIER = 1.12
ALPHA = 0.9
```
You should treat these as experimental parameters, not universal settings. Modify accordingly
---
Output files
Output file	Created by	Description
`today_features_<city>.csv`	`Model_run_template.py`	Single feature row for today.
`today_probabilities_<city>.csv`	`Model_run_template.py`	Probability board for today.
`<city>_temperature_orderbooks.csv`	`backtesting_data_extraction_template.py`	Historical orderbook snapshots.
`historical_predictions_<city>.csv`	`run_model_historical_template.py`	Historical model predictions for orderbook dates.
`<station>_FINAL_ORACLE_DATA.csv`	`mass_scraper_claude_version.py`	Station-level historical high temperatures.
`backtest_trades_<city>.csv`	`backtest_template.py`	One row per simulated trade.
`backtest_daily_<city>.csv`	`backtest_template.py`	Daily aggregated PnL.
`backtest_summary_<city>.txt`	`backtest_template.py`	Human-readable backtest summary.
---
Suggested structure:
```text
.
├── README.md
├── configs/
│   └── milan.yaml
├── notebooks/
│   └── XGBOOST_template.ipynb
├── scripts/
│   ├── Model_run_template.py
│   ├── run_model_historical_template.py
│   ├── backtesting_data_extraction_template.py
│   ├── backtest_template.py
│   ├── Wundergrounds_scraper.py
│   └── oracle_scraper_resumable.py
└── requirements.txt
```
---
Limitations
This project is experimental. Important limitations include:
Backtest results may be unstable with small sample sizes.
Weather forecast APIs and data availability can change.
Market liquidity, spreads, partial fills, and latency are simplified in the backtest.
A low MAE does not automatically imply a profitable trading strategy.
Positive historical ROI does not guarantee future performance.
---
Disclaimer
This repository is for educational and research purposes only. It is not financial advice, trading advice, or a recommendation to trade prediction markets. Prediction markets involve risk, and you can lose money. Always validate your data sources, assumptions, model calibration, and execution logic before using any strategy with real capital.

AI STATEMENT
This project was developed with assistance from AI tools, including Claude Opus 4.7 and ChatGPT 5.5, for feature engineering, code generation, debugging, and documentation support. All outputs were reviewed, tested, and validated by the author.
