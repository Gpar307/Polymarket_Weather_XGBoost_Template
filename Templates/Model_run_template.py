# -*- coding: utf-8 -*-
"""
Model_run_template.py
─────────────────────────────
Generates the SINGLE ROW of features needed to run the City XGBoost
weather model for TODAY's high-temperature forecast.

All city-specific values (coordinates, timezone, IEM station, forecast
hours, model file, MAE) are loaded from a YAML config file passed via
--config. See madrid.yaml for an example.

Run:
    python Model_run_template.py --config madrid.yaml
    python Model_run_template.py -c configs/wageningen.yaml

Output:
    today_features_<suffix>.csv      ← feature row for the XGBoost model
    today_probabilities_<suffix>.csv ← probability board (if model present)
"""

# ─────────────────────────────────────────────────────────────────────────────
# 0.  IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import argparse
import time
import warnings
import yaml
import numpy as np
import pandas as pd
import pytz
import requests
from datetime import datetime, timedelta
from scipy import stats

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# 0b.  LOAD YAML CONFIG
# ─────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="Generate daily features for the XGBoost weather model."
)
parser.add_argument(
    "--config", "-c", required=True,
    help="Path to YAML config file (e.g. madrid.yaml).",
)
args = parser.parse_args()

with open(args.config, "r") as _f:
    CONFIG = yaml.safe_load(_f)

CITY_NAME       = CONFIG["city"]["name"]
OUTPUT_SUFFIX   = CONFIG["city"]["output_suffix"]
CITY_TZ_NAME    = CONFIG["location"]["timezone"]
LAT             = float(CONFIG["location"]["latitude"])
LON_STANDARD    = float(CONFIG["location"]["longitude"])
# GFS uses 0–360 longitude convention. Only shift if originally negative (west).
LON_GFS         = LON_STANDARD + 360 if LON_STANDARD < 0 else LON_STANDARD
STATION         = CONFIG["iem"]["station"]
FXX_GFS_SURFACE = CONFIG["forecast_hours"]["gfs_surface"]
FXX_GFS_850     = CONFIG["forecast_hours"]["gfs_850hpa"]
FXX_ECMWF       = CONFIG["forecast_hours"]["ecmwf_surface"]
MODEL_FILE      = CONFIG.get("model", {}).get("file")
MAE             = float(CONFIG.get("model", {}).get("mae", 1.5))

# ─────────────────────────────────────────────────────────────────────────────
# 1.  DATE ANCHORS
# ─────────────────────────────────────────────────────────────────────────────
utc_now   = datetime.now(pytz.utc)
city_tz   = pytz.timezone(CITY_TZ_NAME)
city_now  = utc_now.astimezone(city_tz)

today          = city_now.date()
yesterday      = today - timedelta(days=1)
two_days_ago   = today - timedelta(days=2)
three_days_ago = today - timedelta(days=3)

print("=" * 60)
print(f"  {CITY_NAME.upper()} XGBOOST — DAILY FEATURE GENERATOR")
print("=" * 60)
print(f"  Today (forecast target) : {today}")
print(f"  Yesterday               : {yesterday}")
print(f"  2 days ago              : {two_days_ago}")
print(f"  3 days ago              : {three_days_ago}")
print()

# ─────────────────────────────────────────────────────────────────────────────
# 3.  HELPER: FETCH IEM ASOS DATA FOR A DATE RANGE
# ─────────────────────────────────────────────────────────────────────────────
def fetch_iem(start_date, end_date, max_retries=5, base_wait=10):
    """
    Download ASOS data from IEM for the configured station between
    start_date and end_date (inclusive). Retries with exponential backoff.
    """
    iem_end = end_date + timedelta(days=1)
    url = (
        f"https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?"
        f"station={STATION}&data=tmpf&data=dwpf&data=drct&data=sknt&"
        f"data=skyc1&data=alti&"
        f"year1={start_date.year}&month1={start_date.month}&day1={start_date.day}&"
        f"year2={iem_end.year}&month2={iem_end.month}&day2={iem_end.day}&"
        f"tz=Etc/UTC&format=onlycomma&latlon=no&missing=empty"
    )
    print(f"[IEM] Downloading {STATION} observations {start_date} → {end_date} …")

    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            from io import StringIO
            df = pd.read_csv(StringIO(resp.text), parse_dates=["valid"])
            break
        except Exception as e:
            wait = base_wait * (2 ** (attempt - 1))
            if attempt < max_retries:
                print(f"  [IEM] Attempt {attempt} failed ({e}). Retrying in {wait}s …")
                time.sleep(wait)
            else:
                print('error in the IEM URL')

    df.rename(columns={
        "valid": "Timestamp",
        "tmpf": "Temp_F",
        "dwpf": "Dew_Point_F",
        "drct": "Wind_Dir_Deg",
        "sknt": "Wind_Speed_Kts",
        "skyc1": "Cloud_Cover_Code",
    }, inplace=True)

    cloud_map = {"CLR": 0, "SKC": 0, "FEW": 2, "SCT": 4, "BKN": 6, "OVC": 8, "VV ": 8}
    df["Cloud_Cover_Okta"] = df["Cloud_Cover_Code"].map(cloud_map).fillna(0)
    df.drop(columns=["Cloud_Cover_Code"], inplace=True)

    df["alti"] = pd.to_numeric(df["alti"], errors="coerce").ffill()

    df = df.dropna(subset=["Temp_F", "Wind_Dir_Deg", "Dew_Point_F"])

    df["Temp_C"] = (df["Temp_F"] - 32) * (5.0 / 9.0)
    df["Dew_Point_C"] = (df["Dew_Point_F"] - 32) * (5.0 / 9.0)
    df.drop(columns=["Temp_F", "Dew_Point_F", "station"], inplace=True, errors="ignore")

    # IEM returns UTC; convert to local city time.
    df["Timestamp"] = (
        df["Timestamp"]
        .dt.tz_localize("UTC")
        .dt.tz_convert(CITY_TZ_NAME)
    )
    df.set_index("Timestamp", inplace=True)
    df.sort_index(inplace=True)

    print(f"[IEM] Fetched {len(df):,} rows.")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 4.  FETCH IEM: 3 DAYS AGO → TODAY
# ─────────────────────────────────────────────────────────────────────────────
df_raw = fetch_iem(three_days_ago, today)


# ─────────────────────────────────────────────────────────────────────────────
# 5.  EXTRACT SNAPSHOTS FROM IEM DATA
# ─────────────────────────────────────────────────────────────────────────────
def snapshot_at(df, date, start_time, end_time, agg="last"):
    """Return a single-row summary for `date` between start_time and end_time."""
    if df.index.tz is None:
        working_df = df.tz_localize("UTC").tz_convert(city_tz)
    else:
        working_df = df.tz_convert(city_tz)

    date_str = str(date)
    day_df = working_df[working_df.index.strftime('%Y-%m-%d') == date_str]
    window = day_df.between_time(start_time, end_time)

    if window.empty:
        return None

    return window.iloc[-1] if agg == "last" else window

# ── 5a. TODAY: 10 AM snapshot
row_10am = snapshot_at(df_raw, today, "09:30", "10:30")
if row_10am is None:
    raise RuntimeError(
        f"No 10 AM observation found for today yet. "
        f"Try running the script after 10:30 AM {CITY_TZ_NAME} local time."
    )

# ── 5b. TODAY: 7 AM snapshot
row_7am = snapshot_at(df_raw, today, "06:30", "07:30")
if row_7am is None:
    raise RuntimeError("No 7 AM observation found for today. Check IEM availability.")

# ── 5c. TODAY: Overnight low (00:00–06:00)
overnight_df = df_raw[df_raw.index.date == today].between_time("00:00", "06:00")
if overnight_df.empty:
    raise RuntimeError("No overnight observations found for today.")
overnight_low_today = overnight_df["Temp_C"].min()

print(f"\n[SNAPSHOT] 10 AM Temp  : {row_10am['Temp_C']:.2f} °C")
print(f"[SNAPSHOT]  7 AM Temp  : {row_7am['Temp_C']:.2f} °C")
print(f"[SNAPSHOT] Overnight Low: {overnight_low_today:.2f} °C")

# ─────────────────────────────────────────────────────────────────────────────
# 6.  ACTUAL AFTERNOON MAX-TEMPS FOR PAST 3 DAYS
# ─────────────────────────────────────────────────────────────────────────────
def actual_max(df, date):
    """Return the observed max temperature 12:00–20:00 for a given date."""
    day_df = df[df.index.date == date]
    afternoon = day_df.between_time("12:00", "20:00")
    if afternoon.empty:
        return np.nan
    return afternoon["Temp_C"].max()

actual_max_yesterday = actual_max(df_raw, yesterday)
actual_max_2days_ago = actual_max(df_raw, two_days_ago)
actual_max_3days_ago = actual_max(df_raw, three_days_ago)

print(f"\n[BIAS] Actual max yesterday   ({yesterday})  : {actual_max_yesterday:.2f} °C")
print(f"[BIAS] Actual max 2 days ago  ({two_days_ago}) : {actual_max_2days_ago:.2f} °C")
print(f"[BIAS] Actual max 3 days ago  ({three_days_ago}) : {actual_max_3days_ago:.2f} °C")


# ─────────────────────────────────────────────────────────────────────────────
# 7.  HERBIE: GFS SURFACE (2 m) TEMPERATURE  — 00 Z RUN FOR TODAY
# ─────────────────────────────────────────────────────────────────────────────
try:
    from herbie import Herbie
    import xarray as xr
except ImportError:
    raise ImportError(
        "herbie-data is not installed. Run: pip install herbie-data xarray cfgrib"
    )

def fetch_herbie_surface(date, model, product, fxx_list, lat, lon, var_name, search_str, priority):
    """Return the max 2 m temperature (°C) across the given forecast hours."""
    date_str = date.strftime("%Y-%m-%d")
    temps = []
    for fxx in fxx_list:
        try:
            H   = Herbie(date_str, model=model, product=product, fxx=fxx, priority=priority)
            ds  = H.xarray(search_str)
            val = ds[var_name].sel(latitude=lat, longitude=lon, method="nearest").values
            temps.append(float(val) - 273.15)
        except Exception as e:
            print(f"  [WARN] {model} fxx={fxx} failed: {e}")
        time.sleep(0.3)
    return round(float(np.max(temps)), 2) if temps else np.nan


print("\n[GFS] Fetching surface 2 m temperature for today …")
gfs_predicted_high = fetch_herbie_surface(
    date     = today,
    model    = "gfs",
    product  = "pgrb2.0p25",
    fxx_list = FXX_GFS_SURFACE,
    lat      = LAT,
    lon      = LON_GFS,
    var_name = "t2m",
    search_str = "TMP:2 m above ground",
    priority = ["aws", "nomads"],
)
print(f"[GFS] Predicted High (surface): {gfs_predicted_high} °C")


# ─────────────────────────────────────────────────────────────────────────────
# 8.  HERBIE: GFS 850 hPa TEMPERATURE — 00 Z RUN FOR TODAY
# ─────────────────────────────────────────────────────────────────────────────
print("\n[GFS 850] Fetching 850 hPa temperature for today …")
gfs_850_temp = fetch_herbie_surface(
    date     = today,
    model    = "gfs",
    product  = "pgrb2.0p25",
    fxx_list = FXX_GFS_850,
    lat      = LAT,
    lon      = LON_GFS,
    var_name = "t",
    search_str = "TMP:850 mb",
    priority = ["aws", "nomads"],
)
print(f"[GFS 850] Predicted 850 hPa High: {gfs_850_temp} °C")


# ─────────────────────────────────────────────────────────────────────────────
# 9.  HERBIE: ECMWF IFS SURFACE (2 m) TEMPERATURE — 00 Z RUN FOR TODAY
# ─────────────────────────────────────────────────────────────────────────────
print("\n[ECMWF] Fetching surface 2 m temperature for today …")
ecmwf_predicted_high = fetch_herbie_surface(
    date     = today,
    model    = "ifs",
    product  = "oper",
    fxx_list = FXX_ECMWF,
    lat      = LAT,
    lon      = LON_STANDARD,   # ECMWF: signed (negative-west) longitude
    var_name = "t2m",
    search_str = ":2t:",
    priority = ["azure", "aws"],
)
print(f"[ECMWF] Predicted High (surface): {ecmwf_predicted_high} °C")


# ─────────────────────────────────────────────────────────────────────────────
# 10. OPEN-METEO ARCHIVE API: SOLAR RADIATION + SOIL  — TODAY AT 10 AM LOCAL
# ─────────────────────────────────────────────────────────────────────────────
def fetch_openmeteo_10am(date):
    """Fetch DNI, DHI, Soil Temp, Soil Moisture at 10:00 AM local city time."""
    date_str = date.strftime("%Y-%m-%d")

    archive_url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude":  LAT,
        "longitude": LON_STANDARD,
        "start_date": date_str,
        "end_date":   date_str,
        "hourly": [
            "direct_normal_irradiance",
            "soil_temperature_0_to_7cm",
            "soil_moisture_0_to_7cm",
            "diffuse_radiation",
        ],
        "timezone": CITY_TZ_NAME,
    }

    try:
        resp = requests.get(archive_url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        hourly = data.get("hourly", {})
        times  = pd.to_datetime(hourly["time"])

        df_h = pd.DataFrame({
            "time":                  times,
            "DNI_Radiation_Wm2":     hourly["direct_normal_irradiance"],
            "Soil_Temp_7cm":         hourly["soil_temperature_0_to_7cm"],
            "Soil_Moisture_7cm":     hourly["soil_moisture_0_to_7cm"],
            "DHI_Radiation_Wm2":     hourly["diffuse_radiation"],
        })
        row_10 = df_h[df_h["time"].dt.hour == 10]
        if not row_10.empty:
            print(f"[Open-Meteo Archive] 10 AM solar/soil data found for {date}.")
            return row_10.iloc[0]
    except Exception as e:
        print(f"  [WARN] Archive API failed ({e}); trying forecast endpoint …")

    forecast_url = "https://api.open-meteo.com/v1/forecast"
    params_fc = {
        "latitude":  LAT,
        "longitude": LON_STANDARD,
        "start_date": date_str,
        "end_date":   date_str,
        "hourly": [
            "direct_normal_irradiance",
            "soil_temperature_0_to_7cm",
            "soil_moisture_0_to_7cm",
            "diffuse_radiation",
        ],
        "timezone": CITY_TZ_NAME,
    }
    resp_fc = requests.get(forecast_url, params=params_fc, timeout=30)
    resp_fc.raise_for_status()
    data_fc = resp_fc.json()
    hourly_fc = data_fc.get("hourly", {})
    times_fc  = pd.to_datetime(hourly_fc["time"])

    df_hf = pd.DataFrame({
        "time":               times_fc,
        "DNI_Radiation_Wm2": hourly_fc["direct_normal_irradiance"],
        "Soil_Temp_7cm":     hourly_fc["soil_temperature_0_to_7cm"],
        "Soil_Moisture_7cm": hourly_fc["soil_moisture_0_to_7cm"],
        "DHI_Radiation_Wm2": hourly_fc["diffuse_radiation"],
    })
    row_10f = df_hf[df_hf["time"].dt.hour == 10]
    if row_10f.empty:
        raise RuntimeError("Open-Meteo: no 10 AM row found even in forecast API.")
    print(f"[Open-Meteo Forecast] 10 AM solar/soil data found for {date}.")
    return row_10f.iloc[0]


print("\n[Open-Meteo] Fetching solar/soil data for today …")
om_today = fetch_openmeteo_10am(today)

dni_radiation   = float(om_today["DNI_Radiation_Wm2"])
dhi_radiation   = float(om_today["DHI_Radiation_Wm2"])
soil_temp_7cm   = float(om_today["Soil_Temp_7cm"])
soil_moist_7cm  = float(om_today["Soil_Moisture_7cm"])

print(f"[Open-Meteo] DNI={dni_radiation:.1f}  DHI={dhi_radiation:.1f}  "
      f"Soil_Temp={soil_temp_7cm:.2f}  Soil_Moist={soil_moist_7cm:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# 11. FETCH GFS PREDICTED HIGH FOR THE PAST 3 DAYS
# ─────────────────────────────────────────────────────────────────────────────
print("\n[GFS BIAS] Fetching GFS surface predictions for past 3 days …")

gfs_high_yesterday = fetch_herbie_surface(
    date=yesterday, model="gfs", product="pgrb2.0p25",
    fxx_list=FXX_GFS_SURFACE, lat=LAT, lon=LON_GFS,
    var_name="t2m", search_str="TMP:2 m above ground", priority=["aws", "nomads"],
)
gfs_high_2days_ago = fetch_herbie_surface(
    date=two_days_ago, model="gfs", product="pgrb2.0p25",
    fxx_list=FXX_GFS_SURFACE, lat=LAT, lon=LON_GFS,
    var_name="t2m", search_str="TMP:2 m above ground", priority=["aws", "nomads"],
)
gfs_high_3days_ago = fetch_herbie_surface(
    date=three_days_ago, model="gfs", product="pgrb2.0p25",
    fxx_list=FXX_GFS_SURFACE, lat=LAT, lon=LON_GFS,
    var_name="t2m", search_str="TMP:2 m above ground", priority=["aws", "nomads"],
)

print(f"[GFS BIAS] GFS yesterday  ({yesterday})  : {gfs_high_yesterday} °C")
print(f"[GFS BIAS] GFS 2 days ago ({two_days_ago}) : {gfs_high_2days_ago} °C")
print(f"[GFS BIAS] GFS 3 days ago ({three_days_ago}) : {gfs_high_3days_ago} °C")


# ─────────────────────────────────────────────────────────────────────────────
# 12. ASSEMBLE ALL FEATURES
# ─────────────────────────────────────────────────────────────────────────────
print("\n[ASSEMBLE] Building the feature row …")

# ── 12a. Base 10 AM observation features
temp_10am     = float(row_10am["Temp_C"])
dew_10am      = float(row_10am["Dew_Point_C"])
wind_dir_10am = float(row_10am["Wind_Dir_Deg"])
wind_spd_10am = float(row_10am["Wind_Speed_Kts"])
cloud_10am    = float(row_10am["Cloud_Cover_Okta"])
alti_10am     = float(row_10am["alti"])

temp_7am      = float(row_7am["Temp_C"])
dew_7am       = float(row_7am["Dew_Point_C"])
alti_7am      = float(row_7am["alti"])

# ── 12b. Derivative features
temp_change_3hr     = temp_10am - temp_7am
dew_change_3hr      = dew_10am  - dew_7am
pressure_change_3hr = alti_10am - alti_7am
heat_momentum_ratio = temp_change_3hr - dew_change_3hr

# ── 12c. Relative Humidity at 10 AM
rh_10am = 100 * (
    np.exp((17.625 * dew_10am)  / (243.04 + dew_10am)) /
    np.exp((17.625 * temp_10am) / (243.04 + temp_10am))
)

# ── 12d. Early heating effort
early_heating_effort = temp_10am - overnight_low_today

# ── 12e. Cyclical encodings
day_of_year = today.timetuple().tm_yday

wind_sin = np.sin(wind_dir_10am * (2.0 * np.pi / 360))
wind_cos = np.cos(wind_dir_10am * (2.0 * np.pi / 360))
day_sin  = np.sin(day_of_year   * (2.0 * np.pi / 366))
day_cos  = np.cos(day_of_year   * (2.0 * np.pi / 366))

# ── 12f. Model-derived composite features
gfs_heat_gap          = gfs_predicted_high - temp_10am
supercomputer_spread  = abs(gfs_predicted_high - ecmwf_predicted_high)
model_bias_direction  = gfs_predicted_high - ecmwf_predicted_high
consensus_predicted   = (gfs_predicted_high + ecmwf_predicted_high) / 2
forecasted_lapse_rate = gfs_predicted_high - gfs_850_temp
scattering_ratio      = dhi_radiation / (dni_radiation + 1)
pressure_10am         = alti_10am
ground_to_air_gradient = soil_temp_7cm - temp_10am

# ── 12g. Bias lag features (using last 3 days)
bias_d1 = actual_max_yesterday - gfs_high_yesterday
bias_d2 = actual_max_2days_ago - gfs_high_2days_ago
bias_d3 = actual_max_3days_ago - gfs_high_3days_ago

yesterday_bias = bias_d1
heat_trend     = actual_max_yesterday - actual_max_2days_ago
bias_3day_mean = np.nanmean([bias_d1, bias_d2, bias_d3])

row_10am_yest = snapshot_at(df_raw, yesterday, "09:30", "10:30")
temp_10am_yesterday = float(row_10am_yest["Temp_C"]) if row_10am_yest is not None else np.nan
temp_10am_vs_yesterday = temp_10am - temp_10am_yesterday

# ── 12h. Wind resistance composite
gap_wind_resistance = gfs_heat_gap / (wind_spd_10am + 1)

# ── 12i. Cooling advection
cooling_advection = wind_spd_10am * wind_cos

# ─────────────────────────────────────────────────────────────────────────────
# 13. BUILD THE FINAL SINGLE-ROW DATAFRAME
# ─────────────────────────────────────────────────────────────────────────────
features = {
    # ── IEM 10 AM snapshot
    "Temp_10AM":             temp_10am,
    "Dew_Point_10AM":        dew_10am,
    "Wind_Speed_10AM":       wind_spd_10am,
    "Cloud_Cover_10AM":      cloud_10am,
    "alti_10AM":             alti_10am,

    # ── GFS / ECMWF supercomputer columns
    "GFS_Predicted_High":    gfs_predicted_high,
    "ECMWF_Predicted_High":  ecmwf_predicted_high,
    "GFS_850hPa_Temp":       gfs_850_temp,

    # ── Open-Meteo radiation + soil
    "DNI_Radiation_Wm2":     dni_radiation,
    "Soil_Temp_7cm":         soil_temp_7cm,
    "Soil_Moisture_7cm":     soil_moist_7cm,
    "DHI_Radiation_Wm2":     dhi_radiation,

    # ── Derived / engineered features
    "Temp_Change_3hr":        temp_change_3hr,
    "Dew_Point_Change_3hr":   dew_change_3hr,
    "Pressure_Change_3hr":    pressure_change_3hr,
    "Heat_Momentum_Ratio":    heat_momentum_ratio,
    "RH_10AM":                rh_10am,
    "Early_Heating_Effort":   early_heating_effort,
    "Overnight_Low":          overnight_low_today,
    "Wind_Sin":               wind_sin,
    "Wind_Cos":               wind_cos,
    "Day_Sin":                day_sin,
    "Day_Cos":                day_cos,

    # ── Model composite features
    "GFS_Heat_Gap":            gfs_heat_gap,
    "Supercomputer_Spread":    supercomputer_spread,
    "Model_Bias_Direction":    model_bias_direction,
    "Consensus_Predicted_High": consensus_predicted,
    "Forecasted_Lapse_Rate":   forecasted_lapse_rate,
    "Scattering_Ratio":        scattering_ratio,
    "Pressure_10AM":           pressure_10am,
    "Ground_to_Air_Gradient":  ground_to_air_gradient,

    # ── Bias / lag features (last 3 days)
    "Yesterday_Bias":         yesterday_bias,
    "Heat_Trend":             heat_trend,
    "Bias_3Day_Mean":         bias_3day_mean,
    "Temp_10AM_vs_Yesterday": temp_10am_vs_yesterday,
    "Gap_Wind_Resistance":    gap_wind_resistance,
    "Cooling_Advection":      cooling_advection,
}

df_today = pd.DataFrame([features])
df_today.insert(0, "Date", str(today))

# ─────────────────────────────────────────────────────────────────────────────
# 14. SAVE & PRINT SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
output_file = f"today_features_{OUTPUT_SUFFIX}.csv"
df_today.to_csv(output_file, index=False)

print("\n" + "=" * 60)
print("  ✅  FEATURE GENERATION COMPLETE")
print("=" * 60)
print(f"  City           : {CITY_NAME}")
print(f"  Date           : {today}")
print(f"  Temp 10 AM     : {temp_10am:.2f} °C")
print(f"  GFS Surface    : {gfs_predicted_high} °C")
print(f"  ECMWF Surface  : {ecmwf_predicted_high} °C")
print(f"  GFS 850 hPa    : {gfs_850_temp} °C")
print(f"  DNI Radiation  : {dni_radiation:.1f} W/m²")
print(f"  Yesterday Bias : {yesterday_bias:.2f} °C")
print(f"  3-Day Mean Bias: {bias_3day_mean:.2f} °C")
print(f"\n  Saved to → {output_file}")
print("=" * 60)

# ─────────────────────────────────────────────────────────────────────────────
# 15. OPTIONAL: QUICK MODEL INFERENCE (if model file is present)
# ─────────────────────────────────────────────────────────────────────────────
import os


# ─────────────────────────────────────────────────────────────────────────────
# 16. PROBABILITY BOARD
# ─────────────────────────────────────────────────────────────────────────────
def calculate_probabilities(prediction, mae=MAE):
    """
    Converts a point prediction into a Polymarket-style probability board.
    Uses a Normal distribution centred on the prediction where:
        std_dev = MAE × 1.2533  (MAE-to-sigma conversion for a Normal)
    Returns a DataFrame of the 10 most likely 1°C brackets, sorted by probability.
    """
    std_dev = mae * 1.2533

    board = []
    center_temp = int(round(prediction))
    scan_range = range(center_temp - 5, center_temp + 6)

    for bracket_temp in scan_range:
        lower_bound = bracket_temp - 0.5
        upper_bound = bracket_temp + 0.5

        prob_lower = stats.norm.cdf(lower_bound, loc=prediction, scale=std_dev)
        prob_upper = stats.norm.cdf(upper_bound, loc=prediction, scale=std_dev)

        bracket_probability = prob_upper - prob_lower
        percent_chance = bracket_probability * 100
        fair_odds = 1.0 / bracket_probability if bracket_probability > 0.01 else 999.0

        board.append({
            'Bracket': f"{bracket_temp}°C",
            'Model_Prob': f"{percent_chance:.1f}%",
            'Fair_Price': f"${bracket_probability:.2f}",
            'Fair_Payout': f"{fair_odds:.2f}x",
            '_raw_prob': bracket_probability,
        })

    df_board = pd.DataFrame(board)
    df_board = df_board.sort_values(by='_raw_prob', ascending=False).drop(columns=['_raw_prob'])
    return df_board.head(10)


if MODEL_FILE and os.path.exists(MODEL_FILE):
    import xgboost as xgb

    X_COLS = [
        "Wind_Speed_10AM", "Cloud_Cover_10AM", "Temp_10AM", "Dew_Point_10AM",
        "Overnight_Low", "Temp_Change_3hr", "Dew_Point_Change_3hr",
        "Pressure_Change_3hr", "Heat_Momentum_Ratio", "RH_10AM",
        "Early_Heating_Effort", "Wind_Sin", "Wind_Cos", "Day_Sin", "Day_Cos",
        "DNI_Radiation_Wm2", "Soil_Temp_7cm", "Soil_Moisture_7cm",
        "DHI_Radiation_Wm2", "GFS_Heat_Gap", "Supercomputer_Spread",
        "Model_Bias_Direction", "Consensus_Predicted_High",
        "Forecasted_Lapse_Rate", "Scattering_Ratio", "Pressure_10AM",
        "Ground_to_Air_Gradient", "Yesterday_Bias", "Heat_Trend",
        "Bias_3Day_Mean", "Cooling_Advection", "Temp_10AM_vs_Yesterday",
        "Gap_Wind_Resistance",
    ]

    model = xgb.XGBRegressor()
    model.load_model(MODEL_FILE)

    X_input = df_today[X_COLS]
    heating_delta = model.predict(X_input)[0]
    predicted_high = temp_10am + heating_delta

    print(f"\n  🌡️  MODEL PREDICTION")
    print(f"  Heating Delta  : +{heating_delta:.2f} °C above 10 AM")
    print(f"  Predicted High : {predicted_high:.1f} °C")

    df_probs = calculate_probabilities(predicted_high)
    print("\n" + "=" * 60)
    print("  📊  POLYMARKET PROBABILITY BOARD")
    print(f"      (centred on {predicted_high:.2f} °C, MAE = {MAE} °C)")
    print("=" * 60)
    print(df_probs.to_string(index=False))
    print("=" * 60)

    probs_file = f"today_probabilities_{OUTPUT_SUFFIX}.csv"
    df_probs.to_csv(probs_file, index=False)
    print(f"\n  Saved to → {probs_file}")
else:
    if MODEL_FILE:
        print(f"\n  (Model file '{MODEL_FILE}' not found — skipping inference.)")
        print("  Place the model JSON in the same directory to enable predictions.")
    else:
        print("\n  (No model file configured — skipping inference.)")
