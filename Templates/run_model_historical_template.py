"""
Historical Predictions Generator
=================================
Runs the city's XGBoost model for every date present in the orderbook CSV
and produces a historical predictions CSV that the backtester consumes.

All city-specific values (coordinates, timezone, IEM station, forecast hours,
model file, MAE, afternoon target window) are loaded from a YAML config file
passed via --config.

Outputs:
    historical_predictions_<suffix>.csv          one row per successful day
    historical_predictions_failures_<suffix>.csv any dates that failed

Usage:
    python run_model_historical.py --config milan.yaml
    python run_model_historical.py --config milan.yaml --resume
"""

import argparse
import time
import warnings
import yaml
from datetime import date, datetime, timedelta
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import pytz
import requests

warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════
# CONFIG LOADING
# ══════════════════════════════════════════════
def _load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


# These are populated by main() after parsing --config.
LAT = LON_STANDARD = LON_GFS = None
TZ = TZ_NAME = STATION = None
MODEL_MAE = None
FXX_GFS_SURFACE = FXX_GFS_850 = FXX_ECMWF = None
TARGET_WIN_START = TARGET_WIN_END = None

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


# ══════════════════════════════════════════════
# 1. IEM
# ══════════════════════════════════════════════
def fetch_iem(start_date, end_date, max_retries=5, base_wait=10):
    iem_end = end_date + timedelta(days=1)
    url = (
        f"https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?"
        f"station={STATION}&data=tmpf&data=dwpf&data=drct&data=sknt&"
        f"data=skyc1&data=alti&"
        f"year1={start_date.year}&month1={start_date.month}&day1={start_date.day}&"
        f"year2={iem_end.year}&month2={iem_end.month}&day2={iem_end.day}&"
        f"tz=Etc/UTC&format=onlycomma&latlon=no&missing=empty"
    )
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            df = pd.read_csv(StringIO(resp.text), parse_dates=["valid"])
            break
        except Exception as e:
            wait = base_wait * (2 ** (attempt - 1))
            if attempt < max_retries:
                print(f"    [IEM] Attempt {attempt} failed ({e}). Retrying in {wait}s")
                time.sleep(wait)
            else:
                raise RuntimeError(f"IEM failed after {max_retries} attempts: {e}")

    df.rename(columns={
        "valid": "Timestamp", "tmpf": "Temp_F", "dwpf": "Dew_Point_F",
        "drct": "Wind_Dir_Deg", "sknt": "Wind_Speed_Kts", "skyc1": "Cloud_Cover_Code",
    }, inplace=True)

    cloud_map = {"CLR": 0, "SKC": 0, "FEW": 2, "SCT": 4, "BKN": 6, "OVC": 8, "VV ": 8}
    df["Cloud_Cover_Okta"] = df["Cloud_Cover_Code"].map(cloud_map).fillna(0)
    df.drop(columns=["Cloud_Cover_Code"], inplace=True)
    df["alti"] = pd.to_numeric(df["alti"], errors="coerce").ffill()
    df = df.dropna(subset=["Temp_F", "Wind_Dir_Deg", "Dew_Point_F"])
    df["Temp_C"] = (df["Temp_F"] - 32) * (5.0 / 9.0)
    df["Dew_Point_C"] = (df["Dew_Point_F"] - 32) * (5.0 / 9.0)
    df.drop(columns=["Temp_F", "Dew_Point_F", "station"], inplace=True, errors="ignore")
    df["Timestamp"] = (
        df["Timestamp"].dt.tz_localize("UTC").dt.tz_convert(TZ_NAME)
    )
    df.set_index("Timestamp", inplace=True)
    df.sort_index(inplace=True)
    return df


def snapshot_at(df, target_date, start_time, end_time, agg="last"):
    working = df.tz_convert(TZ) if df.index.tz is not None else df.tz_localize("UTC").tz_convert(TZ)
    date_str = str(target_date)
    day_df = working[working.index.strftime("%Y-%m-%d") == date_str]
    window = day_df.between_time(start_time, end_time)
    if window.empty:
        return None
    return window.iloc[-1] if agg == "last" else window


def actual_max(df, target_date):
    """Observed max over the configured afternoon window for a given date."""
    day_df = df[df.index.date == target_date]
    afternoon = day_df.between_time(TARGET_WIN_START, TARGET_WIN_END)
    if afternoon.empty:
        return np.nan
    return afternoon["Temp_C"].max()


# ══════════════════════════════════════════════
# 2. HERBIE — GFS / ECMWF
# ══════════════════════════════════════════════
def fetch_herbie_surface(target_date, model, product, fxx_list,
                         lat, lon, var_name, search_str, priority):
    from herbie import Herbie
    date_str = target_date.strftime("%Y-%m-%d")
    temps = []
    for fxx in fxx_list:
        try:
            H   = Herbie(date_str, model=model, product=product, fxx=fxx, priority=priority)
            ds  = H.xarray(search_str)
            val = ds[var_name].sel(latitude=lat, longitude=lon, method="nearest").values
            temps.append(float(val) - 273.15)
        except Exception as e:
            print(f"    [WARN] {model} fxx={fxx} failed: {e}")
        time.sleep(0.3)
    return round(float(np.max(temps)), 2) if temps else np.nan


# ══════════════════════════════════════════════
# 3. OPEN-METEO ARCHIVE
# ══════════════════════════════════════════════
def fetch_openmeteo_10am(target_date):
    date_str = target_date.strftime("%Y-%m-%d")
    for endpoint in ("https://archive-api.open-meteo.com/v1/archive",
                     "https://api.open-meteo.com/v1/forecast"):
        try:
            resp = requests.get(endpoint, params={
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
                "timezone": TZ_NAME,
            }, timeout=30)
            resp.raise_for_status()
            hourly = resp.json().get("hourly", {})
            times  = pd.to_datetime(hourly["time"])
            df_h = pd.DataFrame({
                "time":               times,
                "DNI_Radiation_Wm2":  hourly["direct_normal_irradiance"],
                "Soil_Temp_7cm":      hourly["soil_temperature_0_to_7cm"],
                "Soil_Moisture_7cm":  hourly["soil_moisture_0_to_7cm"],
                "DHI_Radiation_Wm2":  hourly["diffuse_radiation"],
            })
            row_10 = df_h[df_h["time"].dt.hour == 10]
            if not row_10.empty:
                return row_10.iloc[0]
        except Exception as e:
            print(f"    [WARN] Open-Meteo {endpoint} failed: {e}")
    raise RuntimeError("No Open-Meteo 10 AM data found.")


# ══════════════════════════════════════════════
# 4. CORE: BUILD FEATURE ROW FOR A DATE
# ══════════════════════════════════════════════
def build_features_for_date(target_date: date) -> tuple[dict, float]:
    """Returns (features_dict, actual_max_temp) for the given date."""
    yesterday      = target_date - timedelta(days=1)
    two_days_ago   = target_date - timedelta(days=2)
    three_days_ago = target_date - timedelta(days=3)

    df_raw = fetch_iem(three_days_ago, target_date)

    row_10am = snapshot_at(df_raw, target_date, "09:30", "10:30")
    if row_10am is None:
        raise RuntimeError(f"No 10 AM observation on {target_date}")
    row_7am = snapshot_at(df_raw, target_date, "06:30", "07:30")
    if row_7am is None:
        raise RuntimeError(f"No 7 AM observation on {target_date}")

    overnight_df = df_raw[df_raw.index.date == target_date].between_time("00:00", "06:00")
    if overnight_df.empty:
        raise RuntimeError(f"No overnight observations on {target_date}")
    overnight_low = overnight_df["Temp_C"].min()

    actual_max_today     = actual_max(df_raw, target_date)
    actual_max_yesterday = actual_max(df_raw, yesterday)
    actual_max_2days_ago = actual_max(df_raw, two_days_ago)
    actual_max_3days_ago = actual_max(df_raw, three_days_ago)

    # ── 2. GFS + ECMWF
    gfs_predicted_high = fetch_herbie_surface(
        target_date, "gfs", "pgrb2.0p25", FXX_GFS_SURFACE,
        LAT, LON_GFS, "t2m", "TMP:2 m above ground", ["aws", "nomads"],
    )
    gfs_850_temp = fetch_herbie_surface(
        target_date, "gfs", "pgrb2.0p25", FXX_GFS_850,
        LAT, LON_GFS, "t", "TMP:850 mb", ["aws", "nomads"],
    )
    ecmwf_predicted_high = fetch_herbie_surface(
        target_date, "ifs", "oper", FXX_ECMWF,
        LAT, LON_STANDARD, "t2m", ":2t:", ["azure", "aws"],
    )

    gfs_high_yest = fetch_herbie_surface(
        yesterday, "gfs", "pgrb2.0p25", FXX_GFS_SURFACE,
        LAT, LON_GFS, "t2m", "TMP:2 m above ground", ["aws", "nomads"],
    )
    gfs_high_2d = fetch_herbie_surface(
        two_days_ago, "gfs", "pgrb2.0p25", FXX_GFS_SURFACE,
        LAT, LON_GFS, "t2m", "TMP:2 m above ground", ["aws", "nomads"],
    )
    gfs_high_3d = fetch_herbie_surface(
        three_days_ago, "gfs", "pgrb2.0p25", FXX_GFS_SURFACE,
        LAT, LON_GFS, "t2m", "TMP:2 m above ground", ["aws", "nomads"],
    )

    # ── 3. Open-Meteo
    om_today = fetch_openmeteo_10am(target_date)
    dni  = float(om_today["DNI_Radiation_Wm2"])
    dhi  = float(om_today["DHI_Radiation_Wm2"])
    soil_temp  = float(om_today["Soil_Temp_7cm"])
    soil_moist = float(om_today["Soil_Moisture_7cm"])

    # ── 4. Assemble features
    temp_10am = float(row_10am["Temp_C"])
    dew_10am  = float(row_10am["Dew_Point_C"])
    wind_dir  = float(row_10am["Wind_Dir_Deg"])
    wind_spd  = float(row_10am["Wind_Speed_Kts"])
    cloud_10  = float(row_10am["Cloud_Cover_Okta"])
    alti_10   = float(row_10am["alti"])

    temp_7am  = float(row_7am["Temp_C"])
    dew_7am   = float(row_7am["Dew_Point_C"])
    alti_7am  = float(row_7am["alti"])

    temp_change   = temp_10am - temp_7am
    dew_change    = dew_10am - dew_7am
    press_change  = alti_10 - alti_7am
    heat_mom      = temp_change - dew_change

    rh = 100 * (
        np.exp((17.625 * dew_10am)  / (243.04 + dew_10am)) /
        np.exp((17.625 * temp_10am) / (243.04 + temp_10am))
    )
    early_heat = temp_10am - overnight_low

    doy = target_date.timetuple().tm_yday
    wind_sin = np.sin(wind_dir * 2 * np.pi / 360)
    wind_cos = np.cos(wind_dir * 2 * np.pi / 360)
    day_sin  = np.sin(doy * 2 * np.pi / 366)
    day_cos  = np.cos(doy * 2 * np.pi / 366)

    gfs_heat_gap    = gfs_predicted_high - temp_10am
    super_spread    = abs(gfs_predicted_high - ecmwf_predicted_high)
    bias_direction  = gfs_predicted_high - ecmwf_predicted_high
    consensus       = (gfs_predicted_high + ecmwf_predicted_high) / 2
    lapse_rate      = gfs_predicted_high - gfs_850_temp
    scatter_ratio   = dhi / (dni + 1)
    ground_air_grad = soil_temp - temp_10am

    bias_d1 = actual_max_yesterday - gfs_high_yest
    bias_d2 = actual_max_2days_ago - gfs_high_2d
    bias_d3 = actual_max_3days_ago - gfs_high_3d
    yesterday_bias = bias_d1
    heat_trend     = actual_max_yesterday - actual_max_2days_ago
    bias_3d_mean   = np.nanmean([bias_d1, bias_d2, bias_d3])

    row_10_yest = snapshot_at(df_raw, yesterday, "09:30", "10:30")
    temp_10_yest = float(row_10_yest["Temp_C"]) if row_10_yest is not None else np.nan
    temp_vs_yest = temp_10am - temp_10_yest

    gap_wind_res  = gfs_heat_gap / (wind_spd + 1)
    cooling_advec = wind_spd * wind_cos

    features = {
        "Temp_10AM": temp_10am,
        "Dew_Point_10AM": dew_10am,
        "Wind_Speed_10AM": wind_spd,
        "Cloud_Cover_10AM": cloud_10,
        "GFS_Predicted_High": gfs_predicted_high,
        "ECMWF_Predicted_High": ecmwf_predicted_high,
        "GFS_850hPa_Temp": gfs_850_temp,
        "DNI_Radiation_Wm2": dni,
        "Soil_Temp_7cm": soil_temp,
        "Soil_Moisture_7cm": soil_moist,
        "DHI_Radiation_Wm2": dhi,
        "Temp_Change_3hr": temp_change,
        "Dew_Point_Change_3hr": dew_change,
        "Pressure_Change_3hr": press_change,
        "Heat_Momentum_Ratio": heat_mom,
        "RH_10AM": rh,
        "Early_Heating_Effort": early_heat,
        "Overnight_Low": overnight_low,
        "Wind_Sin": wind_sin,
        "Wind_Cos": wind_cos,
        "Day_Sin": day_sin,
        "Day_Cos": day_cos,
        "GFS_Heat_Gap": gfs_heat_gap,
        "Supercomputer_Spread": super_spread,
        "Model_Bias_Direction": bias_direction,
        "Consensus_Predicted_High": consensus,
        "Forecasted_Lapse_Rate": lapse_rate,
        "Scattering_Ratio": scatter_ratio,
        "Pressure_10AM": alti_10,
        "Ground_to_Air_Gradient": ground_air_grad,
        "Yesterday_Bias": yesterday_bias,
        "Heat_Trend": heat_trend,
        "Bias_3Day_Mean": bias_3d_mean,
        "Temp_10AM_vs_Yesterday": temp_vs_yest,
        "Gap_Wind_Resistance": gap_wind_res,
        "Cooling_Advection": cooling_advec,
    }
    return features, float(actual_max_today)


def predict_for_date(target_date: date, model) -> dict:
    features, actual_max_today = build_features_for_date(target_date)
    df_input = pd.DataFrame([features])[X_COLS]
    heating_delta = float(model.predict(df_input)[0])
    predicted_high = features["Temp_10AM"] + heating_delta
    return {
        "date":                target_date.isoformat(),
        "predicted_high":      round(predicted_high, 2),
        "actual_max_temp":     round(actual_max_today, 2) if not np.isnan(actual_max_today) else np.nan,
        "heating_delta":       round(heating_delta, 2),
        "temp_10am":           round(features["Temp_10AM"], 2),
        "gfs_predicted_high":  features["GFS_Predicted_High"],
        "ecmwf_predicted_high": features["ECMWF_Predicted_High"],
        "model_mae":           MODEL_MAE,
    }


# ══════════════════════════════════════════════
# 5. MAIN LOOP
# ══════════════════════════════════════════════
def main():
    global LAT, LON_STANDARD, LON_GFS, TZ, TZ_NAME, STATION, MODEL_MAE
    global FXX_GFS_SURFACE, FXX_GFS_850, FXX_ECMWF
    global TARGET_WIN_START, TARGET_WIN_END

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", "-c", required=True, help="Path to YAML config (e.g. milan.yaml).")
    ap.add_argument("--orderbook", default=None,
                    help="Orderbook CSV. Defaults to <suffix>_temperature_orderbooks.csv from config.")
    ap.add_argument("--model", default=None,
                    help="XGBoost model file. Defaults to model.file from config.")
    ap.add_argument("--out", default=None,
                    help="Predictions output CSV. Defaults to historical_predictions_<suffix>.csv.")
    ap.add_argument("--failures", default=None,
                    help="Failures CSV. Defaults to historical_predictions_failures_<suffix>.csv.")
    ap.add_argument("--resume", action="store_true", help="Skip dates already present in --out")
    args = ap.parse_args()

    # ── Load YAML config and populate module globals
    cfg = _load_config(args.config)
    LAT             = float(cfg["location"]["latitude"])
    LON_STANDARD    = float(cfg["location"]["longitude"])
    LON_GFS         = LON_STANDARD + 360 if LON_STANDARD < 0 else LON_STANDARD
    TZ_NAME         = cfg["location"]["timezone"]
    TZ              = pytz.timezone(TZ_NAME)
    STATION         = cfg["iem"]["station"]
    MODEL_MAE       = float(cfg["model"]["mae"])
    FXX_GFS_SURFACE = list(cfg["forecast_hours"]["gfs_surface"])
    FXX_GFS_850     = list(cfg["forecast_hours"]["gfs_850hpa"])
    FXX_ECMWF       = list(cfg["forecast_hours"]["ecmwf_surface"])
    TARGET_WIN_START = str(cfg["target_window"]["start"])
    TARGET_WIN_END   = str(cfg["target_window"]["end"])

    suffix = cfg["city"]["output_suffix"]

    # ── Resolve file paths (CLI overrides → config defaults)
    orderbook_path = args.orderbook or f"{suffix}_temperature_orderbooks.csv"
    model_path     = args.model     or cfg["model"]["file"]
    out_path       = args.out       or f"historical_predictions_{suffix}.csv"
    failures_path  = args.failures  or f"historical_predictions_failures_{suffix}.csv"

    # ── Read orderbook to learn which dates need predictions
    ob = pd.read_csv(orderbook_path, usecols=["date"])
    target_dates = sorted(set(pd.to_datetime(ob["date"]).dt.date))
    print(f"Orderbook has {len(target_dates)} unique dates.")

    existing = set()
    if args.resume and Path(out_path).exists():
        done = pd.read_csv(out_path)
        existing = set(pd.to_datetime(done["date"]).dt.date)
        print(f"Resume: {len(existing)} dates already predicted, skipping.")

    to_process = [d for d in target_dates if d not in existing]
    print(f"Will process {len(to_process)} date(s).")

    import xgboost as xgb
    model = xgb.XGBRegressor()
    model.load_model(model_path)

    results, failures = [], []

    for i, d in enumerate(to_process, 1):
        print(f"\n[{i}/{len(to_process)}] {d}")
        try:
            row = predict_for_date(d, model)
            print(f"    predicted={row['predicted_high']}°C  "
                  f"actual={row['actual_max_temp']}°C  "
                  f"Δ={row['heating_delta']:+.2f}°C")
            results.append(row)
        except Exception as e:
            print(f"    !! FAILED: {e}")
            failures.append({"date": d.isoformat(), "error": str(e)})

        if i % 5 == 0 or i == len(to_process):
            if results:
                new_df = pd.DataFrame(results)
                if args.resume and Path(out_path).exists():
                    old_df = pd.read_csv(out_path)
                    pd.concat([old_df, new_df], ignore_index=True).to_csv(out_path, index=False)
                else:
                    new_df.to_csv(out_path, index=False)
                print(f"    [saved {len(results)} new rows]")
            if failures:
                pd.DataFrame(failures).to_csv(failures_path, index=False)

    print(f"\nDone. {len(results)} success, {len(failures)} failures.")
    print(f"Predictions → {out_path}")
    if failures:
        print(f"Failures    → {failures_path}")


if __name__ == "__main__":
    main()
