"""
City Temperature Model — Polymarket Backtester
==================================================
Replays the trading strategy against historical Polymarket orderbooks.

City-specific values (timezone, MAE, output filenames) come from a YAML
config file passed via --config.

For every day with both a model prediction AND orderbook data:
    1. Build the model's probability distribution from the predicted high
       (Normal centred on prediction, σ = MAE × 1.2533 × uncertainty_mult).
    2. Within the limit-order window, scan every snapshot. For each (band, side):
         - skip if model_prob outside [MIN_MODEL_PROB, MAX_MODEL_PROB]
         - compute EV = model_prob × (1/ask) − 1
         - if EV ∈ [MIN_EV, MAX_EV], place a Kelly-sized bet at the ask price.
    3. Settle at end of day via Wunderground > orderbook inference > IEM fallback.
    4. Aggregate into ROI, hit rate, equity curve, etc.

Outputs:
    backtest_trades_<suffix>.csv   — one row per trade placed
    backtest_daily_<suffix>.csv    — one row per day (aggregated PnL)
    backtest_summary_<suffix>.txt  — overall ROI, hit rate, Sharpe-ish metrics
"""

import os
import math
import argparse
import yaml
from pathlib import Path
from datetime import date

import numpy as np
import pandas as pd
from scipy import stats


# ══════════════════════════════════════════════
# CONFIG — strategy parameters (universal, not city-specific)
# ══════════════════════════════════════════════
# LIMIT-ORDER WINDOW: orders are "placed" at the start of the window and
# fill the first time a snapshot shows the ask at/below our limit price.
# After filling once, that (band, side) is done for the day.
LIMIT_WINDOW_START = "10:00"
LIMIT_WINDOW_END   = "13:00"
MIN_MODEL_PROB     = 0.5
MAX_MODEL_PROB     = 0.9
MAX_EV             = 0.2
MIN_EV             = 0.1
BANKROLL_START     = 1000.0
KELLY_FRACTION     = 0.1
UNCERTAINTY_MULTIPLIER = 1.12
ALPHA = 0.9

# Populated by main() from the YAML config:
DEFAULT_MAE: float = 1.0
TZ_NAME: str = ""
TEMP_RANGE_LO: int = 0
TEMP_RANGE_HI: int = 46


# ══════════════════════════════════════════════
# 1. MODEL PROBABILITY BOARD
# ══════════════════════════════════════════════
def model_probs_for_day(predicted_high: float, mae: float = None,
                        temps: range = None) -> dict[int, float]:
    """
    Returns a full dict {temp_int: prob} for the integer °C bands in `temps`.
    Mixes a Normal centred on the prediction with a uniform prior (controlled
    by ALPHA) to dampen overconfidence.
    """
    if mae is None:
        mae = DEFAULT_MAE
    if temps is None:
        temps = range(TEMP_RANGE_LO, TEMP_RANGE_HI)

    std = mae * 1.2533 * UNCERTAINTY_MULTIPLIER
    raw = {}
    for t in temps:
        p_lo = stats.norm.cdf(t - 0.5, loc=predicted_high, scale=std)
        p_hi = stats.norm.cdf(t + 0.5, loc=predicted_high, scale=std)
        raw[t] = float(p_hi - p_lo)

    uniform_p = 1.0 / len(temps)
    out = {
        t: ALPHA * raw[t] + (1 - ALPHA) * uniform_p
        for t in temps
    }
    return out


# ══════════════════════════════════════════════
# 2. KELLY SIZING
# ══════════════════════════════════════════════
def kelly_fraction(p: float, price: float) -> float:
    """
    Kelly fraction for a binary bet at `price` with model probability `p`.
    Returns a fraction in [0, 1]; 0 if the bet has no edge.
    """
    if price <= 0 or price >= 1:
        return 0.0
    edge = p - price
    if edge <= 0:
        return 0.0
    f = edge / (1.0 - price)
    return max(0.0, min(1.0, f))


# ══════════════════════════════════════════════
# 3. DATA LOADING
# ══════════════════════════════════════════════
def load_orderbook(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"]).dt.date

    # Unified column name from the data-extraction script.
    # (Old files used "timestamp_madrid" or "timestamp_city" — accept either.)
    ts_col = None
    for cand in ("timestamp_local", "timestamp_city", "timestamp_madrid"):
        if cand in df.columns:
            ts_col = cand
            break
    if ts_col is None:
        raise ValueError(
            "Orderbook CSV must contain a timestamp column "
            "(timestamp_local, timestamp_city, or timestamp_madrid)."
        )
    df = df.rename(columns={ts_col: "timestamp_local"})

    df["timestamp_local"] = pd.to_datetime(df["timestamp_local"], utc=True)
    df["clock"] = df["timestamp_local"].dt.tz_convert(TZ_NAME).dt.strftime("%H:%M")
    return df


def load_predictions(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    if "model_mae" not in df.columns:
        df["model_mae"] = DEFAULT_MAE
    required = {"date", "predicted_high"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"predictions CSV missing columns: {missing}")
    if "actual_max_temp" not in df.columns:
        df["actual_max_temp"] = np.nan
    return df


def load_wunderground(path: str) -> dict:
    """
    Load Wunderground-scraped max temps. AUTHORITATIVE resolution source —
    matches Polymarket's actual settlement because it's the same IBM data.

    Accepts either {Date, Oracle_High_C} or {date, max_temp_c} schemas.
    Returns {date: max_temp_c int}.
    """
    if not path or not Path(path).exists():
        return {}

    df = pd.read_csv(path)

    if "Date" in df.columns and "Oracle_High_C" in df.columns:
        df = df.rename(columns={"Date": "date", "Oracle_High_C": "max_temp_c"})
    elif "date" in df.columns and "max_temp_c" in df.columns:
        pass
    else:
        raise ValueError(
            f"{path} must have columns [Date, Oracle_High_C] or [date, max_temp_c]. "
            f"Got: {list(df.columns)}"
        )

    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df.dropna(subset=["max_temp_c"])
    df["max_temp_c"] = df["max_temp_c"].astype(int)
    return dict(zip(df["date"], df["max_temp_c"]))


# ══════════════════════════════════════════════
# 4. MARKET RESOLUTION — derive winning temp from orderbook
# ══════════════════════════════════════════════
def winning_temp_from_orderbook(ob_today: pd.DataFrame) -> int | None:
    """
    Determine which temperature band WON according to Polymarket's own
    resolution — by inspecting the end-of-day orderbook.
    """
    yes = ob_today[ob_today["option_label"] == "Yes"].copy()
    if yes.empty:
        return None

    yes["resolved_price"] = yes["best_bid"].fillna(yes["best_ask"])
    yes = yes.dropna(subset=["resolved_price"])
    if yes.empty:
        return None

    yes = yes.sort_values("timestamp_local")
    last_per_band = yes.groupby("temperature_c").tail(1)

    max_price = last_per_band["resolved_price"].max()
    if max_price < 0.7:
        return None

    winner = last_per_band.loc[last_per_band["resolved_price"].idxmax()]
    try:
        return int(winner["temperature_c"])
    except (ValueError, TypeError):
        return None


# ══════════════════════════════════════════════
# 5. TRADE SIMULATION — one day
# ══════════════════════════════════════════════
def simulate_day(
    date,
    predicted_high: float,
    actual_max: float,
    mae: float,
    ob_today: pd.DataFrame,
    wunderground_temp: int | None = None,
) -> list[dict]:
    """
    Replay one day of limit-order trading.

    Resolution priority:
        1. Wunderground (Oracle) — Polymarket's actual data source
        2. Orderbook inference — end-of-day price says who won
        3. IEM observed max — last-resort fallback
    """
    window_ob = ob_today[
        (ob_today["clock"] >= LIMIT_WINDOW_START) &
        (ob_today["clock"] <= LIMIT_WINDOW_END)
    ].sort_values("timestamp_local")

    if window_ob.empty:
        return []

    if wunderground_temp is not None:
        winning_temp = int(wunderground_temp)
        resolution_source = "wunderground"
    else:
        winning_temp = winning_temp_from_orderbook(ob_today)
        if winning_temp is not None:
            resolution_source = "orderbook"
        else:
            winning_temp = int(round(actual_max))
            resolution_source = "iem_observed"

    probs = model_probs_for_day(predicted_high, mae)

    trades = []
    filled = set()

    for ts, snap in window_ob.groupby("timestamp_local", sort=True):
        clock = snap["clock"].iloc[0]

        for temp, grp in snap.groupby("temperature_c"):
            try:
                temp_int = int(temp)
            except (ValueError, TypeError):
                continue

            model_p_yes = probs.get(temp_int, 0.0)
            model_p_no  = 1.0 - model_p_yes

            yes_row = grp[grp["option_label"] == "Yes"]
            no_row  = grp[grp["option_label"] == "No"]

            yes_ask = yes_row["best_ask"].iloc[0] if not yes_row.empty else np.nan
            no_ask  = no_row["best_ask"].iloc[0]  if not no_row.empty  else np.nan

            # ── YES side
            key_yes = (temp_int, "Yes")
            if (key_yes not in filled
                and model_p_yes >= MIN_MODEL_PROB
                and not np.isnan(yes_ask)):
                ev = model_p_yes / yes_ask - 1.0
                if ev > MIN_EV and ev < MAX_EV:
                    kelly = kelly_fraction(model_p_yes, yes_ask) * KELLY_FRACTION
                    won = (temp_int == winning_temp)
                    pnl_per_dollar = (1.0 / yes_ask - 1.0) if won else -1.0
                    trades.append({
                        "date":           date,
                        "fill_time":      clock,
                        "temperature_c":  temp_int,
                        "side":           "Yes",
                        "model_prob":     model_p_yes,
                        "market_price":   yes_ask,
                        "ev":             ev,
                        "kelly_frac":     kelly,
                        "won":            won,
                        "pnl_per_dollar": pnl_per_dollar,
                        "predicted_high": predicted_high,
                        "actual_max":     actual_max,
                        "winning_temp":   winning_temp,
                        "resolution":     resolution_source,
                    })
                    filled.add(key_yes)

            # ── NO side
            key_no = (temp_int, "No")
            if (key_no not in filled
                and model_p_no <= MAX_MODEL_PROB
                and not np.isnan(no_ask)):
                ev = model_p_no / no_ask - 1.0
                if ev > MIN_EV and ev < MAX_EV:
                    kelly = kelly_fraction(model_p_no, no_ask) * KELLY_FRACTION
                    won = (temp_int != winning_temp)
                    pnl_per_dollar = (1.0 / no_ask - 1.0) if won else -1.0
                    trades.append({
                        "date":           date,
                        "fill_time":      clock,
                        "temperature_c":  temp_int,
                        "side":           "No",
                        "model_prob":     model_p_no,
                        "market_price":   no_ask,
                        "ev":             ev,
                        "kelly_frac":     kelly,
                        "won":            won,
                        "pnl_per_dollar": pnl_per_dollar,
                        "predicted_high": predicted_high,
                        "actual_max":     actual_max,
                        "winning_temp":   winning_temp,
                        "resolution":     resolution_source,
                    })
                    filled.add(key_no)

    return trades


# ══════════════════════════════════════════════
# 6. EQUITY CURVE — bankroll-based Kelly
# ══════════════════════════════════════════════
def apply_kelly_bankroll(trades_df: pd.DataFrame,
                         bankroll_start: float = BANKROLL_START) -> pd.DataFrame:
    """Simulate a real bankroll applying the Kelly fraction day-by-day."""
    trades_df = trades_df.sort_values("date").copy()
    bankroll = bankroll_start
    stakes, pnls, post_bank = [], [], []

    for date, daily in trades_df.groupby("date", sort=True):
        day_start_bank = bankroll
        total_kelly = daily["kelly_frac"].sum()
        scale = 1.0 if total_kelly <= 1.0 else 1.0 / total_kelly

        day_pnl = 0.0
        for _, trade in daily.iterrows():
            stake = day_start_bank * trade["kelly_frac"] * scale
            pnl   = stake * trade["pnl_per_dollar"]
            stakes.append(stake)
            pnls.append(pnl)
            post_bank.append(None)
            day_pnl += pnl

        bankroll += day_pnl
        for i in range(1, len(daily) + 1):
            post_bank[-i] = bankroll

    trades_df["stake"]          = stakes
    trades_df["pnl"]            = pnls
    trades_df["bankroll_after"] = post_bank
    return trades_df


# ══════════════════════════════════════════════
# 7. REPORTING
# ══════════════════════════════════════════════
def summarize(trades_df: pd.DataFrame, bankroll_start: float, city_name: str) -> str:
    if trades_df.empty:
        return "No trades placed."

    n_trades   = len(trades_df)
    n_wins     = int(trades_df["won"].sum())
    hit_rate   = n_wins / n_trades
    total_stake = trades_df["stake"].sum()
    total_pnl  = trades_df["pnl"].sum()
    roi_on_stake = total_pnl / total_stake if total_stake > 0 else 0.0
    bankroll_end = bankroll_start + total_pnl
    roi_bankroll = (bankroll_end / bankroll_start) - 1.0

    daily = (trades_df.groupby("date")
                      .agg(daily_pnl=("pnl", "sum"),
                           daily_stake=("stake", "sum"))
                      .reset_index())
    daily["daily_return"] = daily["daily_pnl"] / daily["daily_stake"].replace(0, np.nan)
    n_days    = len(daily)
    n_winning_days = int((daily["daily_pnl"] > 0).sum())
    mean_ret  = daily["daily_return"].mean()
    std_ret   = daily["daily_return"].std()
    sharpe    = (mean_ret / std_ret * math.sqrt(252)) if std_ret and std_ret > 0 else float("nan")

    avg_edge  = trades_df["ev"].mean()
    avg_kelly = trades_df["kelly_frac"].mean()

    lines = [
        "═" * 60,
        f"  {city_name.upper()} TEMPERATURE MODEL — BACKTEST SUMMARY",
        "═" * 60,
        f"  Date range          : {trades_df['date'].min()} → {trades_df['date'].max()}",
        f"  Trading days        : {n_days}",
        f"  Winning days        : {n_winning_days} ({n_winning_days/n_days:.1%})",
        f"  Total trades        : {n_trades}",
        f"  Winning trades      : {n_wins} ({hit_rate:.1%})",
        "",
        f"  Starting bankroll   : ${bankroll_start:,.2f}",
        f"  Ending bankroll     : ${bankroll_end:,.2f}",
        f"  Total PnL           : ${total_pnl:,.2f}",
        f"  ROI on bankroll     : {roi_bankroll:+.2%}",
        f"  ROI on capital used : {roi_on_stake:+.2%}",
        "",
        f"  Avg edge (EV)       : {avg_edge:+.2%}",
        f"  Avg Kelly fraction  : {avg_kelly:.3f}",
        f"  Total staked        : ${total_stake:,.2f}",
        f"  Daily Sharpe (252d) : {sharpe:.2f}",
        "═" * 60,
    ]
    return "\n".join(lines)


# ══════════════════════════════════════════════
# 8. MAIN
# ══════════════════════════════════════════════
def _load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    global DEFAULT_MAE, TZ_NAME, TEMP_RANGE_LO, TEMP_RANGE_HI

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", "-c", required=True,
                    help="Path to YAML config (e.g. milan.yaml).")
    ap.add_argument("--orderbook", default=None,
                    help="Path to orderbook CSV (default: <suffix>_temperature_orderbooks.csv)")
    ap.add_argument("--predictions", default=None,
                    help="Path to historical model predictions "
                         "(default: historical_predictions_<suffix>.csv)")
    ap.add_argument("--wunderground", default="Station_FINAL_ORACLE_DATA.csv",
                    help="Path to Wunderground scraper output (authoritative resolution).")
    ap.add_argument("--bankroll", type=float, default=BANKROLL_START)
    ap.add_argument("--outdir", default=".")
    args = ap.parse_args()

    # ── Apply YAML config
    cfg = _load_config(args.config)
    DEFAULT_MAE   = float(cfg["model"]["mae"])
    TZ_NAME       = cfg["location"]["timezone"]
    lo, hi        = cfg.get("temp_range", [0, 46])
    TEMP_RANGE_LO, TEMP_RANGE_HI = int(lo), int(hi)

    suffix     = cfg["city"]["output_suffix"]
    city_name  = cfg["city"]["name"]
    orderbook_path   = args.orderbook   or f"{suffix}_temperature_orderbooks.csv"
    predictions_path = args.predictions or f"historical_predictions_{suffix}.csv"

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ── Load data
    ob = load_orderbook(orderbook_path)
    preds = load_predictions(predictions_path)
    oracle = load_wunderground(args.wunderground)

    dates_ob   = set(ob["date"].unique())
    dates_pred = set(preds["date"].unique())
    common     = sorted(dates_ob & dates_pred)

    print(f"Orderbook dates    : {len(dates_ob)}")
    print(f"Prediction dates   : {len(dates_pred)}")
    print(f"Wunderground dates : {len(oracle)}  (authoritative resolution source)")
    print(f"Overlap to process : {len(common)}")
    if not common:
        raise SystemExit("No common dates — check your inputs.")

    dates_without_oracle = [d for d in common if d not in oracle]
    if dates_without_oracle:
        print(f"\n⚠  {len(dates_without_oracle)} date(s) have no Wunderground data — "
              f"will fall back to orderbook/IEM resolution:")
        for d in dates_without_oracle[:10]:
            print(f"     {d}")
        if len(dates_without_oracle) > 10:
            print(f"     ... and {len(dates_without_oracle) - 10} more")

    # ── Replay each day
    all_trades = []
    for d in common:
        pred_row = preds[preds["date"] == d].iloc[0]
        ob_today = ob[ob["date"] == d]

        if pd.isna(pred_row["predicted_high"]):
            continue
        if oracle.get(d) is None and pd.isna(pred_row["actual_max_temp"]):
            print(f"  [{d}] no resolution source available — skipping")
            continue

        trades = simulate_day(
            date=d,
            predicted_high=float(pred_row["predicted_high"]),
            actual_max=float(pred_row["actual_max_temp"]),
            mae=float(pred_row["model_mae"]),
            ob_today=ob_today,
            wunderground_temp=oracle.get(d),
        )
        all_trades.extend(trades)

    if not all_trades:
        print("No positive-EV trades found.")
        return

    trades_df = pd.DataFrame(all_trades)
    trades_df = apply_kelly_bankroll(trades_df, args.bankroll)

    daily = (trades_df.groupby("date")
                      .agg(n_trades       =("pnl", "size"),
                           n_wins         =("won", "sum"),
                           total_stake    =("stake", "sum"),
                           total_pnl      =("pnl", "sum"),
                           bankroll_after =("bankroll_after", "last"))
                      .reset_index())
    daily["daily_roi"] = daily["total_pnl"] / daily["total_stake"]

    trades_path  = outdir / f"backtest_trades_{suffix}.csv"
    daily_path   = outdir / f"backtest_daily_{suffix}.csv"
    summary_path = outdir / f"backtest_summary_{suffix}.txt"

    trades_df.to_csv(trades_path, index=False)
    daily.to_csv(daily_path, index=False)
    summary = summarize(trades_df, args.bankroll, city_name)
    summary_path.write_text(summary, encoding="utf-8")

    print()
    print(summary)
    print()
    print("Saved:")
    print(f"  {trades_path}")
    print(f"  {daily_path}")
    print(f"  {summary_path}")


if __name__ == "__main__":
    main()
