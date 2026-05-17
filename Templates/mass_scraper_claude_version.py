import os
import pandas as pd
import time
from datetime import datetime
from pathlib import Path
import pytz

from Wundergrounds_scraper import scrape_wunderground_history


def mine_historical_oracle(station, date_list, output_csv=None, resume=True):
    """
    Scrape every date in `date_list` (a list of date strings or Timestamps).
    
    :param station: airport code (LEMD for Madrid-Barajas)
    :param date_list: iterable of dates to scrape
    :param output_csv: where to save results (default: {station}_FINAL_ORACLE_DATA.csv)
    :param resume: if True, skip dates already present in output_csv
    """
    if output_csv is None:
        output_csv = f"{station}_FINAL_ORACLE_DATA.csv"

    # Normalise to YYYY-MM-DD strings
    dates_wanted = [pd.Timestamp(d).strftime("%Y-%m-%d") for d in date_list]

    # Resume: skip dates already scraped
    already_done = set()
    if resume and Path(output_csv).exists():
        prior = pd.read_csv(output_csv)
        already_done = set(prior["Date"].astype(str))
        print(f"📂 Resume: {len(already_done)} dates already in {output_csv}")

    todo = [d for d in dates_wanted if d not in already_done]
    if not todo:
        print("✅ Nothing to scrape — all dates already done.")
        return

    print(f"🚀 INITIATING MASS ORACLE EXTRACTION: {station}")
    print(f"   {len(todo)} date(s) to scrape (of {len(dates_wanted)} requested)")
    print(f"   ETA ≈ {len(todo) * 10 / 60:.1f} minutes\n")

    results = []
    for i, date_str in enumerate(todo, 1):
        dt = pd.Timestamp(date_str)
        url_date = f"{dt.year}-{dt.month}-{dt.day}"   # no leading zeros

        print(f"[{i}/{len(todo)}] {date_str}")
        peak_temp = scrape_wunderground_history(station, url_date)

        row = {"Date": date_str, "Oracle_High_C": peak_temp}
        results.append(row)

        # Append to CSV after every row — no lost work on crash
        header_needed = not Path(output_csv).exists()
        pd.DataFrame([row]).to_csv(output_csv, mode="a",
                                    header=header_needed, index=False)

        if i < len(todo):
            print("⏸️ Sleeping 7s to evade IBM bot-detection...\n")
            time.sleep(7)

    print(f"\n🎉 DONE. Saved {len(results)} new rows to {output_csv}")


def dates_from_orderbook(orderbook_csv: str) -> list[str]:
    """Extract unique dates present in the orderbook CSV."""
    ob = pd.read_csv(orderbook_csv, usecols=["date"])
    dates = sorted(set(pd.to_datetime(ob["date"]).dt.strftime("%Y-%m-%d")))
    return dates


# ══════════════════════════════════════════════
if __name__ == "__main__":
    target_station = "LEMD"

    # ── PICK ONE OF THESE MODES ───────────────────────────
    # Mode A — scrape exactly the dates in your orderbook CSV (RECOMMENDED)
    orderbook_csv = "madrid_temperature_orderbooks.csv"
    if Path(orderbook_csv).exists():
        dates = dates_from_orderbook(orderbook_csv)
        print(f"📅 Using {len(dates)} dates from {orderbook_csv}: "
              f"{dates[0]} → {dates[-1]}")
    else:
        # Mode B — hardcoded date range (fallback if orderbook csv not here)
        start_date = "2026-03-17"
        end_date   = "2026-04-19"
        dates = pd.date_range(start=start_date, end=end_date).strftime("%Y-%m-%d").tolist()
        print(f"📅 Using hardcoded range: {start_date} → {end_date} ({len(dates)} days)")

    mine_historical_oracle(target_station, dates, resume=True)
