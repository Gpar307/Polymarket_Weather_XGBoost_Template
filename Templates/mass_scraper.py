import pandas as pd
import time
from datetime import datetime
import pytz

# Import the single-day Ghost Browser function we built earlier
# Make sure oracle_scraper.py is in the same folder!
from Wundergrounds_scraper import scrape_wunderground_history


def mine_historical_oracle(station, start_date_str, end_date_str):
    print(f"🚀 INITIATING MASS ORACLE EXTRACTION: {station}")
    print("WARNING: This will take several hours. Do not close your laptop.")

    # 1. Generate every single calendar day between your start and end dates
    date_list = pd.date_range(start=start_date_str, end=end_date_str)

    results = []

    for dt in date_list:
        # Format the date exactly how Wunderground's URL expects it (YYYY-M-D)
        # We use .day and .month to avoid leading zeros (e.g., 2023-05-01 -> 2023-5-1)
        url_date = f"{dt.year}-{dt.month}-{dt.day}"

        # 2. Run your Ghost Browser for this specific day
        peak_temp = scrape_wunderground_history(station, url_date)

        # 3. Save the result
        results.append({
            "Date": dt.strftime('%Y-%m-%d'),
            "Oracle_High_C": peak_temp
        })

        # 4. THE THROTTLE (CRITICAL)
        # If you do not pause for 7 seconds, IBM will permanently ban your IP address.
        print("⏸️ Sleeping for 7 seconds to evade IBM bot-detection...\n")
        time.sleep(7)

        # Pro-Tip: Save to CSV every 10 days just in case your internet drops
        if len(results) % 10 == 0:
            pd.DataFrame(results).to_csv(f"{station}_oracle_backup.csv", index=False)

    # 5. Final Save
    df_final = pd.DataFrame(results)
    df_final.to_csv(f"{station}_FINAL_ORACLE_DATA.csv", index=False)
    print(f"🎉 EXTRACTION COMPLETE. Saved {len(df_final)} rows to CSV.")


# ==========================================
# --- RUN THE MASS MINER ---
# ==========================================
if __name__ == "__main__":
    # Change this to whatever city you are mining!
    # (RKSI = Incheon, LEMD = Madrid, LTAC = Ankara)
    target_station = "LIMC"

    # 1. Hardcode the start date
    start_date = "2021-03-31"

    # 2. Dynamically calculate "Today"
    # Using UTC to ensure we don't accidentally ask for a day that hasn't happened yet
    now_utc = datetime.now(pytz.utc)

    # Optional safety: If you run this late at night, Wunderground might not have
    # finalized today's data yet. We scrape up to YESTERDAY to be 100% safe.
    end_date = (now_utc - pd.Timedelta(days=1)).strftime('%Y-%m-%d')

    print(f"📅 Scrape Window: {start_date} to {end_date}")

    # 3. Launch the Fleet
    mine_historical_oracle(
        station=target_station,
        start_date_str=start_date,
        end_date_str=end_date
    )