'''
Script uses Selenium to scrap historical data of max temprature from the
wundergrounds website for a specific airport sensor
'''

#Import statement
import time
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

def scrape_wunderground_history(station, date_str):
    '''

    :param station: the airport station
    :param date_str: the histori date
    :return: the clean tempratures of that date
    '''

    print(f"INITIATING GHOST BROWSER FOR: {station} on {date_str}")
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Runs invisible in the background
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    # Fake a human User-Agent so IBM doesn't block us
    chrome_options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36")

    # Start the browser
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)

    # 2. Construct the exact URL
    # Format: YYYY-M-D (e.g., 2023-5-14)
    url = f"https://www.wunderground.com/history/daily/{station}/date/{date_str}"
    print(f"🔗 Target URL: {url}")

    try:
        driver.get(url)

        print("⏳ Waiting for IBM JavaScript to render the Observation Table...")
        # Wait specifically for the giant data table at the bottom
        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "lib-city-history-observation"))
        )
        time.sleep(2)  # Buffer for the table data to physically populate

        html = driver.page_source
        soup = BeautifulSoup(html, 'html.parser')

        # Find the Observation Section
        obs_section = soup.find('lib-city-history-observation')
        if not obs_section:
            print("❌ WARNING: Could not locate the Observation Table.")
            return None

        # Grab all the rows in that table
        rows = obs_section.find_all('tr')
        daily_temps = []

        # Loop through every single hour of the day
        for row in rows[1:]:  # Skip the header row
            cols = row.find_all('td')
            if len(cols) >= 2:
                # Column 1 is usually the Temperature (Column 0 is Time)
                temp_string = cols[1].text.strip()

                # Extract just the numbers (e.g., turn "85 F" into 85.0)
                clean_string = ''.join(c for c in temp_string if c.isdigit() or c == '.')

                if clean_string:
                    daily_temps.append(float(clean_string))

        if not daily_temps:
            print("❌ WARNING: Table was found, but no temperatures could be extracted.")
            return None

        # Mathematically find the peak temperature of the day (in Fahrenheit)
        wunderground_max_f = max(daily_temps)

        # ==========================================
        # INSTITUTIONAL SANITIZATION (F -> C)
        # ==========================================
        # Convert to Celsius and round to 2 decimal places to match our METAR pipeline
        raw_celsius = (wunderground_max_f - 32) * (5.0 / 9.0)

        # 2. The IBM UI Emulator (Round to nearest whole number)
        wunderground_max_c = int(round(raw_celsius))

        print(f"🇺🇸 Raw IBM Fahrenheit: {wunderground_max_f}°F")
        print(f"🔬 True Metric Math: {raw_celsius:.2f}°C")
        print(f"✅ FINALIZED ORACLE HIGH TEMP: {wunderground_max_c}°C")


        return wunderground_max_c

    except Exception as e:
        print(f"🚨 SCRAPE FAILED: {e}")
        return None
    finally:
        # ALWAYS kill the browser process, otherwise your RAM will fill up and crash your computer
        driver.quit()


from datetime import datetime
import pytz

# --- RUN THE MASS MINER ---
if __name__ == "__main__":
    # Change this to whatever city you are mining!
    # (RKSI = Incheon, LEMD = Madrid, LTAC = Ankara)
    target_station = "LEMD"

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
