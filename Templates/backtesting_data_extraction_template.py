"""
Polymarket City-Temperature Backtesting Data Downloader (Predexon)
=====================================================================
Downloads historical orderbook snapshots for the daily "Highest temperature
in <city>" Polymarket markets — one distinct market per day — sampled every
SAMPLE_INTERVAL minutes from 10:00 to 17:00 in the city's local timezone.

City-specific values (timezone, city slug, output filename) come from a
YAML config file passed via --config.

Pipeline per target date:
    1. Discover that day's markets via /v2/polymarket/markets (filter by slug)
    2. For each market, extract its outcome tokens (YES / NO bands)
    3. Fetch the orderbook range once per token via /v2/polymarket/orderbooks
       (start_time=10:00 local, end_time=17:00 local, both in ms UTC)
    4. Resample — for each tick, take the snapshot closest to the target time
    5. Extract best ask and best bid
    6. Assemble a long-format DataFrame → CSV

Usage:
    export PREDEXON_API_KEY="your_key"
    python backtesting_data_extraction.py --config milan.yaml --probe-only
    python backtesting_data_extraction.py --config milan.yaml
"""

import os
import sys
import time
import logging
import argparse
import yaml
from datetime import datetime, timedelta, date, timezone
from zoneinfo import ZoneInfo

import requests
import pandas as pd
from tqdm import tqdm

# ══════════════════════════════════════════════
# CONFIG — loaded in main()
# ══════════════════════════════════════════════
API_KEY: str = ""
BASE_URL = "https://api.predexon.com"

TZ: ZoneInfo | None = None
TZ_NAME: str = ""
UTC = timezone.utc

WINDOW_START_H  = 10
WINDOW_END_H    = 17
SAMPLE_INTERVAL = 10

LOOKBACK_DAYS = 60
EARLIEST_DATA = date(2026, 1, 1)

OUTPUT_CSV = "city_temperature_orderbooks.csv"
CITY_SLUG: str = ""
TEMP_RANGE = range(0, 46)

REQUEST_DELAY      = 1.1
MAX_RETRIES        = 4
MAX_SNAPSHOT_LIMIT = 200

# ══════════════════════════════════════════════
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

SESSION = requests.Session()


# ══════════════════════════════════════════════
# HTTP HELPER
# ══════════════════════════════════════════════
def _request(method: str, path: str, **kwargs) -> dict | None:
    """Unified request helper with retries + backoff."""
    url = f"{BASE_URL}{path}"
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = SESSION.request(method, url, timeout=30, **kwargs)
            if r.status_code == 404:
                return None
            if r.status_code == 429:
                wait = min(2 ** attempt, 30)
                log.warning("Rate limited. Sleeping %ds", wait)
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r.json()
        except requests.HTTPError as e:
            log.warning("HTTP %s attempt %d: %s", r.status_code, attempt, e)
            if 400 <= r.status_code < 500 and r.status_code not in (408, 429):
                try:
                    log.warning("Body: %s", r.text[:300])
                except Exception:
                    pass
                return None
        except requests.RequestException as e:
            log.warning("Network error attempt %d: %s", attempt, e)
        time.sleep(min(2 ** attempt, 15))
    return None


# ══════════════════════════════════════════════
# 1. DAILY MARKET DISCOVERY
# ══════════════════════════════════════════════
def _fetch_by_slug(slug: str) -> dict | None:
    """Fetch a single market by exact slug. Returns the market dict or None."""
    data = _request("GET", "/v2/polymarket/markets", params={"market_slug": slug})
    if not data:
        return None
    markets = data.get("markets", [])
    return markets[0] if markets else None


def find_markets_for_date(target: date) -> list[dict]:
    """
    Build the exact slug for each integer °C in TEMP_RANGE and look it up:
        highest-temperature-in-<CITY_SLUG>-on-{month}-{day}-{year}-{temp}c
    e.g. highest-temperature-in-milan-on-april-19-2026-27c
    """
    months_full = ["january", "february", "march", "april", "may", "june",
                   "july", "august", "september", "october", "november", "december"]
    month_str = months_full[target.month - 1]
    day_str   = str(target.day)
    year_str  = str(target.year)

    matched     = []
    miss_streak = 0
    found_any   = False

    for temp in TEMP_RANGE:
        slug = (
            f"highest-temperature-in-{CITY_SLUG}-on-"
            f"{month_str}-{day_str}-{year_str}-{temp}c"
        )
        m = _fetch_by_slug(slug)
        time.sleep(REQUEST_DELAY)

        if m:
            matched.append(m)
            found_any   = True
            miss_streak = 0
        else:
            if found_any:
                miss_streak += 1
                if miss_streak >= 5:
                    break

    log.info("Date %s: %d temperature market(s) found",
             target.isoformat(), len(matched))
    return matched


def extract_temperature_from_slug(slug: str) -> str | None:
    """Pull the temperature value from slugs like ...-27c → '27'."""
    import re
    m = re.search(r"-(\d+)c$", slug.lower())
    return m.group(1) if m else None


def extract_outcome_tokens(market: dict) -> list[dict]:
    """Return a market's outcome tokens as [{token_id, label}, ...]."""
    tokens = []
    for outcome in market.get("outcomes", []) or []:
        tid = outcome.get("token_id")
        if not tid:
            continue
        tokens.append({
            "token_id": str(tid),
            "label":    outcome.get("label", ""),
        })
    return tokens


# ══════════════════════════════════════════════
# 2. ORDERBOOK RANGE FETCH
# ══════════════════════════════════════════════
def fetch_orderbook_range(token_id: str, start_ms: int, end_ms: int) -> list[dict]:
    """Fetch all orderbook snapshots for token_id between start_ms and end_ms."""
    snapshots: list[dict] = []
    pagination_key: str | None = None

    while True:
        params = {
            "token_id":   token_id,
            "start_time": start_ms,
            "end_time":   end_ms,
            "limit":      MAX_SNAPSHOT_LIMIT,
        }
        if pagination_key:
            params["pagination_key"] = pagination_key

        data = _request("GET", "/v2/polymarket/orderbooks", params=params)
        if not data:
            break

        batch = data.get("snapshots", [])
        snapshots.extend(batch)

        pg = data.get("pagination", {})
        if pg.get("has_more") and pg.get("pagination_key"):
            pagination_key = pg["pagination_key"]
            time.sleep(REQUEST_DELAY)
        else:
            break

    return snapshots


def best_ask(snapshot: dict) -> float | None:
    asks = snapshot.get("asks") or []
    if not asks:
        return None
    try:
        return min(float(a["price"]) for a in asks if "price" in a)
    except (ValueError, KeyError):
        return None


def best_bid(snapshot: dict) -> float | None:
    bids = snapshot.get("bids") or []
    if not bids:
        return None
    try:
        return max(float(b["price"]) for b in bids if "price" in b)
    except (ValueError, KeyError):
        return None


# ══════════════════════════════════════════════
# 3. RESAMPLING
# ══════════════════════════════════════════════
def sample_snapshots(snapshots: list[dict], target_day: date) -> list[dict]:
    """
    From raw snapshots, pick the one closest to each target tick
    (every SAMPLE_INTERVAL minutes from WINDOW_START_H to WINDOW_END_H local time).
    If the nearest snapshot is more than SAMPLE_INTERVAL/2 minutes off, the
    slot is left as None.
    """
    if not snapshots:
        return []

    tolerance_s = (SAMPLE_INTERVAL / 2) * 60

    parsed = []
    for s in snapshots:
        ts = s.get("timestamp")
        if ts is None:
            continue
        dt_utc = datetime.fromtimestamp(ts / 1000, tz=UTC)
        parsed.append((dt_utc, s))
    if not parsed:
        return []
    parsed.sort(key=lambda x: x[0])

    out = []
    hour, minute = WINDOW_START_H, 0
    while True:
        if hour > WINDOW_END_H:
            break
        if hour == WINDOW_END_H and minute > 0:
            break

        target_local = datetime(
            target_day.year, target_day.month, target_day.day,
            hour, minute, 0, tzinfo=TZ,
        )
        target_utc = target_local.astimezone(UTC)

        nearest = min(parsed, key=lambda x: abs((x[0] - target_utc).total_seconds()))
        dt_utc, snap = nearest

        if abs((dt_utc - target_utc).total_seconds()) > tolerance_s:
            out.append({"timestamp_local": target_local, "snapshot": None})
        else:
            out.append({"timestamp_local": target_local, "snapshot": snap})

        minute += SAMPLE_INTERVAL
        if minute >= 60:
            hour  += minute // 60
            minute = minute % 60

    return out


# ══════════════════════════════════════════════
# 4. ORCHESTRATION
# ══════════════════════════════════════════════
def date_range(lookback: int) -> list[date]:
    today = date.today()
    end   = today - timedelta(days=1)
    start = max(today - timedelta(days=lookback), EARLIEST_DATA)
    days = []
    d = start
    while d <= end:
        days.append(d)
        d += timedelta(days=1)
    return days


def run_one_day(target: date) -> list[dict]:
    """Process a single date end-to-end."""
    rows = []
    markets = find_markets_for_date(target)
    if not markets:
        log.warning("  → no market found for %s", target)
        return rows

    start_local = datetime(target.year, target.month, target.day,
                           WINDOW_START_H, 0, tzinfo=TZ)
    end_local   = datetime(target.year, target.month, target.day,
                           WINDOW_END_H,   0, tzinfo=TZ)
    start_ms = int(start_local.astimezone(UTC).timestamp() * 1000)
    end_ms   = int(end_local.astimezone(UTC).timestamp()   * 1000)

    for m in markets:
        slug   = m.get("market_slug", "")
        title  = m.get("title", "")
        temp_c = extract_temperature_from_slug(slug)

        for tok in extract_outcome_tokens(m):
            token_id = tok["token_id"]
            label    = tok["label"]

            snapshots = fetch_orderbook_range(token_id, start_ms, end_ms)
            hourly    = sample_snapshots(snapshots, target)

            for entry in hourly:
                snap = entry["snapshot"]
                rows.append({
                    "date":            target.isoformat(),
                    "timestamp_local": entry["timestamp_local"].isoformat(),
                    "hour":            entry["timestamp_local"].hour,
                    "market_slug":     slug,
                    "market_title":    title,
                    "temperature_c":   temp_c,
                    "token_id":        token_id,
                    "option_label":    label,
                    "best_ask":        best_ask(snap) if snap else None,
                    "best_bid":        best_bid(snap) if snap else None,
                    "n_snapshots_raw": len(snapshots),
                })
            time.sleep(REQUEST_DELAY)

    return rows


def run(output_csv: str, lookback: int) -> pd.DataFrame:
    days = date_range(lookback)
    log.info("Processing %d days: %s → %s", len(days), days[0], days[-1])

    all_rows = []
    for d in tqdm(days, desc="Days", unit="day"):
        all_rows.extend(run_one_day(d))

    df = pd.DataFrame(all_rows)
    df.to_csv(output_csv, index=False)
    log.info("Saved %d rows → %s", len(df), output_csv)
    return df


# ══════════════════════════════════════════════
# DIAGNOSTIC: one-day probe
# ══════════════════════════════════════════════
def probe(target: date | None = None) -> None:
    target = target or (date.today() - timedelta(days=1))
    print(f"\n=== PROBING {target} ===\n")

    print("── SLUG-BASED MARKET DISCOVERY ──")
    markets_raw = find_markets_for_date(target)
    print(f"  Found {len(markets_raw)} market(s) via direct slug lookup")
    for m in markets_raw:
        print(f"    {m.get('market_slug')} | {m.get('title')}")
    print()

    if markets_raw:
        m0 = markets_raw[0]
        tokens = extract_outcome_tokens(m0)
        print("── FIRST MARKET TOKEN PARSE ──")
        for t in tokens:
            print(f"  {t['label']:<5} id={t['token_id'][:24]}...")
        print()

        print("── ORDERBOOK SAMPLE ──")
        rows = run_one_day(target)
        df = pd.DataFrame(rows)
        print(f"Generated {len(df)} rows")
        if not df.empty:
            cols = ["timestamp_local", "temperature_c", "option_label",
                    "best_ask", "best_bid"]
            print(df[cols].to_string())
    else:
        print("!! No markets found — check that the slug pattern matches.")
        print(f"   Expected: highest-temperature-in-{CITY_SLUG}-on-"
              f"{{month}}-{{day}}-{{year}}-{{temp}}c")


# ══════════════════════════════════════════════
# CONFIG LOAD + MAIN
# ══════════════════════════════════════════════
def _load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _apply_config(cfg: dict) -> None:
    """Populate module-level config globals from the YAML dict."""
    global TZ, TZ_NAME, CITY_SLUG, TEMP_RANGE, OUTPUT_CSV
    TZ_NAME    = cfg["location"]["timezone"]
    TZ         = ZoneInfo(TZ_NAME)
    CITY_SLUG  = cfg["city"]["polymarket_slug"]
    lo, hi     = cfg.get("temp_range", [0, 46])
    TEMP_RANGE = range(int(lo), int(hi))
    suffix     = cfg["city"]["output_suffix"]
    OUTPUT_CSV = f"{suffix}_temperature_orderbooks.csv"


def main() -> None:
    global API_KEY

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", "-c", required=True,
                    help="Path to YAML config (e.g. milan.yaml).")
    ap.add_argument("--probe-only", action="store_true",
                    help="Inspect one day end-to-end instead of running the full download.")
    ap.add_argument("--date", default=None,
                    help="When used with --probe-only, the date to probe (YYYY-MM-DD).")
    ap.add_argument("--output", default=None,
                    help="Override the default output CSV path.")
    ap.add_argument("--lookback", type=int, default=LOOKBACK_DAYS,
                    help=f"How many days back to fetch (default: {LOOKBACK_DAYS}).")
    args = ap.parse_args()

    cfg = _load_config(args.config)
    _apply_config(cfg)

    # API key: env var only — never store in YAML or source.
    API_KEY = os.environ.get("PREDEXON_API_KEY", "")
    if not API_KEY:
        raise SystemExit(
            "PREDEXON_API_KEY environment variable is not set.\n"
            "Run:  export PREDEXON_API_KEY='your_key_here'"
        )
    SESSION.headers.update({
        "x-api-key":    API_KEY,
        "Content-Type": "application/json",
    })

    if args.probe_only:
        d = date.fromisoformat(args.date) if args.date else None
        probe(d)
        return

    out_csv = args.output or OUTPUT_CSV
    df = run(output_csv=out_csv, lookback=args.lookback)
    print(df.head(20).to_string())
    print(f"\nShape: {df.shape}")
    if not df.empty:
        print("\nPer-day coverage:")
        print(df.groupby("date").size().describe())


if __name__ == "__main__":
    main()
