import time
import requests
from datetime import datetime, timedelta
from urllib.parse import quote

BASE_URL = "https://financialmodelingprep.com/api/v3"
CHUNK_DAYS = 365


def fetch_hourly(symbol: str, start: str, end: str, api_key: str) -> list[dict]:
    """Download hourly OHLCV from FMP in year-sized chunks. Returns flat list of bar dicts."""
    symbol_encoded = quote(symbol, safe="")
    url = f"{BASE_URL}/historical-chart/1hour/{symbol_encoded}"

    start_dt = datetime.strptime(start, "%Y-%m-%d")
    end_dt = datetime.strptime(end, "%Y-%m-%d")

    records: list[dict] = []
    current = start_dt

    while current <= end_dt:
        chunk_end = min(current + timedelta(days=CHUNK_DAYS), end_dt)
        params = {
            "from": current.strftime("%Y-%m-%d"),
            "to": chunk_end.strftime("%Y-%m-%d"),
            "apikey": api_key,
        }
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        chunk = resp.json()
        if isinstance(chunk, list):
            records.extend(chunk)
        current = chunk_end + timedelta(days=1)
        time.sleep(0.3)

    return records
