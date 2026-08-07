import time
import requests
from datetime import datetime, timedelta

BASE_URL = "https://financialmodelingprep.com/stable"
CHUNK_DAYS = 80


def fetch_hourly(symbol: str, start: str, end: str, api_key: str) -> list[dict]:
    """Download hourly OHLCV from FMP in 80-day chunks. Returns flat list of bar dicts."""
    url = f"{BASE_URL}/historical-chart/1hour"

    start_dt = datetime.strptime(start, "%Y-%m-%d")
    end_dt = datetime.strptime(end, "%Y-%m-%d")

    records: list[dict] = []
    current = start_dt

    while current <= end_dt:
        chunk_end = min(current + timedelta(days=CHUNK_DAYS), end_dt)
        params = {
            "symbol": symbol,
            "from": current.strftime("%Y-%m-%d"),
            "to": chunk_end.strftime("%Y-%m-%d"),
            "apikey": api_key,
        }
        resp = requests.get(url, params=params, timeout=30)
        try:
            resp.raise_for_status()
        except requests.HTTPError as e:
            safe = resp.url.replace(api_key, "***")
            raise requests.HTTPError(f"{resp.status_code} for {safe}") from e
        chunk = resp.json()
        if not isinstance(chunk, list):
            raise ValueError(
                f"FMP returned unexpected response for {symbol} "
                f"({current.strftime('%Y-%m-%d')} → {chunk_end.strftime('%Y-%m-%d')}): {chunk}"
            )
        records.extend(chunk)
        current = chunk_end + timedelta(days=1)
        if current <= end_dt:
            time.sleep(0.3)

    return records
