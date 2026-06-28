from unittest.mock import patch, MagicMock
from scripts.data.fmp_client import fetch_hourly


def _bar(date_str):
    return {
        "date": date_str,
        "open": 470.0,
        "high": 471.0,
        "low": 469.0,
        "close": 470.5,
        "volume": 1_000_000,
    }


def _mock_get(data):
    m = MagicMock()
    m.json.return_value = data
    m.raise_for_status = MagicMock()
    return m


def test_single_chunk_for_short_range():
    """A range within CHUNK_DAYS makes exactly one API call."""
    with patch("scripts.data.fmp_client.requests.get") as mock_get:
        mock_get.return_value = _mock_get([_bar("2024-01-15 10:00:00")])
        result = fetch_hourly("SPY", "2024-01-01", "2024-03-21", "test_key")  # 80 days
    assert mock_get.call_count == 1
    assert len(result) == 1


def test_multiple_chunks_for_long_range():
    """A range longer than CHUNK_DAYS makes multiple API calls."""
    with patch("scripts.data.fmp_client.requests.get") as mock_get:
        mock_get.return_value = _mock_get([_bar("2024-04-01 10:00:00")])
        # 2024-01-01 to 2024-06-10 = 161 days > 80 → 2 chunks
        fetch_hourly("SPY", "2024-01-01", "2024-06-10", "test_key")
    assert mock_get.call_count == 2


def test_results_are_flat_list_of_dicts():
    with patch("scripts.data.fmp_client.requests.get") as mock_get:
        mock_get.return_value = _mock_get([_bar("2024-01-02 10:00:00"), _bar("2024-01-02 09:30:00")])
        result = fetch_hourly("SPY", "2024-01-01", "2024-03-21", "test_key")
    assert isinstance(result, list)
    assert len(result) == 2
    assert all("date" in r and "close" in r for r in result)


def test_api_key_passed_in_query_params():
    with patch("scripts.data.fmp_client.requests.get") as mock_get:
        mock_get.return_value = _mock_get([])
        fetch_hourly("SPY", "2024-01-01", "2024-03-21", "my_secret_key")
    params = mock_get.call_args.kwargs["params"]
    assert params["apikey"] == "my_secret_key"


def test_symbol_passed_in_query_params():
    """Symbol must appear in query params (not embedded in URL path)."""
    with patch("scripts.data.fmp_client.requests.get") as mock_get:
        mock_get.return_value = _mock_get([])
        fetch_hourly("^VIX", "2024-01-01", "2024-12-31", "key")
    params = mock_get.call_args.kwargs["params"]
    assert params["symbol"] == "^VIX"


def test_raises_on_non_list_response():
    """FMP returns a dict (e.g. error body) when the API key is invalid or quota exceeded."""
    import pytest
    error_body = {"Error Message": "Invalid API KEY. Please retry or visit our documentation..."}
    with patch("scripts.data.fmp_client.requests.get") as mock_get:
        mock_get.return_value = _mock_get(error_body)
        with pytest.raises(ValueError, match="FMP returned unexpected response"):
            fetch_hourly("SPY", "2024-01-01", "2024-03-21", "bad_key")
