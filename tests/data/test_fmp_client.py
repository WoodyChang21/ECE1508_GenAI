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


def test_single_year_makes_one_api_call():
    with patch("scripts.data.fmp_client.requests.get") as mock_get:
        mock_get.return_value = _mock_get([_bar("2024-01-02 10:00:00")])
        result = fetch_hourly("SPY", "2024-01-01", "2024-12-31", "test_key")
    assert mock_get.call_count == 1
    assert len(result) == 1


def test_three_year_range_makes_three_api_calls():
    with patch("scripts.data.fmp_client.requests.get") as mock_get:
        mock_get.return_value = _mock_get([_bar("2022-06-15 10:00:00")])
        fetch_hourly("SPY", "2022-01-01", "2024-12-31", "test_key")
    assert mock_get.call_count == 3


def test_results_are_flat_list_of_dicts():
    with patch("scripts.data.fmp_client.requests.get") as mock_get:
        mock_get.return_value = _mock_get([_bar("2024-01-02 10:00:00"), _bar("2024-01-02 09:30:00")])
        result = fetch_hourly("SPY", "2024-01-01", "2024-12-31", "test_key")
    assert isinstance(result, list)
    assert len(result) == 2
    assert all("date" in r and "close" in r for r in result)


def test_api_key_passed_in_query_params():
    with patch("scripts.data.fmp_client.requests.get") as mock_get:
        mock_get.return_value = _mock_get([])
        fetch_hourly("SPY", "2024-01-01", "2024-12-31", "my_secret_key")
    call_kwargs = mock_get.call_args
    params = mock_get.call_args.kwargs["params"]
    assert params["apikey"] == "my_secret_key"


def test_caret_symbol_is_url_encoded():
    """^VIX must be percent-encoded to %5EVIX in the URL path."""
    with patch("scripts.data.fmp_client.requests.get") as mock_get:
        mock_get.return_value = _mock_get([])
        fetch_hourly("^VIX", "2024-01-01", "2024-12-31", "key")
    url_called = mock_get.call_args[0][0]
    assert "%5EVIX" in url_called or "%5evix" in url_called.lower()


def test_raises_on_non_list_response():
    """FMP returns a dict (e.g. error body) when the API key is invalid or quota exceeded."""
    import pytest
    error_body = {"Error Message": "Invalid API KEY. Please retry or visit our documentation..."}
    with patch("scripts.data.fmp_client.requests.get") as mock_get:
        mock_get.return_value = _mock_get(error_body)
        with pytest.raises(ValueError, match="FMP returned unexpected response"):
            fetch_hourly("SPY", "2024-01-01", "2024-12-31", "bad_key")
