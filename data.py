
---

## 4. `src/data.py`

```python
import pandas as pd
import yfinance as yf
from typing import List, Tuple


def download_price_data(
    tickers: List[str],
    start: str = "2015-01-01",
    end: str = "2024-12-31",
) -> pd.DataFrame:
    """
    Download daily adjusted close prices for a list of tickers.

    Returns
    -------
    prices : pd.DataFrame
        Date index, columns = tickers, values = Adj Close.
    """
    data = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(data, pd.DataFrame) and "Adj Close" in data.columns:
        prices = data["Adj Close"]
    else:
        # If yfinance returns a simple single-ticker dataframe
        prices = data

    prices = prices.dropna(how="all")
    prices = prices.sort_index()
    return prices


def compute_returns(prices: pd.DataFrame, freq: str = "D") -> pd.DataFrame:
    """
    Compute simple returns from price series.

    Parameters
    ----------
    prices : pd.DataFrame
        Price data with date index and ticker columns.
    freq : str
        'D' for daily (default). Kept for future extension.

    Returns
    -------
    returns : pd.DataFrame
        Simple returns.
    """
    rets = prices.pct_change().dropna(how="all")
    return rets


def resample_month_end(prices: pd.DataFrame, returns: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Resample prices and returns to month-end frequency.

    Returns
    -------
    m_prices : pd.DataFrame
    m_returns : pd.DataFrame
    """
    m_prices = prices.resample("M").last()
    m_returns = returns.resample("M").apply(lambda x: (1 + x).prod() - 1)
    return m_prices, m_returns
