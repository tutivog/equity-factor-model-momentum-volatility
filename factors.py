import pandas as pd
import numpy as np
from scipy.stats import zscore


def compute_momentum_12_1(
    prices: pd.DataFrame,
    daily_returns: pd.DataFrame,
    lookback_days: int = 252,
    skip_days: int = 21,
) -> pd.DataFrame:
    """
    Compute 12-1 momentum: cumulative return over past 252 days,
    skipping the most recent 21 days.

    Returns monthly factor values (aligned to month-end).
    """
    # Rolling window cumulative return from t-L to t-1
    rolling_ret = (1 + daily_returns).rolling(window=lookback_days).apply(
        lambda x: np.prod(x) - 1 if np.isfinite(x).all() else np.nan,
        raw=False,
    )

    # Shift forward to skip most recent month
    mom = rolling_ret.shift(skip_days)

    # Convert to month-end series
    mom_m = mom.resample("M").last()
    return mom_m


def compute_volatility_3m(
    daily_returns: pd.DataFrame,
    vol_window: int = 63,
) -> pd.DataFrame:
    """
    Compute 3-month volatility: rolling standard deviation of daily returns.

    Returns monthly factor values (aligned to month-end).
    """
    vol = daily_returns.rolling(window=vol_window).std()
    vol_m = vol.resample("M").last()
    return vol_m


def standardize_factors_cross_sectionally(
    factor_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    For each date, compute cross-sectional z-score of factor values across tickers.
    """
    def _zscore_row(row):
        return pd.Series(zscore(row.dropna()), index=row.dropna().index)

    z = factor_df.apply(_zscore_row, axis=1)
    return z


def build_combined_factor(
    momentum_z: pd.DataFrame,
    vol_z: pd.DataFrame,
    weight_mom: float = 0.5,
    weight_vol: float = 0.5,
) -> pd.DataFrame:
    """
    Combined factor score: higher is better.

    We want:
    - high momentum
    - low volatility  => subtract volatility z-score

    combined = w_mom * momentum_z - w_vol * vol_z
    """
    aligned_mom, aligned_vol = momentum_z.align(vol_z, join="inner")
    combined = weight_mom * aligned_mom - weight_vol * aligned_vol
    return combined
