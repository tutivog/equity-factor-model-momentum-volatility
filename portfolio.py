import pandas as pd
import numpy as np
from typing import Tuple


def build_long_short_portfolios(
    combined_factor: pd.DataFrame,
    daily_returns: pd.DataFrame,
    top_quantile: float = 0.2,
    bottom_quantile: float = 0.2,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Build long/short portfolios based on combined factor scores.

    Rebalanced monthly:
    - At each month-end, rank stocks by factor.
    - Long top X%, short bottom Y%, equal-weight.

    Returns
    -------
    long_ret : pd.Series
    short_ret : pd.Series
    ls_ret : pd.Series (long-short)
    """
    # Reindex factor to daily index using forward fill, so each day in a month
    # holds the weights decided at the previous month-end.
    factor_daily = combined_factor.reindex(daily_returns.index, method="ffill")

    long_ret_list = []
    short_ret_list = []

    for date, row in factor_daily.iterrows():
        scores = row.dropna()
        if scores.empty:
            long_ret_list.append(np.nan)
            short_ret_list.append(np.nan)
            continue

        # Determine thresholds
        q_top = scores.quantile(1 - top_quantile)
        q_bottom = scores.quantile(bottom_quantile)

        long_names = scores[scores >= q_top].index
        short_names = scores[scores <= q_bottom].index

        if len(long_names) == 0 or len(short_names) == 0:
            long_ret_list.append(np.nan)
            short_ret_list.append(np.nan)
            continue

        # Equal-weight portfolios
        day_ret = daily_returns.loc[date]

        long_ret = day_ret[long_names].mean()
        short_ret = day_ret[short_names].mean()

        long_ret_list.append(long_ret)
        short_ret_list.append(short_ret)

    long_ret = pd.Series(long_ret_list, index=daily_returns.index, name="long")
    short_ret = pd.Series(short_ret_list, index=daily_returns.index, name="short")
    ls_ret = long_ret - short_ret
    ls_ret.name = "long_short"

    return long_ret, short_ret, ls_ret
