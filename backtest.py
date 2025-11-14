import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict


def compute_performance_stats(returns: pd.Series, trading_days: int = 252) -> Dict[str, float]:
    """
    Simple performance stats: CAGR, vol, Sharpe, max drawdown.
    """
    rets = returns.dropna()
    if rets.empty:
        return {"cagr": float("nan"), "vol": float("nan"), "sharpe": float("nan"), "max_drawdown": float("nan")}

    cum_ret = (1 + rets).prod()
    n_years = len(rets) / trading_days
    cagr = cum_ret ** (1 / n_years) - 1

    vol = rets.std() * (trading_days ** 0.5)
    sharpe = cagr / vol if vol != 0 else float("nan")

    equity = (1 + rets).cumprod()
    running_max = equity.cummax()
    drawdown = equity / running_max - 1
    max_dd = drawdown.min()

    return {"cagr": cagr, "vol": vol, "sharpe": sharpe, "max_drawdown": max_dd}


def plot_equity_curve(returns_df: pd.DataFrame, out_path: str = "output/equity_curve.png") -> None:
    """
    Plot cumulative return curves for long, short, and long-short portfolios.
    """
    (1 + returns_df).cumprod().plot(figsize=(10, 6))
    plt.title("Equity Curves – Long, Short, Long-Short")
    plt.ylabel("Cumulative Return (Growth of $1)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
