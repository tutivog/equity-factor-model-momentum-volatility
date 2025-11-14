# Simple Equity Factor Model (Momentum + Low Volatility)

This project implements a basic cross-sectional equity **factor model** in Python and backtests
a **monthly-rebalanced long–short portfolio** using:

- **12-1 momentum**: performance over the past 12 months excluding the most recent month
- **3-month volatility**: standard deviation of daily returns over the past 3 months
- Combined factor = z-score(momentum) − z-score(volatility)

Each month, the strategy:
- Goes **long** the top 20% of stocks by combined factor
- Goes **short** the bottom 20%
- Equal-weights stocks in each side
- Rebalances monthly

> ⚠️ This is a **toy research model** for educational purposes only, not investment advice.

## How it works

1. Download daily price data for a universe of large-cap US equities (e.g., S&P 100) using `yfinance`.
2. Compute daily returns for each ticker.
3. Compute factor values:
   - Momentum: cumulative return from t-252 to t-21
   - Volatility: rolling 63-day standard deviation of daily returns
4. Convert factor values to **cross-sectional z-scores** each month.
5. Build long/short portfolios and compute portfolio return time series.
6. Plot cumulative returns and save outputs to `output/`.

## Installation

```bash
git clone <your-repo-url>.git
cd factor-model-demo
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
