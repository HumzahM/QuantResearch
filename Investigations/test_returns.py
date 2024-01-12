from pathlib import Path
import os
import sys
import numpy as np


parent_dir = str(Path(__file__).resolve().parent.parent)
sys.path.append(parent_dir)

from Helper_Functions.total_market_trades import get_event_month_blocks
from Helper_Functions.advanced_fetch_stock_data import advanced_fetch_stock_data
from Helper_Functions.calculate_monthly_returns import calculate_monthly_returns

stocks = advanced_fetch_stock_data(2000, 2019, 1)
event_months, calendar_months = get_event_month_blocks(2520, 2000, 2019)
monthly_returns, sp500_returns = calculate_monthly_returns(stocks, calendar_months, event_months, calculate_sp=True)
sp500_returns['sp500_return_range1'] = sp500_returns['sp500_return_range1'].apply(lambda x: np.exp(x)-1)
sp500_returns['sp500_return_range2'] = sp500_returns['sp500_return_range2'].apply(lambda x: np.exp(x)-1)
monthly_returns['equity_returns'] = monthly_returns['equity_returns'].apply(lambda x: np.exp(x)-1)
monthly_returns['sp500_return'] = monthly_returns['sp500_return'].apply(lambda x: np.exp(x)-1)
monthly_returns.sort_values(by=['sequence #', 'permco' ], inplace=True)
print(monthly_returns)
print(sp500_returns)

