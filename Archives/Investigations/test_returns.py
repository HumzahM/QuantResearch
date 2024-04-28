from pathlib import Path
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

parent_dir = str(Path(__file__).resolve().parent.parent)
sys.path.append(parent_dir)

from Helper_Functions.total_market_trades import get_event_month_blocks
from Helper_Functions.advanced_fetch_stock_data import advanced_fetch_stock_data
from Helper_Functions.calculate_monthly_returns import calculate_monthly_returns

import pickle

window_size = 2520
start_year = 1990
end_year = 2019
n_stocks = 500
start_date = f'{start_year}-01-01'
end_date = f'{end_year}-12-31'

# Directory for storing pickle files
data_directory = 'data'
os.makedirs(data_directory, exist_ok=True)

monthly_returns_filename = os.path.join(data_directory, f'{window_size}_{n_stocks}_{start_year}_{end_year}_monthly_returns.pickle')
spr_returns_filename = os.path.join(data_directory, f'{window_size}_{n_stocks}_{start_year}_{end_year}_spr_returns.pickle')
daily_returns_filename = '500_1990_2019_data.pickle'

with open(monthly_returns_filename, 'rb') as f:
    monthly_returns = pickle.load(f)

with open(daily_returns_filename, 'rb') as f:
    daily_returns = pickle.load(f)['data']

monthly_returns['equity_returns'] = monthly_returns['equity_returns'].apply(lambda x: np.exp(x)-1).fillna(0)
monthly_returns['sp500_return'] = monthly_returns['sp500_return'].apply(lambda x: np.exp(x)-1).fillna(0)

test_beta = monthly_returns[(monthly_returns['permco'] == 4388) & (monthly_returns['type'] == 2)].reset_index(drop=True).iloc[0:60]

bad_returns = monthly_returns[(monthly_returns['permco'] == 4388) & (monthly_returns['type'] == 2)].reset_index(drop=True).iloc[60:72]

def calculate_beta(y, x):
    y = np.array(y)
    x = np.array(x)
    covariance = np.cov(y, x)[0, 1]
    variance = np.var(x)
    beta = covariance / variance
    return beta

# print(calculate_beta(test_beta['equity_returns'], test_beta['sp500_return']))
# print(bad_returns)

# print((1+bad_returns['equity_returns']).prod())

relevant_returns = daily_returns[(daily_returns['permco'] == 4388)]
relevant_returns['ret'] = 0
relevant_returns['market_cap'] = 0

print(len(relevant_returns))
print(len(relevant_returns.drop_duplicates()))

print((1+relevant_returns['ret']).prod())
