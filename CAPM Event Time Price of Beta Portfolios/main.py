import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
import os
import pickle
import matplotlib.pyplot as plt
from scipy.stats import kurtosis, skew
from scipy.stats import ttest_ind
from sklearn.metrics import r2_score
import pandas as pd
import statsmodels.api as sm
import sys
from pathlib import Path
from tqdm import tqdm

#Add the parent directory to sys.path to allow for package imports
parent_dir = str(Path(__file__).resolve().parent.parent)
sys.path.append(parent_dir)

from Helper_Functions.total_market_trades import *
from Helper_Functions.advanced_fetch_stock_data import advanced_fetch_stock_data
from Helper_Functions.calculate_monthly_returns import calculate_monthly_returns
from Helper_Functions.better_calculate_monthly_returns import better_calculate_monthly_returns

#for beta, y is stock return (dependant) and x is market return (independant)
def calculate_beta(y, x):
    y = np.array(y)
    x = np.array(x)
    covariance = np.cov(y, x)[0, 1]
    variance = np.var(x)
    beta = covariance / variance
    return beta

def calculate_beta_force(y, x):
    beta = sm.OLS(y, x).fit().params[0]
    return beta

def calculate_betas_and_portfolio_returns(monthly_returns, spr_returns):
    unique_sequences = monthly_returns['sequence #'].unique()
    monthly_returns['market_cap'].ffill(inplace=True)
    monthly_returns['market_cap'].bfill(inplace=True)
    monthly_returns['market_cap'].fillna(1, inplace=True)
    monthly_returns['equity_returns'] = monthly_returns['equity_returns'].apply(lambda x: np.exp(x)-1).fillna(0)
    monthly_returns['sp500_return'] = monthly_returns['sp500_return'].apply(lambda x: np.exp(x)-1).fillna(0)
    spr_returns['sp500_return_range1'] = spr_returns['sp500_return_range1'].apply(lambda x: np.exp(x)-1).fillna(0)
    spr_returns['sp500_return_range2'] = spr_returns['sp500_return_range2'].apply(lambda x: np.exp(x)-1).fillna(0)
    unique_sequences.sort()
    num_groups = 25
    price_of_beta1, price_of_beta2 = [], []
    for seq in tqdm(unique_sequences):
        if seq >= 60:
            seq_range = range(seq-60, seq+1)
            data_subset = monthly_returns[monthly_returns['sequence #'].isin(seq_range)]

            data_range1 = data_subset[data_subset['type'] == 1]
            data_range2 = data_subset[data_subset['type'] == 2]
            # Sort by 'permco'
            sorted_data1 = data_range1.sort_values(by=['permco', 'sequence #']).groupby('permco').filter(lambda x: len(x) == 61)
            sorted_data2 = data_range2.sort_values(by=['permco', 'sequence #']).groupby('permco').filter(lambda x: len(x) == 61)

            # Group by 'permco' again and calculate the beta for each group
            betas1 = sorted_data1.groupby('permco').apply(lambda x: calculate_beta_force(x['equity_returns'][0:60], x['sp500_return'][0:60]))
            returns1 = sorted_data1.groupby('permco').apply(lambda x: np.log((1+x['equity_returns'][60:]).prod()))
            returns1.fillna(0, inplace=True)
            betas2 = sorted_data2.groupby('permco').apply(lambda x: calculate_beta_force(x['equity_returns'][0:60], x['sp500_return'][0:60]))
            returns2 = sorted_data2.groupby('permco').apply(lambda x: np.log((1+x['equity_returns'][60:]).prod()))
            returns2.fillna(0, inplace=True)
            market_caps1 = sorted_data1.groupby('permco').apply(lambda x: np.mean(x['market_cap'][60:61]))
            adjusted_caps1 = market_caps1 / np.sum(market_caps1)
            market_caps2 = sorted_data2.groupby('permco').apply(lambda x: np.mean(x['market_cap'][60:61]))
            adjusted_caps2 = market_caps2 / np.sum(market_caps2)

            # Filter out items where beta is 0

            # Make 10 groups based on 10 betas
            betas1_grouped = pd.qcut(betas1, num_groups, labels=False)
            betas2_grouped = pd.qcut(betas2, num_groups, labels=False)

            group1returns = []
            group1betas = []
            group2returns = []
            group2betas = []

            for i in range(num_groups):
                group1_indices = betas1_grouped[betas1_grouped == i].index
                group2_indices = betas2_grouped[betas2_grouped == i].index
                group1returns.append(np.average(returns1[group1_indices], weights=adjusted_caps1[group1_indices]))
                group2returns.append(np.average(returns2[group2_indices], weights=adjusted_caps2[group2_indices]))
                group1betas.append(np.average(betas1[group1_indices], weights=adjusted_caps1[group1_indices]))
                group2betas.append(np.average(betas2[group2_indices], weights=adjusted_caps2[group2_indices]))
            
            price_of_beta1.append(r2_score(group1returns, group1betas))
            price_of_beta2.append(r2_score(group2returns, group2betas))

    _, bins , _ = plt.hist(price_of_beta1, bins=20,  histtype=u'step')
    plt.hist(price_of_beta2, bins=bins, histtype=u'step')
    plt.legend(['Calendar Months', 'Event Months'])
    plt.show()


print("running")

window_size = 2520
start_year = 1990
end_year = 2019
n_stocks = 500
start_date = f'{start_year}-01-01'
end_date = f'{end_year}-12-31'

# Rerun flag
rerunMonthlyReturns = False

# Directory for storing pickle files
data_directory = 'data'
os.makedirs(data_directory, exist_ok=True)

# Construct file paths
monthly_returns_filename = os.path.join(data_directory, f'{window_size}_{n_stocks}_{start_year}_{end_year}_monthly_returns.pickle')
spr_returns_filename = os.path.join(data_directory, f'{window_size}_{n_stocks}_{start_year}_{end_year}_spr_returns.pickle')

risk_free = pd.read_csv('../Useful Data/rf daily rate.csv')
risk_free = risk_free[(risk_free['date'] >= start_date) & (risk_free['date'] <= end_date)]
market_returns = pd.read_csv('../Useful Data/value_weighted_return.csv')
market_returns = market_returns[(market_returns['date'] >= start_date) & (market_returns['date'] <= end_date)]
market_returns = market_returns.merge(risk_free, left_on='date', right_on='date')
market_returns['ret'] = market_returns['vwretd'] - market_returns['rf']

try:
    if rerunMonthlyReturns or not (os.path.exists(monthly_returns_filename) and os.path.exists(spr_returns_filename)):
        print("re-running everything")

        event_month_ranges, monthly_day_ranges = get_event_month_blocks(window_size, start_year, end_year)
        print("ranges calculated")

        stocks = advanced_fetch_stock_data(start_year, end_year, n_stocks)
        print("stocks fetched")

        monthly_returns, spr_returns = calculate_monthly_returns(stocks, market_returns, risk_free, monthly_day_ranges, event_month_ranges)
        print("monthly returns calculated")

        with open(monthly_returns_filename, 'wb') as f:
            pickle.dump(monthly_returns, f)
        with open(spr_returns_filename, 'wb') as f:
            pickle.dump(spr_returns, f)
    else:
        # Load existing data
        with open(monthly_returns_filename, 'rb') as f:
            monthly_returns = pickle.load(f)
        with open(spr_returns_filename, 'rb') as f:
            spr_returns = pickle.load(f)
        print("Data loaded from existing files")

    print(monthly_returns[monthly_returns['sequence #'] == 60])
    
    calculate_betas_and_portfolio_returns(monthly_returns, spr_returns)

except Exception as e:
    print(f"An error occurred: {e}")