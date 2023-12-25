import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
import os
import pickle
import matplotlib.pyplot as plt
from scipy.stats import kurtosis, skew
from scipy.stats import ttest_ind
import pandas as pd
import statsmodels.api as sm
import sys
from pathlib import Path

#Add the parent directory to sys.path to allow for package imports
parent_dir = str(Path(__file__).resolve().parent.parent)
sys.path.append(parent_dir)

from Helper_Functions.total_market_trades import *
from Helper_Functions.advanced_fetch_stock_data import advanced_fetch_stock_data
from Helper_Functions.calculate_monthly_returns import calculate_monthly_returns

start_date = '1990-01-01'
end_date = '2019-12-31'

risk_free = pd.read_csv('../Useful Data/rf daily rate.csv')
risk_free = risk_free[(risk_free['date'] >= start_date) & (risk_free['date'] <= end_date)]
market_returns = pd.read_csv('../Useful Data/sp500_return_data.csv')
market_returns = market_returns[(market_returns['date'] >= start_date) & (market_returns['date'] <= end_date)]
market_returns = market_returns.merge(risk_free, left_on='date', right_on='date')
market_returns['log sp500 returns'] = np.log(1+market_returns['sprtrn'] - market_returns['rf'])


def calculate_beta(y, x):
    y = np.array(y)
    x = np.array(x)
    covariance = np.cov(y, x)[0, 1]
    variance = np.var(x)
    beta = covariance / variance
    return beta

def calculate_beta_force(y, x):
    return sm.OLS(y, x).fit().params[0]

def calculate_betas_and_portfolio_returns(monthly_returns, spr_returns):
    unique_sequences = monthly_returns['sequence #'].unique()
    unique_sequences.sort()
    num_groups = 10
    prices_of_beta1 = []
    prices_of_beta2 = []
    for seq in unique_sequences:
        if(seq >= 72 and seq % 12 == 0):
            print(seq)
            seq_range = range(seq-71, seq+1)
            data_subset = monthly_returns[monthly_returns['sequence #'].isin(seq_range)]
            data_range1 = data_subset[data_subset['type'] == 1]
            data_range2 = data_subset[data_subset['type'] == 2]
            # Sort by 'permco'
            sorted_data1 = data_range1.sort_values(by=['permco', 'sequence #']).groupby('permco').filter(lambda x: len(x) == 72)
            sorted_data2 = data_range2.sort_values(by=['permco', 'sequence #']).groupby('permco').filter(lambda x: len(x) == 72)

            #valid_data = sorted_data.groupby('permco').filter(lambda x: len(x) == 72)
            # Group by 'permco' again and calculate the beta for each group
            betas1 = sorted_data1.groupby('permco').apply(lambda x: calculate_beta_force(x['equity_returns'][0:60], x['sp500_return'][0:60]))
            returns1 = sorted_data1.groupby('permco').apply(lambda x: np.sum(x['equity_returns'][60:]))
            betas2 = sorted_data2.groupby('permco').apply(lambda x: calculate_beta_force(x['equity_returns'][0:60], x['sp500_return'][0:60]))
            returns2 = sorted_data2.groupby('permco').apply(lambda x: np.sum(x['equity_returns'][60:]))
            market_caps1 = sorted_data1.groupby('permco').apply(lambda x: np.mean(x['market_cap'][60:]))
            market_caps1.fillna(0, inplace=True)
            market_caps2 = sorted_data2.groupby('permco').apply(lambda x: np.mean(x['market_cap'][60:]))
            market_caps2.fillna(0, inplace=True)

            sp_returns1 = sorted_data1.groupby('permco').apply(lambda x: np.mean(x['sp500_return'][60:]))
            sp_returns2 = sorted_data2.groupby('permco').apply(lambda x: np.mean(x['sp500_return'][60:]))
            # Make 10 groups based on 10 betas

            betas1_grouped = pd.qcut(betas1, num_groups, labels=False)
            betas2_grouped = pd.qcut(betas2, num_groups, labels=False)

            means1 = []
            means2 = []
            betas1 = []
            betas2 = []

            for i in range(num_groups):
                group1_indices = betas1_grouped[betas1_grouped == i].index
                group2_indices = betas2_grouped[betas2_grouped == i].index
                    
                means1.append(np.average(returns1[group1_indices], weights=market_caps1[group1_indices]))
                means2.append(np.average(returns2[group2_indices], weights=market_caps2[group2_indices]))

                betas1.append(calculate_beta_force(returns1[group1_indices], sp_returns1[group1_indices]))
                betas2.append(calculate_beta_force(returns2[group2_indices], sp_returns2[group2_indices]))

            prices_of_beta1.append(calculate_beta(means1, betas1))
            prices_of_beta2.append(calculate_beta(means2, betas2))
            
    plt.figure()
    _, bins, _ = plt.hist(prices_of_beta1, bins=10, histtype=u'step', label="Monthly Returns")
    plt.hist(prices_of_beta2, bins=bins,  histtype=u'step', label="Event Months Returns")
    plt.legend()
    plt.savefig("histogram.png")
                
    print("Portfolios Made")

print("running")

window_size = 2520
start_year = 1990
end_year = 2019
n_stocks = 500

# Rerun flag
rerunMonthlyReturns = False

# Directory for storing pickle files
data_directory = 'data'
os.makedirs(data_directory, exist_ok=True)

# Construct file paths
monthly_returns_filename = os.path.join(data_directory, f'{window_size}_{n_stocks}_{start_year}_{end_year}_monthly_returns.pickle')
spr_returns_filename = os.path.join(data_directory, f'{window_size}_{n_stocks}_{start_year}_{end_year}_spr_returns.pickle')

try:
    if rerunMonthlyReturns or not (os.path.exists(monthly_returns_filename) and os.path.exists(spr_returns_filename)):
        print("re-running everything")

        event_month_ranges, monthly_day_ranges = get_event_month_blocks(window_size)
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

except Exception as e:
    print(f"An error occurred: {e}")

calculate_betas_and_portfolio_returns(monthly_returns, spr_returns)

