import numpy as np
import pandas as pd
import wrds
import statsmodels.api as sm
from scipy.stats import ttest_ind
import os
import pickle
import matplotlib.pyplot as plt
from scipy.stats import kurtosis, skew
from scipy.stats import ttest_ind
import warnings
warnings.filterwarnings('ignore') # :) 

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

#for just the pure beta without forcing intercept to be 0
def calculate_beta(y, x):
    y = np.array(y)
    x = np.array(x)
    covariance_matrix = np.cov(y, x)
    covariance = covariance_matrix[0, 1]
    variance = np.var(x)
    beta = covariance / variance
    return beta

#for when theory suggests that intercept should be 0

def calculate_beta_force(y, x):
    return sm.OLS(y, x).fit().params[0]

import statsmodels.api as sm
import numpy as np

import statsmodels.api as sm
import numpy as np

def calculate_weighted_beta(y, x, weights):
    """
    Calculate the weighted beta of a linear relationship between y and x,
    including an intercept, with scaled weights (market caps).
    """
    y = np.array(y)
    x = np.array(x)
    weights = np.array(weights)

    # Scale down weights by a factor (e.g., 1 million) and round off to avoid zeros
    scale_factor = 1e9
    scaled_weights = np.round(weights / scale_factor)
    scaled_weights[scaled_weights == 0] = 1  # Replace any zero weights with 1

    # Add a column of ones to x for the intercept
    x_with_intercept = sm.add_constant(x)

    # # Fit the weighted least squares model
    model = sm.WLS(y, x_with_intercept, weights=scaled_weights).fit()

    # # Return the intercept and beta (slope)
    intercept, beta = model.params

    return beta


def calculate_betas(monthly_returns):
    unique_permcos = monthly_returns['permco'].unique()
    monthly_returns['beta'] = np.nan
    counter = 1

    for permco in unique_permcos:
        print(counter)
        counter += 1
        data_permco = monthly_returns[monthly_returns['permco'] == permco]
        data_range1 = data_permco[data_permco['type'] == 1].sort_values(by=['sequence #'])
        data_range2 = data_permco[data_permco['type'] == 2].sort_values(by=['sequence #'])

        for i in range(len(data_range1)):
            if i >= 60:
                y_range1 = data_range1['equity_returns'].iloc[i-60:i]
                x_range1 = data_range1['sp500_return'].iloc[i-60:i]
                beta_range1 = calculate_beta_force(y_range1, x_range1)
                monthly_returns.loc[data_range1.index[i], 'beta'] = beta_range1

        for i in range(len(data_range2)):
            if i >= 60:
                y_range2 = data_range2['equity_returns'].iloc[i-60:i]
                x_range2 = data_range2['sp500_return'].iloc[i-60:i]
                beta_range2 = calculate_beta_force(y_range2, x_range2)
                monthly_returns.loc[data_range2.index[i], 'beta'] = beta_range2

    return monthly_returns

def calculate_price_of_betas(monthly_returns):
    monthly_returns['market_cap'].fillna(0, inplace=True)
    monthly_returns = monthly_returns[monthly_returns['beta'].notna()]
    unique_sequences = monthly_returns['sequence #'].unique()
    price_of_beta1, price_of_beta2 = [], []

    for seq in unique_sequences:
        print(f'Calculating price of beta for sequence {seq}')
        data_seq = monthly_returns[monthly_returns['sequence #'] == seq]
        data_seq1 = data_seq[(data_seq['type'] == 1) & (data_seq['market_cap'].notnull())].sort_values(by='permco')
        data_seq2 = data_seq[(data_seq['type'] == 2) & (data_seq['market_cap'].notnull())].sort_values(by='permco')
        price_of_beta1.append(calculate_weighted_beta(data_seq1['equity_returns'], data_seq1['beta'], data_seq1['market_cap']))
        price_of_beta2.append(calculate_weighted_beta(data_seq2['equity_returns'], data_seq2['beta'], data_seq2['market_cap']))

    # Plotting and t-test
    price_of_beta1 = np.array(price_of_beta1)
    price_of_beta2 = np.array(price_of_beta2)

    # Scatter plot
    plt.figure()
    plt.scatter(monthly_returns[monthly_returns['type'] == 1]['beta'], monthly_returns[monthly_returns['type'] == 1]['equity_returns'], label="Monthly Returns")
    plt.scatter(monthly_returns[monthly_returns['type'] == 2]['beta'], monthly_returns[monthly_returns['type'] == 2]['equity_returns'], label="Event Months Returns")
    plt.legend()
    plt.savefig("scatter.png")

    # Histogram
    plt.figure()
    _, bins, _ = plt.hist(price_of_beta1, bins=20, histtype=u'step', label="Monthly Returns")
    plt.hist(price_of_beta2, bins=bins, histtype=u'step', label="Event Months Returns")
    plt.legend()
    plt.savefig("histogram.png")
    
    # T-test
    t_stat, p_value = ttest_ind(price_of_beta1, price_of_beta2)
    print(f"Average price of beta for Range 1: {price_of_beta1.mean()}")
    print(f"Average price of beta for Range 2: {price_of_beta2.mean()}")
    print(f"T-statistic for Difference: {t_stat}, P-value for Difference (One Tailed): {p_value/2}")

print("running")

window_size = 2520
start_year = 1990
end_year = 2019
n_stocks = 500

# Rerun flag
rerunMonthlyReturns = False
rerunBetaCalc = False

if(rerunMonthlyReturns):
    rerunBetaCalc = True

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

betad_monthly_returns_filename = os.path.join(data_directory, f'{window_size}_{n_stocks}_{start_year}_{end_year}_betad_monthly_returns.pickle')

if rerunBetaCalc or not os.path.exists(betad_monthly_returns_filename):
    betad_monthly_returns = calculate_betas(monthly_returns)
    with open(betad_monthly_returns_filename, 'wb') as f:
        pickle.dump(betad_monthly_returns, f)
else:
    with open(betad_monthly_returns_filename, 'rb') as f:
        betad_monthly_returns = pickle.load(f)

calculate_price_of_betas(betad_monthly_returns)

