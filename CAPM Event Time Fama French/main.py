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
from tqdm import tqdm
from sklearn.metrics import r2_score
from math import log
from math import isnan

import sys
from pathlib import Path

#Add the parent directory to sys.path to allow for package imports
parent_dir = str(Path(__file__).resolve().parent.parent)
sys.path.append(parent_dir)

from Helper_Functions.total_market_trades import *
from Helper_Functions.advanced_fetch_stock_data import advanced_fetch_stock_data
from Helper_Functions.calculate_monthly_returns import calculate_monthly_returns

def calculate_ff_returns(ff_data, date_ranges1, date_ranges2):
    ff_returns = pd.DataFrame()

    ff_data['market return'] = np.float64((ff_data['Mkt-RF'] - ff_data['RF']) / 100)
    ff_data['SMB'] = np.float64(ff_data['SMB'] / 100)
    ff_data['HML'] = np.float64(ff_data['HML'] / 100 )
    ff_data['date'] = pd.to_datetime(ff_data['date'])

    date_ranges1 = [(pd.to_datetime(start), pd.to_datetime(end)) for start, end in date_ranges1]
    date_ranges2 = [(pd.to_datetime(start), pd.to_datetime(end)) for start, end in date_ranges2]

        #calculate spr returns 
    for i, ((start1, end1), (start2, end2)) in enumerate(zip(date_ranges1, date_ranges2)): 
        market_return1 = log((ff_data[(ff_data['date'] >= start1) & (ff_data['date'] <= end1)]['market return']+1).prod())
        market_return2 = log((ff_data[(ff_data['date'] >= start2) & (ff_data['date'] <= end2)]['market return']+1).prod())
        smb_return1 = log((ff_data[(ff_data['date'] >= start1) & (ff_data['date'] <= end1)]['SMB']+1).prod())
        smb_return2 = log((ff_data[(ff_data['date'] >= start2) & (ff_data['date'] <= end2)]['SMB']+1).prod())
        hml_return1 = log((ff_data[(ff_data['date'] >= start1) & (ff_data['date'] <= end1)]['HML']+1).prod())
        hml_return2 = log((ff_data[(ff_data['date'] >= start2) & (ff_data['date'] <= end2)]['HML']+1).prod())
        row = pd.DataFrame({
            'sequence #': [i], 
            'market_return_range1': [market_return1],
            'market_return_range2': [market_return2],
            'smb_return_range1': [smb_return1],
            'smb_return_range2': [smb_return2],
            'hml_return_range1': [hml_return1],
            'hml_return_range2': [hml_return2]
        })
        ff_returns = pd.concat([ff_returns, row], ignore_index=True)
    return ff_returns


def calculate_beta(y, x):
    y = np.array(y)
    x = np.array(x)
    covariance_matrix = np.cov(y, x)
    covariance = covariance_matrix[0, 1]
    variance = np.var(x)
    beta = covariance / variance
    return beta

import statsmodels.api as sm

def calculate_multi_beta(y, X):
    model = sm.OLS(y, X).fit()
    return model.params  # Return the beta values directly

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


    model = sm.WLS(y, x, weights=scaled_weights).fit()

    beta = model.params[0]

    return beta

def calculate_betas(monthly_returns, fama_french):
    unique_permcos = monthly_returns['permco'].unique()
    
    # Adding new columns for beta values
    monthly_returns['market_beta'] = np.nan
    monthly_returns['SMB_beta'] = np.nan
    monthly_returns['HML_beta'] = np.nan

    for permco in tqdm(unique_permcos):
        # Filter the monthly_returns for the current permco
        data_permco = monthly_returns[monthly_returns['permco'] == permco]
        #print("-----------------------------------------------------------------------------------------------")
        #print(data_permco)
        
        # Then, merge with fama_french data
        data_permco = data_permco.merge(fama_french, on='sequence #')
        
        for data_type in [1, 2]:
            # Further filter the data for the current type
            data_range = data_permco[data_permco['type'] == data_type].sort_values(by=['sequence #'])
            # Split data into continuous blocks based on 'sequence #'
            data_blocks = [data_range.iloc[i:j] for i, j in zip(np.r_[0, np.where(np.diff(data_range['sequence #']) != 1)[0] + 1],
                                                               np.r_[np.where(np.diff(data_range['sequence #']) != 1)[0] + 1, len(data_range)])]
            for block in data_blocks:
                for i in range(len(block)):
                    if i >= 60:
                        # Get the dependent variable (y) and independent variables (X)
                        y = block['equity_returns'].iloc[i-60:i]
                        X = block[[f'market_return_range{data_type}', f'smb_return_range{data_type}', f'hml_return_range{data_type}']].iloc[i-60:i]
                        
                        # Calculate the betas for the 60-month rolling window
                        betas = calculate_multi_beta(y, X)

                        # Assign the betas to the respective columns
                        #print(monthly_returns.loc[[block.index[i]]])
                        monthly_returns.loc[block.index[i], 'market_beta'] = betas[0]
                        monthly_returns.loc[block.index[i], 'SMB_beta'] = betas[1]
                        monthly_returns.loc[block.index[i], 'HML_beta'] = betas[2]
                        #print(monthly_returns.loc[[block.index[i]]])
                        #print("-----------------------------------------------------------------------------------------------")
            print(data_blocks)
        
        print(data_permco)
    print(monthly_returns)
    return monthly_returns  # Return the DataFrame with the beta columns updated

def calculate_price_of_betas(monthly_returns):
    monthly_returns['market_cap'].fillna(0, inplace=True)
    
    # Filter out rows where any of the betas or market_cap is NaN or market_cap is non-positive
    monthly_returns = monthly_returns[(monthly_returns['market_beta'].notna()) & 
                                      (monthly_returns['SMB_beta'].notna()) & 
                                      (monthly_returns['HML_beta'].notna()) &
                                      (monthly_returns['market_cap'] > 0)]
    
    print(monthly_returns)
    
    unique_sequences = monthly_returns['sequence #'].unique()
    
    # Lists to hold the price of betas for both types
    price_of_market_beta_type1, price_of_SMB_beta_type1, price_of_HML_beta_type1 = [], [], []
    price_of_market_beta_type2, price_of_SMB_beta_type2, price_of_HML_beta_type2 = [], [], []

    for seq in tqdm(unique_sequences):
        data_seq = monthly_returns[monthly_returns['sequence #'] == seq]
        
        # Filter for type 1 and type 2
        data_seq1 = data_seq[data_seq['type'] == 1]
        data_seq2 = data_seq[data_seq['type'] == 2]
        
        # Perform weighted regressions and store results for type 1
        if not data_seq1.empty:
            price_of_market_beta_type1.append(calculate_weighted_beta(
                data_seq1['equity_returns'], data_seq1['market_beta'], data_seq1['market_cap']
            ))
            price_of_SMB_beta_type1.append(calculate_weighted_beta(
                data_seq1['equity_returns'], data_seq1['SMB_beta'], data_seq1['market_cap']
            ))
            price_of_HML_beta_type1.append(calculate_weighted_beta(
                data_seq1['equity_returns'], data_seq1['HML_beta'], data_seq1['market_cap']
            ))
        
        # Perform weighted regressions and store results for type 2
        if not data_seq2.empty:
            price_of_market_beta_type2.append(calculate_weighted_beta(
                data_seq2['equity_returns'], data_seq2['market_beta'], data_seq2['market_cap']
            ))
            price_of_SMB_beta_type2.append(calculate_weighted_beta(
                data_seq2['equity_returns'], data_seq2['SMB_beta'], data_seq2['market_cap']
            ))
            price_of_HML_beta_type2.append(calculate_weighted_beta(
                data_seq2['equity_returns'], data_seq2['HML_beta'], data_seq2['market_cap']
            ))
    
    # Print the means for each beta type for both data types
    print(f"Mean Market Beta Price for Type 1: {np.mean(price_of_market_beta_type1)}")
    print(f"Median Market Beta Price for Type 1: {np.median(price_of_market_beta_type1)}")
    print(f"Mean SMB Beta Price for Type 1: {np.mean(price_of_SMB_beta_type1)}")
    print(f'Median SMB Beta Price for Type 1: {np.median(price_of_SMB_beta_type1)}')
    print(f"Mean HML Beta Price for Type 1: {np.mean(price_of_HML_beta_type1)}")
    print(f'Median HML Beta Price for Type 1: {np.median(price_of_HML_beta_type1)}')

    print(f"Mean Market Beta Price for Type 2: {np.mean(price_of_market_beta_type2)}")
    print(f"Median Market Beta Price for Type 2: {np.median(price_of_market_beta_type2)}")
    print(f"Mean SMB Beta Price for Type 2: {np.mean(price_of_SMB_beta_type2)}")
    print(f"Median SMB Beta Price for Type 2: {np.median(price_of_SMB_beta_type2)}")
    print(f"Mean HML Beta Price for Type 2: {np.mean(price_of_HML_beta_type2)}")
    print(f"Median HML Beta Price for Type 2: {np.median(price_of_HML_beta_type2)}")

    plt.figure()
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(6, 12))
    ax1.hist(price_of_market_beta_type1, bins=20, alpha=0.5, histtype=u'step', label='Calendar')
    ax1.hist(price_of_market_beta_type2, bins=20, alpha=0.5, histtype=u'step', label='Event')
    ax1.set_title('Price of Market Beta')
    ax1.legend()

    ax2.hist(price_of_SMB_beta_type1, bins=20, alpha=0.5, histtype=u'step', label='Calendar')
    ax2.hist(price_of_SMB_beta_type2, bins=20, alpha=0.5, histtype=u'step', label='Event')
    ax2.set_xlim(-20, 20)
    ax2.set_title('Price of SMB Beta')
    ax2.legend()

    ax3.hist(price_of_HML_beta_type1, bins=20, alpha=0.5, histtype=u'step', label='Calendar')
    ax3.hist(price_of_HML_beta_type2, bins=20, alpha=0.5, histtype=u'step', label='Event')
    ax3.set_xlim(-20, 20)
    ax3.set_title('Price of HML Beta')
    ax3.legend()

    plt.savefig("Price of Beta Distributions.png")


print("running")

window_size = 2520
start_year = 1990
end_year = 2019
n_stocks = 500
start_date = f'{start_year}-01-01'
end_date = f'{end_year}-12-31'

risk_free = pd.read_csv('../Useful Data/rf daily rate.csv')
risk_free = risk_free[(risk_free['date'] >= start_date) & (risk_free['date'] <= end_date)]
market_returns = pd.read_csv('../Useful Data/value_weighted_return.csv')
market_returns = market_returns[(market_returns['date'] >= start_date) & (market_returns['date'] <= end_date)]
market_returns = market_returns.merge(risk_free, left_on='date', right_on='date')
market_returns['ret'] = market_returns['vwretd'] - market_returns['rf']
fama_french = pd.read_csv('../Useful Data/FF Factors.csv')
fama_french['date'] = pd.to_datetime(fama_french['date'])

# Rerun flag
rerunMonthlyReturns = False
rerunBetaCalc = True
if(rerunMonthlyReturns):
    rerunBetaCalc = True

# Directory for storing pickle files
data_directory = 'data'
os.makedirs(data_directory, exist_ok=True)

# Construct file paths
monthly_returns_filename = os.path.join(data_directory, f'{window_size}_{n_stocks}_{start_year}_{end_year}_monthly_returns.pickle')
spr_returns_filename = os.path.join(data_directory, f'{window_size}_{n_stocks}_{start_year}_{end_year}_spr_returns.pickle')

event_month_ranges, monthly_day_ranges = get_event_month_blocks(window_size, start_year, end_year)
print("ranges calculated")

try:
    if rerunMonthlyReturns or not (os.path.exists(monthly_returns_filename)):
        print("re-running everything")

        stocks = advanced_fetch_stock_data(start_year, end_year, n_stocks)
        print("stocks fetched")

        monthly_returns = calculate_monthly_returns(stocks, market_returns, risk_free, monthly_day_ranges, event_month_ranges, calculate_sp=False)
        print("monthly returns calculated")

        with open(monthly_returns_filename, 'wb') as f:
            pickle.dump(monthly_returns, f)

    else:
        # Load existing data
        with open(monthly_returns_filename, 'rb') as f:
            monthly_returns = pickle.load(f)

        print("Data loaded from existing files")

except Exception as e:
    print(f"An error occurred: {e}")

betad_monthly_returns_filename = os.path.join(data_directory, f'{window_size}_{n_stocks}_{start_year}_{end_year}_betad_monthly_returns.pickle')

ff_monthly = calculate_ff_returns(fama_french, monthly_day_ranges, event_month_ranges)

if rerunBetaCalc or not os.path.exists(betad_monthly_returns_filename):
    betad_monthly_returns = calculate_betas(monthly_returns, ff_monthly)
    with open(betad_monthly_returns_filename, 'wb') as f:
        pickle.dump(betad_monthly_returns, f)
else:
    with open(betad_monthly_returns_filename, 'rb') as f:
        betad_monthly_returns = pickle.load(f)

print(betad_monthly_returns)

calculate_price_of_betas(betad_monthly_returns)

