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

#Add the parent directory to sys.path to allow for package imports
parent_dir = str(Path(__file__).resolve().parent.parent)
sys.path.append(parent_dir)

from Helper_Functions.total_market_trades import *
from Helper_Functions.advanced_fetch_stock_data import advanced_fetch_stock_data
from Helper_Functions.calculate_monthly_returns import calculate_monthly_returns
from Helper_Functions.better_calculate_monthly_returns import better_calculate_monthly_returns
import return_dispersion

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
    num_groups = 10
    portfolio_range1 = [[] for _ in range(num_groups)]
    portfolio_range2 = [[] for _ in range(num_groups)]
    spreturns1 = []
    spreturns2 = []
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

            for i in range(num_groups):
                group1_indices = betas1_grouped[betas1_grouped == i].index
                group2_indices = betas2_grouped[betas2_grouped == i].index
                portfolio_range1[i].append(np.average(returns1[group1_indices], weights=adjusted_caps1[group1_indices]))
                portfolio_range2[i].append(np.average(returns2[group2_indices], weights=adjusted_caps2[group2_indices]))

            relevant_sp_data = spr_returns[spr_returns['sequence #'].isin(range(seq-11,seq+1))]
            #sp500ret1 = np.sum(relevant_sp_data["sp500_return_range1"])
            #sp500ret2 = np.sum(relevant_sp_data["sp500_return_range2"])
            sp500ret1 = np.log((1+relevant_sp_data["sp500_return_range1"]).prod())
            sp500ret2 = np.log((1+relevant_sp_data["sp500_return_range2"]).prod())
            spreturns1.append(sp500ret1)
            spreturns2.append(sp500ret2)


    print("Portfolios Made")
    betas1 = []
    means1 = []
    betas2 = []
    means2 = []
    skews1 = []
    skews2 = []
    kurtosis1 = []
    kurtosis2 = []

    for point in portfolio_range1:
        beta = calculate_beta_force(point, spreturns1)
        betas1.append(beta)
        means1.append(np.mean(point))
        skews1.append(skew(point))
        kurtosis1.append(kurtosis(point))

    for point in portfolio_range2:
        beta = calculate_beta_force(point, spreturns2)
        betas2.append(beta)
        means2.append(np.mean(point))
        skews2.append(skew(point))
        kurtosis2.append(kurtosis(point))

    betas3 = []
    betas3.append(1)
    betas3.append(1)
    means3 = []
    means3.append(np.mean(spreturns1))
    means3.append(np.mean(spreturns2))
    plt.figure()
    plt.scatter(betas1, means1, color='blue')
    plt.scatter(betas2, means2, color='red')
    plt.scatter(betas3, means3, color='green')
    plt.xlabel('Beta')
    plt.ylabel('Mean Return')
    plt.title(f'Beta vs Mean Return (Blue is Normal Months, Red is Event Months) \n Betas are {calculate_beta(means1, betas1)} and {calculate_beta(means2, betas2)}')
    plt.savefig("beta_vs_mean_return.png")

    # Create a dictionary with the variables
    data = {
        'betas1': betas1,
        'means1': means1,
        'betas2': betas2,
        'means2': means2
    }

    # Create a DataFrame from the dictionary
    df = pd.DataFrame(data)

    # Save the DataFrame to a CSV file
    try:
        df.to_csv('output.csv', index=False)
    except:
        print("Error writing to CSV file.")

    avg_kurtosis1 = sum(kurtosis1) / len(kurtosis1)
    avg_kurtosis2 = sum(kurtosis2) / len(kurtosis2)
    avg_skews1 = sum(skews1) / len(skews1)
    avg_skews2 = sum(skews2) / len(skews2)

    # Perform t-test on the results of the two methods
    t_stat_kurtosis, p_value_kurtosis = ttest_ind(kurtosis1, kurtosis2)
    t_stat_skew, p_value_skew = ttest_ind(skews1, skews2)

    # Print results
    print(f"Average Kurtosis for Range 1: {avg_kurtosis1}")
    print(f"Average Kurtosis for Range 2: {avg_kurtosis2}")
    print(f"Average Skew for Range 1: {avg_skews1}")
    print(f"Average Skew for Range 2: {avg_skews2}")
    print(f"T-statistic for Kurtosis: {t_stat_kurtosis}, P-value for Kurtosis (One Tailed): {p_value_kurtosis/2}")
    print(f"T-statistic for Skew: {t_stat_skew}, P-value for Skew (One Tailed): {p_value_skew/2}")

    # Plot histograms of the portfolio returns - doesn't do anything useful rn but maybe I'll want to come back to it later

    # portfolio_ranges = [portfolio_range1, portfolio_range2]
    # colors = ['blue', 'red']  # Blue for portfolio_range1, Red for portfolio_range2

    # # Create a figure with 20 subplots (2 rows x 10 columns)
    # fig, axs = plt.subplots(2, 10, figsize=(30, 10))

    # # Set a main title for the figure
    # fig.suptitle('Histograms of Portfolio Returns', fontsize=16)

    # for i in range(2):
    #     for j in range(10):
    #         point = portfolio_ranges[i][j]

    #         # Plot histogram in the subplot
    #         axs[i, j].hist(point, bins=20, color=colors[i])
    #         axs[i, j].set_title(f'Portfolio {i+1} - Point {j+1}')
    #         axs[i, j].set_xlabel('Returns')
    #         axs[i, j].set_ylabel('Frequency')

    #         # Highlighting the row for each portfolio range
    #         if j == 0:
    #             axs[i, j].text(-0.3, 0.5, f'Portfolio {i+1}', transform=axs[i, j].transAxes, 
    #                         fontsize=14, color='black', rotation='vertical', verticalalignment='center')

    # # Adjust layout to prevent overlap
    # plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust the rect to account for the main title
    # plt.savefig("histograms_beta_vs_mean_return.png")
    
    plt.figure()
    counter = 0
    colors = plt.cm.Reds(np.linspace(0.2, 1, len(portfolio_range1)))  # Generate a color gradient from yellowish to red
    for point, color in zip(portfolio_range1, colors):
        plt.plot(point, label=f'Beta {round(betas1[counter], 2)}', color=color)
        counter += 1 
    plt.plot(spreturns1, label="SP500 Returns", color='black', linewidth=2.5)
    plt.legend()
    plt.title("Portfolio Returns for Range 1")
    plt.savefig("portfolio_returns1.png")
    plt.figure()

    counter = 0
    colors = plt.cm.Reds(np.linspace(0.2, 1, len(portfolio_range2)))  # Generate a color gradient from yellowish to red
    for point, color in zip(portfolio_range2, colors):
        plt.plot(point, label=f'Beta {round(betas2[counter], 2)}', color=color)
        counter += 1
    plt.plot(spreturns2, label="SP500 Returns", color='black', linewidth=2.5)
    plt.legend()
    plt.title("Portfolio Returns for Range 2")
    plt.savefig("portfolio_returns2.png")

    #plt.show()

print("running")

window_size = 2520
start_year = 1990
end_year = 2019
n_stocks = 500
start_date = f'{start_year}-01-01'
end_date = f'{end_year}-12-31'

# Rerun flag
rerunMonthlyReturns = True

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

        #event_month_ranges, monthly_day_ranges = get_event_month_blocks(window_size, start_year, end_year)
        event_month_ranges, monthly_day_ranges = return_dispersion.get_event_blocks_return_dispersion()
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
    
    calculate_betas_and_portfolio_returns(monthly_returns, spr_returns)

except Exception as e:
    print(f"An error occurred: {e}")



