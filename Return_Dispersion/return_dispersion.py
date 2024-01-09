import wrds
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import sys
from pathlib import Path

# Add the parent directory to sys.path to allow for package imports
parent_dir = str(Path(__file__).resolve().parent.parent)
sys.path.append(parent_dir)

from Helper_Functions.advanced_fetch_stock_data import advanced_fetch_stock_data
from Helper_Functions.total_market_trades import *

def day_return_dispersion(returns, market_caps, spreads):
    # Calculate the weighted mean return
    total_market_cap = np.sum(market_caps)
    weights = market_caps / total_market_cap
    weighted_mean_return = np.sum(weights * returns)

    # Calculate the weighted standard deviation (equity return dispersion)
    squared_diff = (returns - weighted_mean_return) ** 2
    weighted_squared_diff = weights * squared_diff
    weighted_variance = np.sum(weighted_squared_diff)
    #equity_return_dispersion = np.sqrt(weighted_variance)
    equity_return_dispersion = weighted_variance

    final = equity_return_dispersion - np.average(spreads ** 2, weights=market_caps) / 4

    if(final < 0):
        final = 0

    return final

def get_event_blocks_return_dispersion():
    data = advanced_fetch_stock_data(1990, 2019, 500)
    data["percent_spread"].fillna(method="ffill", inplace=True)
    data["market_cap"].fillna(method='ffill', inplace=True)
    data["ret"].fillna(0, inplace=True)
    data['month'] = pd.DatetimeIndex(data['date']).to_period('M')
    return_dispersions = data.groupby("date").apply(lambda x: day_return_dispersion(x["ret"], x["market_cap"], x["percent_spread"]))
    num_events = len(data.groupby('month')['ret'].sum())
    normalized_trading_scaled = np.array(return_dispersions * return_dispersions.shape[0]/np.sum(return_dispersions))
    indices = np.where(normalized_trading_scaled > 20)[0]
    for index in indices:
        print(data['date'][index])
    n = 50
    rolling_mean = np.convolve(normalized_trading_scaled, np.ones(n), mode='full') / n
    # Plot the rolling mean
    plt.plot(rolling_mean)
    plt.xlabel('Days')
    plt.ylabel('Mean Trades per Day')
    plt.title(f'Mean Trades Per Day (n={n})')
    plt.savefig("Rolling Mean (RD as Variance - Bid-Ask Spread)")
    plt.figure()
    plt.plot(normalized_trading_scaled)
    plt.xlabel('Days')
    plt.ylabel('Event Days')
    plt.title(f'Event Days (RD as Variance - Bid-Ask Spread)')
    plt.savefig("Event Days (RD as Variance - Bid-Ask Spread)")
    normalized_days_per_month = np.sum(normalized_trading_scaled)/(num_events) #equal to days per month
    new_blocks = np.empty(num_events, dtype=int)
    event_month_lengths = np.empty(num_events)
    current_sum = 0 
    counter = 0
    othercounter = 0 
    for i in range(normalized_trading_scaled.size):
        current_sum += normalized_trading_scaled[i]
        othercounter += 1
        if(current_sum > normalized_days_per_month):
            current_sum -= normalized_days_per_month 
            new_blocks[counter] = i
            event_month_lengths[counter] = othercounter
            counter += 1
            othercounter = 0
            
    #last one is end 
    new_blocks[-1] = normalized_trading_scaled.size-1
    event_month_lengths[-1] = num_events*normalized_days_per_month - np.sum(event_month_lengths[:-1])
    first_last_pairs_array_event_months = np.empty(num_events, dtype=object)
    first_last_pairs_time_months = []
    for month in data['month'].unique():
        # Filtering the data for the current month
        monthly_data = data[data['month'] == month]
        
        # Getting the first and last date of the month
        first_date = monthly_data['date'].iloc[0].strftime('%Y-%m-%d')
        last_date = monthly_data['date'].iloc[-1].strftime('%Y-%m-%d')
        #first_date = monthly_data['date'].iloc[0]
        #last_date = monthly_data['date'].iloc[-1]

        # Adding the pair to the list
        first_last_pairs_time_months.append([first_date, last_date])

    #data['date'] = data['date'].strftime('%Y-%m-%d')
    # Iterating through the events to get the pairs
    for i in range(num_events):
        # Getting the first date for the current block
        first_date = data['date'][new_blocks[i - 1] + 1] if i > 0 else data['date'][0]

        # Getting the last date for the current block
        last_date = data['date'][new_blocks[i]]

        # Printing the dates for the current block
        last_date = data['date'][new_blocks[i]] if i < num_events - 1 else data['date'].iloc[-1]

        # Storing the pair in the array
        first_last_pairs_array_event_months[i] = [first_date, last_date]


    # Converting the list of pairs to a 2D numpy array
    first_last_pairs_array_time_months = np.array(first_last_pairs_time_months)


    return first_last_pairs_array_event_months, first_last_pairs_array_time_months

def optimize(min, max, step, return_dispersion_normalized_trading_scaled):
    MAEs = []
    correlations = []
    for i in np.arange(min, max, step):
        print(f'N = {i}')
        normalized_trading_scaled = optimize_helper(i, 1990, 2019)
        MAE = np.mean(np.abs(normalized_trading_scaled - return_dispersion_normalized_trading_scaled))
        MAEs.append(MAE)
        correlations.append(np.corrcoef(normalized_trading_scaled, return_dispersion_normalized_trading_scaled)[0, 1])
    plt.figure()
    plt.plot(np.arange(min, max, step), MAEs)
    plt.xlabel('N')
    plt.ylabel('MAE')
    plt.title(f'MAE vs N')
    plt.savefig("MAE vs N (RD as Variance - Bid-Ask Spread))")
    plt.figure()
    plt.plot(np.arange(min, max, step), correlations)
    plt.xlabel('N')
    plt.ylabel('Correlation')
    plt.title(f'Correlation vs N')
    plt.savefig("Correlation vs N (RD as Variance - Bid-Ask Spread))")

# _, _, return_dispersion_normalized_trading_scaled_ = get_event_blocks_return_dispersion()

# optimize(250, 5000, 50, return_dispersion_normalized_trading_scaled_)
