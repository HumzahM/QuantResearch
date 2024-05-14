#################################
#Introduction:
#This file is the final version of the project.
#Every function is in this file for the process of calculating a simple "price of beta"
#All you need is the rf_daily_rate.xlsx and the value_weighted_return.xlsx
#The output will be the price of beta and a T-test result.

#Everything that is "calendar" refers to simple calendar months
#Everything that is "event" refers to event months
#To run do "py -m final" or whatever you use to run python files

#High Level Process:
#A. Build "Event" and "Calendar" Month Ranges
# 1. Get daily numtrades data for the NASDAQ exchange. NYSE is not used because it is not available in the WRDS database.

#################################

#imports
import pandas as pd
import numpy as np
import wrds
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
import os

#Fetch Stock Data: Takes in a start year, end year (both inclusive), and the number of stocks to fetch data for.
#For every year in the range, the function fetches the daily stock data for the top n_stocks by market cap.
#The data is then saved to data/{n_stocks}_{start_year}_{end_year}_stock_data.pickle

#Input: start_year (int), end_year (int), n_stocks (int)
#Output: data (pd.DataFrame) - columns: date, permco, ret, market_cap, percent_spread

def fetch_stock_data(start_year, end_year, n_stocks):
    pickle_file = f'data/{n_stocks}_{start_year}_{end_year}/stock_data.pickle'

    if os.path.exists(pickle_file):
         with open(pickle_file, 'rb') as f:
            saved_data = pickle.load(f)
            saved_n_stocks = saved_data['n_stocks']
            saved_start_year = saved_data['start_year']
            saved_end_year = saved_data['end_year']

            # Check if the saved query matches the current query
            if saved_n_stocks == n_stocks and saved_start_year == start_year and saved_end_year == end_year:
                print("Loading data from saved file.")
                print(saved_data['data'])
                return saved_data['data']

    data = pd.DataFrame()
    db = wrds.Connection(wrds_username='humzahm')
    for year in range(start_year, end_year+1):
        print(f'Fetching data for year {year}...')
        start_date = f'{year}-01-01'
        end_date = f'{year}-12-31'
        query = f"""
        WITH StockAverageMarketCap AS (
            SELECT 
                permco,
                AVG(ABS(shrout * prc)) as avg_market_cap
            FROM 
                crsp.dsf
            WHERE 
                date >= '{start_date}' AND date <= '{end_date}'
                AND prc IS NOT NULL
                AND shrout IS NOT NULL
            GROUP BY permco
        ),
        Top500Stocks AS (
            SELECT 
                permco
            FROM (
                SELECT 
                    permco,
                    RANK() OVER (ORDER BY avg_market_cap DESC) as cap_rank
                FROM 
                    StockAverageMarketCap
            ) as Ranked
            WHERE 
                cap_rank <= '{n_stocks}'
        )
        SELECT 
            dsf.date, 
            dsf.permco, 
            dsf.ret,
            dsf.shrout * dsf.prc as market_cap,
            (dsf.ask - dsf.bid) / dsf.prc as percent_spread
        FROM 
            crsp.dsf as dsf
        JOIN 
            Top500Stocks ON dsf.permco = Top500Stocks.permco
        WHERE 
            dsf.date >= '{start_date}' AND dsf.date <= '{end_date}'
"""
        result = db.raw_sql(query)
        data = pd.concat([data, result])
    
    data = data.sort_values(by=['permco', 'date'])
    data = data.reset_index(drop=True)
    data['ret'] = data['ret'].fillna(0)

    with open (pickle_file, 'wb') as f:
        pickle.dump({'data': data, 'n_stocks': n_stocks, 'start_year': start_year, 'end_year': end_year}, f)

    return data

def symmetric_smoothing(arr, window_size):
    n = len(arr)
    smoothed = np.full(n, -1.0)  # Initialize all elements to -1
    
    half_window = window_size // 2  # Calculate half window size
    
    for i in range(half_window, n - half_window):
        # Define the window range
        window = arr[i - half_window:i + half_window + 1]
        # Calculate the mean of the window
        smoothed[i] = np.mean(window)
    return smoothed

#Input: Start_Year (int), End_Year (int), n_stocks (int), window_size (int)
#N stocks is just a formality so it knows what folder to save the data in
#Output: Two arrays of tuples, each tuple containing the first and last date of the month

def get_event_month_blocks(window_size, start_year, end_year, n_stocks, overwrite=False):
    start_date_data = f'{start_year-5}-01-01'
    start_date = f'{start_year}-01-01'
    end_date_data = f'{end_year+5}-12-31'
    end_date = f'{end_year}-12-31'

    sql_query = f"""
    SELECT date, SUM(NUMTRD) AS "total daily trades" 
    FROM crsp.dsf
    WHERE date BETWEEN '{start_date_data}' AND '{end_date_data}'
    GROUP BY date
    ORDER BY date;
    """

    if os.path.exists(f'data/{n_stocks}_{start_year}_{end_year}/event_and_calendar_months.pickle') and not overwrite:
        with open(f'data/{n_stocks}_{start_year}_{end_year}/event_and_calendar_months.pickle', 'rb') as f:
            data = pickle.load(f)
            return data['event_months'], data['calendar_months']
    
    db = wrds.Connection(wrds_username='humzahm')
    data = db.raw_sql(sql_query, date_cols=['date'])

    #some trading days had 0, 1, or NaN, either bugs or not they mess stuff up
    data["total daily trades"].ffill(inplace=True)
    data["total daily trades"].fillna(value=1000, inplace=True)

    data['month'] = data['date'].dt.strftime('%Y-%m')

    # Applying the function to a single column of the DataFrame
    #data['trading log moving average'] = np.exp(symmetric_smoothing(np.log(data['total daily trades']), window_size))
    os.makedirs(f'figs/{n_stocks}_{start_year}_{end_year}', exist_ok=True)
    data['trading log moving average'] = symmetric_smoothing(np.log(data['total daily trades']), window_size)
    plt.plot(np.log(data['total daily trades']))
    plt.plot(data['trading log moving average'])
    plt.savefig(f'figs/{n_stocks}_{start_year}_{end_year}/trading_log_moving_average_{window_size}.png')
    plt.figure()
    data['month'] = data['date'].dt.strftime('%Y-%m')
    monthly_avg_trades = data.groupby("month")['total daily trades'].transform('mean').drop_duplicates().reset_index(drop=True)
    unique_months = data['month'].drop_duplicates().reset_index(drop=True)
    plt.plot(unique_months, monthly_avg_trades)
    step_size = len(unique_months) // 7  # Adjust this number based on your preference
    plt.xticks(unique_months[::step_size])  # Set x-ticks at an interval of step_size
    plt.ylim(0, 1.1 * monthly_avg_trades.max())
    plt.title("Monthly Average Trades on NASDAQ")
    plt.savefig(f'figs/{n_stocks}_{start_year}_{end_year}/monthly_avg_trade.png')
    data = data[(data['date'] <= end_date) & (data['date'] >= start_date)]

    # Check for -1 values in the smoothed data
    if any(data['trading log moving average'] < 0):
        raise ValueError("Invalid smoothed values found within the specified date range.")

    trading_moving_average_with_log = (data['total daily trades'] / data['trading log moving average']).to_numpy()
    normalized_trading_scaled = trading_moving_average_with_log * trading_moving_average_with_log.shape[0]/np.sum(trading_moving_average_with_log)

    num_events = len(data.groupby('month')['total daily trades'].sum())
    normalized_days_per_month = np.sum(normalized_trading_scaled)/(num_events) #equal to days per month
    new_blocks = np.empty(num_events, dtype=int)
    current_sum = 0 
    counter = 0
    for i in range(normalized_trading_scaled.size):
        current_sum += normalized_trading_scaled[i]
        if(current_sum > normalized_days_per_month):
            current_sum -= normalized_days_per_month
            new_blocks[counter] = i 
            counter += 1

    print(new_blocks)
            
    data['date'] = data['date'].dt.strftime('%Y-%m-%d')
    new_blocks[-1] = normalized_trading_scaled.size-1
    first_last_pairs_array_event_months = []
    first_last_pairs_time_months = []
    for month in data['month'].unique():
        monthly_data = data[data['month'] == month]
        first_date = monthly_data['date'].iloc[0]
        last_date = monthly_data['date'].iloc[-1]
        first_last_pairs_time_months.append([first_date, last_date])

    for i in range(num_events):
        pass
        #print(new_blocks[i])
        #first_date = data['date'][new_blocks[i - 1] + 1] if i > 0 else data['date'][0]
        #last_date = data['date'][new_blocks[i]] if i < num_events - 1 else data['date'].iloc[-1]
        #first_last_pairs_array_event_months.append([first_date, last_date])
    
    # Converting the list of pairs to a 2D numpy array
    first_last_pairs_array_time_months = np.array(first_last_pairs_time_months)
    first_last_pairs_array_event_months = np.array(first_last_pairs_array_event_months)
    os.makedirs(f'data/{n_stocks}_{start_year}_{end_year}', exist_ok=True)
    with open(f'data/{n_stocks}_{start_year}_{end_year}/event_and_calendar_months.pickle', 'wb') as f:
        pickle.dump({'event_months': first_last_pairs_array_event_months, 'calendar_months': first_last_pairs_array_time_months}, f)

    return first_last_pairs_array_event_months, first_last_pairs_array_time_months

get_event_month_blocks(2000, 2007, 2008, 500, overwrite=True)