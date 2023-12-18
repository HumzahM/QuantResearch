import numpy as np
import pandas as pd
import wrds
from scipy.stats import ttest_ind
import os
import pickle
import matplotlib.pyplot as plt
from scipy.stats import kurtosis, skew
from scipy.stats import ttest_ind
import random
import pandas as pd

start_date = '1990-01-01'
end_date = '2022-12-31'
risk_free = pd.read_csv('rf daily rate.csv')

def fetch_stock_data(start_date, end_date):
    # Define the pickle file path
    pickle_file = 'stock_data.pickle'
    num_stocks = 5 #should be 500 later

    # Prepare the query
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
                RANK() OVER (ORDER BY AVG(ABS(shrout * prc)) DESC) as cap_rank
            FROM 
                crsp.dsf
            WHERE 
                date >= '{start_date}' AND date <= '{end_date}'
                AND prc IS NOT NULL
                AND shrout IS NOT NULL
            GROUP BY permco
        ) as Ranked
        WHERE 
            cap_rank <= '{num_stocks}'
    )
    SELECT 
        dsf.date, 
        dsf.permco, 
        dsf.ret,
        dsf.numtrd as "NUMTRD"
    FROM 
        crsp.dsf as dsf
    JOIN 
        Top500Stocks ON dsf.permco = Top500Stocks.permco
    WHERE 
        dsf.date >= '{start_date}' AND dsf.date <= '{end_date}'
    """

    #Check if the pickle file exists
    if os.path.exists(pickle_file):
        with open(pickle_file, 'rb') as f:
            saved_data = pickle.load(f)
            saved_query = saved_data['query']

            # Check if the saved query matches the current query
            if saved_query == query:
                print("Loading data from saved file.")
                print(saved_data['data'])
                return saved_data['data']

    # If no saved file or query doesn't match, execute the new query
    db = wrds.Connection(wrds_username='humzahm')
    data = db.raw_sql(query)
    data['ret'] = data['ret'].fillna(0)
    #data['market_cap'] = data['market_cap'].fillna(1) #this might be bad
    print(data)
    # Save the new query and data to pickle
    with open(pickle_file, 'wb') as f:
        pickle.dump({'query': query, 'data': data}, f)

    return data

def symmetric_smoothing(series, window, method='average'):
    # Extended series with NaNs for handling edge cases
    extended_series = pd.Series([np.nan] * window + series.tolist() + [np.nan] * window)
    
    # List to store the results
    results = []

    for i in range(len(series)):
        # Adjust the window size symmetrically
        actual_window_size = min(i, len(series) - i - 1, window)

        # Calculate the start and end indices for the adjusted window
        start_index = i + window - actual_window_size
        end_index = i + window + actual_window_size + 1

        # Extract the window values, ignoring NaNs
        window_values = extended_series[start_index:end_index].dropna()

        # Calculate the average or median based on the method
        if method == 'average':
            result = window_values.mean()
        else:  # 'median'
            result = window_values.median()
        
        results.append(result)

    return results

def get_event_month_blocks(data, window_size):

    #some trading days had 0, 1, or NaN, either bugs or not they mess stuff up
    data["NUMTRD"].fillna(value=0)
    data = data[data["NUMTRD"] > 1]
    data['month'] = data['date'].str[:-3]

    # Applying the function to a single column of the DataFrame
    data['trading moving average'] = np.exp(symmetric_smoothing(np.log(data['NUMTRD']), window_size, method='average'))
    data['trading moving median'] = symmetric_smoothing(data['NUMTRD'], window_size, method='average')

    data = data[data['date'] <= end_date]
    data = data[data['date'] >= start_date].reset_index(drop=True)

    trading_moving_average = (data['NUMTRD'] / data['trading moving average']).to_numpy()

    normalized_trading_scaled = trading_moving_average * trading_moving_average.shape[0]/np.sum(trading_moving_average)

    num_events = len(data.groupby('month')['NUMTRD'].sum())
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
    plt.savefig("total_market_trades.png")
    plt.figure()
    first_last_pairs_array_event_months = np.empty(num_events, dtype=object)
    first_last_pairs_time_months = []

    for month in data['month'].unique():
        # Filtering the data for the current month
        monthly_data = data[data['date'].dt.to_period('M') == month]
        
        # Getting the first and last date of the month
        first_date = monthly_data['date'].iloc[0].strftime('%Y-%m-%d')
        last_date = monthly_data['date'].iloc[-1].strftime('%Y-%m-%d')
        first_last_pairs_time_months.append([first_date, last_date])

    data['date'] = data['date'].dt.strftime('%Y-%m-%d')
# Iterating through the events to get the pairs
    for i in range(num_events):
        # Getting the first date for the current block
        first_date = data['date'][new_blocks[i - 1] + 1] if i > 0 else data['date'][0]

        # Getting the last date for the current block
        last_date = data['date'][new_blocks[i]] if i < num_events - 1 else data['date'].iloc[-1]

        # Storing the pair in the array
        first_last_pairs_array_event_months[i] = [first_date, last_date]
    
    
    # Converting the list of pairs to a 2D numpy array
    first_last_pairs_array_time_months = np.array(first_last_pairs_time_months)
    
    return first_last_pairs_array_event_months, first_last_pairs_array_time_months

def process_data(data, rf, window_size):
    unique_permcos = data['permco'].unique()

    skew_range1, skew_range2 = [], []
    kurtosis_range1, kurtosis_range2 = [], []
    counter = 1
    for stock in unique_permcos:
        print(stock)
        print(counter)
        counter += 1
        #setup data
        stock_data = data[data['permco'] == stock]
        stock_data["NUMTRD"].fillna(value=0)
        rf_filtered = rf[rf['date'].isin(stock_data['date'])]
        stock_data = stock_data.merge(rf_filtered, on='date')
        stock_data["log returns"] = np.log(stock_data["ret"] + 1 - stock_data["rf"])
        #calculate normalized trading/event blocks
        event_months, months = get_event_month_blocks(stock_data, window_size)
        #calculate monthly returns for each set 

        stock_dates = stock_data['date']
        #print(permco_dates)
        first_date_stock = stock_dates.min()
        last_date_stock = stock_dates.max()

        monthly_returns1, monthly_returns2 = [], []

        for i, ((start1, end1), (start2, end2)) in enumerate(zip(event_months, months)):
            if start1 >= first_date_stock and end1 <= last_date_stock and start2 >= first_date_stock and end2 <= last_date_stock:
                filtered_stock_data1 = pd.merge(stock_data[(stock_data['date'] >= start1) & (stock_data['date'] <= end1)], rf, on='date')
                filtered_stock_data2 = pd.merge(stock_data[(stock_data['date'] >= start2) & (stock_data['date'] <= end2)], rf, on='date')

                filtered_stock_data1['log equity returns'] = np.log(1 + filtered_stock_data1['ret'] - filtered_stock_data1['rf'])
                filtered_stock_data2['log equity returns'] = np.log(1 + filtered_stock_data2['ret'] - filtered_stock_data2['rf'])

                monthly_return1 = filtered_stock_data1['log equity returns'].sum()
                monthly_return2 = filtered_stock_data2['log equity returns'].sum()

                monthly_returns1.append(monthly_return1)
                monthly_returns2.append(monthly_return2)
        #calculate skew and kurtosis for each set
        skew_range1.append(skew(monthly_returns1))
        skew_range2.append(skew(monthly_returns2))
        kurtosis_range1.append(kurtosis(monthly_returns1))
        kurtosis_range2.append(kurtosis(monthly_returns2))
    
    avg_kurtosis_range1 = sum(kurtosis_range1) / len(kurtosis_range1)
    avg_kurtosis_range2 = sum(kurtosis_range2) / len(kurtosis_range2)
    avg_skew_range1 = sum(skew_range1) / len(skew_range1)
    avg_skew_range2 = sum(skew_range2) / len(skew_range2)

    # Perform t-test on the results of the two methods
    t_stat_kurtosis, p_value_kurtosis = ttest_ind(kurtosis_range1, kurtosis_range2)
    t_stat_skew, p_value_skew = ttest_ind(skew_range1, skew_range2)

    # Print results
    print(f"Average Kurtosis for Range 1: {avg_kurtosis_range1}")
    print(f"Average Kurtosis for Range 2: {avg_kurtosis_range2}")
    print(f"Average Skew for Range 1: {avg_skew_range1}")
    print(f"Average Skew for Range 2: {avg_skew_range2}")
    print(f"T-statistic for Kurtosis: {t_stat_kurtosis}, P-value for Kurtosis (One Tailed): {p_value_kurtosis/2}")
    print(f"T-statistic for Skew: {t_stat_skew}, P-value for Skew (One Tailed): {p_value_skew/2}")


print("running")

data = fetch_stock_data(start_date, end_date)

process_data(data, rf=risk_free, window_size=1008)

