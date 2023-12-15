import wrds
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression 
import pickle
import os

def run_or_load_query(query, pickle_file='query_cache.pkl'):
    # Check if the pickle file exists and load the cached query and result
    if os.path.exists(pickle_file):
        with open(pickle_file, 'rb') as file:
            cached_query, cached_result = pickle.load(file)
        
        # If the query matches the cached query, return the cached result
        if query == cached_query:
            #print("loaded cached query")
            return cached_result

    # If the query is different, run it and cache the result
    #print("ran query online")
    db = wrds.Connection(wrds_username='humzahm')
    result = db.raw_sql(query, date_cols=['date'])

    # Cache the new query and result
    with open(pickle_file, 'wb') as file:
        pickle.dump((query, result), file)

    return result

    # Modified function for both average and median smoothing to adjust the window size symmetrically
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

def get_event_month_blocks():
    start_date_data = '1987-01-01'
    start_date = '1990-01-01'
    end_date_data = '2021-12-30'
    end_date = '2019-12-31'

    STOCK_NAME = "Market"

    sql_query = f"""
    SELECT date, SUM(NUMTRD) AS "total daily trades" 
    FROM crsp.dsf
    WHERE date BETWEEN '{start_date_data}' AND '{end_date_data}'
    GROUP BY date
    ORDER BY date;
    """

    data = run_or_load_query(sql_query)

    #some trading days had 0, 1, or NaN, either bugs or not they mess stuff up
    data["total daily trades"].fillna(value=0)
    data = data[data["total daily trades"] > 1]

    data['month'] = data['date'].dt.strftime('%Y-%m')
    data['year'] = data['date'].dt.strftime('%Y')
    data['monthly avg trade'] = data.groupby("month")['total daily trades'].transform('mean')

    # Applying the function to a single column of the DataFrame
    window_size = 1512  # Change this value as needed
    data['trading moving average'] = np.exp(symmetric_smoothing(np.log(data['total daily trades']), window_size, method='average'))
    data['trading moving median'] = symmetric_smoothing(data['total daily trades'], window_size, method='average')

    data = data[data['date'] <= end_date]
    data = data[data['date'] >= start_date].reset_index(drop=True)

    plt.plot(data[['trading moving average']], color='red', label='Front and Back Mean')
    plt.plot(data[['trading moving median']], color='yellow', label='Front and Back Median')

    y_data = data['monthly avg trade']
    y_data_log = np.log(y_data)
    x_data = np.arange(1, y_data.shape[0]+1, 1).reshape(-1,1)

    lin_reg = LinearRegression()
    lin_reg.fit(x_data, y_data_log)
            
    # Generate points for plotting the linear regression line in log scale
    y_model_log = lin_reg.predict(x_data)
    y_model = np.exp(y_model_log)

    plt.scatter(x_data, y_data)
    plt.plot(x_data, y_model, color="black", zorder=2, label='Exponential Model')

    plt.ylabel("# of daily trades")
    plt.xlabel("Day")

    plt.title(STOCK_NAME + " number daily trades" + str(window_size) + " window size ")
    plt.legend()

    normalized_trading = (y_data / y_model).to_numpy()

    trading_moving_average = (data['total daily trades'] / data['trading moving average']).to_numpy()

    trading_moving_median = (data['total daily trades'] / data['trading moving median']).to_numpy()

    normalized_trading_scaled = normalized_trading * normalized_trading.shape[0]/np.sum(normalized_trading)
    normalized_trading_scaled_2 = trading_moving_average * trading_moving_average.shape[0]/np.sum(trading_moving_average)
    normalized_trading_scaled_3 = trading_moving_median * trading_moving_median.shape[0]/np.sum(trading_moving_median)

    """PICK WHICH ONE I WANT -> no comment, exponential, #2, average, #3, median"""
    normalized_trading_scaled = normalized_trading_scaled_2 #mean

    num_events = len(data.groupby('month')['total daily trades'].sum())
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
        #first_date = monthly_data['date'].iloc[0].strftime('%Y-%m-%d')
        #last_date = monthly_data['date'].iloc[-1].strftime('%Y-%m-%d')
        first_date = monthly_data['date'].iloc[0]
        last_date = monthly_data['date'].iloc[-1]

        # Adding the pair to the list
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
