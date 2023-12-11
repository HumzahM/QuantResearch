import total_market_trades
import datetime
import numpy as np
import pandas as pd
import wrds
import datetime
import statsmodels.api as sm
from scipy.stats import ttest_ind
import os
import pickle
import matplotlib.pyplot as plt
from scipy.stats import kurtosis, skew
from scipy.stats import ttest_ind


start_date = '1990-01-01'
end_date = '2021-12-31'

risk_free = pd.read_csv('rf daily rate.csv')
risk_free = risk_free[(risk_free['date'] >= start_date) & (risk_free['date'] <= end_date)]
market_returns = pd.read_csv('msft test data.csv')
market_returns = market_returns[(market_returns['date'] >= start_date) & (market_returns['date'] <= end_date)]
market_returns = market_returns.merge(risk_free, left_on='date', right_on='date')
market_returns['log sp500 returns'] = np.log(1+market_returns['sprtrn'] - market_returns['rf'])

#for beta, y is stock return (dependant) and x is market return (independant)
def calculate_beta(y, x):
    y = np.array(y)
    x = np.array(x)
    covariance = np.cov(y, x)[0, 1]
    variance = np.var(x)
    beta = covariance / variance
    return beta


def fetch_stock_data(start_date, end_date):
    # Define the pickle file path
    pickle_file = 'stock_data.pickle'
    num_stocks = 500 #should be 500 later

    # Prepare the query
    query = f"""
    WITH StockAverageMarketCap AS (
        SELECT 
            permno,
            AVG(ABS(shrout * prc)) as avg_market_cap
        FROM 
            crsp.dsf
        WHERE 
            date >= '{start_date}' AND date <= '{end_date}'
            AND prc IS NOT NULL
            AND shrout IS NOT NULL
        GROUP BY permno
    ),
    Top500Stocks AS (
        SELECT 
            permno
        FROM (
            SELECT 
                permno,
                RANK() OVER (ORDER BY AVG(ABS(shrout * prc)) DESC) as cap_rank
            FROM 
                crsp.dsf
            WHERE 
                date >= '{start_date}' AND date <= '{end_date}'
                AND prc IS NOT NULL
                AND shrout IS NOT NULL
            GROUP BY permno
        ) as Ranked
        WHERE 
            cap_rank <= '{num_stocks}'
    )
    SELECT 
        dsf.date, 
        dsf.permno, 
        dsf.ret,
        dsf.shrout * dsf.prc as market_cap
    FROM 
        crsp.dsf as dsf
    JOIN 
        Top500Stocks ON dsf.permno = Top500Stocks.permno
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
                return saved_data['data']

    # If no saved file or query doesn't match, execute the new query
    db = wrds.Connection(wrds_username='humzahm')
    data = db.raw_sql(query)
    data['ret'] = data['ret'].fillna(0)
    print(data[data.isna().any(axis=1)])
    print(data)
    # Save the new query and data to pickle
    with open(pickle_file, 'wb') as f:
        pickle.dump({'query': query, 'data': data}, f)

    return data


def calculate_monthly_returns(stock_data, sp500_data, risk_free_rate_data, date_ranges1, date_ranges2):
    stock_data['date'] = pd.to_datetime(stock_data['date'])
    sp500_data['date'] = pd.to_datetime(sp500_data['date'])
    risk_free_rate_data['date'] = pd.to_datetime(risk_free_rate_data['date'])

    date_ranges1 = [(pd.to_datetime(start), pd.to_datetime(end)) for start, end in date_ranges1]
    date_ranges2 = [(pd.to_datetime(start), pd.to_datetime(end)) for start, end in date_ranges2]

    final_results = pd.DataFrame()

    unique_permnos = stock_data['permno'].unique()
    counter = 1
    for permno in unique_permnos:
        print(counter)
        counter += 1
        stock_data_permno = stock_data[stock_data['permno'] == permno]

        permno_dates = stock_data_permno['date']
        first_date_permno = permno_dates.min()
        last_date_permno = permno_dates.max()

        for i, ((start1, end1), (start2, end2)) in enumerate(zip(date_ranges1, date_ranges2)):
            if start1 >= first_date_permno and end1 <= last_date_permno and start2 >= first_date_permno and end2 <= last_date_permno:
                filtered_stock_data1 = pd.merge(stock_data_permno[(stock_data_permno['date'] >= start1) & (stock_data_permno['date'] <= end1)], risk_free_rate_data, on='date')
                filtered_stock_data2 = pd.merge(stock_data_permno[(stock_data_permno['date'] >= start2) & (stock_data_permno['date'] <= end2)], risk_free_rate_data, on='date')

                filtered_stock_data1['log equity returns'] = np.log(1 + filtered_stock_data1['ret'] - filtered_stock_data1['rf'])
                filtered_stock_data2['log equity returns'] = np.log(1 + filtered_stock_data2['ret'] - filtered_stock_data2['rf'])

                monthly_return1 = filtered_stock_data1['log equity returns'].sum()
                monthly_return2 = filtered_stock_data2['log equity returns'].sum()

                sp500_return1 = sp500_data[(sp500_data['date'] >= start1) & (sp500_data['date'] <= end1)]['log sp500 returns'].sum()
                sp500_return2 = sp500_data[(sp500_data['date'] >= start2) & (sp500_data['date'] <= end2)]['log sp500 returns'].sum()

                row = pd.DataFrame({
                    'sequence #': [i], 
                    'permno': [permno], 
                    'equity_returns_range1': [monthly_return1], 
                    'equity_returns_range2': [monthly_return2],
                    'sp500_return_range1': [sp500_return1],
                    'sp500_return_range2': [sp500_return2],
                    'market cap': [np.nanmean(filtered_stock_data1['market_cap'])]
                })
                final_results = pd.concat([final_results, row], ignore_index=True)

    return final_results

def calculate_kurtosis_skew(final_results):
    unique_permnos = final_results['permno'].unique()

    # Arrays to hold kurtosis and skew values
    kurtosis_range1 = []
    kurtosis_range2 = []
    skew_range1 = []
    skew_range2 = []

    for permno in unique_permnos:           
        data_permno = final_results[final_results['permno'] == permno]

        # Calculate kurtosis and skew for each range and append to arrays
        kurtosis_range1.append(kurtosis(data_permno['equity_returns_range1'], fisher=True))
        kurtosis_range2.append(kurtosis(data_permno['equity_returns_range2'], fisher=True))
        skew_range1.append(skew(data_permno['equity_returns_range1']))
        skew_range2.append(skew(data_permno['equity_returns_range2']))

    # Calculate averages
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

def calculate_beta(y, x):
    y = np.array(y)
    x = np.array(x)
    covariance_matrix = np.cov(y, x)
    covariance = covariance_matrix[0, 1]
    variance = np.var(x)
    beta = covariance / variance
    return beta

def calculate_weighted_beta(y, x, weights):
    y = np.array(y)
    x = np.array(x)
    weights = np.array(weights)

    # Ensure weights sum to 1
    weights /= weights.sum()

    # Calculate weighted average of y and x
    avg_y = np.average(y, weights=weights)
    avg_x = np.average(x, weights=weights)

    # Calculate weighted covariance and variance
    covariance = np.sum(weights * (y - avg_y) * (x - avg_x))
    variance = np.sum(weights * (x - avg_x)**2)

    if variance == 0:
        return np.nan

    beta = covariance / variance
    return beta


def calculate_betas_and_price_of_beta(monthly_returns):
    unique_permnos = monthly_returns['permno'].unique()

    monthly_returns['beta_range1'] = np.nan
    monthly_returns['beta_range2'] = np.nan
    counter = 1
    for permno in unique_permnos:
        print(counter)
        counter += 1
        data_permno = monthly_returns[monthly_returns['permno'] == permno]

        # Calculate beta for each range
        for i in range(len(data_permno)):
            if i >= 60:
                y_range1 = data_permno['equity_returns_range1'].iloc[i-60:i]
                x_range1 = data_permno['sp500_return_range1'].iloc[i-60:i]
                beta_range1 = calculate_beta(y_range1, x_range1)
                monthly_returns.loc[data_permno.index[i], 'beta_range1'] = beta_range1

                y_range2 = data_permno['equity_returns_range2'].iloc[i-60:i]
                x_range2 = data_permno['sp500_return_range2'].iloc[i-60:i]
                beta_range2 = calculate_beta(y_range2, x_range2)
                monthly_returns.loc[data_permno.index[i], 'beta_range2'] = beta_range2
    
    price_of_beta_range1, price_of_beta_range2 = [], []

    monthly_returns_filtered = monthly_returns[monthly_returns['sequence #'] >= 61]
    monthly_returns_filtered = monthly_returns_filtered.dropna(subset=['beta_range1', 'beta_range2'])
    print(monthly_returns_filtered)
    unique_sequences = monthly_returns_filtered['sequence #'].unique()
    
    for seq in unique_sequences:
        data_seq = monthly_returns[monthly_returns['sequence #'] == seq].dropna(subset=['beta_range1', 'beta_range2'])
        
        # For range 1
        x_range1 = data_seq['beta_range1']
        y_range1 = data_seq['equity_returns_range1']
        weights = data_seq['market cap']
        #price_of_beta_1 = calculate_beta(y_range1, x_range1)
        price_of_beta_1 = calculate_weighted_beta(y_range1, x_range1, weights)
        price_of_beta_range1.append(price_of_beta_1)

        # For range 2
        x_range2 = data_seq['beta_range2']
        y_range2 = data_seq['equity_returns_range2']
        weights = data_seq['market cap']
        #price_of_beta_2 = calculate_beta(y_range2, x_range2)
        price_of_beta_2 = calculate_weighted_beta(y_range2, x_range2, weights)
        price_of_beta_range2.append(price_of_beta_2)

    avg_range1 = sum(price_of_beta_range1) / len(price_of_beta_range1)
    avg_range2 = sum(price_of_beta_range2) / len(price_of_beta_range2)
    
    # Perform t-test on the results of the two methods
    t_stat, p_value = ttest_ind(price_of_beta_range1, price_of_beta_range2)
    _, bins, _, = plt.hist(price_of_beta_range1, bins=30, histtype=u'step', label="Monthly Returns")
    plt.hist(price_of_beta_range2, bins=bins,  histtype=u'step', label="Event Months Returns")
    plt.savefig("histogram.png")
    # Print results
    print(f"Average price of beta for Range 1: {avg_range1}")
    print(f"Average price of beta for Range 2: {avg_range2}")
    print(f"T-statistic for Difference: {t_stat}, P-value for Difference (One Tailed): {p_value/2}")

print("running")

stocks = fetch_stock_data('1990-01-01', '2021-12-30')
print("stocks fetched")

monthly_day_ranges, event_month_ranges = total_market_trades.get_event_month_blocks()
print("ranges calculated")

monthly_returns = calculate_monthly_returns(stocks, market_returns, risk_free, monthly_day_ranges, event_month_ranges)
print("monthly returns calculated")
print(monthly_returns)

calculate_kurtosis_skew(monthly_returns)
calculate_betas_and_price_of_beta(monthly_returns)

