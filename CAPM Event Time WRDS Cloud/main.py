#import total_market_trades_old as total_market_trades
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
import warnings
warnings.filterwarnings('ignore')


start_date = '1990-01-01'
end_date = '2019-12-31'

risk_free = pd.read_csv('rf daily rate.csv')
risk_free = risk_free[(risk_free['date'] >= start_date) & (risk_free['date'] <= end_date)]
market_returns = pd.read_csv('msft test data.csv')
market_returns = market_returns[(market_returns['date'] >= start_date) & (market_returns['date'] <= end_date)]
market_returns = market_returns.merge(risk_free, left_on='date', right_on='date')
market_returns['log sp500 returns'] = np.log(1+market_returns['sprtrn'] - market_returns['rf'])

def fetch_stock_data(start_date, end_date):
    # Define the pickle file path
    pickle_file = 'stock_data.pickle'
    num_stocks = 25 #should be 500 later

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
    TopNStocks AS (
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
        dsf.ret
    FROM 
        crsp.dsf as dsf
    JOIN 
        TopNStocks ON dsf.permco = TopNStocks.permco
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


def calculate_monthly_returns(stock_data, sp500_data, risk_free_rate_data, date_ranges1, date_ranges2):
    stock_data['date'] = pd.to_datetime(stock_data['date'])
    sp500_data['date'] = pd.to_datetime(sp500_data['date'])
    risk_free_rate_data['date'] = pd.to_datetime(risk_free_rate_data['date'])

    date_ranges1 = [(pd.to_datetime(start), pd.to_datetime(end)) for start, end in date_ranges1]
    date_ranges2 = [(pd.to_datetime(start), pd.to_datetime(end)) for start, end in date_ranges2]

    final_results = pd.DataFrame()

    unique_permcos = stock_data['permco'].unique()
    counter = 1
    for permco in unique_permcos:
        print(counter)
        counter += 1
        stock_data_permco = stock_data[stock_data['permco'] == permco]

        permco_dates = stock_data_permco['date']
        #print(permco_dates)
        first_date_permco = permco_dates.min()
        last_date_permco = permco_dates.max()

        for i, (start1, end1) in enumerate(date_ranges1):
            if start1 >= first_date_permco and end1 <= last_date_permco:
                filtered_stock_data1 = pd.merge(stock_data_permco[(stock_data_permco['date'] >= start1) & (stock_data_permco['date'] <= end1)], risk_free_rate_data, on='date')

                filtered_stock_data1['log equity returns'] = np.log(1 + filtered_stock_data1['ret'] - filtered_stock_data1['rf'])

                monthly_return1 = filtered_stock_data1['log equity returns'].sum()

                sp500_return1 = sp500_data[(sp500_data['date'] >= start1) & (sp500_data['date'] <= end1)]['log sp500 returns'].sum()

                row = pd.DataFrame({
                    'sequence #': [i],
                    'type': [1],
                    'permco': [permco], 
                    'equity_returns': [monthly_return1], 
                    'sp500_return': [sp500_return1]
                })
                final_results = pd.concat([final_results, row], ignore_index=True)
        
        for i, (start2, end2) in enumerate(date_ranges2):
            if start2 >= first_date_permco and end2 <= last_date_permco:
                filtered_stock_data2 = pd.merge(stock_data_permco[(stock_data_permco['date'] >= start2) & (stock_data_permco['date'] <= end2)], risk_free_rate_data, on='date')

                filtered_stock_data2['log equity returns'] = np.log(1 + filtered_stock_data2['ret'] - filtered_stock_data2['rf'])

                monthly_return2 = filtered_stock_data2['log equity returns'].sum()

                sp500_return2 = sp500_data[(sp500_data['date'] >= start2) & (sp500_data['date'] <= end2)]['log sp500 returns'].sum()

                row = pd.DataFrame({
                    'sequence #': [i],
                    'type': [2],
                    'permco': [permco], 
                    'equity_returns': [monthly_return2], 
                    'sp500_return': [sp500_return2]
                })
                final_results = pd.concat([final_results, row], ignore_index=True)

    return final_results

def calculate_kurtosis_skew(final_results):
    unique_permcos = final_results['permco'].unique()

    # Arrays to hold kurtosis and skew values
    kurtosis_range1 = []
    kurtosis_range2 = []
    skew_range1 = []
    skew_range2 = []

    for permco in unique_permcos:           
        data_permco = final_results[final_results['permco'] == permco]
        data_range1 = data_permco[data_permco['type'] == 1]
        data_range2 = data_permco[data_permco['type'] == 2]

        # Calculate kurtosis and skew for each range and append to arrays
        kurtosis_range1.append(kurtosis(data_range1['equity_returns'], fisher=True))
        kurtosis_range2.append(kurtosis(data_range2['equity_returns'], fisher=True))
        skew_range1.append(skew(data_range1['equity_returns']))
        skew_range2.append(skew(data_range2['equity_returns']))

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

def calculate_stock_beta(y, x):
    return sm.OLS(y, x).fit().params[0]

def calculate_betas_and_price_of_beta(monthly_returns):
    unique_permcos = monthly_returns['permco'].unique()

    monthly_returns['beta'] = np.nan
    counter = 1
    
    for permco in unique_permcos:
        print(counter)
        counter += 1
        data_permco = monthly_returns[monthly_returns['permco'] == permco]
        data_range1 = data_permco[data_permco['type'] == 1]
        data_range2 = data_permco[data_permco['type'] == 2]
        # Calculate beta for each range
        #print("range: " + str(len(data_permco)))
        for i in range(len(data_range1)):
            if i >= 60:
                y_range1 = data_range1['equity_returns'].iloc[i-60:i]
                x_range1 = data_range1['sp500_return'].iloc[i-60:i]
                beta_range1 = calculate_stock_beta(y_range1, x_range1)
                monthly_returns.loc[data_range1.index[i], 'beta'] = beta_range1
        for i in range(len(data_range2)):
            if i >= 60:
                y_range2 = data_range2['equity_returns'].iloc[i-60:i]
                x_range2 = data_range2['sp500_return'].iloc[i-60:i]
                beta_range2 = calculate_stock_beta(y_range2, x_range2)
                monthly_returns.loc[data_range2.index[i], 'beta'] = beta_range2

    price_of_beta1, price_of_beta2 = [], []
    monthly_returns_filtered = monthly_returns[monthly_returns['beta'].notna()]
    unique_sequences = monthly_returns_filtered['sequence #'].unique()

    avg_beta = monthly_returns[monthly_returns['type'] == 1]['beta'].mean()
    print("Avg Betas: ")
    print(avg_beta)
    avg_beta = monthly_returns[monthly_returns['type'] == 2]['beta'].mean()
    print(avg_beta)

    plt.figure()
    plt.scatter(monthly_returns[monthly_returns['type'] == 1]['beta'], monthly_returns[monthly_returns['type'] == 1]['equity_returns'], label="Monthly Returns")
    plt.scatter(monthly_returns[monthly_returns['type'] == 2]['beta'], monthly_returns[monthly_returns['type'] == 2]['equity_returns'], label="Event Months Returns")
    plt.legend()
    plt.savefig("scatter.png")

    for seq in unique_sequences:
        data_seq = monthly_returns_filtered[monthly_returns_filtered['sequence #'] == seq]
        data_seq1 = data_seq[data_seq['type'] == 1]
        data_seq2 = data_seq[data_seq['type'] == 2]
        price_of_beta1.append(calculate_beta(data_seq1['equity_returns'], data_seq1['beta']))
        price_of_beta2.append(calculate_beta(data_seq2['equity_returns'], data_seq2['beta']))

    price_of_beta1 = np.array(price_of_beta1)
    price_of_beta2 = np.array(price_of_beta2)

    price_of_beta1 = price_of_beta1[~np.isnan(price_of_beta1)]
    price_of_beta2 = price_of_beta2[~np.isnan(price_of_beta2)]

    plt.figure()
    _, bins, _ = plt.hist(price_of_beta1, bins=20, histtype=u'step', label="Monthly Returns")
    plt.hist(price_of_beta2, bins=bins,  histtype=u'step', label="Event Months Returns")
    plt.legend()
    plt.savefig("histogram.png")
    
    # Perform t-test on the results of the two methods
    t_stat, p_value = ttest_ind(price_of_beta1, price_of_beta2)
    # Print results
    print(f"Average price of beta for Range 1: {price_of_beta1.mean()}")
    print(f"Average price of beta for Range 2: {price_of_beta2.mean()}")
    print(f"T-statistic for Difference: {t_stat}, P-value for Difference (One Tailed): {p_value/2}")

print("running")

rerunMonthlyReturns = False

if rerunMonthlyReturns:
    event_month_ranges, monthly_day_ranges = total_market_trades.get_event_month_blocks()
    print("ranges calculated")

    stocks = fetch_stock_data('1990-01-01', '2019-12-31')
    print("stocks fetched")

    monthly_returns = calculate_monthly_returns(stocks, market_returns, risk_free, monthly_day_ranges, event_month_ranges)
    print("monthly returns calculated")

    with open('final_results.pickle', 'wb') as f:
        pickle.dump(monthly_returns, f)


else:
    # Load monthly_returns from a pickle file
    with open('final_results.pickle', 'rb') as f:
        monthly_returns = pickle.load(f)

calculate_kurtosis_skew(monthly_returns)

calculate_betas_and_price_of_beta(monthly_returns)

