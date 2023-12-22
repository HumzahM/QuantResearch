#import total_market_trades_old as total_market_trades
import total_market_trades
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
import statsmodels.api as sm


start_date = '1990-01-01'
end_date = '2019-12-31'

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

def calculate_beta_force(y, x):
    return sm.OLS(y, x).fit().params[0]


def fetch_stock_data(start_date, end_date):
    # Define the pickle file path
    pickle_file = 'stock_data.pickle'
    num_stocks = 100 #should be 500 later

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
        dsf.shrout * dsf.prc as market_cap
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

        for i, ((start1, end1), (start2, end2)) in enumerate(zip(date_ranges1, date_ranges2)):
            if start1 >= first_date_permco and end1 <= last_date_permco and start2 >= first_date_permco and end2 <= last_date_permco:
                filtered_stock_data1 = pd.merge(stock_data_permco[(stock_data_permco['date'] >= start1) & (stock_data_permco['date'] <= end1)], risk_free_rate_data, on='date')
                filtered_stock_data2 = pd.merge(stock_data_permco[(stock_data_permco['date'] >= start2) & (stock_data_permco['date'] <= end2)], risk_free_rate_data, on='date')

                filtered_stock_data1['log equity returns'] = np.log(1 + filtered_stock_data1['ret'] - filtered_stock_data1['rf'])
                filtered_stock_data2['log equity returns'] = np.log(1 + filtered_stock_data2['ret'] - filtered_stock_data2['rf'])

                monthly_return1 = filtered_stock_data1['log equity returns'].sum()
                monthly_return2 = filtered_stock_data2['log equity returns'].sum()
                
                row = pd.DataFrame({
                    'sequence #': [i], 
                    'permco': [permco], 
                    'equity_returns_range1': [monthly_return1], 
                    'equity_returns_range2': [monthly_return2],
                    'sp500_return_range1': [sp500_return1],
                    'sp500_return_range2': [sp500_return2],
                    'market_cap': [filtered_stock_data1['market_cap'].fillna(0).mean()]
                })
                final_results = pd.concat([final_results, row], ignore_index=True)

    spr_returns = pd.DataFrame()

    #calculate spr returns 
    for i, ((start1, end1), (start2, end2)) in enumerate(zip(date_ranges1, date_ranges2)): 
        sp500_return1 = sp500_data[(sp500_data['date'] >= start1) & (sp500_data['date'] <= end1)]['log sp500 returns'].sum()
        sp500_return2 = sp500_data[(sp500_data['date'] >= start2) & (sp500_data['date'] <= end2)]['log sp500 returns'].sum()
        row = pd.DataFrame({
            'sequence #': [i], 
            'sp500_return_range1': [sp500_return1],
            'sp500_return_range2': [sp500_return2]
        })
        spr_returns = pd.concat([spr_returns, row], ignore_index=True)
    print(spr_returns)
    return final_results, spr_returns

def calculate_betas_and_price_of_beta(monthly_returns, spr_returns):
    unique_sequences = monthly_returns['sequence #'].unique()
    portfolio_range1 = [[] for _ in range(10)]
    portfolio_range2 = [[] for _ in range(10)]
    spreturns1 = []
    spreturns2 = []
    for seq in unique_sequences:
        if(seq >= 72 and seq % 12 == 0):
            print(seq)
            seq_range = range(seq-71, seq+1)
            data_subset = monthly_returns[monthly_returns['sequence #'].isin(seq_range)]

            # Sort by 'permco'
            sorted_data = data_subset.sort_values(by='permco')
            # Group by 'permco' and filter out any group that doesn't have exactly 72 points
            valid_data = sorted_data.groupby('permco').filter(lambda x: len(x) == 72)
            # Group by 'permco' again and calculate the beta for each group
            betas1 = valid_data.groupby('permco').apply(lambda x: calculate_beta_force(x['equity_returns_range1'][0:60], x['sp500_return_range1'][0:60]))
            returns1 = valid_data.groupby('permco').apply(lambda x: np.sum(x['equity_returns_range1'][60:]))
            betas2 = valid_data.groupby('permco').apply(lambda x: calculate_beta_force(x['equity_returns_range2'][0:60], x['sp500_return_range2'][0:60]))
            returns2 = valid_data.groupby('permco').apply(lambda x: np.sum(x['equity_returns_range2'][60:]))
            market_caps1 = valid_data.groupby('permco').apply(lambda x: np.mean(x['market_cap'][60:]))
            market_caps1.fillna(0, inplace=True)
            market_caps2 = valid_data.groupby('permco').apply(lambda x: np.mean(x['market_cap'][60:]))
            market_caps2.fillna(0, inplace=True)

            # Make 10 groups based on 10 betas

            betas1_grouped = pd.qcut(betas1, 10, labels=False)
            betas2_grouped = pd.qcut(betas2, 10, labels=False)
            for i in range(10):
                group1_indices = betas1_grouped[betas1_grouped == i].index
                group2_indices = betas2_grouped[betas2_grouped == i].index

                group1_indices = betas1_grouped[betas1_grouped == i].index
                group2_indices = betas2_grouped[betas2_grouped == i].index
                    
                #portfolio_range1[i].append(np.average(returns1[group1_indices], weights=market_caps1[group1_indices]))
                #portfolio_range2[i].append(np.average(returns2[group2_indices], weights=market_caps2[group2_indices]))
                portfolio_range1[i].append(np.average(returns1[group1_indices]))
                portfolio_range2[i].append(np.average(returns2[group2_indices]))

            relevant_sp_data = spr_returns[spr_returns['sequence #'].isin(range(seq-11,seq+1))]
            sp500ret1 = np.sum(relevant_sp_data["sp500_return_range1"])
            #convert sp500ret1 to simple return from log return
            sp500ret2 = np.sum(relevant_sp_data["sp500_return_range2"])
            spreturns1.append(sp500ret1)
            spreturns2.append(sp500ret2)

            # sp500ret1 = np.exp(sp500ret1) - 1
            # sp500ret2 = np.exp(sp500ret2) - 1

            # print("risk adjusted return of sp500 is: " + str(seq/12) + " " + str(sp500ret1))
            # print("risk adjusted return of sp500 (event year) is: " + str(sp500ret2))

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

    plt.scatter(betas1, means1, color='blue')
    plt.scatter(betas2, means2, color='red')
    plt.scatter(betas3, means3, color='green')
    plt.xlabel('Beta')
    plt.ylabel('Mean Return')
    plt.title('Beta vs Mean Return (Blue is Normal Months, Red is Event Months)')
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

    print(calculate_beta(means1, betas1))
    print(calculate_beta(means2, betas2))

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

rerunMonthlyReturns = False

print("running")

if rerunMonthlyReturns:
    event_month_ranges, monthly_day_ranges = total_market_trades.get_event_month_blocks(window_size=150)
    print("ranges calculated")

    stocks = fetch_stock_data('1990-01-01', '2021-12-30')
    print("stocks fetched")

    monthly_returns, spr_returns = calculate_monthly_returns(stocks, market_returns, risk_free, monthly_day_ranges, event_month_ranges)
    print("monthly returns calculated")

    with open('final_results.pickle', 'wb') as f:
        pickle.dump(monthly_returns, f)

    # Save spr_returns to a pickle file
    with open('spr_returns.pickle', 'wb') as f:
        pickle.dump(spr_returns, f)

else:
    # Load monthly_returns from a pickle file
    with open('final_results.pickle', 'rb') as f:
        monthly_returns = pickle.load(f)

    # Load spr_returns from a pickle file
    with open('spr_returns.pickle', 'rb') as f:
        spr_returns = pickle.load(f)

calculate_betas_and_price_of_beta(monthly_returns, spr_returns)

