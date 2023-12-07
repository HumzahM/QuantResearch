import total_market_trades
import datetime
import numpy as np
import pandas as pd
import wrds
import datetime
import statsmodels.api as sm
from scipy.stats import ttest_ind

start_date = '1990-01-01'
end_date = '2021-12-31'
trading_dates = []

dates_risk_free = pd.read_csv('rf daily rate.csv')
dates_market_returns = pd.read_csv('sprtrn daily returns.csv')

market_event_months, market_time_months, trading_dates = total_market_trades.get_event_month_blocks()

def fetch_stock_data(start_date, end_date, dates_risk_free):
    query = f"""
    SELECT 
        date, 
        permno, 
        ret,
        ABS(shrout * prc) as market_cap
    FROM 
        crsp.dsf
    WHERE 
        date >= '{start_date}' AND date <= '{end_date}'
        AND permno IN (SELECT permno FROM crsp.msenames WHERE ncusip IS NOT NULL AND namedt <= date AND nameendt >= date)
    """

    data = db.raw_sql(query)
    data = data.merge(dates_risk_free, left_on='date', right_on='date')
    data["log equity returns"] = np.log(1+data['RET'] - data['rf'])
    #data = data.dropna()
    return data

#for beta, y is stock return (dependant) and x is market return (independant)
def calculate_beta(y, x):
    y = np.array(y)
    x = np.array(x)
    covariance = np.cov(y, x)[0, 1]
    variance = np.var(x)
    beta = covariance / variance
    return beta

def calculate_weighted_price_of_beta(monthly_returns_data, market_returns_data):
    # Calculate the beta for each stock
    monthly_returns_data['beta'] = monthly_returns_data.groupby('permno').apply(lambda x: calculate_beta(x['log equity returns'], market_returns_data['vwretd']))
    #monthly_returns_data = monthly_returns_data.dropna()
    
    #Now regress betas against returns
    
    return 0 
    
db = wrds.Connection(wrds_username='humzahm')
fetch_stock_data('1990-01-01', '1990-01-31', dates_risk_free)

monthly_price_of_beta = []
event_month_price_of_beta = []

for start_index, end_index in market_time_months:
    start_date = trading_dates[start_index]
    print("start date: ")
    print(start_date)
    end_date = trading_dates[end_index]

    rf_data = dates_risk_free[(dates_risk_free['date'] >= start_date) & (dates_risk_free['date'] <= end_date)]
    stock_returns_data = fetch_stock_data(start_date, end_date, rf_data)
    market_returns_data = dates_market_returns[(dates_market_returns['date'] >= start_date) & (dates_market_returns['date'] <= end_date)]

    weighted_price_of_beta = calculate_weighted_price_of_beta(stock_returns_data, market_returns_data)
    monthly_price_of_beta.append(weighted_price_of_beta)

for start_index, end_index in market_event_months:
    start_date = trading_dates[start_index]
    print("start date: ")
    print(start_date)
    end_date = trading_dates[end_index]


# Perform t-test on the results of the two methods
t_stat, p_value = ttest_ind(monthly_price_of_beta, event_month_price_of_beta)

# Print t-test results
print(f"T-statistic: {t_stat}, P-value: {p_value}")

