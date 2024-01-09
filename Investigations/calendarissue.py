import sys
from pathlib import Path
import os
import pickle
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
# Add the parent directory to sys.path to allow for package imports
parent_dir = str(Path(__file__).resolve().parent.parent)
sys.path.append(parent_dir)

from Helper_Functions.total_market_trades import get_event_month_blocks
from Helper_Functions.advanced_fetch_stock_data import advanced_fetch_stock_data
import pandas as pd

data_directory = 'data'
os.makedirs(data_directory, exist_ok=True)

# load from /data/1900_500_1990_2019_monthly_returns.pickle
# load from /data/2520_500_1990_2019_monthly_returns.pickle

returns2520 = pickle.load(open(f'{data_directory}/2520_500_1990_2019_monthly_returns.pickle', 'rb'))

event_months, time_months, event_month_lengths, time_month_lengths = get_event_month_blocks(2520, 1990, 2019, return_extra_data=True)

monthly_returns = returns2520[['type', 'sequence #', 'sp500_return']].drop_duplicates().reset_index(drop=True)

monthly_returns['month length'] = monthly_returns.apply(lambda row: time_month_lengths[int(row['sequence #'])] if row['type'] == 1 else event_month_lengths[int(row['sequence #'])], axis=1)
monthly_returns.sort_values(by=['sequence #'], inplace=True)
monthly_returns.reset_index(drop=True, inplace=True)

time_months = monthly_returns[monthly_returns['type'] == 1]
event_months = monthly_returns[monthly_returns['type'] == 2]

plt.close('all')
plt.figure()
plt.scatter(time_months['month length'], time_months['sp500_return'], color='blue', label='Time Months')
plt.scatter(event_months['month length'], event_months['sp500_return'], color='red', label='Event Months')

# Calculate the 90th and 10th percentiles for time_months and event_months
time_months_90th_percentile = time_months['sp500_return'].quantile(0.9)
time_months_10th_percentile = time_months['sp500_return'].quantile(0.1)
event_months_90th_percentile = event_months['sp500_return'].quantile(0.9)
event_months_10th_percentile = event_months['sp500_return'].quantile(0.1)

# Draw horizontal lines for the 90th and 10th percentiles
plt.axhline(y=time_months_90th_percentile, color='black', linestyle='--', label='90th Percentile (Time Months)')
plt.axhline(y=time_months_10th_percentile, color='black', linestyle='--', label='10th Percentile (Time Months)')
plt.axhline(y=event_months_90th_percentile, color='black', linestyle='--', label='90th Percentile (Event Months)')
plt.axhline(y=event_months_10th_percentile, color='black', linestyle='--', label='10th Percentile (Event Months)')

plt.legend()
plt.xlabel('Month Length')
plt.ylabel('S&P 500 Return')
plt.title('S&P 500 Return vs. Month Length')
plt.axhline(y=0, color='black', linestyle='--')
plt.savefig('sp500_return_vs_month_length.png')
plt.figure()
_, bins, _ = plt.hist(event_months['sp500_return'], bins=20, color='red', histtype=u'step')
plt.hist(time_months['sp500_return'], bins=bins, color='blue', histtype=u'step')
plt.legend(['Event Months', 'Time Months'])
plt.xlabel('S&P 500 Return (Log)')
plt.ylabel('Frequency')
plt.savefig('sp500_return_histogram.png')
plt.show()
print(skew(event_months['sp500_return']))
print(kurtosis(event_months['sp500_return']))
print(skew(time_months['sp500_return']))
print(kurtosis(time_months['sp500_return']))