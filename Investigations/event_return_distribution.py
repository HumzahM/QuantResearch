import pickle
import os
import pandas as pd
import numpy as np

print("running")

window_size = 2520
start_year = 1990
end_year = 2019
n_stocks = 500
start_date = f'{start_year}-01-01'
end_date = f'{end_year}-12-31'

# Directory for storing pickle files
data_directory = 'data'
os.makedirs(data_directory, exist_ok=True)

monthly_returns_filename = os.path.join(data_directory, f'{window_size}_{n_stocks}_{start_year}_{end_year}_monthly_returns.pickle')
spr_returns_filename = os.path.join(data_directory, f'{window_size}_{n_stocks}_{start_year}_{end_year}_spr_returns.pickle')

with open(monthly_returns_filename, 'rb') as f:
    monthly_returns = pickle.load(f)

with open(spr_returns_filename, 'rb') as f:
    spr_returns = pickle.load(f)

import sys
from pathlib import Path

#Add the parent directory to sys.path to allow for package imports
parent_dir = str(Path(__file__).resolve().parent.parent)
sys.path.append(parent_dir)

from Helper_Functions.total_market_trades import *
from Helper_Functions.advanced_fetch_stock_data import advanced_fetch_stock_data
from Helper_Functions.calculate_monthly_returns import calculate_monthly_returns

"""Make sure that the configuration (+- edges) matches the one used to generate the data"""

# first_analysis_date = '2000-01-01'
# last_analysis_date = '2002-06-31'

# Name = "Dotcom"

event_months, months = get_event_month_blocks(window_size, start_year, end_year)

def get_dates(sequence_num, type_num):
    # Select the correct date range array based on type
    date_range = months if type_num == 1 else event_months
    sequence_num = int(sequence_num)
    # Fetch the date range for the given sequence number
    start_date, end_date = date_range[sequence_num]

    return start_date, end_date

# Assuming your dataframe is named monthly_returns
monthly_returns['start_date'], monthly_returns['end_date'] = zip(*monthly_returns.apply(lambda row: get_dates(row['sequence #'], row['type']), axis=1))

#monthly_returns = monthly_returns[(monthly_returns['start_date'] >= first_analysis_date) & (monthly_returns['end_date'] <= last_analysis_date)]

# Selecting only the required columns
unique_data = monthly_returns[['type', 'sequence #', 'sp500_return', 'start_date', 'end_date']].drop_duplicates()

# Save unique_data as a CSV file
unique_data.to_csv('unique_data1000.csv', index=False)

# _, bins, _ = plt.hist(unique_data[unique_data['type'] == 2]['sp500_return'], bins=10, histtype=u'step', label="Event Months", density=True)
# plt.hist(unique_data[unique_data['type'] == 1]['sp500_return'], bins=bins,  histtype=u'step', label="Time Months", density=True)
# plt.legend()
# plt.ylabel("# Months (Adjusted)")

# plt.xlabel("Return (Log)")
# plt.title(f"{Name} {start_date}->{end_date} \n {len(unique_data[unique_data['type'] == 1])} months and {len(unique_data[unique_data['type'] == 2])} event months")
# plt.savefig(f"{Name}.png")






