import pickle
import os
import pandas as pd
import numpy as np

print("running")

window_size = 2520
start_year = 1990
end_year = 2019
n_stocks = 500

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
from Return_Dispersion.return_dispersion import get_event_blocks_return_dispersion

start_date = '1990-01-01'
end_date = '2019-12-31'

event_months, months = get_event_month_blocks(window_size)

def get_dates(sequence_num, type_num):
    # Select the correct date range array based on type
    date_range = months if type_num == 1 else event_months
    sequence_num = int(sequence_num)
    # Fetch the date range for the given sequence number
    start_date, end_date = date_range[sequence_num]

    return start_date, end_date


# Assuming your dataframe is named monthly_returns
monthly_returns['start_date'], monthly_returns['end_date'] = zip(*monthly_returns.apply(lambda row: get_dates(row['sequence #'], row['type']), axis=1))

first_analysis_date = '2000-01-01'
last_analysis_date = '2002-12-31'

monthly_returns = monthly_returns[(monthly_returns['start_date'] >= first_analysis_date) & (monthly_returns['end_date'] <= last_analysis_date)]

# Selecting only the required columns
selected_data = monthly_returns[['type', 'sequence #', 'sp500_return', 'start_date', 'end_date']]

# Dropping duplicate rows
unique_data = selected_data.drop_duplicates()
#unique_data.sort_values(by='sp500_return', inplace=True)

# Now, you can save this filtered and unique data to a new file
_, bins, _ = plt.hist(unique_data[unique_data['type'] == 2]['sp500_return'], bins=10, histtype=u'step', label="Event Months")
plt.hist(unique_data[unique_data['type'] == 1]['sp500_return'], bins=bins,  histtype=u'step', label="Time Months")
plt.legend()
plt.ylabel("# Months")
plt.xlabel("Return (Log)")
#plt.title("Returns 2000-2002 (Dotcom Bubble)")
plt.title(f"Recession (2008-2009) \n {len(unique_data[unique_data['type'] == 1])} months and {len(unique_data[unique_data['type'] == 2])} event months")
plt.show()




