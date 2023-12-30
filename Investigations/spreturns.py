import pickle
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

print("running")

window_size = 2520
start_year = 1990
end_year = 2019
n_stocks = 500

# Rerun flag
rerunMonthlyReturns = True

# Directory for storing pickle files
data_directory = 'data'
os.makedirs(data_directory, exist_ok=True)

monthly_returns_filename = os.path.join(data_directory, f'{window_size}_{n_stocks}_{start_year}_{end_year}_monthly_returns.pickle')
spr_returns_filename = os.path.join(data_directory, f'{window_size}_{n_stocks}_{start_year}_{end_year}_spr_returns.pickle')

with open(monthly_returns_filename, 'rb') as f:
    monthly_returns = pickle.load(f)

with open(spr_returns_filename, 'rb') as f:
    spr_returns = pickle.load(f)

_, bins, _ = plt.hist(spr_returns['sp500_return_range1'], bins=10, histtype=u'step', label="Monthly Returns")
plt.hist(spr_returns['sp500_return_range2'], bins=bins,  histtype=u'step', label="Event Months Returns")
plt.legend()
plt.savefig("spreturns.png")

from scipy.stats import kurtosis, skew
