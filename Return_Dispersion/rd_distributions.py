import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

window_size = 2520
start_year = 1990
end_year = 2019
n_stocks = 500
start_date = f'{start_year}-01-01'
end_date = f'{end_year}-12-31'

data_directory = 'data'
os.makedirs(data_directory, exist_ok=True)

# Construct file paths
monthly_returns_filename = os.path.join(data_directory, f'{window_size}_{n_stocks}_{start_year}_{end_year}_monthly_returns.pickle')

with open(monthly_returns_filename, 'rb') as f:
    monthly_returns = pickle.load(f)

print(monthly_returns)

#Not adjusted for bid ask spreads (yet)

def month_return_dispersion(returns, market_caps):
    # Calculate the weighted mean return
    total_market_cap = np.sum(market_caps)
    weights = market_caps / total_market_cap
    weighted_mean_return = np.sum(weights * returns)

    # Calculate the weighted standard deviation (equity return dispersion)
    squared_diff = (returns - weighted_mean_return) ** 2
    weighted_squared_diff = weights * squared_diff
    weighted_variance = np.sum(weighted_squared_diff)
    #equity_return_dispersion = np.sqrt(weighted_variance)
    equity_return_dispersion = weighted_variance

    final = equity_return_dispersion

    return final

calendar_dispersion = []
event_dispersion = []

for seq in monthly_returns['sequence #'].unique():
    data_calendar = monthly_returns[(monthly_returns['sequence #'] == seq) & (monthly_returns['type'] == 1)]
    data_event = monthly_returns[(monthly_returns['sequence #'] == seq) & (monthly_returns['type'] == 2)]

    calendar_dispersion.append(month_return_dispersion(data_calendar['equity_returns'], data_calendar['market_cap']))
    event_dispersion.append(month_return_dispersion(data_event['equity_returns'], data_event['market_cap']))

_, bins, _ = plt.hist(calendar_dispersion, bins=25, label='Calendar', histtype=u'step')
plt.hist(event_dispersion, bins=bins, label='Event', histtype=u'step')
plt.legend()
plt.xlabel('Dispersion')
plt.ylabel('Frequency')
plt.title('Return Dispersion')
plt.savefig("dispersion.png")
print("Means: (Calendar then Event)")
print(np.mean(calendar_dispersion))
print(np.mean(event_dispersion))
print("Standard Deviations: (Calendar then Event)")
print(np.std(calendar_dispersion))
print(np.std(event_dispersion))

def autocorrelation(time_series, N):
    """
    Calculate the average autocorrelation for the last N points in a time series.

    Args:
    time_series (list or numpy array): The time series data.
    N (int): The number of last points to consider for autocorrelation.

    Returns:
    float: The average autocorrelation value.
    """
    n = len(time_series)
    autocorrelations = []
    
    for lag in range(1, N + 1):
        if n - lag < 1:  # Check if lag is too large for time series
            break

        # Calculate autocorrelation for the given lag
        autocorr = np.corrcoef(time_series[:-lag], time_series[lag:])[0, 1]
        autocorrelations.append(autocorr)

    # Return the average autocorrelation
    return np.nanmean(autocorrelations)

N = 3 # for last 3 points, change this number as needed

average_autocorr_calendar = autocorrelation(calendar_dispersion, N)
average_autocorr_event = autocorrelation(event_dispersion, N)

print(f"Average Autocorrelation for Calendar Dispersion for {N} months:", average_autocorr_calendar)
print(f"Average Autocorrelation for Event Dispersion for {N} months:", average_autocorr_event)

max_N = 12

calendar_autocorrs = [autocorrelation(calendar_dispersion, N) for N in range(1, max_N + 1)]
event_autocorrs = [autocorrelation(event_dispersion, N) for N in range(1, max_N + 1)]

plt.figure()

plt.plot(range(1, max_N + 1), calendar_autocorrs, label='Calendar Dispersion')
plt.plot(range(1, max_N + 1), event_autocorrs, label='Event Dispersion')

plt.xlabel('N (Number of Lags)')
plt.ylabel('Average Autocorrelation')
plt.title('Autocorrelation vs. N for Calendar and Event Dispersion')
plt.legend()
plt.savefig("autocorrelation vs n.png")

import numpy as np
import pandas as pd
import statsmodels.api as sm

def regression_with_lags_and_r2(data, N):
    """
    Perform a linear regression on time series data with N lags.

    Args:
    data (list or numpy array): The time series data.
    N (int): The number of lags to include in the regression.

    Returns:
    list: Coefficients of the regression model (constant and betas).
    """
    # Create a DataFrame from the data
    df = pd.DataFrame(data, columns=['ReturnDispersion'])
    
    # Generate lagged data
    for lag in range(1, N + 1):
        df[f'lag_{lag}'] = df['ReturnDispersion'].shift(lag)

    # Drop rows with NaN values (due to lagging)
    df = df.dropna()

    # Define the independent variables (lags) and dependent variable
    X = df.iloc[:, 1:]  # Independent variables (lagged data)
    y = df['ReturnDispersion']  # Dependent variable

    # Add a constant to the model (for the intercept)
    X = sm.add_constant(X)

    # Fit the regression model
    model = sm.OLS(y, X).fit()

    # Return the model coefficients
    return model.params, model.rsquared

N = 3  # Number of lags, change this number as needed

coefficients_calendar, r2_calendar = regression_with_lags_and_r2(calendar_dispersion, N)
coefficients_event, r2_event = regression_with_lags_and_r2(event_dispersion, N)

print("Coefficients for Calendar Dispersion:", coefficients_calendar)
print("R2 Score for Calendar Dispersion:", r2_calendar)
print("Coefficients for Event Dispersion:", coefficients_event)
print("R2 Score for Event Dispersion:", r2_event)

    
