import requests
from datetime import datetime, timedelta
import re

# Function to get the close price for a given option name and date
def get_close_price(option_name, date, api_key):
    url = f"https://api.polygon.io/v1/open-close/O:{option_name}/{date}?adjusted=true&apiKey={api_key}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return data.get('close')
    else:
        print(f"Error fetching data: {response.text}")
        return None

# Function to get the close price for a given stock ticker and date
def get_stock_price(ticker, date, api_key):
    url = f"https://api.polygon.io/v1/open-close/{ticker}/{date}?adjusted=true&apiKey={api_key}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return data.get('close')
    else:
        print(f"Error fetching data: {response.text}")
        return None

# Your API key
api_key = 'FD6_gTCE8RWdLi3Z9DdLyxd5ebfjeWBG'
today = datetime.now()

putInOptionOrAskQuestions = input("Would you like to input an option or ask questions? (I/A): ").strip().upper()
if putInOptionOrAskQuestions == "I":
    option_name = input("Enter the option name: ").upper()
    strike_day = re.search(r'[A-Za-z]+(\d{6})', option_name).group(1)
elif putInOptionOrAskQuestions == "A":
    ticker_name = input("Enter the name of the ticker: ").upper()
    option_type = input("Is this a call or a put? (C/P): ").strip().upper()
    strike_day = input("Enter the strike day of the option (YYMMDD): ")
    strike_price_input = float(input("Enter the strike price (e.g., 14 for $14): "))
    strike_price = f"{int(strike_price_input * 1000):08d}"
    option_name = f"{ticker_name}{strike_day}{option_type}{strike_price}"

# User inputs
daysAgoOrDate = input("Would you like to input the days ago or the date? (D/A): ").strip().upper()
if daysAgoOrDate == "D":
    days_ago_for_api = int(input("How many days ago was this posted on WSB?"))
    query_date = (today - timedelta(days=days_ago_for_api)).strftime('%Y-%m-%d')
elif daysAgoOrDate == "A":
    query_date = input("Enter the date (YYYY-MM-DD): ")

# Get the close price for the query date
close_price_query_date = get_close_price(option_name, query_date, api_key)

# Get the close price for the strike day (YYYY-MM-DD format needed)
close_price_strike_date = get_close_price(option_name, f"20{strike_day[:2]}-{strike_day[2:4]}-{strike_day[4:]}", api_key)

# Get the stock price for the query date
stock_price_query_date = get_stock_price(ticker_name, query_date, api_key)

# Get the stock price for the strike day
stock_price_strike_date = get_stock_price(ticker_name, f"20{strike_day[:2]}-{strike_day[2:4]}-{strike_day[4:]}", api_key)

# Compare and print the result
if close_price_query_date and close_price_strike_date and stock_price_query_date and stock_price_strike_date:
    difference = 100 * close_price_strike_date / close_price_query_date 
    print("This was likely bought for " + str(close_price_query_date) + " on " + query_date + ".")
    print("This was worth " + str(close_price_strike_date) + " on the strike day.")
    print("This option was worth", difference, "% of its bought value on the strike day.")
    print("The stock price on the day it was bought was", stock_price_query_date)
    print("The stock price on the day the option closed was", stock_price_strike_date)
else:
    print("Could not retrieve one or more close prices or stock prices.")
