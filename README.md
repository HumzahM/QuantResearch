This is a collection of projects I am doing as Quant Research with professor Dr. Larry Harris at USC 

Helper Libraries contains some functions that I use in every part
Advanced Fetch Stock Data.py pulls the N largest stocks by market cap from start to end year and combines them (Set to 500, 1990, 2019 usually)
Total Market Trades.py calculates market level event months using data from the whole NASDAQ (not just the N stocks from the previous) 

CAPM Event Time Test 1 is a Jupyter Notebook that tests the basic idea of calculating a stock's return based on some kind of normalized num trades (using MSFT and SBUX)

CAPM Event Time WRDS Cloud is attempting to calculate the "price of beta." If the price of beta is positive then that means risk is priced in the market

CAPM Event Time Portfolio replicates the Black Jensen Scholes 1972 portfolio/CAPM study. 
**Right now it demonstrates that there is evidence of less skew/kurtosis among the whole portfolio and much less noise in expected returns **

CAPM Event Time Individual Skew Kurtosis is to test skew and kurtosis on calculating a stock's own "event months" rather then market level event months. Pretty sure this file is messed up right now 
