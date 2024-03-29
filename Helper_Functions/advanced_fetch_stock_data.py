import pandas as pd
import pickle
import os
import wrds
#inclusive, inclusive 
def advanced_fetch_stock_data(start_year, end_year, n_stocks):
    pickle_file = f'{n_stocks}_{start_year}_{end_year}_data.pickle'

    if os.path.exists(pickle_file):
         with open(pickle_file, 'rb') as f:
            saved_data = pickle.load(f)
            saved_n_stocks = saved_data['n_stocks']
            saved_start_year = saved_data['start_year']
            saved_end_year = saved_data['end_year']

            # Check if the saved query matches the current query
            if saved_n_stocks == n_stocks and saved_start_year == start_year and saved_end_year == end_year:
                print("Loading data from saved file.")
                print(saved_data['data'])
                return saved_data['data']

    data = pd.DataFrame()
    db = wrds.Connection(wrds_username='humzahm')
    for year in range(start_year, end_year+1):
        print(f'Fetching data for year {year}...')
        start_date = f'{year}-01-01'
        end_date = f'{year}-12-31'
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
                    RANK() OVER (ORDER BY avg_market_cap DESC) as cap_rank
                FROM 
                    StockAverageMarketCap
            ) as Ranked
            WHERE 
                cap_rank <= '{n_stocks}'
        )
        SELECT 
            dsf.date, 
            dsf.permco, 
            dsf.ret,
            dsf.shrout * dsf.prc as market_cap,
            (dsf.ask - dsf.bid) / dsf.prc as percent_spread
        FROM 
            crsp.dsf as dsf
        JOIN 
            Top500Stocks ON dsf.permco = Top500Stocks.permco
        WHERE 
            dsf.date >= '{start_date}' AND dsf.date <= '{end_date}'
"""
        result = db.raw_sql(query)
        data = pd.concat([data, result])
    
    data = data.sort_values(by=['permco', 'date'])
    data = data.reset_index(drop=True)
    data['ret'] = data['ret'].fillna(0)

    with open (pickle_file, 'wb') as f:
        pickle.dump({'data': data, 'n_stocks': n_stocks, 'start_year': start_year, 'end_year': end_year}, f)

    return data

# def unbiased_fetch_stock_data(calendar_months, event_months, n_stocks):
#     pickle_file = f'unbiased_{n_stocks}_{calendar_months[0][0]}_{calendar_months[-1][1]}_data.pickle'

#     if os.path.exists(pickle_file):
#          with open(pickle_file, 'rb') as f:
#             saved_data = pickle.load(f)
#             saved_n_stocks = saved_data['n_stocks']
#             saved_start_day = saved_data['start_day']
#             saved_end_day = saved_data['end_day']

#             # Check if the saved query matches the current query
#             if saved_n_stocks == n_stocks and saved_start_day == calendar_months[0][0] and saved_end_day == calendar_months[-1][1]:
#                 print("Loading data from saved file.")
#                 print(saved_data['calendar_data'])
#                 return saved_data['calendar_data', ]
            
#     calendar_returns = queue_data(calendar_months)
#     event_returns = queue_data(event_months)

#     with open (pickle_file, 'wb') as f:
#         pickle.dump({'data': data, 'n_stocks': n_stocks, 'start_year': start_year, 'end_year': end_year}, f)

#     return calendar_returns, event_returns


# def queue_data(days):
#     data = pd.DataFrame()
#     db = wrds.Connection('humzahm')
        
#      for year in range(start_year, end_year+1):
#         print(f'Fetching data for year {year}...')
#         start_date = f'{year}-01-01'
#         end_date = f'{year}-12-31'
#         query = f"""
#         WITH StockAverageMarketCap AS (
#             SELECT 
#                 permco,
#                 AVG(ABS(shrout * prc)) as avg_market_cap
#             FROM 
#                 crsp.dsf
#             WHERE 
#                 date >= '{start_date}' AND date <= '{end_date}'
#                 AND prc IS NOT NULL
#                 AND shrout IS NOT NULL
#             GROUP BY permco
#         ),
#         Top500Stocks AS (
#             SELECT 
#                 permco
#             FROM (
#                 SELECT 
#                     permco,
#                     RANK() OVER (ORDER BY AVG(ABS(shrout * prc)) DESC) as cap_rank
#                 FROM 
#                     crsp.dsf
#                 WHERE 
#                     date >= '{start_date}' AND date <= '{end_date}'
#                     AND prc IS NOT NULL
#                     AND shrout IS NOT NULL
#                 GROUP BY permco
#             ) as Ranked
#             WHERE 
#                 cap_rank <= '{n_stocks}'
#         )
#         SELECT 
#             dsf.date, 
#             dsf.permco, 
#             dsf.ret,
#             dsf.shrout * dsf.prc as market_cap,
#             (dsf.ask - dsf.bid) / dsf.prc as percent_spread
#         FROM 
#             crsp.dsf as dsf
#         JOIN 
#             Top500Stocks ON dsf.permco = Top500Stocks.permco
#         WHERE 
#             dsf.date >= '{start_date}' AND dsf.date <= '{end_date}'
#         """
#         result = db.raw_sql(query)
#         data = pd.concat([data, result])
    
#     data = data.sort_values(by=['permco', 'date'])
#     data = data.reset_index(drop=True)
#     data['ret'] = data['ret'].fillna(0)