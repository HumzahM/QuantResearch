import pandas as pd
import numpy as np
from math import log

def calculate_monthly_returns(stock_data, sp500_data, risk_free_rate_data, date_ranges1, date_ranges2, calculate_sp=True):
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

        for i, (start1, end1) in enumerate(date_ranges1):
            if start1 >= first_date_permco and end1 <= last_date_permco:
                filtered_stock_data1 = pd.merge(stock_data_permco[(stock_data_permco['date'] >= start1) & (stock_data_permco['date'] <= end1)], risk_free_rate_data, on='date')
                monthly_return1 = log((1 + filtered_stock_data1['ret'] - filtered_stock_data1['rf']).prod())
                sp500_return1 = log((sp500_data[(sp500_data['date'] >= start1) & (sp500_data['date'] <= end1)]['ret']+1).prod())
                
                row = pd.DataFrame({
                    'sequence #': [i],
                    'type': [1],
                    'permco': [permco], 
                    'equity_returns': [monthly_return1], 
                    'sp500_return': [sp500_return1],
                    'market_cap': [filtered_stock_data1['market_cap'].fillna(1).mean()]
                })
                final_results = pd.concat([final_results, row], ignore_index=True)
        
        for i, (start2, end2) in enumerate(date_ranges2):
            if start2 >= first_date_permco and end2 <= last_date_permco:
                filtered_stock_data2 = pd.merge(stock_data_permco[(stock_data_permco['date'] >= start2) & (stock_data_permco['date'] <= end2)], risk_free_rate_data, on='date')

                monthly_return2 = log((1 + filtered_stock_data2['ret'] - filtered_stock_data2['rf']).prod())
                
                sp500_return2 = log((sp500_data[(sp500_data['date'] >= start2) & (sp500_data['date'] <= end2)]['ret']+1).prod())

                row = pd.DataFrame({
                    'sequence #': [i],
                    'type': [2],
                    'permco': [permco], 
                    'equity_returns': [monthly_return2], 
                    'sp500_return': [sp500_return2],
                    'market_cap': [filtered_stock_data1['market_cap'].fillna(1).mean()]
                })
                final_results = pd.concat([final_results, row], ignore_index=True)

    if(calculate_sp):

        spr_returns = pd.DataFrame()

        #calculate spr returns 
        for i, ((start1, end1), (start2, end2)) in enumerate(zip(date_ranges1, date_ranges2)): 
            sp500_return1 = log((sp500_data[(sp500_data['date'] >= start1) & (sp500_data['date'] <= end1)]['ret']+1).prod())
            sp500_return2 = log((sp500_data[(sp500_data['date'] >= start2) & (sp500_data['date'] <= end2)]['ret']+1).prod())
            row = pd.DataFrame({
                'sequence #': [i], 
                'sp500_return_range1': [sp500_return1],
                'sp500_return_range2': [sp500_return2]
            })
            spr_returns = pd.concat([spr_returns, row], ignore_index=True)
        return final_results, spr_returns
    
    else:
        return final_results