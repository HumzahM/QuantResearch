import pandas as pd
from numpy import log
from tqdm import tqdm  # for progress bar

def better_calculate_monthly_returns(stock_data, sp500_data, risk_free_rate_data, date_ranges1, date_ranges2, calculate_sp=True):
    # Convert dates to datetime objects and sort data
    stock_data['date'] = pd.to_datetime(stock_data['date'])
    sp500_data['date'] = pd.to_datetime(sp500_data['date'])
    risk_free_rate_data['date'] = pd.to_datetime(risk_free_rate_data['date'])

    stock_data.sort_values(by=['permco', 'date'], inplace=True)
    sp500_data.sort_values(by='date', inplace=True)
    risk_free_rate_data.sort_values(by='date', inplace=True)

    date_ranges1 = [(pd.to_datetime(start), pd.to_datetime(end)) for start, end in date_ranges1]
    date_ranges2 = [(pd.to_datetime(start), pd.to_datetime(end)) for start, end in date_ranges2]

    final_results_list = []
    spr_returns_list = []

    unique_permcos = stock_data['permco'].unique()

    for permco in tqdm(unique_permcos, desc="Processing companies"):  # Progress bar
        stock_data_permco = stock_data[stock_data['permco'] == permco]

        # Identify continuous blocks
        continuous_blocks = []
        current_block = []
        last_date = None

        for date in stock_data_permco['date']:
            if last_date is None or (date - last_date).days <= 31:
                current_block.append(date)
            else:
                continuous_blocks.append((min(current_block), max(current_block)))
                current_block = [date]
            last_date = date

        if current_block:
            continuous_blocks.append((min(current_block), max(current_block)))

        # Calculate returns for each block within date ranges
        for block_start, block_end in continuous_blocks:
            for i, date_range in enumerate(date_ranges1 + date_ranges2):
                type_val = 1 if i < len(date_ranges1) else 2
                start, end = date_range

                if start >= block_start and end <= block_end:
                    filtered_stock_data = pd.merge(stock_data_permco[(stock_data_permco['date'] >= start) & (stock_data_permco['date'] <= end)], risk_free_rate_data, on='date', how='left')
                    monthly_return = log((1 + filtered_stock_data['ret'] - filtered_stock_data['rf']).prod())
                    sp500_return = log((sp500_data[(sp500_data['date'] >= start) & (sp500_data['date'] <= end)]['ret']+1).prod())
                    
                    row = {
                        'sequence #': i if i < len(date_ranges1) else i - len(date_ranges1),
                        'type': type_val,
                        'permco': permco, 
                        'equity_returns': monthly_return, 
                        'sp500_return': sp500_return,
                        'market_cap': filtered_stock_data['market_cap'].fillna(1).mean()
                    }
                    final_results_list.append(row)

    final_results = pd.DataFrame(final_results_list)

    if calculate_sp:
        print("fix this")
        for i, date_range in enumerate(date_ranges1 + date_ranges2):
            start, end = date_range
            sp500_return = log((sp500_data[(sp500_data['date'] >= start) & (sp500_data['date'] <= end)]['ret']+1).prod())
            
            row = {
                'sequence #': i if i < len(date_ranges1) else i - len(date_ranges1),
                'sp500_return': sp500_return
            }
            spr_returns_list.append(row)

        spr_returns = pd.DataFrame(spr_returns_list)
        return final_results, spr_returns
    else:
        return final_results
