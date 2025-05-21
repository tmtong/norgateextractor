import logging
import norgatedata
import multiprocessing as mp
import pandas as pd
from multiprocessing import Pool, cpu_count
from functools import partial
import os
import re
def get_all_market_symbols():
    print("Retrieving all symbols from US Equities and US Equities Delisted...")
    
    us_equities = norgatedata.database_symbols('US Equities')
    us_delisted = norgatedata.database_symbols('US Equities Delisted')
    
    print(f"Retrieved {len(us_equities)} active symbols and {len(us_delisted)} delisted symbols.")
    return us_equities, us_delisted




def process_symbol(inputs):
    symbol = inputs[0]
    indexname = inputs[1]
    print('Processing ' + symbol)
    try:
        df = norgatedata.index_constituent_timeseries(
            symbol,
            indexname,
            timeseriesformat="pandas-dataframe"
        )
    except Exception as e:
        # print(f"Error fetching data for {symbol}: {e}")
        return []

    if df.empty:
        return []

    df_filtered = df[df['Index Constituent'] == 1]
    dates = df_filtered.index.get_level_values(0).strftime('%Y-%m-%d').tolist()
    return [(date, symbol) for date in dates]


def build_constituents(all_symbols, indexname):

    dirpath = 'data/index/'
    indexfilename = indexname
    indexfilename = indexfilename.replace(' ', '')
    indexfilename = re.sub(r'[^a-zA-Z0-9]', '', indexfilename)
    indexfilename = indexfilename + '.csv'
    indexfilename = dirpath  + indexfilename
    if os.path.isfile(indexfilename):
        return

    inputs = [[symbol, indexname] for symbol in all_symbols]


    print(f"Processing {len(all_symbols)} symbols using multiprocessing...")

    # Use all available CPU cores minus one to avoid overloading the system
    num_processes = 8
    print(f"Using {num_processes} processes.")

    with Pool(num_processes) as pool:
        results = pool.map(partial(process_symbol), inputs)

    # Flatten the list of results
    date_to_symbols = {}
    for result in results:
        for date, symbol in result:
            if date not in date_to_symbols:
                date_to_symbols[date] = []
            date_to_symbols[date].append(symbol)

    # Convert to DataFrame
    df = pd.DataFrame(list(date_to_symbols.items()), columns=['Date', 'Constituents'])
    df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
    df = df.sort_values(by='Date')
    # Save to CSV
    df.to_csv(indexfilename, index=False)
    print("CSV file saved as " + indexfilename + " .")



def all_index():
    return ['Dow Jones Industrial Average', 'Nasdaq-100', 'Nasdaq Q-50', 'Nasdaq Next Generation 100', 'Nasdaq-100 Technology Sector', 'Nasdaq Biotechnology', 'Russell Top 200', 'Russell 1000', 'Russell 2000', 'Russell 3000', 'Russell Mid Cap', 'Russell Micro Cap', 'Russell Small Cap Completeness', 'S&P 100', 'S&P 500', 'S&P MidCap 400', 'S&P SmallCap 600', 'S&P Composite 1500', 'S&P 1000', 'S&P 900', 'S&P 500 Dividend Aristocrats', 'S&P 500 ESG', 'Nasdaq-100 + Q-50 Superset', 'Nasdaq-100 + Next Generation 100 Superset','Russell 1000 2000 + Micro Cap Superset', 'Russell 2000 + Micro Cap Superset', 'Russell Micro Cap excl Russell 2000,' 'Russell 2000 bottom 1000', 'S&P 500 excl S&P 100']




if __name__ == "__main__":
    active_symbols, delisted_symbols = get_all_market_symbols()
    all_symbols = list(set(active_symbols + delisted_symbols))
    all_indexs = all_index() 
    # all_indexs = ['S&P 500'] # WARNING
    # all_symbols = all_symbols[0:30] # WARNING
    for indexname in all_indexs:
        build_constituents(all_symbols, indexname)
