import logging
import norgatedata
import multiprocessing as mp
import pandas as pd
from multiprocessing import Pool, cpu_count
from functools import partial

def get_all_market_symbols():
    print("Retrieving all symbols from US Equities and US Equities Delisted...")
    
    us_equities = norgatedata.database_symbols('US Equities')
    us_delisted = norgatedata.database_symbols('US Equities Delisted')
    
    print(f"Retrieved {len(us_equities)} active symbols and {len(us_delisted)} delisted symbols.")
    return us_equities, us_delisted




def process_symbol(symbol, index_symbol="S&P 500"):
    """Process a single symbol to extract its S&P 500 membership history"""
    print('Processing ' + symbol)
    try:
        df = norgatedata.index_constituent_timeseries(
            symbol,
            index_symbol,
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


def build_sp500_constituents_map_mp():
    active_symbols, delisted_symbols = get_all_market_symbols()
    all_symbols = list(set(active_symbols + delisted_symbols))

    print(f"Processing {len(all_symbols)} symbols using multiprocessing...")

    # Use all available CPU cores minus one to avoid overloading the system
    num_processes = 4
    print(f"Using {num_processes} processes.")

    with Pool(num_processes) as pool:
        results = pool.map(partial(process_symbol), all_symbols)

    # Flatten the list of results
    date_to_symbols = {}
    for result in results:
        for date, symbol in result:
            if date not in date_to_symbols:
                date_to_symbols[date] = []
            date_to_symbols[date].append(symbol)

    # Convert to DataFrame
    df = pd.DataFrame(list(date_to_symbols.items()), columns=['Date', 'Constituents'])

    # Save to CSV
    df.to_csv('sp500_constituents_by_date.csv', index=False)
    print("CSV file saved as 'sp500_constituents_by_date.csv'.")

    return df


if __name__ == "__main__":
    build_sp500_constituents_map_mp()
