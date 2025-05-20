import logging
import norgatedata
import multiprocessing as mp
import pandas as pd

def get_all_market_symbols():
    print("Retrieving all symbols from US Equities and US Equities Delisted...")
    
    us_equities = norgatedata.database_symbols('US Equities')
    us_delisted = norgatedata.database_symbols('US Equities Delisted')
    
    print(f"Retrieved {len(us_equities)} active symbols and {len(us_delisted)} delisted symbols.")
    return us_equities, us_delisted



def build_sp500_constituents_map():
    # Get all symbols
    active_symbols, delisted_symbols = get_all_market_symbols()
    all_symbols = list(set(active_symbols + delisted_symbols))

    # Set up parameters
    index_symbol = "S&P 500"  # S&P 500
    padding_setting = norgatedata.PaddingSetting.COMPACT  # No forward filling
    constituents_map = {}

    print(f"Processing {len(all_symbols)} symbols...")

    for symbol in all_symbols:
        try:
            # Get index constituent time series for this symbol
            df = norgatedata.index_constituent_timeseries(
                symbol,
                index_symbol,
                padding_setting=padding_setting,
                timeseriesformat="pandas-dataframe"
            )
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            continue

        if df.empty:
            continue

        # Filter only dates where the symbol was in the index
        df_filtered = df[df['Index Constituent'] == 1]

        for date in df_filtered.index.get_level_values(0):
            date_str = date.strftime('%Y-%m-%d')
            if date_str not in constituents_map:
                constituents_map[date_str] = []
            constituents_map[date_str].append(symbol)

    # Convert to DataFrame
    result = pd.DataFrame(list(constituents_map.items()), columns=['Date', 'Constituents'])

    # Save to CSV
    result.to_csv('sp500_constituents_by_date.csv', index=False)
    print("CSV file saved as 'sp500_constituents_by_date.csv'.")

    return result


if __name__ == "__main__":
    build_sp500_constituents_map()