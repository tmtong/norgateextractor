import os
import norgatedata
import pandas as pd
import multiprocessing

# Specify the directories
symbol_directory = 'data/index/'
stock_directory = 'data/stock/'

# Ensure the stock directory exists
os.makedirs(stock_directory, exist_ok=True)

# Read all symbols from .symbol files
def load_symbols():
    symbols = []
    for filename in os.listdir(symbol_directory):
        if filename.endswith('.symbol'):
            file_path = os.path.join(symbol_directory, filename)
            with open(file_path, 'r') as f:
                symbols.extend([line.strip() for line in f if line.strip()])
    return list(set(symbols))  # Remove duplicates

# Retrieve and save data for a single symbol
def fetch_and_save_symbol_data(symbol):
    output_file = f'{symbol}.csv'
    output_path = os.path.join(stock_directory, output_file)
    
    # Skip if the file already exists
    if os.path.exists(output_path):
        print(f"Skipping {symbol}, file already exists: {output_path}")
        return
    
    try:
        # Retrieve all available data for the symbol
        data = norgatedata.get_timeseries(
            symbol,
            timeseriesformat='pandas-dataframe'
        )
        
        # Save the data to a CSV file
        data.to_csv(output_path, index=True)
        
        print(f"Data for {symbol} has been saved to {output_path}")
    except Exception as e:
        print(f"Error retrieving data for {symbol}: {e}")

# Main function to process all symbols in parallel
def main():
    symbols = load_symbols()
    
    # Determine the number of CPU cores available
    num_workers = multiprocessing.cpu_count()
    print(f"Using {num_workers} worker processes")
    
    # Use Pool to fetch data in parallel
    with multiprocessing.Pool(processes=num_workers) as pool:
        pool.map(fetch_and_save_symbol_data, symbols)

if __name__ == "__main__":
    main()
