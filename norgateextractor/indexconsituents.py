import norgatedata
import pandas as pd
import pandas_market_calendars as mcal
import json
from datetime import datetime

# Get the start and end dates for which you want to retrieve data
start_date = '2000-01-01'
end_date = datetime.now().strftime('%Y-%m-%d')

# Get the trading calendar for NYSE
nyse = mcal.get_calendar('NYSE')
trading_days = nyse.valid_days(start_date=start_date, end_date=end_date)

# Define the list of US indices
us_indices = [
    "Dow Jones Industrial Average",
    "Nasdaq-100",
    "Nasdaq Q-50",
    "Nasdaq Next Generation 100",
    "Nasdaq-100 Technology Sector",
    "Nasdaq Biotechnology",
    "Russell Top 200",
    "Russell 1000",
    "Russell 2000",
    "Russell 3000",
    "Russell Mid Cap",
    "Russell Micro Cap",
    "Russell Small Cap Completeness",
    "S&P 100",
    "S&P 500",
    "S&P MidCap 400",
    "S&P SmallCap 600",
    "S&P Composite 1500",
    "S&P 1000",
    "S&P 900",
    "S&P 500 Dividend Aristocrats",
    "S&P 500 ESG",
    "Nasdaq-100 + Q-50 Superset",
    "Nasdaq-100 + Next Generation 100 Superset",
    "Russell 1000 2000 + Micro Cap Superset",
    "Russell 2000 + Micro Cap Superset",
    "Russell Micro Cap excl Russell 2000",
    "Russell 2000 bottom 1000",
    "S&P 500 excl S&P 100"
]

# Iterate over each US index
for index in us_indices:
    # Create a dictionary to store the constituents for each trading day
    constituents_by_date = {}

    # Iterate over each trading day
    for date in trading_days:
        date_str = date.strftime('%Y-%m-%d')
        try:
            # Get the constituents for the current index and date
            index_constituents = norgatedata.index_constituent_timeseries(
                index,
                timeseriesformat="pandas-dataframe",
                date=date_str
            )
            
            # Extract the constituents from the dataframe
            if not index_constituents.empty:
                constituents = index_constituents['symbol'].tolist()
                # Store the constituents for the current trading day
                constituents_by_date[date_str] = constituents
        except Exception as e:
            print(f"Error retrieving constituents for {index} on {date_str}: {e}")
    dirpath = './data/index/'

    # Save the data to a JSON file for the current index
    with open(dirpath + index.replace(' ', '_') + '.json', 'w') as f:
        json.dump(constituents_by_date, f, indent=4)

    print(f"Data for {index} has been saved to {index.replace(' ', '_')}.json")

