#!/usr/bin/env python3

from __future__ import annotations
import os
import pandas as pd
import requests
import datetime
import pytz
from pathlib import Path
import subprocess
import json

import pandas_market_calendars as mcal

# Configuration
URL = "https://www.ssga.com/us/en/intermediary/library-content/products/fund-data/etfs/us/holdings-daily-us-en-spy.xlsx"
OUTPUT_DIR = './rawindex' 
CSV_FILE = OUTPUT_DIR + "/INDEX-SPY.csv"
NY_TIME = pytz.timezone("America/New_York")



def is_trading_day():
    """Check if today is a NYSE trading day."""
    today = datetime.datetime.now(NY_TIME).date()
    nyse = mcal.get_calendar('NYSE')
    schedule = nyse.schedule(start_date=today, end_date=today)
    return not schedule.empty

def download_file(url, tempfilename):
    """Download the xlsx file using wget."""
    print("Downloading file using wget...")
    tempfilename
    cmd = ["wget", "-O", str(tempfilename), url]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    if result.returncode != 0:
        raise Exception(f"Download failed: {result.stderr.decode()}")

    return tempfilename

def parse_ssgaxlsx(file_path):
    """Parse the xlsx file starting from row 6 (skip 5 header rows)."""
    df = pd.read_excel(file_path, skiprows=5)
    # Assuming Ticker column exists
    df.columns = ['Name', 'Ticker', 'Identifier',	'SEDOL', 'Weight',	'Sector',	'Shares Held',	'Local Currency']
    tickers = df['Ticker'].dropna().tolist()
    return tickers

def append_to_csv(date_str, constituents):
    new_row = date_str + ',' + constituents + '\n'
    if not os.path.isfile(CSV_FILE):
        with open(CSV_FILE, 'w') as f:
            f.write("Date,Constituents\n")
            f.write(new_row)
    else:
        with open(CSV_FILE, 'a+') as f:
            f.write(new_row)

def spy():
    xlsx_file = download_file(URL, )

def main():
    if not is_trading_day():
        print("Today is not a trading day.")
        return

    today_ny = datetime.datetime.now(NY_TIME).strftime("%Y-%m-%d")
    print(f"Processing for {today_ny}...")

    try:
        xlsx_file = download_file(URL)
        constituents = parse_xlsx(xlsx_file)
        constituents_str = "', '".join(constituents)
        constituents_str = "\"['" + constituents_str + "']\""
        append_to_csv(today_ny, constituents_str)
        print("Successfully updated constituents.")
    except Exception as e:
        print(f"Error occurred: {e}")
    finally:
        if 'xlsx_file' in locals() and os.path.isfile(xlsx_file):
            os.remove(xlsx_file)

if __name__ == "__main__":
    main()