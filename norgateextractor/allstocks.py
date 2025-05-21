import logging
import norgatedata
import multiprocessing as mp
import re

def get_all_market_symbols():
    print("Retrieving all symbols from US Equities and US Equities Delisted...")
    
    us_equities = norgatedata.database_symbols('US Equities')
    us_delisted = norgatedata.database_symbols('US Equities Delisted')
    
    print(f"Retrieved {len(us_equities)} active symbols and {len(us_delisted)} delisted symbols.")
    return us_equities, us_delisted

def get_all_indexs():
    return ['$DJI', '$NDX', '$NXTQ', '$NGX', '$NDXT', '$NBI', '$RT200', '$RUI', '$RUT', '$RUA', '$RMC', '$RUMIC', '$RSCC', '$OEX', '$SPX', '$MID', '$SML', '$SP1500', '$SP1000', '$SP900', '$SPDAUDP', '$SPESG', '$N1-150', '$N1-200', '$R1-4000', '$R1001-4000', '$R3001-4000', '$R2001-3000', '$SP101-500']



def download_stock_data(symbol):
    print(f"Downloading data for {symbol} ...")
    priceadjust = norgatedata.StockPriceAdjustmentType.TOTALRETURN
    padding_setting = norgatedata.PaddingType.NONE
    timeseriesformat = 'pandas-dataframe'
    
    pricedata = norgatedata.price_timeseries(
        symbol,
        stock_price_adjustment_setting=priceadjust,
        padding_setting=padding_setting,
        timeseriesformat=timeseriesformat
    )
    dirpath = './data/stock/'
    symbol = re.sub(r'[^a-zA-Z0-9]', '', symbol)
    filename = dirpath + symbol
    pricedata.to_csv(filename  + '.symbol')
    return pricedata


def single_download(symbols):
    for symbol in symbols:
        download_stock_data(symbol)
def multi_download(symbols):
    cpu_count = max(1, int(mp.cpu_count() * 0.8))
    # cpu_count = 1
    pool = mp.Pool(processes=cpu_count)
    pool.map(download_stock_data, symbols)

if __name__ == "__main__":
    active_symbols, delisted_symbols = get_all_market_symbols()
    all_symbols = active_symbols + delisted_symbols
    all_indexs = get_all_indexs()
    multi_download(all_indexs)
