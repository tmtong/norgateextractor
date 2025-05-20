import logging
import norgatedata
import multiprocessing as mp

def get_all_market_symbols():
    logging.info("Retrieving all symbols from US Equities and US Equities Delisted...")
    
    us_equities = norgatedata.database_symbols('US Equities')
    us_delisted = norgatedata.database_symbols('US Equities Delisted')
    
    logging.info(f"Retrieved {len(us_equities)} active symbols and {len(us_delisted)} delisted symbols.")
    return us_equities, us_delisted




def download_stock_data(symbol):
    logging.info(f"Downloading data for {symbol} ...")
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
    pricedata.to_csv(dirpath + symbol  + '.symbol')
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
    single_download(all_symbols)