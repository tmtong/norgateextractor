from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import pandas as pd
import os
import re
from datetime import datetime
import pyarrow.feather as paft
import zstandard as zstd
import math
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import pickle
from threading import Lock
import traceback
import uvicorn
from tqdm import tqdm
from contextlib import asynccontextmanager

app = FastAPI()

# === Configuration ===
MOUNTPOINT = "./norgatedata"
INDEX_DIR = os.path.join(MOUNTPOINT, "index")
STOCK_DIR = os.path.join(MOUNTPOINT, "stock")
METRICS_DIR = os.path.join(MOUNTPOINT, "stockmetrics")
CACHE_DIR = os.path.join(MOUNTPOINT, "cache")

DATE_FORMAT = "%Y-%m-%d"
startdate_str = '2000-01-01'
enddate_str = '2025-01-01'
# indexsymbol = "INDEX-SPX"
indexsymbol = "INDEX-SP900"

os.makedirs(CACHE_DIR, exist_ok=True)

# === Global In-Memory Cache (no LRU) ===
GLOBAL_CACHE = {
    "index": {},  # index_symbol -> { date_str: [symbols] }
    "stock": {}   # symbol -> { date_str: row_dict }
}

CACHE_LOCK = Lock()  # Thread-safe access to cache

# Ensure directories exist
os.makedirs(CACHE_DIR, exist_ok=True)


# === Pydantic Models ===
class IndexRequest(BaseModel):
    date: str
    indexsymbol: str

class IndexResponse(BaseModel):
    date: str
    indexsymbol: str
    constituents: List[str]

class StockDataRequest(BaseModel):
    symbol: str
    date: str

class ClosestRequest(BaseModel):
    symbol: str
    date: str
    field: str


# === Helper Functions ===
def sanitize_date(date_str: str) -> str:
    try:
        dt = datetime.strptime(date_str, DATE_FORMAT)
        return dt.strftime(DATE_FORMAT)
    except ValueError:
        raise ValueError(f"Invalid date format. Expected format: {DATE_FORMAT}")

def sanitize_symbol(symbol: str) -> str:
    symbol = symbol.strip().upper()
    if not re.fullmatch(r"[A-Za-z0-9._\-_]+", symbol):
        raise ValueError("Symbol contains invalid characters")
    return symbol


# === Lazy Load Helpers ===
def load_index_to_cache(index_symbol: str) -> dict:
    filename = f"{index_symbol}.components"
    file_path = os.path.join(INDEX_DIR, filename)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Index file not found: {file_path}")

    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date']).dt.strftime(DATE_FORMAT)
    data_map = {}

    for _, row in df.iterrows():
        date = row['Date']
        try:
            constituents = eval(row['Constituents'])  # convert string to list
            data_map[date] = constituents
        except Exception as e:
            print(f"Error parsing row in index {index_symbol}: {str(e)}")
            continue

    return data_map


def load_stock_to_cache(symbol: str) -> dict:
    filename = f"{symbol}.feather"
    file_path = os.path.join(METRICS_DIR, filename)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Stock file not found: {file_path}")
    print("Loading " + symbol + " to cache")
    df = pd.read_feather(file_path)
    df['date'] = pd.to_datetime(df['date']).dt.strftime(DATE_FORMAT)
    data_map = {}

    for _, row in df.iterrows():
        date = row['date']
        data_map[date] = row.to_dict()

    return data_map

def savedata(indexsymbol: str, start_date: str, end_date: str):
    """
    Saves all stocks that were part of an index between two dates into a compressed pickle file
    named like: full_cache.INDEXSYMBOL.START.END.zst
    """
    print(f"Precomputing data for {indexsymbol} from {start_date} to {end_date}...")

    index_file = os.path.join(INDEX_DIR, f"{indexsymbol}.components")
    if not os.path.exists(index_file):
        raise FileNotFoundError(f"Index file not found: {index_file}")

    df_index = pd.read_csv(index_file)
    df_index['Date'] = pd.to_datetime(df_index['Date']).dt.strftime(DATE_FORMAT)

    # Filter by date range
    df_index = df_index[(df_index['Date'] >= start_date) & (df_index['Date'] <= end_date)]

    # Get unique symbols
    all_symbols = set()
    for _, row in df_index.iterrows():
        try:
            constituents = eval(row['Constituents'])  # convert string to list
            for c in constituents:
                all_symbols.add(sanitize_symbol(c))
        except Exception as e:
            print(f"Error parsing constituents: {e}")

    print(f"Found {len(all_symbols)} unique symbols. Loading data...")

    # Load all stock data
    stock_cache = {}
    for symbol in tqdm(list(all_symbols), desc="Loading Stocks"):
        filename = f"{symbol}.feather"
        file_path = os.path.join(METRICS_DIR, filename)
        if not os.path.exists(file_path):
            print(f"Stock file not found for {symbol}")
            continue
        try:
            df = paft.read_feather(file_path)
            df['date'] = pd.to_datetime(df['date']).dt.strftime(DATE_FORMAT)
            data_map = {}
            for _, row in df.iterrows():
                date_str = row['date']
                data_map[date_str] = row.to_dict()
            stock_cache[symbol] = data_map
        except Exception as e:
            print(f"Failed to load {symbol}: {str(e)}")

    # Build index_data
    index_cache = {}
    try:
        df_index = pd.read_csv(index_file)
        df_index['Date'] = pd.to_datetime(df_index['Date']).dt.strftime(DATE_FORMAT)
        for _, row in df_index.iterrows():
            date = row['Date']
            try:
                constituents = eval(row['Constituents'])
                index_cache[date] = constituents
            except Exception as e:
                print(f"Error parsing row in index {indexsymbol}: {str(e)}")
    except Exception as e:
        print(f"Error reading index file: {str(e)}")

    # Save both index and stock data
    cache_data = {
        "stock": stock_cache,
        "index": index_cache,
    }

    cache_file = os.path.join(CACHE_DIR, f"full_cache.{indexsymbol}.{start_date}.{end_date}.zst")
    print(f"Saving cache to {cache_file}...")

    with open(cache_file, 'wb') as f:
        compressor = zstd.ZstdCompressor(level=3)
        with compressor.stream_writer(f) as stream:
            pickle.dump(cache_data, stream)

    print("✅ Cache saved.")


def loaddata(indexsymbol: str, start_date: str, end_date: str) -> bool:
    """Loads precomputed cache based on indexsymbol, start_date, end_date"""
    cache_file = os.path.join(CACHE_DIR, f"full_cache.{indexsymbol}.{start_date}.{end_date}.zst")
    if not os.path.isfile(cache_file):
        print(f"⚠️ No precomputed cache found for {indexsymbol}, {start_date}, {end_date}")
        return False

    print(f"🔄 Loading precomputed cache from {cache_file}...")
    dctx = zstd.ZstdDecompressor()
    with open(cache_file, 'rb') as f:
        with dctx.stream_reader(f) as reader:
            raw_data = reader.read()
    loaded = pickle.loads(raw_data)

    GLOBAL_CACHE["stock"] = loaded.get("stock", {})
    GLOBAL_CACHE["index"] = loaded.get("index", {})

    print(f"Loaded {len(GLOBAL_CACHE['stock'])} stocks and {len(GLOBAL_CACHE['index'])} index entries.")
    return True



@app.on_event("startup")
def startup():
    print("Attempt to load data from cache")
    success = loaddata(indexsymbol, startdate_str, enddate_str)
    if not success:
        print("🛑 Falling back to lazy loading.")
    else:
        print("🎉 Loaded from precomputed cache!")


# === API Endpoints ===

@app.post("/indexconstituents", response_model=IndexResponse)
def get_index_constituents(request: IndexRequest):
    try:
        input_date = sanitize_date(request.date)
        index_symbol = sanitize_symbol(request.indexsymbol)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    with CACHE_LOCK:
        if index_symbol in GLOBAL_CACHE["index"]:
            index_data = GLOBAL_CACHE["index"][index_symbol]
        else:
            try:
                index_data = load_index_to_cache(index_symbol)
                GLOBAL_CACHE["index"][index_symbol] = index_data
            except FileNotFoundError:
                raise HTTPException(status_code=404, detail=f"Index file not found for {index_symbol}")
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

    constituents = index_data.get(input_date)
    if not constituents:
        raise HTTPException(status_code=404, detail=f"No data for date: {input_date} in index {index_symbol}")

    return IndexResponse(
        date=input_date,
        indexsymbol=index_symbol,
        constituents=constituents
    )


@app.post("/stockdata")
def get_stock_data(request: StockDataRequest):
    try:
        input_date = sanitize_date(request.date)
        symbol = sanitize_symbol(request.symbol)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    with CACHE_LOCK:
        if symbol in GLOBAL_CACHE["stock"]:
            stock_data = GLOBAL_CACHE["stock"][symbol]
        else:
            try:
                stock_data = load_stock_to_cache(symbol)
                GLOBAL_CACHE["stock"][symbol] = stock_data
            except FileNotFoundError:
                raise HTTPException(status_code=404, detail=f"Stock file not found for {symbol}")
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

    example_row = next(iter(stock_data.values())) if stock_data else {}
    nan_row = {col: None for col in example_row.keys()}
    row = stock_data.get(input_date, nan_row)

    cleaned_row = {
        k: None if isinstance(v, float) and (math.isnan(v) or math.isinf(v)) else v
        for k, v in row.items()
    }

    return cleaned_row


@app.post("/getclosest")
def get_closest(request: ClosestRequest):
    try:
        input_date = sanitize_date(request.date)
        symbol = sanitize_symbol(request.symbol)
        field = request.field
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    with CACHE_LOCK:
        if symbol in GLOBAL_CACHE["stock"]:
            stock_data = GLOBAL_CACHE["stock"][symbol]
        else:
            try:
                stock_data = load_stock_to_cache(symbol)
                GLOBAL_CACHE["stock"][symbol] = stock_data
            except FileNotFoundError:
                raise HTTPException(status_code=404, detail=f"Stock file not found for {symbol}")
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

    if not stock_data:
        return {"symbol": symbol, "requested_date": input_date, "found_date": None, "value": None}

    all_dates = list(stock_data.keys())
    min_date = min(all_dates)
    max_date = max(all_dates)
    current_date = pd.to_datetime(input_date)

    if current_date < pd.to_datetime(min_date):
        return {"symbol": symbol, "requested_date": input_date, "found_date": None, "value": None}

    while current_date >= pd.to_datetime(min_date):
        date_str = current_date.strftime(DATE_FORMAT)
        row = stock_data.get(date_str)
        if row:
            value = row.get(field)
            if value is not None and not (isinstance(value, float) and math.isnan(value)):
                return {"symbol": symbol, "requested_date": input_date, "found_date": date_str, "value": value}
        current_date -= pd.Timedelta(days=1)

    return {"symbol": symbol, "requested_date": input_date, "found_date": None, "value": None}

@app.get("/getall")
def get_all_data(symbol: str,
                 start_date: Optional[str] = None,
                 end_date: Optional[str] = None):
    try:
        symbol = sanitize_symbol(symbol)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    with CACHE_LOCK:
        if symbol in GLOBAL_CACHE["stock"]:
            stock_data = GLOBAL_CACHE["stock"][symbol]
        else:
            try:
                stock_data = load_stock_to_cache(symbol)
                GLOBAL_CACHE["stock"][symbol] = stock_data
            except FileNotFoundError:
                raise HTTPException(status_code=404,
                                    detail=f"Stock file not found for {symbol}")
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

    def _safe_value(v):
        # convert NaN/Inf/-Inf -> None
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return None
        return v

    result = []
    for date_str, row in stock_data.items():
        dt = datetime.strptime(date_str, DATE_FORMAT)
        if start_date and dt < datetime.strptime(start_date, DATE_FORMAT):
            continue
        if end_date and dt > datetime.strptime(end_date, DATE_FORMAT):
            continue

        # clean every scalar
        cleaned = {k: _safe_value(v) for k, v in row.items()}
        cleaned["date"] = date_str
        result.append(cleaned)

    return {"symbol": symbol, "data": result}



# === Run the server via main() for debugging in VSCode ===
if __name__ == "__main__":

    cache_file = os.path.join(CACHE_DIR, f"full_cache.{indexsymbol}.{startdate_str}.{enddate_str}.zst")
    if not os.path.exists(cache_file):
        print(f"Cache file {cache_file} not found. Generating cache...")
        savedata(indexsymbol, startdate_str, enddate_str)

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        loop="asyncio",
        http="httptools",
    )