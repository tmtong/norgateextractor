from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import pandas as pd
import os
import re
from datetime import datetime
from cachetools import LRUCache
from threading import Lock
import uvicorn
import math
import numpy as np

from typing import Any, Optional
app = FastAPI()

# === Configuration ===
MOUNTPOINT = "./norgatedata"
INDEX_DIR = os.path.join(MOUNTPOINT, "index")
STOCK_DIR = os.path.join(MOUNTPOINT, "stock")
METRICS_DIR =  os.path.join(MOUNTPOINT, "stockmetrics")

DATE_FORMAT = "%Y-%m-%d"
MAX_SYMBOL_LENGTH = 10

# === Global Lazy Cache with LRU Eviction ===
INDEX_CACHE_SIZE = 50   # Max number of index datasets cached
STOCK_CACHE_SIZE = 200  # Max number of stock datasets cached

GLOBAL_CACHE = {
    "index": LRUCache(maxsize=INDEX_CACHE_SIZE),
    "stock": LRUCache(maxsize=STOCK_CACHE_SIZE),
}

CACHE_LOCK = Lock()  # Ensure thread-safe access to cache

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
    """
    Sanitizes and validates a stock/index symbol.

    Allows: uppercase letters, numbers, dot (.), dash (-), underscore (_)
    Returns: sanitized symbol in uppercase
    """
    symbol = symbol.strip().upper()
    if not re.fullmatch(r"[A-Za-z0-9._\-_]+", symbol):
        raise ValueError("Symbol contains invalid characters")
    return symbol

def safe_file_path(base_dir: str, filename: str) -> str:
    full_path = os.path.join(base_dir, filename)
    resolved_path = os.path.realpath(full_path)
    base_realpath = os.path.realpath(base_dir)
    if not resolved_path.startswith(base_realpath):
        raise ValueError("Path traversal detected")
    return resolved_path


# === Lazy Load Helpers ===

def load_index_to_cache(index_symbol: str) -> Dict[str, List[str]]:
    filename = f"{index_symbol}.components"
    file_path = os.path.join(INDEX_DIR, filename)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Index file not found: {file_path}")

    try:
        df = pd.read_csv(file_path)
        data_map = {}

        for _, row in df.iterrows():
            try:
                date = pd.to_datetime(row['Date']).strftime(DATE_FORMAT)
                constituents = eval(row['Constituents'])  # convert string to list
                data_map[date] = constituents
            except Exception as e:
                print(f"Error parsing row in index {index_symbol}: {str(e)}")
                continue

        return data_map

    except Exception as e:
        raise RuntimeError(f"Failed to load index {index_symbol}: {str(e)}")


def load_stock_to_cache(symbol: str) -> Dict[str, Dict]:
    filename = f"{symbol}.csv"
    file_path = os.path.join(METRICS_DIR, filename)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Stock file not found: {file_path}")

    try:
        df = pd.read_csv(file_path)
        data_map = {}

        for _, row in df.iterrows():
            try:
                date = pd.to_datetime(row['date']).strftime(DATE_FORMAT)
                data_map[date] = row.to_dict()
            except Exception as e:
                print(f"Error parsing row in stock {symbol}: {str(e)}")
                continue

        return data_map

    except Exception as e:
        raise RuntimeError(f"Failed to load stock {symbol}: {str(e)}")


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


    # Build NaN row based on available columns
    if stock_data:
        example_row = next(iter(stock_data.values()))
        nan_row = {col: None for col in example_row.keys()}
    else:
        nan_row = {}

    row = stock_data.get(input_date, nan_row)
    row = {k: None if isinstance(v, float) and math.isnan(v) else v for k, v in row.items()}
    return row


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
        return {
            "symbol": symbol,
            "requested_date": input_date,
            "found_date": None,
            "value": None
        }

    # Get min/max date once
    all_dates = list(stock_data.keys())
    min_date = min(all_dates)
    max_date = max(all_dates)

    current_date = pd.to_datetime(input_date)

    # Early exit if before earliest data
    if current_date < pd.to_datetime(min_date):
        return {
            "symbol": symbol,
            "requested_date": input_date,
            "found_date": None,
            "value": None
        }

    # Start from input date and go backward
    while current_date >= pd.to_datetime(min_date):
        date_str = current_date.strftime(DATE_FORMAT)
        row = stock_data.get(date_str)

        if row:
            value = row.get(field)
            if value is not None and not (isinstance(value, float) and np.isnan(value)):
                return {
                    "symbol": symbol,
                    "requested_date": input_date,
                    "found_date": date_str,
                    "value": value
                }

        # Move one day back
        current_date = current_date - pd.Timedelta(days=1)

    # No valid data found
    return {
        "symbol": symbol,
        "requested_date": input_date,
        "found_date": None,
        "value": None
    }





@app.get("/getall")
def get_all_data(symbol: str, start_date: Optional[str] = None, end_date: Optional[str] = None):
    try:
        symbol = sanitize_symbol(symbol)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Load stock data (from cache or file)
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
        return {"symbol": symbol, "data": []}

    # Parse date range (if provided)
    try:
        start = datetime.strptime(start_date, DATE_FORMAT) if start_date else None
        end = datetime.strptime(end_date, DATE_FORMAT) if end_date else None
    except ValueError as e:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")

    def _safe_value(v: Any) -> Any:
        # Replace NaN with None for JSON compatibility
        if isinstance(v, float) and math.isnan(v):
            return None
        return v

    result = []
    for date_str, row in stock_data.items():
        date_obj = datetime.strptime(date_str, DATE_FORMAT)

        # Apply date filter if bounds are provided
        if start and date_obj < start:
            continue
        if end and date_obj > end:
            continue

        # Clean row: replace NaN and add date
        cleaned_row = {k: _safe_value(v) for k, v in row.items()}
        cleaned_row["date"] = date_str
        result.append(cleaned_row)

    return {"symbol": symbol, "data": result}



# === Run the server via main() for debugging in VSCode ===
if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        loop="asyncio",
        http="httptools",
    )