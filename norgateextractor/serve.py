from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import pandas as pd
import os
import re
from datetime import datetime

app = FastAPI()

# === Configuration ===
MOUNTPOINT = "./norgatedata"
INDEX_DIR = os.path.join(MOUNTPOINT, "data", "index")
STOCK_DIR = os.path.join(MOUNTPOINT, "data", "stock")

DATE_FORMAT = "%Y-%m-%d"
MAX_SYMBOL_LENGTH = 10

# === Global Lazy Cache ===
GLOBAL_CACHE = {
    "index": {},  # { index_symbol: { date_str: [symbols] } }
    "stock": {}   # { symbol: { date_str: { col: val } } }
}

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


# === Helper Functions ===

def sanitize_date(date_str: str) -> str:
    try:
        dt = datetime.strptime(date_str, DATE_FORMAT)
        return dt.strftime(DATE_FORMAT)
    except ValueError:
        raise ValueError(f"Invalid date format. Expected format: {DATE_FORMAT}")

def sanitize_symbol(symbol: str, max_length: int = MAX_SYMBOL_LENGTH) -> str:
    symbol = symbol.strip().upper()
    if not re.fullmatch(r"[A-Z0-9]+", symbol):
        raise ValueError("Symbol must contain only letters and numbers")
    if len(symbol) > max_length:
        raise ValueError(f"Symbol exceeds maximum length of {max_length}")
    return symbol

def safe_file_path(base_dir: str, filename: str) -> str:
    full_path = os.path.join(base_dir, filename)
    resolved_path = os.path.realpath(full_path)
    base_realpath = os.path.realpath(base_dir)
    if not resolved_path.startswith(base_realpath):
        raise ValueError("Path traversal detected")
    return resolved_path


# === Lazy Load Helpers ===

def load_index_to_cache(index_symbol: str):
    filename = f"{index_symbol.lower()}.components"
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

        GLOBAL_CACHE["index"][index_symbol] = data_map

    except Exception as e:
        raise RuntimeError(f"Failed to load index {index_symbol}: {str(e)}")


def load_stock_to_cache(symbol: str):
    filename = f"{symbol.lower()}.symbol"
    file_path = os.path.join(STOCK_DIR, filename)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Stock file not found: {file_path}")

    try:
        df = pd.read_csv(file_path)
        data_map = {}

        for _, row in df.iterrows():
            try:
                date = pd.to_datetime(row['Date']).strftime(DATE_FORMAT)
                data_map[date] = row.to_dict()
            except Exception as e:
                print(f"Error parsing row in stock {symbol}: {str(e)}")
                continue

        GLOBAL_CACHE["stock"][symbol] = data_map

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

    # Lazy load index if not in cache
    if index_symbol not in GLOBAL_CACHE["index"]:
        try:
            load_index_to_cache(index_symbol)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail=f"Index file not found for {index_symbol}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    index_data = GLOBAL_CACHE["index"][index_symbol]
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

    # Lazy load stock if not in cache
    if symbol not in GLOBAL_CACHE["stock"]:
        try:
            load_stock_to_cache(symbol)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail=f"Stock file not found for {symbol}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    stock_data = GLOBAL_CACHE["stock"][symbol]
    row = stock_data.get(input_date)

    if not row:
        raise HTTPException(status_code=404, detail=f"No data for date: {input_date} for stock {symbol}")

    return row
