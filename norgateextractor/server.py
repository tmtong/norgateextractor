"""
FastAPI server for Norgate-style data.
Pre-cleans rows and adds closest-valid-adjclose ordinal column
so that /stockdata and /getclosest are both O(1) at request time.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import pandas as pd
import os
import re
from datetime import datetime, date as _date
import pyarrow.feather as paft
import zstandard as zstd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import pickle
from threading import Lock
import traceback
import uvicorn
from tqdm import tqdm
from contextlib import asynccontextmanager
import math

app = FastAPI()

# ---------- configuration ---------------------------------------------------
MOUNTPOINT   = "./norgatedata"
INDEX_DIR    = os.path.join(MOUNTPOINT, "index")
STOCK_DIR    = os.path.join(MOUNTPOINT, "stock")
METRICS_DIR  = os.path.join(MOUNTPOINT, "stockmetrics")
CACHE_DIR    = os.path.join(MOUNTPOINT, "cache")

DATE_FORMAT  = "%Y-%m-%d"
START_DATE   = "1998-01-01"
END_DATE     = "2025-01-01"
INDEX_SYMBOL = "INDEX-SP900"

os.makedirs(CACHE_DIR, exist_ok=True)

# ---------- global in-memory cache ------------------------------------------
GLOBAL_CACHE: Dict[str, Dict[str, Any]] = {"stock": {}, "index": {}}
CACHE_LOCK = Lock()

# ---------- pydantic models --------------------------------------------------
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

# ---------- helpers ---------------------------------------------------------
def sanitize_date(date_str: str) -> str:
    try:
        return _date.fromisoformat(date_str).isoformat()
    except ValueError:
        raise ValueError(f"Invalid date format. Expected {DATE_FORMAT}")

def sanitize_symbol(symbol: str) -> str:
    symbol = symbol.strip().upper()
    if not re.fullmatch(r"[A-Z0-9._-]+", symbol):
        raise ValueError("Symbol contains invalid characters")
    return symbol

def _clean_scalar(v: Any) -> Any:
    """Convert NaN/inf/-inf -> None once and for all."""
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    return v

def _build_closest_adj_column(rows: List[Dict[str, Any]]) -> None:
    """
    Mutate rows in place: add 'closest_adj_ord' column.
    Walks backwards so we always know the nearest earlier date with valid adjclose.
    """
    last_good_ord: Optional[int] = None
    for row in reversed(rows):
        adj = row.get("adjclose")
        adj = _clean_scalar(adj)
        row["adjclose"] = adj

        cur_ord = _date.fromisoformat(row["date"]).toordinal()
        if adj is not None:
            last_good_ord = cur_ord
        row["closest_adj_ord"] = last_good_ord

# ---------- lazy-load helpers (kept for on-demand fallback) -----------------
def load_index_to_cache(index_symbol: str) -> Dict[str, List[str]]:
    filename = f"{index_symbol}.components"
    file_path = os.path.join(INDEX_DIR, filename)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Index file not found: {file_path}")

    df = pd.read_csv(file_path)
    df["Date"] = pd.to_datetime(df["Date"]).dt.strftime(DATE_FORMAT)
    data_map = {}
    for _, row in df.iterrows():
        try:
            constituents = eval(row["Constituents"])
            data_map[row["Date"]] = [sanitize_symbol(c) for c in constituents]
        except Exception as e:
            print(f"Error parsing index row: {e}")
    return data_map

def load_stock_to_cache(symbol: str) -> Dict[str, Dict[str, Any]]:
    """Legacy lazy loader – NOT used if cache file exists."""
    feather_path = os.path.join(METRICS_DIR, f"{symbol}.feather")
    if not os.path.exists(feather_path):
        raise FileNotFoundError(f"Stock file not found for {symbol}")

    df = paft.read_feather(feather_path)
    df["date"] = pd.to_datetime(df["date"]).dt.strftime(DATE_FORMAT)
    df = df.sort_values("date")
    rows = df.to_dict(orient="records")

    # clean once
    for row in rows:
        for k, v in list(row.items()):
            row[k] = _clean_scalar(v)
        row["closest_adj_ord"] = None  # not needed for lazy path

    return {row["date"]: row for row in rows}

# ---------- pre-computation (savedata) --------------------------------------
def savedata(index_symbol: str, start_date: str, end_date: str) -> None:
    print(f"Pre-computing cache for {index_symbol}  {start_date} … {end_date}")

    # --- load index ------------------------------------------------------------
    index_file = os.path.join(INDEX_DIR, f"{index_symbol}.components")
    if not os.path.exists(index_file):
        raise FileNotFoundError(index_file)
    df_idx = pd.read_csv(index_file)
    df_idx["Date"] = pd.to_datetime(df_idx["Date"]).dt.strftime(DATE_FORMAT)
    df_idx = df_idx[(df_idx["Date"] >= start_date) & (df_idx["Date"] <= end_date)]
    index_cache: Dict[str, List[str]] = {}
    for _, r in df_idx.iterrows():
        try:
            index_cache[r["Date"]] = [sanitize_symbol(c) for c in eval(r["Constituents"])]
        except Exception as e:
            print(f"Bad index row: {e}")

    # --- discover symbols ------------------------------------------------------
    all_symbols = {c for lst in index_cache.values() for c in lst}
    print(f"Unique symbols: {len(all_symbols)}")

    all_symbols.add(index_symbol) # ADD INDEX-SPY with "INDEX-"
    all_symbols.add("INDEX-SP900")
    all_symbols.add("INDEX-SPX")

    # --- build stock cache -----------------------------------------------------
    stock_cache: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for symbol in tqdm(all_symbols, desc="Cleaning stocks"):
        feather_path = os.path.join(METRICS_DIR, f"{symbol}.feather")
        if not os.path.exists(feather_path):
            continue
        df = paft.read_feather(feather_path)
        df["date"] = pd.to_datetime(df["date"]).dt.strftime(DATE_FORMAT)
        df = df.sort_values("date")
        rows: List[Dict[str, Any]] = df.to_dict(orient="records")

        # 1. clean all columns
        for row in rows:
            for k, v in list(row.items()):
                row[k] = _clean_scalar(v)

        # 2. add closest_adj_ord
        _build_closest_adj_column(rows)

        # 3. build date->row map
        date_map = {row["date"]: row for row in rows}
        stock_cache[symbol] = date_map

    # --- persist ---------------------------------------------------------------
    cache_file = os.path.join(
        CACHE_DIR, f"full_cache.{index_symbol}.{start_date}.{end_date}.zst"
    )
    print(f"Writing {cache_file}")
    with open(cache_file, "wb") as f:
        compressor = zstd.ZstdCompressor(level=3)
        with compressor.stream_writer(f) as writer:
            pickle.dump({"stock": stock_cache, "index": index_cache}, writer)
    print("✅ Cache saved")

def loaddata(index_symbol: str, start_date: str, end_date: str) -> bool:
    cache_file = os.path.join(
        CACHE_DIR, f"full_cache.{index_symbol}.{start_date}.{end_date}.zst"
    )
    if not os.path.isfile(cache_file):
        return False
    print(f"Loading cache {cache_file}")
    with open(cache_file, "rb") as f:
        decompressor = zstd.ZstdDecompressor()
        with decompressor.stream_reader(f) as r:
            loaded = pickle.load(r)
    GLOBAL_CACHE.update(loaded)
    print("Cache loaded")
    return True

# ---------- startup ---------------------------------------------------------
@app.on_event("startup")
def startup():
    ok = loaddata(INDEX_SYMBOL, START_DATE, END_DATE)
    if not ok:
        print("Cache absent – use savedata() to build it")

# ---------- API  endpoints  -------------------------------------------------
@app.post("/indexconstituents", response_model=IndexResponse)
def get_index_constituents(req: IndexRequest):
    try:
        d = sanitize_date(req.date)
        idx = sanitize_symbol(req.indexsymbol)
    except ValueError as e:
        raise HTTPException(400, str(e))

    with CACHE_LOCK:
        idx_map = GLOBAL_CACHE["index"].get(idx)
        if idx_map is None:
            idx_map = load_index_to_cache(idx)
            GLOBAL_CACHE["index"][idx] = idx_map

    cons = idx_map.get(d)
    if cons is None:
        raise HTTPException(404, f"No constituents for {d}")
    return IndexResponse(date=d, indexsymbol=idx, constituents=cons)

@app.post("/stockdata")
def get_stock_data(req: StockDataRequest):
    try:
        sym = sanitize_symbol(req.symbol)
        d   = sanitize_date(req.date)
    except ValueError as e:
        raise HTTPException(400, str(e))

    with CACHE_LOCK:
        stock_map = GLOBAL_CACHE["stock"].get(sym)
        if stock_map is None:
            raise HTTPException(404, f"Symbol {sym} not in cache")

    row = stock_map.get(d)
    if row is None:
        empty = {k: None for k in next(iter(stock_map.values())).keys()}
        return empty
    return row  # already cleaned

@app.post("/getclosest")
def get_closest(req: ClosestRequest):
    try:
        sym   = sanitize_symbol(req.symbol)
        tgt_d = sanitize_date(req.date)
        field = req.field
    except ValueError as e:
        raise HTTPException(400, str(e))

    with CACHE_LOCK:
        stock_map = GLOBAL_CACHE["stock"].get(sym)
        if stock_map is None:
            raise HTTPException(404, f"Symbol {sym} not in cache")

    # --- fast path for adjclose ----------------------------------------------
    if field == "adjclose":
        start_row = stock_map.get(tgt_d)
        if start_row and start_row["adjclose"] is not None:
            return {"symbol": sym,
                    "requested_date": tgt_d,
                    "found_date": tgt_d,
                    "value": start_row["adjclose"]}
        # walk backwards via pre-computed ordinal
        tgt_ord = _date.fromisoformat(tgt_d).toordinal()
        for date_str in reversed(stock_map.keys()):
            if date_str > tgt_d:
                continue
            row = stock_map[date_str]
            closest_ord = row["closest_adj_ord"]
            if closest_ord is None:
                return {"symbol": sym,
                        "requested_date": tgt_d,
                        "found_date": None,
                        "value": None}
            closest_date = _date.fromordinal(closest_ord).isoformat()
            closest_row  = stock_map[closest_date]
            return {"symbol": sym,
                    "requested_date": tgt_d,
                    "found_date": closest_date,
                    "value": closest_row["adjclose"]}

    # --- generic path (any other field) – simple backward scan ---------------
    dt = _date.fromisoformat(tgt_d)
    while dt >= _date.min:
        ds = dt.isoformat()
        row = stock_map.get(ds)
        if row:
            v = row.get(field)
            if v is not None:
                return {"symbol": sym,
                        "requested_date": tgt_d,
                        "found_date": ds,
                        "value": v}
        dt -= _date.resolution
    return {"symbol": sym,
            "requested_date": tgt_d,
            "found_date": None,
            "value": None}

@app.get("/getall")
def get_all_data(symbol: str,
                 start_date: Optional[str] = None,
                 end_date: Optional[str] = None):
    try:
        sym = sanitize_symbol(symbol)
    except ValueError as e:
        raise HTTPException(400, str(e))

    with CACHE_LOCK:
        stock_map = GLOBAL_CACHE["stock"].get(sym)
        if stock_map is None:
            raise HTTPException(404, f"Symbol {sym} not in cache")

    def _safe(v):
        return None if isinstance(v, float) and (math.isnan(v) or math.isinf(v)) else v

    out = []
    for date_str, row in stock_map.items():
        dt = _date.fromisoformat(date_str)
        if start_date and dt < _date.fromisoformat(start_date):
            continue
        if end_date and dt > _date.fromisoformat(end_date):
            continue
        cleaned = {k: _safe(v) for k, v in row.items()}
        cleaned["date"] = date_str
        out.append(cleaned)

    return {"symbol": sym, "data": out}

# ---------- readiness probe -------------------------------------------------
@app.get("/getready")
def get_ready():
    """
    Returns HTTP-200 {“ready”: true} when the pre-computed cache has been
    loaded into GLOBAL_CACHE, otherwise HTTP-503 {“ready”: false}.
    """
    # We consider the service “ready” when the index cache is non-empty
    with CACHE_LOCK:
        ready = bool(GLOBAL_CACHE["index"])
    if ready:
        return {"ready": True}
    raise HTTPException(503, {"ready": False})

# ---------- manual cache build ----------------------------------------------
if __name__ == "__main__":
    print("Running using "  + INDEX_SYMBOL + ' data')
    cache_file = os.path.join(
        CACHE_DIR, f"full_cache.{INDEX_SYMBOL}.{START_DATE}.{END_DATE}.zst"
    )
    if not os.path.exists(cache_file):
        print("Cache absent – building now …")
        savedata(INDEX_SYMBOL, START_DATE, END_DATE)
    uvicorn.run(app, host="0.0.0.0", port=8000, loop="asyncio", http="httptools")
