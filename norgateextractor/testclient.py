import requests
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, List

# Base URL of your running FastAPI server
BASE_URL = "http://localhost:8000"

def get_index_constituents(index_symbol: str, date: str) -> Optional[dict]:
    url = f"{BASE_URL}/indexconstituents"
    payload = {"indexsymbol": index_symbol, "date": date}
    try:
        start = time.time()
        response = requests.post(url, json=payload)
        duration = time.time() - start
        print(f"[Index] {index_symbol} @ {date} | Status: {response.status_code} | Time: {duration:.4f}s")
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"Request failed: {e}")
    return None


def get_stock_data(symbol: str, date: str) -> Optional[dict]:
    url = f"{BASE_URL}/stockdata"
    payload = {"symbol": symbol, "date": date}
    try:
        start = time.time()
        response = requests.post(url, json=payload)
        duration = time.time() - start
        print(f"[Stock] {symbol} @ {date} | Status: {response.status_code} | Time: {duration:.4f}s")
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"Request failed: {e}")
    return None


def test_single_requests():
    print("\n=== Testing Single Requests ===")
    print(get_index_constituents("SPX", "2023-12-29"))
    print(get_stock_data("AAPL", "2023-12-29"))


def test_concurrent_requests():
    print("\n=== Testing Concurrent Requests ===")
    requests = [
        ("AAPL", "2023-12-29"),
        ("GOOG", "2023-12-29"),
        ("MSFT", "2023-12-29"),
        ("TSLA", "2023-12-29"),
        ("SPX", "2023-12-29"),
    ]

    def task(symbol, date):
        print(get_stock_data(symbol, date))

    with ThreadPoolExecutor(max_workers=5) as executor:
        for symbol, date in requests:
            executor.submit(task, symbol, date)


if __name__ == "__main__":
    # test_single_requests()
    test_concurrent_requests()