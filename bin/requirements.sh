#!/usr/bin/env bash
pip install --upgrade pip setuptools wheel
pip install pandas requests httptools tqdm matplotlib pathos psutil fastapi uvicorn pandas_market_calendars scipy filelock pytz pyarrow zstandard ta-lib
echo ""
echo "✅ All dependencies installed."
echo ""
echo "👉 To activate the virtual environment:"
echo "   source $VENV_DIR/bin/activate"
echo ""
echo "👉 To run the downloader:"
echo "   python fmpextractor/downloaddata.py"
