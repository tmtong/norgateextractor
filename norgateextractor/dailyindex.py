import requests
import os

import pandas as pd
from datetime import date

OUTPUTDIR = 'dailyindexdata/'
INDEXDIR = OUTPUTDIR + 'index/'
TEMPDIR = OUTPUTDIR + 'temp/'
STOCKDIR = OUTPUTDIR + 'stock/'
def prepare_dir():
    for dir in [OUTPUTDIR, TEMPDIR, INDEXDIR, STOCKDIR]:
        if not os.path.isdir(dir):
            os.mkdir(dir)



class StockWebSite:
    def download_file(self, url, outputfilename):
        try:
            response = requests.get(url)
            response.raise_for_status()  # Raise an exception for HTTP errors

            with open( outputfilename, 'wb') as file:
                file.write(response.content)
            print(f"File downloaded successfully and saved to { outputfilename}")
        except requests.exceptions.RequestException as e:
            print(f"An error occurred: {e}")
    def archive(self, indexname):
        indexpath = TEMPDIR + indexname + '.xlsx'
        url = self.indexurls[indexname]
        self.download_file(url, indexpath)
        return indexpath
    def update_csv(self, indexname, df):
        indexpath = INDEXDIR + indexname + '.csv'
        # Extract the Ticker column
        df = df.dropna()
        tickers = df['Ticker'].tolist()
        
        # Format the tickers as a string in the required format
        tickers_str = str(tickers)
        tickers_str = tickers_str.replace(", nan", '')
        
        # Get today's date in YYYY-mm-dd format
        today_date = date.today().strftime('%Y-%m-%d')
        
        # Combine the date and tickers into the final output string
        output_str = today_date + ',' + tickers_str + '\n'
        with open(indexpath, 'a+') as f:
            f.write(output_str )

class SSGA(StockWebSite):
    indexurls = {
        'INDEX-SPY': 'https://www.ssga.com/us/en/intermediary/library-content/products/fund-data/etfs/us/holdings-daily-us-en-spy.xlsx',
        'INDEX-MDY': 'https://www.ssga.com/us/en/intermediary/library-content/products/fund-data/etfs/us/holdings-daily-us-en-mdy.xlsx'
    }
   


    def clean_excelfile(self, filepath):
        # Read the Excel file
        df = pd.read_excel(filepath, header=None)
        
        # Find the index of the row that starts with "Name" (the actual header)
        header_row = df[df.iloc[:, 0] == 'Name'].index[0]
        
        # Set the header row as the new header
        df.columns = df.iloc[header_row]
        
        # Remove all rows before the header row
        df = df[header_row + 1:]
        
        # Reset the index
        df.reset_index(drop=True, inplace=True)
        
        return df

   
    def dailyrun(self):
        for indexname in self.indexurls.keys():
            xlsxpath = self.archive(indexname)
            df = self.clean_excelfile(xlsxpath)
            self.update_csv(indexname, df)            

class ISHARE(StockWebSite):
    indexurls = {
        'INDEX-ACWI': 'https://www.ishares.com/us/products/239696/ishares-msci-world-etf/1467271812596.ajax?fileType=csv&fileName=URTH_holdings&dataType=fund'
    }

    def clean_excelfile(self, filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        start = 0
        end = len(lines)
        for line in lines:
            if line.startswith("Ticker"):
                break
            start = start + 1
        for index in range(len(lines) - 1, 0, -1):
            line = lines[index]
            if line.startswith('"'):
                break
            end = end - 1
        lines = lines[start:end - 1]
        content = '\n'.join(lines)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        # Read the Excel file
        df = pd.read_csv(filepath, header=None)
        df = df.dropna()
        
        # Find the index of the row that starts with "Name" (the actual header)
        header_row = df[df.iloc[:, 0] == 'Ticker'].index[0]
        
        # Set the header row as the new header
        df.columns = df.iloc[header_row]
        
        # Remove all rows before the header row
        df = df[header_row + 1:]
        
        # Reset the index
        df.reset_index(drop=True, inplace=True)
        
        return df

    def dailyrun(self):
        for indexname in self.indexurls.keys():
            xlsxpath = self.archive(indexname)
            df = self.clean_excelfile(xlsxpath)
            self.update_csv(indexname, df)     

if __name__ == "__main__":
    prepare_dir()
    # ssga = SSGA()
    # ssga.dailyrun()
    ishare = ISHARE()
    ishare.dailyrun()
    

