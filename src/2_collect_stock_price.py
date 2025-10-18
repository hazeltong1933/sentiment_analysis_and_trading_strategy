import pandas as pd
from datetime import datetime
import requests

API_KEY = 'AVCRHVMJJ54QV4X8'

def download_price_data(symbol, month):
    print(f"Downloading {symbol} data for {month}...")
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={symbol}&interval=5min&month={month}&outputsize=full&apikey={API_KEY}&extended_hours=false'
    r = requests.get(url)
    data = r.json()

    if 'Time Series (5min)' not in data:
        print(f"Warning: No intraday data for {symbol} in {month}")
        return pd.DataFrame()

    df = pd.DataFrame.from_dict(data['Time Series (5min)'], orient='index')
    df.columns = [col.strip().lower() for col in df.columns]
    df = df.astype(float)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df['symbol'] = symbol
    df['timestamp'] = df.index
    print(df)
    return df

def main():
    
    symbols = ['AAPL', 'MSFT', 'META', 'GOOG', 'AMZN']
    start_date = datetime(2015, 1, 1)
    end_date = datetime(2024, 12, 31)

    all_price_data = pd.DataFrame()

    for symbol in symbols:
        for date in pd.date_range(start_date, end_date, freq='M'):
            month = date.strftime("%Y-%m")
            price_data = download_price_data(symbol, month)
            all_price_data = pd.concat([all_price_data, price_data])

    all_price_data.to_csv("stock_price.csv", index=False)
    print("Stock price data saved to 'stock_price.csv'.")

if __name__ == '__main__':
    main()
