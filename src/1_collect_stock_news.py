import pandas as pd
from datetime import datetime
import requests

API_KEY = 'AVCRHVMJJ54QV4X8'

def download_news_data(symbol, date):
    print(f"Downloading {symbol} news data for {date.strftime('%Y-%m-%d')}...")

    time_from = date.strftime("%Y%m%dT0000")
    time_to = date.strftime("%Y%m%dT2359")
    
    url = (
        f'https://www.alphavantage.co/query?function=NEWS_SENTIMENT'
        f'&tickers={symbol}&time_from={time_from}&time_to={time_to}'
        f'&sort=RELEVANCE&limit=1000&apikey={API_KEY}'
    )
    
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()  # Raise exception for HTTP error codes
        if not r.text.strip().startswith('{'):
            print(f"Warning: Non-JSON response for {symbol} on {date.strftime('%Y-%m-%d')}")
            return pd.DataFrame()
        data = r.json()
    except Exception as e:
        print(f"Error fetching news for {symbol} on {date.strftime('%Y-%m-%d')}: {e}")
        return pd.DataFrame()

    feed = data.get('feed', [])
    if not feed:
        print(f"Warning: No news data found for {symbol} on {date.strftime('%Y-%m-%d')}")
        return pd.DataFrame()

    df = pd.DataFrame([{
        'symbol': symbol,
        'title': item.get('title'),
        'summary': item.get('summary'),
        'published_timestamp': pd.to_datetime(item.get('time_published')),
        'source': item.get('source'),
        'relevance_score': next(
            (t.get('relevance_score') for t in item.get('ticker_sentiment', []) if t.get('ticker') == symbol),
            None
        )
    } for item in feed])

    print(df)
    return df

def main():
    
    symbols = ['AAPL', 'MSFT', 'META', 'GOOG', 'AMZN']
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2024, 12, 31)

    all_news_data = pd.DataFrame()

    for symbol in symbols:
        for date in pd.date_range(start_date, end_date):
            news_data = download_news_data(symbol, date)
            all_news_data = pd.concat([all_news_data, news_data])

    all_news_data.to_csv("news.csv", index=False)
    print("Stock news data saved to 'news.csv'.")

    relevant_news_data = all_news_data[all_news_data['relevance_score']>=0.5]
    relevant_news_data.to_csv("news_relevant.csv")

if __name__ == '__main__':
    main()
