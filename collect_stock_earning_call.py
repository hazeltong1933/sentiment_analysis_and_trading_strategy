import pandas as pd
import requests

API_KEY = 'AVCRHVMJJ54QV4X8'

def download_earning_call(symbol, quarter):
    print(f"Downloading {symbol} earnings call transcript for {quarter}...")

    url = (
        f'https://www.alphavantage.co/query?function=EARNINGS_CALL_TRANSCRIPT'
        f'&symbol={symbol}&quarter={quarter}&apikey={API_KEY}'
    )

    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()  # Raise exception for HTTP error codes
        if not r.text.strip().startswith('{'):
            print(f"Warning: Non-JSON response for {symbol} on {quarter}")
            return pd.DataFrame()
        data = r.json()
    except Exception as e:
        print(f"Error fetching transcript for {symbol} on {quarter}: {e}")
        return pd.DataFrame()
    
    transcript = data.get('transcript', [])
    if not transcript:
        print(f"Warning: No transcript data for {symbol} {quarter}")
        return pd.DataFrame()

    df = pd.DataFrame([{
        'symbol': symbol,
        'quarter': quarter,
        'speaker': section.get('speaker'),
        'title': section.get('title'),
        'content': section.get('content')
    } for section in transcript])

    print(df)
    return df

def main():
    
    symbols = ['AAPL', 'MSFT', 'META', 'GOOG', 'AMZN']
    start_year = 2010
    end_year = 2024

    all_transcripts = pd.DataFrame()

    for symbol in symbols:
        for year in range(start_year, end_year + 1):
            for q in range(1, 5):
                quarter = f"{year}Q{q}"
                df = download_earning_call(symbol, quarter)
                if not df.empty:
                    all_transcripts = pd.concat([all_transcripts, df], ignore_index=True)

    all_transcripts.to_csv("earning_call_transcripts.csv", index=False)
    print("Earning call transcripts saved to 'earning_call_transcripts.csv'.")

if __name__ == '__main__':
    main()
