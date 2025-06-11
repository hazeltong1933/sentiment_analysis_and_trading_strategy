import pandas as pd
import requests

API_KEY = "AVCRHVMJJ54QV4X8"
url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol=SPY&outputsize=full&apikey={API_KEY}"

response = requests.get(url)
data = response.json()

ts = data['Time Series (Daily)']
df = pd.DataFrame.from_dict(ts, orient='index')
df = df.rename(columns={"5. adjusted close": "sp500_adj_close"})
df['sp500_adj_close'] = df['sp500_adj_close'].astype(float)
df.index = pd.to_datetime(df.index)
df.sort_index(inplace=True)

# Keep only date and adj_close
df_out = df[['sp500_adj_close']].copy()
df_out.index.name = 'date'
df_out.reset_index(inplace=True)

df_out.to_csv("sp500_adj_close.csv", index=False)
print("Saved SP500 adjusted close prices to sp500_adj_close.csv.")
