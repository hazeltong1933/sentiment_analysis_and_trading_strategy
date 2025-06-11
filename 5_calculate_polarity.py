import pandas as pd

def calculate_polarity(sentiment_data):
    polarity_data = []
    
    for (symbol, time_bin), group in sentiment_data.groupby(['symbol', 'time_bin']):
        p = group[group['sentiment'] == 'positive'].shape[0]
        n = group[group['sentiment'] == 'negative'].shape[0]

        if p + n > 0:
            polarity = (p - n) / (p + n)
        else:
            polarity = 0

        polarity_data.append({
            'symbol': symbol,
            'time_bin': pd.to_datetime(time_bin), 
            'polarity': polarity
        })

    return pd.DataFrame(polarity_data)

# Load and process sentiment data
news_sentiment = pd.read_csv('news_relevant_with_sentiment.csv')
news_sentiment['published_timestamp'] = pd.to_datetime(news_sentiment['published_timestamp'])
news_sentiment['time_bin'] = news_sentiment['published_timestamp'].dt.date

# Calculate polarity
polarity_df = calculate_polarity(news_sentiment)

if not polarity_df.empty:
    polarity_df.sort_values(['symbol', 'time_bin'], inplace=True)
    polarity_df['polarity_3d'] = polarity_df.groupby('symbol')['polarity'] \
        .transform(lambda x: x.ewm(span=3, adjust=False).mean())
    polarity_df['polarity_5d'] = polarity_df.groupby('symbol')['polarity'] \
        .transform(lambda x: x.ewm(span=5, adjust=False).mean())

    polarity_df.to_csv("stock_polarity.csv", index=False)
    print("Saved polarity to stock_polarity.csv.")
else: 
    print("No polarity calculation results.")
