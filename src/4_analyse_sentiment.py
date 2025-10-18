import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn.functional import softmax
import torch

MODEL_NAME = "ProsusAI/finbert"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

label_map = {0: 'negative', 1: 'neutral', 2: 'positive'}

def get_finbert_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = softmax(outputs.logits, dim=1)
        predicted_label = torch.argmax(probs).item()
        return label_map[predicted_label]

def analyze_sentiment_for_file(input_csv, output_csv, title_col, content_col):
    df = pd.read_csv(input_csv)

    sentiments = []
    for index, row in df.iterrows():
        company = row['symbol']
        title = str(row[title_col]) if pd.notnull(row[title_col]) else ''
        content = str(row[content_col]) if pd.notnull(row[content_col]) else ''
        text = f"Title: {title}. Content: {content}."
        sentiment = get_finbert_sentiment(text)
        sentiments.append(sentiment)
        print(f"[{index + 1}/{len(df)}] {company} → {sentiment}")

    df['sentiment'] = sentiments
    df.to_csv(output_csv, index=False)
    print(f"Sentiment analysis saved to {output_csv}")

# Analyze sentiment for news
analyze_sentiment_for_file(
    input_csv="news_relevant.csv",
    output_csv="news_relevant_with_sentiment.csv",
    title_col="title",
    content_col="summary"
)

# Analyze sentiment for earnings calls
analyze_sentiment_for_file(
    input_csv="earning_call_transcripts.csv",
    output_csv="earning_call_transcripts_with_sentiment.csv",
    title_col="title",
    content_col="content"
)

