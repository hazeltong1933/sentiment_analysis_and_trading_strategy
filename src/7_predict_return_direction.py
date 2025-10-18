import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier, plot_importance
import matplotlib.pyplot as plt
import ta

# Load data
price_df = pd.read_csv("stock_price.csv", parse_dates=['timestamp'])
polarity_df = pd.read_csv("stock_polarity.csv", parse_dates=['time_bin'])
sp500_df = pd.read_csv("sp500_adj_close.csv", parse_dates=['date'])

# Convert to daily bin for price
price_df['time_bin'] = price_df['timestamp'].dt.date
daily_df = price_df.groupby(['symbol', 'time_bin']).agg({
    'open': 'first',
    'high': 'max',
    'low': 'min',
    'close': 'last',
    'volume': 'sum'
}).reset_index()

# Convert daily_df time_bin to datetime for filtering
daily_df['time_bin'] = pd.to_datetime(daily_df['time_bin'])

# Date filter for 2023-2024
start_date = pd.to_datetime("2023-01-01")
end_date = pd.to_datetime("2024-12-31")
daily_df = daily_df[(daily_df['time_bin'] >= start_date) & (daily_df['time_bin'] <= end_date)]

# Sort for rolling calculations
daily_df = daily_df.sort_values(['symbol', 'time_bin'])

# Calculate returns
daily_df['return'] = daily_df.groupby('symbol')['close'].pct_change()

def compute_ta(group):
    group['rsi'] = ta.momentum.RSIIndicator(close=group['close'], window=14).rsi()
    macd = ta.trend.MACD(close=group['close'])
    group['macd'] = macd.macd()
    group['macd_signal'] = macd.macd_signal()
    group['macd_diff'] = macd.macd_diff()
    return group

daily_df = daily_df.groupby('symbol').apply(compute_ta).reset_index(drop=True)

# Rolling features
for window in [3, 5]:
    daily_df[f'rolling_return_{window}d'] = daily_df.groupby('symbol')['return'].rolling(window).mean().reset_index(0, drop=True)
    daily_df[f'volatility_{window}d'] = daily_df.groupby('symbol')['return'].rolling(window).std().reset_index(0, drop=True)
    daily_df[f'volume_change_{window}d'] = daily_df.groupby('symbol')['volume'].pct_change(window)

# Lag next-day return, then convert to direction: 1 if next return > 0 else 0
daily_df['next_return'] = daily_df.groupby('symbol')['return'].shift(-1)
daily_df['next_direction'] = daily_df['next_return'].apply(lambda x: 1 if x > 0 else 0)

# Filter polarity for date range
polarity_df = polarity_df[(polarity_df['time_bin'] >= start_date) & (polarity_df['time_bin'] <= end_date)]

# Merge polarity
merged_df = pd.merge(daily_df, polarity_df[['symbol', 'time_bin', 'polarity', 'polarity_3d', 'polarity_5d']],
                     on=['symbol', 'time_bin'], how='left')

# Process SP500
sp500_df = sp500_df.sort_values('date')
sp500_df['sp500_return'] = sp500_df['sp500_adj_close'].pct_change()
sp500_df = sp500_df.rename(columns={'date': 'time_bin'})
sp500_df = sp500_df[(sp500_df['time_bin'] >= start_date) & (sp500_df['time_bin'] <= end_date)]

# Merge SP500 data
merged_df = pd.merge(merged_df, sp500_df[['time_bin', 'sp500_adj_close', 'sp500_return']], on='time_bin', how='left')

# Drop NA target rows and sort by time
model_df = merged_df.dropna(subset=['next_direction']).sort_values('time_bin')
features = [
    'volume', 'return',
    'rsi', 'macd_diff',
    'rolling_return_3d', 'rolling_return_5d',
    'volatility_3d', 'volatility_5d',
    'volume_change_3d', 'volume_change_5d',
    'polarity', 'polarity_3d', 'polarity_5d',
    'sp500_return'
]

X = model_df[features]
y = model_df['next_direction']

# TimeSeries cross-validation with XGBClassifier
tscv = TimeSeriesSplit(n_splits=5)
model = XGBClassifier(n_estimators=1000, learning_rate=0.05, max_depth=3, use_label_encoder=False, eval_metric='logloss', random_state=42)

print("Time Series Cross-Validation Results:")
for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    model.fit(X_train, y_train,
              eval_set=[(X_test, y_test)],
              verbose=False)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"Fold {fold} — Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred, zero_division=0))

plot_importance(model)
plt.title("Feature Importance (Last Fold Model)")
plt.show()

# Final hold-out evaluation (80/20 time split)
split_index = int(len(X) * 0.8)
X_train_final, X_test_final = X.iloc[:split_index], X.iloc[split_index:]
y_train_final, y_test_final = y.iloc[:split_index], y.iloc[split_index:]

final_model = XGBClassifier(n_estimators=1000, learning_rate=0.05, max_depth=3, use_label_encoder=False, eval_metric='logloss', random_state=42)
final_model.fit(X_train_final, y_train_final)

y_pred_final = final_model.predict(X_test_final)

acc_final = accuracy_score(y_test_final, y_pred_final)
print("\n--- Final 80/20 Split Evaluation ---")
print(f"Final Accuracy: {acc_final:.4f}")
print(classification_report(y_test_final, y_pred_final, zero_division=0))
print("Confusion Matrix:")
print(confusion_matrix(y_test_final, y_pred_final))

plot_importance(final_model)
plt.title("Feature Importance (Final 80/20 Model)")
plt.show()

# Save final predictions
results_df = model_df.iloc[split_index:].copy()
results_df['predicted_next_direction'] = y_pred_final

results_df[['symbol', 'time_bin', 'predicted_next_direction']].to_csv("predictions_directional.csv", index=False)

print("Final predictions saved to 'predictions_directional.csv'.")
