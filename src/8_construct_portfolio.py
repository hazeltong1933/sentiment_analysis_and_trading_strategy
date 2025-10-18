import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- Load predictions and close prices ---
transaction_cost = 0.001  # 0.1% per trade

prediction_type = 'directional'

if prediction_type == 'numerical':
    test_df = pd.read_csv("predictions_numerical.csv")
    test_df['position'] = np.where(test_df['predicted_next_return'] > 0, 1,
                               np.where(test_df['predicted_next_return'] <= 0, -1, 0))
else:
    test_df = pd.read_csv("predictions_directional.csv")
    test_df['position'] = np.where(test_df['predicted_next_direction'] == 1, 1,
                               np.where(test_df['predicted_next_direction'] == 0, -1, 0))
test_df['time_bin'] = pd.to_datetime(test_df['time_bin'])  # ensure datetime
price_df = pd.read_csv("stock_price.csv", parse_dates=['timestamp'])

price_df['time_bin'] = price_df['timestamp'].dt.date
daily_df = price_df.groupby(['symbol', 'time_bin']).agg({
    'open': 'first',
    'high': 'max',
    'low': 'min',
    'close': 'last',
    'volume': 'sum'
}).reset_index()
daily_df['time_bin'] = pd.to_datetime(daily_df['time_bin'])
start_date = pd.to_datetime("2023-01-01")
end_date = pd.to_datetime("2024-12-31")
daily_df = daily_df[(daily_df['time_bin'] >= start_date) & (daily_df['time_bin'] <= end_date)]
daily_df = daily_df.sort_values(['symbol', 'time_bin'])
daily_df['return'] = daily_df.groupby('symbol')['close'].pct_change()
daily_df['next_return'] = daily_df.groupby('symbol')['return'].shift(-1)

# Merge close price into test_df
test_df = pd.merge(test_df, daily_df, on=['symbol', 'time_bin'], how='left')

# --- Position assignment ---


# --- Normalize long-short weights ---
def normalize_positions(group):
    longs = group[group['position'] == 1].copy()
    shorts = group[group['position'] == -1].copy()
    neutral = group[group['position'] == 0].copy()

    if not longs.empty:
        longs.loc[:, 'weight'] = 1 / len(longs)
    if not shorts.empty:
        shorts.loc[:, 'weight'] = -1 / len(shorts)
    if not neutral.empty:
        neutral.loc[:, 'weight'] = 0

    weights = pd.concat([longs, shorts, neutral])
    return weights

weighted_positions = test_df.groupby('time_bin', group_keys=False).apply(normalize_positions).reset_index(drop=True)

# --- Merge weights back to test_df ---
test_df = test_df.drop(columns=['weight'], errors='ignore')
test_df = pd.merge(
    test_df.drop(columns=['position'], errors='ignore'),
    weighted_positions[['symbol', 'time_bin', 'weight']],
    on=['symbol', 'time_bin'],
    how='left'
)

# --- Compute weighted returns ---
test_df['weighted_return'] = test_df['weight'] * test_df['next_return']

# --- Compute transaction cost ---
test_df = test_df.sort_values(['symbol', 'time_bin'])
test_df['prev_weight'] = test_df.groupby('symbol')['weight'].shift(1).fillna(0)
test_df['trade_size'] = (test_df['weight'] - test_df['prev_weight']).abs()
test_df['transaction_cost'] = test_df['trade_size'] * transaction_cost
test_df['net_return'] = test_df['weighted_return'] - test_df['transaction_cost']

# --- Aggregate daily returns ---
daily_returns = test_df.groupby('time_bin')['net_return'].sum().sort_index()
daily_returns.index = pd.to_datetime(daily_returns.index)

# --- Cumulative return ---
cumulative_return = (1 + daily_returns).cumprod() - 1

# --- Performance metrics ---
annualized_return = (1 + cumulative_return.iloc[-1]) ** (252 / len(daily_returns)) - 1
annualized_vol = daily_returns.std() * np.sqrt(252)
sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)

print("\n--- Strategy Performance ---")
print(f"Final Cumulative Return: {cumulative_return.iloc[-1]:.2%}")
print(f"Annualized Return: {annualized_return:.2%}")
print(f"Annualized Volatility: {annualized_vol:.2%}")
print(f"Sharpe Ratio: {sharpe_ratio:.2f}")

# --- Plot ---
plt.figure(figsize=(10,6))
plt.plot(cumulative_return.index, cumulative_return.values)
plt.title("Cumulative Return of Long-Short Strategy with Transaction Cost")
plt.xlabel("Date")
plt.ylabel("Cumulative Return")
plt.grid(True)
plt.tight_layout()
plt.show()
