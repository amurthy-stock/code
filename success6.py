# Fibonacci LSTM Forecast Multi-Timeframe Script (with progress bars)

import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.callbacks import EarlyStopping
import gc
import csv
import os
import sys
import argparse
from tqdm import tqdm

# --- Parse command-line arguments ---
def parse_arguments():
    """
    Parse command-line arguments for the LSTM forecast script.
    
    Returns:
        argparse.Namespace: Parsed arguments containing:
            - ticker_file: Path to file containing ticker symbols
            - output_dir: Directory for output files
            - batch_size: Number of tickers to process in each batch
    """
    parser = argparse.ArgumentParser(description='Fibonacci LSTM Forecast Multi-Timeframe Script')
    parser.add_argument('--ticker-file', type=str, 
                       default=os.getenv('TICKER_FILE', '/content/drive/MyDrive/Outputs/Test_ticker.txt'),
                       help='Path to ticker file (default: from TICKER_FILE env var or Google Drive path)')
    parser.add_argument('--output-dir', type=str,
                       default=os.getenv('OUTPUT_DIR', '/content/drive/MyDrive/Outputs'),
                       help='Output directory for results (default: from OUTPUT_DIR env var or Google Drive path)')
    parser.add_argument('--batch-size', type=int, default=5,
                       help='Batch size for processing tickers (default: 5)')
    return parser.parse_args()

# Parse arguments
args = parse_arguments()

# --- MOUNT GOOGLE DRIVE IN COLAB (if available) ---
try:
    from google.colab import drive
    drive.mount('/content/drive')
except ImportError:
    print("Google Colab not detected, skipping drive mount")
except Exception as e:
    print(f"Warning: Failed to mount Google Drive: {str(e)}")

# --- Load tickers from file and verify ---
ticker_file = args.ticker_file
if not os.path.exists(ticker_file):
    raise FileNotFoundError(f"Ticker file '{ticker_file}' not found. Please check the file path.")

with open(ticker_file, 'r') as f:
    tickers = [line.strip() for line in f if line.strip()]

if len(tickers) == 0:
    raise ValueError("No tickers found in file. Please check the contents of the ticker file!")

print(f"Loaded {len(tickers)} tickers from {ticker_file}")

def batch_tickers(ticker_list, batch_size):
    """
    Generator function to yield batches of tickers for rate-limited API calls.
    
    Args:
        ticker_list: List of ticker symbols
        batch_size: Number of tickers per batch
        
    Yields:
        List of tickers in each batch
    """
    for i in range(0, len(ticker_list), batch_size):
        yield ticker_list[i:i+batch_size]

batch_size = args.batch_size  # Configurable via command line

# --- Helper Functions ---
def build_lstm_model(input_shape):
    """
    Build and compile an LSTM model for time series forecasting.
    
    Args:
        input_shape: Tuple of (timesteps, features) for the input layer
        
    Returns:
        Compiled Keras Sequential model
    """
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

# --- Weekly LSTM Forecast ---
print("\n" + "="*60)
print("STARTING WEEKLY LSTM FORECAST")
print("="*60)
results = []
window_size = 7
forecast_steps = [1,2,3]
fib_idxs = [-13, -8, -5, -3, -2, -1]

for ticker_batch in batch_tickers(tickers, batch_size):
    data_batch = yf.download(ticker_batch, period='2y', interval='1wk', auto_adjust=True, progress=False, group_by='ticker')
    for ticker in tqdm(ticker_batch, desc="Processing Weekly Tickers", leave=False):
        try:
            data = data_batch[ticker] if len(ticker_batch) > 1 else data_batch
            if data is None or 'Close' not in data:
                continue
            closes = data['Close'].values
            closes = [float(np.squeeze(c)) for c in closes if c is not None]
            if len(closes) < abs(fib_idxs[0]):
                continue

            scaler = MinMaxScaler()
            closes_scaled = scaler.fit_transform(np.array(closes).reshape(-1, 1)).flatten()
            X, y = [], []
            for i in range(len(closes_scaled) - window_size - max(forecast_steps) + 1):
                X.append(closes_scaled[i:i+window_size])
                y.append(closes_scaled[i+window_size+max(forecast_steps)-1])
            X, y = np.array(X), np.array(y)
            if len(X) == 0 or len(y) == 0:
                continue
            X = X.reshape((X.shape[0], X.shape[1], 1))
            split = int(0.8 * len(X))
            X_train, X_test = X[:split], X[split:]
            y_train, y_test = y[:split], y[split:]

            model = build_lstm_model((window_size, 1))
            es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            model.fit(X_train, y_train, epochs=100, batch_size=8,
                      validation_data=(X_test, y_test), callbacks=[es], verbose=0)

            y_pred = model.predict(X_test, verbose=0)
            y_pred_inv = scaler.inverse_transform(y_pred)
            y_test_inv = scaler.inverse_transform(y_test.reshape(-1,1))
            mape_rounded = round(np.mean(np.abs((y_test_inv - y_pred_inv) / y_test_inv)) * 100, 1)

            fib_points = [closes[i] for i in fib_idxs]
            last_window = closes_scaled[-window_size:]
            last_window = last_window.reshape((1, window_size, 1))

            forecasted = {}
            for step in forecast_steps:
                window = last_window.copy()
                pred_scaled = None
                for s in range(1, step + 1):
                    pred_scaled = model.predict(window, verbose=0)[0][0]
                    new_window = np.append(window.flatten()[1:], pred_scaled)
                    window = new_window.reshape((1, window_size, 1))
                forecasted[f'+{step}P'] = scaler.inverse_transform([[pred_scaled]])[0][0]

            last_close = fib_points[5]
            gain = max([forecasted['+1P'], forecasted['+2P'], forecasted['+3P']]) - last_close

            row = {
                'Symbol': ticker,
                '13P': fib_points[0],
                '8P': fib_points[1],
                '5P': fib_points[2],
                '3P': fib_points[3],
                '2P': fib_points[4],
                '1P': fib_points[5],
                'Last Close': last_close,
                '+1P': forecasted['+1P'],
                '+2P': forecasted['+2P'],
                '+3P': forecasted['+3P'],
                'Gain': gain,
                'Test MAPE (%)': mape_rounded
            }
            row['cum_gain'] = [
                0,
                row['8P'] - row['13P'],
                row['5P'] - row['13P'],
                row['3P'] - row['13P'],
                row['2P'] - row['13P'],
                row['1P'] - row['13P'],
                row['+1P'] - row['13P'],
                row['+2P'] - row['13P'],
                row['+3P'] - row['13P'],
            ]
            results.append(row)
        except Exception as e:
            print(f"Error processing {ticker} (weekly): {str(e)}", file=sys.stderr)
    time.sleep(2)

results_sorted = sorted([row for row in results if row['Gain'] > 0], key=lambda x: x['Gain'], reverse=True)[:15]
print(f"Weekly forecast complete: {len(results)} processed, {len(results_sorted)} with positive gain")

# --- Daily LSTM Forecast ---
print("\n" + "="*60)
print("STARTING DAILY LSTM FORECAST")
print("="*60)
daily_results = []
daily_window_size = 12
daily_forecast_steps = [1,2,5]
daily_fib_idxs = [-21, -13, -8, -5, -3, -2, -1]

for ticker_batch in batch_tickers(tickers, batch_size):
    data_batch = yf.download(ticker_batch, period='60d', interval='1d', auto_adjust=True, progress=False, group_by='ticker')
    for ticker in tqdm(ticker_batch, desc="Processing Daily Tickers", leave=False):
        try:
            data = data_batch[ticker] if len(ticker_batch) > 1 else data_batch
            if data is None or 'Close' not in data:
                continue
            closes = data['Close'].values
            closes = [float(np.squeeze(c)) for c in closes if c is not None]
            if len(closes) < abs(daily_fib_idxs[0]):
                continue

            scaler = MinMaxScaler()
            closes_scaled = scaler.fit_transform(np.array(closes).reshape(-1, 1)).flatten()
            X, y = [], []
            for i in range(len(closes_scaled) - daily_window_size - max(daily_forecast_steps) + 1):
                X.append(closes_scaled[i:i+daily_window_size])
                y.append(closes_scaled[i+daily_window_size+max(daily_forecast_steps)-1])
            X, y = np.array(X), np.array(y)
            if len(X) == 0 or len(y) == 0:
                continue
            X = X.reshape((X.shape[0], X.shape[1], 1))
            split = int(0.8 * len(X))
            X_train, X_test = X[:split], X[split:]
            y_train, y_test = y[:split], y[split:]

            model = build_lstm_model((daily_window_size, 1))
            es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            model.fit(X_train, y_train, epochs=100, batch_size=8,
                      validation_data=(X_test, y_test), callbacks=[es], verbose=0)

            y_pred = model.predict(X_test, verbose=0)
            y_pred_inv = scaler.inverse_transform(y_pred)
            y_test_inv = scaler.inverse_transform(y_test.reshape(-1,1))
            mape_rounded = round(np.mean(np.abs((y_test_inv - y_pred_inv) / y_test_inv)) * 100, 1)

            fib_points = [closes[i] for i in daily_fib_idxs]
            last_window = closes_scaled[-daily_window_size:]
            last_window = last_window.reshape((1, daily_window_size, 1))

            forecasted = {}
            for step in daily_forecast_steps:
                window = last_window.copy()
                pred_scaled = None
                for s in range(1, step + 1):
                    pred_scaled = model.predict(window, verbose=0)[0][0]
                    new_window = np.append(window.flatten()[1:], pred_scaled)
                    window = new_window.reshape((1, daily_window_size, 1))
                forecasted[f'+{step}P'] = scaler.inverse_transform([[pred_scaled]])[0][0]

            last_close = fib_points[6]
            gain = max([forecasted['+1P'], forecasted['+2P'], forecasted['+5P']]) - last_close

            row = {
                'Symbol': ticker,
                '21P': fib_points[0],
                '13P': fib_points[1],
                '8P': fib_points[2],
                '5P': fib_points[3],
                '3P': fib_points[4],
                '2P': fib_points[5],
                '1P': fib_points[6],
                'Last Close': last_close,
                '+1P': forecasted['+1P'],
                '+2P': forecasted['+2P'],
                '+5P': forecasted['+5P'],
                'Gain': gain,
                'Test MAPE (%)': mape_rounded
            }
            row['cum_gain'] = [
                0,
                row['13P'] - row['21P'],
                row['8P'] - row['21P'],
                row['5P'] - row['21P'],
                row['3P'] - row['21P'],
                row['2P'] - row['21P'],
                row['1P'] - row['21P'],
                row['+1P'] - row['21P'],
                row['+2P'] - row['21P'],
                row['+5P'] - row['21P'],
            ]
            daily_results.append(row)
        except Exception as e:
            print(f"Error processing {ticker} (daily): {str(e)}", file=sys.stderr)
    time.sleep(2)

daily_results_sorted = sorted([row for row in daily_results if row['Gain'] > 0], key=lambda x: x['Gain'], reverse=True)[:15]
print(f"Daily forecast complete: {len(daily_results)} processed, {len(daily_results_sorted)} with positive gain")

# --- Hourly (60m) LSTM Forecast ---
print("\n" + "="*60)
print("STARTING HOURLY (60m) LSTM FORECAST")
print("="*60)
hourly_results = []
hourly_window_size = 12
hourly_forecast_steps = [3, 5, 8]
hourly_fib_idxs = [-34, -21, -13, -8, -5, -3, -2, -1]

for ticker_batch in batch_tickers(tickers, batch_size):
    data_batch = yf.download(
        ticker_batch,
        period='5d',
        interval='60m',
        auto_adjust=True,
        progress=False,
        group_by='ticker'
    )
    for ticker in tqdm(ticker_batch, desc="Processing Hourly Tickers", leave=False):
        try:
            if len(ticker_batch) > 1:
                data = data_batch.get(ticker)
            else:
                data = data_batch
            if data is None or 'Close' not in data:
                continue

            closes = data['Close'].values
            closes = [float(np.squeeze(c)) for c in closes if c is not None]
            if len(closes) < abs(hourly_fib_idxs[0]):
                continue

            scaler = MinMaxScaler()
            closes_scaled = scaler.fit_transform(np.array(closes).reshape(-1, 1)).flatten()
            X, y = [], []
            for i in range(len(closes_scaled) - hourly_window_size - max(hourly_forecast_steps) + 1):
                X.append(closes_scaled[i:i+hourly_window_size])
                y.append(closes_scaled[i+hourly_window_size+max(hourly_forecast_steps)-1])
            X, y = np.array(X), np.array(y)
            if len(X) == 0 or len(y) == 0:
                continue

            X = X.reshape((X.shape[0], X.shape[1], 1))
            split = int(0.8 * len(X))
            X_train, X_test = X[:split], X[split:]
            y_train, y_test = y[:split], y[split:]

            model = build_lstm_model((hourly_window_size, 1))
            es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            model.fit(X_train, y_train, epochs=100, batch_size=8,
                      validation_data=(X_test, y_test), callbacks=[es], verbose=0)

            y_pred = model.predict(X_test, verbose=0)
            y_pred_inv = scaler.inverse_transform(y_pred)
            y_test_inv = scaler.inverse_transform(y_test.reshape(-1,1))
            mape_rounded = round(np.mean(np.abs((y_test_inv - y_pred_inv) / y_test_inv)) * 100, 1)

            fib_points = [closes[i] for i in hourly_fib_idxs]
            last_window = closes_scaled[-hourly_window_size:]
            last_window = last_window.reshape((1, hourly_window_size, 1))

            forecasted = {}
            for step in hourly_forecast_steps:
                window = last_window.copy()
                pred_scaled = None
                for s in range(1, step + 1):
                    pred_scaled = model.predict(window, verbose=0)[0][0]
                    new_window = np.append(window.flatten()[1:], pred_scaled)
                    window = new_window.reshape((1, hourly_window_size, 1))
                forecasted[f'+{step}P_Hourly'] = scaler.inverse_transform([[pred_scaled]])[0][0]

            last_close = fib_points[7]
            gain_hourly = max([
                forecasted['+3P_Hourly'],
                forecasted['+5P_Hourly'],
                forecasted['+8P_Hourly']
            ]) - last_close

            if gain_hourly > 0:
                row = {
                    'Symbol_daily': ticker,
                    '34P': fib_points[0],
                    '21P': fib_points[1],
                    '13P_daily': fib_points[2],
                    '8P_daily': fib_points[3],
                    '5P_daily': fib_points[4],
                    '3P_daily': fib_points[5],
                    '2P_daily': fib_points[6],
                    '1P_daily': fib_points[7],
                    'Last Close_daily': last_close,
                    '+3P_Hourly': forecasted['+3P_Hourly'],
                    '+5P_Hourly': forecasted['+5P_Hourly'],
                    '+8P_Hourly': forecasted['+8P_Hourly'],
                    'Gain_Hourly': gain_hourly,
                    'Test MAPE (%)_daily': mape_rounded
                }
                row['cum_gain'] = [
                    0,
                    row['21P'] - row['34P'],
                    row['13P_daily'] - row['34P'],
                    row['8P_daily'] - row['34P'],
                    row['5P_daily'] - row['34P'],
                    row['3P_daily'] - row['34P'],
                    row['2P_daily'] - row['34P'],
                    row['1P_daily'] - row['34P'],
                    row['+3P_Hourly'] - row['34P'],
                    row['+5P_Hourly'] - row['34P'],
                    row['+8P_Hourly'] - row['34P'],
                ]
                hourly_results.append(row)
        except Exception as e:
            print(f"Error processing {ticker} (hourly): {str(e)}", file=sys.stderr)
    time.sleep(2)

hourly_results_sorted = sorted(hourly_results, key=lambda x: x['Gain_Hourly'], reverse=True)[:15]
print(f"Hourly forecast complete: {len(hourly_results)} processed, {len(hourly_results_sorted)} with positive gain")

# --- Combined Output CSV ---
print("\n" + "="*60)
print("GENERATING OUTPUT FILES")
print("="*60)
output_dir = args.output_dir
os.makedirs(output_dir, exist_ok=True)
csv_path = os.path.join(output_dir, 'Fibonacci_LSTM_results.csv')
with open(csv_path, 'w', newline='') as csvfile:
    fieldnames = [
        # Weekly
        "Symbol", "13P", "8P", "5P", "3P", "2P", "1P", "Last Close", "+1P", "+2P", "+3P", "Gain", "Test MAPE (%)",
        "", "", "", "", "",  # Blank columns

        # Daily
        "Symbol_daily", "21P", "13P_daily", "8P_daily", "5P_daily", "3P_daily", "2P_daily", "1P_daily", "Last Close_daily",
        "+1P_daily", "+2P_daily", "+5P_daily", "Gain_daily", "Test MAPE (%)_daily",
        "", "", "",  # Blank columns

        # Hourly
        "Symbol_hourly", "34P", "21P_hourly", "13P_hourly", "8P_hourly", "5P_hourly", "3P_hourly", "2P_hourly", "1P_hourly", "Last Close_hourly",
        "+3P_Hourly", "+5P_Hourly", "+8P_Hourly", "Gain_Hourly", "Test MAPE (%)_hourly"
    ]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    max_len = max(len(results_sorted), len(daily_results_sorted), len(hourly_results_sorted))
    for i in range(max_len):
        row = {}

        # Weekly
        if i < len(results_sorted):
            r = results_sorted[i]
            row["Symbol"] = r.get("Symbol", "")
            row["13P"] = r.get("13P", "")
            row["8P"] = r.get("8P", "")
            row["5P"] = r.get("5P", "")
            row["3P"] = r.get("3P", "")
            row["2P"] = r.get("2P", "")
            row["1P"] = r.get("1P", "")
            row["Last Close"] = r.get("Last Close", "")
            row["+1P"] = r.get("+1P", "")
            row["+2P"] = r.get("+2P", "")
            row["+3P"] = r.get("+3P", "")
            row["Gain"] = r.get("Gain", "")
            row["Test MAPE (%)"] = r.get("Test MAPE (%)", "")
        else:
            for key in ["Symbol", "13P", "8P", "5P", "3P", "2P", "1P", "Last Close", "+1P", "+2P", "+3P", "Gain", "Test MAPE (%)"]:
                row[key] = ""

        # Daily
        if i < len(daily_results_sorted):
            d = daily_results_sorted[i]
            row["Symbol_daily"] = d.get("Symbol", "")
            row["21P"] = d.get("21P", "")
            row["13P_daily"] = d.get("13P", "")
            row["8P_daily"] = d.get("8P", "")
            row["5P_daily"] = d.get("5P", "")
            row["3P_daily"] = d.get("3P", "")
            row["2P_daily"] = d.get("2P", "")
            row["1P_daily"] = d.get("1P", "")
            row["Last Close_daily"] = d.get("Last Close", "")
            row["+1P_daily"] = d.get("+1P", "")
            row["+2P_daily"] = d.get("+2P", "")
            row["+5P_daily"] = d.get("+5P", "")
            row["Gain_daily"] = d.get("Gain", "")
            row["Test MAPE (%)_daily"] = d.get("Test MAPE (%)", "")
        else:
            for key in [
                "Symbol_daily", "21P", "13P_daily", "8P_daily", "5P_daily",
                "3P_daily", "2P_daily", "1P_daily", "Last Close_daily",
                "+1P_daily", "+2P_daily", "+5P_daily", "Gain_daily", "Test MAPE (%)_daily"
            ]:
                row[key] = ""

        # Hourly
        if i < len(hourly_results_sorted):
            h = hourly_results_sorted[i]
            row["Symbol_hourly"] = h.get("Symbol_daily", "")
            row["34P"] = h.get("34P", "")
            row["21P_hourly"] = h.get("21P", "")
            row["13P_hourly"] = h.get("13P_daily", "")
            row["8P_hourly"] = h.get("8P_daily", "")
            row["5P_hourly"] = h.get("5P_daily", "")
            row["3P_hourly"] = h.get("3P_daily", "")
            row["2P_hourly"] = h.get("2P_daily", "")
            row["1P_hourly"] = h.get("1P_daily", "")
            row["Last Close_hourly"] = h.get("Last Close_daily", "")
            row["+3P_Hourly"] = h.get("+3P_Hourly", "")
            row["+5P_Hourly"] = h.get("+5P_Hourly", "")
            row["+8P_Hourly"] = h.get("+8P_Hourly", "")
            row["Gain_Hourly"] = h.get("Gain_Hourly", "")
            row["Test MAPE (%)_hourly"] = h.get("Test MAPE (%)_daily", "")
        else:
            for key in [
                "Symbol_hourly", "34P", "21P_hourly", "13P_hourly", "8P_hourly", "5P_hourly",
                "3P_hourly", "2P_hourly", "1P_hourly", "Last Close_hourly",
                "+3P_Hourly", "+5P_Hourly", "+8P_Hourly", "Gain_Hourly", "Test MAPE (%)_hourly"
            ]:
                row[key] = ""

        writer.writerow(row)

print(f"All results (weekly, daily, hourly) saved to {csv_path}")

# --- Combined Chart: Weekly, Daily, Hourly ---
points_weekly = ["13P", "8P", "5P", "3P", "2P", "1P", "+1P", "+2P", "+3P"]
points_daily = ["21P", "13P", "8P", "5P", "3P", "2P", "1P", "+1P", "+2P", "+5P"]
points_hourly = ["34P", "21P", "13P_daily", "8P_daily", "5P_daily", "3P_daily", "2P_daily", "1P_daily", "+3P_Hourly", "+5P_Hourly", "+8P_Hourly"]

fig, axs = plt.subplots(1, 3, figsize=(30, 8))

# Weekly Chart
axs[0].set_title("Weekly Cumulative Gain (Top 15 by Gain)")
for row in results_sorted:
    x = np.arange(len(points_weekly))
    y = row['cum_gain']
    axs[0].plot(points_weekly, y, marker='o', linestyle='-', label=row['Symbol'])
    axs[0].text(x[-1], y[-1], row['Symbol'], fontsize=11, fontweight='bold', color='blue', va='center', ha='left')
axs[0].set_xlabel("Fibonacci Points")
axs[0].set_ylabel("Cumulative Gain")
axs[0].grid(True)

# Daily Chart
axs[1].set_title("Daily Cumulative Gain (Top 15 by Gain)")
for row in daily_results_sorted:
    x = np.arange(len(points_daily))
    y = row['cum_gain']
    axs[1].plot(points_daily, y, marker='x', linestyle='--', label=row['Symbol'])
    axs[1].text(x[-1], y[-1], row['Symbol'], fontsize=11, fontweight='bold', color='green', va='center', ha='left')
axs[1].set_xlabel("Fibonacci Points")
axs[1].set_ylabel("Cumulative Gain")
axs[1].grid(True)

# Hourly Chart (Only Positive Gain)
for row in hourly_results_sorted:
    x = np.arange(len(points_hourly))
    y = row['cum_gain']
    axs[2].plot(points_hourly, y, marker='s', linestyle='-', label=row['Symbol_daily'])
    axs[2].text(x[-1], y[-1], row['Symbol_daily'], fontsize=11, fontweight='bold', color='purple', va='center', ha='left')
axs[2].set_title("Hourly Cumulative Gain (Top 15 by Gain, Positive Only)")
axs[2].set_xlabel("Fibonacci Points")
axs[2].set_ylabel("Cumulative Gain")
axs[2].grid(True)

plt.tight_layout()
combined_chart_path = os.path.join(output_dir, 'Fibonacci_LSTM_combined_cumgain.png')
plt.savefig(combined_chart_path)
plt.show()
print(f"Weekly, Daily, and Hourly cumulative gain charts saved to {combined_chart_path}")

# ---------------- FINAL PART OF SCRIPT ----------------
print("\n========== LSTM Fibonacci Forecast Run Complete ==========")
print("Output files generated:")
print(f"  - Results CSV: {csv_path}")
print(f"  - Combined Cumulative Gain Chart PNG: {combined_chart_path}")
print("\nReview your charts and CSVs in Google Drive. If you need further analysis, visualization, or export options, just ask!")
print("\n=========================================================")

tf.keras.backend.clear_session()
gc.collect()
print("TensorFlow session cleared and memory released.")

print("--------END OF SCRIPT----------------")