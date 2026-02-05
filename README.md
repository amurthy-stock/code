# Fibonacci LSTM Forecast Script

This repository contains a Python script for multi-timeframe stock price forecasting using LSTM (Long Short-Term Memory) neural networks and Fibonacci retracement levels.

## Overview

The `fibonacci_lstm_forecast.py` script performs the following tasks:
- Downloads stock price data using Yahoo Finance API
- Trains LSTM models on weekly, daily, and hourly timeframes
- Generates forecasts based on Fibonacci retracement points
- Creates cumulative gain charts and CSV reports
- Saves results to Google Drive (when run in Google Colab)

## Features

- **Multi-timeframe Analysis**: Weekly, daily, and hourly forecasts
- **Fibonacci Points**: Uses Fibonacci retracement levels for analysis
- **LSTM Neural Networks**: Deep learning for time series prediction
- **Progress Tracking**: Progress bars using tqdm
- **Batch Processing**: Handles multiple stock tickers efficiently
- **Data Visualization**: Generates cumulative gain charts

## Requirements

Install the required dependencies:

```bash
pip install -r requirements.txt
```

### Dependencies:
- yfinance
- numpy
- matplotlib
- scikit-learn
- tensorflow
- tqdm

## Usage

### Google Colab (Recommended)
1. Upload the script to Google Colab
2. Create a `Test_ticker.txt` file with stock symbols (one per line)
3. Save it to `/content/drive/MyDrive/Outputs/Test_ticker.txt`
4. Run the script

### Local Execution
1. Modify the file paths to point to local directories
2. Create a ticker file with stock symbols
3. Run: `python fibonacci_lstm_forecast.py`

## Input

The script expects a text file containing stock ticker symbols (one per line):
```
AAPL
GOOGL
MSFT
```

## Output

The script generates:
- CSV file with forecast results and cumulative gains
- PNG charts showing cumulative gains across multiple timeframes
- Console output with progress and summary statistics

## Configuration

Key parameters that can be adjusted in the script:
- `batch_size`: Number of tickers to process in each batch (default: 5)
- `window_size`: Size of the sliding window for LSTM (default: 7)
- `forecast_steps`: Steps ahead to forecast (default: [1,2,3])
- `fib_idxs`: Fibonacci point indices for analysis

## Notes

- The script is optimized for Google Colab environment
- Requires Google Drive mount for file I/O in Colab
- Memory management included with garbage collection
- Early stopping implemented to prevent overfitting

## License

Please refer to the repository license for usage terms.
