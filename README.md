# Fibonacci LSTM Stock Forecast

This script performs multi-timeframe (weekly, daily, hourly) LSTM-based stock price forecasting using Fibonacci sequence analysis.

## Features

- Weekly, daily, and hourly LSTM forecasts
- Fibonacci point analysis
- Configurable input/output paths
- Progress bars for batch processing
- CSV and chart outputs

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```bash
python3 success6.py --ticker-file /path/to/tickers.txt --output-dir /path/to/output
```

### Command-line Arguments

- `--ticker-file`: Path to the ticker file (default: `/content/drive/MyDrive/Outputs/Test_ticker.txt` or `TICKER_FILE` env var)
- `--output-dir`: Output directory for results (default: `/content/drive/MyDrive/Outputs` or `OUTPUT_DIR` env var)
- `--batch-size`: Batch size for processing tickers (default: 5)

### Environment Variables

You can also set the following environment variables:

- `TICKER_FILE`: Default path to ticker file
- `OUTPUT_DIR`: Default output directory

Example:

```bash
export TICKER_FILE=/path/to/tickers.txt
export OUTPUT_DIR=/path/to/output
python3 success6.py
```

## Input Format

The ticker file should contain one stock ticker symbol per line:

```
AAPL
GOOGL
MSFT
TSLA
```

## Output

The script generates:

1. `Fibonacci_LSTM_results.csv` - CSV file with forecast results
2. `Fibonacci_LSTM_combined_cumgain.png` - Combined chart showing cumulative gains

## Google Colab

This script is designed to work in Google Colab with Google Drive integration. When run in Colab, it will automatically mount Google Drive.

## Notes

- The script uses rate limiting to avoid API throttling
- Error messages are printed to stderr for easier debugging
- Progress bars show processing status for each timeframe
