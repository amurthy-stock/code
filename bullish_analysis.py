"""
Bullish Stock Scoring System for Google Colab
Custom implementation for quantitative stock analysis
"""

import yfinance as yf
import numpy as np
import pandas as pd
import os
from datetime import datetime

# Mount Google Drive if in Colab
try:
    from google.colab import drive
    drive.mount('/content/drive')
    DRIVE_ROOT = '/content/drive/MyDrive/Outputs'
except:
    DRIVE_ROOT = '.'

# File paths
TICKER_INPUT = os.path.join(DRIVE_ROOT, 't', 'Test_ticker.txt')
RESULT_OUTPUT = os.path.join(DRIVE_ROOT, 'Bullish9_Feb5.csv')


class StockMetrics:
    """Calculate various stock metrics for scoring"""
    
    def __init__(self, price_data, volume_data=None):
        self.prices = np.array(price_data)
        self.volumes = np.array(volume_data) if volume_data is not None else None
        
    def gfe_metric(self, lookback=252):
        """Generalized Fractal Exponent using variance ratio method"""
        subset = self.prices[-min(lookback, len(self.prices)):]
        if len(subset) < 30:
            return None
            
        log_returns = np.diff(np.log(subset))
        
        # Calculate variance ratios at different lags
        variance_ratios = []
        test_lags = [2, 4, 8, 16, 32]
        
        for lag in test_lags:
            if lag >= len(log_returns):
                continue
                
            # Variance of lag-period returns
            lag_returns = []
            for i in range(0, len(log_returns) - lag + 1, lag):
                lag_returns.append(np.sum(log_returns[i:i+lag]))
            
            if len(lag_returns) > 1:
                var_lag = np.var(lag_returns, ddof=1)
                var_one = np.var(log_returns, ddof=1)
                
                if var_one > 0:
                    ratio = var_lag / (lag * var_one)
                    variance_ratios.append((np.log(lag), np.log(ratio)))
        
        if len(variance_ratios) < 2:
            return None
            
        # Fit to get fractal dimension
        x_vals = [vr[0] for vr in variance_ratios]
        y_vals = [vr[1] for vr in variance_ratios]
        
        coefficients = np.polyfit(x_vals, y_vals, 1)
        gfe_value = 0.5 + coefficients[0] / 2
        
        return gfe_value
    
    def risk_adjusted_return(self, annual_rf=0.02):
        """Compute risk-adjusted return metric"""
        if len(self.prices) < 3:
            return None
            
        pct_changes = np.diff(self.prices) / self.prices[:-1]
        
        avg_return = np.mean(pct_changes) * 252
        volatility = np.std(pct_changes, ddof=1) * np.sqrt(252)
        
        if volatility < 1e-6:
            return None
            
        risk_adj = (avg_return - annual_rf) / volatility
        return risk_adj
    
    def momentum_by_period(self, window):
        """Calculate momentum score for given window"""
        if len(self.prices) < window + 1:
            return None
            
        subset = self.prices[-window:]
        x_axis = np.arange(len(subset))
        
        # Weighted least squares with recency bias
        weights = np.exp(x_axis / len(subset))
        fit = np.polyfit(x_axis, subset, 1, w=weights)
        
        momentum = (fit[0] / np.mean(subset)) * 100
        return momentum
    
    def projection_model(self, train_days=90, horizon=5):
        """Project future price using exponential smoothing"""
        if len(self.prices) < train_days:
            return None, None
            
        recent = self.prices[-train_days:]
        
        # Triple exponential smoothing parameters
        alpha, beta, gamma = 0.3, 0.1, 0.05
        
        # Initialize
        level = recent[0]
        trend = (recent[-1] - recent[0]) / len(recent)
        
        # Update through data
        for price in recent[1:]:
            prev_level = level
            level = alpha * price + (1 - alpha) * (level + trend)
            trend = beta * (level - prev_level) + (1 - beta) * trend
        
        # Project forward
        forecast = level + horizon * trend
        current = self.prices[-1]
        
        return_pct = ((forecast - current) / current) * 100
        
        return forecast, return_pct
    
    def momentum_velocity(self, span=21):
        """Rate of change momentum indicator"""
        if len(self.prices) < span + 1:
            return None
            
        current = self.prices[-1]
        past = self.prices[-(span + 1)]
        
        velocity = ((current - past) / past) * 100
        return velocity
    
    def convergence_divergence(self, fast_span=12, slow_span=26, signal_span=9):
        """Moving average convergence divergence signal"""
        if len(self.prices) < slow_span + signal_span:
            return None
            
        series = pd.Series(self.prices)
        
        ema_fast = series.ewm(span=fast_span, adjust=False).mean()
        ema_slow = series.ewm(span=slow_span, adjust=False).mean()
        
        macd = ema_fast - ema_slow
        signal = macd.ewm(span=signal_span, adjust=False).mean()
        
        histogram = macd.iloc[-1] - signal.iloc[-1]
        
        return histogram
    
    def strength_index(self, period=14):
        """Relative strength indicator"""
        if len(self.prices) < period + 1:
            return None
            
        changes = np.diff(self.prices)
        
        gains = np.where(changes > 0, changes, 0)
        losses = np.where(changes < 0, -changes, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss < 1e-9:
            return 100.0
            
        relative_strength = avg_gain / avg_loss
        rsi_value = 100.0 - (100.0 / (1.0 + relative_strength))
        
        return rsi_value
    
    def balance_volume_metric(self):
        """On-balance volume calculation"""
        if self.volumes is None or len(self.prices) < 2:
            return None
            
        obv_total = 0
        for i in range(1, len(self.prices)):
            if self.prices[i] > self.prices[i-1]:
                obv_total += self.volumes[i]
            elif self.prices[i] < self.prices[i-1]:
                obv_total -= self.volumes[i]
                
        return obv_total


class ScoreCalculator:
    """Convert raw metrics to normalized scores"""
    
    @staticmethod
    def score_range(values, higher_better=True):
        """Map values to 0-10 scale"""
        arr = np.array(values, dtype=float)
        valid_mask = np.isfinite(arr)
        
        if not np.any(valid_mask):
            return np.full_like(arr, np.nan)
        
        min_val = np.nanmin(arr[valid_mask])
        max_val = np.nanmax(arr[valid_mask])
        
        if abs(max_val - min_val) < 1e-9:
            return np.full_like(arr, 5.0)
        
        scores = np.full_like(arr, np.nan)
        
        normalized = (arr[valid_mask] - min_val) / (max_val - min_val)
        
        if higher_better:
            scores[valid_mask] = normalized * 10.0
        else:
            scores[valid_mask] = (1.0 - normalized) * 10.0
            
        return scores
    
    @staticmethod
    def score_rsi_special(rsi_values, optimal=46, worst=78):
        """Special RSI scoring where 46 is optimal"""
        arr = np.array(rsi_values, dtype=float)
        scores = np.full_like(arr, np.nan)
        
        valid_mask = np.isfinite(arr)
        if not np.any(valid_mask):
            return scores
        
        for idx in np.where(valid_mask)[0]:
            rsi = arr[idx]
            distance = abs(rsi - optimal)
            max_distance = abs(worst - optimal)
            
            score = 10.0 * (1.0 - min(distance / max_distance, 1.0))
            scores[idx] = max(0.0, score)
            
        return scores


def fetch_ticker_data(symbol, period='1y'):
    """Download and process ticker data"""
    try:
        ticker_obj = yf.Ticker(symbol)
        history = ticker_obj.history(period=period, auto_adjust=True)
        
        if history is None or len(history) < 50:
            return None
            
        return {
            'prices': history['Close'].values,
            'volumes': history['Volume'].values if 'Volume' in history else None
        }
    except:
        return None


def analyze_single_ticker(symbol):
    """Perform complete analysis on one ticker"""
    data = fetch_ticker_data(symbol)
    
    if data is None:
        return None
    
    metrics_calc = StockMetrics(data['prices'], data['volumes'])
    
    # Calculate all metrics
    gfe = metrics_calc.gfe_metric(lookback=252)
    risk_adj = metrics_calc.risk_adjusted_return()
    
    mom_monthly = metrics_calc.momentum_by_period(21)
    mom_weekly = metrics_calc.momentum_by_period(5)
    mom_daily = metrics_calc.momentum_by_period(1)
    
    forecast_price, forecast_return = metrics_calc.projection_model(90, 5)
    
    roc = metrics_calc.momentum_velocity(21)
    macd_hist = metrics_calc.convergence_divergence()
    rsi = metrics_calc.strength_index()
    obv = metrics_calc.balance_volume_metric()
    
    current_price = data['prices'][-1]
    
    return {
        'Ticker': symbol,
        'Current_Price': current_price,
        'GFE_Hurst': gfe,
        'Risk_Adj_Return': risk_adj,
        'Momentum_Monthly': mom_monthly,
        'Momentum_Weekly': mom_weekly,
        'Momentum_Daily': mom_daily,
        'Forecast_5D_Price': forecast_price,
        'Forecast_5D_Return': forecast_return,
        'ROC_21D': roc,
        'MACD_Histogram': macd_hist,
        'RSI_Value': rsi,
        'OBV_Metric': obv
    }


def process_all_tickers():
    """Main processing function"""
    print("=" * 70)
    print("BULLISH STOCK ANALYSIS SYSTEM")
    print("=" * 70)
    
    # Read ticker symbols
    try:
        with open(TICKER_INPUT, 'r') as f:
            symbols = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"ERROR: Cannot find {TICKER_INPUT}")
        return
    
    print(f"\nProcessing {len(symbols)} ticker symbols...")
    
    # Analyze each ticker
    results_list = []
    for idx, symbol in enumerate(symbols, 1):
        print(f"  {idx}/{len(symbols)}: {symbol}...", end=' ')
        result = analyze_single_ticker(symbol)
        
        if result:
            results_list.append(result)
            print("OK")
        else:
            print("SKIP")
    
    if not results_list:
        print("\nERROR: No valid results")
        return
    
    print(f"\nAnalyzed {len(results_list)} tickers successfully")
    
    # Create dataframe
    df = pd.DataFrame(results_list)
    
    # Calculate scores
    print("\nCalculating scores...")
    
    scorer = ScoreCalculator()
    
    df['Score_GFE'] = scorer.score_range(df['GFE_Hurst'].values, True)
    df['Score_RiskAdj'] = scorer.score_range(df['Risk_Adj_Return'].values, True)
    df['Score_Mom_Month'] = scorer.score_range(df['Momentum_Monthly'].values, True)
    df['Score_Mom_Week'] = scorer.score_range(df['Momentum_Weekly'].values, True)
    df['Score_Mom_Day'] = scorer.score_range(df['Momentum_Daily'].values, True)
    df['Score_Forecast'] = scorer.score_range(df['Forecast_5D_Return'].values, True)
    df['Score_ROC'] = scorer.score_range(df['ROC_21D'].values, True)
    df['Score_MACD'] = scorer.score_range(df['MACD_Histogram'].values, True)
    df['Score_RSI'] = scorer.score_rsi_special(df['RSI_Value'].values, 46, 78)
    df['Score_OBV'] = scorer.score_range(df['OBV_Metric'].values, True)
    
    # Total score
    score_cols = [col for col in df.columns if col.startswith('Score_')]
    df['Total_Score'] = df[score_cols].sum(axis=1, skipna=True)
    
    # Sort by total score
    df_sorted = df.sort_values('Total_Score', ascending=False)
    
    # Prepare output columns
    output_cols = [
        'Ticker', 'Current_Price', 'Total_Score',
        'GFE_Hurst', 'Score_GFE',
        'Risk_Adj_Return', 'Score_RiskAdj',
        'Momentum_Monthly', 'Score_Mom_Month',
        'Momentum_Weekly', 'Score_Mom_Week',
        'Momentum_Daily', 'Score_Mom_Day',
        'Forecast_5D_Return', 'Score_Forecast',
        'ROC_21D', 'Score_ROC',
        'MACD_Histogram', 'Score_MACD',
        'RSI_Value', 'Score_RSI',
        'OBV_Metric', 'Score_OBV'
    ]
    
    df_output = df_sorted[output_cols]
    
    # Save results
    os.makedirs(os.path.dirname(RESULT_OUTPUT), exist_ok=True)
    df_output.to_csv(RESULT_OUTPUT, index=False, float_format='%.4f')
    
    print(f"\nResults saved to: {RESULT_OUTPUT}")
    
    # Display top results
    print("\n" + "=" * 70)
    print("TOP 10 RANKED STOCKS")
    print("=" * 70)
    
    display_cols = ['Ticker', 'Current_Price', 'Total_Score', 'GFE_Hurst', 
                    'Risk_Adj_Return', 'Forecast_5D_Return', 'RSI_Value']
    
    print(df_sorted[display_cols].head(10).to_string(index=False))
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    process_all_tickers()
