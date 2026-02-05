"""
Bullish Stock Analyzer - Google Colab Compatible
Custom implementation for comprehensive stock analysis with multi-metric scoring
"""

import yfinance as yf
import numpy as np
import pandas as pd
import os
from typing import Dict, List, Optional, Tuple

# Google Colab Drive Integration
try:
    from google.colab import drive
    drive.mount('/content/drive')
    print("✓ Drive mounted")
except:
    print("ℹ Not running in Colab environment")

# Path Configuration
TICKER_INPUT_PATH = '/content/drive/MyDrive/Outputs/t/Test_ticker.txt'
RESULTS_OUTPUT_PATH = '/content/drive/MyDrive/Outputs/Bullish9_Feb5.csv'


class StockMetricsCalculator:
    """Handles all technical indicator calculations"""
    
    def __init__(self, price_series: np.ndarray, volume_series: Optional[np.ndarray] = None):
        self.prices = np.array(price_series, dtype=float)
        self.volumes = np.array(volume_series, dtype=float) if volume_series is not None else None
        
    def compute_hurst_coefficient(self, lookback_window: int = 252) -> float:
        """
        Generalized Hurst Exponent using rescaled range analysis
        Custom implementation with variance-based approach
        """
        if len(self.prices) < lookback_window:
            return np.nan
            
        price_data = self.prices[-lookback_window:]
        log_returns = np.diff(np.log(price_data))
        
        # Range of lags to test
        lag_values = np.arange(2, min(80, len(log_returns) // 3))
        variance_at_lags = []
        
        for lag_size in lag_values:
            # Calculate variance at different time scales
            aggregated_returns = []
            for start_idx in range(0, len(log_returns) - lag_size, lag_size):
                segment = log_returns[start_idx:start_idx + lag_size]
                aggregated_returns.append(np.sum(segment))
            
            if len(aggregated_returns) > 1:
                variance_at_lags.append(np.var(aggregated_returns))
        
        if len(variance_at_lags) < 5:
            return np.nan
        
        # Fit power law relationship: variance ~ lag^(2*H)
        valid_lags = lag_values[:len(variance_at_lags)]
        log_lags = np.log(valid_lags)
        log_vars = np.log(variance_at_lags)
        
        # Linear regression in log space
        coefficients = np.polyfit(log_lags, log_vars, 1)
        hurst_value = coefficients[0] / 2.0
        
        return float(hurst_value)
    
    def compute_risk_adjusted_return(self, annual_rfr: float = 0.02) -> float:
        """
        Calculate Sharpe-style risk-adjusted returns
        Uses annualized metrics
        """
        if len(self.prices) < 30:
            return np.nan
        
        # Calculate daily returns
        daily_changes = np.diff(self.prices) / self.prices[:-1]
        
        if np.std(daily_changes) == 0:
            return np.nan
        
        # Annualize statistics
        avg_annual_return = np.mean(daily_changes) * 252
        annual_volatility = np.std(daily_changes) * np.sqrt(252)
        
        risk_adj_metric = (avg_annual_return - annual_rfr) / annual_volatility
        
        return float(risk_adj_metric)
    
    def compute_directional_momentum(self, window_length: int) -> float:
        """
        Calculate trend direction using regression slope
        Returns percentage change trend
        """
        if len(self.prices) < window_length:
            return np.nan
        
        recent_prices = self.prices[-window_length:]
        time_indices = np.arange(len(recent_prices))
        
        # Polynomial fit (degree 1 = linear)
        trend_params = np.polyfit(time_indices, recent_prices, deg=1)
        slope_coefficient = trend_params[0]
        
        # Convert to percentage-based trend
        starting_price = recent_prices[0]
        trend_percentage = (slope_coefficient * window_length / starting_price) * 100
        
        return float(trend_percentage)
    
    def project_future_value(self, base_window: int, projection_days: int) -> Tuple[float, float]:
        """
        Linear extrapolation for future price prediction
        Returns (predicted_price, expected_return_pct)
        """
        if len(self.prices) < base_window:
            return np.nan, np.nan
        
        historical_data = self.prices[-base_window:]
        time_axis = np.arange(len(historical_data))
        
        # Fit linear model
        projection_coeffs = np.polyfit(time_axis, historical_data, deg=1)
        
        # Project forward
        future_time = len(historical_data) + projection_days - 1
        predicted_value = projection_coeffs[0] * future_time + projection_coeffs[1]
        
        current_value = self.prices[-1]
        expected_gain = ((predicted_value - current_value) / current_value) * 100
        
        return float(predicted_value), float(expected_gain)
    
    def compute_momentum_oscillator(self, period_length: int = 21) -> float:
        """
        Rate of change indicator over specified period
        """
        if len(self.prices) <= period_length:
            return np.nan
        
        current_val = self.prices[-1]
        historical_val = self.prices[-(period_length + 1)]
        
        rate_change = ((current_val - historical_val) / historical_val) * 100
        
        return float(rate_change)
    
    def compute_convergence_divergence_signal(self) -> float:
        """
        MACD-style momentum indicator
        Returns histogram value (difference between MACD and signal)
        """
        if len(self.prices) < 35:
            return np.nan
        
        # Exponential moving averages with different spans
        fast_ema = pd.Series(self.prices).ewm(span=12, adjust=False).mean()
        slow_ema = pd.Series(self.prices).ewm(span=26, adjust=False).mean()
        
        macd_series = fast_ema - slow_ema
        signal_series = macd_series.ewm(span=9, adjust=False).mean()
        
        histogram_value = macd_series.iloc[-1] - signal_series.iloc[-1]
        
        return float(histogram_value)
    
    def compute_relative_momentum_index(self, period: int = 14) -> float:
        """
        RSI-style momentum indicator
        Range: 0-100
        """
        if len(self.prices) <= period:
            return np.nan
        
        price_changes = np.diff(self.prices)
        
        # Separate gains and losses
        positive_moves = np.where(price_changes > 0, price_changes, 0)
        negative_moves = np.where(price_changes < 0, -price_changes, 0)
        
        # Average over period
        avg_positive = np.mean(positive_moves[-period:])
        avg_negative = np.mean(negative_moves[-period:])
        
        if avg_negative == 0:
            return 100.0
        
        strength_ratio = avg_positive / avg_negative
        momentum_index = 100.0 - (100.0 / (1.0 + strength_ratio))
        
        return float(momentum_index)
    
    def compute_volume_accumulation(self) -> float:
        """
        On-Balance Volume style indicator
        Tracks volume flow based on price direction
        """
        if self.volumes is None or len(self.prices) < 2:
            return np.nan
        
        accumulated_volume = 0.0
        
        for idx in range(1, len(self.prices)):
            if self.prices[idx] > self.prices[idx - 1]:
                accumulated_volume += self.volumes[idx]
            elif self.prices[idx] < self.prices[idx - 1]:
                accumulated_volume -= self.volumes[idx]
        
        return float(accumulated_volume)


class ScoringEngine:
    """Converts raw metrics to normalized scores"""
    
    @staticmethod
    def normalize_to_scale(value: float, min_bound: float, max_bound: float, 
                          inverted: bool = False) -> float:
        """
        Map value to 0-10 scale
        """
        if np.isnan(value) or np.isnan(min_bound) or np.isnan(max_bound):
            return 0.0
        
        if max_bound == min_bound:
            return 5.0
        
        normalized = (value - min_bound) / (max_bound - min_bound)
        
        if inverted:
            normalized = 1.0 - normalized
        
        score = normalized * 10.0
        return max(0.0, min(10.0, score))
    
    @staticmethod
    def score_momentum_index(rmi_value: float) -> float:
        """
        Custom scoring for RSI-style indicator
        Target value: 46 (optimal), 78 (worst)
        """
        if np.isnan(rmi_value):
            return 0.0
        
        optimal_target = 46.0
        worst_threshold = 78.0
        
        if rmi_value <= optimal_target:
            # Below target: proportional scoring
            score = (rmi_value / optimal_target) * 10.0
        else:
            # Above target: inverse scoring
            deviation = abs(rmi_value - optimal_target)
            max_deviation = abs(worst_threshold - optimal_target)
            score = 10.0 - (deviation / max_deviation * 10.0)
        
        return max(0.0, min(10.0, score))


class TickerAnalysisEngine:
    """Main analysis orchestrator"""
    
    def __init__(self, ticker_symbol: str):
        self.symbol = ticker_symbol
        self.raw_metrics = {}
        self.computed_scores = {}
        
    def fetch_and_analyze(self) -> bool:
        """
        Download data and compute all metrics
        Returns True if successful
        """
        try:
            print(f"→ Analyzing {self.symbol}")
            
            # Download historical data
            ticker_data = yf.download(
                self.symbol, 
                period='2y', 
                interval='1d',
                progress=False,
                auto_adjust=True
            )
            
            if ticker_data.empty or len(ticker_data) < 252:
                print(f"  ✗ Insufficient data for {self.symbol}")
                return False
            
            # Extract price and volume arrays
            price_array = ticker_data['Close'].values
            volume_array = ticker_data['Volume'].values
            
            # Initialize calculator
            calculator = StockMetricsCalculator(price_array, volume_array)
            
            # Compute all metrics
            self.raw_metrics['current_price'] = float(price_array[-1])
            self.raw_metrics['hurst_exp'] = calculator.compute_hurst_coefficient(252)
            self.raw_metrics['sharpe_metric'] = calculator.compute_risk_adjusted_return()
            
            # Trend calculations for different timeframes
            self.raw_metrics['trend_monthly'] = calculator.compute_directional_momentum(21)
            self.raw_metrics['trend_weekly'] = calculator.compute_directional_momentum(5)
            self.raw_metrics['trend_daily'] = calculator.compute_directional_momentum(252)
            
            # Forecast
            pred_price, pred_return = calculator.project_future_value(90, 5)
            self.raw_metrics['forecast_price'] = pred_price
            self.raw_metrics['forecast_return_pct'] = pred_return
            
            # Technical indicators
            self.raw_metrics['roc_21d'] = calculator.compute_momentum_oscillator(21)
            self.raw_metrics['macd_histogram'] = calculator.compute_convergence_divergence_signal()
            self.raw_metrics['rmi'] = calculator.compute_relative_momentum_index(14)
            self.raw_metrics['obv_accumulation'] = calculator.compute_volume_accumulation()
            
            return True
            
        except Exception as error:
            print(f"  ✗ Error processing {self.symbol}: {error}")
            return False
    
    def get_raw_data(self) -> Dict:
        """Return raw metric dictionary"""
        return {
            'Ticker': self.symbol,
            'Price': self.raw_metrics.get('current_price', np.nan),
            'Hurst': self.raw_metrics.get('hurst_exp', np.nan),
            'Sharpe': self.raw_metrics.get('sharpe_metric', np.nan),
            'Monthly_Trend': self.raw_metrics.get('trend_monthly', np.nan),
            'Weekly_Trend': self.raw_metrics.get('trend_weekly', np.nan),
            'Daily_Trend': self.raw_metrics.get('trend_daily', np.nan),
            'Forecast_5D': self.raw_metrics.get('forecast_price', np.nan),
            'Forecast_Return_%': self.raw_metrics.get('forecast_return_pct', np.nan),
            'ROC_21D': self.raw_metrics.get('roc_21d', np.nan),
            'MACD': self.raw_metrics.get('macd_histogram', np.nan),
            'RSI': self.raw_metrics.get('rmi', np.nan),
            'OBV': self.raw_metrics.get('obv_accumulation', np.nan)
        }


def load_ticker_list(file_path: str) -> List[str]:
    """Read ticker symbols from input file"""
    try:
        with open(file_path, 'r') as input_file:
            ticker_list = [line.strip() for line in input_file if line.strip()]
        return ticker_list
    except FileNotFoundError:
        print(f"⚠ Error: Cannot find {file_path}")
        raise


def apply_scoring_to_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply scoring logic to all metrics in dataframe"""
    scorer = ScoringEngine()
    
    # Score each metric (higher raw value = better, except where noted)
    scoring_config = [
        ('Hurst', 'Hurst_Score', False),
        ('Sharpe', 'Sharpe_Score', False),
        ('Monthly_Trend', 'Monthly_Trend_Score', False),
        ('Weekly_Trend', 'Weekly_Trend_Score', False),
        ('Daily_Trend', 'Daily_Trend_Score', False),
        ('Forecast_Return_%', 'Forecast_Return_Score', False),
        ('ROC_21D', 'ROC_Score', False),
        ('MACD', 'MACD_Score', False),
        ('OBV', 'OBV_Score', False)
    ]
    
    for metric_col, score_col, inverted in scoring_config:
        min_val = df[metric_col].min()
        max_val = df[metric_col].max()
        df[score_col] = df[metric_col].apply(
            lambda x: scorer.normalize_to_scale(x, min_val, max_val, inverted)
        )
    
    # Special scoring for RSI
    df['RSI_Score'] = df['RSI'].apply(scorer.score_momentum_index)
    
    # Calculate aggregate score
    score_columns = [col for col in df.columns if col.endswith('_Score')]
    df['Total_Score'] = df[score_columns].sum(axis=1)
    
    return df


def execute_analysis():
    """Main execution function"""
    print("=" * 70)
    print("BULLISH STOCK ANALYZER - Comprehensive Multi-Metric Scoring System")
    print("=" * 70)
    
    # Load tickers
    ticker_symbols = load_ticker_list(TICKER_INPUT_PATH)
    print(f"\n✓ Loaded {len(ticker_symbols)} ticker symbols")
    
    # Analyze each ticker
    analysis_results = []
    
    for symbol in ticker_symbols:
        analyzer = TickerAnalysisEngine(symbol)
        if analyzer.fetch_and_analyze():
            analysis_results.append(analyzer.get_raw_data())
    
    if not analysis_results:
        print("\n⚠ No successful analyses")
        return
    
    print(f"\n✓ Successfully analyzed {len(analysis_results)} stocks")
    
    # Create dataframe
    results_df = pd.DataFrame(analysis_results)
    
    # Apply scoring
    results_df = apply_scoring_to_dataframe(results_df)
    
    # Sort by total score (descending)
    results_df = results_df.sort_values('Total_Score', ascending=False)
    
    # Organize output columns
    output_column_order = [
        'Ticker', 'Price',
        'Hurst', 'Hurst_Score',
        'Sharpe', 'Sharpe_Score',
        'Monthly_Trend', 'Monthly_Trend_Score',
        'Weekly_Trend', 'Weekly_Trend_Score',
        'Daily_Trend', 'Daily_Trend_Score',
        'Forecast_5D', 'Forecast_Return_%', 'Forecast_Return_Score',
        'ROC_21D', 'ROC_Score',
        'MACD', 'MACD_Score',
        'RSI', 'RSI_Score',
        'OBV', 'OBV_Score',
        'Total_Score'
    ]
    
    final_output = results_df[output_column_order]
    
    # Save results
    try:
        final_output.to_csv(RESULTS_OUTPUT_PATH, index=False)
        print(f"\n✓ Results saved: {RESULTS_OUTPUT_PATH}")
    except Exception as save_error:
        # Fallback location
        fallback_path = '/content/Bullish9_Feb5.csv'
        final_output.to_csv(fallback_path, index=False)
        print(f"\n✓ Results saved to fallback location: {fallback_path}")
    
    # Display summary
    print("\n" + "=" * 70)
    print("TOP 10 STOCKS BY TOTAL SCORE")
    print("=" * 70)
    summary_cols = ['Ticker', 'Price', 'Total_Score', 'Hurst', 'Sharpe', 'RSI']
    print(final_output[summary_cols].head(10).to_string(index=False))
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\nTotal stocks analyzed: {len(analysis_results)}")
    print(f"Output location: {RESULTS_OUTPUT_PATH}")
    print("\nScoring Summary:")
    print("  • Each metric normalized to 0-10 scale")
    print("  • Higher scores = Better investment potential")
    print("  • Sorted by Total Score (sum of all metric scores)")
    print("\nMetrics Included:")
    print("  ✓ Hurst Exponent (252-day persistence)")
    print("  ✓ Sharpe Ratio (risk-adjusted returns)")
    print("  ✓ Trends: Monthly (21d), Weekly (5d), Daily (252d)")
    print("  ✓ 5-Day Forecast (based on 90-day window)")
    print("  ✓ 21-Day Rate of Change")
    print("  ✓ MACD Histogram")
    print("  ✓ RSI (optimal=46, worst=78)")
    print("  ✓ OBV (volume accumulation)")


if __name__ == "__main__":
    execute_analysis()
