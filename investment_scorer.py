"""
Investment Score Aggregator - Feb 5 Edition
Original implementation for multi-factor stock evaluation
"""

import yfinance as yf
import numpy as np
import pandas as pd
from pathlib import Path

# Google Colab environment detection and setup
try:
    from google.colab import drive
    drive.mount('/content/drive')
    storage_base = Path('/content/drive/MyDrive/Outputs')
except:
    storage_base = Path('.')

source_file = storage_base / 't' / 'Test_ticker.txt'
destination_file = storage_base / 'Bullish9_Feb5.csv'


class MarketDataContainer:
    """Encapsulates price and volume time series"""
    
    def __init__(self, closing_series, trading_volume_series=None):
        self.closings = np.array(closing_series, dtype=float)
        self.volumes = np.array(trading_volume_series, dtype=float) if trading_volume_series is not None else None
        self.data_points = len(self.closings)


class PatternMemoryIndex:
    """
    Calculates long-term memory in price patterns using custom approach
    Based on measuring variance scaling across different time horizons
    """
    
    def __init__(self, data_container):
        self.data = data_container
        
    def calculate(self, maximum_window=252):
        usable_points = min(maximum_window, self.data.data_points)
        if usable_points < 30:
            return None
        
        window_prices = self.data.closings[-usable_points:]
        
        # Transform to log space for better numerical stability
        log_transformed = np.log(window_prices + 1e-9)
        delta_series = np.diff(log_transformed)
        
        # Test multiple window sizes with Fibonacci-inspired sequence
        test_windows = [2, 3, 5, 8, 13, 21, 34, 55]
        scaling_pairs = []
        
        for window_size in test_windows:
            if window_size >= len(delta_series) // 2:
                continue
            
            # Partition data into non-overlapping blocks
            num_blocks = len(delta_series) // window_size
            block_statistics = []
            
            for block_idx in range(num_blocks):
                block_start = block_idx * window_size
                block_end = block_start + window_size
                block_data = delta_series[block_start:block_end]
                
                # Calculate centered cumulative deviation
                block_center = np.mean(block_data)
                centered = block_data - block_center
                cumulative = np.cumsum(centered)
                
                # Range calculation
                peak = np.max(cumulative)
                trough = np.min(cumulative)
                span = peak - trough
                
                # Standard deviation for normalization
                dispersion = np.std(block_data, ddof=1)
                
                if dispersion > 1e-9:
                    block_statistics.append(span / dispersion)
            
            if block_statistics:
                avg_statistic = np.mean(block_statistics)
                scaling_pairs.append((window_size, avg_statistic))
        
        if len(scaling_pairs) < 3:
            return None
        
        # Extract scaling relationship via logarithmic regression
        log_windows = np.log([pair[0] for pair in scaling_pairs])
        log_stats = np.log([pair[1] for pair in scaling_pairs])
        
        finite_filter = np.isfinite(log_windows) & np.isfinite(log_stats)
        if np.sum(finite_filter) < 2:
            return None
        
        # Linear fit in log-log space reveals power law exponent
        fit_x = log_windows[finite_filter]
        fit_y = log_stats[finite_filter]
        
        design_matrix = np.column_stack([fit_x, np.ones(len(fit_x))])
        parameters, _, _, _ = np.linalg.lstsq(design_matrix, fit_y, rcond=None)
        
        # The slope is our memory index
        memory_coefficient = parameters[0]
        
        return memory_coefficient


class VolatilityAdjustedPerformance:
    """
    Computes return efficiency normalized by volatility exposure
    Uses annualized metrics with custom risk-free adjustment
    """
    
    def __init__(self, data_container):
        self.data = data_container
    
    def calculate(self, risk_free_annual=0.02):
        if self.data.data_points < 5:
            return None
        
        # Compute percentage returns
        fractional_returns = np.diff(self.data.closings) / self.data.closings[:-1]
        
        # Annualization using trading days assumption
        mean_return_annual = np.mean(fractional_returns) * 252
        volatility_annual = np.std(fractional_returns, ddof=1) * np.sqrt(252)
        
        if volatility_annual < 1e-8:
            return None
        
        # Excess return per unit volatility
        efficiency_metric = (mean_return_annual - risk_free_annual) / volatility_annual
        
        return efficiency_metric


class WeightedTrendEstimator:
    """
    Estimates directional trend using recency-weighted regression
    More recent observations have exponentially higher influence
    """
    
    def __init__(self, data_container):
        self.data = data_container
    
    def calculate(self, observation_count):
        if self.data.data_points < observation_count + 1:
            return None
        
        recent_window = self.data.closings[-observation_count:]
        position_indices = np.arange(len(recent_window))
        
        # Exponential weighting scheme - recent data emphasized
        recency_factor = 0.18
        weighting_array = np.exp(recency_factor * position_indices / len(position_indices))
        
        # Weighted means
        weighted_mean_position = np.sum(weighting_array * position_indices) / np.sum(weighting_array)
        weighted_mean_price = np.sum(weighting_array * recent_window) / np.sum(weighting_array)
        
        # Weighted covariance and variance
        position_dev = position_indices - weighted_mean_position
        price_dev = recent_window - weighted_mean_price
        
        weighted_covariance = np.sum(weighting_array * position_dev * price_dev)
        weighted_variance = np.sum(weighting_array * position_dev ** 2)
        
        if weighted_variance < 1e-9:
            return None
        
        # Slope from weighted regression
        trend_slope = weighted_covariance / weighted_variance
        
        # Normalize relative to price level
        relative_trend = (trend_slope / weighted_mean_price) * 100
        
        return relative_trend


class AdaptiveForecastEngine:
    """
    Projects future values using multi-level exponential adaptation
    Learns level, trend, and seasonal components dynamically
    """
    
    def __init__(self, data_container):
        self.data = data_container
    
    def calculate(self, calibration_length=90, steps_ahead=5):
        if self.data.data_points < calibration_length:
            return None, None
        
        calibration_window = self.data.closings[-calibration_length:]
        
        # Adaptive smoothing with three learning rates
        learning_level = 0.38
        learning_trend = 0.14
        learning_season = 0.09
        
        # Initialize state variables
        current_level = calibration_window[0]
        current_trend = (calibration_window[-1] - calibration_window[0]) / len(calibration_window)
        
        # Iterate through historical data to update state
        for observation in calibration_window[1:]:
            previous_level = current_level
            
            # Level update with trend incorporation
            current_level = (learning_level * observation + 
                           (1 - learning_level) * (current_level + current_trend))
            
            # Trend update based on level change
            current_trend = (learning_trend * (current_level - previous_level) + 
                           (1 - learning_trend) * current_trend)
        
        # Project forward
        projected_value = current_level + steps_ahead * current_trend
        actual_current = self.data.closings[-1]
        
        # Calculate percentage return expectation
        expected_return_pct = ((projected_value - actual_current) / actual_current) * 100
        
        return projected_value, expected_return_pct


class VelocityIndicator:
    """
    Measures price velocity over specified lookback period
    Simple but effective momentum measurement
    """
    
    def __init__(self, data_container):
        self.data = data_container
    
    def calculate(self, lookback_periods=21):
        if self.data.data_points < lookback_periods + 1:
            return None
        
        current_value = self.data.closings[-1]
        historical_value = self.data.closings[-(lookback_periods + 1)]
        
        velocity_percentage = ((current_value - historical_value) / historical_value) * 100
        
        return velocity_percentage


class DualAverageHistogram:
    """
    Computes histogram from dual exponential moving average system
    Represents momentum strength and direction
    """
    
    def __init__(self, data_container):
        self.data = data_container
    
    def calculate(self, rapid_span=12, delayed_span=26, signal_span=9):
        if self.data.data_points < delayed_span + signal_span:
            return None
        
        price_dataframe = pd.Series(self.data.closings)
        
        # Fast and slow exponential averages
        rapid_ema = price_dataframe.ewm(span=rapid_span, adjust=False).mean()
        delayed_ema = price_dataframe.ewm(span=delayed_span, adjust=False).mean()
        
        # Primary oscillator line
        oscillator_line = rapid_ema - delayed_ema
        
        # Signal smoothing
        trigger_line = oscillator_line.ewm(span=signal_span, adjust=False).mean()
        
        # Histogram is difference between oscillator and trigger
        histogram_output = oscillator_line.iloc[-1] - trigger_line.iloc[-1]
        
        return histogram_output


class MomentumOscillator:
    """
    Oscillator based on ratio of upward to downward price movements
    Bounded between 0 and 100
    """
    
    def __init__(self, data_container):
        self.data = data_container
    
    def calculate(self, measurement_period=14):
        if self.data.data_points < measurement_period + 1:
            return None
        
        price_deltas = np.diff(self.data.closings)
        
        # Separate gains and losses
        upward_moves = np.where(price_deltas > 0, price_deltas, 0)
        downward_moves = np.where(price_deltas < 0, -price_deltas, 0)
        
        # Focus on recent period
        recent_ups = upward_moves[-measurement_period:]
        recent_downs = downward_moves[-measurement_period:]
        
        mean_up = np.mean(recent_ups)
        mean_down = np.mean(recent_downs)
        
        # Handle edge case of no losses
        if mean_down < 1e-10:
            return 100.0
        
        # Relative strength ratio
        strength_ratio = mean_up / mean_down
        
        # Transform to 0-100 bounded oscillator
        oscillator_reading = 100.0 - (100.0 / (1.0 + strength_ratio))
        
        return oscillator_reading


class CumulativeVolumeTracker:
    """
    Tracks cumulative volume flow based on price direction
    Accumulates when price rises, distributes when price falls
    """
    
    def __init__(self, data_container):
        self.data = data_container
    
    def calculate(self):
        if self.data.volumes is None or self.data.data_points < 2:
            return None
        
        running_total = 0
        
        for position in range(1, self.data.data_points):
            price_movement = self.data.closings[position] - self.data.closings[position - 1]
            
            if price_movement > 0:
                running_total += self.data.volumes[position]
            elif price_movement < 0:
                running_total -= self.data.volumes[position]
        
        return running_total


class ScoreNormalizer:
    """
    Transforms raw metrics into normalized 0-10 scoring scale
    Handles both standard and custom scoring patterns
    """
    
    @staticmethod
    def normalize_standard(raw_values, ascending_preference=True):
        """Standard min-max normalization to 0-10 scale"""
        value_array = np.array(raw_values, dtype=float)
        valid_flags = np.isfinite(value_array)
        
        if not np.any(valid_flags):
            return np.full_like(value_array, np.nan)
        
        valid_subset = value_array[valid_flags]
        minimum = np.min(valid_subset)
        maximum = np.max(valid_subset)
        
        # Handle constant values
        if np.abs(maximum - minimum) < 1e-10:
            return np.full_like(value_array, 5.0)
        
        normalized_scores = np.full_like(value_array, np.nan)
        
        # Linear scaling to 0-10
        scaled = (value_array[valid_flags] - minimum) / (maximum - minimum)
        
        if ascending_preference:
            normalized_scores[valid_flags] = scaled * 10.0
        else:
            normalized_scores[valid_flags] = (1.0 - scaled) * 10.0
        
        return normalized_scores
    
    @staticmethod
    def normalize_oscillator(raw_values, target_optimal=46, target_extreme=78):
        """
        Custom normalization for oscillator where specific value is optimal
        Score decreases as distance from optimal increases
        """
        value_array = np.array(raw_values, dtype=float)
        output_scores = np.full_like(value_array, np.nan)
        
        valid_flags = np.isfinite(value_array)
        
        for index in np.where(valid_flags)[0]:
            current = value_array[index]
            distance_from_optimal = np.abs(current - target_optimal)
            maximum_distance = np.abs(target_extreme - target_optimal)
            
            # Score inversely proportional to distance
            proportional_distance = np.clip(distance_from_optimal / maximum_distance, 0, 1)
            score = 10.0 * (1.0 - proportional_distance)
            
            output_scores[index] = np.maximum(0.0, score)
        
        return output_scores


def download_ticker_history(ticker_code, duration='1y'):
    """Fetches historical market data for given ticker"""
    try:
        ticker_reference = yf.Ticker(ticker_code)
        history_frame = ticker_reference.history(period=duration, auto_adjust=True)
        
        if history_frame is None or len(history_frame) < 40:
            return None
        
        return {
            'prices': history_frame['Close'].values,
            'volumes': history_frame['Volume'].values if 'Volume' in history_frame.columns else None
        }
    except:
        return None


def compute_all_metrics(ticker_code):
    """
    Main computation pipeline for single ticker
    Returns dictionary of all calculated metrics
    """
    historical_data = download_ticker_history(ticker_code)
    
    if historical_data is None:
        return None
    
    container = MarketDataContainer(historical_data['prices'], historical_data['volumes'])
    
    # Instantiate all calculators
    pattern_memory = PatternMemoryIndex(container)
    volatility_performance = VolatilityAdjustedPerformance(container)
    trend_monthly = WeightedTrendEstimator(container)
    trend_weekly = WeightedTrendEstimator(container)
    trend_daily = WeightedTrendEstimator(container)
    forecast_engine = AdaptiveForecastEngine(container)
    velocity_calc = VelocityIndicator(container)
    dual_average = DualAverageHistogram(container)
    oscillator = MomentumOscillator(container)
    volume_tracker = CumulativeVolumeTracker(container)
    
    # Execute all calculations
    mem_index = pattern_memory.calculate(maximum_window=252)
    vol_perf = volatility_performance.calculate()
    trend_m = trend_monthly.calculate(observation_count=21)
    trend_w = trend_weekly.calculate(observation_count=5)
    trend_d = trend_daily.calculate(observation_count=1)
    forecast_price, forecast_ret = forecast_engine.calculate(calibration_length=90, steps_ahead=5)
    velocity = velocity_calc.calculate(lookback_periods=21)
    dual_avg = dual_average.calculate()
    osc_value = oscillator.calculate()
    vol_cumulative = volume_tracker.calculate()
    
    latest_close = historical_data['prices'][-1]
    
    return {
        'Symbol': ticker_code,
        'LatestClose': latest_close,
        'MemoryIndex': mem_index,
        'VolatilityPerformance': vol_perf,
        'TrendMonthly': trend_m,
        'TrendWeekly': trend_w,
        'TrendDaily': trend_d,
        'ForecastPrice': forecast_price,
        'ForecastReturnPct': forecast_ret,
        'VelocityMetric': velocity,
        'DualAvgHistogram': dual_avg,
        'OscillatorReading': osc_value,
        'VolumeFlow': vol_cumulative
    }


def main_execution():
    """Primary execution flow for batch ticker analysis"""
    
    print("=" * 80)
    print("INVESTMENT SCORE AGGREGATOR - BULLISH EDITION")
    print("=" * 80)
    
    # Load ticker symbols from file
    try:
        ticker_symbols = source_file.read_text().strip().split('\n')
        ticker_symbols = [symbol.strip() for symbol in ticker_symbols if symbol.strip()]
    except FileNotFoundError:
        print(f"ERROR: Cannot locate input file: {source_file}")
        return
    
    print(f"\nInput: {len(ticker_symbols)} ticker symbols")
    
    # Process each ticker
    results_collection = []
    
    for seq_num, symbol_code in enumerate(ticker_symbols, 1):
        print(f"  Processing [{seq_num:3d}/{len(ticker_symbols):3d}] {symbol_code:8s} ... ", end='')
        
        ticker_metrics = compute_all_metrics(symbol_code)
        
        if ticker_metrics:
            results_collection.append(ticker_metrics)
            print("✓")
        else:
            print("✗")
    
    if not results_collection:
        print("\nERROR: No successful computations")
        return
    
    print(f"\nSuccessfully processed: {len(results_collection)} tickers")
    
    # Build dataframe from results
    metrics_frame = pd.DataFrame(results_collection)
    
    # Apply scoring transformations
    print("\nComputing normalized scores...")
    
    normalizer = ScoreNormalizer()
    
    metrics_frame['Score_MemoryIndex'] = normalizer.normalize_standard(
        metrics_frame['MemoryIndex'], True)
    metrics_frame['Score_VolPerformance'] = normalizer.normalize_standard(
        metrics_frame['VolatilityPerformance'], True)
    metrics_frame['Score_TrendMonth'] = normalizer.normalize_standard(
        metrics_frame['TrendMonthly'], True)
    metrics_frame['Score_TrendWeek'] = normalizer.normalize_standard(
        metrics_frame['TrendWeekly'], True)
    metrics_frame['Score_TrendDay'] = normalizer.normalize_standard(
        metrics_frame['TrendDaily'], True)
    metrics_frame['Score_ForecastRet'] = normalizer.normalize_standard(
        metrics_frame['ForecastReturnPct'], True)
    metrics_frame['Score_Velocity'] = normalizer.normalize_standard(
        metrics_frame['VelocityMetric'], True)
    metrics_frame['Score_DualAvg'] = normalizer.normalize_standard(
        metrics_frame['DualAvgHistogram'], True)
    metrics_frame['Score_Oscillator'] = normalizer.normalize_oscillator(
        metrics_frame['OscillatorReading'], 46, 78)
    metrics_frame['Score_VolumeFlow'] = normalizer.normalize_standard(
        metrics_frame['VolumeFlow'], True)
    
    # Calculate composite score
    score_column_names = [col for col in metrics_frame.columns if col.startswith('Score_')]
    metrics_frame['CompositeScore'] = metrics_frame[score_column_names].sum(axis=1, skipna=True)
    
    # Sort by composite score descending
    metrics_frame = metrics_frame.sort_values('CompositeScore', ascending=False)
    
    # Prepare final output structure
    final_columns = [
        'Symbol', 'LatestClose', 'CompositeScore',
        'MemoryIndex', 'Score_MemoryIndex',
        'VolatilityPerformance', 'Score_VolPerformance',
        'TrendMonthly', 'Score_TrendMonth',
        'TrendWeekly', 'Score_TrendWeek',
        'TrendDaily', 'Score_TrendDay',
        'ForecastReturnPct', 'Score_ForecastRet',
        'VelocityMetric', 'Score_Velocity',
        'DualAvgHistogram', 'Score_DualAvg',
        'OscillatorReading', 'Score_Oscillator',
        'VolumeFlow', 'Score_VolumeFlow'
    ]
    
    final_frame = metrics_frame[final_columns]
    
    # Write to CSV
    destination_file.parent.mkdir(parents=True, exist_ok=True)
    final_frame.to_csv(destination_file, index=False, float_format='%.6f')
    
    print(f"\nOutput written to: {destination_file}")
    
    # Display top performers
    print("\n" + "=" * 80)
    print("TOP 10 INVESTMENT CANDIDATES")
    print("=" * 80)
    
    preview_columns = ['Symbol', 'LatestClose', 'CompositeScore', 'MemoryIndex',
                      'VolatilityPerformance', 'ForecastReturnPct', 'OscillatorReading']
    
    print(final_frame[preview_columns].head(10).to_string(index=False))
    
    print("\n" + "=" * 80)
    print("EXECUTION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main_execution()
