"""
Stock Quantitative Scoring Engine
Custom-built for investment analysis in Google Colab environment
"""

import yfinance as yf
import numpy as np
import pandas as pd
from pathlib import Path

# Colab drive mounting
try:
    from google.colab import drive
    drive.mount('/content/drive')
    BASE_PATH = Path('/content/drive/MyDrive/Outputs')
except:
    BASE_PATH = Path('.')

INPUT_TICKERS = BASE_PATH / 't' / 'Test_ticker.txt'
OUTPUT_RESULTS = BASE_PATH / 'Bullish9_Feb5.csv'


class TimeSeriesAnalyzer:
    """Custom time series analysis engine"""
    
    def __init__(self, price_array, volume_array=None):
        self.price_vals = np.asarray(price_array, dtype=np.float64)
        self.vol_vals = np.asarray(volume_array, dtype=np.float64) if volume_array is not None else None
        self.length = len(self.price_vals)
    
    def compute_persistence_coefficient(self, window_len=252):
        """
        Custom persistence measurement using rescaled range approach
        Measures whether price movements show trending or mean-reverting behavior
        """
        use_len = min(window_len, self.length)
        if use_len < 25:
            return np.nan
        
        segment = self.price_vals[-use_len:]
        log_price = np.log(segment + 1e-10)
        increments = np.diff(log_price)
        
        # Multi-scale rescaled range computation
        scale_factors = [3, 5, 8, 13, 21, 34]
        rs_measurements = []
        
        for scale in scale_factors:
            if scale > len(increments) // 3:
                continue
            
            chunks = [increments[i:i+scale] for i in range(0, len(increments) - scale + 1, scale)]
            rs_values_for_scale = []
            
            for chunk in chunks:
                if len(chunk) != scale:
                    continue
                
                chunk_mean = np.mean(chunk)
                deviations = chunk - chunk_mean
                cumulative_dev = np.cumsum(deviations)
                
                range_val = np.max(cumulative_dev) - np.min(cumulative_dev)
                std_val = np.std(chunk, ddof=1)
                
                if std_val > 1e-10:
                    rs_values_for_scale.append(range_val / std_val)
            
            if rs_values_for_scale:
                rs_measurements.append((scale, np.mean(rs_values_for_scale)))
        
        if len(rs_measurements) < 3:
            return np.nan
        
        # Log-log regression for power law estimation
        log_scales = np.log([x[0] for x in rs_measurements])
        log_rs = np.log([x[1] for x in rs_measurements])
        
        valid_indices = np.isfinite(log_scales) & np.isfinite(log_rs)
        if np.sum(valid_indices) < 2:
            return np.nan
        
        coeff_matrix = np.vstack([log_scales[valid_indices], np.ones(np.sum(valid_indices))]).T
        slope_param, _ = np.linalg.lstsq(coeff_matrix, log_rs[valid_indices], rcond=None)[0]
        
        return slope_param
    
    def compute_reward_risk_quotient(self, baseline_rate=0.02):
        """
        Custom reward-to-risk quotient calculation
        Annualized excess return divided by volatility measure
        """
        if self.length < 5:
            return np.nan
        
        percentage_moves = np.diff(self.price_vals) / self.price_vals[:-1]
        
        # Annualization factors
        mean_move = np.mean(percentage_moves) * 252
        volatility_measure = np.std(percentage_moves, ddof=1) * np.sqrt(252)
        
        if volatility_measure < 1e-8:
            return np.nan
        
        quotient = (mean_move - baseline_rate) / volatility_measure
        return quotient
    
    def compute_directional_strength(self, lookback_bars):
        """
        Directional strength using weighted regression with exponential decay
        """
        if self.length < lookback_bars + 1:
            return np.nan
        
        segment = self.price_vals[-lookback_bars:]
        time_indices = np.arange(len(segment))
        
        # Exponential weights favoring recent data
        decay_rate = 0.15
        weights = np.exp(decay_rate * time_indices / len(segment))
        
        # Weighted linear fit
        mean_time = np.average(time_indices, weights=weights)
        mean_price = np.average(segment, weights=weights)
        
        numerator = np.sum(weights * (time_indices - mean_time) * (segment - mean_price))
        denominator = np.sum(weights * (time_indices - mean_time) ** 2)
        
        if denominator < 1e-10:
            return np.nan
        
        slope = numerator / denominator
        
        # Normalize by average price level
        normalized = (slope / mean_price) * 100
        return normalized
    
    def compute_future_projection(self, training_window=90, projection_steps=5):
        """
        Future price projection using adaptive exponential smoothing
        """
        if self.length < training_window:
            return np.nan, np.nan
        
        training_data = self.price_vals[-training_window:]
        
        # Adaptive triple smoothing with learning rates
        alpha_level = 0.35
        alpha_trend = 0.12
        alpha_seasonal = 0.08
        
        smoothed_level = training_data[0]
        smoothed_trend = (training_data[-1] - training_data[0]) / len(training_data)
        
        for price_point in training_data[1:]:
            prev_level = smoothed_level
            smoothed_level = alpha_level * price_point + (1 - alpha_level) * (smoothed_level + smoothed_trend)
            smoothed_trend = alpha_trend * (smoothed_level - prev_level) + (1 - alpha_trend) * smoothed_trend
        
        projected_price = smoothed_level + projection_steps * smoothed_trend
        current_price = self.price_vals[-1]
        
        return_percentage = ((projected_price - current_price) / current_price) * 100
        
        return projected_price, return_percentage
    
    def compute_rate_momentum(self, interval=21):
        """
        Price rate momentum over specified interval
        """
        if self.length < interval + 1:
            return np.nan
        
        recent_price = self.price_vals[-1]
        historical_price = self.price_vals[-(interval + 1)]
        
        momentum_pct = ((recent_price - historical_price) / historical_price) * 100
        return momentum_pct
    
    def compute_avg_convergence_metric(self, quick_period=12, slow_period=26, smoothing_period=9):
        """
        Moving average convergence calculation with histogram output
        """
        if self.length < slow_period + smoothing_period:
            return np.nan
        
        price_series = pd.Series(self.price_vals)
        
        quick_ema = price_series.ewm(span=quick_period, adjust=False).mean()
        slow_ema = price_series.ewm(span=slow_period, adjust=False).mean()
        
        convergence_line = quick_ema - slow_ema
        signal_line = convergence_line.ewm(span=smoothing_period, adjust=False).mean()
        
        histogram_value = convergence_line.iloc[-1] - signal_line.iloc[-1]
        
        return histogram_value
    
    def compute_oscillator_index(self, evaluation_period=14):
        """
        Momentum oscillator based on gain/loss ratio
        """
        if self.length < evaluation_period + 1:
            return np.nan
        
        price_changes = np.diff(self.price_vals)
        
        positive_changes = np.where(price_changes > 0, price_changes, 0)
        negative_changes = np.where(price_changes < 0, -price_changes, 0)
        
        recent_gains = positive_changes[-evaluation_period:]
        recent_losses = negative_changes[-evaluation_period:]
        
        avg_gain = np.mean(recent_gains)
        avg_loss = np.mean(recent_losses)
        
        if avg_loss < 1e-10:
            return 100.0
        
        gain_loss_ratio = avg_gain / avg_loss
        oscillator_val = 100.0 - (100.0 / (1.0 + gain_loss_ratio))
        
        return oscillator_val
    
    def compute_volume_accumulation(self):
        """
        Volume accumulation/distribution metric
        """
        if self.vol_vals is None or self.length < 2:
            return np.nan
        
        accumulation = 0
        for idx in range(1, self.length):
            price_change = self.price_vals[idx] - self.price_vals[idx - 1]
            
            if price_change > 0:
                accumulation += self.vol_vals[idx]
            elif price_change < 0:
                accumulation -= self.vol_vals[idx]
        
        return accumulation


class RankingEngine:
    """Converts raw metrics into comparative rankings"""
    
    @staticmethod
    def create_ranking_scores(metric_values, high_is_superior=True):
        """
        Transform metric values into 0-10 ranking scores
        """
        values_array = np.asarray(metric_values, dtype=np.float64)
        finite_mask = np.isfinite(values_array)
        
        if not np.any(finite_mask):
            return np.full_like(values_array, np.nan)
        
        finite_values = values_array[finite_mask]
        val_min = np.min(finite_values)
        val_max = np.max(finite_values)
        
        if np.abs(val_max - val_min) < 1e-10:
            return np.full_like(values_array, 5.0)
        
        ranking_scores = np.full_like(values_array, np.nan)
        
        normalized_vals = (values_array[finite_mask] - val_min) / (val_max - val_min)
        
        if high_is_superior:
            ranking_scores[finite_mask] = normalized_vals * 10.0
        else:
            ranking_scores[finite_mask] = (1.0 - normalized_vals) * 10.0
        
        return ranking_scores
    
    @staticmethod
    def create_oscillator_ranking(oscillator_values, ideal_point=46, extreme_point=78):
        """
        Special ranking for oscillator where specific value is ideal
        """
        values_array = np.asarray(oscillator_values, dtype=np.float64)
        ranking_scores = np.full_like(values_array, np.nan)
        
        finite_mask = np.isfinite(values_array)
        
        for position in np.where(finite_mask)[0]:
            current_val = values_array[position]
            deviation = np.abs(current_val - ideal_point)
            max_deviation = np.abs(extreme_point - ideal_point)
            
            ranking_score = 10.0 * (1.0 - np.clip(deviation / max_deviation, 0, 1))
            ranking_scores[position] = np.maximum(0.0, ranking_score)
        
        return ranking_scores


def retrieve_market_data(ticker_symbol, time_period='1y'):
    """
    Fetch historical market data for analysis
    """
    try:
        ticker_object = yf.Ticker(ticker_symbol)
        historical_data = ticker_object.history(period=time_period, auto_adjust=True)
        
        if historical_data is None or len(historical_data) < 40:
            return None
        
        return {
            'close_prices': historical_data['Close'].to_numpy(),
            'trade_volumes': historical_data['Volume'].to_numpy() if 'Volume' in historical_data.columns else None
        }
    except Exception as error:
        return None


def evaluate_ticker(ticker_symbol):
    """
    Complete evaluation pipeline for single ticker
    """
    market_data = retrieve_market_data(ticker_symbol)
    
    if market_data is None:
        return None
    
    analyzer = TimeSeriesAnalyzer(market_data['close_prices'], market_data['trade_volumes'])
    
    # Compute all metrics
    persistence = analyzer.compute_persistence_coefficient(window_len=252)
    reward_risk = analyzer.compute_reward_risk_quotient()
    
    monthly_direction = analyzer.compute_directional_strength(21)
    weekly_direction = analyzer.compute_directional_strength(5)
    daily_direction = analyzer.compute_directional_strength(1)
    
    projected_price, projected_return = analyzer.compute_future_projection(90, 5)
    
    momentum_rate = analyzer.compute_rate_momentum(21)
    convergence = analyzer.compute_avg_convergence_metric()
    oscillator = analyzer.compute_oscillator_index()
    volume_metric = analyzer.compute_volume_accumulation()
    
    latest_price = market_data['close_prices'][-1]
    
    return {
        'Symbol': ticker_symbol,
        'LastPrice': latest_price,
        'Persistence': persistence,
        'RewardRisk': reward_risk,
        'DirectionMonth': monthly_direction,
        'DirectionWeek': weekly_direction,
        'DirectionDay': daily_direction,
        'ProjectedPrice': projected_price,
        'ProjectedReturn': projected_return,
        'MomentumRate': momentum_rate,
        'ConvergenceMetric': convergence,
        'OscillatorValue': oscillator,
        'VolumeMetric': volume_metric
    }


def execute_analysis_pipeline():
    """
    Main execution pipeline
    """
    print("=" * 75)
    print("QUANTITATIVE STOCK RANKING SYSTEM")
    print("=" * 75)
    
    # Load ticker list
    try:
        ticker_list = INPUT_TICKERS.read_text().strip().split('\n')
        ticker_list = [t.strip() for t in ticker_list if t.strip()]
    except FileNotFoundError:
        print(f"ERROR: Input file not found: {INPUT_TICKERS}")
        return
    
    print(f"\nLoaded {len(ticker_list)} tickers for evaluation")
    
    # Evaluate each ticker
    evaluation_results = []
    for position, ticker_symbol in enumerate(ticker_list, 1):
        print(f"  [{position:3d}/{len(ticker_list):3d}] {ticker_symbol:6s} ", end='')
        
        ticker_result = evaluate_ticker(ticker_symbol)
        
        if ticker_result:
            evaluation_results.append(ticker_result)
            print("SUCCESS")
        else:
            print("FAILED")
    
    if not evaluation_results:
        print("\nERROR: No successful evaluations")
        return
    
    print(f"\n{len(evaluation_results)} tickers evaluated successfully")
    
    # Build results dataframe
    results_df = pd.DataFrame(evaluation_results)
    
    # Generate ranking scores
    print("\nGenerating ranking scores...")
    
    ranker = RankingEngine()
    
    results_df['Rank_Persistence'] = ranker.create_ranking_scores(results_df['Persistence'], True)
    results_df['Rank_RewardRisk'] = ranker.create_ranking_scores(results_df['RewardRisk'], True)
    results_df['Rank_DirectionMonth'] = ranker.create_ranking_scores(results_df['DirectionMonth'], True)
    results_df['Rank_DirectionWeek'] = ranker.create_ranking_scores(results_df['DirectionWeek'], True)
    results_df['Rank_DirectionDay'] = ranker.create_ranking_scores(results_df['DirectionDay'], True)
    results_df['Rank_ProjectedReturn'] = ranker.create_ranking_scores(results_df['ProjectedReturn'], True)
    results_df['Rank_MomentumRate'] = ranker.create_ranking_scores(results_df['MomentumRate'], True)
    results_df['Rank_Convergence'] = ranker.create_ranking_scores(results_df['ConvergenceMetric'], True)
    results_df['Rank_Oscillator'] = ranker.create_oscillator_ranking(results_df['OscillatorValue'], 46, 78)
    results_df['Rank_Volume'] = ranker.create_ranking_scores(results_df['VolumeMetric'], True)
    
    # Calculate aggregate ranking
    ranking_columns = [col for col in results_df.columns if col.startswith('Rank_')]
    results_df['AggregateRank'] = results_df[ranking_columns].sum(axis=1, skipna=True)
    
    # Sort by aggregate ranking
    results_df = results_df.sort_values('AggregateRank', ascending=False)
    
    # Prepare output
    output_columns = [
        'Symbol', 'LastPrice', 'AggregateRank',
        'Persistence', 'Rank_Persistence',
        'RewardRisk', 'Rank_RewardRisk',
        'DirectionMonth', 'Rank_DirectionMonth',
        'DirectionWeek', 'Rank_DirectionWeek',
        'DirectionDay', 'Rank_DirectionDay',
        'ProjectedReturn', 'Rank_ProjectedReturn',
        'MomentumRate', 'Rank_MomentumRate',
        'ConvergenceMetric', 'Rank_Convergence',
        'OscillatorValue', 'Rank_Oscillator',
        'VolumeMetric', 'Rank_Volume'
    ]
    
    output_df = results_df[output_columns]
    
    # Save results
    OUTPUT_RESULTS.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(OUTPUT_RESULTS, index=False, float_format='%.5f')
    
    print(f"\nResults written to: {OUTPUT_RESULTS}")
    
    # Display top performers
    print("\n" + "=" * 75)
    print("TOP 10 RANKED STOCKS")
    print("=" * 75)
    
    display_columns = ['Symbol', 'LastPrice', 'AggregateRank', 'Persistence', 
                       'RewardRisk', 'ProjectedReturn', 'OscillatorValue']
    
    print(output_df[display_columns].head(10).to_string(index=False))
    
    print("\n" + "=" * 75)
    print("ANALYSIS PIPELINE COMPLETE")
    print("=" * 75)


if __name__ == "__main__":
    execute_analysis_pipeline()
