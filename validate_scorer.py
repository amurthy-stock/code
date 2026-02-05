"""
Validation tests for stock scoring system
Tests core mathematical functions with synthetic data
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stock_scorer import (
    detrended_fluctuation_hurst,
    reward_volatility_ratio,
    polynomial_direction,
    exponential_projection,
    velocity_indicator,
    dual_ema_divergence,
    strength_index_oscillator,
    directional_volume_flow,
    map_to_decile,
    custom_oscillator_scoring
)


def test_persistence_calculation():
    """Validate DFA Hurst calculation"""
    print("Testing persistence exponent...")
    
    # Create synthetic trending data
    np.random.seed(123)
    trending = np.cumsum(np.random.randn(300) * 0.5) + 100
    
    persist = detrended_fluctuation_hurst(trending, 252)
    
    assert not np.isnan(persist), "Should return valid number"
    assert 0 < persist < 2, f"Persistence {persist} out of expected range"
    print(f"  ✓ Persistence: {persist:.3f}")


def test_reward_risk_calculation():
    """Validate risk-adjusted return metric"""
    print("Testing reward-risk ratio...")
    
    np.random.seed(456)
    prices = np.exp(np.cumsum(np.random.randn(300) * 0.01)) * 100
    
    ratio = reward_volatility_ratio(prices)
    
    assert not np.isnan(ratio), "Should return valid number"
    print(f"  ✓ Reward-Risk: {ratio:.3f}")


def test_directional_analysis():
    """Validate polynomial trend detection"""
    print("Testing directional analysis...")
    
    # Create uptrend
    uptrend = np.linspace(100, 120, 21)
    direction = polynomial_direction(uptrend, 21)
    
    assert direction > 0, "Should detect uptrend"
    print(f"  ✓ Uptrend detected: {direction:.2f}%")
    
    # Create downtrend
    downtrend = np.linspace(120, 100, 21)
    direction = polynomial_direction(downtrend, 21)
    
    assert direction < 0, "Should detect downtrend"
    print(f"  ✓ Downtrend detected: {direction:.2f}%")


def test_projection_system():
    """Validate exponential weighted projection"""
    print("Testing projection system...")
    
    np.random.seed(789)
    prices = np.linspace(100, 110, 90) + np.random.randn(90) * 0.5
    
    projected, gain = exponential_projection(prices, 90, 5)
    
    assert not np.isnan(projected), "Should return projected price"
    assert not np.isnan(gain), "Should return gain percentage"
    assert projected > 0, "Projected price should be positive"
    print(f"  ✓ Projection: ${projected:.2f}, Gain: {gain:.2f}%")


def test_velocity_calculation():
    """Validate velocity indicator"""
    print("Testing velocity indicator...")
    
    prices = np.linspace(100, 110, 50)
    velocity = velocity_indicator(prices, 21)
    
    assert not np.isnan(velocity), "Should return valid velocity"
    assert velocity > 0, "Should show positive momentum"
    print(f"  ✓ Velocity: {velocity:.2f}%")


def test_divergence_indicator():
    """Validate dual EMA divergence"""
    print("Testing EMA divergence...")
    
    np.random.seed(101)
    prices = np.cumsum(np.random.randn(100)) + 100
    
    divergence = dual_ema_divergence(prices)
    
    assert not np.isnan(divergence), "Should return valid divergence"
    print(f"  ✓ Divergence: {divergence:.3f}")


def test_oscillator_calculation():
    """Validate strength oscillator"""
    print("Testing strength oscillator...")
    
    np.random.seed(202)
    prices = 100 + np.cumsum(np.random.randn(100) * 0.5)
    
    oscillator = strength_index_oscillator(prices, 14)
    
    assert not np.isnan(oscillator), "Should return valid oscillator"
    assert 0 <= oscillator <= 100, f"Oscillator {oscillator} out of 0-100 range"
    print(f"  ✓ Oscillator: {oscillator:.1f}")


def test_volume_flow_indicator():
    """Validate directional volume flow"""
    print("Testing volume flow...")
    
    prices = np.array([100, 101, 102, 101, 103, 104, 103])
    volumes = np.array([1000, 1100, 1200, 1150, 1300, 1400, 1350])
    
    flow = directional_volume_flow(prices, volumes)
    
    assert not np.isnan(flow), "Should return valid flow"
    print(f"  ✓ Volume Flow: {flow:.0f}")


def test_scoring_normalization():
    """Validate decile mapping"""
    print("Testing score normalization...")
    
    # Normal case
    score = map_to_decile(5.0, 0.0, 10.0)
    assert score == 5.0, "Should map to midpoint"
    
    # Boundaries
    score = map_to_decile(0.0, 0.0, 10.0)
    assert score == 0.0, "Should map to minimum"
    
    score = map_to_decile(10.0, 0.0, 10.0)
    assert score == 10.0, "Should map to maximum"
    
    # Inverted
    score = map_to_decile(2.0, 0.0, 10.0, reverse=True)
    assert score == 8.0, "Should invert correctly"
    
    print("  ✓ Normalization working correctly")


def test_oscillator_scoring():
    """Validate custom oscillator scoring"""
    print("Testing oscillator scoring...")
    
    # Optimal value
    score = custom_oscillator_scoring(46.0)
    assert score == 10.0, f"Should score 10 at optimal, got {score}"
    
    # Worst value
    score = custom_oscillator_scoring(78.0)
    assert score < 1.0, f"Should score near 0 at worst, got {score}"
    
    # Middle value
    score = custom_oscillator_scoring(62.0)
    assert 0 < score < 10, f"Should score between 0-10, got {score}"
    
    print("  ✓ Oscillator scoring working correctly")


def test_edge_cases():
    """Test edge case handling"""
    print("Testing edge cases...")
    
    # Empty array
    result = detrended_fluctuation_hurst(np.array([]), 252)
    assert np.isnan(result), "Should handle empty array"
    
    # Insufficient data
    short_data = np.array([100, 101, 102])
    result = reward_volatility_ratio(short_data)
    assert np.isnan(result), "Should handle insufficient data"
    
    # NaN in scoring
    score = map_to_decile(np.nan, 0.0, 10.0)
    assert score == 0.0, "Should return 0 for NaN"
    
    score = custom_oscillator_scoring(np.nan)
    assert score == 0.0, "Should return 0 for NaN"
    
    print("  ✓ Edge cases handled correctly")


def run_all_tests():
    """Execute all validation tests"""
    print("=" * 60)
    print("STOCK SCORER VALIDATION TESTS")
    print("=" * 60)
    print()
    
    tests = [
        test_persistence_calculation,
        test_reward_risk_calculation,
        test_directional_analysis,
        test_projection_system,
        test_velocity_calculation,
        test_divergence_indicator,
        test_oscillator_calculation,
        test_volume_flow_indicator,
        test_scoring_normalization,
        test_oscillator_scoring,
        test_edge_cases
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
            print()
        except AssertionError as e:
            print(f"  ✗ FAILED: {e}")
            failed += 1
            print()
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            failed += 1
            print()
    
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
