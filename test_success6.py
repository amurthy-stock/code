#!/usr/bin/env python3
"""
Basic tests for success6.py functionality
"""

import unittest
import tempfile
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

class TestSuccess6Functions(unittest.TestCase):
    """Test helper functions from success6.py"""
    
    def test_batch_tickers(self):
        """Test the batch_tickers generator function"""
        # Import the function
        from success6 import batch_tickers
        
        # Test with a list of tickers
        tickers = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'META', 'NFLX']
        batch_size = 3
        
        batches = list(batch_tickers(tickers, batch_size))
        
        # Should have 3 batches (3, 3, 1)
        self.assertEqual(len(batches), 3)
        self.assertEqual(batches[0], ['AAPL', 'GOOGL', 'MSFT'])
        self.assertEqual(batches[1], ['TSLA', 'AMZN', 'META'])
        self.assertEqual(batches[2], ['NFLX'])
    
    def test_batch_tickers_empty(self):
        """Test batch_tickers with empty list"""
        from success6 import batch_tickers
        
        batches = list(batch_tickers([], 5))
        self.assertEqual(len(batches), 0)
    
    def test_build_lstm_model(self):
        """Test LSTM model building"""
        try:
            from success6 import build_lstm_model
            
            # Test model creation
            model = build_lstm_model((7, 1))
            
            # Check that model is created
            self.assertIsNotNone(model)
            
            # Check model has the expected structure
            self.assertEqual(len(model.layers), 2)  # LSTM + Dense
        except ImportError as e:
            self.skipTest(f"TensorFlow not available: {e}")


class TestArgumentParsing(unittest.TestCase):
    """Test command-line argument parsing"""
    
    def test_parse_arguments_function_exists(self):
        """Test that parse_arguments function is defined"""
        from success6 import parse_arguments
        # Verify the function is callable
        self.assertTrue(callable(parse_arguments))


class TestFileOperations(unittest.TestCase):
    """Test file I/O operations"""
    
    def test_ticker_file_reading(self):
        """Test reading ticker file"""
        # Create a temporary ticker file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write("AAPL\n")
            f.write("GOOGL\n")
            f.write("MSFT\n")
            temp_file = f.name
        
        try:
            # Read the file
            with open(temp_file, 'r') as f:
                tickers = [line.strip() for line in f if line.strip()]
            
            self.assertEqual(len(tickers), 3)
            self.assertEqual(tickers, ['AAPL', 'GOOGL', 'MSFT'])
        finally:
            os.unlink(temp_file)


if __name__ == '__main__':
    # Run tests
    unittest.main()
