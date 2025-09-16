"""
Unit tests for the crypto price prediction system.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crypto_predictor.config import Config
from crypto_predictor.data_fetcher import CryptoDataFetcher
from crypto_predictor.indicators import TechnicalIndicators
from crypto_predictor.models import LinearRegressionModel, LSTMModel
from crypto_predictor.predictor import CryptoPricePredictor


class TestConfig(unittest.TestCase):
    """Test configuration class."""
    
    def test_validate_symbol(self):
        """Test symbol validation."""
        self.assertTrue(Config.validate_symbol('BTCUSDT'))
        self.assertTrue(Config.validate_symbol('btcusdt'))
        self.assertFalse(Config.validate_symbol('INVALID'))
        
    def test_validate_timeframe(self):
        """Test timeframe validation."""
        self.assertTrue(Config.validate_timeframe('1h'))
        self.assertTrue(Config.validate_timeframe('1H'))
        self.assertFalse(Config.validate_timeframe('2h'))
        
    def test_get_supported_symbols(self):
        """Test getting supported symbols."""
        symbols = Config.get_supported_symbols()
        self.assertIsInstance(symbols, list)
        self.assertIn('BTCUSDT', symbols)
        self.assertGreater(len(symbols), 0)


class TestCryptoDataFetcher(unittest.TestCase):
    """Test data fetcher class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.data_fetcher = CryptoDataFetcher()
        
    def test_generate_sample_data(self):
        """Test sample data generation."""
        df = self.data_fetcher.generate_sample_data('BTCUSDT', days=10)
        
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df), 0)
        self.assertIn('open', df.columns)
        self.assertIn('high', df.columns)
        self.assertIn('low', df.columns)
        self.assertIn('close', df.columns)
        self.assertIn('volume', df.columns)
        
        # Test OHLC consistency
        for _, row in df.iterrows():
            self.assertGreaterEqual(row['high'], row['open'])
            self.assertGreaterEqual(row['high'], row['close'])
            self.assertLessEqual(row['low'], row['open'])
            self.assertLessEqual(row['low'], row['close'])
            
    def test_sample_data_realistic_prices(self):
        """Test that sample data generates realistic prices."""
        df = self.data_fetcher.generate_sample_data('BTCUSDT', days=5)
        
        # Prices should be in a reasonable range for BTC
        self.assertGreater(df['close'].min(), 10000)  # At least $10k
        self.assertLess(df['close'].max(), 200000)    # Less than $200k
        
        # Volume should be positive
        self.assertTrue((df['volume'] > 0).all())


class TestTechnicalIndicators(unittest.TestCase):
    """Test technical indicators class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create sample data
        dates = pd.date_range(start='2023-01-01', periods=100, freq='1H')
        np.random.seed(42)
        prices = 50000 + np.cumsum(np.random.randn(100) * 100)
        
        self.sample_data = pd.DataFrame({
            'open': prices + np.random.randn(100) * 50,
            'high': prices + np.abs(np.random.randn(100)) * 100,
            'low': prices - np.abs(np.random.randn(100)) * 100,
            'close': prices,
            'volume': np.random.uniform(1000, 5000, 100)
        }, index=dates)
        
        # Ensure OHLC consistency
        for i in range(len(self.sample_data)):
            row = self.sample_data.iloc[i]
            high = max(row['open'], row['close']) + abs(row['high'] - max(row['open'], row['close']))
            low = min(row['open'], row['close']) - abs(min(row['open'], row['close']) - row['low'])
            self.sample_data.loc[row.name, 'high'] = high
            self.sample_data.loc[row.name, 'low'] = low
        
    def test_simple_moving_average(self):
        """Test SMA calculation."""
        sma = TechnicalIndicators.simple_moving_average(self.sample_data['close'], 10)
        
        self.assertIsInstance(sma, pd.Series)
        self.assertEqual(len(sma), len(self.sample_data))
        # First 9 values should be NaN
        self.assertTrue(pd.isna(sma.iloc[:9]).all())
        # 10th value should be the average of first 10 values
        expected = self.sample_data['close'].iloc[:10].mean()
        self.assertAlmostEqual(sma.iloc[9], expected, places=2)
        
    def test_exponential_moving_average(self):
        """Test EMA calculation."""
        ema = TechnicalIndicators.exponential_moving_average(self.sample_data['close'], 12)
        
        self.assertIsInstance(ema, pd.Series)
        self.assertEqual(len(ema), len(self.sample_data))
        self.assertFalse(pd.isna(ema.iloc[-1]))
        
    def test_rsi(self):
        """Test RSI calculation."""
        rsi = TechnicalIndicators.relative_strength_index(self.sample_data['close'])
        
        self.assertIsInstance(rsi, pd.Series)
        # RSI should be between 0 and 100
        valid_rsi = rsi.dropna()
        self.assertTrue((valid_rsi >= 0).all())
        self.assertTrue((valid_rsi <= 100).all())
        
    def test_macd(self):
        """Test MACD calculation."""
        macd_data = TechnicalIndicators.macd(self.sample_data['close'])
        
        self.assertIsInstance(macd_data, dict)
        self.assertIn('macd', macd_data)
        self.assertIn('signal', macd_data)
        self.assertIn('histogram', macd_data)
        
        # All should be pandas Series
        for key, value in macd_data.items():
            self.assertIsInstance(value, pd.Series)
            
    def test_bollinger_bands(self):
        """Test Bollinger Bands calculation."""
        bb_data = TechnicalIndicators.bollinger_bands(self.sample_data['close'])
        
        self.assertIsInstance(bb_data, dict)
        self.assertIn('upper', bb_data)
        self.assertIn('middle', bb_data)
        self.assertIn('lower', bb_data)
        
        # Upper should be greater than middle, middle greater than lower
        for i in range(20, len(self.sample_data)):  # Skip initial NaN values
            upper = bb_data['upper'].iloc[i]
            middle = bb_data['middle'].iloc[i]
            lower = bb_data['lower'].iloc[i]
            
            if not (pd.isna(upper) or pd.isna(middle) or pd.isna(lower)):
                self.assertGreater(upper, middle)
                self.assertGreater(middle, lower)
                
    def test_calculate_all_indicators(self):
        """Test calculating all indicators."""
        df_with_indicators = TechnicalIndicators.calculate_all_indicators(self.sample_data)
        
        self.assertIsInstance(df_with_indicators, pd.DataFrame)
        
        # Check that new columns were added
        expected_columns = ['SMA_5', 'SMA_10', 'SMA_20', 'SMA_50', 'EMA_12', 'EMA_26',
                          'RSI', 'MACD', 'MACD_Signal', 'MACD_Histogram',
                          'BB_Upper', 'BB_Middle', 'BB_Lower', 'Stoch_K', 'Stoch_D',
                          'Williams_R', 'ATR']
        
        for col in expected_columns:
            self.assertIn(col, df_with_indicators.columns)


class TestLinearRegressionModel(unittest.TestCase):
    """Test Linear Regression model."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model = LinearRegressionModel()
        
        # Create sample data with indicators
        data_fetcher = CryptoDataFetcher()
        raw_data = data_fetcher.generate_sample_data('BTCUSDT', days=30)
        self.sample_data = TechnicalIndicators.calculate_all_indicators(raw_data)
        
    def test_model_initialization(self):
        """Test model initialization."""
        self.assertIsNotNone(self.model.model)
        self.assertIsNotNone(self.model.scaler)
        self.assertFalse(self.model.is_trained)
        
    def test_prepare_features(self):
        """Test feature preparation."""
        X, y = self.model.prepare_features(self.sample_data)
        
        self.assertIsInstance(X, np.ndarray)
        self.assertIsInstance(y, np.ndarray)
        self.assertEqual(len(X), len(y))
        self.assertGreater(X.shape[1], 0)  # Should have features
        
    def test_training(self):
        """Test model training."""
        metrics = self.model.train(self.sample_data)
        
        self.assertIsInstance(metrics, dict)
        self.assertIn('r2', metrics)
        self.assertIn('mae', metrics)
        self.assertIn('mse', metrics)
        self.assertIn('rmse', metrics)
        self.assertTrue(self.model.is_trained)
        
    def test_prediction_after_training(self):
        """Test making predictions after training."""
        self.model.train(self.sample_data)
        prediction = self.model.predict_next_price(self.sample_data)
        
        self.assertIsInstance(prediction, float)
        self.assertGreater(prediction, 0)


class TestCryptoPricePredictor(unittest.TestCase):
    """Test main predictor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.predictor = CryptoPricePredictor('BTCUSDT', use_sample_data=True)
        
    def test_initialization(self):
        """Test predictor initialization."""
        self.assertEqual(self.predictor.symbol, 'BTCUSDT')
        self.assertTrue(self.predictor.use_sample_data)
        self.assertIsNotNone(self.predictor.data_fetcher)
        self.assertIsNotNone(self.predictor.technical_indicators)
        
    def test_fetch_data(self):
        """Test data fetching."""
        df = self.predictor.fetch_data(days=10)
        
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df), 0)
        self.assertIsNotNone(self.predictor.raw_data)
        
    def test_process_data(self):
        """Test data processing."""
        self.predictor.fetch_data(days=10)
        df = self.predictor.process_data()
        
        self.assertIsInstance(df, pd.DataFrame)
        self.assertIsNotNone(self.predictor.processed_data)
        # Should have technical indicators
        self.assertIn('RSI', df.columns)
        self.assertIn('MACD', df.columns)
        
    def test_generate_technical_signals(self):
        """Test technical signal generation."""
        self.predictor.fetch_data(days=10)
        self.predictor.process_data()
        signals = self.predictor.generate_technical_signals()
        
        self.assertIsInstance(signals, dict)
        self.assertIn('RSI', signals)
        self.assertIn('MACD', signals)
        
        # Signals should be valid strings
        valid_signals = ['BULLISH', 'BEARISH', 'NEUTRAL', 'OVERBOUGHT', 'OVERSOLD']
        for signal_value in signals.values():
            self.assertIn(signal_value, valid_signals)


if __name__ == '__main__':
    unittest.main()