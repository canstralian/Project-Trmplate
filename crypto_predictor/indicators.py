"""
Technical indicators for cryptocurrency price analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from .config import Config


class TechnicalIndicators:
    """Class for calculating various technical indicators."""
    
    @staticmethod
    def simple_moving_average(data: pd.Series, period: int) -> pd.Series:
        """Calculate Simple Moving Average (SMA)."""
        return data.rolling(window=period).mean()
    
    @staticmethod
    def exponential_moving_average(data: pd.Series, period: int) -> pd.Series:
        """Calculate Exponential Moving Average (EMA)."""
        return data.ewm(span=period).mean()
    
    @staticmethod
    def relative_strength_index(data: pd.Series, period: int = Config.RSI_PERIOD) -> pd.Series:
        """
        Calculate Relative Strength Index (RSI).
        
        Args:
            data: Price data series
            period: Period for RSI calculation
            
        Returns:
            RSI values as pandas Series
        """
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def macd(data: pd.Series, 
             fast: int = Config.MACD_FAST, 
             slow: int = Config.MACD_SLOW, 
             signal: int = Config.MACD_SIGNAL) -> Dict[str, pd.Series]:
        """
        Calculate MACD (Moving Average Convergence Divergence).
        
        Args:
            data: Price data series
            fast: Fast EMA period
            slow: Slow EMA period
            signal: Signal line EMA period
            
        Returns:
            Dictionary with MACD, signal line, and histogram
        """
        ema_fast = TechnicalIndicators.exponential_moving_average(data, fast)
        ema_slow = TechnicalIndicators.exponential_moving_average(data, slow)
        
        macd_line = ema_fast - ema_slow
        signal_line = TechnicalIndicators.exponential_moving_average(macd_line, signal)
        histogram = macd_line - signal_line
        
        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }
    
    @staticmethod
    def bollinger_bands(data: pd.Series, 
                       period: int = Config.BOLLINGER_PERIOD, 
                       std_dev: float = Config.BOLLINGER_STD) -> Dict[str, pd.Series]:
        """
        Calculate Bollinger Bands.
        
        Args:
            data: Price data series
            period: Moving average period
            std_dev: Standard deviation multiplier
            
        Returns:
            Dictionary with upper, middle, and lower bands
        """
        sma = TechnicalIndicators.simple_moving_average(data, period)
        std = data.rolling(window=period).std()
        
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        
        return {
            'upper': upper_band,
            'middle': sma,
            'lower': lower_band
        }
    
    @staticmethod
    def stochastic_oscillator(high: pd.Series, low: pd.Series, close: pd.Series, 
                            k_period: int = 14, d_period: int = 3) -> Dict[str, pd.Series]:
        """
        Calculate Stochastic Oscillator.
        
        Args:
            high: High price series
            low: Low price series
            close: Close price series
            k_period: %K period
            d_period: %D period
            
        Returns:
            Dictionary with %K and %D values
        """
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_period).mean()
        
        return {
            'k_percent': k_percent,
            'd_percent': d_percent
        }
    
    @staticmethod
    def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Williams %R indicator."""
        highest_high = high.rolling(window=period).max()
        lowest_low = low.rolling(window=period).min()
        
        williams_r = -100 * ((highest_high - close) / (highest_high - lowest_low))
        return williams_r
    
    @staticmethod
    def average_true_range(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Average True Range (ATR)."""
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        
        return atr
    
    @classmethod
    def calculate_all_indicators(cls, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all technical indicators for a given dataframe.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with all technical indicators added
        """
        result_df = df.copy()
        
        # Moving averages
        for period in Config.SMA_PERIODS:
            result_df[f'SMA_{period}'] = cls.simple_moving_average(df['close'], period)
        
        for period in Config.EMA_PERIODS:
            result_df[f'EMA_{period}'] = cls.exponential_moving_average(df['close'], period)
        
        # RSI
        result_df['RSI'] = cls.relative_strength_index(df['close'])
        
        # MACD
        macd_data = cls.macd(df['close'])
        result_df['MACD'] = macd_data['macd']
        result_df['MACD_Signal'] = macd_data['signal']
        result_df['MACD_Histogram'] = macd_data['histogram']
        
        # Bollinger Bands
        bb_data = cls.bollinger_bands(df['close'])
        result_df['BB_Upper'] = bb_data['upper']
        result_df['BB_Middle'] = bb_data['middle']
        result_df['BB_Lower'] = bb_data['lower']
        
        # Stochastic
        stoch_data = cls.stochastic_oscillator(df['high'], df['low'], df['close'])
        result_df['Stoch_K'] = stoch_data['k_percent']
        result_df['Stoch_D'] = stoch_data['d_percent']
        
        # Williams %R
        result_df['Williams_R'] = cls.williams_r(df['high'], df['low'], df['close'])
        
        # ATR
        result_df['ATR'] = cls.average_true_range(df['high'], df['low'], df['close'])
        
        return result_df