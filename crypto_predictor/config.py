"""
Configuration settings for the crypto price prediction system.
"""

import os
from typing import Dict, List, Any

class Config:
    """Configuration class for crypto prediction settings."""
    
    # Data settings
    DEFAULT_SYMBOL = "BTCUSDT"
    DEFAULT_INTERVAL = "1h"
    DEFAULT_LOOKBACK_DAYS = 30
    
    # Technical indicator settings
    SMA_PERIODS = [5, 10, 20, 50]
    EMA_PERIODS = [12, 26]
    RSI_PERIOD = 14
    MACD_FAST = 12
    MACD_SLOW = 26
    MACD_SIGNAL = 9
    BOLLINGER_PERIOD = 20
    BOLLINGER_STD = 2
    
    # Machine learning settings
    LSTM_SEQUENCE_LENGTH = 60
    LSTM_EPOCHS = 50
    LSTM_BATCH_SIZE = 32
    TRAIN_TEST_SPLIT = 0.8
    
    # Trading settings
    RISK_PERCENTAGE = 0.02  # 2% risk per trade
    STOP_LOSS_PERCENTAGE = 0.03  # 3% stop loss
    TAKE_PROFIT_PERCENTAGE = 0.06  # 6% take profit
    
    # API settings (can be overridden with environment variables)
    BINANCE_API_KEY = os.getenv('BINANCE_API_KEY', '')
    BINANCE_SECRET_KEY = os.getenv('BINANCE_SECRET_KEY', '')
    
    # Logging settings
    LOG_LEVEL = "INFO"
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    @classmethod
    def get_supported_symbols(cls) -> List[str]:
        """Return list of supported cryptocurrency symbols."""
        return [
            "BTCUSDT", "ETHUSDT", "ADAUSDT", "DOTUSDT", "LINKUSDT",
            "BNBUSDT", "XRPUSDT", "LTCUSDT", "BCHUSDT", "XLMUSDT"
        ]
    
    @classmethod
    def get_timeframes(cls) -> List[str]:
        """Return list of supported timeframes."""
        return ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]
    
    @classmethod
    def validate_symbol(cls, symbol: str) -> bool:
        """Validate if symbol is supported."""
        return symbol.upper() in cls.get_supported_symbols()
    
    @classmethod
    def validate_timeframe(cls, timeframe: str) -> bool:
        """Validate if timeframe is supported."""
        return timeframe.lower() in cls.get_timeframes()