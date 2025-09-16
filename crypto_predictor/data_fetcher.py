"""
Data fetching utilities for cryptocurrency price data.
"""

import pandas as pd
import numpy as np
# Optional imports
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    yf = None
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging
from .config import Config

# Configure logging
logging.basicConfig(level=Config.LOG_LEVEL, format=Config.LOG_FORMAT)
logger = logging.getLogger(__name__)


class CryptoDataFetcher:
    """Class for fetching cryptocurrency price data from various sources."""
    
    def __init__(self):
        self.session = requests.Session()
        
    def fetch_binance_data(self, symbol: str, interval: str = Config.DEFAULT_INTERVAL, 
                          limit: int = 1000) -> Optional[pd.DataFrame]:
        """
        Fetch data from Binance API.
        
        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            interval: Time interval (e.g., '1h', '4h', '1d')
            limit: Number of data points to fetch
            
        Returns:
            DataFrame with OHLCV data or None if failed
        """
        try:
            url = "https://api.binance.com/api/v3/klines"
            params = {
                'symbol': symbol.upper(),
                'interval': interval,
                'limit': limit
            }
            
            response = self.session.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            # Convert to DataFrame
            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Convert price columns to float
            price_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in price_columns:
                df[col] = df[col].astype(float)
            
            # Keep only essential columns
            df = df[price_columns]
            
            logger.info(f"Successfully fetched {len(df)} data points for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching Binance data for {symbol}: {str(e)}")
            return None
    
    def fetch_yahoo_data(self, symbol: str, period: str = "30d", 
                        interval: str = "1h") -> Optional[pd.DataFrame]:
        """
        Fetch data from Yahoo Finance (for crypto symbols ending with -USD).
        
        Args:
            symbol: Trading symbol (e.g., 'BTC-USD')
            period: Time period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
            interval: Time interval ('1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')
            
        Returns:
            DataFrame with OHLCV data or None if failed
        """
        if not YFINANCE_AVAILABLE:
            logger.warning("yfinance not available. Please install it: pip install yfinance")
            return None
            
        try:
            # Convert symbol format if needed (BTCUSDT -> BTC-USD)
            if symbol.endswith('USDT'):
                yahoo_symbol = symbol.replace('USDT', '-USD')
            else:
                yahoo_symbol = symbol
            
            ticker = yf.Ticker(yahoo_symbol)
            df = ticker.history(period=period, interval=interval)
            
            if df.empty:
                logger.warning(f"No data found for {yahoo_symbol}")
                return None
            
            # Rename columns to match our format
            df.columns = [col.lower() for col in df.columns]
            df.index.name = 'timestamp'
            
            logger.info(f"Successfully fetched {len(df)} data points for {yahoo_symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching Yahoo data for {symbol}: {str(e)}")
            return None
    
    def fetch_coinapi_data(self, symbol: str, start_date: datetime, 
                          end_date: datetime, api_key: str) -> Optional[pd.DataFrame]:
        """
        Fetch data from CoinAPI (requires API key).
        
        Args:
            symbol: Trading symbol (e.g., 'BITSTAMP_SPOT_BTC_USD')
            start_date: Start date for data
            end_date: End date for data
            api_key: CoinAPI key
            
        Returns:
            DataFrame with OHLCV data or None if failed
        """
        try:
            url = f"https://rest.coinapi.io/v1/ohlcv/{symbol}/history"
            headers = {'X-CoinAPI-Key': api_key}
            params = {
                'period_id': '1HRS',
                'time_start': start_date.isoformat(),
                'time_end': end_date.isoformat()
            }
            
            response = self.session.get(url, headers=headers, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            if not data:
                logger.warning(f"No data found for {symbol}")
                return None
            
            df = pd.DataFrame(data)
            df['time_period_start'] = pd.to_datetime(df['time_period_start'])
            df.set_index('time_period_start', inplace=True)
            df.index.name = 'timestamp'
            
            # Rename columns
            column_mapping = {
                'price_open': 'open',
                'price_high': 'high',
                'price_low': 'low',
                'price_close': 'close',
                'volume_traded': 'volume'
            }
            df.rename(columns=column_mapping, inplace=True)
            
            logger.info(f"Successfully fetched {len(df)} data points for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching CoinAPI data for {symbol}: {str(e)}")
            return None
    
    def get_crypto_data(self, symbol: str, source: str = "binance", 
                       **kwargs) -> Optional[pd.DataFrame]:
        """
        Generic method to fetch cryptocurrency data from specified source.
        
        Args:
            symbol: Trading symbol
            source: Data source ('binance', 'yahoo', 'coinapi')
            **kwargs: Additional arguments for specific data sources
            
        Returns:
            DataFrame with OHLCV data or None if failed
        """
        if not Config.validate_symbol(symbol):
            logger.warning(f"Symbol {symbol} is not in supported symbols list")
        
        if source.lower() == "binance":
            return self.fetch_binance_data(symbol, **kwargs)
        elif source.lower() == "yahoo":
            return self.fetch_yahoo_data(symbol, **kwargs)
        elif source.lower() == "coinapi":
            return self.fetch_coinapi_data(symbol, **kwargs)
        else:
            logger.error(f"Unsupported data source: {source}")
            return None
    
    def generate_sample_data(self, symbol: str = Config.DEFAULT_SYMBOL, 
                           days: int = Config.DEFAULT_LOOKBACK_DAYS) -> pd.DataFrame:
        """
        Generate sample cryptocurrency data for testing purposes.
        
        Args:
            symbol: Trading symbol
            days: Number of days of sample data
            
        Returns:
            DataFrame with sample OHLCV data
        """
        logger.info(f"Generating {days} days of sample data for {symbol}")
        
        # Create date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        date_range = pd.date_range(start=start_date, end=end_date, freq='1H')
        
        # Generate sample price data with some realistic volatility
        np.random.seed(42)  # For reproducible results
        
        # Starting price
        base_price = 50000 if 'BTC' in symbol else 3000
        
        # Generate price series with random walk and trend
        returns = np.random.normal(0.0001, 0.02, len(date_range))  # Small upward trend with volatility
        price_series = base_price * np.exp(np.cumsum(returns))
        
        # Generate OHLCV data
        data = []
        for i, (timestamp, close_price) in enumerate(zip(date_range, price_series)):
            # Generate high/low around close price
            volatility = np.random.uniform(0.005, 0.02)  # 0.5% to 2% volatility
            high = close_price * (1 + volatility)
            low = close_price * (1 - volatility)
            
            # Open price is previous close (for first entry, use close)
            if i == 0:
                open_price = close_price
            else:
                open_price = data[i-1][4]  # Previous close
            
            # Ensure OHLC consistency
            high = max(high, open_price, close_price)
            low = min(low, open_price, close_price)
            
            # Generate volume
            volume = np.random.uniform(1000, 5000)
            
            data.append([timestamp, open_price, high, low, close_price, volume])
        
        df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df.set_index('timestamp', inplace=True)
        df.index.name = 'timestamp'
        
        logger.info(f"Generated {len(df)} sample data points for {symbol}")
        return df
    
    def get_latest_price(self, symbol: str, source: str = "binance") -> Optional[float]:
        """
        Get the latest price for a cryptocurrency.
        
        Args:
            symbol: Trading symbol
            source: Data source
            
        Returns:
            Latest price or None if failed
        """
        try:
            if source.lower() == "binance":
                url = "https://api.binance.com/api/v3/ticker/price"
                params = {'symbol': symbol.upper()}
                response = self.session.get(url, params=params)
                response.raise_for_status()
                data = response.json()
                return float(data['price'])
        except Exception as e:
            logger.error(f"Error fetching latest price for {symbol}: {str(e)}")
            return None