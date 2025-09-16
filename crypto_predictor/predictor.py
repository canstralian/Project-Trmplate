"""
Main cryptocurrency price prediction system.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import logging
import warnings
warnings.filterwarnings('ignore')

from .config import Config
from .data_fetcher import CryptoDataFetcher
from .indicators import TechnicalIndicators
from .models import LinearRegressionModel, LSTMModel, EnsembleModel

# Configure logging
logging.basicConfig(level=Config.LOG_LEVEL, format=Config.LOG_FORMAT)
logger = logging.getLogger(__name__)


class CryptoPricePredictor:
    """
    Main class for cryptocurrency price prediction using multiple algorithms.
    
    This class combines technical analysis, machine learning models, and ensemble methods
    to provide comprehensive price predictions for cryptocurrency day trading.
    """
    
    def __init__(self, symbol: str = Config.DEFAULT_SYMBOL, 
                 data_source: str = "binance", use_sample_data: bool = True):
        """
        Initialize the crypto price predictor.
        
        Args:
            symbol: Cryptocurrency symbol (e.g., 'BTCUSDT')
            data_source: Data source ('binance', 'yahoo', 'sample')
            use_sample_data: Whether to use sample data for demonstration
        """
        self.symbol = symbol.upper()
        self.data_source = data_source
        self.use_sample_data = use_sample_data
        
        # Initialize components
        self.data_fetcher = CryptoDataFetcher()
        self.technical_indicators = TechnicalIndicators()
        
        # Initialize models
        self.linear_model = LinearRegressionModel()
        self.lstm_model = LSTMModel()
        self.ensemble_model = EnsembleModel()
        
        # Data storage
        self.raw_data = None
        self.processed_data = None
        self.latest_predictions = None
        
        logger.info(f"CryptoPricePredictor initialized for {self.symbol}")
        
    def fetch_data(self, days: int = Config.DEFAULT_LOOKBACK_DAYS, 
                   interval: str = Config.DEFAULT_INTERVAL) -> pd.DataFrame:
        """
        Fetch cryptocurrency data.
        
        Args:
            days: Number of days of historical data
            interval: Time interval for data
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            if self.use_sample_data:
                logger.info("Using sample data for demonstration")
                self.raw_data = self.data_fetcher.generate_sample_data(self.symbol, days)
            else:
                logger.info(f"Fetching real data for {self.symbol}")
                self.raw_data = self.data_fetcher.get_crypto_data(
                    self.symbol, self.data_source, interval=interval, limit=days*24
                )
                
                if self.raw_data is None:
                    logger.warning("Failed to fetch real data, falling back to sample data")
                    self.raw_data = self.data_fetcher.generate_sample_data(self.symbol, days)
            
            logger.info(f"Data fetched successfully: {len(self.raw_data)} data points")
            return self.raw_data
            
        except Exception as e:
            logger.error(f"Error fetching data: {str(e)}")
            raise
            
    def process_data(self) -> pd.DataFrame:
        """
        Process raw data by adding technical indicators.
        
        Returns:
            DataFrame with technical indicators added
        """
        if self.raw_data is None:
            raise ValueError("No raw data available. Call fetch_data() first.")
            
        try:
            logger.info("Processing data and calculating technical indicators")
            self.processed_data = self.technical_indicators.calculate_all_indicators(self.raw_data)
            
            # Remove rows with NaN values (from indicator calculations)
            initial_count = len(self.processed_data)
            self.processed_data = self.processed_data.dropna()
            final_count = len(self.processed_data)
            
            logger.info(f"Data processing complete. {initial_count - final_count} rows removed due to NaN values")
            return self.processed_data
            
        except Exception as e:
            logger.error(f"Error processing data: {str(e)}")
            raise
            
    def train_models(self) -> Dict[str, Any]:
        """
        Train all prediction models.
        
        Returns:
            Dictionary with training metrics for all models
        """
        if self.processed_data is None:
            raise ValueError("No processed data available. Call process_data() first.")
            
        try:
            logger.info("Training prediction models...")
            
            # Train individual models
            linear_metrics = self.linear_model.train(self.processed_data)
            
            # Try to train LSTM model if TensorFlow is available
            lstm_metrics = {}
            try:
                from .models import TENSORFLOW_AVAILABLE
                if TENSORFLOW_AVAILABLE:
                    lstm_metrics = self.lstm_model.train(self.processed_data)
                else:
                    logger.warning("TensorFlow not available. Skipping LSTM training.")
                    lstm_metrics = {'error': 'TensorFlow not available'}
            except Exception as e:
                logger.warning(f"LSTM training failed: {str(e)}")
                lstm_metrics = {'error': str(e)}
            
            # Train ensemble model
            ensemble_metrics = self.ensemble_model.train(self.processed_data)
            
            training_results = {
                'linear_regression': linear_metrics,
                'lstm': lstm_metrics,
                'ensemble': ensemble_metrics,
                'training_data_points': len(self.processed_data)
            }
            
            logger.info("All models trained successfully")
            return training_results
            
        except Exception as e:
            logger.error(f"Error training models: {str(e)}")
            raise
            
    def generate_technical_signals(self) -> Dict[str, str]:
        """
        Generate trading signals based on technical indicators.
        
        Returns:
            Dictionary with technical analysis signals
        """
        if self.processed_data is None:
            raise ValueError("No processed data available.")
            
        try:
            latest = self.processed_data.iloc[-1]
            signals = {}
            
            # RSI signals
            if latest['RSI'] > 70:
                signals['RSI'] = 'OVERBOUGHT'
            elif latest['RSI'] < 30:
                signals['RSI'] = 'OVERSOLD'
            else:
                signals['RSI'] = 'NEUTRAL'
                
            # MACD signals
            if latest['MACD'] > latest['MACD_Signal']:
                signals['MACD'] = 'BULLISH'
            else:
                signals['MACD'] = 'BEARISH'
                
            # Bollinger Bands signals
            if latest['close'] > latest['BB_Upper']:
                signals['Bollinger'] = 'OVERBOUGHT'
            elif latest['close'] < latest['BB_Lower']:
                signals['Bollinger'] = 'OVERSOLD'
            else:
                signals['Bollinger'] = 'NEUTRAL'
                
            # Moving Average signals
            if latest['close'] > latest['SMA_20']:
                signals['SMA_Trend'] = 'BULLISH'
            else:
                signals['SMA_Trend'] = 'BEARISH'
                
            # Stochastic signals
            if latest['Stoch_K'] > 80:
                signals['Stochastic'] = 'OVERBOUGHT'
            elif latest['Stoch_K'] < 20:
                signals['Stochastic'] = 'OVERSOLD'
            else:
                signals['Stochastic'] = 'NEUTRAL'
                
            return signals
            
        except Exception as e:
            logger.error(f"Error generating technical signals: {str(e)}")
            return {}
            
    def predict_price(self) -> Dict[str, Any]:
        """
        Generate comprehensive price predictions.
        
        Returns:
            Dictionary with predictions from all models and technical analysis
        """
        if self.processed_data is None:
            raise ValueError("No processed data available.")
            
        try:
            current_price = self.processed_data['close'].iloc[-1]
            
            # Get model predictions
            predictions = {}
            
            # Linear regression prediction
            if self.linear_model.is_trained:
                lr_pred = self.linear_model.predict_next_price(self.processed_data)
                predictions['linear_regression'] = {
                    'price': lr_pred,
                    'change_percent': ((lr_pred - current_price) / current_price) * 100
                }
            
            # LSTM prediction (only if available and trained)
            try:
                from .models import TENSORFLOW_AVAILABLE
                if TENSORFLOW_AVAILABLE and self.lstm_model.is_trained:
                    lstm_pred = self.lstm_model.predict_next_price(self.processed_data)
                    predictions['lstm'] = {
                        'price': lstm_pred,
                        'change_percent': ((lstm_pred - current_price) / current_price) * 100
                    }
            except Exception as e:
                logger.warning(f"LSTM prediction failed: {str(e)}")
            
            # Ensemble prediction
            if self.ensemble_model.is_trained:
                try:
                    ensemble_preds = self.ensemble_model.predict_next_price(self.processed_data)
                    ensemble_price = ensemble_preds['ensemble']
                    predictions['ensemble'] = {
                        'price': ensemble_price,
                        'change_percent': ((ensemble_price - current_price) / current_price) * 100,
                        'individual_predictions': ensemble_preds
                    }
                except Exception as e:
                    logger.warning(f"Ensemble prediction failed: {str(e)}")
            
            # Technical analysis signals
            technical_signals = self.generate_technical_signals()
            
            # Overall prediction
            prediction_result = {
                'symbol': self.symbol,
                'current_price': current_price,
                'timestamp': datetime.now().isoformat(),
                'predictions': predictions,
                'technical_signals': technical_signals,
                'data_points_used': len(self.processed_data)
            }
            
            self.latest_predictions = prediction_result
            logger.info(f"Price prediction generated for {self.symbol}")
            
            return prediction_result
            
        except Exception as e:
            logger.error(f"Error generating price prediction: {str(e)}")
            raise
            
    def get_trading_recommendation(self) -> Dict[str, Any]:
        """
        Generate trading recommendation based on all available signals.
        
        Returns:
            Dictionary with trading recommendation
        """
        if self.latest_predictions is None:
            self.predict_price()
            
        try:
            predictions = self.latest_predictions['predictions']
            technical_signals = self.latest_predictions['technical_signals']
            current_price = self.latest_predictions['current_price']
            
            # Count bullish/bearish signals
            bullish_count = 0
            bearish_count = 0
            
            # Technical analysis voting
            for signal, value in technical_signals.items():
                if value in ['BULLISH', 'OVERSOLD']:
                    bullish_count += 1
                elif value in ['BEARISH', 'OVERBOUGHT']:
                    bearish_count += 1
                    
            # Model predictions voting
            if 'ensemble' in predictions:
                change_percent = predictions['ensemble']['change_percent']
                if change_percent > 1:  # More than 1% increase predicted
                    bullish_count += 2  # Give more weight to ML prediction
                elif change_percent < -1:  # More than 1% decrease predicted
                    bearish_count += 2
                    
            # Generate recommendation
            if bullish_count > bearish_count:
                recommendation = 'BUY'
                confidence = min(bullish_count / (bullish_count + bearish_count + 1), 0.95)
            elif bearish_count > bullish_count:
                recommendation = 'SELL'
                confidence = min(bearish_count / (bullish_count + bearish_count + 1), 0.95)
            else:
                recommendation = 'HOLD'
                confidence = 0.5
                
            # Calculate suggested price levels
            stop_loss = current_price * (1 - Config.STOP_LOSS_PERCENTAGE)
            take_profit = current_price * (1 + Config.TAKE_PROFIT_PERCENTAGE)
            
            if recommendation == 'SELL':
                stop_loss = current_price * (1 + Config.STOP_LOSS_PERCENTAGE)
                take_profit = current_price * (1 - Config.TAKE_PROFIT_PERCENTAGE)
                
            trading_recommendation = {
                'recommendation': recommendation,
                'confidence': confidence,
                'entry_price': current_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'risk_reward_ratio': Config.TAKE_PROFIT_PERCENTAGE / Config.STOP_LOSS_PERCENTAGE,
                'bullish_signals': bullish_count,
                'bearish_signals': bearish_count,
                'reasoning': {
                    'technical_analysis': technical_signals,
                    'price_predictions': predictions
                }
            }
            
            return trading_recommendation
            
        except Exception as e:
            logger.error(f"Error generating trading recommendation: {str(e)}")
            raise
            
    def run_full_analysis(self) -> Dict[str, Any]:
        """
        Run complete analysis pipeline.
        
        Returns:
            Dictionary with complete analysis results
        """
        try:
            logger.info(f"Starting full analysis for {self.symbol}")
            
            # Step 1: Fetch data
            self.fetch_data()
            
            # Step 2: Process data
            self.process_data()
            
            # Step 3: Train models
            training_metrics = self.train_models()
            
            # Step 4: Generate predictions
            predictions = self.predict_price()
            
            # Step 5: Generate trading recommendation
            recommendation = self.get_trading_recommendation()
            
            # Compile complete analysis
            complete_analysis = {
                'analysis_timestamp': datetime.now().isoformat(),
                'symbol': self.symbol,
                'data_source': self.data_source,
                'training_metrics': training_metrics,
                'predictions': predictions,
                'trading_recommendation': recommendation,
                'market_data_summary': {
                    'data_points': len(self.processed_data),
                    'date_range': {
                        'start': self.processed_data.index[0].isoformat(),
                        'end': self.processed_data.index[-1].isoformat()
                    },
                    'price_range': {
                        'min': float(self.processed_data['close'].min()),
                        'max': float(self.processed_data['close'].max()),
                        'current': float(self.processed_data['close'].iloc[-1])
                    }
                }
            }
            
            logger.info(f"Full analysis completed for {self.symbol}")
            return complete_analysis
            
        except Exception as e:
            logger.error(f"Error in full analysis: {str(e)}")
            raise
            
    def get_latest_data_summary(self) -> Dict[str, Any]:
        """
        Get summary of latest market data and indicators.
        
        Returns:
            Dictionary with latest market data summary
        """
        if self.processed_data is None:
            raise ValueError("No processed data available.")
            
        latest = self.processed_data.iloc[-1]
        
        return {
            'timestamp': latest.name.isoformat(),
            'ohlcv': {
                'open': float(latest['open']),
                'high': float(latest['high']),
                'low': float(latest['low']),
                'close': float(latest['close']),
                'volume': float(latest['volume'])
            },
            'technical_indicators': {
                'RSI': float(latest['RSI']) if not pd.isna(latest['RSI']) else None,
                'MACD': float(latest['MACD']) if not pd.isna(latest['MACD']) else None,
                'SMA_20': float(latest['SMA_20']) if not pd.isna(latest['SMA_20']) else None,
                'EMA_12': float(latest['EMA_12']) if not pd.isna(latest['EMA_12']) else None,
                'Bollinger_Upper': float(latest['BB_Upper']) if not pd.isna(latest['BB_Upper']) else None,
                'Bollinger_Lower': float(latest['BB_Lower']) if not pd.isna(latest['BB_Lower']) else None,
                'Stochastic_K': float(latest['Stoch_K']) if not pd.isna(latest['Stoch_K']) else None,
                'ATR': float(latest['ATR']) if not pd.isna(latest['ATR']) else None
            }
        }