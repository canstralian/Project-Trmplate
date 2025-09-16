"""
Machine learning models for cryptocurrency price prediction.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# TensorFlow imports - optional for LSTM model
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    tf = None
    Sequential = None
from typing import Dict, Tuple, Optional, Any
import logging
from .config import Config

# Configure logging
logging.basicConfig(level=Config.LOG_LEVEL, format=Config.LOG_FORMAT)
logger = logging.getLogger(__name__)


class BaseModel:
    """Base class for prediction models."""
    
    def __init__(self):
        self.model = None
        self.scaler = MinMaxScaler()
        self.is_trained = False
        
    def prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """Prepare features from dataframe."""
        raise NotImplementedError
        
    def train(self, df: pd.DataFrame) -> Dict[str, float]:
        """Train the model."""
        raise NotImplementedError
        
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        raise NotImplementedError
        
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance."""
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        
        return {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2': r2
        }


class LinearRegressionModel(BaseModel):
    """Linear Regression model for price prediction."""
    
    def __init__(self):
        super().__init__()
        self.model = LinearRegression()
        
    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features for linear regression.
        
        Args:
            df: DataFrame with technical indicators
            
        Returns:
            Tuple of (features, target)
        """
        # Select relevant columns for features
        feature_columns = [
            'open', 'high', 'low', 'volume',
            'SMA_5', 'SMA_10', 'SMA_20', 'SMA_50',
            'EMA_12', 'EMA_26', 'RSI', 'MACD', 'MACD_Signal',
            'BB_Upper', 'BB_Middle', 'BB_Lower',
            'Stoch_K', 'Stoch_D', 'Williams_R', 'ATR'
        ]
        
        # Filter columns that exist in the dataframe
        available_columns = [col for col in feature_columns if col in df.columns]
        
        if not available_columns:
            raise ValueError("No suitable feature columns found in dataframe")
        
        # Prepare features and target
        features = df[available_columns].dropna()
        target = df['close'].loc[features.index]
        
        return features.values, target.values
        
    def train(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Train the linear regression model.
        
        Args:
            df: DataFrame with features and target
            
        Returns:
            Dictionary with training metrics
        """
        try:
            X, y = self.prepare_features(df)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=1-Config.TRAIN_TEST_SPLIT, random_state=42
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train model
            self.model.fit(X_train_scaled, y_train)
            self.is_trained = True
            
            # Make predictions for evaluation
            y_pred = self.model.predict(X_test_scaled)
            
            # Evaluate
            metrics = self.evaluate(y_test, y_pred)
            
            logger.info(f"Linear Regression trained successfully. R² Score: {metrics['r2']:.4f}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error training Linear Regression model: {str(e)}")
            raise
            
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            df: DataFrame with features
            
        Returns:
            Array of predictions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
            
        X, _ = self.prepare_features(df)
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        
        return predictions
        
    def predict_next_price(self, df: pd.DataFrame) -> float:
        """
        Predict the next price based on the latest data.
        
        Args:
            df: DataFrame with latest data
            
        Returns:
            Predicted next price
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
            
        # Use the last row for prediction
        latest_data = df.tail(1)
        prediction = self.predict(latest_data)
        
        return float(prediction[0])


class LSTMModel(BaseModel):
    """LSTM Neural Network model for price prediction."""
    
    def __init__(self, sequence_length: int = Config.LSTM_SEQUENCE_LENGTH):
        super().__init__()
        self.sequence_length = sequence_length
        self.model = None
        
    def create_sequences(self, data: np.ndarray, sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM training.
        
        Args:
            data: Input data array
            sequence_length: Length of each sequence
            
        Returns:
            Tuple of (X_sequences, y_sequences)
        """
        X, y = [], []
        for i in range(sequence_length, len(data)):
            X.append(data[i-sequence_length:i])
            y.append(data[i])
        return np.array(X), np.array(y)
        
    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features for LSTM model.
        
        Args:
            df: DataFrame with price data
            
        Returns:
            Tuple of (features, target)
        """
        # Use close price for LSTM
        close_prices = df['close'].values.reshape(-1, 1)
        
        # Scale the data
        scaled_data = self.scaler.fit_transform(close_prices)
        
        # Create sequences
        X, y = self.create_sequences(scaled_data, self.sequence_length)
        
        return X, y
        
    def build_model(self, input_shape: Tuple[int, int]):
        """
        Build LSTM model architecture.
        
        Args:
            input_shape: Shape of input data (sequence_length, features)
            
        Returns:
            Compiled LSTM model
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required for LSTM model. Install with: pip install tensorflow")
            
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(50, return_sequences=True),
            Dropout(0.2),
            LSTM(50),
            Dropout(0.2),
            Dense(1)
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
        return model
        
    def train(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Train the LSTM model.
        
        Args:
            df: DataFrame with price data
            
        Returns:
            Dictionary with training metrics
        """
        if not TENSORFLOW_AVAILABLE:
            logger.warning("TensorFlow not available. Skipping LSTM training.")
            return {'error': 'TensorFlow not available'}
            
        try:
            X, y = self.prepare_features(df)
            
            if len(X) < self.sequence_length:
                raise ValueError(f"Not enough data points. Need at least {self.sequence_length}")
            
            # Split data
            split_index = int(len(X) * Config.TRAIN_TEST_SPLIT)
            X_train, X_test = X[:split_index], X[split_index:]
            y_train, y_test = y[:split_index], y[split_index:]
            
            # Build model
            self.model = self.build_model((X_train.shape[1], X_train.shape[2]))
            
            # Train model
            history = self.model.fit(
                X_train, y_train,
                epochs=Config.LSTM_EPOCHS,
                batch_size=Config.LSTM_BATCH_SIZE,
                validation_data=(X_test, y_test),
                verbose=0
            )
            
            self.is_trained = True
            
            # Make predictions for evaluation
            y_pred = self.model.predict(X_test)
            
            # Inverse transform predictions and actual values
            y_test_actual = self.scaler.inverse_transform(y_test)
            y_pred_actual = self.scaler.inverse_transform(y_pred)
            
            # Evaluate
            metrics = self.evaluate(y_test_actual.flatten(), y_pred_actual.flatten())
            metrics['final_loss'] = history.history['loss'][-1]
            metrics['final_val_loss'] = history.history['val_loss'][-1]
            
            logger.info(f"LSTM model trained successfully. R² Score: {metrics['r2']:.4f}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error training LSTM model: {str(e)}")
            raise
            
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the trained LSTM model.
        
        Args:
            df: DataFrame with price data
            
        Returns:
            Array of predictions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
            
        # Use the last sequence for prediction
        close_prices = df['close'].tail(self.sequence_length).values.reshape(-1, 1)
        scaled_data = self.scaler.transform(close_prices)
        
        # Reshape for LSTM
        X = scaled_data.reshape(1, self.sequence_length, 1)
        
        # Make prediction
        prediction_scaled = self.model.predict(X)
        prediction = self.scaler.inverse_transform(prediction_scaled)
        
        return prediction.flatten()
        
    def predict_next_price(self, df: pd.DataFrame) -> float:
        """
        Predict the next price based on the latest sequence.
        
        Args:
            df: DataFrame with latest data
            
        Returns:
            Predicted next price
        """
        if not TENSORFLOW_AVAILABLE:
            raise ValueError("TensorFlow is required for LSTM predictions")
            
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
            
        if len(df) < self.sequence_length:
            raise ValueError(f"Need at least {self.sequence_length} data points for prediction")
            
        prediction = self.predict(df)
        return float(prediction[0])


class EnsembleModel:
    """Ensemble model combining multiple prediction models."""
    
    def __init__(self):
        self.linear_model = LinearRegressionModel()
        self.lstm_model = LSTMModel()
        self.weights = {'linear': 0.3, 'lstm': 0.7}  # Can be optimized
        self.is_trained = False
        
    def train(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Train all models in the ensemble.
        
        Args:
            df: DataFrame with features and target
            
        Returns:
            Dictionary with training metrics for all models
        """
        try:
            logger.info("Training ensemble models...")
            
            # Train linear model
            linear_metrics = self.linear_model.train(df)
            
            # Train LSTM model only if TensorFlow is available
            lstm_metrics = {}
            if TENSORFLOW_AVAILABLE:
                lstm_metrics = self.lstm_model.train(df)
            else:
                logger.warning("TensorFlow not available. Skipping LSTM training in ensemble.")
                lstm_metrics = {'error': 'TensorFlow not available'}
            
            self.is_trained = True
            
            logger.info("Ensemble models trained successfully")
            
            return {
                'linear_regression': linear_metrics,
                'lstm': lstm_metrics
            }
            
        except Exception as e:
            logger.error(f"Error training ensemble models: {str(e)}")
            raise
            
    def predict_next_price(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Make ensemble prediction for next price.
        
        Args:
            df: DataFrame with latest data
            
        Returns:
            Dictionary with individual and ensemble predictions
        """
        if not self.is_trained:
            raise ValueError("Ensemble must be trained before making predictions")
            
        try:
            predictions = {}
            
            # Get predictions from individual models
            if self.linear_model.is_trained:
                linear_pred = self.linear_model.predict_next_price(df)
                predictions['linear_regression'] = linear_pred
            
            # Only try LSTM if TensorFlow is available and model is trained
            if TENSORFLOW_AVAILABLE and self.lstm_model.is_trained:
                lstm_pred = self.lstm_model.predict_next_price(df)
                predictions['lstm'] = lstm_pred
            
            # Calculate ensemble prediction based on available models
            if 'lstm' in predictions and 'linear_regression' in predictions:
                ensemble_pred = (self.weights['linear'] * predictions['linear_regression'] + 
                               self.weights['lstm'] * predictions['lstm'])
            elif 'linear_regression' in predictions:
                # Only linear model available
                ensemble_pred = predictions['linear_regression']
                logger.warning("LSTM not available, using only Linear Regression for ensemble")
            else:
                raise ValueError("No trained models available for ensemble prediction")
            
            predictions['ensemble'] = ensemble_pred
            return predictions
            
        except Exception as e:
            logger.error(f"Error making ensemble predictions: {str(e)}")
            raise