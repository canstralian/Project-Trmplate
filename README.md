# Crypto Price Prediction Algorithm

A comprehensive cryptocurrency day trading price prediction system that combines technical analysis, machine learning models, and ensemble methods to provide accurate price forecasts and trading recommendations.

## 🚀 Features

- **Multiple Prediction Models**: Linear Regression, LSTM Neural Networks, and Ensemble methods
- **Technical Analysis**: RSI, MACD, Bollinger Bands, Stochastic Oscillator, Williams %R, ATR
- **Real-time Data Fetching**: Support for Binance API, Yahoo Finance, and sample data generation
- **Trading Recommendations**: Automated buy/sell/hold signals with confidence levels
- **Risk Management**: Stop-loss and take-profit calculations
- **Backtesting Framework**: Evaluate strategy performance on historical data
- **CLI Interface**: Easy-to-use command-line interface
- **Comprehensive Testing**: Unit tests for all components

## 📊 Supported Cryptocurrencies

- Bitcoin (BTCUSDT)
- Ethereum (ETHUSDT)
- Cardano (ADAUSDT)
- Polkadot (DOTUSDT)
- Chainlink (LINKUSDT)
- Binance Coin (BNBUSDT)
- Ripple (XRPUSDT)
- Litecoin (LTCUSDT)
- Bitcoin Cash (BCHUSDT)
- Stellar (XLMUSDT)

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/canstralian/Project-Trmplate.git
   cd Project-Trmplate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

   **Optional dependencies for advanced features:**
   ```bash
   # For LSTM neural network models
   pip install tensorflow>=2.8.0
   
   # For real-time data from Yahoo Finance
   pip install yfinance>=0.1.70
   
   # For additional technical indicators and visualization
   pip install seaborn ta plotly
   ```

3. **Optional - Set up API keys (for real data):**
   ```bash
   export BINANCE_API_KEY="your_api_key"
   export BINANCE_SECRET_KEY="your_secret_key"
   ```

## 🎯 Quick Start

### Basic Usage

```bash
# Run full analysis for Bitcoin
python main.py --symbol BTCUSDT --analyze

# Quick prediction for Ethereum
python main.py --symbol ETHUSDT --predict-only

# Use real market data instead of sample data (requires optional dependencies)
python main.py --symbol BTCUSDT --analyze --use-real-data

# Save results to file
python main.py --symbol ADAUSDT --analyze --output results.json

# Run the demo example
python example.py
```

### Python API Usage

```python
from crypto_predictor import CryptoPricePredictor

# Initialize predictor
predictor = CryptoPricePredictor('BTCUSDT', use_sample_data=True)

# Run complete analysis
results = predictor.run_full_analysis()

# Get trading recommendation
recommendation = predictor.get_trading_recommendation()

print(f"Recommendation: {recommendation['recommendation']}")
print(f"Confidence: {recommendation['confidence']:.2%}")
print(f"Entry Price: ${recommendation['entry_price']:,.2f}")
```

## 📈 Algorithm Overview

### 1. Technical Analysis
- **Moving Averages**: SMA (5, 10, 20, 50 periods) and EMA (12, 26 periods)
- **Momentum Indicators**: RSI (14 period), Stochastic Oscillator, Williams %R
- **Trend Indicators**: MACD, Bollinger Bands
- **Volatility Indicators**: Average True Range (ATR)

### 2. Machine Learning Models

#### Linear Regression
- Uses technical indicators as features
- Fast training and prediction
- Provides baseline performance

#### LSTM Neural Network
- Processes sequential price data
- Captures complex patterns and dependencies
- 3-layer LSTM architecture with dropout

#### Ensemble Model
- Combines predictions from multiple models
- Weighted voting system
- Improved accuracy and robustness

### 3. Trading Signal Generation
- **Signal Aggregation**: Combines technical and ML signals
- **Confidence Scoring**: Probabilistic confidence levels
- **Risk Management**: Automatic stop-loss and take-profit levels
- **Multi-timeframe Analysis**: Supports various timeframes (1m to 1d)

## 📊 Model Performance Metrics

The system provides comprehensive evaluation metrics:

- **Mean Absolute Error (MAE)**: Average prediction error
- **Root Mean Square Error (RMSE)**: Penalizes larger errors
- **R² Score**: Coefficient of determination
- **Confidence Intervals**: Prediction uncertainty bounds

## 🔧 Configuration

Edit `crypto_predictor/config.py` to customize:

```python
# Technical indicator settings
SMA_PERIODS = [5, 10, 20, 50]
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26

# Machine learning settings
LSTM_SEQUENCE_LENGTH = 60
LSTM_EPOCHS = 50
TRAIN_TEST_SPLIT = 0.8

# Trading settings
RISK_PERCENTAGE = 0.02  # 2% risk per trade
STOP_LOSS_PERCENTAGE = 0.03  # 3% stop loss
TAKE_PROFIT_PERCENTAGE = 0.06  # 6% take profit
```

## 📋 Command Line Options

```bash
python main.py [OPTIONS]

Options:
  -s, --symbol SYMBOL     Cryptocurrency symbol (default: BTCUSDT)
  -d, --days DAYS         Days of historical data (default: 30)
  -i, --interval INTERVAL Time interval (default: 1h)
  -a, --analyze          Run full analysis with training
  -p, --predict-only     Generate predictions only
  --list-symbols         List supported symbols
  -o, --output FILE      Save results to JSON file
  --use-real-data        Use real market data
  -v, --verbose          Enable verbose output
```

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_crypto_predictor.py

# Run with coverage
python -m pytest tests/ --cov=crypto_predictor
```

## 📁 Project Structure

```
Project-Trmplate/
├── crypto_predictor/
│   ├── __init__.py
│   ├── config.py          # Configuration settings
│   ├── data_fetcher.py    # Data fetching utilities
│   ├── indicators.py     # Technical indicators
│   ├── models.py         # ML models
│   └── predictor.py      # Main predictor class
├── tests/
│   └── test_crypto_predictor.py
├── main.py               # CLI interface
├── requirements.txt      # Dependencies
├── README.md            # Documentation
└── LICENSE              # Apache 2.0 License
```

## 🔮 Future Enhancements

- [ ] Additional ML models (Random Forest, XGBoost, Transformer)
- [ ] Sentiment analysis integration
- [ ] Real-time streaming predictions
- [ ] Web dashboard interface
- [ ] Portfolio optimization
- [ ] Advanced backtesting with transaction costs
- [ ] Integration with trading platforms
- [ ] Multi-asset correlation analysis

## ⚠️ Disclaimer

**This software is for educational and research purposes only. Cryptocurrency trading involves substantial risk and may result in significant financial losses. The predictions and recommendations provided by this system should not be considered as financial advice. Always conduct your own research and consult with qualified financial advisors before making investment decisions.**

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📞 Support

If you encounter any issues or have questions:

1. Check the existing issues on GitHub
2. Create a new issue with detailed information
3. Join our community discussions

## 🌟 Acknowledgments

- Technical analysis formulas from [TA-Lib](https://ta-lib.org/)
- Market data provided by [Binance API](https://binance-docs.github.io/apidocs/)
- Machine learning frameworks: [scikit-learn](https://scikit-learn.org/) and [TensorFlow](https://tensorflow.org/)

---

**Happy Trading! 🚀📈**
