#!/usr/bin/env python3
"""
Example script demonstrating the Crypto Price Prediction system.
"""

from crypto_predictor import CryptoPricePredictor

def main():
    """
    Example usage of the crypto price prediction system.
    """
    print("=" * 60)
    print("Crypto Price Prediction System - Demo")
    print("=" * 60)
    
    # Initialize predictor for Bitcoin
    predictor = CryptoPricePredictor('BTCUSDT', use_sample_data=True)
    
    print(f"\n1. Fetching data for {predictor.symbol}...")
    predictor.fetch_data(days=15)  # Get 15 days of data
    
    print("2. Processing data and calculating technical indicators...")
    predictor.process_data()
    
    print("3. Training machine learning models...")
    training_metrics = predictor.train_models()
    
    print("4. Generating price predictions...")
    predictions = predictor.predict_price()
    
    print("5. Getting trading recommendation...")
    recommendation = predictor.get_trading_recommendation()
    
    # Display results
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    
    current_price = predictions['current_price']
    print(f"Symbol: {predictions['symbol']}")
    print(f"Current Price: ${current_price:,.2f}")
    
    # Show model predictions
    if 'linear_regression' in predictions['predictions']:
        lr_pred = predictions['predictions']['linear_regression']
        print(f"Linear Regression Prediction: ${lr_pred['price']:,.2f} ({lr_pred['change_percent']:+.2f}%)")
    
    if 'ensemble' in predictions['predictions']:
        ensemble_pred = predictions['predictions']['ensemble']
        print(f"Ensemble Prediction: ${ensemble_pred['price']:,.2f} ({ensemble_pred['change_percent']:+.2f}%)")
    
    # Show technical signals
    print("\nTechnical Analysis Signals:")
    for indicator, signal in predictions['technical_signals'].items():
        print(f"  {indicator}: {signal}")
    
    # Show trading recommendation
    print(f"\nTrading Recommendation: {recommendation['recommendation']}")
    print(f"Confidence: {recommendation['confidence']:.1%}")
    print(f"Entry Price: ${recommendation['entry_price']:,.2f}")
    print(f"Stop Loss: ${recommendation['stop_loss']:,.2f}")
    print(f"Take Profit: ${recommendation['take_profit']:,.2f}")
    
    # Show model performance
    print(f"\nModel Performance (R² Score):")
    if 'linear_regression' in training_metrics:
        lr_r2 = training_metrics['linear_regression'].get('r2', 'N/A')
        print(f"  Linear Regression: {lr_r2:.4f}" if isinstance(lr_r2, float) else f"  Linear Regression: {lr_r2}")
    
    print(f"\nData Points Used: {training_metrics['training_data_points']}")
    
    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()