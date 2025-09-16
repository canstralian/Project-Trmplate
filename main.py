#!/usr/bin/env python3
"""
Command-line interface for the Crypto Price Prediction system.
"""

import argparse
import json
import sys
from datetime import datetime
from crypto_predictor import CryptoPricePredictor
from crypto_predictor.config import Config


def format_price(price: float) -> str:
    """Format price with appropriate decimal places."""
    return f"${price:,.2f}"


def format_percentage(percentage: float) -> str:
    """Format percentage with sign and color coding."""
    sign = "+" if percentage >= 0 else ""
    return f"{sign}{percentage:.2f}%"


def print_header(title: str):
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)


def print_predictions(predictions: dict):
    """Print prediction results in a formatted way."""
    print_header("PRICE PREDICTIONS")
    
    current_price = predictions.get('current_price', 0)
    print(f"Current Price: {format_price(current_price)}")
    print(f"Timestamp: {predictions.get('timestamp', 'N/A')}")
    print()
    
    model_predictions = predictions.get('predictions', {})
    
    for model_name, pred_data in model_predictions.items():
        if isinstance(pred_data, dict):
            predicted_price = pred_data.get('price', 0)
            change_percent = pred_data.get('change_percent', 0)
            
            print(f"{model_name.upper()}:")
            print(f"  Predicted Price: {format_price(predicted_price)}")
            print(f"  Expected Change: {format_percentage(change_percent)}")
            print()


def print_technical_signals(signals: dict):
    """Print technical analysis signals."""
    print_header("TECHNICAL ANALYSIS SIGNALS")
    
    for indicator, signal in signals.items():
        print(f"{indicator:15}: {signal}")


def print_trading_recommendation(recommendation: dict):
    """Print trading recommendation."""
    print_header("TRADING RECOMMENDATION")
    
    action = recommendation.get('recommendation', 'HOLD')
    confidence = recommendation.get('confidence', 0) * 100
    entry_price = recommendation.get('entry_price', 0)
    stop_loss = recommendation.get('stop_loss', 0)
    take_profit = recommendation.get('take_profit', 0)
    
    print(f"Recommendation: {action}")
    print(f"Confidence: {confidence:.1f}%")
    print(f"Entry Price: {format_price(entry_price)}")
    print(f"Stop Loss: {format_price(stop_loss)}")
    print(f"Take Profit: {format_price(take_profit)}")
    print(f"Risk/Reward Ratio: {recommendation.get('risk_reward_ratio', 0):.2f}")
    
    print(f"\nSignal Count:")
    print(f"  Bullish: {recommendation.get('bullish_signals', 0)}")
    print(f"  Bearish: {recommendation.get('bearish_signals', 0)}")


def print_training_metrics(metrics: dict):
    """Print model training metrics."""
    print_header("MODEL TRAINING METRICS")
    
    for model_name, model_metrics in metrics.items():
        if model_name == 'training_data_points':
            print(f"Training Data Points: {model_metrics}")
            continue
            
        print(f"\n{model_name.upper()}:")
        if isinstance(model_metrics, dict):
            for metric_name, metric_value in model_metrics.items():
                if isinstance(metric_value, (int, float)):
                    print(f"  {metric_name}: {metric_value:.4f}")


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Cryptocurrency Price Prediction System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --symbol BTCUSDT --analyze
  python main.py --symbol ETHUSDT --predict-only
  python main.py --list-symbols
  python main.py --symbol ADAUSDT --days 60 --output results.json
        """
    )
    
    parser.add_argument(
        '--symbol', '-s',
        type=str,
        default=Config.DEFAULT_SYMBOL,
        help=f'Cryptocurrency symbol (default: {Config.DEFAULT_SYMBOL})'
    )
    
    parser.add_argument(
        '--days', '-d',
        type=int,
        default=Config.DEFAULT_LOOKBACK_DAYS,
        help=f'Number of days of historical data (default: {Config.DEFAULT_LOOKBACK_DAYS})'
    )
    
    parser.add_argument(
        '--interval', '-i',
        type=str,
        default=Config.DEFAULT_INTERVAL,
        help=f'Data interval (default: {Config.DEFAULT_INTERVAL})'
    )
    
    parser.add_argument(
        '--analyze', '-a',
        action='store_true',
        help='Run full analysis including training and predictions'
    )
    
    parser.add_argument(
        '--predict-only', '-p',
        action='store_true',
        help='Only generate predictions (requires pre-trained models)'
    )
    
    parser.add_argument(
        '--list-symbols',
        action='store_true',
        help='List all supported cryptocurrency symbols'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Save results to JSON file'
    )
    
    parser.add_argument(
        '--use-real-data',
        action='store_true',
        help='Use real market data instead of sample data'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    args = parser.parse_args()
    
    # Handle list symbols command
    if args.list_symbols:
        print("Supported Cryptocurrency Symbols:")
        for symbol in Config.get_supported_symbols():
            print(f"  {symbol}")
        return
    
    # Validate symbol
    if not Config.validate_symbol(args.symbol):
        print(f"Warning: {args.symbol} is not in the supported symbols list.")
        print("Use --list-symbols to see supported symbols.")
    
    # Initialize predictor
    try:
        print(f"Initializing Crypto Price Predictor for {args.symbol}...")
        predictor = CryptoPricePredictor(
            symbol=args.symbol,
            use_sample_data=not args.use_real_data
        )
        
        if args.analyze:
            # Run full analysis
            print("Running full analysis (this may take a few minutes)...")
            results = predictor.run_full_analysis()
            
            # Print results
            print_predictions(results['predictions'])
            print_technical_signals(results['predictions']['technical_signals'])
            print_trading_recommendation(results['trading_recommendation'])
            
            if args.verbose:
                print_training_metrics(results['training_metrics'])
            
            # Save to file if requested
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(results, f, indent=2, default=str)
                print(f"\nResults saved to {args.output}")
                
        elif args.predict_only:
            # Only run prediction (would need pre-trained models in real scenario)
            print("For prediction-only mode, models need to be pre-trained.")
            print("Running quick analysis instead...")
            
            # Fetch and process data
            predictor.fetch_data(args.days, args.interval)
            predictor.process_data()
            
            # Generate technical signals only
            signals = predictor.generate_technical_signals()
            print_technical_signals(signals)
            
        else:
            print("Please specify --analyze or --predict-only")
            print("Use --help for more information")
            
    except Exception as e:
        print(f"Error: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()