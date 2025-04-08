# Advanced Indian Market Options Trading Bot

## Overview
This is an advanced options trading bot specifically designed for the Indian market, focusing on Bank Nifty and Nifty indices. The bot implements multiple options trading strategies and aims to achieve a 10% overall profit on the invested capital.

## Features
- Multiple trading strategies:
  - Covered Call
  - Protective Put
  - Iron Condor
  - Straddle
  - Vertical Spreads (Bull Call and Bear Put)
- Real-time position monitoring and management
- Automatic profit taking at 50% target
- Trade history tracking with Excel export
- Portfolio value and P&L tracking
- Market hours compliance (9:15 AM to 3:30 PM IST)

## Strategies
1. **Covered Call**: Buy stock and sell OTM call options
2. **Protective Put**: Buy stock and buy ATM put options for downside protection
3. **Iron Condor**: Sell OTM call and put spreads for premium collection
4. **Straddle**: Buy ATM call and put options for volatility exposure
5. **Vertical Spreads**: Bull call spread for bullish outlook, bear put spread for bearish outlook

## Requirements
- Python 3.8+
- Required packages listed in requirements.txt
- Internet connection for market data access

## Installation
1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage
Run the bot with:
```bash
python indian_market_bot.py
```

## Configuration
- Initial capital: ₹10,00,000
- Position size: 20% of capital per trade
- Target profit: 10% overall return
- Profit taking: 50% per position

## Trade History
- Trade details are saved in Excel files
- Location: ./trading_history/
- Format: trading_history_YYYYMMDD_HHMMSS.xlsx
- Saved every 5 trades and at program end

## Risk Management
- Position size limits
- Strategy-specific stop losses
- Automatic position closure at expiry
- Market hours trading only

## Disclaimer
This bot is for educational and demonstration purposes only. Real trading involves significant risk of loss. Always paper trade first and consult with a financial advisor before using any trading strategy.

## License
MIT License

## Author
[Your Name]

## Version
1.0.0 