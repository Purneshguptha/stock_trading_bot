import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import os
import sys

class AdvancedOptionsBot:
    def __init__(self):
        print("Initializing Advanced Options Trading Bot...")
        self.portfolio = {
            'cash': 1000000,  # Starting with ₹10,00,000
            'positions': {},
            'initial_capital': 1000000  # Track initial capital
        }
        self.symbols = {
            'BANKNIFTY': '^NSEBANK',
            'NIFTY': '^NSEI'
        }
        self.position_size = 0.2  # Use 20% of capital per trade
        self.strategies = {
            'COVERED_CALL': self.covered_call,
            'PROTECTIVE_PUT': self.protective_put,
            'IRON_CONDOR': self.iron_condor,
            'STRADDLE': self.straddle,
            'VERTICAL_SPREAD': self.vertical_spread
        }
        self.current_strategy = None
        self.trade_history = []
        self.target_profit_percentage = 10  # Target 10% overall profit
        self.trades_completed = 0
        print("Bot initialized successfully!")
    
    def get_options_chain(self, symbol, strike_range=5):
        """Get options chain data with multiple strikes"""
        try:
            print(f"\nFetching options chain for {symbol}...")
            stock = yf.Ticker(self.symbols[symbol])
            current_price = stock.history(period='1d')['Close'].iloc[-1]
            print(f"Current price: ₹{current_price:.2f}")
            
            # Calculate ATM strike price
            strike_interval = 100 if symbol == 'BANKNIFTY' else 50
            atm_strike = round(current_price / strike_interval) * strike_interval
            
            # Get options data for current expiry
            expiry = self.get_expiry_dates()[0]
            options_data = {
                'current_price': current_price,
                'expiry': expiry,
                'strikes': {}
            }
            
            # Get strikes around ATM
            for i in range(-strike_range, strike_range + 1):
                strike = atm_strike + (i * strike_interval)
                options_data['strikes'][strike] = {
                    'CALL': {
                        'premium': self.get_option_premium(symbol, expiry, strike, 'CALL'),
                        'delta': self.calculate_delta(symbol, expiry, strike, 'CALL')
                    },
                    'PUT': {
                        'premium': self.get_option_premium(symbol, expiry, strike, 'PUT'),
                        'delta': self.calculate_delta(symbol, expiry, strike, 'PUT')
                    }
                }
            
            return options_data
        except Exception as e:
            print(f"Error in get_options_chain: {str(e)}")
            return None
    
    def get_expiry_dates(self):
        """Get upcoming expiry dates (simplified for demo)"""
        today = datetime.now()
        # Get next Thursday (standard expiry day)
        days_until_thursday = (3 - today.weekday()) % 7
        next_thursday = today + timedelta(days=days_until_thursday)
        return [next_thursday.strftime('%Y-%m-%d')]
    
    def get_option_premium(self, symbol, expiry, strike, option_type):
        """Get option premium (simplified for demo)"""
        try:
            stock = yf.Ticker(self.symbols[symbol])
            current_price = stock.history(period='1d')['Close'].iloc[-1]
            
            # Simplified premium calculation based on Black-Scholes model
            time_to_expiry = (datetime.strptime(expiry, '%Y-%m-%d') - datetime.now()).days / 365
            volatility = 0.2  # 20% volatility (simplified)
            
            if option_type == 'CALL':
                return max(0, current_price - strike) + volatility * current_price * np.sqrt(time_to_expiry)
            else:  # PUT
                return max(0, strike - current_price) + volatility * current_price * np.sqrt(time_to_expiry)
        except:
            return None
    
    def calculate_delta(self, symbol, expiry, strike, option_type):
        """Calculate option delta (simplified)"""
        try:
            stock = yf.Ticker(self.symbols[symbol])
            current_price = stock.history(period='1d')['Close'].iloc[-1]
            
            # Simplified delta calculation
            if option_type == 'CALL':
                return min(1, max(0, (current_price - strike) / (strike * 0.1)))
            else:  # PUT
                return max(-1, min(0, (strike - current_price) / (strike * 0.1)))
        except:
            return None
    
    def calculate_indicators(self, data):
        """Calculate technical indicators for options trading"""
        # Calculate moving averages
        data['SMA20'] = data['Close'].rolling(window=20).mean()
        data['SMA50'] = data['Close'].rolling(window=50).mean()
        
        # Calculate volatility
        data['Returns'] = data['Close'].pct_change()
        data['Volatility'] = data['Returns'].rolling(window=20).std() * np.sqrt(252)
        
        # Calculate RSI
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
        
        return data
    
    def generate_signals(self, data):
        """Generate options trading signals"""
        signals = pd.DataFrame(index=data.index)
        signals['Signal'] = 0  # 0: Hold, 1: Buy Call, -1: Buy Put
        
        # Buy Call conditions
        call_conditions = (
            (data['SMA20'] > data['SMA50']) &  # Uptrend
            (data['RSI'] < 70) &  # Not overbought
            (data['Volatility'] > 0.15)  # High volatility
        )
        
        # Buy Put conditions
        put_conditions = (
            (data['SMA20'] < data['SMA50']) &  # Downtrend
            (data['RSI'] > 30) &  # Not oversold
            (data['Volatility'] > 0.15)  # High volatility
        )
        
        signals.loc[call_conditions, 'Signal'] = 1
        signals.loc[put_conditions, 'Signal'] = -1
        
        return signals
    
    def save_trade_history(self):
        """Save trade history to Excel file"""
        if not self.trade_history:
            return
            
        df = pd.DataFrame(self.trade_history)
        filename = f"trading_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        
        # Create directory if it doesn't exist
        if not os.path.exists('trading_history'):
            os.makedirs('trading_history')
            
        filepath = os.path.join('trading_history', filename)
        df.to_excel(filepath, index=False)
        print(f"\nTrade history saved to {filepath}")
    
    def add_to_trade_history(self, trade_info):
        """Add trade information to history"""
        self.trade_history.append(trade_info)
        self.trades_completed += 1
        
        # Save to Excel after every 5 trades
        if self.trades_completed % 5 == 0:
            self.save_trade_history()
    
    def execute_trade(self, symbol, action, options_data):
        """Execute options trades"""
        if not options_data:
            return
            
        expiry = list(options_data.keys())[0]  # Use nearest expiry
        option_data = options_data[expiry]
        
        # Check existing positions for profit taking
        for pos_symbol, position in list(self.portfolio['positions'].items()):
            if pos_symbol.startswith(symbol):
                stock = yf.Ticker(self.symbols[symbol])
                current_price = stock.history(period='1d')['Close'].iloc[-1]
                
                if position['type'] == 'CALL':
                    profit = max(0, current_price - position['strike']) - position['premium']
                else:  # PUT
                    profit = max(0, position['strike'] - current_price) - position['premium']
                
                total_profit = profit * position['quantity']
                initial_investment = position['quantity'] * position['premium']
                
                # Calculate profit percentage
                profit_percentage = (total_profit / initial_investment) * 100
                
                # Take profit at 50%
                if profit_percentage >= 50:
                    print(f"\nTaking profit on {pos_symbol} at {profit_percentage:.2f}%")
                    self.portfolio['cash'] += (initial_investment + total_profit)
                    
                    # Add to trade history
                    trade_info = {
                        'Symbol': symbol,
                        'Type': position['type'],
                        'Strike': position['strike'],
                        'Quantity': position['quantity'],
                        'Premium': position['premium'],
                        'Entry_Time': position['entry_time'],
                        'Exit_Time': datetime.now(),
                        'Profit': total_profit,
                        'Profit_Percentage': profit_percentage,
                        'Status': 'Profit'
                    }
                    self.add_to_trade_history(trade_info)
                    
                    del self.portfolio['positions'][pos_symbol]
                    print(f"Profit taken: ₹{total_profit:.2f}")
                    continue
        
        # Execute new trades if no position exists
        if action == 1:  # Buy Call
            if f"{symbol}_CALL" not in self.portfolio['positions']:
                trade_amount = self.portfolio['cash'] * self.position_size
                premium = option_data['CALL']['premium']
                if premium and premium > 0:
                    quantity = int(trade_amount / premium)
                    if quantity > 0:
                        cost = quantity * premium
                        if cost <= self.portfolio['cash']:
                            self.portfolio['positions'][f"{symbol}_CALL"] = {
                                'quantity': quantity,
                                'strike': option_data['CALL']['strike'],
                                'premium': premium,
                                'expiry': expiry,
                                'type': 'CALL',
                                'entry_time': datetime.now()
                            }
                            self.portfolio['cash'] -= cost
                            
                            # Add to trade history
                            trade_info = {
                                'Symbol': symbol,
                                'Type': 'CALL',
                                'Strike': option_data['CALL']['strike'],
                                'Quantity': quantity,
                                'Premium': premium,
                                'Entry_Time': datetime.now(),
                                'Exit_Time': None,
                                'Profit': None,
                                'Profit_Percentage': None,
                                'Status': 'Open'
                            }
                            self.add_to_trade_history(trade_info)
                            
                            print(f"\nBOUGHT: {quantity} {symbol} CALL options at strike {option_data['CALL']['strike']}")
                            print(f"Premium: ₹{premium:.2f}")
                            print(f"Trade Value: ₹{cost:.2f}")
                            
        elif action == -1:  # Buy Put
            if f"{symbol}_PUT" not in self.portfolio['positions']:
                trade_amount = self.portfolio['cash'] * self.position_size
                premium = option_data['PUT']['premium']
                if premium and premium > 0:
                    quantity = int(trade_amount / premium)
                    if quantity > 0:
                        cost = quantity * premium
                        if cost <= self.portfolio['cash']:
                            self.portfolio['positions'][f"{symbol}_PUT"] = {
                                'quantity': quantity,
                                'strike': option_data['PUT']['strike'],
                                'premium': premium,
                                'expiry': expiry,
                                'type': 'PUT',
                                'entry_time': datetime.now()
                            }
                            self.portfolio['cash'] -= cost
                            
                            # Add to trade history
                            trade_info = {
                                'Symbol': symbol,
                                'Type': 'PUT',
                                'Strike': option_data['PUT']['strike'],
                                'Quantity': quantity,
                                'Premium': premium,
                                'Entry_Time': datetime.now(),
                                'Exit_Time': None,
                                'Profit': None,
                                'Profit_Percentage': None,
                                'Status': 'Open'
                            }
                            self.add_to_trade_history(trade_info)
                            
                            print(f"\nBOUGHT: {quantity} {symbol} PUT options at strike {option_data['PUT']['strike']}")
                            print(f"Premium: ₹{premium:.2f}")
                            print(f"Trade Value: ₹{cost:.2f}")
    
    def check_expiry(self):
        """Check and close expired positions"""
        today = datetime.now().strftime('%Y-%m-%d')
        positions_to_close = []
        
        for symbol, position in self.portfolio['positions'].items():
            if position['expiry'] <= today:
                positions_to_close.append(symbol)
        
        for symbol in positions_to_close:
            position = self.portfolio['positions'][symbol]
            # Simplified P&L calculation
            stock_symbol = symbol.split('_')[0]
            stock = yf.Ticker(self.symbols[stock_symbol])
            current_price = stock.history(period='1d')['Close'].iloc[-1]
            
            if position['type'] == 'CALL':
                profit = max(0, current_price - position['strike']) - position['premium']
            else:  # PUT
                profit = max(0, position['strike'] - current_price) - position['premium']
            
            total_profit = profit * position['quantity']
            self.portfolio['cash'] += total_profit
            del self.portfolio['positions'][symbol]
            
            print(f"\nPosition expired: {symbol}")
            print(f"Profit/Loss: ₹{total_profit:.2f}")
    
    def display_portfolio(self):
        """Display current portfolio status"""
        print("\n" + "="*80)
        print("OPTIONS PORTFOLIO SUMMARY".center(80))
        print("="*80)
        print(f"Cash: ₹{self.portfolio['cash']:,.2f}")
        
        if self.portfolio['positions']:
            total_value = self.portfolio['cash']
            print("\nCurrent Positions:")
            print("-"*80)
            for symbol, position in self.portfolio['positions'].items():
                stock_symbol = symbol.split('_')[0]
                stock = yf.Ticker(self.symbols[stock_symbol])
                current_price = stock.history(period='1d')['Close'].iloc[-1]
                
                # Handle different position types
                if position['type'] in ['CALL', 'PUT']:
                    if position['type'] == 'CALL':
                        profit = max(0, current_price - position['strike']) - position['premium']
                    else:  # PUT
                        profit = max(0, position['strike'] - current_price) - position['premium']
                    
                    position_value = profit * position['quantity']
                    total_value += position_value
                    initial_investment = position['quantity'] * position['premium']
                    profit_percentage = (position_value / initial_investment) * 100
                    
                    print(f"{symbol}:")
                    print(f"  Type: {position['type']}")
                    print(f"  Quantity: {position['quantity']}")
                    print(f"  Strike: ₹{position['strike']:.2f}")
                    print(f"  Premium: ₹{position['premium']:.2f}")
                    print(f"  Expiry: {position['expiry']}")
                    print(f"  Current P/L: ₹{position_value:,.2f}")
                    print(f"  Profit %: {profit_percentage:.2f}%")
                    print(f"  Entry Time: {position['entry_time'].strftime('%Y-%m-%d %H:%M:%S')}")
                
                elif position['type'] == 'COVERED_CALL':
                    stock_value = position['stock_quantity'] * current_price
                    call_value = max(0, current_price - position['call_strike']) * position['stock_quantity']
                    total_value += stock_value - call_value
                    
                    print(f"{symbol}:")
                    print(f"  Type: Covered Call")
                    print(f"  Stock Quantity: {position['stock_quantity']}")
                    print(f"  Stock Price: ₹{position['stock_price']:.2f}")
                    print(f"  Call Strike: ₹{position['call_strike']:.2f}")
                    print(f"  Call Premium: ₹{position['call_premium']:.2f}")
                    print(f"  Current Stock Value: ₹{stock_value:,.2f}")
                    print(f"  Current Call Value: ₹{call_value:,.2f}")
                
                elif position['type'] == 'PROTECTIVE_PUT':
                    stock_value = position['stock_quantity'] * current_price
                    put_value = max(0, position['put_strike'] - current_price) * position['stock_quantity']
                    total_value += stock_value + put_value
                    
                    print(f"{symbol}:")
                    print(f"  Type: Protective Put")
                    print(f"  Stock Quantity: {position['stock_quantity']}")
                    print(f"  Stock Price: ₹{position['stock_price']:.2f}")
                    print(f"  Put Strike: ₹{position['put_strike']:.2f}")
                    print(f"  Put Premium: ₹{position['put_premium']:.2f}")
                    print(f"  Current Stock Value: ₹{stock_value:,.2f}")
                    print(f"  Current Put Value: ₹{put_value:,.2f}")
                
                elif position['type'] == 'IRON_CONDOR':
                    short_call_value = max(0, current_price - position['short_call_strike']) * position['quantity']
                    short_put_value = max(0, position['short_put_strike'] - current_price) * position['quantity']
                    long_call_value = max(0, current_price - position['long_call_strike']) * position['quantity']
                    long_put_value = max(0, position['long_put_strike'] - current_price) * position['quantity']
                    
                    net_value = (position['short_call_premium'] + position['short_put_premium'] - 
                               position['long_call_premium'] - position['long_put_premium']) * position['quantity']
                    total_value += net_value
                    
                    print(f"{symbol}:")
                    print(f"  Type: Iron Condor")
                    print(f"  Quantity: {position['quantity']}")
                    print(f"  Short Call Strike: ₹{position['short_call_strike']:.2f}")
                    print(f"  Short Put Strike: ₹{position['short_put_strike']:.2f}")
                    print(f"  Long Call Strike: ₹{position['long_call_strike']:.2f}")
                    print(f"  Long Put Strike: ₹{position['long_put_strike']:.2f}")
                    print(f"  Net Value: ₹{net_value:,.2f}")
                
                elif position['type'] == 'STRADDLE':
                    call_value = max(0, current_price - position['strike']) * position['quantity']
                    put_value = max(0, position['strike'] - current_price) * position['quantity']
                    net_value = (call_value + put_value) - (position['call_premium'] + position['put_premium']) * position['quantity']
                    total_value += net_value
                    
                    print(f"{symbol}:")
                    print(f"  Type: Straddle")
                    print(f"  Quantity: {position['quantity']}")
                    print(f"  Strike: ₹{position['strike']:.2f}")
                    print(f"  Call Premium: ₹{position['call_premium']:.2f}")
                    print(f"  Put Premium: ₹{position['put_premium']:.2f}")
                    print(f"  Net Value: ₹{net_value:,.2f}")
                
                elif position['type'] in ['BULL_SPREAD', 'BEAR_SPREAD']:
                    if position['type'] == 'BULL_SPREAD':
                        long_value = max(0, current_price - position['long_strike']) * position['quantity']
                        short_value = max(0, current_price - position['short_strike']) * position['quantity']
                    else:  # BEAR_SPREAD
                        long_value = max(0, position['long_strike'] - current_price) * position['quantity']
                        short_value = max(0, position['short_strike'] - current_price) * position['quantity']
                    
                    net_value = (long_value - short_value) - (position['long_premium'] - position['short_premium']) * position['quantity']
                    total_value += net_value
                    
                    print(f"{symbol}:")
                    print(f"  Type: {position['type']}")
                    print(f"  Quantity: {position['quantity']}")
                    print(f"  Long Strike: ₹{position['long_strike']:.2f}")
                    print(f"  Short Strike: ₹{position['short_strike']:.2f}")
                    print(f"  Net Value: ₹{net_value:,.2f}")
                
                print("-"*80)
            print(f"Total Portfolio Value: ₹{total_value:,.2f}")
        else:
            print("\nNo open positions")
        print("="*80)
    
    def covered_call(self, symbol, options_data):
        """Covered Call Strategy"""
        if not options_data:
            return False
            
        current_price = options_data['current_price']
        expiry = options_data['expiry']
        
        # Find OTM call with good premium
        for strike, data in options_data['strikes'].items():
            if strike > current_price and data['CALL']['premium']:
                premium = data['CALL']['premium']
                quantity = int((self.portfolio['cash'] * self.position_size) / current_price)
                
                if quantity > 0:
                    # Buy stock
                    stock_cost = quantity * current_price
                    # Sell call
                    call_premium = premium * quantity
                    
                    if stock_cost <= self.portfolio['cash']:
                        self.portfolio['positions'][f"{symbol}_COVERED_CALL"] = {
                            'type': 'COVERED_CALL',
                            'stock_quantity': quantity,
                            'stock_price': current_price,
                            'call_strike': strike,
                            'call_premium': premium,
                            'expiry': expiry,
                            'entry_time': datetime.now()
                        }
                        self.portfolio['cash'] -= (stock_cost - call_premium)
                        print(f"\nExecuted Covered Call on {symbol}:")
                        print(f"Bought {quantity} shares at ₹{current_price:.2f}")
                        print(f"Sold {quantity} CALL options at strike ₹{strike:.2f}")
                        print(f"Received premium: ₹{call_premium:.2f}")
                        return True
        return False
    
    def protective_put(self, symbol, options_data):
        """Protective Put Strategy"""
        if not options_data:
            return False
            
        current_price = options_data['current_price']
        expiry = options_data['expiry']
        
        # Find ATM put
        for strike, data in options_data['strikes'].items():
            if abs(strike - current_price) < 100 and data['PUT']['premium']:
                premium = data['PUT']['premium']
                quantity = int((self.portfolio['cash'] * self.position_size) / current_price)
                
                if quantity > 0:
                    # Buy stock
                    stock_cost = quantity * current_price
                    # Buy put
                    put_cost = premium * quantity
                    
                    if (stock_cost + put_cost) <= self.portfolio['cash']:
                        self.portfolio['positions'][f"{symbol}_PROTECTIVE_PUT"] = {
                            'type': 'PROTECTIVE_PUT',
                            'stock_quantity': quantity,
                            'stock_price': current_price,
                            'put_strike': strike,
                            'put_premium': premium,
                            'expiry': expiry,
                            'entry_time': datetime.now()
                        }
                        self.portfolio['cash'] -= (stock_cost + put_cost)
                        print(f"\nExecuted Protective Put on {symbol}:")
                        print(f"Bought {quantity} shares at ₹{current_price:.2f}")
                        print(f"Bought {quantity} PUT options at strike ₹{strike:.2f}")
                        print(f"Paid premium: ₹{put_cost:.2f}")
                        return True
        return False
    
    def iron_condor(self, symbol, options_data):
        """Iron Condor Strategy"""
        if not options_data:
            return False
            
        current_price = options_data['current_price']
        expiry = options_data['expiry']
        
        # Find strikes for iron condor
        strikes = sorted(options_data['strikes'].keys())
        atm_index = next(i for i, strike in enumerate(strikes) if strike >= current_price)
        
        if atm_index >= 2 and atm_index + 2 < len(strikes):
            # Sell OTM call and put
            short_call_strike = strikes[atm_index + 1]
            short_put_strike = strikes[atm_index - 1]
            # Buy further OTM call and put
            long_call_strike = strikes[atm_index + 2]
            long_put_strike = strikes[atm_index - 2]
            
            short_call_premium = options_data['strikes'][short_call_strike]['CALL']['premium']
            short_put_premium = options_data['strikes'][short_put_strike]['PUT']['premium']
            long_call_premium = options_data['strikes'][long_call_strike]['CALL']['premium']
            long_put_premium = options_data['strikes'][long_put_strike]['PUT']['premium']
            
            if all([short_call_premium, short_put_premium, long_call_premium, long_put_premium]):
                quantity = int((self.portfolio['cash'] * self.position_size) / 
                             (long_call_premium + long_put_premium))
                
                if quantity > 0:
                    total_cost = (long_call_premium + long_put_premium - 
                                short_call_premium - short_put_premium) * quantity
                    
                    if total_cost <= self.portfolio['cash']:
                        self.portfolio['positions'][f"{symbol}_IRON_CONDOR"] = {
                            'type': 'IRON_CONDOR',
                            'quantity': quantity,
                            'short_call_strike': short_call_strike,
                            'short_put_strike': short_put_strike,
                            'long_call_strike': long_call_strike,
                            'long_put_strike': long_put_strike,
                            'short_call_premium': short_call_premium,
                            'short_put_premium': short_put_premium,
                            'long_call_premium': long_call_premium,
                            'long_put_premium': long_put_premium,
                            'expiry': expiry,
                            'entry_time': datetime.now()
                        }
                        self.portfolio['cash'] -= total_cost
                        print(f"\nExecuted Iron Condor on {symbol}:")
                        print(f"Short CALL at ₹{short_call_strike:.2f}, Premium: ₹{short_call_premium:.2f}")
                        print(f"Short PUT at ₹{short_put_strike:.2f}, Premium: ₹{short_put_premium:.2f}")
                        print(f"Long CALL at ₹{long_call_strike:.2f}, Premium: ₹{long_call_premium:.2f}")
                        print(f"Long PUT at ₹{long_put_strike:.2f}, Premium: ₹{long_put_premium:.2f}")
                        return True
        return False
    
    def straddle(self, symbol, options_data):
        """Straddle Strategy"""
        if not options_data:
            return False
            
        current_price = options_data['current_price']
        expiry = options_data['expiry']
        
        # Find ATM strike
        for strike, data in options_data['strikes'].items():
            if abs(strike - current_price) < 100:
                call_premium = data['CALL']['premium']
                put_premium = data['PUT']['premium']
                
                if call_premium and put_premium:
                    quantity = int((self.portfolio['cash'] * self.position_size) / 
                                 (call_premium + put_premium))
                    
                    if quantity > 0:
                        total_cost = (call_premium + put_premium) * quantity
                        
                        if total_cost <= self.portfolio['cash']:
                            self.portfolio['positions'][f"{symbol}_STRADDLE"] = {
                                'type': 'STRADDLE',
                                'quantity': quantity,
                                'strike': strike,
                                'call_premium': call_premium,
                                'put_premium': put_premium,
                                'expiry': expiry,
                                'entry_time': datetime.now()
                            }
                            self.portfolio['cash'] -= total_cost
                            print(f"\nExecuted Straddle on {symbol}:")
                            print(f"Bought CALL and PUT at strike ₹{strike:.2f}")
                            print(f"Total premium paid: ₹{total_cost:.2f}")
                            return True
        return False
    
    def vertical_spread(self, symbol, options_data, direction='bull'):
        """Vertical Spread Strategy"""
        if not options_data:
            return False
            
        current_price = options_data['current_price']
        expiry = options_data['expiry']
        
        strikes = sorted(options_data['strikes'].keys())
        atm_index = next(i for i, strike in enumerate(strikes) if strike >= current_price)
        
        if direction == 'bull' and atm_index + 1 < len(strikes):
            # Bull call spread
            long_strike = strikes[atm_index]
            short_strike = strikes[atm_index + 1]
            
            long_premium = options_data['strikes'][long_strike]['CALL']['premium']
            short_premium = options_data['strikes'][short_strike]['CALL']['premium']
            
            if long_premium and short_premium:
                quantity = int((self.portfolio['cash'] * self.position_size) / 
                             (long_premium - short_premium))
                
                if quantity > 0:
                    total_cost = (long_premium - short_premium) * quantity
                    
                    if total_cost <= self.portfolio['cash']:
                        self.portfolio['positions'][f"{symbol}_BULL_SPREAD"] = {
                            'type': 'BULL_SPREAD',
                            'quantity': quantity,
                            'long_strike': long_strike,
                            'short_strike': short_strike,
                            'long_premium': long_premium,
                            'short_premium': short_premium,
                            'expiry': expiry,
                            'entry_time': datetime.now()
                        }
                        self.portfolio['cash'] -= total_cost
                        print(f"\nExecuted Bull Call Spread on {symbol}:")
                        print(f"Long CALL at ₹{long_strike:.2f}, Premium: ₹{long_premium:.2f}")
                        print(f"Short CALL at ₹{short_strike:.2f}, Premium: ₹{short_premium:.2f}")
                        return True
        elif direction == 'bear' and atm_index > 0:
            # Bear put spread
            long_strike = strikes[atm_index]
            short_strike = strikes[atm_index - 1]
            
            long_premium = options_data['strikes'][long_strike]['PUT']['premium']
            short_premium = options_data['strikes'][short_strike]['PUT']['premium']
            
            if long_premium and short_premium:
                quantity = int((self.portfolio['cash'] * self.position_size) / 
                             (long_premium - short_premium))
                
                if quantity > 0:
                    total_cost = (long_premium - short_premium) * quantity
                    
                    if total_cost <= self.portfolio['cash']:
                        self.portfolio['positions'][f"{symbol}_BEAR_SPREAD"] = {
                            'type': 'BEAR_SPREAD',
                            'quantity': quantity,
                            'long_strike': long_strike,
                            'short_strike': short_strike,
                            'long_premium': long_premium,
                            'short_premium': short_premium,
                            'expiry': expiry,
                            'entry_time': datetime.now()
                        }
                        self.portfolio['cash'] -= total_cost
                        print(f"\nExecuted Bear Put Spread on {symbol}:")
                        print(f"Long PUT at ₹{long_strike:.2f}, Premium: ₹{long_premium:.2f}")
                        print(f"Short PUT at ₹{short_strike:.2f}, Premium: ₹{short_premium:.2f}")
                        return True
        return False
    
    def check_positions(self):
        """Check and manage existing positions"""
        for symbol, position in list(self.portfolio['positions'].items()):
            stock_symbol = symbol.split('_')[0]
            stock = yf.Ticker(self.symbols[stock_symbol])
            current_price = stock.history(period='1d')['Close'].iloc[-1]
            
            if position['type'] == 'COVERED_CALL':
                # Check if stock price is above call strike
                if current_price > position['call_strike']:
                    print(f"\nStock price above call strike for {symbol}")
                    # Close position
                    stock_value = position['stock_quantity'] * current_price
                    self.portfolio['cash'] += stock_value
                    del self.portfolio['positions'][symbol]
                    print(f"Closed position with profit: ₹{stock_value - (position['stock_quantity'] * position['stock_price']):.2f}")
            
            elif position['type'] == 'PROTECTIVE_PUT':
                # Check if put is in the money
                if current_price < position['put_strike']:
                    print(f"\nPut option in the money for {symbol}")
                    # Exercise put
                    put_value = (position['put_strike'] - current_price) * position['stock_quantity']
                    self.portfolio['cash'] += put_value
                    del self.portfolio['positions'][symbol]
                    print(f"Exercised put with profit: ₹{put_value - (position['put_premium'] * position['stock_quantity']):.2f}")
            
            elif position['type'] in ['IRON_CONDOR', 'STRADDLE']:
                # Check if any leg is in the money
                if position['type'] == 'IRON_CONDOR':
                    if current_price > position['long_call_strike'] or current_price < position['long_put_strike']:
                        print(f"\nIron Condor outside range for {symbol}")
                        # Close position
                        del self.portfolio['positions'][symbol]
                elif position['type'] == 'STRADDLE':
                    if abs(current_price - position['strike']) > position['strike'] * 0.1:
                        print(f"\nStraddle reached target for {symbol}")
                        # Close position
                        del self.portfolio['positions'][symbol]
            
            elif position['type'] in ['BULL_SPREAD', 'BEAR_SPREAD']:
                # Check if spread reached target
                if position['type'] == 'BULL_SPREAD':
                    if current_price > position['short_strike']:
                        print(f"\nBull spread reached target for {symbol}")
                        # Close position
                        del self.portfolio['positions'][symbol]
                else:  # BEAR_SPREAD
                    if current_price < position['short_strike']:
                        print(f"\nBear spread reached target for {symbol}")
                        # Close position
                        del self.portfolio['positions'][symbol]
    
    def calculate_total_profit(self):
        """Calculate total profit including open positions"""
        total_value = self.portfolio['cash']
        
        for symbol, position in self.portfolio['positions'].items():
            stock_symbol = symbol.split('_')[0]
            stock = yf.Ticker(self.symbols[stock_symbol])
            current_price = stock.history(period='1d')['Close'].iloc[-1]
            
            if position['type'] in ['CALL', 'PUT']:
                if position['type'] == 'CALL':
                    profit = max(0, current_price - position['strike']) - position['premium']
                else:  # PUT
                    profit = max(0, position['strike'] - current_price) - position['premium']
                total_value += profit * position['quantity']
            
            elif position['type'] == 'COVERED_CALL':
                stock_value = position['stock_quantity'] * current_price
                call_value = max(0, current_price - position['call_strike']) * position['stock_quantity']
                total_value += stock_value - call_value
            
            elif position['type'] == 'PROTECTIVE_PUT':
                stock_value = position['stock_quantity'] * current_price
                put_value = max(0, position['put_strike'] - current_price) * position['stock_quantity']
                total_value += stock_value + put_value
            
            elif position['type'] == 'IRON_CONDOR':
                short_call_value = max(0, current_price - position['short_call_strike']) * position['quantity']
                short_put_value = max(0, position['short_put_strike'] - current_price) * position['quantity']
                long_call_value = max(0, current_price - position['long_call_strike']) * position['quantity']
                long_put_value = max(0, position['long_put_strike'] - current_price) * position['quantity']
                net_value = (position['short_call_premium'] + position['short_put_premium'] - 
                           position['long_call_premium'] - position['long_put_premium']) * position['quantity']
                total_value += net_value
            
            elif position['type'] == 'STRADDLE':
                call_value = max(0, current_price - position['strike']) * position['quantity']
                put_value = max(0, position['strike'] - current_price) * position['quantity']
                net_value = (call_value + put_value) - (position['call_premium'] + position['put_premium']) * position['quantity']
                total_value += net_value
            
            elif position['type'] in ['BULL_SPREAD', 'BEAR_SPREAD']:
                if position['type'] == 'BULL_SPREAD':
                    long_value = max(0, current_price - position['long_strike']) * position['quantity']
                    short_value = max(0, current_price - position['short_strike']) * position['quantity']
                else:  # BEAR_SPREAD
                    long_value = max(0, position['long_strike'] - current_price) * position['quantity']
                    short_value = max(0, position['short_strike'] - current_price) * position['quantity']
                net_value = (long_value - short_value) - (position['long_premium'] - position['short_premium']) * position['quantity']
                total_value += net_value
        
        return total_value
    
    def run(self):
        """Main bot loop"""
        print("\nStarting Advanced Options Trading Bot...")
        print("Monitoring Bank Nifty and Nifty options...")
        print(f"Target: Achieve {self.target_profit_percentage}% overall profit")
        
        while True:
            try:
                current_time = datetime.now().time()
                print(f"\nCurrent time: {current_time}")
                
                # Calculate current profit percentage
                total_value = self.calculate_total_profit()
                profit_percentage = ((total_value - self.portfolio['initial_capital']) / self.portfolio['initial_capital']) * 100
                print(f"Current Profit: {profit_percentage:.2f}%")
                
                # Check if we've reached the target profit
                if profit_percentage >= self.target_profit_percentage:
                    print(f"\nTarget profit of {self.target_profit_percentage}% reached!")
                    print(f"Final Portfolio Value: ₹{total_value:,.2f}")
                    print(f"Total Profit: ₹{total_value - self.portfolio['initial_capital']:,.2f}")
                    self.save_trade_history()
                    break
                
                # Indian market hours: 9:15 AM to 3:30 PM
                if current_time.hour < 9 or (current_time.hour == 9 and current_time.minute < 15) or \
                   current_time.hour > 15 or (current_time.hour == 15 and current_time.minute > 30):
                    print("\nMarket is closed. Waiting for market hours...")
                    time.sleep(60)
                    continue
                
                # Check existing positions
                self.check_positions()
                
                for name, symbol in self.symbols.items():
                    print(f"\nProcessing {name}...")
                    # Get options chain
                    options_data = self.get_options_chain(name)
                    
                    if options_data:
                        # Choose strategy based on market conditions
                        if len(self.portfolio['positions']) == 0:
                            # Implement different strategies based on market conditions
                            if options_data['current_price'] > options_data['strikes'][list(options_data['strikes'].keys())[0]]['CALL']['premium']:
                                self.covered_call(name, options_data)
                            elif options_data['current_price'] < options_data['strikes'][list(options_data['strikes'].keys())[-1]]['PUT']['premium']:
                                self.protective_put(name, options_data)
                            else:
                                # Check volatility for other strategies
                                volatility = self.calculate_volatility(name)
                                if volatility > 0.2:
                                    self.straddle(name, options_data)
                                else:
                                    self.iron_condor(name, options_data)
                
                # Display portfolio status
                self.display_portfolio()
                
                # Small delay to prevent overwhelming the API
                time.sleep(1)
                
            except Exception as e:
                print(f"\nError in main loop: {str(e)}")
                print(f"Error type: {type(e).__name__}")
                import traceback
                traceback.print_exc()
                print("Retrying in 5 seconds...")
                time.sleep(5)
                continue

if __name__ == "__main__":
    try:
        bot = AdvancedOptionsBot()
        bot.run()
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 