import datetime
import pandas as pd

class Trade:
    def __init__(self, entry_time, pair, direction, entry_price, stop_loss, take_profit, volume, reason, risk_amount=None, entry_volume=0.0):
        print(f"DEBUG: Trade.__init__ received entry_volume: {entry_volume}")
        self.entry_time = entry_time
        self.pair = pair
        self.direction = direction
        self.entry_price = entry_price
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.volume = volume
        self.reason = reason
        self.exit_time = None
        self.exit_price = None
        self.exit_reason = None
        self.pips_gained = 0
        self.usd_pnl = 0
        self.duration_hours = 0
        self.is_closed = False
        self.risk_amount = risk_amount
        self.balance_after = 0.0 # Initialize to 0.0
        self.sl_pips = 0.0 # New: Store Stop Loss in pips
        self.tp_pips = 0.0 # New: Store Take Profit in pips
        self.entry_volume = entry_volume # New: Store entry candle volume
    
    def calculate_pnl(self, exit_price):
        """Calculate P&L for this trade"""
        pip_size = 0.01 if "JPY" in self.pair else 0.0001
        pip_value = 9.09 if "JPY" in self.pair else 10.0
        
        if self.direction == "BUY":
            pips = (exit_price - self.entry_price) / pip_size
        else:  # SELL
            pips = (self.entry_price - exit_price) / pip_size
        
        self.pips_gained = pips
        self.usd_pnl = pips * pip_value * self.volume
        return self.usd_pnl
    
    def close_trade(self, exit_time, exit_price, exit_reason):
        """Close the trade and calculate all metrics"""
        self.exit_time = exit_time
        self.exit_price = exit_price
        self.exit_reason = exit_reason
        self.is_closed = True
        
        # Calculate duration
        if isinstance(self.entry_time, (datetime.datetime, pd.Timestamp)) and \
           isinstance(self.exit_time, (datetime.datetime, pd.Timestamp)):
            duration = self.exit_time - self.entry_time
            self.duration_hours = duration.total_seconds() / 3600
        
        # Calculate P&L
        self.calculate_pnl(exit_price)
        
        return self.usd_pnl

    def __str__(self):
        return (f"Trade({self.direction} {self.pair}, PnL: ${self.usd_pnl:+.2f}, "
                f"Entry: {self.entry_time.strftime('%Y-%m-%d %H:%M') if self.entry_time else 'N/A'}, "
                f"Exit: {self.exit_time.strftime('%Y-%m-%d %H:%M') if self.exit_time else 'N/A'}, "
                f"SL: {self.sl_pips:.1f} pips, TP: {self.tp_pips:.1f} pips, Entry Volume: {self.entry_volume:.0f}, Position Volume: {self.volume:.2f} lots)")