import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import logging
import json
import pickle
from collections import defaultdict
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest.trade_models import Trade
import warnings

# 🎯 DYNAMIC STRATEGY IMPORTS - ONE LINE PER IMPORT
from strategy.usdjpy_strategy import USDJPYStrategy
from strategy.eurgbp_strategy import EURGBPStrategy
from strategy.eurjpy_strategy import EURJPYStrategy
from strategy.gbpjpy_strategy import GBPJPYStrategy
# from strategy.eur_usd_fibonacci import EURUSDFIBONACCI

# Ignore FutureWarnings from pandas
warnings.filterwarnings('ignore', category=FutureWarning, module='pandas')

# Setup logging
# timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# log_file_path = f"backtest_logs/{self.target_pair.replace('/', '_')}_strategy_debug_{timestamp}.log"
# logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)


class BacktestEngineM30:
    def __init__(self, strategy, target_pair: str = "EUR/USD", start_balance: float = 1000.0, is_autotuning: bool = False, start_date: str = None, end_date: str = None):
        self.strategy = strategy # Accept strategy instance directly
        self.target_pair = target_pair
        self.initial_balance = start_balance
        self.current_balance = start_balance
        self.trades = []
        self.open_trades = []
        self.peak_balance = start_balance
        self.lowest_balance = start_balance
        # self.trade_candle_data = {}  # Initialize trade_candle_data
        self.is_autotuning = is_autotuning # Flag to control logging verbosity
        self.detailed_trade_data = [] # To store detailed trade and candle data
        self.trade_log_detailed_file = None # To define the output path for the detailed trade log
        self.summary_log_file = None # To define the output path for the backtest summary
        self.start_date = pd.to_datetime(start_date) if start_date else datetime.now() - timedelta(days=365)
        self.end_date = pd.to_datetime(end_date) if end_date else datetime.now()

        # Define min/max position size (standard lots)
        self.min_position_size = 0.01
        self.max_position_size = 5.0

        # Setup logging for trades and candle data
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.trade_log_detailed_file = f"backtest_logs/{self.target_pair.replace('/', '_')}_detailed_trades_{timestamp}.xlsx"
        self.summary_log_file = f"backtest_logs/{self.target_pair.replace('/', '_')}_summary_{timestamp}.txt"
        # # self.trade_log_file = f"backtest_logs/{self.target_pair.reZZplace('/', '_')}_FULL_trades_{timestamp}.xlsx"
        # self.trade_data_path = f"candle_data/{self.target_pair.replace('/', '_')}_candle_data_{timestamp}.pkl"

        # Ensure directories exist
        os.makedirs('backtest_logs', exist_ok=True)
        os.makedirs('candle_data', exist_ok=True)

        if not self.is_autotuning:
            logger.info(f"Backtest Engine: ${self.initial_balance} initial balance")
            # logger.info(f"🎯 TARGET PAIR: {self.target_pair}")
            # logger.info(f"🧠 STRATEGY: {self.strategy.__class__.__name__}")
            # logger.info(f"🔧 REALISTIC SIMULATION: Cron timing + execution delays + state persistence")
            # logger.info(f"🕯️ COMPREHENSIVE CANDLE DATA: Entry to Exit tracking")

    def find_cron_execution_points(self, df):
        """
        🔧 REALISTIC SIMULATION: Find execution points for native timeframe data
        - EUR/USD (H1): Execute on every H1 bar (cron runs hourly)
        - H4 pairs: Execute on every H4 bar (cron runs every 4 hours)
        
        Since we have native timeframe data, each bar represents the exact period
        when cron would execute, so we process most bars but skip some for realism.
        """
        execution_points = []
        
        pair_timeframes = {
            "EUR/USD": "M30",
            "USD/JPY": "M30",
            "EUR/JPY": "M30"
        }
        
        expected_timeframe = pair_timeframes.get(self.target_pair, "M30")
        
        try:
            # Ensure timestamp column is datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # PURE STRATEGY TESTING: Process ALL bars for maximum strategy evaluation
            for idx in range(len(df)):
                execution_points.append(idx)
            
            execution_rate = (len(execution_points) / len(df)) * 100
            
            logger.info(f"Found {len(execution_points)} execution points for {self.target_pair} ({expected_timeframe})")
            logger.info(f"Timeframe: {expected_timeframe} | Execution frequency: {execution_rate:.1f}% of total bars")
            logger.info(f"PURE STRATEGY TESTING: Processing ALL bars for maximum evaluation")
            
            return execution_points
            
        except Exception as e:
            logger.error(f"Error finding execution points: {e}")
            # Fallback: use every 30th bar (rough approximation)
            fallback_points = list(range(0, len(df), 30))
            logger.warning(f"Using fallback execution points: {len(fallback_points)}")
            return fallback_points
    
    def simulate_execution_delay(self, signal_index, df):
        """
        🎯 PURE STRATEGY TESTING: No execution delays - immediate execution
        """
        try:
            # Immediate execution for pure strategy testing
            actual_price = df.iloc[signal_index]['close']
            actual_timestamp = df.iloc[signal_index]['timestamp']
            
            return signal_index, actual_price, actual_timestamp
            
        except Exception as e:
            logger.error(f"Error in execution: {e}")
            return signal_index, df.iloc[signal_index]['close'], df.iloc[signal_index]['timestamp']
    
    def get_strategy_file_info(self):
        """
        📝 GET STRATEGY FILE CREATION INFO FOR MISSING PAIRS
        Provides guidance on creating new strategy files
        """
        pair_clean = ''.join(self.target_pair.lower().split('/'))
        class_name = ''.join(self.target_pair.split('/'))
        
        return {
            'file_path': f"strategy/{pair_clean}_strategy.py",
            'class_name': f"{class_name}Strategy",
            'import_statement': f"from strategy.{pair_clean}_strategy import {class_name}Strategy",
            'uncomment_line': f"# from strategy.{pair_clean}_strategy import {class_name}Strategy",
            'example_template': f'''
# strategy/{pair_clean}_strategy.py
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class {class_name}Strategy:
    def __init__(self):
        self.check_count = 0
        self.filtered_count = 0
        self.passed_count = 0
    
    def analyze_trade_signal(self, df: pd.DataFrame, pair: str):
        """
        {self.target_pair} specific trading strategy
        Implement your {self.target_pair} logic here
        """
        self.check_count += 1
        
        try:
            # Add your {self.target_pair} strategy logic here
            # Return format should be:
            # {{
            #     "decision": "BUY" | "SELL" | "NO TRADE",
            #     "entry_price": float,
            #     "stop_loss": float,
            #     "take_profit": float,
            #     "volume": float,
            #     "reason": str
            # }}
            
            # Placeholder - replace with actual strategy
            return {{"decision": "NO TRADE", "reason": "Strategy not implemented"}}
            
        except Exception as e:
            logger.error(f"{class_name}Strategy error: {{e}}")
            return {{"decision": "NO TRADE", "reason": f"Error: {{str(e)}}"}}
    
    def get_statistics(self):
        return {{
            'total_checks': self.check_count,
            'filtered_out': self.filtered_count,
            'passed_to_strategy': self.passed_count
        }}
    
    def print_final_stats(self):
        stats = self.get_statistics()
        print(f"\\n🎯 {class_name.upper()} STRATEGY STATS:")
        print(f"   Total Checks: {{stats['total_checks']:,}}")
        print(f"   Signals Generated: {{stats['passed_to_strategy']:,}}")
'''
        }
    
    def get_pip_size(self, pair):
        """Get pip size for any currency pair"""
        if "JPY" in pair:
            return 0.01
        else:
            return 0.0001
    
    def calculate_position_size_for_risk(self, pair, entry_price, stop_loss, risk_amount_usd=100):
        """
        Calculate position size (lots) to risk exactly the specified USD amount
        
        Formula:
        Position Size = Risk Amount / (Stop Loss Distance in Pips × Pip Value)
        """
        pip_size = self.get_pip_size(pair)
        
        # Calculate stop loss distance in pips
        sl_distance_price = abs(entry_price - stop_loss)
        sl_distance_pips = sl_distance_price / pip_size
        
        # Get pip value per standard lot
        # This is simplified - in reality, you'd need current exchange rates
        pip_values_per_lot = {
            "EUR/USD": 10.0,
            "GBP/USD": 10.0,
            "USD/JPY": 10.0 # Assuming similar pip value for now
        }
        
        pip_value = pip_values_per_lot.get(pair, 10.0)  # Default to $10
        
        # Avoid division by zero if sl_distance_pips is 0
        if sl_distance_pips == 0:
            return self.min_position_size

        # Calculate position size in lots
        position_size = risk_amount_usd / (sl_distance_pips * pip_value)
        
        # Round to 2 decimal places (0.01 lot increments)
        position_size = round(position_size, 2)
        
        # Apply min/max constraints
        position_size = max(self.min_position_size, min(position_size, self.max_position_size))
        
        return position_size

    def get_trading_costs(self, pair):
        """Get trading costs for different pairs (PURE STRATEGY TESTING: zero costs)"""
        # Pure strategy testing: no spread/slippage for maximum strategy evaluation
        return {"spread": 0.0, "slippage": 0.0}
    
    def collect_comprehensive_candle_data(self, trade, full_df, entry_index, exit_index):
        """
        Collect comprehensive candle data and a focused slice of up to 500 post-entry candles
        🕯️ COMPLETE PRICE ACTION HISTORY + 500-candle post-entry window
        """
        try:
            trade_id = len(self.trades)
            
            # Calculate full range: 500 candles before entry until exit
            start_index = max(0, entry_index - 500)
            end_index = min(len(full_df) - 1, exit_index)
            
            # Extract all candles in range
            candle_range = full_df.iloc[start_index:end_index + 1].copy()
            candle_range.reset_index(drop=True, inplace=True)
            
            # Add relative positioning
            entry_relative_index = entry_index - start_index
            exit_relative_index = exit_index - start_index
            
            # Mark special candles
            candle_range['candle_type'] = 'normal'
            candle_range.loc[:entry_relative_index - 1, 'candle_type'] = 'pre_entry'
            candle_range.loc[entry_relative_index, 'candle_type'] = 'entry'
            candle_range.loc[entry_relative_index + 1:exit_relative_index - 1, 'candle_type'] = 'in_trade'
            candle_range.loc[exit_relative_index, 'candle_type'] = 'exit'
            
            # Calculate price movements relative to entry
            entry_price = trade.entry_price
            pip_size = self.get_pip_size(trade.pair)
            
            candle_range['pips_from_entry'] = ((candle_range['close'] - entry_price) / pip_size).round(1)
            candle_range['high_pips_from_entry'] = ((candle_range['high'] - entry_price) / pip_size).round(1)
            candle_range['low_pips_from_entry'] = ((candle_range['low'] - entry_price) / pip_size).round(1)
            
            # Mark SL/TP levels hit
            if trade.direction == "BUY":
                candle_range['sl_hit'] = candle_range['low'] <= trade.stop_loss
                candle_range['tp_hit'] = candle_range['high'] >= trade.take_profit
            else:
                candle_range['sl_hit'] = candle_range['high'] >= trade.stop_loss
                candle_range['tp_hit'] = candle_range['low'] <= trade.take_profit
            
            # Calculate candle patterns
            candle_range['body_size'] = abs(candle_range['close'] - candle_range['open'])
            candle_range['upper_wick'] = candle_range['high'] - candle_range[['open', 'close']].max(axis=1)
            candle_range['lower_wick'] = candle_range[['open', 'close']].min(axis=1) - candle_range['low']
            candle_range['total_range'] = candle_range['high'] - candle_range['low']
            candle_range['body_pct'] = (candle_range['body_size'] / candle_range['total_range'] * 100).round(1)
            
            # Bullish/Bearish classification
            candle_range['bullish'] = candle_range['close'] > candle_range['open']
            candle_range['bearish'] = candle_range['close'] < candle_range['open']
            candle_range['doji'] = abs(candle_range['close'] - candle_range['open']) < (candle_range['total_range'] * 0.1)
            
            # Build a focused slice: up to 500 candles after entry (or until exit)
            post_entry_start = entry_index + 1
            post_entry_end = min(entry_index + 500, exit_index)
            post_entry_range = full_df.iloc[post_entry_start:post_entry_end + 1].copy()
            post_entry_range.reset_index(drop=True, inplace=True)

            # Add relative indexing for post-entry window
            post_entry_range['candle_index_from_entry'] = range(1, len(post_entry_range) + 1)

            # Store comprehensive trade candle data
            trade_candle_info = {
                'trade_id': trade_id,
                'pair': trade.pair,
                'direction': trade.direction,
                'entry_time': trade.entry_time,
                'exit_time': trade.exit_time,
                'entry_price': trade.entry_price,
                'exit_price': trade.exit_price,
                'stop_loss': trade.stop_loss,
                'take_profit': trade.take_profit,
                'sl_pips': trade.sl_pips,
                'tp_pips': trade.tp_pips,
                'entry_volume': trade.entry_volume, # New: Add entry candle volume
                'exit_reason': trade.exit_reason,
                'pips_gained': trade.pips_gained,
                'duration_hours': trade.duration_hours,
                'total_candles': len(candle_range),
                'pre_entry_candles': entry_relative_index,
                'in_trade_candles': exit_relative_index - entry_relative_index,
                'entry_index_in_data': entry_index,
                'exit_index_in_data': exit_index,
                'candle_data': candle_range.to_dict('records'),  # All candles in full range
                'post_entry_candles_500': post_entry_range.to_dict('records')  # Up to 500
            }
            
            # Store in trade_candle_data dictionary
            self.detailed_trade_data.append(trade_candle_info)
            
            logger.info(f"Collected {len(candle_range)} candles for Trade #{trade_id} ({trade.direction} {trade.pair})")
            
            return trade_candle_info
            
        except Exception as e:
            logger.error(f"Error collecting candle data for trade: {e}")
            return None
    
    def aggregate_to_timeframe(self, df, target_timeframe):
        """
        🔧 Convert 30-minute data to target timeframe (H1 or H4)
        """
        try:
            if target_timeframe == "M30":
                return df  # Already 30-minute data
            
            # Ensure timestamp is datetime and set as index
            df = df.copy()
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            
            # Define aggregation rules
            agg_rules = {
                'open': 'first',
                'high': 'max', 
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }
            
            # Resample based on target timeframe
            if target_timeframe == "H1":
                # Aggregate to 1-hour bars
                resampled = df.resample('1H').agg(agg_rules)
            elif target_timeframe == "H4":
                # Aggregate to 4-hour bars  
                resampled = df.resample('4H').agg(agg_rules)
            else:
                logger.warning(f"Unknown timeframe {target_timeframe}, returning original data")
                return df.reset_index()
            
            # Remove any NaN rows and reset index
            resampled = resampled.dropna()
            resampled.reset_index(inplace=True)
            
            # 🔧 CRITICAL: Recalculate candle_range for aggregated data
            # Strategies depend on this column for analysis
            resampled['candle_range'] = resampled['high'] - resampled['low']
            
            logger.info(f"Aggregated {len(df)} M30 bars to {len(resampled)} {target_timeframe} bars")
            logger.info(f"Recalculated candle_range for {target_timeframe} data")
            return resampled
            
        except Exception as e:
            logger.error(f"Error aggregating to {target_timeframe}: {e}")
            return df
    
    def load_excel_data(self, file_path='data/forex_data1.xlsx'):
        """
        🚀 ENHANCED: Load NATIVE timeframe data directly from fetch_data.py
        - EUR/USD: Native H1 data (no aggregation needed)
        - H4 pairs: Native H4 data (no aggregation needed)
        - Maximum accuracy with broker's official candles
        """
        # 🔧 FIX: Handle both relative paths (from backtest dir) and absolute paths
        possible_paths = [
            file_path,
            f'backtest/{file_path}',
            f'data/forex_data1.xlsx',
            f'backtest/data/forex_data1.xlsx'
        ]
        
        actual_file_path = None
        for path in possible_paths:
            if os.path.exists(path):
                actual_file_path = path
                break
        
        if not actual_file_path:
            logger.error(f"Data file not found. Tried paths: {possible_paths}")
            return {}
        
        # Get pair's expected timeframe
        pair_timeframes = {
            "EUR/USD": "M30",
            "GBP/USD": "M30",
            "USD/JPY": "M30", # Add USD/JPY
            "EUR/GBP": "M30", # Added EUR/GBP
            "EUR/JPY": "M30" # Added EUR/JPY
        }
        
        expected_timeframe = pair_timeframes.get(self.target_pair, "M30")
        
        logger.info(f"Loading {self.target_pair} data from {actual_file_path}")
        logger.info(f"Expected timeframe: {expected_timeframe} (NATIVE)")
        data = {}
        
        try:
            with pd.ExcelFile(actual_file_path) as excel_file:
                sheet_names = excel_file.sheet_names
                target_sheet = None
                
                # Try different possible sheet names for target pair
                pair_variations = [
                    self.target_pair,
                    self.target_pair.replace("/", "_"),
                    self.target_pair.replace("/", ""),
                    self.target_pair.replace("-", "_"),
                    self.target_pair.replace("-", ""),
                    self.target_pair.lower(),
                    self.target_pair.lower().replace("/", "_"),
                    self.target_pair.lower().replace("/", ""),
                    self.target_pair.upper(),
                    self.target_pair.upper().replace("/", "_"),
                    self.target_pair.upper().replace("/", ""),
                    f'{self.target_pair.replace("/", "_")}_M30', # Explicit M30 sheet name
                    f'{self.target_pair.replace("/", "")}_M30'  # Explicit M30 sheet name
                ]
                
                for variation in pair_variations:
                    if variation in sheet_names:
                        target_sheet = variation
                        break
                
                if not target_sheet:
                    logger.error(f"{self.target_pair} sheet not found. Available sheets: {sheet_names}")
                    return {}
                
                # Read target pair sheet (now contains NATIVE timeframe data)
                df = pd.read_excel(excel_file, sheet_name=target_sheet, engine='openpyxl')
                
                # Ensure timestamp column is datetime
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                
                # Sort by timestamp (ascending for backtest)
                df.sort_values('timestamp', inplace=True, ascending=True)
                df.reset_index(drop=True, inplace=True)
                
                # Filter data by start and end dates
                df = df[(df['timestamp'] >= self.start_date) & (df['timestamp'] <= self.end_date)]
                
                # Ensure candle_range exists (should be pre-calculated)
                if 'candle_range' not in df.columns:
                    df['candle_range'] = df['high'] - df['low']
                    logger.warning(f"Added missing candle_range column")
                
                # --- DEBUGGING VOLUME --- #
                print(f"DEBUG: DataFrame columns: {df.columns.tolist()}")
                if 'volume' in df.columns:
                    print(f"DEBUG: First 5 volume values: {df['volume'].head().tolist()}")
                    print(f"DEBUG: Volume column dtype: {df['volume'].dtype}")
                else:
                    print("DEBUG: 'volume' column NOT found in DataFrame after loading.")
                # --- END DEBUGGING VOLUME --- #

                logger.info(f"Loaded NATIVE {self.target_pair}: {len(df):,} {expected_timeframe} bars")
                logger.info(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
                logger.info(f"NATIVE TIMEFRAME: No aggregation needed - maximum broker accuracy!")
                
                data[self.target_pair] = df

            
            return data
            
        except Exception as e:
            logger.error(f"Error loading {self.target_pair} data: {e}")
            return {}
    
    def open_trade(self, signal, timestamp, pair, market_price, current_index, full_df, entry_candle_volume: float = 0.0):
        """Open a new trade with PROPER risk-based position sizing"""
        if signal.get("decision", "NO TRADE").upper() == "NO TRADE":
            return
        
        # Validate signal
        required = ["entry_price", "stop_loss", "take_profit"]
        if not all(field in signal for field in required):
            logger.warning(f"Invalid signal for {pair}: missing required fields")
            return
        
        # FIXED: Calculate position size based on $100 risk
        risk_per_trade = 100.0  # Risk $100 per trade
        
        # Use strategy's SL for risk calculation
        calculated_volume = self.calculate_position_size_for_risk(
            pair, 
            signal["entry_price"],  # Use strategy's entry price
            signal["stop_loss"],  # Stop loss from strategy
            risk_per_trade
        )
        
        # Apply trading costs (for realistic mode - set to 0 for pure strategy testing)
        costs = self.get_trading_costs(pair)
        pip_size = self.get_pip_size(pair)
        total_cost_pips = (costs["spread"] + costs["slippage"])
        total_cost_price_units = total_cost_pips * pip_size

        if signal["decision"].upper() == "BUY":
            actual_entry = market_price + total_cost_price_units
        else:
            actual_entry = market_price - total_cost_price_units
        
        # Create trade with strategy's original values
        trade = Trade(
            timestamp, pair, signal["decision"].upper(), actual_entry,
            signal["stop_loss"], signal["take_profit"], 
            calculated_volume, # Use calculated volume for position size
            signal.get("reason", "Strategy signal"),
            risk_amount=risk_per_trade,  # Store risk amount for PnL calculation
            entry_volume=entry_candle_volume # New: Pass entry candle volume
        )
        
        # Store entry index for candle data collection
        trade.entry_index = current_index
        trade.full_df_reference = full_df
        
        logger.info(f"OPENED {trade.direction} {pair} @ {trade.entry_price:.5f}")
        logger.info(f"   SL: {trade.stop_loss:.5f}, TP: {trade.take_profit:.5f}")
        logger.info(f"   Volume: {calculated_volume:.2f} lots (Risk: ${risk_per_trade})")
        
        self.open_trades.append(trade)
    
    def check_trade_exits(self, timestamp, bar_data, pair, current_index):
        """Check if any open trades should be closed and collect candle data"""
        trades_to_close = []
        
        for trade in self.open_trades:
            if trade.pair != pair:
                continue
            
            high = bar_data['high']
            low = bar_data['low']
            open_price = bar_data['open']
            
            exit_price = None
            exit_reason = None
            
            if trade.direction == "BUY":
                sl_hit = low <= trade.stop_loss
                tp_hit = high >= trade.take_profit
                
                if sl_hit and tp_hit:
                    # Both hit - determine order by distance to open
                    distance_to_sl = abs(open_price - trade.stop_loss)
                    distance_to_tp = abs(open_price - trade.take_profit)
                    
                    if distance_to_sl <= distance_to_tp:
                        exit_price = trade.stop_loss
                        exit_reason = "Stop Loss"
                    else:
                        exit_price = trade.take_profit
                        exit_reason = "Take Profit"
                elif sl_hit:
                    exit_price = trade.stop_loss
                    exit_reason = "Stop Loss"
                elif tp_hit:
                    exit_price = trade.take_profit
                    exit_reason = "Take Profit"
                
            else:  # SELL
                sl_hit = high >= trade.stop_loss
                tp_hit = low <= trade.take_profit
                
                if sl_hit and tp_hit:
                    distance_to_sl = abs(open_price - trade.stop_loss)
                    distance_to_tp = abs(open_price - trade.take_profit)
                    
                    if distance_to_sl <= distance_to_tp:
                        exit_price = trade.stop_loss
                        exit_reason = "Stop Loss"
                    else:
                        exit_price = trade.take_profit
                        exit_reason = "Take Profit"
                elif sl_hit:
                    exit_price = trade.stop_loss
                    exit_reason = "Stop Loss"
                elif tp_hit:
                    exit_price = trade.take_profit
                    exit_reason = "Take Profit"
        
            if exit_price:
                # 🕯️ COLLECT COMPREHENSIVE CANDLE DATA
                self.close_trade_with_candle_data(trade, timestamp, exit_price, exit_reason, current_index)
                trades_to_close.append(trade)
        
        # Remove closed trades
        for trade in trades_to_close:
            if trade in self.open_trades:
                self.open_trades.remove(trade)
    
    def close_trade_with_candle_data(self, trade, exit_time, exit_price, exit_reason, exit_index):
        """Close trade with PROPER PnL calculation based on fixed risk"""
        
        # Calculate actual PnL based on risk model
        pip_size = self.get_pip_size(trade.pair)
        
        if trade.direction == "BUY":
            pips_gained = (exit_price - trade.entry_price) / pip_size
            trade.sl_pips = abs(trade.entry_price - trade.stop_loss) / pip_size
            trade.tp_pips = abs(trade.take_profit - trade.entry_price) / pip_size
        else:  # SELL
            pips_gained = (trade.entry_price - exit_price) / pip_size
            trade.sl_pips = abs(trade.entry_price - trade.stop_loss) / pip_size
            trade.tp_pips = abs(trade.entry_price - trade.take_profit) / pip_size
        
        # Calculate PnL based on fixed risk
        sl_distance_pips = abs(trade.entry_price - trade.stop_loss) / pip_size
        
        if exit_reason == "Stop Loss":
            # Exactly -$100 (minus risk amount)
            pnl = -trade.risk_amount  # Should be -100
        elif exit_reason == "Take Profit":
            # Calculate R:R ratio and apply
            tp_distance_pips = abs(trade.take_profit - trade.entry_price) / pip_size
            rr_ratio = tp_distance_pips / sl_distance_pips
            pnl = trade.risk_amount * rr_ratio  # e.g., $100 * 2 = $200 for 1:2 RR
        else:
            # Exit at market (end of backtest or manual close)
            # Calculate based on actual pips vs risk pips
            pnl = (pips_gained / sl_distance_pips) * trade.risk_amount
        
        # Update trade object
        trade.exit_time = exit_time
        trade.exit_price = exit_price
        trade.exit_reason = exit_reason
        trade.pips_gained = pips_gained
        trade.usd_pnl = pnl
        trade.duration_hours = (exit_time - trade.entry_time).total_seconds() / 3600 if trade.entry_time else 0
        
        # Update balance
        self.current_balance += pnl
        trade.balance_after = self.current_balance # Update trade object with balance after closing
        if self.current_balance > self.peak_balance:
            self.peak_balance = self.current_balance
        if self.current_balance < self.lowest_balance:
            self.lowest_balance = self.current_balance
        
        # Collect comprehensive candle data
        self.collect_comprehensive_candle_data(
            trade, 
            trade.full_df_reference, 
            trade.entry_index, 
            exit_index
        )
        
        # Add to completed trades
        self.trades.append(trade)
        
        # Log to Excel
        # self.log_trade_to_excel(trade)
        
        # Log results
        logger.info(f"CLOSED {trade.direction} {trade.pair}")
        logger.info(f"   Exit: {exit_reason} | Pips: {pips_gained:+.1f} | PnL: ${pnl:+.2f}")
        logger.info(f"   SL Pips: {trade.sl_pips:.1f}, TP Pips: {trade.tp_pips:.1f}")
        logger.info(f"   Balance: ${self.current_balance:.2f}")
        
        return pnl
    
    def run_backtest(self):
        """
        🎯 PURE STRATEGY BACKTEST: Focus on strategy performance only
        - No execution delays
        - No slippage/spread
        - Process all bars for maximum evaluation
        - Clean strategy testing environment
        """
        logger.info(f"Starting PURE STRATEGY {self.target_pair} backtest...")

        # 🎯 CHECK IF STRATEGY IS AVAILABLE
        if self.strategy is None:
            strategy_info = self.get_strategy_file_info()
            logger.error(f"No strategy available for {self.target_pair}")
            print(f"\nBACKTEST ABORTED - NO STRATEGY AVAILABLE")
            print(f"Create strategy file for {self.target_pair}:")
            print(f"   File: {strategy_info['file_path']}")
            print(f"   Class: {strategy_info['class_name']}")
            return
        
        # Load target pair data
        data = self.load_excel_data()
        if not data or self.target_pair not in data:
            logger.error(f"No {self.target_pair} data loaded. Aborting backtest.")
            return
        
        pair_df = data[self.target_pair]
        if pair_df.empty:
            logger.error(f"{self.target_pair} data is empty. Aborting backtest.")
            return
        
        logger.info(f"{self.target_pair} data loaded: {len(pair_df):,} bars")
        
        # 🔧 Precompute indicator columns once for performance (EMA and RSI)
        try:
            ema_periods = getattr(self.strategy, 'ema_periods', [20, 50, 200]) or []
            for p in ema_periods:
                col = f"ema_{p}"
                if col not in pair_df.columns:
                    pair_df[col] = pair_df['close'].ewm(span=p, adjust=False).mean()

            rsi_period = getattr(self.strategy, 'rsi_period', 14)
            rsi_col = f"rsi_{rsi_period}"
            if rsi_col not in pair_df.columns:
                delta = pair_df['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
                rs = gain / loss
                pair_df[rsi_col] = 100 - (100 / (1 + rs))
        except Exception as e:
            logger.warning(f"Failed to precompute indicators: {e}")

        # Minimum data requirement
        min_bars_for_strategy = 250
        if len(pair_df) < min_bars_for_strategy:
            logger.error(f"Insufficient data. Need {min_bars_for_strategy} bars, got {len(pair_df)}")
            return
        
        # 🎯 PURE STRATEGY TESTING: Process all bars for maximum evaluation
        execution_points = self.find_cron_execution_points(pair_df)
        
        # Filter execution points to have minimum data
        valid_execution_points = [ep for ep in execution_points if ep >= min_bars_for_strategy - 1]
        
        logger.info(f"Processing {len(valid_execution_points)} execution points for pure strategy testing...")
        logger.info(f"Processing {len(valid_execution_points)}/{len(pair_df)} bars ({len(valid_execution_points)/len(pair_df)*100:.1f}%)")
        
        # 🎯 MAIN PURE STRATEGY TESTING LOOP
        processed_count = 0
        
        # Get the strategy for the target pair
        strategy = self.strategy # Use the injected strategy
        if not strategy:
            logger.error(f"No strategy provided to BacktestEngineM30 for {self.target_pair}")
            return {}

        # 🔧 FIXED: Simulate real-time data feed by iterating through cron points
        for i, idx in enumerate(valid_execution_points):
            current_candle = pair_df.iloc[idx]
            timestamp = current_candle['timestamp']
            current_price = current_candle['close']
            current_volume = current_candle.get('volume', 0.0) # Get volume, default to 0 if not present

            # Update strategy with new candle (for internal state management)
            # strategy.update_state(current_candle)

            # Close existing trades if TP/SL hit
            self.check_trade_exits(timestamp, current_candle, self.target_pair, idx)
            
            # Check for new signals (only if no open trade)
            has_open_trade = any(t.pair == self.target_pair for t in self.open_trades)
            
            if not has_open_trade:
                try:
                    # 🔧 FIXED: Use continuous data (like ctrader.py)
                    # Give strategy the FULL continuous data up to current point
                    current_data = pair_df.iloc[:idx + 1].copy()
                    
                    if len(current_data) >= min_bars_for_strategy:
                        # 🎯 PURE STRATEGY EVALUATION
                        signal = strategy.analyze_trade_signal(current_data, self.target_pair)
                        
                        if signal.get("decision", "NO TRADE").upper() != "NO TRADE":
                            # 🎯 IMMEDIATE EXECUTION: No delays or slippage for pure strategy testing
                            exec_result = self.simulate_execution_delay(idx, pair_df)
                            
                            if exec_result and len(exec_result) == 3:
                                actual_exec_idx, actual_price, actual_timestamp = exec_result
                                
                                # 🎯 NO TRADING COSTS: Pure strategy performance
                                # Use exact signal entry price for clean testing
                                
                                self.open_trade(signal, actual_timestamp, self.target_pair, 
                                              actual_price, actual_exec_idx, pair_df, current_volume)
                                
                                # Log every trade for pure strategy evaluation
                                if len(self.trades) % 10 == 1:  # Log every 10th trade
                                    logger.info(f"Trade #{len(self.trades)}: {signal['decision']} signal executed")
                        else:
                            # logger.debug(f"NO TRADE at {timestamp}: {signal.get('reason', 'Unknown reason')}")
                            pass
                            
                except Exception as e:
                    logger.error(f"Strategy error at {timestamp}: {e}")
            
            # Progress reporting for pure strategy testing
            progress = processed_count / len(valid_execution_points) * 100
            if processed_count % 500 == 0:  # Every 500 bars for cleaner output
                logger.info(f"Progress: {progress:.1f}% | Trades: {len(self.trades):,} | Balance: ${self.current_balance:.2f}")
                
            processed_count += 1
        
        # Close remaining trades
        if self.open_trades:
            logger.info("Closing remaining trades...")
            final_bar = pair_df.iloc[-1]
            final_index = len(pair_df) - 1
            for trade in self.open_trades[:]:
                self.close_trade_with_candle_data(trade, final_bar['timestamp'], final_bar['close'], "End of backtest", final_index)
        
        # Save all candle data
        # self.save_candle_data_to_files()
        self.save_detailed_trade_data_to_excel()
        
        # Print final statistics
        results = self.print_realistic_results(len(valid_execution_points), len(pair_df), pair_df)
        
        logger.info(f"PURE STRATEGY {self.target_pair} backtest completed!")
        return results
    
    def print_realistic_results(self, execution_points, total_bars, pair_df: pd.DataFrame):
        """Print pure strategy backtest results focused on strategy performance only"""
        total_trades = len(self.trades)
        if total_trades == 0:
            # Returning an empty dict when no trades, to prevent errors in autotuner.
            return {
                'final_balance': self.current_balance,
                'total_pnl': self.current_balance - self.initial_balance,
                'total_trades': 0,
                'win_rate': 0,
                'winning_trades': [],
                'losing_trades': [],
                'overall_rr_ratio': 0,
                'max_drawdown_percent': 0.0,
                'sharpe_ratio': 0.0
            }
        
        winning_trades = [t for t in self.trades if t.pips_gained > 0]
        win_rate = len(winning_trades) / total_trades * 100
        
        # Calculate overall Risk-Reward Ratio
        total_winning_pips = sum(t.pips_gained for t in winning_trades)
        losing_trades = [t for t in self.trades if t.pips_gained <= 0]
        total_losing_pips = sum(abs(t.pips_gained) for t in losing_trades)
        
        overall_rr = (total_winning_pips / len(winning_trades)) / (total_losing_pips / len(losing_trades)) if len(winning_trades) > 0 and len(losing_trades) > 0 else 0
        
        # Calculate common metrics before conditional branches
        total_pnl = self.current_balance - self.initial_balance
        max_drawdown_percent = self._calculate_max_drawdown(pair_df)
        
        # Initialize results dictionary
        results = {
            "final_balance": self.current_balance,
            "total_pnl": total_pnl,
            "total_trades": total_trades,
            "win_rate": win_rate,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "overall_rr_ratio": overall_rr,
            "max_drawdown_percent": max_drawdown_percent,
            "sharpe_ratio": 0.0 # Placeholder, not computed in pure strategy
        }
        
        # If autotuning, return results as a dictionary
        if self.is_autotuning:
            return results
        
        # If not autotuning, print detailed results
        # Construct the summary string
        summary_output = f"\n{self.target_pair} PURE STRATEGY BACKTEST RESULTS\n"
        summary_output += f"================================================================================\n"
        summary_output += f"Strategy: {self.strategy.__class__.__name__}\n"
        summary_output += f"\nPURE STRATEGY TESTING:\n"
        summary_output += f"   Processed Bars: {execution_points:,}\n"
        summary_output += f"   Total Available Bars: {total_bars:,}\n"
        summary_output += f"   Processing Rate: {execution_points/total_bars*100:.1f}% of bars\n"
        summary_output += f"   Trading Costs: ZERO (Pure strategy performance)\n"
        summary_output += f"\nTRADING RESULTS:\n"
        summary_output += f"   Total Trades: {results['total_trades']}\n"
        summary_output += f"   Win Rate: {results['win_rate']:.2f}%\n"
        summary_output += f"   Winning Trades:\n"
        for trade in results['winning_trades']:
            summary_output += f"     - {trade}\n"
        summary_output += f"   Losing Trades:\n"
        for trade in results['losing_trades']:
            summary_output += f"     - {trade}\n"
        summary_output += f"\nTRADING SUMMARY:\n"
        summary_output += f"   Starting Balance: ${self.initial_balance:,.2f}\n"
        summary_output += f"   Final Balance: ${results['final_balance']:,.2f}\n"
        summary_output += f"   Total PnL: ${results['total_pnl']:,.2f}\n"
        summary_output += f"   Peak Balance: ${self.peak_balance:,.2f}\n"
        summary_output += f"   Lowest Balance: ${self.lowest_balance:,.2f}\n"
        summary_output += f"   Max Drawdown: {results['max_drawdown_percent']:.2f}%\n"
        summary_output += f"   Overall R:R Ratio: {results['overall_rr_ratio']:.2f}:1\n"
        # summary_output += f"   Sharpe Ratio: {sharpe_ratio:.2f}\n" # Temporarily disabled, needs trades for std dev

        # Print to console
        print(summary_output)
        
        # Save to file
        try:
            with open(self.summary_log_file, "w") as f:
                f.write(summary_output)
            logger.info(f"Backtest summary saved to {self.summary_log_file}")
        except Exception as e:
            logger.error(f"Error saving backtest summary to file: {e}")
        
        return results
    
    
    
    # def print_final_results(self):
    #     """Print final backtest results (fallback for old method)"""
    #     total_trades = len(self.trades)
    #     if total_trades == 0:
    #         print(f"\n🚫 No trades executed for {self.target_pair}")
    #         return
    #     
    #     winning_trades = [t for t in self.trades if t.usd_pnl > 0]
    #     win_rate = len(winning_trades) / total_trades * 100
    #     total_pnl = sum(t.usd_pnl for t in self.trades)
    #     
    #     print(f"\n" + "="*80)
    #     print(f"🎯 {self.target_pair} BACKTEST RESULTS")
    #     print(f"="*80)
    #     print(f"🧠 Strategy: {self.strategy.__class__.__name__ if self.strategy else 'No Strategy'}")
    #     print(f"📊 Total Trades: {total_trades:,}")
    #     print(f"🎯 Win Rate: {win_rate:.2f}%")
    #     print(f"💰 Total P&L: ${total_pnl:+,.2f}")
    #     print(f"📈 Final Balance: ${self.current_balance:,.2f}")
    #     print(f"📊 Return: {((self.current_balance - self.initial_balance) / self.initial_balance * 100):+.2f}%")
    #     print(f"================================================================================")
    
    # def run_legacy_backtest(self):
    #     """
    #     🗄️ LEGACY METHOD: Old backtest for comparison (processes every bar)
    #     Use this to compare with realistic results
    #     """
    #     logger.info(f"🗄️ Running LEGACY {self.target_pair} backtest for comparison...")
    #     
    #     # Reset state for fair comparison
    #     self.current_balance = self.initial_balance
    #     self.trades = []
    #     self.open_trades = []
    #     
    #     # Create fresh strategy instance for legacy test
    #     legacy_strategies = {
    #         "EUR/USD": EURUSDSTRATEGY()
    #     }
    #     legacy_strategy = legacy_strategies.get(self.target_pair)
    #     
    #     if not legacy_strategy:
    #         logger.error(f"❌ No legacy strategy available for {self.target_pair}")
    #         return
    #     
    #     # Load data
    #     data = self.load_excel_data()
    #     if not data or self.target_pair not in data:
    #         logger.error(f"❌ No {self.target_pair} data loaded. Aborting legacy backtest.")
    #         return
    #     
    #     pair_df = data[self.target_pair]
    #     min_bars_for_strategy = 250
    #     
    #     logger.info(f"🗄️ Legacy processing {len(pair_df) - min_bars_for_strategy + 1:,} bars...")
    #     
    #     # OLD METHOD: Process every bar
    #     for i in range(min_bars_for_strategy - 1, len(pair_df)):
    #         current_bar = pair_df.iloc[i]
    #         timestamp = current_bar['timestamp']
    #         
    #         self.check_trade_exits(timestamp, current_bar, self.target_pair, i)
    #         
    #         has_open_trade = any(t.pair == self.target_pair for t in self.open_trades)
    #         
    #         if not has_open_trade:
    #             try:
    #                 # OLD METHOD: Create fresh data slice (destroys state)
    #                 start_idx = max(0, i - min_bars_for_strategy + 1)
    #                 strategy_data = pair_df.iloc[start_idx:i + 1].copy()
    #                 strategy_data.reset_index(drop=True, inplace=True)  # DESTROYS STATE!
    #                 
    #                 if len(strategy_data) >= min_bars_for_strategy:
    #                     signal = legacy_strategy.analyze_trade_signal(strategy_data, self.target_pair)
    #                     
    #                     if signal.get("decision", "NO TRADE").upper() != "NO TRADE":
    #                         self.open_trade(signal, timestamp, self.target_pair, current_bar['close'], i, pair_df)
    #                         
    #             except Exception as e:
    #                 logger.error(f"❌ Legacy strategy error at {timestamp}: {e}")
    #         
    #     # Close remaining trades
    #     if self.open_trades:
    #         final_bar = pair_df.iloc[-1]
    #         final_index = len(pair_df) - 1
    #         for trade in self.open_trades[:]:
    #             self.close_trade_with_candle_data(trade, final_bar['timestamp'], final_bar['close'], "End of legacy test", final_index)
    #     
    #     # Print legacy results
    #     total_trades = len(self.trades)
    #     if total_trades > 0:
    #         winning_trades = [t for t in self.trades if t.pips_gained > 0]
    #         win_rate = len(winning_trades) / total_trades * 100
    #         
    #         print(f"\n🗄️ LEGACY BACKTEST RESULTS:")
    #         print(f"   Total Trades: {total_trades:,}")
    #         print(f"   Win Rate: {win_rate:.2f}%")
    #         print(f"   Final Balance: ${self.current_balance:,.2f}")
    #     
    #     logger.info(f"🗄️ Legacy {self.target_pair} backtest completed!")
    #     return {
    #         'trades': total_trades,
    #         'win_rate': win_rate if total_trades > 0 else 0,
    #         'balance': self.current_balance
    #     }
    
    # def run_comparison_backtest(self):
    #     """
    #     🔀 COMPARISON: Run both realistic and legacy methods
    #     Shows the difference between old and new approaches
    #     """
    #     print(f"\n🔀 RUNNING COMPARISON BACKTEST FOR {self.target_pair}")
    #     print(f"="*80)
    #     
    #     # Run realistic first
    #     print(f"1️⃣ Running REALISTIC simulation...")
    #     self.run_backtest()
    #     realistic_results = {
    #         'trades': len(self.trades),
    #         'win_rate': len([t for t in self.trades if t.pips_gained > 0]) / len(self.trades) * 100 if self.trades else 0,
    #         'balance': self.current_balance
    #     }
    #     
    #     print(f"\n2️⃣ Running LEGACY simulation...")
    #     legacy_results = self.run_legacy_backtest()
    #     
    #     # Print comparison
    #     print(f"\n" + "="*80)
    #     print(f"🔀 REALISTIC vs LEGACY COMPARISON")
    #     print(f"="*80)
    #     print(f"📊 TRADE FREQUENCY:")
    #     print(f"   Realistic: {realistic_results['trades']:,} trades")
    #     print(f"   Legacy: {legacy_results['trades']:,} trades")
    #     if legacy_results['trades'] > 0:
    #         reduction = (1 - realistic_results['trades'] / legacy_results['trades']) * 100
    #         print(f"   Reduction: {reduction:.1f}% fewer trades (more realistic)")
    #     print(f"")
    #     print(f"🎯 WIN RATE:")
    #     print(f"   Realistic: {realistic_results['win_rate']:.2f}%")
    #     print(f"   Legacy: {legacy_results['win_rate']:.2f}%")
    #     print(f"   Difference: {realistic_results['win_rate'] - legacy_results['win_rate']:+.2f}%")
    #     print(f"")
    #     print(f"🔧 CRITICAL FIXES APPLIED:")
    #     print(f"   ✅ Matches real ctrader.py execution")
    #     print(f"   ✅ Proper strategy state persistence")
    #     print(f"   ✅ Correct timeframe data per pair")
    #     print(f"   ✅ Realistic execution timing per timeframe")
    #     print(f"   ✅ Accounts for delays and slippage")
    #     print(f"   ✅ Simulates timeout constraints")
    #     print(f"")
    #     print(f"📊 NATIVE TIMEFRAME OPTIMIZATION:")
    #     if self.target_pair == "EUR/USD":
    #         print(f"   ✅ EUR/USD: Native H1 data + hourly execution")
    #     else:
    #         print(f"   ✅ {self.target_pair}: Native H4 data + 4-hourly execution")
    #     print(f"   🎯 No aggregation needed - maximum broker accuracy!")
    #     print(f"   ⚡ Faster processing with smaller datasets")
    #     print(f"="*80)
    
    def verify_timeframe_setup(self):
        """
        🔍 VERIFICATION: Show the user exactly what timeframe setup is being used
        """
        pair_timeframes = {
            "EUR/USD": "M30",
            "GBP/USD": "M30", # Add GBP/USD
            "USD/JPY": "M30", # Add USD/JPY
            "EUR/GBP": "M30", # Add EUR/GBP
            "EUR/JPY": "M30" # Add EUR/JPY
        }
        
        expected_timeframe = pair_timeframes.get(self.target_pair, "M30")
        
        print(f"\nTIMEFRAME VERIFICATION FOR {self.target_pair}")
        print(f"="*60)
        print(f"Expected Strategy Timeframe: {expected_timeframe}")
        print(f"Backtest Data Timeframe: {expected_timeframe} (NATIVE)")
        print(f"Processing Schedule:")
        
        print(f"   PURE STRATEGY TESTING: All M30 bars processed")
        print(f"   Uses NATIVE 30-minute candles from broker API")
        print(f"   Zero trading costs for clean strategy evaluation")
        
        print(f"")
        print(f"PURE STRATEGY TESTING: Maximum strategy evaluation!")
        print(f"Clean testing environment for accurate strategy assessment!")
        print(f"="*60)

    def _calculate_max_drawdown(self, df: pd.DataFrame) -> float:
        """
        Calculate the maximum drawdown percentage from the equity curve.
        The drawdown is the percentage decline from a peak in the equity curve to a subsequent trough.
        """
        if not self.trades:
            return 0.0
        
        # Create a series of account balances after each trade
        balance_history = [self.initial_balance] + [trade.balance_after for trade in self.trades]
        balance_series = pd.Series(balance_history)
        
        # Calculate cumulative returns
        # If using balance, we need to convert to equity curve for drawdown calculation
        # Assuming balance_series represents the equity curve directly
        equity_curve = balance_series
        
        # Calculate the running maximum
        running_max = equity_curve.expanding().max()
        
        # Calculate the drawdown
        drawdown = (equity_curve - running_max) / running_max
        
        # Max drawdown is the smallest (most negative) drawdown
        max_drawdown = drawdown.min() * 100
        
        return max_drawdown

    def save_detailed_trade_data_to_excel(self):
        """
        Save detailed trade data, including candle information, to an Excel file.
        """
        if not self.detailed_trade_data:
            logger.info("No detailed trade data to save.")
            return

        logger.info(f"Saving detailed trade data to {self.trade_log_detailed_file}")
        
        # Flatten the dictionary for easier DataFrame conversion
        df_data = []
        for trade_info in self.detailed_trade_data:
            # Convert list of dicts for candle data to JSON string for single cell storage
            trade_info_copy = trade_info.copy()
            if 'candle_data' in trade_info_copy and trade_info_copy['candle_data']:
                for candle in trade_info_copy['candle_data']:
                    if 'timestamp' in candle and isinstance(candle['timestamp'], pd.Timestamp):
                        candle['timestamp'] = candle['timestamp'].isoformat()
            if 'post_entry_candles_500' in trade_info_copy and trade_info_copy['post_entry_candles_500']:
                for candle in trade_info_copy['post_entry_candles_500']:
                    if 'timestamp' in candle and isinstance(candle['timestamp'], pd.Timestamp):
                        candle['timestamp'] = candle['timestamp'].isoformat()
            
            trade_info_copy['candle_data'] = json.dumps(trade_info_copy['candle_data'])
            trade_info_copy['post_entry_candles_500'] = json.dumps(trade_info_copy['post_entry_candles_500'])
            df_data.append(trade_info_copy)

        df = pd.DataFrame(df_data)

        try:
            df.to_excel(self.trade_log_detailed_file, index=False)
            logger.info(f"Detailed trade data saved to {self.trade_log_detailed_file}")
        except Exception as e:
            logger.error(f"Error saving detailed trade data to Excel: {e}")

if __name__ == "__main__":
    try:
        # Define the one-year period for backtesting
        end_date_str = datetime.now().strftime("%Y-%m-%d")
        start_date_str = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")

        # Revert logging to default behavior (or simple console output if needed)
        # Remove any file handlers set previously in this session
        for handler in logging.root.handlers[:]:
            if isinstance(handler, logging.FileHandler):
                logging.root.removeHandler(handler)
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        logger = logging.getLogger(__name__)

        def execute_backtest_for_pair(strategy_instance, target_pair, start_date_str, end_date_str):
            engine = BacktestEngineM30(
                strategy=strategy_instance,
                target_pair=target_pair,
                start_balance=1000,
                is_autotuning=False,
                start_date=start_date_str,
                end_date=end_date_str
            )
            
            print(f"\n{'='*80}")
            print(f"STARTING SINGLE BACKTEST FOR {target_pair} with Supply and Demand Strategy")
            print(f"{'='*80}")

            engine.verify_timeframe_setup()
            results = engine.run_backtest()

            if results:
                print(f"\nFinal Balance for {target_pair}: ${results['final_balance']:,.2f}")
                print(f"Total PnL for {target_pair}: ${results['total_pnl']:,.2f}")
                print(f"Win Rate for {target_pair}: {results['win_rate']:.2f}%")

            print(f"\nBlocking Reasons Summary for {target_pair}:")
            blocking_counts = strategy_instance.get_blocking_reasons_counts()
            if blocking_counts:
                for reason, count in sorted(blocking_counts.items(), key=lambda item: item[1], reverse=True):
                    print(f"  - {reason}: {count} times")
            else:
                print(f"  No trades were blocked by specific strategy parameters for {target_pair}.")

            print(f"\n{'='*80}")
            print(f"SINGLE BACKTEST COMPLETED FOR {target_pair}")
            print(f"{'='*80}\n")
            return results

        # --- Run backtests for EUR/USD and GBP/USD ---
        # EUR/USD Backtest
        # eurusd_strategy = EURUSDSTRATEGY(target_pair="EUR/USD")
        # eurusd_results = execute_backtest_for_pair(eurusd_strategy, "EUR/USD", start_date_str, end_date_str)

        # GBP/USD Backtest
        # gbpusd_strategy = GBPUSDStrategy(target_pair="GBP/USD")
        # gbpusd_results = execute_backtest_for_pair(gbpusd_strategy, "GBP/USD", start_date_str, end_date_str)

        # USD/JPY Backtest
        usdjpy_strategy = USDJPYStrategy(target_pair="USD/JPY")
        usdjpy_results = execute_backtest_for_pair(usdjpy_strategy, "USD/JPY", start_date_str, end_date_str)

        # EUR/JPY Backtest
        eurjpy_strategy = EURJPYStrategy(target_pair="EUR/JPY")
        #eurjpy_results = execute_backtest_for_pair(eurjpy_strategy, "EUR/JPY", start_date_str, end_date_str)

        # EUR/GBP Backtest
        eurgbp_strategy = EURGBPStrategy(target_pair="EUR/GBP")
        #eurgbp_results = execute_backtest_for_pair(eurgbp_strategy, "EUR/GBP", start_date_str, end_date_str)

        # GBP/JPY Backtest
        gbpjpy_strategy = GBPJPYStrategy(target_pair="GBP/JPY")
        #gbpjpy_results = execute_backtest_for_pair(gbpjpy_strategy, "GBP/JPY", start_date_str, end_date_str)

    except KeyboardInterrupt:
        logger.info(f"Backtest interrupted by user")
    except Exception as e:
        logger.error(f"An error occurred during backtesting: {e}", exc_info=True)
    finally:
        pass