
# EUR/USD Candle Data Access Methods
# Generated on 2025-09-10 21:04:38.012794

import pickle
import pandas as pd
import json

def load_candle_data():
    """Load all candle data from pickle file"""
    with open('candle_data/EUR_USD_candle_data_20250910_210232.pkl', 'rb') as f:
        return pickle.load(f)

def get_trade_candles(trade_id):
    """Get candle data for specific trade"""
    data = load_candle_data()
    return data.get(trade_id, None)

def get_trade_as_dataframe(trade_id):
    """Get trade candle data as pandas DataFrame"""
    trade_data = get_trade_candles(trade_id)
    if trade_data:
        return pd.DataFrame(trade_data['candle_data'])
    return None

def list_all_trades():
    """List all available trades"""
    data = load_candle_data()
    trades = []
    for trade_id, trade_info in data.items():
        trades.append({
            'trade_id': trade_id,
            'pair': trade_info['pair'],
            'direction': trade_info['direction'],
            'pips_gained': trade_info['pips_gained'],
            'exit_reason': trade_info['exit_reason']
        })
    return trades

def get_winning_trades():
    """Get only winning trades"""
    data = load_candle_data()
    return {k: v for k, v in data.items() if v['pips_gained'] > 0}

def get_losing_trades():
    """Get only losing trades"""
    data = load_candle_data()
    return {k: v for k, v in data.items() if v['pips_gained'] <= 0}

# Example usage:
# data = load_candle_data()
# trade_1_df = get_trade_as_dataframe(1)
# all_trades = list_all_trades()
