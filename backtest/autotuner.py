import pandas as pd
from datetime import datetime, timedelta
from itertools import product
import logging
import sys
import os

# Add parent directory to sys.path to allow imports from `backtest` and `strategy`
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest.backtest_engine_m30 import BacktestEngineM30
from strategy.eurusd_strategy import EURUSDSTRATEGY

# Setup logging for autotuner (can be less verbose)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_autotuner():
    logger.info("Starting Autotuner for EUR/USD Supply & Demand Strategy (fixed 1:3 R:R)...")

    # Define parameter ranges for tuning (kept modest to finish quickly)
    # EXACTLY 10 combinations: 2 (lookbacks) × 5 (move ratios) × 1 × 1 × 1 × 1 × 1 = 10
    zone_lookbacks = [300, 350]                 # 2
    base_max_candles_list = [4]                 # 1
    move_min_ratios = [2.5, 2.75, 3.0, 3.25, 3.5]  # 5
    zone_width_max_pips_list = [18]             # 1
    sl_buffer_pips_list = [4.0]                 # 1
    use_trend_filter_options = [True]           # 1
    rsi_threshold_sets = [ (30.0, 70.0) ]       # 1 (oversold, overbought)

    # Total combinations
    total_combinations = (
        len(zone_lookbacks)
        * len(base_max_candles_list)
        * len(move_min_ratios)
        * len(zone_width_max_pips_list)
        * len(sl_buffer_pips_list)
        * len(use_trend_filter_options)
        * len(rsi_threshold_sets)
    )

    best_score = -float('inf')
    best_params = {}
    best_results = None
    all_results = []
    current_combination_num = 0

    end_date_str = datetime.now().strftime("%Y-%m-%d")
    start_date_str = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")

    for zone_lookback, base_max_candles, move_min_ratio, zone_width_max_pips, sl_buffer_pips, use_trend_filter, (rsi_oversold, rsi_overbought) in product(
        zone_lookbacks,
        base_max_candles_list,
        move_min_ratios,
        zone_width_max_pips_list,
        sl_buffer_pips_list,
        use_trend_filter_options,
        rsi_threshold_sets,
    ):
        current_combination_num += 1
        logger.info(
            f"Combo {current_combination_num}/{total_combinations}: "
            f"LB={zone_lookback}, base={base_max_candles}, move_ratio={move_min_ratio}, "
            f"zone_w={zone_width_max_pips}, sl_buf={sl_buffer_pips}, trend={'ON' if use_trend_filter else 'OFF'}, "
            f"RSI=({rsi_oversold},{rsi_overbought})"
        )

        ema_periods = [20, 50, 200] if use_trend_filter else []

        strategy_instance = EURUSDSTRATEGY(
            target_pair="EUR/USD",
            zone_lookback=zone_lookback,
            base_max_candles=base_max_candles,
            move_min_ratio=move_min_ratio,
            zone_width_max_pips=zone_width_max_pips,
            risk_reward_ratio=3.0,  # Enforce 1:3 R:R
            sl_buffer_pips=sl_buffer_pips,
            ema_periods=ema_periods,
            rsi_period=14,
            rsi_oversold=rsi_oversold,
            rsi_overbought=rsi_overbought,
            enable_volume_filter=False,  # volume often unavailable in excel
            min_volume_factor=1.2,
            session_hours_utc=("07:00-08:30", "10:30-11:00", "15:00-16:00"),
            enable_session_hours_filter=True,
            enable_news_sentiment_filter=False,
        )

        engine = BacktestEngineM30(
            strategy=strategy_instance,
            target_pair="EUR/USD",
            start_balance=1000,
            is_autotuning=True,
            start_date=start_date_str,
            end_date=end_date_str,
        )

        results = engine.run_backtest()

        if results:
            # Composite score: prioritize win rate, penalize many trades; require minimum trades
            win_rate = float(results.get('win_rate', 0.0) or 0.0)
            trades = int(results.get('total_trades', 0) or 0)
            total_pnl = float(results.get('total_pnl', 0.0) or 0.0)
            overall_rr = float(results.get('overall_rr_ratio', 0.0) or 0.0)

            # Hard constraints
            min_trades_required = 8
            if trades < min_trades_required:
                score = -1e9  # discard extremely low-sample configs even if high win rate
            else:
                # penalty weight per trade; scaled so 40 trades costs 6 pts
                trade_penalty = 0.15 * trades
                score = win_rate - trade_penalty

            all_results.append({
                'zone_lookback': zone_lookback,
                'base_max_candles': base_max_candles,
                'move_min_ratio': move_min_ratio,
                'zone_width_max_pips': zone_width_max_pips,
                'sl_buffer_pips': sl_buffer_pips,
                'use_trend_filter': use_trend_filter,
                'rsi_oversold': rsi_oversold,
                'rsi_overbought': rsi_overbought,
                'final_balance': results.get('final_balance', 1000.0),
                'total_pnl': total_pnl,
                'win_rate': win_rate,
                'total_trades': trades,
                'overall_rr_ratio': overall_rr,
                'max_drawdown_percent': results.get('max_drawdown_percent', 0.0),
                'score': score,
            })

            if score > best_score:
                best_score = score
                best_params = {
                    'zone_lookback': zone_lookback,
                    'base_max_candles': base_max_candles,
                    'move_min_ratio': move_min_ratio,
                    'zone_width_max_pips': zone_width_max_pips,
                    'sl_buffer_pips': sl_buffer_pips,
                    'use_trend_filter': use_trend_filter,
                    'rsi_oversold': rsi_oversold,
                    'rsi_overbought': rsi_overbought,
                }
                best_results = results

        # Reset engine state for next run
        engine.trades = []
        engine.open_trades = []
        engine.current_balance = engine.initial_balance
        engine.peak_balance = engine.initial_balance
        engine.lowest_balance = engine.initial_balance
        engine.detailed_trade_data = []

    logger.info("Autotuner completed!")

    if all_results:
        results_df = pd.DataFrame(all_results)
        results_df_sorted = results_df.sort_values(by='score', ascending=False).reset_index(drop=True)

        print(f"\nTOP 5 PARAMETER COMBINATIONS BY SCORE (win_rate - 0.15*trades):")
        for i, row in results_df_sorted.head(5).iterrows():
            print(f"\n--- Rank {i+1} ---")
            print(f"   Score: {row['score']:.2f}")
            print(f"   Win Rate: {row['win_rate']:.2f}% | Trades: {int(row['total_trades'])}")
            print(f"   Final Balance: ${row['final_balance']:,.2f} | PnL: ${row['total_pnl']:,.2f}")
            print(f"   Overall R:R (observed): {row['overall_rr_ratio']:.2f}:1 | Max DD: {row['max_drawdown_percent']:.2f}%")
            print(f"   Params -> LB={row['zone_lookback']}, base={row['base_max_candles']}, move_ratio={row['move_min_ratio']}, zone_w={row['zone_width_max_pips']}, sl_buf={row['sl_buffer_pips']}, trend={'ON' if row['use_trend_filter'] else 'OFF'}, RSI=({row['rsi_oversold']},{row['rsi_overbought']})")

        if best_params:
            print(f"\nBEST CONFIG (by score):\n   {best_params}")
            if best_results:
                print(f"   Win Rate: {best_results.get('win_rate', 0.0):.2f}% | Trades: {best_results.get('total_trades', 0)} | PnL: ${best_results.get('total_pnl', 0.0):,.2f}")
    else:
        print("\nNo combinations produced trades under current constraints.")

if __name__ == "__main__":
    run_autotuner() 