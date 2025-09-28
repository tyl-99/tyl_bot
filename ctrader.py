# Standard library imports
import datetime
import calendar
import json
import logging
import os
import re
import sys
import threading
import time

# Third-party imports
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from twisted.internet import reactor, defer
from twisted.internet.defer import TimeoutError

# cTrader API imports
from ctrader_open_api import Client, Protobuf, TcpProtocol, Auth, EndPoints
from ctrader_open_api.endpoints import EndPoints
from ctrader_open_api.messages.OpenApiCommonMessages_pb2 import *
from ctrader_open_api.messages.OpenApiMessages_pb2 import *
from ctrader_open_api.messages.OpenApiModelMessages_pb2 import *

# Strategy imports (optional, handle missing files gracefully)
try:
    from strategy.eurusd_strategy import EURUSDSTRATEGY
except ImportError:
    EURUSDSTRATEGY = None
try:
    from strategy.gbpusd_strategy import GBPUSDSTRATEGY
except ImportError:
    GBPUSDSTRATEGY = None
try:
    from strategy.eurgbp_strategy import EURGBPSTRATEGY
except ImportError:
    EURGBPSTRATEGY = None
try:
    from strategy.usdjpy_strategy import USDJPYSTRATEGY
except ImportError:
    USDJPYSTRATEGY = None
try:
    from strategy.gbpjpy_strategy import GBPJPYSTRATEGY
except ImportError:
    GBPJPYSTRATEGY = None
try:
    from strategy.eurjpy_strategy import EURJPYSTRATEGY
except ImportError:
    EURJPYSTRATEGY = None

# Forex symbols mapping with IDs
forex_symbols = {
    "EUR/USD": 1,
    "GBP/USD": 2,
    "EUR/JPY": 3,
    "EUR/GBP": 9,
    "USD/JPY": 4,
    "GBP/JPY": 7
}

# 🔄 DYNAMIC TIMEFRAME CONFIGURATION BY PAIR
PAIR_TIMEFRAMES = {
    "EUR/USD": "M30",
    "GBP/USD": "M30",
    "EUR/JPY": "M30",
    "EUR/GBP": "M30",
    "USD/JPY": "M30",
    "GBP/JPY": "M30"
}

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('forex_trading.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class Trader:
    def __init__(self):
        self.client_id = os.getenv("CTRADER_CLIENT_ID")
        self.client_secret = os.getenv("CTRADER_CLIENT_SECRET")
        self.account_id = int(os.getenv("CTRADER_ACCOUNT_ID"))
        
        self.host = EndPoints.PROTOBUF_DEMO_HOST
        self.client = Client(self.host, EndPoints.PROTOBUF_PORT, TcpProtocol)
        
        self.trendbarReq = None
        self.trendbar = None
        # To store pending order params
        self.pending_order = None
        self.action = None
        self.df = None

        self.pairIndex = 0
        
        # Load pairs from environment variable or use default
        pairs_str = os.getenv("TRADING_PAIRS")
        default_pairs = [
            {'from': 'USD', 'to': 'JPY'},
            {'from': 'EUR', 'to': 'USD'},
            {'from': 'EUR', 'to': 'JPY'},
            {'from': 'GBP', 'to': 'JPY'},
            {'from': 'GBP', 'to': 'USD'},
            {'from': 'EUR', 'to': 'GBP'},
        ]

        if pairs_str:
            print(f"✅ Loading trading pairs from environment: {pairs_str}")
            parsed_pairs = []
            for pair in pairs_str.split(','):
                currencies = pair.strip().split('/')
                if len(currencies) == 2:
                    parsed_pairs.append({'from': currencies[0], 'to': currencies[1]})
                else:
                    logger.warning(f"Invalid pair format in TRADING_PAIRS: '{pair}'. Skipping.")
            
            if parsed_pairs:
                self.pairs = parsed_pairs
            else:
                logger.error("No valid pairs found in TRADING_PAIRS. Falling back to default pairs.")
                self.pairs = default_pairs
        else:
            self.pairs = default_pairs

        self.current_pair = None

        self.active_order = []
        
        # Add pending orders tracking
        self.pending_orders = []  # Store pending limit orders
        
        # Add retry tracking
        self.retry_count = 0
        self.max_retries = 1  # Only retry once with volume/2
        self.original_trade_data = None
        
        # Add timeout and API retry tracking
        self.api_retry_count = 0
        self.max_api_retries = 3
        self.api_timeout = 15  # seconds
        self.request_delay = 2  # seconds between requests
        
        # Risk management - minimum R:R ratio filter (aligned with strategy default)
        self.min_rr_ratio = 2.0  # Centralized minimum R:R requirement
        
        # Track current position ID for closing if needed
        self.current_position_id = None
        # Track current position volume for accurate closing
        self.current_position_volume = None
        
        # Store closed deals list
        self.closed_deals_list = []

        # Initialize strategy instances for each available pair
        self.strategies = {}
        if EURUSDSTRATEGY:
            self.strategies["EUR/USD"] = EURUSDSTRATEGY()
        if GBPUSDSTRATEGY:
            self.strategies["GBP/USD"] = GBPUSDSTRATEGY()
        if EURGBPSTRATEGY:
            self.strategies["EUR/GBP"] = EURGBPSTRATEGY()
        if USDJPYSTRATEGY:
            self.strategies["USD/JPY"] = USDJPYSTRATEGY()
        if GBPJPYSTRATEGY:
            self.strategies["GBP/JPY"] = GBPJPYSTRATEGY()
        if EURJPYSTRATEGY:
            self.strategies["EUR/JPY"] = EURJPYSTRATEGY()

        # Filter configured pairs to only those with available strategies
        available_pairs = set(self.strategies.keys())
        self.pairs = [p for p in self.pairs if f"{p['from']}/{p['to']}" in available_pairs]
        if not self.pairs:
            logger.error("No available strategies for configured pairs. Exiting.")
            return

        self.connect()
        
    def onError(self, failure):
        """Enhanced error handler with timeout handling"""
        error_type = type(failure.value).__name__
        
        if "TimeoutError" in error_type:
            logger.warning(f"⏰ API timeout for {self.current_pair}. Retry {self.api_retry_count + 1}/{self.max_api_retries}")
            
            if self.api_retry_count < self.max_api_retries:
                self.api_retry_count += 1
                # Wait before retry
                reactor.callLater(self.request_delay * self.api_retry_count, self.retry_last_request)
                return
            else:
                logger.error(f"❌ Max API retries reached for {self.current_pair}. Skipping.")
                self.reset_api_retry_state()
                self.move_to_next_pair()
        else:
            print(f"Error: {failure}")
            # For non-timeout errors, also move to next pair
            self.reset_api_retry_state()
            self.move_to_next_pair()

    def retry_last_request(self):
        """Retry the last API request that timed out"""
        logger.info(f"🔄 Retrying API request for {self.current_pair}")
        
        # Add small delay to avoid overwhelming the API
        time.sleep(1)
        
        # Retry the trendbar request
        self.sendTrendbarReq(weeks=6, symbolId=self.current_pair)

    def reset_api_retry_state(self):
        """Reset API retry tracking"""
        self.api_retry_count = 0

    def get_pair_timeframe(self, pair_name):
        """Get optimal timeframe for specific pair"""
        timeframe = PAIR_TIMEFRAMES.get(pair_name, "M30")  # Default to M30
        logger.info(f"📊 {pair_name} using {timeframe} timeframe")
        return timeframe

    def connected(self, client):
        print("Connected to server.")
        self.authenticate_app()

    def authenticate_app(self):
        appAuth = ProtoOAApplicationAuthReq()
        appAuth.clientId = self.client_id
        appAuth.clientSecret = self.client_secret
        deferred = self.client.send(appAuth)
        deferred.addCallbacks(self.onAppAuthSuccess, self.onError)

    def onAppAuthSuccess(self, response):
        print("App authenticated.")
        accessToken = os.getenv("CTRADER_ACCESS_TOKEN")
        self.authenticate_user(accessToken)

    def authenticate_user(self, accessToken):
        userAuth = ProtoOAAccountAuthReq()
        userAuth.ctidTraderAccountId = self.account_id
        userAuth.accessToken = accessToken
        deferred = self.client.send(userAuth)
        deferred.addCallbacks(self.onUserAuthSuccess, self.onError)

    def disconnected(self, client, reason):
        print("Disconnected:", reason)

    def onMessageReceived(self, client, message):
        print("Message received:")
        #print(Protobuf.extract(message))

    def connect(self):

        self.client.setConnectedCallback(self.connected)
        self.client.setDisconnectedCallback(self.disconnected)
        self.client.setMessageReceivedCallback(self.onMessageReceived)

        self.client.startService()

        reactor.run()

    def onUserAuthSuccess(self, response):
        print("User authenticated.")
        self.getActivePosition()

    def sendOrderReq(self, symbol, trade_data):
        # Extract data from the trade_data object
        volume = round(float(trade_data.get("volume")), 2) * 100000
        # Ensure volume is multiple of 1000 (cTrader requirement)
        volume = round(volume / 1000) * 1000
        # Ensure minimum volume of 1000
        volume = max(volume, 1000)
        stop_loss = float(trade_data.get("stop_loss"))
        take_profit = float(trade_data.get("take_profit"))
        decision = trade_data.get("decision")
        
        # Store additional trade info for notifications/logging
        self.pending_order = {
            "symbol": symbol,
            "volume": volume,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "decision": decision,
            "entry_price": float(trade_data.get("entry_price", 0)),
            "reason": trade_data.get("reason", ""),
            "winrate": trade_data.get("winrate", ""),
            "risk_reward_ratio": trade_data.get("risk_reward_ratio", ""),
            "potential_loss_usd": trade_data.get("potential_loss_usd", ""),
            "potential_win_usd": trade_data.get("potential_win_usd", ""),
            "volume_calculation": trade_data.get("volume_calculation", ""),
            "loss_calculation": trade_data.get("loss_calculation", ""),
            "win_calculation": trade_data.get("win_calculation", "")
        }
        self.active_order.append(self.pending_order)
        symbol_id = forex_symbols.get(self.pending_order["symbol"])
        
        if symbol_id is not None:
            print(f"Placing limit order for {self.pending_order['symbol']} at {self.pending_order['entry_price']}")

            order = ProtoOANewOrderReq()
            order.ctidTraderAccountId = self.account_id
            order.symbolId = symbol_id
            order.volume = int(self.pending_order["volume"])*100

            # Change to LIMIT order instead of MARKET
            order.orderType = ProtoOAOrderType.MARKET
            
            # For market orders, limitPrice is not set. Entry price will be the actual market price.
            # We still set stop loss and take profit directly.
            # if 'JPY' in self.current_pair:
            #     order.limitPrice = round(float(self.pending_order["entry_price"]), 3)
            # else:
            #     order.limitPrice = round(float(self.pending_order["entry_price"]), 5)
            
            # No expiration - limit orders stay active until filled or manually cancelled
            # order.expirationTimestamp = expiration_timestamp  # Removed expiration
            
            if(self.pending_order["decision"] == "BUY"):
                order.tradeSide = ProtoOATradeSide.BUY
            elif(self.pending_order["decision"] =="SELL"):
                order.tradeSide = ProtoOATradeSide.SELL
            
            # Set stop loss and take profit directly in the order request
            if 'JPY' in self.current_pair:
                order.stopLoss = round(float(self.pending_order["stop_loss"]), 3)
                order.takeProfit = round(float(self.pending_order["take_profit"]), 3)
            else:
                order.stopLoss = round(float(self.pending_order["stop_loss"]), 5)
                order.takeProfit = round(float(self.pending_order["take_profit"]), 5)
            
            print(f"Placing {order.tradeSide} LIMIT order for symbol {order.symbolId} with volume {order.volume} at price {order.limitPrice}")
            print(f"Stop Loss: {order.stopLoss}, Take Profit: {order.takeProfit}")

            deferred = self.client.send(order)
            # Add timeout to order request
            deferred.addTimeout(self.api_timeout, reactor)
            deferred.addCallbacks(self.onOrderSent, self.onError)
        else:
            print(f"{self.pending_order['symbol']} symbol not found in the dictionary!")

    def onOrderSent(self, response):
        print("Limit order sent successfully!")
        message = Protobuf.extract(response)
        print(message)
        
        # Check if order was successful (errorCode exists but is empty string when successful)
        if hasattr(message, 'errorCode') and message.errorCode:
            description = getattr(message, 'description', 'No description available')
            print(f"❌ Order failed: {message.errorCode} - {description}")
            
            # Check for TRADING_BAD_STOPS error during order creation
            if message.errorCode == "TRADING_BAD_STOPS":
                print(f"🚨 TRADING_BAD_STOPS detected for {self.current_pair} during order creation!")
                logger.warning(f"TRADING_BAD_STOPS error for {self.current_pair} during order creation")
            
            self.move_to_next_pair()
            return
        
        if hasattr(message, 'position') and message.position:
            position_id = message.position.positionId
            position_volume = message.position.tradeData.volume
            
            # Store the current position ID and volume for potential closing
            self.current_position_id = position_id
            self.current_position_volume = position_volume
            
            print(f"✅ Position created with SL/TP - ID: {position_id}, Volume: {position_volume}")
            print(f"✅ Stop Loss: {self.pending_order['stop_loss']}, Take Profit: {self.pending_order['take_profit']}")
            
            # Send notification immediately since SL/TP are already set
            if os.getenv("sendNotification") == "true":
                self.send_pushover_notification()
            self.reset_retry_state()
            self.move_to_next_pair()
        else:
            print("❌ No position created in response")
            self.move_to_next_pair()

    def amend_sl_tp(self, position_id, stop_loss_price, take_profit_price):
        amend = ProtoOAAmendPositionSLTPReq()
        amend.ctidTraderAccountId = self.account_id
        amend.positionId = position_id
        
        # Round prices to appropriate precision (5 decimal places for most pairs, 3 for JPY pairs)
        if 'JPY' in self.current_pair:
            amend.stopLoss = round(float(stop_loss_price), 3)
            amend.takeProfit = round(float(take_profit_price), 3)
        else:
            amend.stopLoss = round(float(stop_loss_price), 5)
            amend.takeProfit = round(float(take_profit_price), 5)

        print(f"Setting SL {amend.stopLoss} and TP {amend.takeProfit} for position {position_id}")

        deferred = self.client.send(amend)
        # Add timeout to amend request
        deferred.addTimeout(self.api_timeout, reactor)
        deferred.addCallbacks(self.onAmendSent, self.onError)

    def close_position(self, position_id, volume=None):
        """Close a position completely or partially
        
        Args:
            position_id: The position ID to close
            volume: Volume to close in 0.01 units (e.g., 1000 = 10.00 units). If None, closes full position.
        """
        close_req = ProtoOAClosePositionReq()
        close_req.ctidTraderAccountId = self.account_id
        close_req.positionId = position_id
        
        # Use stored position volume if available, otherwise use provided volume
        if volume is None and self.current_position_volume:
            volume = self.current_position_volume  # Already in protocol format
        elif volume is None and self.pending_order:
            volume = int(self.pending_order["volume"]) * 100  # Convert to protocol format
        elif volume is None:
            # Default to a small volume if we don't have position info
            volume = 1000  # 10.00 units
        else:
            volume = int(volume)
        
        close_req.volume = volume
        
        print(f"🔄 Closing position {position_id} with volume {volume}")
        
        deferred = self.client.send(close_req)
        deferred.addTimeout(self.api_timeout, reactor)
        deferred.addCallbacks(self.onPositionClosed, self.onError)

    def onPositionClosed(self, response):
        """Handle position close response"""
        print("Position close request sent!")
        message = Protobuf.extract(response)
        print(message)
        
        # Check if close was successful
        if hasattr(message, 'errorCode') and message.errorCode:
            description = getattr(message, 'description', 'No description available')
            print(f"❌ Position close failed: {message.errorCode} - {description}")
        else:
            print("✅ Position closed successfully!")
            # Clear the current position ID and volume since it's now closed
            self.current_position_id = None
            self.current_position_volume = None
        
        # Reset retry state and move to next pair
        self.reset_retry_state()
        self.move_to_next_pair()

    def onAmendSent(self, response):
        message = Protobuf.extract(response)
        print(message)
        
        # Check if amendment was successful (errorCode exists but is empty string when successful)
        if hasattr(message, 'errorCode') and message.errorCode:
            description = getattr(message, 'description', 'No description available')
            print(f"❌ SL/TP amendment failed: {message.errorCode} - {description}")
            
            # Check for TRADING_BAD_STOPS error - close position immediately
            if message.errorCode == "TRADING_BAD_STOPS":
                print(f"🚨 TRADING_BAD_STOPS detected for {self.current_pair}. Closing position immediately!")
                logger.warning(f"TRADING_BAD_STOPS error for {self.current_pair} - closing position")
                
                if self.current_position_id:
                    self.close_position(self.current_position_id)
                    return  # Don't move to next pair yet, wait for close confirmation
                else:
                    print("❌ No position ID available to close")
                    self.reset_retry_state()
                    self.move_to_next_pair()
                    return
            
            # Check if it's a POSITION_NOT_FOUND error and we haven't retried yet
            elif message.errorCode == "POSITION_NOT_FOUND" and self.retry_count < self.max_retries:
                print(f"🔄 Retrying trade with volume/2 (attempt {self.retry_count + 1}/{self.max_retries})")
                self.retry_count += 1
                
                # Retry with volume/2
                if self.original_trade_data:
                    retry_trade_data = self.original_trade_data.copy()
                    retry_trade_data["volume"] = retry_trade_data["volume"] / 2
                    print(f"📉 Reducing volume from {self.original_trade_data['volume']:.2f} to {retry_trade_data['volume']:.2f} lots")
                    self.sendOrderReq(self.current_pair, retry_trade_data)
                    return
            else:
                if self.retry_count >= self.max_retries:
                    print(f"❌ Max retries ({self.max_retries}) reached for {self.current_pair}. Skipping.")
                self.reset_retry_state()
        else:
            print("✅ Amend SL/TP sent successfully!")
            # Send notification only after successful SL/TP setting
            if os.getenv("sendNotification") == "true":
                self.send_pushover_notification()
            self.reset_retry_state()
        
        self.move_to_next_pair()

    def reset_retry_state(self):
        """Reset retry tracking variables"""
        self.retry_count = 0
        self.original_trade_data = None
        self.current_position_id = None
        self.current_position_volume = None

    def sendTrendbarReq(self, weeks, symbolId):
        """Enhanced trendbar request with dynamic timeframe per pair"""
        # 🔄 Get pair-specific timeframe dynamically
        timeframe = self.get_pair_timeframe(symbolId)
        
        self.trendbarReq = (weeks, timeframe, symbolId)
        request = ProtoOAGetTrendbarsReq()
        request.ctidTraderAccountId = self.account_id
        request.period = ProtoOATrendbarPeriod.Value(timeframe)
        
        # 🔄 Adjust weeks based on timeframe for optimal data
        if timeframe == "H4":
            # H4 = 8x M30 candles in same time, so need more weeks for same analysis depth
            adjusted_weeks = weeks * 2  # Double weeks for H4 to get enough data
        elif timeframe == "H1":
            # H1 = 2x M30 candles, so slightly more weeks  
            adjusted_weeks = int(weeks * 1.5)
        else:  # M30, M15, M5, etc.
            adjusted_weeks = weeks
            
        if timeframe != "M1":
            request.fromTimestamp = int(calendar.timegm((datetime.datetime.utcnow() - datetime.timedelta(weeks=int(adjusted_weeks))).utctimetuple())) * 1000
            request.toTimestamp = int(calendar.timegm(datetime.datetime.utcnow().utctimetuple())) * 1000
        elif timeframe == "M1":
            request.fromTimestamp = int(calendar.timegm((datetime.datetime.utcnow() - datetime.timedelta(minutes=40)).utctimetuple())) * 1000
            request.toTimestamp = int(calendar.timegm(datetime.datetime.utcnow().utctimetuple())) * 1000
            
        request.symbolId = int(forex_symbols.get(symbolId))
        self.trendbarReq = None
        
        print(f"📊 Requesting {timeframe} data for {symbolId} ({adjusted_weeks} weeks)")
        
        deferred = self.client.send(request, clientMsgId=None)
        # Add timeout handling
        deferred.addTimeout(self.api_timeout, reactor)
        deferred.addCallbacks(self.onTrendbarDataReceived, self.onError)

    def onTrendbarDataReceived(self, response):
        """Enhanced trendbar data processing with validation"""
        print("Trendbar data received:")
        
        # Reset API retry count on successful response
        self.reset_api_retry_state()
        
        try:
            parsed = Protobuf.extract(response)
            trendbars = parsed.trendbar  # This is a list of trendbar objects
            
            if not trendbars:
                logger.warning(f"⚠️ No trendbar data received for {self.current_pair}")
                self.move_to_next_pair()
                return
            
            # Convert trendbars to DataFrame
            data = []
            for tb in trendbars:
                data.append({
                    'timestamp': datetime.datetime.utcfromtimestamp(tb.utcTimestampInMinutes * 60),
                    'open': (tb.low + tb.deltaOpen) / 1e5,
                    'high': (tb.low + tb.deltaHigh) / 1e5,
                    'low': tb.low / 1e5,
                    'close': (tb.low + tb.deltaClose) / 1e5,
                    'volume': tb.volume
                })
            
            df = pd.DataFrame(data)
            # Keep timestamp as datetime for strategy filters (trend/session)
            df.sort_values('timestamp', inplace=True, ascending=False)
            
            if self.trendbar.empty:
                # First call - store M30 data as base
                # Keep as datetime; strategy will handle conversion if needed
                self.trendbar = df
            # else:
            #     # Second call - aggregate M1 data into proper 30-minute candles
            #     df_30min = self.aggregate_1min_to_30min(df)
                
            #     # Combine with existing M30 data and remove duplicates
            #     df_30min['timestamp'] = df_30min['timestamp'].astype(str)
            #     self.trendbar = pd.concat([df_30min, self.trendbar], ignore_index=True)
                
            #     # Remove duplicates based on timestamp and sort
            #     self.trendbar = self.trendbar.drop_duplicates(subset=['timestamp'], keep='first')
            #     self.trendbar = self.trendbar.head(500)
            
            # if not self.latest_data:
            #     self.latest_data = True
            #     # Add delay before next request for M1 data
            #     reactor.callLater(self.request_delay, lambda: self.sendTrendbarReq(weeks=6, period="M1", symbolId=self.current_pair))
            #     return
            
            # Print last rows of trendbar data before sorting (for debugging)
            
            
            self.trendbar.sort_values('timestamp', inplace=True, ascending=True)
            print(f"\n📊 {self.current_pair} - Trendbar data after sorting (showing last 5 rows):")
            print(self.trendbar.tail().to_string())
            self.analyze_with_our_strategy()
            
        except Exception as e:
            logger.error(f"❌ Error processing trendbar data for {self.current_pair}: {e}")
            self.move_to_next_pair()

    def aggregate_1min_to_30min(self, df_1min):
        """Aggregate 1-minute candles into proper 30-minute candles aligned to 01-30 and 31-00"""
        try:
            # Ensure timestamp is datetime for grouping
            df_1min = df_1min.copy()
            df_1min['timestamp'] = pd.to_datetime(df_1min['timestamp'])
            df_1min.sort_values('timestamp', inplace=True, ascending=True)
            
            # Use all 40 minutes of M1 data - the complete period filter will handle the rest
            df_recent = df_1min.copy()
            
            print(f"📊 Using all M1 data: {len(df_recent)} candles (full 40min)")
            
            if df_recent.empty:
                logger.warning("⚠️ No recent M1 data after filtering")
                return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # Create custom 30-minute periods: 01-30 and 31-00
            def get_custom_period(timestamp):
                minute = timestamp.minute
                if 1 <= minute <= 30:
                    # Period: XX:01 to XX:30, label as XX:30
                    return timestamp.replace(minute=30, second=0, microsecond=0)
                else:  # 31-59 and 00
                    # Period: XX:31 to (XX+1):00, label as (XX+1):00
                    if minute == 0:
                        # 00 minute belongs to previous period (XX-1):31 to XX:00
                        return timestamp.replace(minute=0, second=0, microsecond=0)
                    else:
                        # 31-59 minutes: XX:31 to (XX+1):00
                        next_hour = timestamp + pd.Timedelta(hours=1)
                        return next_hour.replace(minute=0, second=0, microsecond=0)
            
            df_recent['period'] = df_recent['timestamp'].apply(get_custom_period)
            
            # Group by custom 30-minute periods and get period info
            period_groups = df_recent.groupby('period')
            
            # Create aggregated data with period counts
            aggregated_data = []
            for period, group in period_groups:
                period_data = {
                    'timestamp': period,
                    'open': group['open'].iloc[0],
                    'high': group['high'].max(),
                    'low': group['low'].min(),
                    'close': group['close'].iloc[-1],
                    'volume': group['volume'].sum(),
                    'candle_count': len(group)  # Track how many 1-min candles in this period
                }
                aggregated_data.append(period_data)
            
            aggregated = pd.DataFrame(aggregated_data)
            
            if aggregated.empty:
                return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # Sort by timestamp (newest first)
            aggregated.sort_values('timestamp', inplace=True, ascending=False)
            
            # Filter to only complete periods (should have close to 30 candles)
            # Allow some flexibility for missing candles (>=25 candles = complete enough)
            complete_periods = aggregated[aggregated['candle_count'] >= 25].copy()
            
            if complete_periods.empty:
                # If no complete periods, take the period with most candles
                complete_periods = aggregated.head(1)
                print(f"⚠️ No complete 30-min periods found, using period with {aggregated.iloc[0]['candle_count']} candles")
            else:
                print(f"✅ Found {len(complete_periods)} complete 30-min periods")
            
            # Take the most recent complete period
            result = complete_periods.head(1).copy()
            
            # Remove the helper column
            result = result.drop(columns=['candle_count'])
            
            if not result.empty:
                period_end = result.iloc[0]['timestamp']
                print(f"✅ Selected complete 30-min candle ending at: {period_end}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error aggregating 1-min to 30-min data: {e}")
            # Return empty DataFrame on error
            return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

    def analyze_with_our_strategy(self):
        """Analyze market data using our optimized supply/demand strategies"""
        try:
            # Get the appropriate strategy for this pair
            strategy = self.strategies.get(self.current_pair)
            if not strategy:
                logger.warning(f"No strategy found for {self.current_pair}")
                self.move_to_next_pair()
                return
            
            # Analyze the market data
            signal = strategy.analyze_trade_signal(self.trendbar, self.current_pair)
            
            logger.info(f"\n=== Strategy Decision for {self.current_pair} ===")
            logger.info(f"Decision: {signal.get('decision') or signal.get('action')}")
            
            # Check for both 'decision' and 'action' keys to handle different response formats
            if signal.get("decision") == "NO TRADE" or signal.get("action") == "HOLD" or signal.get("action") == "NONE":
                logger.info(f"No trade signal for {self.current_pair}: {signal.get('reason', 'No reason provided')}")
                self.move_to_next_pair()
            else:
                # CENTRALIZED R:R FILTER - Check R:R ratio before executing trade
                entry_price = signal['entry_price']
                stop_loss = signal['stop_loss']
                take_profit = signal['take_profit']
                
                # Calculate R:R using direct price distances (simpler than pip conversion)
                risk_distance = abs(entry_price - stop_loss)
                reward_distance = abs(take_profit - entry_price)
                
                rr_ratio = reward_distance / risk_distance if risk_distance > 0 else 0
                
                # Pip calculations for logging and sizing only (min-pips gate removed)
                pip_size = 0.01 if 'JPY' in self.current_pair else 0.0001
                risk_pips = risk_distance / pip_size
                
                if rr_ratio < self.min_rr_ratio:
                    reward_pips = reward_distance / pip_size  # Calculate for logging only
                    logger.info(f"❌ Trade REJECTED for {self.current_pair}: R:R {rr_ratio:.2f} < {self.min_rr_ratio}")
                    logger.info(f"   Risk: {risk_pips:.1f} pips | Reward: {reward_pips:.1f} pips")
                    print(f"⚠️ {self.current_pair}: R:R {rr_ratio:.2f} too low, minimum required: {self.min_rr_ratio}")
                    self.move_to_next_pair()
                    return
                
                logger.info(f"ℹ️ Stop Loss distance: {risk_pips:.1f} pips (ATR/minDistance handled in strategy)")
                logger.info(f"✅ R:R Check PASSED: {rr_ratio:.2f} ≥ {self.min_rr_ratio}")
                
                # Convert our strategy signal to the format expected by sendOrderReq
                trade_data = self.format_trade_data(signal)
                
                # Store original trade data for potential retry
                if self.retry_count == 0:
                    self.original_trade_data = trade_data.copy()
                
                logger.info(f"Trade signal: {trade_data}")
                self.sendOrderReq(self.current_pair, trade_data)
                
        except Exception as e:
            logger.error(f"Error analyzing {self.current_pair}: {str(e)}")
            self.reset_retry_state()
            self.move_to_next_pair()
    
    def get_trade_reason(self, signal):
        """Safely extract trade reason from signal"""
        try:
            if 'meta' in signal and signal['meta']:
                meta = signal['meta']
                zone_type = meta.get('zone_type', 'Unknown')
                zone_low = meta.get('zone_low', 0)
                zone_high = meta.get('zone_high', 0)
                
                if zone_low and zone_high:
                    return f"Supply/Demand zone: {zone_type} zone at {zone_low:.5f}-{zone_high:.5f}"
                else:
                    return f"Supply/Demand zone: {zone_type} zone"
            else:
                return f"Supply/Demand strategy setup for {self.current_pair}"
        except Exception as e:
            logger.warning(f"Error extracting trade reason: {e}")
            return f"Technical analysis setup for {self.current_pair}"
    
    def format_trade_data(self, signal):
        """Convert our strategy signal to ctrader format"""
        entry_price = signal['entry_price']
        stop_loss = signal['stop_loss']
        take_profit = signal['take_profit']
        
        # Calculate risk in pips for volume calculation
        pip_size = 0.01 if 'JPY' in self.current_pair else 0.0001
        risk_pips = abs(entry_price - stop_loss) / pip_size
        
        # Target $50 risk per trade (as requested by user)
        target_risk_usd = 100.0
        
        # More accurate pip values for different pairs
        if 'JPY' in self.current_pair:
            if self.current_pair == 'USD/JPY':
                pip_value = 10.0  # USD/JPY: $10 per pip for 1 lot
            else:  # EUR/JPY, GBP/JPY
                pip_value = 7.0   # Cross JPY pairs: ~$7 per pip for 1 lot
        else:
            pip_value = 10.0  # Major pairs: $10 per pip for 1 lot
        
        volume_lots = target_risk_usd / (risk_pips * pip_value)
        volume_lots = max(0.01, min(volume_lots, 2.0))  # Clamp between 0.01 and 2.0 lots
        
        # Calculate R:R using direct price distances (simpler and cleaner)
        risk_distance = abs(entry_price - stop_loss)
        reward_distance = abs(take_profit - entry_price)
        rr_ratio = reward_distance / risk_distance if risk_distance > 0 else 0
        
        # Calculate potential P&L (still need pips for dollar calculations)
        reward_pips = reward_distance / pip_size
        potential_loss = risk_pips * pip_value * volume_lots
        potential_win = reward_pips * pip_value * volume_lots
        
        return {
            "decision": signal.get('decision', signal.get('action', 'UNKNOWN')),
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "volume": volume_lots,
            "reason": self.get_trade_reason(signal),
            "risk_reward_ratio": f"{rr_ratio:.2f}",
            "potential_loss_usd": f"${potential_loss:.2f}",
            "potential_win_usd": f"${potential_win:.2f}",
            "winrate": "55%+",  # Based on our backtest results
            "volume_calculation": f"{volume_lots:.2f} lots for ${target_risk_usd} risk",
            "loss_calculation": f"{risk_pips:.1f} pips × ${pip_value:.1f}/pip × {volume_lots:.2f} lots",
            "win_calculation": f"{reward_pips:.1f} pips × ${pip_value:.1f}/pip × {volume_lots:.2f} lots"
        }
    
    def move_to_next_pair(self):
        """Move to the next trading pair or stop if all pairs are done"""
        # Reset retry state when moving to next pair
        self.reset_retry_state()
        self.reset_api_retry_state()
        
        if self.pairIndex < len(self.pairs) - 1:
            self.pairIndex += 1
            # Add delay before processing next pair
            reactor.callLater(self.request_delay, lambda: self.run_trading_cycle(self.pairs[self.pairIndex]))
        else:
            print("All trading pairs analyzed.")
            reactor.stop()
    
    def send_pushover_notification(self):
        APP_TOKEN = "ah7dehvsrm6j3pmwg9se5h7svwj333"
        USER_KEY = "u4ipwwnphbcs2j8iiosg3gqvompfs2"

        # Create organized message with all trade details
        message = self.format_trade_notification()

        payload = {
            "token": APP_TOKEN,
            "user": USER_KEY,
            "message": message,
            "title": f"🚀 {self.pending_order['decision']} Trade Alert - {self.pending_order['symbol']}",
            "priority": 1,  # High priority for trade notifications
            "sound": "cashregister"  # Custom sound for trade alerts
        }

        try:
            response = requests.post("https://api.pushover.net/1/messages.json", data=payload)
            if response.status_code == 200:
                print("📲 Enhanced Pushover notification sent successfully.")
                logger.info(f"Trade notification sent for {self.pending_order['symbol']}")
            else:
                print(f"❌ Failed to send notification. Status: {response.status_code}, Error: {response.text}")
                logger.error(f"Pushover notification failed: {response.text}")
        except Exception as e:
            print(f"❌ Error sending Pushover notification: {e}")
            logger.error(f"Pushover notification error: {str(e)}")

    def format_trade_notification(self):
        """Format comprehensive trade notification message"""
        
        # Header with trade action and pair
        message_parts = [
            f"🎯 {self.pending_order['decision']} TRADE EXECUTED",
            f"💱 Pair: {self.pending_order['symbol']}",
            "",
            "📊 TRADE DETAILS:",
            f"Entry: ${self.pending_order.get('entry_price', 'Market Price'):.5f}",
            f"Stop Loss: ${self.pending_order['stop_loss']:.5f}",
            f"Take Profit: ${self.pending_order['take_profit']:.5f}",
            f"Volume: {self.pending_order['volume'] / 100000:.2f} lots",
            "",
            "📈 RISK ANALYSIS:",
            f"R:R Ratio: {self.pending_order.get('risk_reward_ratio', 'N/A')}",
            f"Max Risk: {self.pending_order.get('potential_loss_usd', '$50.00')}",
            f"Potential Win: {self.pending_order.get('potential_win_usd', 'N/A')}",
            f"Confidence: {self.pending_order.get('winrate', 'N/A')}",
            "",
            "💡 STRATEGY REASON:",
            f"{self.pending_order.get('reason', 'Technical analysis setup')[:100]}{'...' if len(self.pending_order.get('reason', '')) > 100 else ''}",
            "",
            "🔢 CALCULATIONS:",
            f"Volume: {self.pending_order.get('volume_calculation', 'Risk-based sizing')}",
            f"Loss: {self.pending_order.get('loss_calculation', 'SL distance calculation')}",
            f"Win: {self.pending_order.get('win_calculation', 'TP distance calculation')}",
            "",
            f"⏰ {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC"
        ]
        
        return "\n".join(message_parts)

    def format_compact_notification(self):
        """Alternative compact version for shorter notifications"""
        
        compact_message = (
            f"🚀 {self.pending_order['decision']} {self.pending_order['symbol']}\n"
            f"📊 Entry: ${self.pending_order.get('entry_price', 0):.5f}\n"
            f"🛑 SL: ${self.pending_order['stop_loss']:.5f} | 🎯 TP: ${self.pending_order['take_profit']:.5f}\n"
            f"📈 R:R: {self.pending_order.get('risk_reward_ratio', 'N/A')} | 🎲 Conf: {self.pending_order.get('winrate', 'N/A')}\n"
            f"💰 Risk: {self.pending_order.get('potential_loss_usd', '$50.00')} | Win: {self.pending_order.get('potential_win_usd', 'N/A')}\n"
            f"💡 {self.pending_order.get('reason', '')[:80]}{'...' if len(self.pending_order.get('reason', '')) > 80 else ''}\n"
            f"⏰ {datetime.datetime.now().strftime('%H:%M:%S')}"
        )
        
        return compact_message
    
    def getActivePosition(self):
        req = ProtoOAReconcileReq()
        req.ctidTraderAccountId = self.account_id
        deferred = self.client.send(req)
        # Add timeout to reconcile request
        deferred.addTimeout(self.api_timeout, reactor)
        deferred.addCallbacks(self.onActivePositionReceived, self.onError)
                
    def onActivePositionReceived(self, response):
        parsed = Protobuf.extract(response)
        
        # Process active positions
        positions = parsed.position if hasattr(parsed, 'position') else []
        self.active_positions = []

        for pos in positions:
            self.active_positions.append({
                "positionId": pos.positionId,
                "symbolId": pos.tradeData.symbolId,
                "side": "BUY" if pos.tradeData.tradeSide == 1 else "SELL",
                "volume": pos.tradeData.volume,
                "openPrice": pos.price,
                "stopLoss": pos.stopLoss,
                "takeProfit": pos.takeProfit,
            })

        # Process pending orders
        orders = parsed.order if hasattr(parsed, 'order') else []
        self.pending_orders = []

        for order in orders:
            self.pending_orders.append({
                "orderId": order.orderId,
                "symbolId": order.tradeData.symbolId,
                "side": "BUY" if order.tradeData.tradeSide == 1 else "SELL",
                "volume": order.tradeData.volume,
                "orderType": order.orderType,
                "limitPrice": getattr(order, 'limitPrice', None),
                "stopPrice": getattr(order, 'stopPrice', None),
                "expirationTimestamp": getattr(order, 'expirationTimestamp', None)
            })

        print(f"📊 Found {len(self.active_positions)} active positions and {len(self.pending_orders)} pending orders")
        
        self.pairIndex = 0
        self.run_trading_cycle(self.pairs[self.pairIndex])

    def get_symbol_list(self):
        req = ProtoOASymbolsListReq()
        req.ctidTraderAccountId = self.account_id
        req.includeArchivedSymbols = True
        deferred = self.client.send(req)
        deferred.addCallbacks(self.onSymbolsReceived, self.onError)

    def onSymbolsReceived(self, message):
        print("Message received:")
        try:
            parsed_message = Protobuf.extract(message)
            
            with open("symbols.txt", "w") as f:
                for symbol in parsed_message.symbol:
                    line = f"{symbol.symbolName} (ID: {symbol.symbolId})\n"
                    print(line.strip())  # print to console
                    f.write(line)        # write to file

        except Exception as e:
            print(f"Failed to parse message: {e}")
            print(message)

    def is_symbol_active(self, symbol_id):
        return any(pos["symbolId"] == symbol_id for pos in self.active_positions)
    
    def has_pending_order(self, symbol_id):
        """Check if there's a pending limit order for the given symbol"""
        return any(order["symbolId"] == symbol_id for order in self.pending_orders)
    
    def get_pending_order_details(self, symbol_id):
        """Get details of pending orders for the given symbol"""
        symbol_orders = [order for order in self.pending_orders if order["symbolId"] == symbol_id]
        return symbol_orders

    def get_deals_from_current_week(self):
        """Get all deals from the current week before starting trendbar collection"""
        try:
            # Calculate current week's start and end timestamps
            now = datetime.datetime.now()
            start_of_week = now - datetime.timedelta(days=now.weekday())  # Current week
            start_of_week = start_of_week.replace(hour=0, minute=0, second=0, microsecond=0)
            end_of_week = start_of_week + datetime.timedelta(days=7)
            
            # Convert to Unix timestamps
            from_timestamp = int(start_of_week.timestamp() * 1000)  # Convert to milliseconds
            to_timestamp = int(end_of_week.timestamp() * 1000)
            
            print(f"📊 Fetching deals from current week: {start_of_week.strftime('%Y-%m-%d')} to {end_of_week.strftime('%Y-%m-%d')}")
            
            # Create ProtoOADealListReq
            deal_req = ProtoOADealListReq()
            deal_req.ctidTraderAccountId = self.account_id
            deal_req.fromTimestamp = from_timestamp
            deal_req.toTimestamp = to_timestamp
            deal_req.maxRows = 1000  # Get up to 1000 deals
            
            # Send the request
            deferred = self.client.send(deal_req)
            deferred.addCallbacks(self.onDealsReceived, self.onError)
            
        except Exception as e:
            logger.error(f"Error creating deal request: {str(e)}")
            # Continue with trendbar collection even if deal request fails
            self.sendTrendbarReq(weeks=6, symbolId=self.current_pair)

    def onDealsReceived(self, response):
        """Handle the response from ProtoOADealListRes"""
        try:
            parsed = Protobuf.extract(response)
            
            # print(f"\n📈 DEALS FROM CURRENT WEEK (CLOSED POSITIONS ONLY):")
            # print("=" * 80)
            
            if hasattr(parsed, 'deal') and parsed.deal:
                closed_deals = []
                total_gross_profit = 0
                closed_trades = 0
                
                for deal in parsed.deal:
                    # Only process deals that have closePositionDetail (closed positions)
                    if hasattr(deal, 'closePositionDetail') and deal.closePositionDetail:
                        # Get gross profit from closePositionDetail
                        gross_profit = deal.closePositionDetail.grossProfit
                        
                        # Only include deals with actual profit/loss (not $0.00)
                        if gross_profit != 0:
                            # Convert timestamp from milliseconds to UTC datetime for consistency
                            deal_time = datetime.datetime.utcfromtimestamp(deal.executionTimestamp / 1000)
                            
                            # Get symbol name from symbol ID
                            symbol_name = "Unknown"
                            for symbol_id, name in forex_symbols.items():
                                if name == deal.symbolId:
                                    symbol_name = symbol_id
                                    break
                            
                            # Store deal info in list
                            closed_deals.append({
                                'timestamp': deal_time,
                                'symbol_id': deal.symbolId,
                                'gross_profit': gross_profit
                            })
                            
                            # # Format the deal information
                            # print(f"🕐 {deal_time.strftime('%Y-%m-%d %H:%M:%S')} | "
                            #       f"💱 {symbol_name} (ID: {deal.symbolId}) | "
                            #       f"📈 Gross Profit: ${gross_profit:.2f}")
                            
                            # Update totals
                            total_gross_profit += gross_profit
                            closed_trades += 1
                
                
                # Store the closed deals list for potential future use
                self.closed_deals_list = closed_deals
                
            else:
                print("📭 No deals found for the current week.")
                self.closed_deals_list = []
            
            print(f"\n🚀 Starting trendbar data collection for {self.current_pair}...")
            
            # Continue with trendbar collection
            self.sendTrendbarReq(weeks=6, symbolId=self.current_pair)
            
        except Exception as e:
            logger.error(f"Error processing deals response: {str(e)}")
            # Continue with trendbar collection even if deal processing fails
            self.sendTrendbarReq(weeks=6, symbolId=self.current_pair)

    def check_recent_loss_trade(self, pair_name):
        """Check if there's a loss trade in the last 12 hours for the given pair"""
        try:
            if not hasattr(self, 'closed_deals_list') or not self.closed_deals_list:
                logger.info(f"🔍 {pair_name}: No closed deals list available")
                return False  # No deals to check
            
            # Use UTC time for consistency with cTrader server
            now = datetime.datetime.utcnow()
            twelve_hours_ago = now - datetime.timedelta(hours=12)
            
            logger.info(f"🔍 {pair_name}: Checking for loss trades between {twelve_hours_ago.strftime('%Y-%m-%d %H:%M:%S')} UTC and {now.strftime('%Y-%m-%d %H:%M:%S')} UTC")
            
            # Get symbol ID for the pair
            symbol_id = forex_symbols.get(pair_name)
            if symbol_id is None:
                logger.warning(f"🔍 {pair_name}: Symbol ID not found in forex_symbols")
                return False  # Pair not found
            
            # Track deals for this pair for debugging
            pair_deals = []
            loss_deals = []
            recent_loss_deals = []
            
            # Check for loss trades in the last 12 hours
            for deal in self.closed_deals_list:
                # Track all deals for this pair
                if deal['symbol_id'] == symbol_id:
                    pair_deals.append(deal)
                    
                    # Track loss deals for this pair
                    if deal['gross_profit'] < 0:
                        loss_deals.append(deal)
                        
                        # Check if within 12 hours (convert deal timestamp to UTC for comparison)
                        if deal['timestamp'] >= twelve_hours_ago:
                            recent_loss_deals.append(deal)
                            
                            print(f"🚫 {pair_name} has a loss trade in the last 12 hours!")
                            print(f"   Loss: ${deal['gross_profit']:.2f} at {deal['timestamp'].strftime('%Y-%m-%d %H:%M:%S')} UTC")
                            print(f"   Time difference: {(now - deal['timestamp']).total_seconds() / 3600:.2f} hours ago")
                            logger.info(f"12-hour check BLOCKED {pair_name}: Loss ${deal['gross_profit']:.2f} at {deal['timestamp']}")
                            return True
            
            # Enhanced logging for debugging
            logger.info(f"🔍 {pair_name}: Found {len(pair_deals)} total deals, {len(loss_deals)} loss deals, {len(recent_loss_deals)} recent loss deals")
            
            if pair_deals:
                latest_deal = max(pair_deals, key=lambda x: x['timestamp'])
                logger.info(f"🔍 {pair_name}: Latest deal was {(now - latest_deal['timestamp']).total_seconds() / 3600:.2f} hours ago (${latest_deal['gross_profit']:.2f})")
            
            logger.info(f"✅ {pair_name}: 12-hour check PASSED - No recent loss trades")
            return False
            
        except Exception as e:
            logger.error(f"Error checking recent loss trade for {pair_name}: {str(e)}")
            return False

    def run_trading_cycle(self, pair):
    
        try:
            from_curr = pair['from']
            to_curr = pair['to']
            pair_name = f"{from_curr}/{to_curr}"
            logger.info(f"Processing {pair_name}")

            symbol_id = forex_symbols.get(pair_name)
            self.latest_data  = False
            self.trendbar = pd.DataFrame()

            # Check for both active positions and pending orders
            if self.is_symbol_active(symbol_id):
                print(f"⚠️ {pair_name} is currently Active (has open position)!")
                self.move_to_next_pair()
            elif self.has_pending_order(symbol_id):
                # Get details of pending orders for better logging
                pending_details = self.get_pending_order_details(symbol_id)
                for order in pending_details:
                    order_type = order.get('orderType', 'Unknown')
                    side = order.get('side', 'Unknown')
                    volume = order.get('volume', 0) / 10000  # Convert to lots
                    limit_price = order.get('limitPrice', 'N/A')
                    print(f"⚠️ {pair_name} has pending {order_type} {side} order: {volume:.2f} lots @ {limit_price}")
                logger.info(f"Skipping {pair_name} - has {len(pending_details)} pending order(s)")
                self.move_to_next_pair()
            else:
                self.current_pair = pair_name
            
                # Directly request trendbars for decision-making
                self.sendTrendbarReq(weeks=6, symbolId=pair_name)
                # #self.getActivePosition()
                # #self.get_symbol_list()

        except Exception as e:
            logger.error(f"Error processing {pair_name}: {str(e)}")

if __name__ == "__main__":
    print("🚀 Starting cTrader Live Trading Bot...")
    print("=" * 50)
    
    try:
        load_dotenv()
        
        def force_exit():
            print("\n⏰ Program exceeded 5 minutes. Exiting safely.")
            reactor.stop()
            sys.exit(0)

        timer = threading.Timer(300, force_exit)  # 300 seconds = 5 minutes
        timer.start()

        print("📡 Connecting to cTrader...")
        trader = Trader()
        
    except KeyboardInterrupt:
        print("\n🛑 Manual stop requested. Exiting...")
        reactor.stop()
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        reactor.stop()
        sys.exit(1)
    

