import pandas as pd
import datetime
import calendar
import os
import logging
from twisted.internet import reactor
from ctrader_open_api import Client, Protobuf, TcpProtocol, EndPoints
from ctrader_open_api.messages.OpenApiCommonMessages_pb2 import *
from ctrader_open_api.messages.OpenApiMessages_pb2 import *
from ctrader_open_api.messages.OpenApiModelMessages_pb2 import *
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('backtest/data_fetch.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 🔄 UPDATED: Match ctrader.py forex symbols exactly
forex_symbols = {
    "EUR/USD": 1,
    "GBP/USD": 2,
    "EUR/JPY": 3,
    "EUR/GBP": 9,
    "USD/JPY": 4,
    "GBP/JPY": 7
}

# 🎯 PAIR TIMEFRAME CONFIGURATION (H4 for Fibonacci strategies)
PAIR_TIMEFRAMES = {
    "EUR/USD": "M30",   # M30 - default for all pairs
    "GBP/USD": "M30",   # M30 - default for all pairs
    "EUR/JPY": "M30",   # M30 - default for all pairs
    "EUR/GBP": "M30",   # M30 - default for all pairs
    "USD/JPY": "M30",   # M30 - default for all pairs
    "GBP/JPY": "M30"    # M30 - default for all pairs
}

class DataFetcher:
    def __init__(self):
        self.client_id = os.getenv("CTRADER_CLIENT_ID")
        self.client_secret = os.getenv("CTRADER_CLIENT_SECRET")
        self.account_id = int(os.getenv("CTRADER_ACCOUNT_ID"))
        self.access_token = os.getenv("CTRADER_ACCESS_TOKEN")
        
        self.host = EndPoints.PROTOBUF_DEMO_HOST
        self.client = Client(self.host, EndPoints.PROTOBUF_PORT, TcpProtocol)
        
        self.pairs_to_fetch = []
        self.current_pair_index = 0
        self.start_date = None
        self.end_date = None
        self.all_data = {}
        
        # 🎯 ENHANCED: Add data quality tracking
        self.fetch_stats = {
            'total_pairs': 0,
            'successful_pairs': 0,
            'failed_pairs': 0,
            'total_bars': 0
        }
    
    def get_pair_timeframe(self, pair_name):
        """Get optimal timeframe for specific pair (matches ctrader.py)"""
        timeframe = PAIR_TIMEFRAMES.get(pair_name, "M30")  # Default to M30
        logger.info(f"📊 {pair_name} using {timeframe} timeframe")
        return timeframe
        
    def fetch_data(self, pairs, start_date, end_date):
        self.pairs_to_fetch = pairs
        self.start_date = start_date
        self.end_date = end_date
        self.current_pair_index = 0
        self.all_data = {}
        
        # 🎯 ENHANCED: Initialize fetch stats
        self.fetch_stats['total_pairs'] = len(pairs)
        
        logger.info(f"🚀 Fetching data for {pairs}")
        logger.info(f"📅 Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        # 🔧 TIMEFRAME INFO
        logger.info(f"📊 TIMEFRAME CONFIGURATION:")
        for pair in pairs:
            expected_tf = PAIR_TIMEFRAMES.get(pair, "M30")
            logger.info(f"   {pair}: {expected_tf}")
        
        logger.info(f"📝 NOTE: Fetching M30 base data - backtest engine will aggregate to target timeframes")
        
        self.client.setConnectedCallback(self.connected)
        self.client.setDisconnectedCallback(self.disconnected)
        self.client.startService()
        
        reactor.run()
    
    def connected(self, client):
        logger.info("✅ Connected to cTrader server")
        self.authenticate_app()
    
    def authenticate_app(self):
        logger.info("🔐 Authenticating application...")
        appAuth = ProtoOAApplicationAuthReq()
        appAuth.clientId = self.client_id
        appAuth.clientSecret = self.client_secret
        deferred = self.client.send(appAuth)
        deferred.addCallbacks(self.onAppAuthSuccess, self.onError)
    
    def onAppAuthSuccess(self, response):
        logger.info("✅ Application authenticated")
        self.authenticate_user()
    
    def authenticate_user(self):
        logger.info("🔐 Authenticating user...")
        userAuth = ProtoOAAccountAuthReq()
        userAuth.ctidTraderAccountId = self.account_id
        userAuth.accessToken = self.access_token
        deferred = self.client.send(userAuth)
        deferred.addCallbacks(self.onUserAuthSuccess, self.onError)
    
    def onUserAuthSuccess(self, response):
        logger.info("✅ User authenticated successfully")
        logger.info(f"📊 Starting data fetch for {len(self.pairs_to_fetch)} pairs")
        self.fetch_next_pair()
    
    def fetch_next_pair(self):
        if self.current_pair_index >= len(self.pairs_to_fetch):
            logger.info("✅ All data fetched. Saving to Excel...")
            self.save_to_excel()
            reactor.stop()
            return
        
        pair = self.pairs_to_fetch[self.current_pair_index]
        logger.info(f"📈 Fetching {pair} data... ({self.current_pair_index + 1}/{len(self.pairs_to_fetch)})")
        self.sendTrendbarReq(pair)
    
    def sendTrendbarReq(self, pair):
        """🚀 ENHANCED: Send trendbar request with NATIVE timeframe per pair"""
        request = ProtoOAGetTrendbarsReq()
        request.ctidTraderAccountId = self.account_id
        
        # 🎯 GET NATIVE TIMEFRAME FOR THIS PAIR
        native_timeframe = self.get_pair_timeframe(pair)
        request.period = ProtoOATrendbarPeriod.Value(native_timeframe)
        
        logger.info(f"📊 {pair}: Requesting {native_timeframe} data directly")
        
        # Set time range
        request.fromTimestamp = int(calendar.timegm(self.start_date.utctimetuple())) * 1000
        request.toTimestamp = int(calendar.timegm(self.end_date.utctimetuple())) * 1000
        
        # 🔧 ADJUST COUNT BASED ON TIMEFRAME (more realistic limits)
        if native_timeframe == "H1":
            request.count = 10000  # H1: ~4,320 bars for 6 months
        elif native_timeframe == "H4": 
            request.count = 5000   # H4: ~1,080 bars for 6 months
        else:
            request.count = 20000  # M30: fallback
        
        # Get symbol ID
        symbol_id = forex_symbols.get(pair)
        if not symbol_id:
            logger.error(f"❌ Symbol ID not found for {pair}. Skipping.")
            self.all_data[pair] = pd.DataFrame()
            self.current_pair_index += 1
            reactor.callLater(0.5, self.fetch_next_pair)
            return
            
        request.symbolId = symbol_id
        
        deferred = self.client.send(request, clientMsgId=None)  # Same as ctrader.py
        deferred.addCallbacks(self.onTrendbarDataReceived, self.onError)
    
    def onTrendbarDataReceived(self, response):
        """🔧 ENHANCED: Process trendbar data with quality checks and candle_range calculation"""
        pair = self.pairs_to_fetch[self.current_pair_index]
        
        try:
            # Extract the response using Protobuf.extract (same as ctrader.py)
            parsed = Protobuf.extract(response)
            trendbars = parsed.trendbar  # This is a list of trendbar objects
            
            if not trendbars:
                logger.warning(f"⚠️ No data received for {pair}")
                self.all_data[pair] = pd.DataFrame()
                self.fetch_stats['failed_pairs'] += 1
            else:
                # Convert to DataFrame (exact same logic as ctrader.py)
                data = []
                for tb in trendbars:
                    # Calculate OHLC values
                    low_price = tb.low / 1e5
                    open_price = (tb.low + tb.deltaOpen) / 1e5
                    high_price = (tb.low + tb.deltaHigh) / 1e5
                    close_price = (tb.low + tb.deltaClose) / 1e5
                    
                    data.append({
                        'timestamp': datetime.datetime.utcfromtimestamp(tb.utcTimestampInMinutes * 60),
                        'open': open_price,
                        'high': high_price,
                        'low': low_price,
                        'close': close_price,
                        'volume': tb.volume
                    })
                
                df = pd.DataFrame(data)
                df.sort_values('timestamp', inplace=True, ascending=True)  # Sort ascending for backtest
                
                # 🔧 CRITICAL: Add candle_range calculation (required by strategies)
                df['candle_range'] = df['high'] - df['low']
                
                # 🎯 DATA QUALITY CHECKS
                initial_count = len(df)
                
                # Remove any invalid data (where high < low, etc.)
                df = df[(df['high'] >= df['low']) & (df['high'] >= df['open']) & 
                       (df['high'] >= df['close']) & (df['low'] <= df['open']) & 
                       (df['low'] <= df['close'])]
                
                # Check for data loss
                if len(df) < initial_count:
                    logger.warning(f"⚠️ {pair}: Removed {initial_count - len(df)} invalid bars")
                
                self.all_data[pair] = df
                self.fetch_stats['successful_pairs'] += 1
                self.fetch_stats['total_bars'] += len(df)
                
                # Get native timeframe for logging
                native_tf = self.get_pair_timeframe(pair)
                logger.info(f"✅ {pair}: {len(df)} {native_tf} bars loaded NATIVELY")
                
                # 📊 Enhanced data quality summary with efficiency metrics
                if len(df) > 0:
                    date_range = f"{df['timestamp'].min().strftime('%Y-%m-%d')} to {df['timestamp'].max().strftime('%Y-%m-%d')}"
                    avg_volume = df['volume'].mean()
                    avg_range = df['candle_range'].mean()
                    
                    # Calculate efficiency gain vs M30
                    if native_tf == "H1":
                        m30_equivalent = len(df) * 2  # 1 H1 = 2 M30
                        efficiency = f"50% less data than M30 ({m30_equivalent:,} M30 equivalent)"
                    elif native_tf == "H4":
                        m30_equivalent = len(df) * 8  # 1 H4 = 8 M30  
                        efficiency = f"87.5% less data than M30 ({m30_equivalent:,} M30 equivalent)"
                    else:
                        efficiency = "No efficiency gain (M30)"
                    
                    logger.info(f"   📅 Date range: {date_range}")
                    logger.info(f"   📊 Avg volume: {avg_volume:,.0f}")
                    logger.info(f"   🕯️ Avg range: {avg_range:.5f}")
                    logger.info(f"   ⚡ Efficiency: {efficiency}")
            
        except Exception as e:
            logger.error(f"❌ Error processing data for {pair}: {e}")
            self.all_data[pair] = pd.DataFrame()
            self.fetch_stats['failed_pairs'] += 1
        
        # Move to next pair
        self.current_pair_index += 1
        reactor.callLater(0.5, self.fetch_next_pair)  # Small delay between requests
    
    def save_to_excel(self):
        """🔧 ENHANCED: Save all data to Excel file with improved directory structure and reporting"""
        # 🎯 UPDATED: Use backtest/data directory as requested
        output_dir = 'backtest/data'
        os.makedirs(output_dir, exist_ok=True)
        
        output_file = f'{output_dir}/forex_data1.xlsx'
        
        try:
            with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                logger.info(f"📝 SAVING DATA SUMMARY:")
                logger.info(f"="*50)
                
                for pair, df in self.all_data.items():
                    if not df.empty:
                        sheet_name = pair.replace('/', '_')
                        df.to_excel(writer, sheet_name=sheet_name, index=False)
                        
                        # Enhanced logging per pair with native timeframe info
                        native_tf = PAIR_TIMEFRAMES.get(pair, "M30")
                        date_range = f"{df['timestamp'].min().strftime('%Y-%m-%d')} to {df['timestamp'].max().strftime('%Y-%m-%d')}"
                        
                        # Calculate data efficiency
                        if native_tf == "H1":
                            m30_equivalent = len(df) * 2
                            efficiency_info = f"({m30_equivalent:,} M30 equivalent - 50% less data)"
                        elif native_tf == "H4":
                            m30_equivalent = len(df) * 8
                            efficiency_info = f"({m30_equivalent:,} M30 equivalent - 87.5% less data)"
                        else:
                            efficiency_info = "(M30 baseline)"
                        
                        logger.info(f"💾 {pair} → sheet '{sheet_name}'")
                        logger.info(f"   📊 Bars: {len(df):,} {native_tf} candles {efficiency_info}")
                        logger.info(f"   📅 Range: {date_range}")
                        logger.info(f"   🕯️ Avg Range: {df['candle_range'].mean():.5f}")
                        logger.info(f"   ✅ NATIVE TIMEFRAME - No aggregation needed!")
                        
                    else:
                        logger.warning(f"⚠️ No data to save for {pair}")
                
                logger.info(f"="*50)
            
            # 🎯 FINAL SUMMARY REPORT
            logger.info(f"🎉 DATA FETCH COMPLETED!")
            logger.info(f"📂 File saved: {output_file}")
            logger.info(f"")
            logger.info(f"📊 FETCH STATISTICS:")
            logger.info(f"   Total pairs: {self.fetch_stats['total_pairs']}")
            logger.info(f"   Successful: {self.fetch_stats['successful_pairs']}")
            logger.info(f"   Failed: {self.fetch_stats['failed_pairs']}")
            logger.info(f"   Total bars: {self.fetch_stats['total_bars']:,}")
            logger.info(f"   Avg per pair: {self.fetch_stats['total_bars'] // max(1, self.fetch_stats['successful_pairs']):,}")
            logger.info(f"")
            logger.info(f"🔧 NEXT STEPS:")
            logger.info(f"   1. Run backtest engine with: TARGET_PAIR = 'EUR/USD'")
            logger.info(f"   2. Check backtest/data_fetch.log for detailed info")
            logger.info(f"   3. ✅ NATIVE TIMEFRAMES - No aggregation needed!")
            logger.info(f"   4. 🚀 Faster backtests with accurate broker data")
            
            # Calculate total efficiency gain
            total_m30_equivalent = 0
            actual_bars = self.fetch_stats['total_bars']
            
            for pair, df in self.all_data.items():
                if not df.empty:
                    native_tf = PAIR_TIMEFRAMES.get(pair, "M30")
                    if native_tf == "H1":
                        total_m30_equivalent += len(df) * 2
                    elif native_tf == "H4":
                        total_m30_equivalent += len(df) * 8
                    else:
                        total_m30_equivalent += len(df)
            
            if total_m30_equivalent > actual_bars:
                efficiency_pct = ((total_m30_equivalent - actual_bars) / total_m30_equivalent) * 100
                logger.info(f"")
                logger.info(f"⚡ EFFICIENCY GAIN:")
                logger.info(f"   Old approach: {total_m30_equivalent:,} M30 bars")
                logger.info(f"   New approach: {actual_bars:,} native bars")
                logger.info(f"   Data reduction: {efficiency_pct:.1f}% smaller!")
                logger.info(f"   🎯 Same accuracy, much faster!")
            
        except Exception as e:
            logger.error(f"❌ Error saving to Excel: {e}")
            logger.error(f"💡 Try installing openpyxl: pip install openpyxl")
    
    def onError(self, failure):
        logger.error(f"❌ cTrader API Error: {failure}")
        reactor.stop()
    
    def disconnected(self, client, reason):
        logger.info(f"🔌 Disconnected from cTrader: {reason}")

if __name__ == "__main__":
    try:
        # 🎯 ENHANCED: Updated configuration and better date range
        fetcher = DataFetcher()
        
        # 🔄 UPDATED: Use exact pairs from ctrader.py in priority order
        pairs = [
            'EUR/USD',  # H1 - Most liquid, good for testing
            'EUR/GBP',  # H4 - Cross pair
            'GBP/USD',  # H4 - Major volatile
            'USD/JPY',  # H4 - Major trend-following
            'EUR/JPY',  # H4 - Carry trade
            'GBP/JPY'   # H4 - "The Beast"
        ]
        
        # 🗓️ UPDATED: 6-month recent data with NATIVE timeframes
        start = datetime.datetime(2025, 1, 1)     # July 1, 2024
        end = datetime.datetime(2025, 9, 1)       # January 9, 2025 (today)
        
        logger.info(f"🚀 ENHANCED DATA FETCHER v3.0 - NATIVE TIMEFRAMES")
        logger.info(f"🎯 Fetching 6 months of recent data: {start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}")
        logger.info(f"🔧 Output: backtest/data/forex_data1.xlsx")
        logger.info(f"")
        logger.info(f"📊 NATIVE TIMEFRAME CONFIGURATION:")
        
        total_expected_bars = 0
        for pair in pairs:
            native_tf = PAIR_TIMEFRAMES.get(pair, "M30")
            if native_tf == "H1":
                expected_bars = 4320  # 180 days × 24 H1 bars/day
                old_bars = 8640      # 180 days × 48 M30 bars/day
                efficiency = "50% less data"
            elif native_tf == "H4":
                expected_bars = 1080  # 180 days × 6 H4 bars/day  
                old_bars = 8640      # 180 days × 48 M30 bars/day
                efficiency = "87.5% less data"
            else:
                expected_bars = 8640  # M30 fallback
                old_bars = 8640
                efficiency = "no change"
            
            total_expected_bars += expected_bars
            logger.info(f"   {pair}: {native_tf} NATIVE (~{expected_bars:,} bars, {efficiency})")
        
        logger.info(f"")
        logger.info(f"⚡ EFFICIENCY SUMMARY:")
        old_total = len(pairs) * 8640  # Old M30 approach
        logger.info(f"   Old M30 approach: {old_total:,} total bars")
        logger.info(f"   New native approach: ~{total_expected_bars:,} total bars")
        efficiency_pct = ((old_total - total_expected_bars) / old_total) * 100
        logger.info(f"   Data reduction: {efficiency_pct:.1f}% less data to download!")
        logger.info(f"   🎯 Native broker accuracy + faster processing")
        logger.info(f"")
        
        # Environment debug info
        logger.info(f"🔧 Python version: {os.sys.version}")
        logger.info(f"🔧 Platform: {os.name}")
        logger.info(f"🔧 Working directory: {os.getcwd()}")
        
        # Check environment variables
        required_vars = ["CTRADER_CLIENT_ID", "CTRADER_CLIENT_SECRET", "CTRADER_ACCOUNT_ID", "CTRADER_ACCESS_TOKEN"]
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        
        if missing_vars:
            logger.error(f"❌ Missing environment variables: {missing_vars}")
            logger.error(f"💡 Check your .env file")
            exit(1)
        else:
            logger.info(f"✅ All environment variables present")
        
        logger.info(f"")
        logger.info(f"🚀 Starting data fetch...")
        
        fetcher.fetch_data(pairs, start, end)
        
    except KeyboardInterrupt:
        logger.info("🛑 Data fetch interrupted by user")
        if reactor.running:
            reactor.stop()
    except Exception as e:
        import traceback
        logger.error(f"❌ Error in data fetcher: {e}")
        logger.error(f"❌ Full traceback: {traceback.format_exc()}")
        if reactor.running:
            reactor.stop()