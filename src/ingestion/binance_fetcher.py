"""
Binance API fetcher for cryptocurrency OHLCV data.
No API key required for public market data.
"""
from binance.client import Client
from datetime import datetime, timedelta
import pandas as pd
import time
import logging
from typing import Optional, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BinanceFetcher:
    """
    Fetches historical OHLCV data from Binance API.
    Free, no API key required for historical data.
    """
    
    # Binance symbol mappings
    SYMBOL_MAP = {
        'bitcoin': 'BTCUSDT',
        'ethereum': 'ETHUSDT',
        'binancecoin': 'BNBUSDT',
        'cardano': 'ADAUSDT',
        'solana': 'SOLUSDT',
        'ripple': 'XRPUSDT',
        'polkadot': 'DOTUSDT',
        'dogecoin': 'DOGEUSDT',
    }
    
    INTERVAL_MAP = {
        '1m': Client.KLINE_INTERVAL_1MINUTE,
        '5m': Client.KLINE_INTERVAL_5MINUTE,
        '15m': Client.KLINE_INTERVAL_15MINUTE,
        '1h': Client.KLINE_INTERVAL_1HOUR,
        '4h': Client.KLINE_INTERVAL_4HOUR,
        '1d': Client.KLINE_INTERVAL_1DAY,
    }
    
    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None):
        """
        Initialize Binance client.
        
        Args:
            api_key: Optional API key (not needed for public data)
            api_secret: Optional API secret
        """
        if api_key and api_secret:
            self.client = Client(api_key, api_secret)
            logger.info("Initialized Binance client with API credentials")
        else:
            self.client = Client()  # Public client, no auth needed
            logger.info("Initialized Binance public client (no API key)")
    
    def fetch_ohlcv(
        self,
        coin_id: str,
        interval: str = '1h',
        days: int = 365,
        limit: int = 1000
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data from Binance.
        
        Args:
            coin_id: Coin identifier (e.g., 'bitcoin', 'ethereum')
            interval: Timeframe ('1m', '5m', '15m', '1h', '4h', '1d')
            days: Number of days of historical data
            limit: Max records per request (max 1000)
            
        Returns:
            DataFrame with OHLCV data
        """
        # Map coin_id to Binance symbol
        symbol = self.SYMBOL_MAP.get(coin_id.lower())
        if not symbol:
            raise ValueError(f"Unknown coin: {coin_id}. Available: {list(self.SYMBOL_MAP.keys())}")
        
        # Map interval
        binance_interval = self.INTERVAL_MAP.get(interval)
        if not binance_interval:
            raise ValueError(f"Unknown interval: {interval}. Available: {list(self.INTERVAL_MAP.keys())}")
        
        logger.info(f"Fetching {symbol} data (interval={interval}, days={days})")
        
        # Calculate start time
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
        
        # Fetch data in chunks (Binance limits to 1000 candles per request)
        all_klines = []
        current_start = start_time
        
        while current_start < end_time:
            try:
                # Fetch klines
                klines = self.client.get_historical_klines(
                    symbol=symbol,
                    interval=binance_interval,
                    start_str=str(int(current_start.timestamp() * 1000)),
                    end_str=str(int(end_time.timestamp() * 1000)),
                    limit=limit
                )
                
                if not klines:
                    break
                
                all_klines.extend(klines)
                
                # Update start time for next batch
                last_timestamp = klines[-1][0]
                current_start = datetime.fromtimestamp(last_timestamp / 1000) + timedelta(seconds=1)
                
                logger.info(f"Fetched {len(klines)} candles, total: {len(all_klines)}")
                
                # Rate limiting (be nice to Binance)
                time.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Error fetching data: {e}")
                break
        
        # Convert to DataFrame
        df = self._klines_to_dataframe(all_klines, coin_id)
        
        logger.info(f"Successfully fetched {len(df)} records for {coin_id}")
        return df
    
    def _klines_to_dataframe(self, klines: List, coin_id: str) -> pd.DataFrame:
        """
        Convert Binance klines to DataFrame.
        
        Binance kline format:
        [
            [
                1499040000000,      # Open time
                "0.01634790",       # Open
                "0.80000000",       # High
                "0.01575800",       # Low
                "0.01577100",       # Close
                "148976.11427815",  # Volume
                1499644799999,      # Close time
                "2434.19055334",    # Quote asset volume
                308,                # Number of trades
                "1756.87402397",    # Taker buy base asset volume
                "28.46694368",      # Taker buy quote asset volume
                "17928899.62484339" # Ignore
            ]
        ]
        """
        df = pd.DataFrame(klines, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 
            'taker_buy_base', 'taker_buy_quote', 'ignore'
        ])
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms')
        
        # Convert price columns to float
        price_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in price_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Add coin_id
        df['coin_id'] = coin_id
        
        # Select and rename columns to match our schema
        df = df[[
            'timestamp', 'open', 'high', 'low', 'close', 'volume', 'coin_id'
        ]]
        
        # Remove duplicates and sort
        df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
        
        return df
    
    def fetch_multiple_coins(
        self,
        coin_ids: List[str],
        interval: str = '1h',
        days: int = 365
    ) -> dict:
        """
        Fetch data for multiple coins.
        
        Args:
            coin_ids: List of coin identifiers
            interval: Timeframe
            days: Number of days of historical data
            
        Returns:
            Dictionary mapping coin_id to DataFrame
        """
        results = {}
        
        for coin_id in coin_ids:
            try:
                logger.info(f"Processing {coin_id}...")
                df = self.fetch_ohlcv(coin_id, interval, days)
                results[coin_id] = df
                
                # Rate limiting between coins
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Failed to fetch {coin_id}: {e}")
                continue
        
        return results
    
    @staticmethod
    def get_available_symbols() -> List[str]:
        """Get list of available cryptocurrencies."""
        return list(BinanceFetcher.SYMBOL_MAP.keys())
