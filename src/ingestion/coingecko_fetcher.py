"""
CoinGecko API fetcher for cryptocurrency OHLCV data.
"""
import time
import os
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import pandas as pd
from pycoingecko import CoinGeckoAPI
from dotenv import load_dotenv
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


class CoinGeckoFetcher:
    """
    Fetches historical OHLCV data from CoinGecko API.
    """
    
    def __init__(self, rate_limit_delay: float = 1.5, api_key: Optional[str] = None):
        """
        Initialize CoinGecko fetcher.
        
        Args:
            rate_limit_delay: Delay between API requests in seconds
            api_key: Optional CoinGecko Pro API key (or set in .env file)
        """
        # Get API key from parameter or environment variable
        api_key = api_key or os.getenv('COINGECKO_API_KEY')
        
        if api_key:
            logger.info("Using CoinGecko Pro API with API key")
            self.cg = CoinGeckoAPI(api_key=api_key)
        else:
            logger.info("Using CoinGecko Free API (no API key)")
            self.cg = CoinGeckoAPI()
            
        self.rate_limit_delay = rate_limit_delay
        
    def fetch_ohlc(
        self, 
        coin_id: str, 
        vs_currency: str = "usd",
        days: int = 365
    ) -> pd.DataFrame:
        """
        Fetch OHLC data for a cryptocurrency.
        
        Args:
            coin_id: CoinGecko coin ID (e.g., 'bitcoin', 'ethereum')
            vs_currency: Currency to compare against (default: 'usd')
            days: Number of days of historical data
            
        Returns:
            DataFrame with columns: timestamp, open, high, low, close
        """
        logger.info(f"Fetching OHLC data for {coin_id} ({days} days)")
        
        try:
            # CoinGecko OHLC endpoint
            # Note: Free API only supports up to 90 days for hourly data
            if days > 90:
                logger.warning(f"CoinGecko free tier only supports 90 days for hourly data. Adjusting from {days} to 90 days.")
                days = 90
                
            ohlc_data = self.cg.get_coin_ohlc_by_id(
                id=coin_id,
                vs_currency=vs_currency,
                days=days
            )
            
            # Convert to DataFrame
            df = pd.DataFrame(
                ohlc_data,
                columns=['timestamp', 'open', 'high', 'low', 'close']
            )
            
            # Convert timestamp from milliseconds to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Sort by timestamp
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            logger.info(f"Fetched {len(df)} OHLC records for {coin_id}")
            
            # Rate limiting
            time.sleep(self.rate_limit_delay)
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching OHLC data for {coin_id}: {e}")
            raise
            
    def fetch_market_chart(
        self,
        coin_id: str,
        vs_currency: str = "usd",
        days: int = 365
    ) -> pd.DataFrame:
        """
        Fetch market chart data (prices, market caps, volumes).
        
        Args:
            coin_id: CoinGecko coin ID
            vs_currency: Currency to compare against
            days: Number of days of historical data
            
        Returns:
            DataFrame with columns: timestamp, price, market_cap, volume
        """
        logger.info(f"Fetching market chart for {coin_id} ({days} days)")
        
        try:
            market_data = self.cg.get_coin_market_chart_by_id(
                id=coin_id,
                vs_currency=vs_currency,
                days=days
            )
            
            # Extract prices, market caps, and volumes
            prices = pd.DataFrame(market_data['prices'], columns=['timestamp', 'price'])
            market_caps = pd.DataFrame(market_data['market_caps'], columns=['timestamp', 'market_cap'])
            volumes = pd.DataFrame(market_data['total_volumes'], columns=['timestamp', 'volume'])
            
            # Merge on timestamp
            df = prices.merge(market_caps, on='timestamp', how='left')
            df = df.merge(volumes, on='timestamp', how='left')
            
            # Convert timestamp
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            logger.info(f"Fetched {len(df)} market chart records for {coin_id}")
            
            # Rate limiting
            time.sleep(self.rate_limit_delay)
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching market chart for {coin_id}: {e}")
            raise
            
    def fetch_combined(
        self,
        coin_id: str,
        vs_currency: str = "usd",
        days: int = 90
    ) -> pd.DataFrame:
        """
        Fetch and combine OHLC and market chart data.
        
        Args:
            coin_id: CoinGecko coin ID
            vs_currency: Currency to compare against
            days: Number of days of historical data
            
        Returns:
            Combined DataFrame
        """
        logger.info(f"Fetching combined data for {coin_id}")
        
        # Fetch OHLC
        ohlc_df = self.fetch_ohlc(coin_id, vs_currency, days)
        
        # Fetch market chart for volume data
        market_df = self.fetch_market_chart(coin_id, vs_currency, days)
        
        # Merge on timestamp (round to nearest hour for matching)
        ohlc_df['timestamp_hour'] = ohlc_df['timestamp'].dt.floor('h')
        market_df['timestamp_hour'] = market_df['timestamp'].dt.floor('h')
        
        combined = ohlc_df.merge(
            market_df[['timestamp_hour', 'volume', 'market_cap']],
            on='timestamp_hour',
            how='left'
        )
        
        # Drop the temporary column
        combined = combined.drop('timestamp_hour', axis=1)
        
        # Add coin_id column
        combined['coin_id'] = coin_id
        
        logger.info(f"Combined data ready: {len(combined)} records")
        
        return combined
        
    def fetch_multiple_coins(
        self,
        coin_ids: List[str],
        vs_currency: str = "usd",
        days: int = 90
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple coins.
        
        Args:
            coin_ids: List of CoinGecko coin IDs
            vs_currency: Currency to compare against
            days: Number of days of historical data
            
        Returns:
            Dictionary mapping coin_id to DataFrame
        """
        results = {}
        
        for coin_id in coin_ids:
            try:
                df = self.fetch_combined(coin_id, vs_currency, days)
                results[coin_id] = df
            except Exception as e:
                logger.error(f"Failed to fetch {coin_id}: {e}")
                continue
                
        return results
