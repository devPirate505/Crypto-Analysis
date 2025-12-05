"""
Alternative data fetcher that creates OHLCV from market chart data.
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

load_dotenv()


class MarketChartFetcher:
    """
    Fetches market chart data and converts to OHLCV candles.
    """
    
    def __init__(self, rate_limit_delay: float = 1.5, api_key: Optional[str] = None):
        """Initialize fetcher."""
        api_key = api_key or os.getenv('COINGECKO_API_KEY')
        
        if api_key:
            logger.info("Using CoinGecko Pro API with API key")
            self.cg = CoinGeckoAPI(api_key=api_key)
        else:
            logger.info("Using CoinGecko Free API (no API key)")
            self.cg = CoinGeckoAPI()
            
        self.rate_limit_delay = rate_limit_delay
        
    def fetch_and_convert_to_ohlcv(
        self,
        coin_id: str,
        vs_currency: str = "usd",
        days: int = 90,
        resample_freq: str = '1h'
    ) -> pd.DataFrame:
        """
        Fetch market chart data and convert to OHLCV candles.
        
        Args:
            coin_id: CoinGecko coin ID
            vs_currency: Currency to compare against
            days: Number of days of historical data
            resample_freq: Resampling frequency ('1h', '4h', '1D', etc.)
            
        Returns:
            DataFrame with OHLCV data
        """
        logger.info(f"Fetching market chart for {coin_id} ({days} days)")
        
        try:
            # Fetch market chart data
            market_data = self.cg.get_coin_market_chart_by_id(
                id=coin_id,
                vs_currency=vs_currency,
                days=days
            )
            
            # Convert prices to DataFrame
            prices = pd.DataFrame(market_data['prices'], columns=['timestamp', 'price'])
            prices['timestamp'] = pd.to_datetime(prices['timestamp'], unit='ms')
            prices = prices.set_index('timestamp')
            
            # Resample to create OHLC
            ohlc = prices['price'].resample(resample_freq).ohlc()
            ohlc.columns = ['open', 'high', 'low', 'close']
            
            # Get volume data
            volumes = pd.DataFrame(market_data['total_volumes'], columns=['timestamp', 'volume'])
            volumes['timestamp'] = pd.to_datetime(volumes['timestamp'], unit='ms')
            volumes = volumes.set_index('timestamp')
            volume_resampled = volumes['volume'].resample(resample_freq).sum()
            
            # Get market cap data
            market_caps = pd.DataFrame(market_data['market_caps'], columns=['timestamp', 'market_cap'])
            market_caps['timestamp'] = pd.to_datetime(market_caps['timestamp'], unit='ms')
            market_caps = market_caps.set_index('timestamp')
            market_cap_resampled = market_caps['market_cap'].resample(resample_freq).last()
            
            # Combine all data
            df = ohlc.copy()
            df['volume'] = volume_resampled
            df['market_cap'] = market_cap_resampled
            df['coin_id'] = coin_id
            
            # Reset index to get timestamp as column
            df = df.reset_index()
            df = df.rename(columns={'index': 'timestamp'})
            
            # Drop rows with NaN values
            df = df.dropna()
            
            logger.info(f"Created {len(df)} {resample_freq} candles for {coin_id}")
            
            # Rate limiting
            time.sleep(self.rate_limit_delay)
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data for {coin_id}: {e}")
            raise
            
    def fetch_multiple_coins(
        self,
        coin_ids: List[str],
        vs_currency: str = "usd",
        days: int = 90,
        resample_freq: str = '1h'
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple coins.
        
        Args:
            coin_ids: List of CoinGecko coin IDs
            vs_currency: Currency to compare against
            days: Number of days of historical data
            resample_freq: Resampling frequency
            
        Returns:
            Dictionary mapping coin_id to DataFrame
        """
        results = {}
        
        for coin_id in coin_ids:
            try:
                df = self.fetch_and_convert_to_ohlcv(coin_id, vs_currency, days, resample_freq)
                results[coin_id] = df
            except Exception as e:
                logger.error(f"Failed to fetch {coin_id}: {e}")
                continue
                
        return results
