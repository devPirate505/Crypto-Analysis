"""
Backfill script using Binance API for historical data.
Much better than CoinGecko: no rate limits, 3+ years of data.
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.ingestion.binance_fetcher import BinanceFetcher
from src.ingestion.storage import DataStorage
from src.utils.config import load_config
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """
    Main function to fetch cryptocurrency data from Binance.
    """
    # Load configuration
    config = load_config()
    
    # Initialize fetcher and storage
    fetcher = BinanceFetcher()  # No API key needed for public data
    storage = DataStorage()
    
    # Get configuration
    coin_ids = config['data'].get('symbols', ['bitcoin', 'ethereum', 'binancecoin'])
    
    # Binance-specific settings
    interval = '1h'  # Can be '1m', '5m', '15m', '1h', '4h', '1d'
    days = config['data'].get('history_days', 365)
    
    # Override if too large (Binance can handle it, but start reasonable)
    if days > 730:
        logger.warning(f"Reducing history_days from {days} to 730 (2 years)")
        days = 730
    
    logger.info(f"Fetching data for {len(coin_ids)} coins: {coin_ids}")
    logger.info(f"Interval: {interval}, Days: {days}")
    
    # Show available symbols
    available = BinanceFetcher.get_available_symbols()
    logger.info(f"Available symbols: {available}")
    
    # Fetch data for each coin
    successful = 0
    for coin_id in coin_ids:
        try:
            logger.info(f"Processing {coin_id}...")
            
            # Fetch data
            df = fetcher.fetch_ohlcv(
                coin_id,
                interval=interval,
                days=days
            )
            
            if df.empty:
                logger.warning(f"No data returned for {coin_id}")
                continue
            
            # Quality check
            quality = storage.check_quality(df)
            logger.info(f"Quality metrics for {coin_id}: {quality}")
            
            # Save raw data
            filepath = storage.save_raw(df, coin_id)
            logger.info(f"Saved {coin_id} data to {filepath}")
            
            successful += 1
            
        except Exception as e:
            logger.error(f"Failed to process {coin_id}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    logger.info(f"\nData ingestion complete!")
    logger.info(f"Successfully processed: {successful}/{len(coin_ids)} coins")
    
    if successful == 0:
        logger.error("No data was successfully fetched. Please check the logs above.")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
