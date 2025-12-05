"""
Script to fetch and store historical cryptocurrency data.
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.ingestion.market_chart_fetcher import MarketChartFetcher
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
    Main function to fetch and store cryptocurrency data.
    """
    # Load configuration
    config = load_config()
    
    # Initialize fetcher and storage
    fetcher = MarketChartFetcher(
        rate_limit_delay=config['api']['coingecko']['rate_limit_delay']
    )
    # Use default base_path="data" which will create data/raw, data/processed, etc.
    storage = DataStorage()
    
    # Get coins from config
    coin_ids = config['data']['symbols']
    days = config['data']['history_days']
    
    # Adjust for CoinGecko API limits
    if days > 90:
        logger.warning(f"CoinGecko free tier limited to 90 days. Using 90 instead of {days}.")
        days = 90
    
    logger.info(f"Fetching data for {len(coin_ids)} coins: {coin_ids}")
    
    # Fetch data for each coin
    for coin_id in coin_ids:
        try:
            logger.info(f"Processing {coin_id}...")
            
            # Fetch data and convert to hourly OHLCV
            df = fetcher.fetch_and_convert_to_ohlcv(
                coin_id,
                days=days,
                resample_freq='1h'  # Hourly candles
            )
            
            # Quality check
            quality = storage.check_quality(df)
            logger.info(f"Quality metrics for {coin_id}: {quality}")
            
            # Save raw data
            filepath = storage.save_raw(df, coin_id)
            logger.info(f"Saved {coin_id} data to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to process {coin_id}: {e}")
            continue
    
    logger.info("Data ingestion complete!")
    
    # List saved files
    raw_files = storage.list_raw_files()
    logger.info(f"Total raw files saved: {len(raw_files)}")


if __name__ == "__main__":
    main()
