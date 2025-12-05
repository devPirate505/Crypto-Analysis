"""
Feature engineering pipeline - combine all feature generators.
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.features.indicators import TechnicalIndicators
from src.features.transform import FeatureTransforms
from src.ingestion.storage import DataStorage
from src.utils.config import load_config
import pandas as pd
import json
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FeaturePipeline:
    """
    Complete feature engineering pipeline.
    """
    
    def __init__(self, config_path: str = "configs/default.yaml"):
        """
        Initialize feature pipeline.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = load_config(config_path)
        self.indicators = TechnicalIndicators(self.config['features'])
        self.transforms = FeatureTransforms(self.config['features'])
        self.storage = DataStorage()
        
    def process_coin(self, coin_id: str) -> pd.DataFrame:
        """
        Process a single coin's data through the feature pipeline.
        
        Args:
            coin_id: Coin identifier
            
        Returns:
            DataFrame with all features
        """
        logger.info(f"Processing features for {coin_id}")
        
        # Load raw data
        df = self.storage.load_raw(coin_id, latest=True)
        
        if df is None:
            raise ValueError(f"No raw data found for {coin_id}")
            
        # Add technical indicators
        df = self.indicators.add_all_indicators(df)
        
        # Add price-based features with REDUCED windows to preserve data
        # Use smaller windows: max 24h instead of 72h
        df = self.transforms.create_all_features(df)
        
        # Drop rows with NaN values ONLY in core columns to preserve data
        # Don't drop based on lag features or advanced indicators
        core_columns = ['close', 'rsi_14', 'macd', 'ema_20', 'return_1h', 'rolling_std_24h']
        initial_rows = len(df)
        df = df.dropna(subset=core_columns)
        logger.info(f"Dropped {initial_rows - len(df)} rows with NaN in core columns (kept {len(df)} rows)")
        
        # Fill any remaining NaN values with forward fill, then backward fill
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        # Ensure timestamp is preserved
        if 'timestamp' not in df.columns:
            logger.error("Timestamp column missing after feature engineering!")
            
        logger.info(f"Final dataset: {len(df)} rows, {len(df.columns)} columns")
        
        # Save processed data
        self.storage.save_processed(df, coin_id)
        
        logger.info(f"Processed {coin_id}: {len(df)} rows, {len(df.columns)} features")
        
        return df
        
    def create_feature_manifest(self, df: pd.DataFrame, coin_id: str) -> dict:
        """
        Create a manifest of all features.
        
        Args:
            df: DataFrame with features
            coin_id: Coin identifier
            
        Returns:
            Feature manifest dictionary
        """
        # Categorize features
        raw_features = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'market_cap', 'coin_id']
        
        indicator_features = [
            col for col in df.columns 
            if any(x in col for x in ['rsi', 'macd', 'ema', 'bb', 'adx', 'cci', 'atr', 'stoch', 'obv'])
        ]
        
        transform_features = [
            col for col in df.columns
            if any(x in col for x in ['return', 'rolling', 'hl_range', 'position', 'roc', 'volume_', 'lag', 'norm'])
        ]
        
        manifest = {
            'coin_id': coin_id,
            'total_features': len(df.columns),
            'total_rows': len(df),
            'raw_features': raw_features,
            'indicator_features': indicator_features,
            'transform_features': transform_features,
            'all_columns': list(df.columns)
        }
        
        return manifest
        
    def save_manifest(self, manifest: dict, coin_id: str):
        """
        Save feature manifest to JSON.
        
        Args:
            manifest: Feature manifest dictionary
            coin_id: Coin identifier
        """
        manifest_path = self.storage.processed_path / f"{coin_id}_feature_manifest.json"
        
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
            
        logger.info(f"Saved feature manifest to {manifest_path}")
        
    def run(self, coin_ids: list = None):
        """
        Run feature engineering for specified coins.
        
        Args:
            coin_ids: List of coin IDs (if None, uses config)
        """
        if coin_ids is None:
            coin_ids = self.config['data']['symbols']
            
        for coin_id in coin_ids:
            try:
                # Process features
                df = self.process_coin(coin_id)
                
                # Create and save manifest
                manifest = self.create_feature_manifest(df, coin_id)
                self.save_manifest(manifest, coin_id)
                
                logger.info(f"Completed feature engineering for {coin_id}")
                
            except Exception as e:
                logger.error(f"Failed to process {coin_id}: {e}")
                continue


if __name__ == "__main__":
    pipeline = FeaturePipeline()
    pipeline.run()
