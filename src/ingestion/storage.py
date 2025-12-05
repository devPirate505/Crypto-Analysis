"""
Data storage module for saving and loading data in Parquet format.
"""
import pandas as pd
from pathlib import Path
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataStorage:
    """
    Handles data storage and retrieval using Parquet format.
    """
    
    def __init__(self, base_path: str = "data"):
        """
        Initialize data storage.
        
        Args:
            base_path: Base directory for data storage
        """
        self.base_path = Path(base_path)
        self.raw_path = self.base_path / "raw"
        self.processed_path = self.base_path / "processed"
        self.external_path = self.base_path / "external"
        
        # Create directories
        self.raw_path.mkdir(parents=True, exist_ok=True)
        self.processed_path.mkdir(parents=True, exist_ok=True)
        self.external_path.mkdir(parents=True, exist_ok=True)
        
    def save_raw(
        self,
        df: pd.DataFrame,
        coin_id: str,
        suffix: str = ""
    ) -> Path:
        """
        Save raw data to Parquet.
        
        Args:
            df: DataFrame to save
            coin_id: Coin identifier
            suffix: Optional suffix for filename
            
        Returns:
            Path to saved file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{coin_id}_{timestamp}{suffix}.parquet"
        filepath = self.raw_path / filename
        
        df.to_parquet(filepath, index=False, compression='snappy')
        logger.info(f"Saved raw data to {filepath}")
        
        return filepath
        
    def save_processed(
        self,
        df: pd.DataFrame,
        coin_id: str,
        suffix: str = ""
    ) -> Path:
        """
        Save processed data to Parquet.
        
        Args:
            df: DataFrame to save
            coin_id: Coin identifier
            suffix: Optional suffix for filename
            
        Returns:
            Path to saved file
        """
        filename = f"{coin_id}_processed{suffix}.parquet"
        filepath = self.processed_path / filename
        
        df.to_parquet(filepath, index=False, compression='snappy')
        logger.info(f"Saved processed data to {filepath}")
        
        return filepath
        
    def load_raw(
        self,
        coin_id: str,
        latest: bool = True
    ) -> Optional[pd.DataFrame]:
        """
        Load raw data from Parquet.
        
        Args:
            coin_id: Coin identifier
            latest: If True, load the most recent file
            
        Returns:
            DataFrame or None if not found
        """
        pattern = f"{coin_id}_*.parquet"
        files = list(self.raw_path.glob(pattern))
        
        if not files:
            logger.warning(f"No raw data found for {coin_id}")
            return None
            
        if latest:
            # Sort by modification time
            files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            filepath = files[0]
        else:
            filepath = files[0]
            
        logger.info(f"Loading raw data from {filepath}")
        return pd.read_parquet(filepath)
        
    def load_processed(self, coin_id: str, suffix: str = "") -> Optional[pd.DataFrame]:
        """
        Load processed data from Parquet.
        
        Args:
            coin_id: Coin identifier
            suffix: Optional suffix for filename
            
        Returns:
            DataFrame or None if not found
        """
        filename = f"{coin_id}_processed{suffix}.parquet"
        filepath = self.processed_path / filename
        
        if not filepath.exists():
            logger.warning(f"Processed data not found: {filepath}")
            return None
            
        logger.info(f"Loading processed data from {filepath}")
        return pd.read_parquet(filepath)
        
    def list_raw_files(self, coin_id: Optional[str] = None) -> List[Path]:
        """
        List raw data files.
        
        Args:
            coin_id: Optional filter by coin ID
            
        Returns:
            List of file paths
        """
        if coin_id:
            pattern = f"{coin_id}_*.parquet"
        else:
            pattern = "*.parquet"
            
        return list(self.raw_path.glob(pattern))
        
    def list_processed_files(self, coin_id: Optional[str] = None) -> List[Path]:
        """
        List processed data files.
        
        Args:
            coin_id: Optional filter by coin ID
            
        Returns:
            List of file paths
        """
        if coin_id:
            pattern = f"{coin_id}_*.parquet"
        else:
            pattern = "*.parquet"
            
        return list(self.processed_path.glob(pattern))
        
    def check_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Check data quality.
        
        Args:
            df: DataFrame to check
            
        Returns:
            Dictionary with quality metrics
        """
        quality = {
            'total_rows': len(df),
            'duplicates': df.duplicated().sum(),
            'missing_values': df.isnull().sum().to_dict(),
            'timestamp_monotonic': False,
            'timestamp_gaps': []
        }
        
        # Check timestamp monotonicity
        if 'timestamp' in df.columns:
            quality['timestamp_monotonic'] = df['timestamp'].is_monotonic_increasing
            
            # Check for gaps (assuming hourly data)
            time_diffs = df['timestamp'].diff()
            expected_diff = pd.Timedelta(hours=1)
            gaps = time_diffs[time_diffs > expected_diff * 1.5]
            quality['timestamp_gaps'] = len(gaps)
            
        logger.info(f"Quality check: {quality}")
        return quality
