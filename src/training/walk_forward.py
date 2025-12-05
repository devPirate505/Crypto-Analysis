"""
Walk-forward cross-validation for time series.
"""
import pandas as pd
import numpy as np
from datetime import timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WalkForwardCV:
    """
    Implements walk-forward cross-validation for time series data.
    """
    
    def __init__(
        self,
        train_size_days: int = 180,
        validation_size_days: int = 30,
        test_size_days: int = 30,
        step_size_days: int = 30
    ):
        """
        Initialize walk-forward CV.
        
        Args:
            train_size_days: Size of training window in days
            validation_size_days: Size of validation window in days
            test_size_days: Size of test window in days
            step_size_days: Step size between folds in days
        """
        self.train_size = timedelta(days=train_size_days)
        self.val_size = timedelta(days=validation_size_days)
        self.test_size = timedelta(days=test_size_days)
        self.step_size = timedelta(days=step_size_days)
        
    def split(self, df: pd.DataFrame, time_column: str = 'timestamp'):
        """
        Generate train/validation/test splits.
        
        Args:
            df: DataFrame with time series data
            time_column: Name of timestamp column
            
        Yields:
            Tuples of (train_df, val_df, test_df, fold_number)
        """
        # Ensure dataframe is sorted by time
        df = df.sort_values(time_column).reset_index(drop=True)
        
        start_date = df[time_column].min()
        end_date = df[time_column].max()
        
        # Calculate first possible test start
        current_date = start_date + self.train_size + self.val_size
        
        fold = 0
        
        while current_date + self.test_size <= end_date:
            # Define window boundaries
            train_start = current_date - self.train_size - self.val_size
            train_end = current_date - self.val_size
            val_start = train_end
            val_end = current_date
            test_start = current_date
            test_end = current_date + self.test_size
            
            # Split data
            train_df = df[
                (df[time_column] >= train_start) &
                (df[time_column] < train_end)
            ]
            
            val_df = df[
                (df[time_column] >= val_start) &
                (df[time_column] < val_end)
            ]
            
            test_df = df[
                (df[time_column] >= test_start) &
                (df[time_column] < test_end)
            ]
            
            logger.info(f"Fold {fold}: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
            logger.info(f"  Train: {train_start.date()} to {train_end.date()}")
            logger.info(f"  Val: {val_start.date()} to {val_end.date()}")
            logger.info(f"  Test: {test_start.date()} to {test_end.date()}")
            
            yield train_df, val_df, test_df, fold
            
            # Move forward
            current_date += self.step_size
            fold += 1
            
        logger.info(f"Generated {fold} folds")
        
    def simple_split(
        self,
        df: pd.DataFrame,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        time_column: str = 'timestamp'
    ):
        """
        Simple chronological train/val/test split (non-overlapping).
        
        Args:
            df: DataFrame with time series data
            train_ratio: Proportion for training
            val_ratio: Proportion for validation
            time_column: Name of timestamp column
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        df = df.sort_values(time_column).reset_index(drop=True)
        
        n = len(df)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        train_df = df.iloc[:train_end]
        val_df = df.iloc[train_end:val_end]
        test_df = df.iloc[val_end:]
        
        logger.info(f"Simple split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
        
        return train_df, val_df, test_df
