"""
Price-based feature transforms and engineering.
"""
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureTransforms:
    """
    Create price-based features and transformations.
    """
    
    def __init__(self, config: dict = None):
        """
        Initialize with configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
    def add_returns(
        self,
        df: pd.DataFrame,
        periods: list = [1, 3, 6, 12, 24]
    ) -> pd.DataFrame:
        """
        Add log returns for multiple periods.
        
        Args:
            df: DataFrame with price data
            periods: List of periods for returns calculation
            
        Returns:
            DataFrame with return features
        """
        for period in periods:
            df[f'return_{period}h'] = np.log(df['close'] / df['close'].shift(period))
            
        return df
        
    def add_rolling_stats(
        self,
        df: pd.DataFrame,
        windows: list = [6, 12, 24, 48, 72]
    ) -> pd.DataFrame:
        """
        Add rolling statistics (mean, std, min, max).
        
        Args:
            df: DataFrame with price data
            windows: List of window sizes in hours
            
        Returns:
            DataFrame with rolling features
        """
        for window in windows:
            # Rolling mean
            df[f'rolling_mean_{window}h'] = df['close'].rolling(window).mean()
            
            # Rolling std (volatility)
            df[f'rolling_std_{window}h'] = df['close'].rolling(window).std()
            
            # Rolling min/max
            df[f'rolling_min_{window}h'] = df['close'].rolling(window).min()
            df[f'rolling_max_{window}h'] = df['close'].rolling(window).max()
            
            # Price position within range
            df[f'price_position_{window}h'] = (
                (df['close'] - df[f'rolling_min_{window}h']) /
                (df[f'rolling_max_{window}h'] - df[f'rolling_min_{window}h'])
            )
            
        return df
        
    def add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add volatility-based features.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with volatility features
        """
        # High-Low range
        df['hl_range'] = df['high'] - df['low']
        df['hl_range_pct'] = (df['high'] - df['low']) / df['close']
        
        # Close relative to High-Low
        df['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
        
        # Open-Close change
        df['oc_change'] = df['close'] - df['open']
        df['oc_change_pct'] = (df['close'] - df['open']) / df['open']
        
        return df
        
    def add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add momentum-based features.
        
        Args:
            df: DataFrame with price data
            
        Returns:
            DataFrame with momentum features
        """
        # Rate of change
        for period in [3, 6, 12, 24]:
            df[f'roc_{period}h'] = (
                (df['close'] - df['close'].shift(period)) / df['close'].shift(period)
            )
            
        # Price acceleration (2nd derivative)
        df['price_acceleration'] = df['return_1h'].diff()
        
        return df
        
    def add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add volume-based features.
        
        Args:
            df: DataFrame with volume data
            
        Returns:
            DataFrame with volume features
        """
        if 'volume' not in df.columns:
            logger.warning("No volume data available")
            return df
            
        # Volume changes
        df['volume_change'] = df['volume'].pct_change()
        
        # Rolling volume stats
        for window in [6, 12, 24]:
            df[f'volume_ma_{window}h'] = df['volume'].rolling(window).mean()
            df[f'volume_ratio_{window}h'] = df['volume'] / df[f'volume_ma_{window}h']
            
        # Price-volume correlation
        df['price_volume_corr_24h'] = (
            df['close'].rolling(24).corr(df['volume'])
        )
        
        return df
        
    def add_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add microstructure features if bid/ask data available.
        
        Args:
            df: DataFrame with bid/ask data
            
        Returns:
            DataFrame with microstructure features
        """
        if 'bid' in df.columns and 'ask' in df.columns:
            df['spread'] = df['ask'] - df['bid']
            df['mid_price'] = (df['ask'] + df['bid']) / 2
            df['relative_spread'] = df['spread'] / df['mid_price']
        else:
            logger.info("No bid/ask data for microstructure features")
            
        return df
        
    def add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add temporal features (hour, day of week, weekend).
        
        Args:
            df: DataFrame with timestamp
            
        Returns:
            DataFrame with temporal features
        """
        if 'timestamp' in df.columns:
            df['hour'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        return df

    def add_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add market regime features (volatility, trend).
        
        Args:
            df: DataFrame with price data
            
        Returns:
            DataFrame with regime features
        """
        # Volatility Regime (Low/Medium/High)
        # Use rolling_std_24h if available, else calculate it
        if 'rolling_std_24h' not in df.columns:
            df['rolling_std_24h'] = df['close'].rolling(24).std()
            
        # We use quantile-based discretization for regimes
        # Note: In production, these thresholds should be fixed from training data
        # to avoid look-ahead bias. For now we use rolling quantiles.
        
        # Calculate rolling quantiles (30-day window) to determine relative volatility
        rolling_30d = df['rolling_std_24h'].rolling(720, min_periods=24)
        low_threshold = rolling_30d.quantile(0.33)
        high_threshold = rolling_30d.quantile(0.66)
        
        # 0=Low, 1=Medium, 2=High
        df['volatility_regime'] = 1  # Default Medium
        df.loc[df['rolling_std_24h'] <= low_threshold, 'volatility_regime'] = 0
        df.loc[df['rolling_std_24h'] >= high_threshold, 'volatility_regime'] = 2
        
        # Trend Strength (ADX-based)
        if 'adx_14' in df.columns:
            # 0=Choppy (<25), 1=Trending (>25)
            df['adx_regime'] = (df['adx_14'] > 25).astype(int)
            
        # Trend Strength (EMA-based)
        if 'ema_20' in df.columns and 'ema_50' in df.columns:
            df['trend_strength'] = abs(df['ema_20'] - df['ema_50']) / df['close']
            
        return df
        
    def add_lag_features(
        self,
        df: pd.DataFrame,
        columns: list,
        lags: list = [1, 2, 3, 6, 12]
    ) -> pd.DataFrame:
        """
        Add lagged features.
        
        Args:
            df: DataFrame
            columns: Columns to create lags for
            lags: List of lag periods
            
        Returns:
            DataFrame with lag features
        """
        for col in columns:
            if col in df.columns:
                for lag in lags:
                    df[f'{col}_lag_{lag}'] = df[col].shift(lag)
                    
        return df
        
    def normalize_features(
        self,
        df: pd.DataFrame,
        columns: list,
        window: int = 720
    ) -> pd.DataFrame:
        """
        Rolling z-score normalization.
        
        Args:
            df: DataFrame
            columns: Columns to normalize
            window: Rolling window for stats calculation
            
        Returns:
            DataFrame with normalized features
        """
        for col in columns:
            if col in df.columns:
                rolling_mean = df[col].rolling(window, min_periods=1).mean()
                rolling_std = df[col].rolling(window, min_periods=1).std()
                df[f'{col}_norm'] = (df[col] - rolling_mean) / (rolling_std + 1e-8)
                
        return df
        
    def create_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create all price-based features.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with all features
        """
        logger.info("Creating price-based features...")
        
        # Get config parameters - USE SMALLER WINDOWS to preserve data
        rolling_windows = self.config.get('rolling_windows', [6, 12, 24])  # Max 24h instead of 72h
        lag_periods = self.config.get('lag_periods', [1, 2, 3, 6])  # Max 6 instead of 12
        
        # Add features
        df = self.add_returns(df)
        df = self.add_rolling_stats(df, rolling_windows)
        df = self.add_volatility_features(df)
        df = self.add_momentum_features(df)
        df = self.add_volume_features(df)
        df = self.add_microstructure_features(df)
        df = self.add_temporal_features(df)
        df = self.add_regime_features(df)
        
        # Add lags for key features
        lag_columns = [
            'rsi_14', 'macd', 'ema_20',
            'return_1h', 'rolling_std_24h', 'volume_ratio_24h',
            'volatility_regime', 'adx_regime'
        ]
        df = self.add_lag_features(df, lag_columns, lag_periods)
        
        logger.info(f"Created features, total columns: {len(df.columns)}")
        
        return df
