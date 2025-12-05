"""
Technical indicators using the ta library.
"""
import pandas as pd
import numpy as np
from ta import momentum, trend, volatility, volume
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TechnicalIndicators:
    """
    Calculate technical indicators for cryptocurrency data.
    """
    
    def __init__(self, config: dict = None):
        """
        Initialize with configuration.
        
        Args:
            config: Configuration dictionary with indicator parameters
        """
        self.config = config or {}
        
    def add_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Add RSI (Relative Strength Index)."""
        df[f'rsi_{period}'] = momentum.RSIIndicator(
            close=df['close'],
            window=period
        ).rsi()
        return df
        
    def add_macd(
        self,
        df: pd.DataFrame,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> pd.DataFrame:
        """Add MACD indicators."""
        macd = trend.MACD(
            close=df['close'],
            window_fast=fast,
            window_slow=slow,
            window_sign=signal
        )
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()
        return df
        
    def add_ema(self, df: pd.DataFrame, periods: list = [5, 10, 20, 50]) -> pd.DataFrame:
        """Add Exponential Moving Averages."""
        for period in periods:
            df[f'ema_{period}'] = trend.EMAIndicator(
                close=df['close'],
                window=period
            ).ema_indicator()
        return df
        
    def add_bollinger_bands(self, df: pd.DataFrame, period: int = 20, std: int = 2) -> pd.DataFrame:
        """Add Bollinger Bands."""
        bb = volatility.BollingerBands(
            close=df['close'],
            window=period,
            window_dev=std
        )
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_middle'] = bb.bollinger_mavg()
        df['bb_lower'] = bb.bollinger_lband()
        df['bb_width'] = bb.bollinger_wband()
        df['bb_pband'] = bb.bollinger_pband()
        return df
        
    def add_adx(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Add ADX (Average Directional Index)."""
        adx = trend.ADXIndicator(
            high=df['high'],
            low=df['low'],
            close=df['close'],
            window=period
        )
        df[f'adx_{period}'] = adx.adx()
        df[f'adx_pos'] = adx.adx_pos()
        df[f'adx_neg'] = adx.adx_neg()
        return df
        
    def add_cci(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """Add CCI (Commodity Channel Index)."""
        df[f'cci_{period}'] = trend.CCIIndicator(
            high=df['high'],
            low=df['low'],
            close=df['close'],
            window=period
        ).cci()
        return df
        
    def add_atr(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Add ATR (Average True Range)."""
        df[f'atr_{period}'] = volatility.AverageTrueRange(
            high=df['high'],
            low=df['low'],
            close=df['close'],
            window=period
        ).average_true_range()
        return df
        
    def add_stochastic(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Add Stochastic Oscillator."""
        stoch = momentum.StochasticOscillator(
            high=df['high'],
            low=df['low'],
            close=df['close'],
            window=period
        )
        df[f'stoch_{period}'] = stoch.stoch()
        df[f'stoch_signal_{period}'] = stoch.stoch_signal()
        return df
        
    def add_obv(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add On-Balance Volume."""
        if 'volume' in df.columns:
            df['obv'] = volume.OnBalanceVolumeIndicator(
                close=df['close'],
                volume=df['volume']
            ).on_balance_volume()
        return df
        
    def add_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add all technical indicators.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with all indicators added
        """
        logger.info("Adding technical indicators...")
        
        # Get parameters from config or use defaults
        rsi_period = self.config.get('rsi_period', 14)
        macd_fast = self.config.get('macd_fast', 12)
        macd_slow = self.config.get('macd_slow', 26)
        macd_signal = self.config.get('macd_signal', 9)
        ema_periods = self.config.get('ema_periods', [5, 10, 20, 50])
        adx_period = self.config.get('adx_period', 14)
        atr_period = self.config.get('atr_period', 14)
        
        # Add indicators
        df = self.add_rsi(df, rsi_period)
        df = self.add_macd(df, macd_fast, macd_slow, macd_signal)
        df = self.add_ema(df, ema_periods)
        df = self.add_bollinger_bands(df)
        df = self.add_adx(df, adx_period)
        df = self.add_cci(df)
        df = self.add_atr(df, atr_period)
        df = self.add_stochastic(df)
        df = self.add_obv(df)
        
        logger.info(f"Added {len(df.columns)} total columns (including indicators)")
        
        return df
