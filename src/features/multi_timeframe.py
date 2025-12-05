"""
Multi-timeframe data fetcher for 4H and Daily candles from Binance.
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from binance.client import Client
from binance.exceptions import BinanceAPIException
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import os
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Coin ID to Binance symbol mapping
COIN_TO_SYMBOL = {
    'bitcoin': 'BTCUSDT',
    'ethereum': 'ETHUSDT',
    'binancecoin': 'BNBUSDT',
}


class MultiTimeframeFetcher:
    """Fetch multiple timeframe data from Binance."""
    
    def __init__(self):
        api_key = os.getenv('BINANCE_API_KEY', '')
        api_secret = os.getenv('BINANCE_API_SECRET', '')
        self.client = Client(api_key, api_secret)
        
    def fetch_klines(self, symbol: str, interval: str, days: int = 730) -> pd.DataFrame:
        """
        Fetch klines for a specific interval.
        
        Args:
            symbol: Binance symbol (e.g., 'BTCUSDT')
            interval: Binance interval (e.g., '4h', '1d')
            days: Number of days of history
            
        Returns:
            DataFrame with OHLCV data
        """
        logger.info(f"Fetching {interval} data for {symbol} ({days} days)...")
        
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
        
        klines = self.client.get_historical_klines(
            symbol,
            interval,
            start_time.strftime("%d %b %Y %H:%M:%S"),
            end_time.strftime("%d %b %Y %H:%M:%S")
        )
        
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        
        # Convert types
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
            
        # Keep only needed columns
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        
        logger.info(f"Fetched {len(df)} {interval} candles for {symbol}")
        return df
        
    def fetch_all_timeframes(self, coin_id: str, days: int = 730) -> dict:
        """
        Fetch 1H, 4H, and Daily data for a coin.
        
        Returns:
            Dict with '1h', '4h', '1d' DataFrames
        """
        symbol = COIN_TO_SYMBOL.get(coin_id)
        if not symbol:
            raise ValueError(f"Unknown coin: {coin_id}")
            
        data = {
            '4h': self.fetch_klines(symbol, Client.KLINE_INTERVAL_4HOUR, days),
            '1d': self.fetch_klines(symbol, Client.KLINE_INTERVAL_1DAY, days),
        }
        
        return data


def calculate_htf_features(df: pd.DataFrame, suffix: str) -> pd.DataFrame:
    """
    Calculate trend features from higher timeframe data.
    
    Args:
        df: OHLCV DataFrame
        suffix: Suffix to add to column names (e.g., '_4h', '_1d')
        
    Returns:
        DataFrame with HTF features
    """
    features = pd.DataFrame()
    features['timestamp'] = df['timestamp']
    
    # Trend direction (EMA-based)
    df['ema_10'] = df['close'].ewm(span=10).mean()
    df['ema_20'] = df['close'].ewm(span=20).mean()
    features[f'trend{suffix}'] = (df['ema_10'] > df['ema_20']).astype(int)  # 1=bullish, 0=bearish
    
    # Trend strength
    features[f'trend_strength{suffix}'] = abs(df['ema_10'] - df['ema_20']) / df['close']
    
    # Price position in range (0-1)
    rolling_high = df['high'].rolling(20).max()
    rolling_low = df['low'].rolling(20).min()
    features[f'price_position{suffix}'] = (df['close'] - rolling_low) / (rolling_high - rolling_low + 1e-8)
    
    # Volatility (ATR-like)
    tr = pd.DataFrame()
    tr['hl'] = df['high'] - df['low']
    tr['hc'] = abs(df['high'] - df['close'].shift(1))
    tr['lc'] = abs(df['low'] - df['close'].shift(1))
    df['atr'] = tr.max(axis=1).rolling(14).mean()
    features[f'volatility{suffix}'] = df['atr'] / df['close']
    
    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-8)
    features[f'rsi{suffix}'] = 100 - (100 / (1 + rs))
    
    # MACD trend
    ema_12 = df['close'].ewm(span=12).mean()
    ema_26 = df['close'].ewm(span=26).mean()
    macd = ema_12 - ema_26
    signal = macd.ewm(span=9).mean()
    features[f'macd_trend{suffix}'] = (macd > signal).astype(int)  # 1=bullish, 0=bearish
    
    # Volume trend
    features[f'volume_trend{suffix}'] = (df['volume'] > df['volume'].rolling(20).mean()).astype(int)
    
    return features


def merge_mtf_features(df_1h: pd.DataFrame, df_4h: pd.DataFrame, df_1d: pd.DataFrame) -> pd.DataFrame:
    """
    Merge 4H and Daily features into 1H dataframe.
    
    Uses forward-fill to align higher timeframe features with 1H data.
    """
    logger.info("Calculating HTF features...")
    
    # Calculate features for each timeframe
    features_4h = calculate_htf_features(df_4h, '_4h')
    features_1d = calculate_htf_features(df_1d, '_1d')
    
    # Merge with 1H data using asof join (forward-fill)
    df_1h = df_1h.sort_values('timestamp')
    features_4h = features_4h.sort_values('timestamp')
    features_1d = features_1d.sort_values('timestamp')
    
    # Merge 4H features
    df_merged = pd.merge_asof(
        df_1h, 
        features_4h,
        on='timestamp',
        direction='backward'
    )
    
    # Merge Daily features
    df_merged = pd.merge_asof(
        df_merged,
        features_1d,
        on='timestamp', 
        direction='backward'
    )
    
    logger.info(f"Merged MTF features. New columns: {len(df_merged.columns)}")
    
    return df_merged


if __name__ == "__main__":
    # Test fetch
    fetcher = MultiTimeframeFetcher()
    
    for coin_id in ['bitcoin', 'ethereum', 'binancecoin']:
        try:
            data = fetcher.fetch_all_timeframes(coin_id, days=730)
            print(f"\n{coin_id}:")
            print(f"  4H: {len(data['4h'])} candles")
            print(f"  1D: {len(data['1d'])} candles")
        except Exception as e:
            print(f"Error for {coin_id}: {e}")
