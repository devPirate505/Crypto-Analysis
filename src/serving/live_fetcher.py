"""
Live data fetcher for real-time crypto prices from Binance.
"""
from binance.client import Client
import pandas as pd
import numpy as np
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()


# Coin ID to Binance symbol mapping
COIN_TO_SYMBOL = {
    'bitcoin': 'BTCUSDT',
    'ethereum': 'ETHUSDT', 
    'binancecoin': 'BNBUSDT',
}


class LivePriceFetcher:
    """Fetch real-time prices from Binance."""
    
    def __init__(self):
        api_key = os.getenv('BINANCE_API_KEY', '')
        api_secret = os.getenv('BINANCE_API_SECRET', '')
        self.client = Client(api_key, api_secret)
        
    def get_current_price(self, coin_id: str) -> float:
        """Get current price for a coin."""
        symbol = COIN_TO_SYMBOL.get(coin_id)
        if not symbol:
            raise ValueError(f"Unknown coin: {coin_id}")
            
        ticker = self.client.get_symbol_ticker(symbol=symbol)
        return float(ticker['price'])
        
    def get_recent_klines(self, coin_id: str, interval: str = '1h', limit: int = 100) -> pd.DataFrame:
        """Get recent klines for feature calculation."""
        symbol = COIN_TO_SYMBOL.get(coin_id)
        if not symbol:
            raise ValueError(f"Unknown coin: {coin_id}")
            
        klines = self.client.get_klines(
            symbol=symbol,
            interval=interval,
            limit=limit
        )
        
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
            
        return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        
    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate ATR for stop-loss calculation."""
        high = df['high']
        low = df['low'] 
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(period).mean().iloc[-1]
        
        return atr
        
    def get_live_data_with_features(self, coin_id: str) -> dict:
        """
        Get live data with calculated features for prediction.
        
        Returns dict with:
        - current_price: Current market price
        - atr: Average True Range (for stop-loss)
        - recent_high: Recent swing high
        - recent_low: Recent swing low
        - klines: Recent OHLCV data
        """
        # Get recent data
        klines = self.get_recent_klines(coin_id, '1h', 100)
        
        current_price = klines['close'].iloc[-1]
        atr = self.calculate_atr(klines)
        
        # Recent swing high/low (last 24 hours)
        recent_high = klines['high'].tail(24).max()
        recent_low = klines['low'].tail(24).min()
        
        return {
            'current_price': current_price,
            'atr': atr,
            'recent_high': recent_high,
            'recent_low': recent_low,
            'klines': klines,
            'timestamp': datetime.now()
        }


def calculate_sl_tp(entry_price: float, atr: float, predicted_return: float, 
                    sl_multiplier: float = 2.0, risk_reward: float = 2.0) -> dict:
    """
    Calculate Stop Loss and Take Profit levels.
    
    Args:
        entry_price: Entry price
        atr: Average True Range
        predicted_return: Model's predicted return
        sl_multiplier: ATR multiplier for stop loss (default 2x ATR)
        risk_reward: Risk/Reward ratio (default 2:1)
        
    Returns:
        dict with stop_loss, take_profit, risk_amount, reward_amount
    """
    # Stop loss based on ATR
    stop_loss = entry_price - (atr * sl_multiplier)
    risk_amount = entry_price - stop_loss
    
    # Take profit based on risk:reward ratio
    reward_amount = risk_amount * risk_reward
    take_profit = entry_price + reward_amount
    
    # Calculate percentages
    sl_pct = (stop_loss - entry_price) / entry_price * 100
    tp_pct = (take_profit - entry_price) / entry_price * 100
    
    return {
        'stop_loss': stop_loss,
        'take_profit': take_profit,
        'risk_amount': risk_amount,
        'reward_amount': reward_amount,
        'sl_pct': sl_pct,
        'tp_pct': tp_pct,
        'risk_reward': risk_reward
    }


if __name__ == "__main__":
    # Test
    fetcher = LivePriceFetcher()
    
    for coin_id in ['bitcoin', 'ethereum', 'binancecoin']:
        try:
            data = fetcher.get_live_data_with_features(coin_id)
            print(f"\n{coin_id.upper()}:")
            print(f"  Price: ${data['current_price']:,.2f}")
            print(f"  ATR: ${data['atr']:,.2f}")
            print(f"  24h High: ${data['recent_high']:,.2f}")
            print(f"  24h Low: ${data['recent_low']:,.2f}")
            
            # Example SL/TP
            levels = calculate_sl_tp(data['current_price'], data['atr'], 0.001)
            print(f"  Stop Loss: ${levels['stop_loss']:,.2f} ({levels['sl_pct']:.2f}%)")
            print(f"  Take Profit: ${levels['take_profit']:,.2f} ({levels['tp_pct']:.2f}%)")
        except Exception as e:
            print(f"Error for {coin_id}: {e}")
