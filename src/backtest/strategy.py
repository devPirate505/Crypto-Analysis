"""
Trading strategy based on model predictions.
"""
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MLStrategy:
    """
    Machine learning-based trading strategy.
    """
    
    def __init__(
        self,
        probability_threshold: float = 0.55,
        stop_loss: float = 0.02,
        take_profit: float = 0.04,
        position_size: float = 1.0
    ):
        """
        Initialize strategy.
        
        Args:
            probability_threshold: Minimum probability to take a position
            stop_loss: Stop loss percentage (e.g., 0.02 = 2%)
            take_profit: Take profit percentage (e.g., 0.04 = 4%)
            position_size: Position size as fraction of capital (1.0 = 100%)
        """
        self.prob_threshold = probability_threshold
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.position_size = position_size
        
    def generate_signals(self, predictions: np.ndarray, probabilities: np.ndarray = None) -> pd.Series:
        """
        Generate trading signals from model predictions.
        
        Args:
            predictions: Binary predictions (0 or 1)
            probabilities: Prediction probabilities (optional)
            
        Returns:
            Series of signals: 1 (buy), 0 (hold), -1 (sell)
        """
        signals = pd.Series(0, index=range(len(predictions)))
        
        if probabilities is not None:
            # Use probability threshold
            signals[probabilities > self.prob_threshold] = 1
            signals[probabilities < (1 - self.prob_threshold)] = -1
        else:
            # Use binary predictions
            signals[predictions == 1] = 1
            signals[predictions == 0] = -1
            
        return signals
        
    def apply_risk_management(
        self,
        df: pd.DataFrame,
        signals: pd.Series,
        entry_price_col: str = 'close'
    ) -> pd.DataFrame:
        """
        Apply stop-loss and take-profit rules.
        
        Args:
            df: DataFrame with price data
            signals: Trading signals
            entry_price_col: Column to use for entry price
            
        Returns:
            DataFrame with risk-managed signals
        """
        df = df.copy()
        df['signal'] = signals
        df['position'] = 0
        
        current_position = 0
        entry_price = 0
        
        for i in range(len(df)):
            if current_position == 0:
                # No position, check for entry signal
                if df.loc[i, 'signal'] == 1:
                    current_position = 1
                    entry_price = df.loc[i, entry_price_col]
                    df.loc[i, 'position'] = 1
                elif df.loc[i, 'signal'] == -1:
                    current_position = -1
                    entry_price = df.loc[i, entry_price_col]
                    df.loc[i, 'position'] = -1
            else:
                # In position, check for exit conditions
                current_price = df.loc[i, entry_price_col]
                
                if current_position == 1:
                    # Long position
                    pnl_pct = (current_price - entry_price) / entry_price
                    
                    if pnl_pct <= -self.stop_loss or pnl_pct >= self.take_profit:
                        # Exit
                        current_position = 0
                        df.loc[i, 'position'] = 0
                    else:
                        df.loc[i, 'position'] = 1
                        
                elif current_position == -1:
                    # Short position
                    pnl_pct = (entry_price - current_price) / entry_price
                    
                    if pnl_pct <= -self.stop_loss or pnl_pct >= self.take_profit:
                        # Exit
                        current_position = 0
                        df.loc[i, 'position'] = 0
                    else:
                        df.loc[i, 'position'] = -1
                        
        return df


class SimpleRebalanceStrategy:
    """
    Simple rebalancing strategy that rebalances at each period.
    """
    
    def __init__(self, probability_threshold: float = 0.55):
        """
        Initialize strategy.
        
        Args:
            probability_threshold: Minimum probability to go long
        """
        self.prob_threshold = probability_threshold
        
    def generate_positions(self, probabilities: np.ndarray) -> pd.Series:
        """
        Generate positions from probabilities.
        
        Args:
            probabilities: Prediction probabilities for upward movement
            
        Returns:
            Series of positions: 1 (long), 0 (neutral), -1 (short)
        """
        positions = pd.Series(0, index=range(len(probabilities)))
        
        # Long if probability > threshold
        positions[probabilities > self.prob_threshold] = 1
        
        # Short if probability < (1 - threshold)
        positions[probabilities < (1 - self.prob_threshold)] = -1
        
        return positions
