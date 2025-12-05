"""
Advanced trading strategy with:
1. Ensemble predictions (LightGBM + RandomForest)
2. ATR-based adaptive stop-losses
3. Kelly Criterion position sizing
4. Volatility regime filtering
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
from src.models.lightgbm_model import LightGBMModel
from src.models.rf_model import RandomForestModel


class AdvancedStrategy:
    """
    Advanced trading strategy with ensemble predictions and risk management.
    """
    
    def __init__(
        self,
        buy_threshold: float = 0.0003,
        atr_multiplier: float = 2.0,
        max_position_size: float = 0.2,
        use_kelly: bool = True,
        use_regime: bool = True
    ):
        """
        Args:
            buy_threshold: Minimum predicted return to trade
            atr_multiplier: ATR multiplier for stop-loss (e.g., 2x ATR)
            max_position_size: Maximum position as fraction of capital
            use_kelly: Whether to use Kelly Criterion for sizing
            use_regime: Whether to filter by volatility regime
        """
        self.buy_threshold = buy_threshold
        self.atr_multiplier = atr_multiplier
        self.max_position_size = max_position_size
        self.use_kelly = use_kelly
        self.use_regime = use_regime
        
    def calculate_kelly_fraction(self, win_rate: float, avg_win: float, avg_loss: float) -> float:
        """
        Calculate Kelly Criterion fraction for optimal position sizing.
        
        Kelly % = W - [(1-W) / R]
        Where: W = win probability, R = win/loss ratio
        """
        if avg_loss == 0 or win_rate <= 0:
            return 0.0
            
        win_loss_ratio = abs(avg_win / avg_loss)
        kelly = win_rate - ((1 - win_rate) / win_loss_ratio)
        
        # Use half-Kelly for safety
        half_kelly = kelly / 2
        
        # Clamp to max position size
        return max(0, min(half_kelly, self.max_position_size))
        
    def calculate_stop_loss(self, atr_value: float, entry_price: float) -> float:
        """
        Calculate stop-loss price based on ATR.
        
        Args:
            atr_value: Current ATR value
            entry_price: Entry price for the trade
            
        Returns:
            Stop-loss price
        """
        stop_distance = atr_value * self.atr_multiplier
        return entry_price - stop_distance
        
    def generate_signals(
        self,
        lgb_predictions: np.ndarray,
        rf_predictions: np.ndarray,
        volatility_regimes: np.ndarray = None,
        atr_values: np.ndarray = None,
        prices: np.ndarray = None
    ) -> dict:
        """
        Generate trading signals with ensemble and risk management.
        
        Returns:
            dict with 'positions', 'stop_losses', 'position_sizes'
        """
        n = len(lgb_predictions)
        
        # Ensemble: weighted average (LightGBM gets more weight due to better performance)
        ensemble_predictions = 0.7 * lgb_predictions + 0.3 * rf_predictions
        
        positions = np.zeros(n)
        stop_losses = np.zeros(n)
        position_sizes = np.full(n, self.max_position_size)  # Default size
        
        for i in range(n):
            pred_return = ensemble_predictions[i]
            
            # Check if predicted return exceeds threshold
            if pred_return > self.buy_threshold:
                # Check volatility regime if enabled
                if self.use_regime and volatility_regimes is not None:
                    # Only trade in low/medium volatility (not high=2)
                    if volatility_regimes[i] >= 2:
                        continue
                        
                positions[i] = 1  # Long signal
                
                # Calculate stop-loss if ATR available
                if atr_values is not None and prices is not None:
                    stop_losses[i] = self.calculate_stop_loss(atr_values[i], prices[i])
                    
        return {
            'positions': positions,
            'stop_losses': stop_losses,
            'position_sizes': position_sizes,
            'ensemble_predictions': ensemble_predictions
        }


class AdvancedBacktester:
    """
    Backtester with stop-loss and position sizing support.
    """
    
    def __init__(
        self,
        initial_capital: float = 10000,
        commission: float = 0.001,
        slippage: float = 0.0005
    ):
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        
    def run(
        self,
        df: pd.DataFrame,
        positions: np.ndarray,
        stop_losses: np.ndarray = None,
        position_sizes: np.ndarray = None
    ) -> pd.DataFrame:
        """
        Run backtest with stop-losses and position sizing.
        """
        n = len(df)
        prices = df['close'].values
        
        # Initialize tracking
        equity = np.full(n, self.initial_capital, dtype=float)
        cash = self.initial_capital
        holdings = 0
        entry_price = 0
        current_stop = 0
        
        trades = []
        
        for i in range(n):
            price = prices[i]
            pos = positions[i]
            pos_size = position_sizes[i] if position_sizes is not None else 0.2
            stop = stop_losses[i] if stop_losses is not None else 0
            
            # Check stop-loss
            if holdings > 0 and current_stop > 0:
                if price <= current_stop:
                    # Stop-loss triggered
                    sell_value = holdings * price * (1 - self.commission - self.slippage)
                    cash += sell_value
                    pnl = (price - entry_price) / entry_price
                    trades.append({'type': 'stop', 'pnl': pnl, 'idx': i})
                    holdings = 0
                    entry_price = 0
                    current_stop = 0
                    
            # Entry signal
            if pos == 1 and holdings == 0:
                # Calculate position size
                trade_value = cash * pos_size
                buy_cost = trade_value * (1 + self.commission + self.slippage)
                
                if buy_cost <= cash:
                    holdings = trade_value / price
                    cash -= buy_cost
                    entry_price = price
                    current_stop = stop if stop > 0 else 0
                    
            # Exit on signal change
            elif pos == 0 and holdings > 0:
                sell_value = holdings * price * (1 - self.commission - self.slippage)
                cash += sell_value
                pnl = (price - entry_price) / entry_price
                trades.append({'type': 'exit', 'pnl': pnl, 'idx': i})
                holdings = 0
                entry_price = 0
                current_stop = 0
                
            # Update equity
            equity[i] = cash + holdings * price
            
        # Create result dataframe
        result_df = df.copy()
        result_df['equity'] = equity
        result_df['returns'] = pd.Series(equity).pct_change().fillna(0)
        result_df['cumulative_returns'] = (1 + result_df['returns']).cumprod() - 1
        
        # Calculate metrics
        total_return = (equity[-1] - self.initial_capital) / self.initial_capital
        
        if len(trades) > 0:
            wins = [t for t in trades if t['pnl'] > 0]
            losses = [t for t in trades if t['pnl'] <= 0]
            win_rate = len(wins) / len(trades)
            avg_win = np.mean([t['pnl'] for t in wins]) if wins else 0
            avg_loss = np.mean([t['pnl'] for t in losses]) if losses else 0
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            
        result_df.attrs['metrics'] = {
            'total_return': total_return,
            'total_trades': len(trades),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'final_equity': equity[-1]
        }
        
        return result_df


def load_ensemble_models(coin_id: str):
    """Load both LightGBM and RandomForest models."""
    lgb_model = LightGBMModel(task_type='regression')
    rf_model = RandomForestModel(task_type='regression')
    
    lgb_path = f"models/{coin_id}_lightgbm_optimized.joblib"
    rf_path = f"models/{coin_id}_random_forest.joblib"
    
    if Path(lgb_path).exists():
        lgb_model.load(lgb_path)
    else:
        lgb_model.load(f"models/{coin_id}_lightgbm.joblib")
        
    if Path(rf_path).exists():
        rf_model.load(rf_path)
        
    return lgb_model, rf_model
