"""
Backtesting engine for strategy evaluation.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Backtester:
    """
    Backtest trading strategies on historical data.
    """
    
    def __init__(
        self,
        initial_capital: float = 10000,
        commission: float = 0.001,
        slippage: float = 0.0005
    ):
        """
        Initialize backtester.
        
        Args:
            initial_capital: Starting capital
            commission: Trading commission (e.g., 0.001 = 0.1%)
            slippage: Slippage (e.g., 0.0005 = 0.05%)
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        
    def run(
        self,
        df: pd.DataFrame,
        positions: pd.Series,
        price_col: str = 'close'
    ) -> pd.DataFrame:
        """
        Run backtest.
        
        Args:
            df: DataFrame with price data
            positions: Series of positions (1 = long, -1 = short, 0 = neutral)
            price_col: Column to use for prices
            
        Returns:
            DataFrame with backtest results
        """
        df = df.copy()
        df['position'] = positions
        
        # Calculate returns
        df['returns'] = df[price_col].pct_change()
        
        # Position changes (for transaction costs)
        df['position_change'] = df['position'].diff().fillna(0)
        
        # Strategy returns (position * market return)
        df['strategy_returns'] = df['position'].shift(1) * df['returns']
        
        # Apply transaction costs
        df['transaction_costs'] = np.abs(df['position_change']) * (self.commission + self.slippage)
        df['strategy_returns_net'] = df['strategy_returns'] - df['transaction_costs']
        
        # Cumulative returns
        df['cumulative_returns'] = (1 + df['returns']).cumprod() - 1
        df['cumulative_strategy_returns'] = (1 + df['strategy_returns_net']).cumprod() - 1
        
        # Equity curve
        df['equity'] = self.initial_capital * (1 + df['cumulative_strategy_returns'])
        
        return df
        
    def calculate_metrics(self, backtest_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate performance metrics.
        
        Args:
            backtest_df: DataFrame with backtest results
            
        Returns:
            Dictionary of metrics
        """
        returns = backtest_df['strategy_returns_net'].dropna()
        
        # Total return
        total_return = backtest_df['cumulative_strategy_returns'].iloc[-1]
        
        # Annualized return (assuming hourly data)
        n_periods = len(returns)
        hours_per_year = 24 * 365
        annualized_return = (1 + total_return) ** (hours_per_year / n_periods) - 1
        
        # Volatility
        volatility = returns.std() * np.sqrt(hours_per_year)
        
        # Sharpe ratio
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        
        # Maximum drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Win rate
        winning_trades = (returns > 0).sum()
        total_trades = (returns != 0).sum()
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Profit factor
        gains = returns[returns > 0].sum()
        losses = abs(returns[returns < 0].sum())
        profit_factor = gains / losses if losses > 0 else np.inf
        
        # Average win/loss
        avg_win = returns[returns > 0].mean() if winning_trades > 0 else 0
        avg_loss = returns[returns < 0].mean() if (total_trades - winning_trades) > 0 else 0
        
        # Final equity
        final_equity = backtest_df['equity'].iloc[-1]
        
        metrics = {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'total_trades': int(total_trades),
            'winning_trades': int(winning_trades),
            'losing_trades': int(total_trades - winning_trades),
            'final_equity': final_equity,
            'initial_capital': self.initial_capital
        }
        
        return metrics
        
    def print_summary(self, metrics: Dict[str, Any]):
        """
        Print backtest summary.
        
        Args:
            metrics: Metrics dictionary
        """
        logger.info("\n" + "="*60)
        logger.info("BACKTEST RESULTS")
        logger.info("="*60)
        logger.info(f"Initial Capital: ${metrics['initial_capital']:,.2f}")
        logger.info(f"Final Equity: ${metrics['final_equity']:,.2f}")
        logger.info(f"Total Return: {metrics['total_return']*100:.2f}%")
        logger.info(f"Annualized Return: {metrics['annualized_return']*100:.2f}%")
        logger.info(f"Volatility: {metrics['volatility']*100:.2f}%")
        logger.info(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        logger.info(f"Max Drawdown: {metrics['max_drawdown']*100:.2f}%")
        logger.info("-"*60)
        logger.info(f"Total Trades: {metrics['total_trades']}")
        logger.info(f"Win Rate: {metrics['win_rate']*100:.2f}%")
        logger.info(f"Profit Factor: {metrics['profit_factor']:.2f}")
        logger.info(f"Avg Win: {metrics['avg_win']*100:.2f}%")
        logger.info(f"Avg Loss: {metrics['avg_loss']*100:.2f}%")
        logger.info("="*60 + "\n")
