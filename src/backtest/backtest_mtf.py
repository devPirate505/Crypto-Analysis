"""
Quick backtest for MTF models.
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.lightgbm_model import LightGBMModel
from src.training.walk_forward import WalkForwardCV
from src.backtest.engine import Backtester
from src.utils.config import load_config
import pandas as pd
import numpy as np
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def backtest_mtf_model(coin_id: str, config: dict) -> dict:
    """Backtest MTF model."""
    
    logger.info(f"\n{'='*60}")
    logger.info(f"BACKTESTING MTF MODEL: {coin_id.upper()}")
    logger.info(f"{'='*60}\n")
    
    # Load MTF data
    df = pd.read_parquet(f"data/processed/{coin_id}_processed_mtf.parquet")
    
    # Create target
    df['future_return'] = (df['close'].shift(-1) / df['close']) - 1
    df = df.dropna(subset=['future_return'])
    
    exclude_cols = [
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'market_cap', 'coin_id', 'target', 'future_return'
    ]
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Split
    data = df.copy()
    wf_cv = WalkForwardCV()
    train_df, val_df, test_df = wf_cv.simple_split(data, train_ratio=0.7, val_ratio=0.15)
    
    # Load model
    model = LightGBMModel(task_type='regression')
    model.load(f"models/{coin_id}_lightgbm_mtf.joblib")
    
    # Get predictions
    X_test = test_df[feature_cols]
    predictions = model.predict(X_test)
    
    # Strategy: buy when predicted return > 0.03%
    threshold = 0.0003
    positions = np.where(predictions > threshold, 1, 0)
    
    logger.info(f"Positions: {np.sum(positions==1)} long, {np.sum(positions==0)} flat")
    
    # Calculate returns
    prices = test_df['close'].values
    actual_returns = test_df['future_return'].values
    
    # Strategy returns
    strategy_returns = positions * actual_returns
    
    # Metrics
    total_trades = int(np.sum(np.diff(positions) != 0))
    
    # Win rate
    trade_returns = strategy_returns[positions == 1]
    if len(trade_returns) > 0:
        wins = np.sum(trade_returns > 0)
        losses = np.sum(trade_returns <= 0)
        win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0
        avg_win = np.mean(trade_returns[trade_returns > 0]) if np.any(trade_returns > 0) else 0
        avg_loss = np.mean(trade_returns[trade_returns <= 0]) if np.any(trade_returns <= 0) else 0
    else:
        win_rate = 0
        avg_win = 0
        avg_loss = 0
        
    # Cumulative return
    cumulative_return = np.prod(1 + strategy_returns) - 1
    
    # Calculate equity curve
    test_df['strategy_return'] = strategy_returns
    test_df['equity_curve'] = (1 + test_df['strategy_return']).cumprod()
    test_df['signal'] = positions
    test_df['predicted_return'] = predictions
    
    # Save detailed backtest results for dashboard
    backtest_df = test_df[['timestamp', 'close', 'signal', 'predicted_return', 'strategy_return', 'equity_curve', 'future_return']]
    backtest_df.to_parquet(f"data/processed/{coin_id}_backtest_mtf.parquet")
    
    # Compare to previous results
    logger.info(f"\n{'='*40}")
    logger.info(f"MTF BACKTEST RESULTS: {coin_id.upper()}")
    logger.info(f"{'='*40}")
    logger.info(f"Total Return: {cumulative_return*100:.2f}%")
    logger.info(f"Win Rate: {win_rate*100:.1f}%")
    logger.info(f"Avg Win: {avg_win*100:.3f}%")
    logger.info(f"Avg Loss: {avg_loss*100:.3f}%")
    logger.info(f"Total Trades: {total_trades}")
    logger.info(f"Features Used: {len(feature_cols)}")
    logger.info(f"MTF Features: {len([c for c in feature_cols if '_4h' in c or '_1d' in c])}")
    
    # Calculate additional metrics
    volatility = np.std(strategy_returns) * np.sqrt(24*365) if np.std(strategy_returns) > 0 else 0
    
    gross_profit = np.sum(strategy_returns[strategy_returns > 0])
    gross_loss = np.abs(np.sum(strategy_returns[strategy_returns < 0]))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
    
    return {
        'total_return': cumulative_return,
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'total_trades': total_trades,
        'features': len(feature_cols),
        'sharpe_ratio': np.mean(strategy_returns) / np.std(strategy_returns) * np.sqrt(24*365) if np.std(strategy_returns) > 0 else 0,
        'max_drawdown': (test_df['equity_curve'] / test_df['equity_curve'].cummax() - 1).min(),
        'volatility': volatility,
        'profit_factor': profit_factor
    }


def main():
    """Backtest all MTF models."""
    config = load_config()
    coin_ids = config['data']['symbols']
    
    all_results = {}
    
    for coin_id in coin_ids:
        try:
            results = backtest_mtf_model(coin_id, config)
            all_results[coin_id] = results
        except Exception as e:
            logger.error(f"Failed for {coin_id}: {e}")
            import traceback
            traceback.print_exc()
            
    # Save results
    with open('models/backtest_results_mtf.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
        
    # Summary comparison
    logger.info("\n" + "="*60)
    logger.info("COMPARISON: BEFORE vs AFTER MTF")
    logger.info("="*60)
    
    # Load previous results for comparison
    try:
        with open('models/backtest_results_advanced.json', 'r') as f:
            old_results = json.load(f)
            
        for coin_id in coin_ids:
            if coin_id in all_results and coin_id in old_results:
                old = old_results[coin_id]
                new = all_results[coin_id]
                
                logger.info(f"\n{coin_id.upper()}:")
                logger.info(f"  Return: {old.get('total_return', 0)*100:.2f}% → {new['total_return']*100:.2f}%")
                logger.info(f"  Win Rate: {old.get('win_rate', 0)*100:.1f}% → {new['win_rate']*100:.1f}%")
                logger.info(f"  Features: 115 → {new['features']}")
    except FileNotFoundError:
        pass


if __name__ == "__main__":
    main()
