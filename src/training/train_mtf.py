"""
Train and backtest models using multi-timeframe features.
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.lightgbm_model import LightGBMModel
from src.models.rf_model import RandomForestModel
from src.training.walk_forward import WalkForwardCV
from src.backtest.engine import Backtester
from src.utils.config import load_config
import pandas as pd
import numpy as np
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def prepare_mtf_data(df: pd.DataFrame, prediction_horizon: int = 1):
    """Prepare MTF data for training."""
    # Create target
    df['future_return'] = (df['close'].shift(-prediction_horizon) / df['close']) - 1
    df = df.dropna(subset=['future_return'])
    
    exclude_cols = [
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'market_cap', 'coin_id', 'target', 'future_return'
    ]
    
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    X = df[feature_cols]
    y = df['future_return']
    timestamps = df['timestamp']
    
    logger.info(f"Prepared {len(feature_cols)} features, {len(X)} samples")
    
    return X, y, feature_cols, timestamps


def train_and_backtest_mtf(coin_id: str, config: dict) -> dict:
    """Train and backtest a coin with MTF features."""
    
    logger.info(f"\n{'='*60}")
    logger.info(f"TRAINING MTF MODEL: {coin_id.upper()}")
    logger.info(f"{'='*60}\n")
    
    # Load MTF data
    mtf_path = Path(f"data/processed/{coin_id}_processed_mtf.parquet")
    if not mtf_path.exists():
        raise FileNotFoundError(f"MTF data not found: {mtf_path}")
        
    df = pd.read_parquet(mtf_path)
    
    # Prepare data
    X, y, feature_cols, timestamps = prepare_mtf_data(df, config['model']['prediction_horizon'])
    
    # Create dataframe for splitting
    data = pd.DataFrame(X, columns=feature_cols)
    data['timestamp'] = timestamps.values
    data['target'] = y.values
    data['close'] = df['close'].iloc[:len(data)].values
    
    # Split
    wf_cv = WalkForwardCV()
    train_df, val_df, test_df = wf_cv.simple_split(data, train_ratio=0.7, val_ratio=0.15)
    
    X_train, y_train = train_df[feature_cols], train_df['target']
    X_val, y_val = val_df[feature_cols], val_df['target']
    X_test, y_test = test_df[feature_cols], test_df['target']
    
    # Load best params if available
    params = {
        'objective': 'regression',
        'metric': 'mae',
        'verbosity': -1,
        'n_jobs': -1,
    }
    
    try:
        with open('configs/best_params.json', 'r') as f:
            best_params = json.load(f)
        if coin_id in best_params:
            params.update(best_params[coin_id])
    except FileNotFoundError:
        pass
        
    # Train LightGBM
    lgb_model = LightGBMModel(task_type='regression', params=params)
    lgb_model.train(X_train, y_train, X_val, y_val)
    
    # Evaluate
    train_metrics = lgb_model.evaluate(X_train, y_train)
    val_metrics = lgb_model.evaluate(X_val, y_val)
    test_metrics = lgb_model.evaluate(X_test, y_test)
    
    logger.info(f"Train MAE: {train_metrics['mae']:.6f}")
    logger.info(f"Val MAE: {val_metrics['mae']:.6f}")
    logger.info(f"Test MAE: {test_metrics['mae']:.6f}")
    
    # Save model
    lgb_model.save(f"models/{coin_id}_lightgbm_mtf.joblib")
    
    # Train RandomForest
    rf_model = RandomForestModel(task_type='regression')
    rf_model.train(X_train, y_train, X_val, y_val)
    rf_model.save(f"models/{coin_id}_random_forest_mtf.joblib")
    
    # Simple backtest
    predictions = lgb_model.predict(X_test)
    
    # Trading strategy: buy if predicted return > 0.03%
    positions = np.where(predictions > 0.0003, 1, 0)
    
    # Calculate returns
    prices = test_df['close'].values
    returns = np.diff(prices) / prices[:-1]
    
    # Strategy returns (with 1-period lag for execution)
    strategy_returns = positions[:-1] * returns
    
    # Metrics
    total_trades = np.sum(np.diff(positions) != 0) // 2 + np.sum(positions == 1)
    
    if np.sum(positions[:-1] == 1) > 0:
        wins = np.sum((strategy_returns > 0) & (positions[:-1] == 1))
        losses = np.sum((strategy_returns <= 0) & (positions[:-1] == 1))
        win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0
    else:
        win_rate = 0
        
    cumulative_return = np.prod(1 + strategy_returns) - 1
    
    logger.info(f"\n--- BACKTEST RESULTS ---")
    logger.info(f"Total Return: {cumulative_return*100:.2f}%")
    logger.info(f"Win Rate: {win_rate*100:.1f}%")
    logger.info(f"Trades: {total_trades}")
    
    return {
        'model_metrics': {
            'train_mae': train_metrics['mae'],
            'val_mae': val_metrics['mae'],
            'test_mae': test_metrics['mae']
        },
        'backtest': {
            'total_return': cumulative_return,
            'win_rate': win_rate,
            'total_trades': total_trades
        },
        'features': len(feature_cols),
        'mtf_features': [col for col in feature_cols if col.endswith('_4h') or col.endswith('_1d')]
    }


def main():
    """Train MTF models for all coins."""
    config = load_config()
    coin_ids = config['data']['symbols']
    
    all_results = {}
    
    for coin_id in coin_ids:
        try:
            results = train_and_backtest_mtf(coin_id, config)
            all_results[coin_id] = results
        except Exception as e:
            logger.error(f"Failed for {coin_id}: {e}")
            import traceback
            traceback.print_exc()
            
    # Save results
    with open('models/mtf_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
        
    # Summary
    logger.info("\n" + "="*60)
    logger.info("MTF TRAINING COMPLETE - SUMMARY")
    logger.info("="*60)
    
    for coin_id, res in all_results.items():
        bt = res.get('backtest', {})
        logger.info(f"\n{coin_id.upper()}:")
        logger.info(f"  Features: {res.get('features', 0)} (incl. {len(res.get('mtf_features', []))} MTF)")
        logger.info(f"  Return: {bt.get('total_return', 0)*100:.2f}%")
        logger.info(f"  Win Rate: {bt.get('win_rate', 0)*100:.1f}%")


if __name__ == "__main__":
    main()
