"""
Hyperparameter tuning using Optuna.
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

import optuna
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error
import logging
import json
import yaml

from src.ingestion.storage import DataStorage
from src.utils.config import load_config
from src.training.train import prepare_data
from src.training.walk_forward import WalkForwardCV

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def objective(trial, X, y, task_type):
    """
    Optuna objective function.
    """
    # Define hyperparameters to tune
    params = {
        'objective': 'binary' if task_type == 'classification' else 'regression',
        'metric': 'binary_logloss' if task_type == 'classification' else 'mae',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'n_jobs': -1
    }
    
    # Simple train/val split for tuning (last 20% as validation)
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # Train model
    if task_type == 'classification':
        model = lgb.LGBMClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(stopping_rounds=20)])
        preds = model.predict(X_val)
        from sklearn.metrics import f1_score
        score = f1_score(y_val, preds, zero_division=0) # Maximize F1
        return score
    else:
        model = lgb.LGBMRegressor(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(stopping_rounds=20)])
        preds = model.predict(X_val)
        score = mean_absolute_error(y_val, preds) # Minimize MAE
        return score


def tune_coin(coin_id, n_trials=50):
    """
    Tune hyperparameters for a specific coin.
    """
    config = load_config()
    storage = DataStorage()
    
    # Load data
    df = storage.load_processed(coin_id)
    if df is None:
        logger.error(f"No data for {coin_id}")
        return None
        
    # Prepare data
    task_type = config['model']['target_type']
    prediction_horizon = config['model']['prediction_horizon']
    
    X, y, feature_cols, _ = prepare_data(df, prediction_horizon, task_type)
    
    # Create study
    direction = 'maximize' if task_type == 'classification' else 'minimize'
    study = optuna.create_study(direction=direction)
    
    logger.info(f"Starting optimization for {coin_id} ({task_type})...")
    study.optimize(lambda trial: objective(trial, X, y, task_type), n_trials=n_trials)
    
    logger.info(f"Best params for {coin_id}: {study.best_params}")
    logger.info(f"Best score: {study.best_value}")
    
    return study.best_params


def main():
    """
    Main tuning function.
    """
    config = load_config()
    coin_ids = config['data']['symbols']
    
    best_params_all = {}
    
    for coin_id in coin_ids:
        best_params = tune_coin(coin_id, n_trials=30) # 30 trials per coin
        if best_params:
            best_params_all[coin_id] = best_params
            
    # Save best params to file
    with open('configs/best_params.json', 'w') as f:
        json.dump(best_params_all, f, indent=2)
        
    logger.info("Tuning complete! Saved to configs/best_params.json")


if __name__ == "__main__":
    main()
