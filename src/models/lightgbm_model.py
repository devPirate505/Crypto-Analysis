"""
LightGBM model wrapper for cryptocurrency prediction.
"""
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error
import joblib
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LightGBMModel:
    """
    LightGBM model for cryptocurrency prediction.
    Supports both classification (price direction) and regression (price change).
    """
    
    def __init__(self, task_type: str = "classification", params: dict = None):
        """
        Initialize LightGBM model.
        
        Args:
            task_type: 'classification' or 'regression'
            params: Model hyperparameters
        """
        self.task_type = task_type
        self.model = None
        
        # Default parameters
        default_params = {
            'objective': 'binary' if task_type == 'classification' else 'regression',
            'metric': 'binary_logloss' if task_type == 'classification' else 'mae',
            'n_estimators': 500,
            'learning_rate': 0.05,
            'max_depth': 7,
            'num_leaves': 31,
            'min_child_samples': 20,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1
        }
        
        self.params = params if params else default_params
        
    def prepare_target(self, df: pd.DataFrame, horizon: int = 1, threshold: float = 0.0):
        """
        Prepare target variable.
        
        Args:
            df: DataFrame with price data
            horizon: Prediction horizon (hours)
            threshold: Classification threshold for returns
            
        Returns:
            Target series
        """
        # Calculate future return
        future_return = (df['close'].shift(-horizon) / df['close']) - 1
        
        if self.task_type == 'classification':
            # Binary: 1 if return > threshold, 0 otherwise
            target = (future_return > threshold).astype(int)
        else:
            # Regression: actual return
            target = future_return
            
        return target
        
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame = None,
        y_val: pd.Series = None,
        categorical_features: list = None
    ):
        """
        Train the model.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            categorical_features: List of categorical feature names
        """
        logger.info(f"Training LightGBM model ({self.task_type})")
        logger.info(f"Training samples: {len(X_train)}, Features: {X_train.shape[1]}")
        
        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]
            logger.info(f"Validation samples: {len(X_val)}")
        else:
            eval_set = None
            
        # Initialize model
        if self.task_type == 'classification':
            self.model = lgb.LGBMClassifier(**self.params)
        else:
            self.model = lgb.LGBMRegressor(**self.params)
            
        # Train
        self.model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            callbacks=[lgb.log_evaluation(period=100), lgb.early_stopping(stopping_rounds=50)]
            if eval_set else [lgb.log_evaluation(period=100)]
        )
        
        logger.info("Training complete")
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Features
            
        Returns:
            Predictions
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
            
        return self.model.predict(X)
        
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict probabilities (classification only).
        
        Args:
            X: Features
            
        Returns:
            Prediction probabilities
        """
        if self.task_type != 'classification':
            raise ValueError("predict_proba only available for classification")
            
        if self.model is None:
            raise ValueError("Model not trained yet")
            
        return self.model.predict_proba(X)
        
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> dict:
        """
        Evaluate model performance.
        
        Args:
            X: Features
            y: True labels/values
            
        Returns:
            Dictionary of metrics
        """
        predictions = self.predict(X)
        
        if self.task_type == 'classification':
            metrics = {
                'accuracy': accuracy_score(y, predictions),
                'precision': precision_score(y, predictions, zero_division=0),
                'recall': recall_score(y, predictions, zero_division=0),
                'f1': f1_score(y, predictions, zero_division=0)
            }
            
            # Add probability-based metrics if available
            if hasattr(self.model, 'predict_proba'):
                proba = self.predict_proba(X)[:, 1]
                from sklearn.metrics import roc_auc_score
                try:
                    metrics['roc_auc'] = roc_auc_score(y, proba)
                except:
                    pass
                    
        else:
            metrics = {
                'mae': mean_absolute_error(y, predictions),
                'rmse': np.sqrt(((predictions - y) ** 2).mean()),
                'mape': np.mean(np.abs((y - predictions) / (y + 1e-8))) * 100
            }
            
        return metrics
        
    def get_feature_importance(self, feature_names: list = None) -> pd.DataFrame:
        """
        Get feature importance.
        
        Args:
            feature_names: List of feature names
            
        Returns:
            DataFrame with feature importance
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
            
        importance = self.model.feature_importances_
        
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(len(importance))]
            
        df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return df
        
    def save(self, filepath: str):
        """
        Save model to file.
        
        Args:
            filepath: Path to save model
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, filepath)
        logger.info(f"Model saved to {filepath}")
        
    def load(self, filepath: str):
        """
        Load model from file.
        
        Args:
            filepath: Path to model file
        """
        self.model = joblib.load(filepath)
        logger.info(f"Model loaded from {filepath}")
