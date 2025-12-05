"""
RandomForest model wrapper for cryptocurrency prediction.
"""
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error
import numpy as np
import pandas as pd
import joblib
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RandomForestModel:
    """
    RandomForest model for cryptocurrency prediction.
    CPU-optimized with parallelization.
    """
    
    def __init__(self, task_type: str = "classification", params: dict = None):
        """
        Initialize RandomForest model.
        
        Args:
            task_type: 'classification' or 'regression'
            params: Model hyperparameters
        """
        self.task_type = task_type
        self.model = None
        
        # Default parameters
        default_params = {
            'n_estimators': 200,
            'max_depth': 10,
            'min_samples_split': 10,
            'min_samples_leaf': 5,
            'random_state': 42,
            'n_jobs': -1
        }
        
        self.params = params if params else default_params
        
    def train(self, X_train: pd.DataFrame, y_train: pd.Series):
        """
        Train the model.
        
        Args:
            X_train: Training features
            y_train: Training target
        """
        logger.info(f"Training RandomForest model ({self.task_type})")
        logger.info(f"Training samples: {len(X_train)}, Features: {X_train.shape[1]}")
        
        # Initialize model
        if self.task_type == 'classification':
            self.model = RandomForestClassifier(**self.params)
        else:
            self.model = RandomForestRegressor(**self.params)
            
        # Train
        self.model.fit(X_train, y_train)
        
        logger.info("Training complete")
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if self.model is None:
            raise ValueError("Model not trained yet")
        return self.model.predict(X)
        
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities (classification only)."""
        if self.task_type != 'classification':
            raise ValueError("predict_proba only available for classification")
        if self.model is None:
            raise ValueError("Model not trained yet")
        return self.model.predict_proba(X)
        
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> dict:
        """Evaluate model performance."""
        predictions = self.predict(X)
        
        if self.task_type == 'classification':
            metrics = {
                'accuracy': accuracy_score(y, predictions),
                'precision': precision_score(y, predictions, zero_division=0),
                'recall': recall_score(y, predictions, zero_division=0),
                'f1': f1_score(y, predictions, zero_division=0)
            }
        else:
            metrics = {
                'mae': mean_absolute_error(y, predictions),
                'rmse': np.sqrt(((predictions - y) ** 2).mean()),
                'mape': np.mean(np.abs((y - predictions) / (y + 1e-8))) * 100
            }
            
        return metrics
        
    def get_feature_importance(self, feature_names: list = None) -> pd.DataFrame:
        """Get feature importance."""
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
        """Save model to file."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, filepath)
        logger.info(f"Model saved to {filepath}")
        
    def load(self, filepath: str):
        """Load model from file."""
        self.model = joblib.load(filepath)
        logger.info(f"Model loaded from {filepath}")
