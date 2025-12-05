"""
Analyze feature importance and identify features to keep/remove.
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.lightgbm_model import LightGBMModel
from src.ingestion.storage import DataStorage
from src.utils.config import load_config
import pandas as pd
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def analyze_feature_importance(coin_id: str):
    """Analyze and report feature importance for a coin."""
    
    # Load optimized model
    model_path = f"models/{coin_id}_lightgbm_optimized.joblib"
    if not Path(model_path).exists():
        model_path = f"models/{coin_id}_lightgbm.joblib"
        
    model = LightGBMModel(task_type='regression')
    model.load(model_path)
    
    # Load data to get feature names
    storage = DataStorage()
    df = storage.load_processed(coin_id)
    
    exclude_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 
                    'market_cap', 'coin_id', 'target', 'future_return']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Get feature importance
    importance_df = model.get_feature_importance(feature_cols)
    
    # Analyze
    total_features = len(importance_df)
    zero_importance = len(importance_df[importance_df['importance'] == 0])
    non_zero = importance_df[importance_df['importance'] > 0]
    
    logger.info(f"\n{'='*60}")
    logger.info(f"FEATURE IMPORTANCE ANALYSIS: {coin_id.upper()}")
    logger.info(f"{'='*60}")
    logger.info(f"Total Features: {total_features}")
    logger.info(f"Features with 0 importance: {zero_importance} ({zero_importance/total_features*100:.1f}%)")
    logger.info(f"Features actually used: {len(non_zero)} ({len(non_zero)/total_features*100:.1f}%)")
    
    logger.info(f"\nTop 20 Most Important Features:")
    for i, row in importance_df.head(20).iterrows():
        logger.info(f"  {row['feature']}: {row['importance']}")
        
    logger.info(f"\nFeatures with 0 importance (can be removed):")
    zero_features = importance_df[importance_df['importance'] == 0]['feature'].tolist()
    for feat in zero_features[:10]:  # Show first 10
        logger.info(f"  - {feat}")
    if len(zero_features) > 10:
        logger.info(f"  ... and {len(zero_features) - 10} more")
        
    return importance_df, zero_features


def get_important_features(min_coins_required: int = 2):
    """Get features that are important across multiple coins."""
    
    config = load_config()
    coin_ids = config['data']['symbols']
    
    all_importance = {}
    all_zero_features = {}
    
    for coin_id in coin_ids:
        try:
            importance_df, zero_features = analyze_feature_importance(coin_id)
            all_importance[coin_id] = importance_df
            all_zero_features[coin_id] = set(zero_features)
        except Exception as e:
            logger.error(f"Failed for {coin_id}: {e}")
            
    # Find features with 0 importance in ALL coins (safe to remove)
    common_zero = set.intersection(*all_zero_features.values()) if all_zero_features else set()
    
    # Find features important in at least min_coins_required coins
    important_across_coins = set()
    for coin_id, importance_df in all_importance.items():
        non_zero = importance_df[importance_df['importance'] > 0]['feature'].tolist()
        important_across_coins.update(non_zero)
        
    logger.info(f"\n{'='*60}")
    logger.info("SUMMARY ACROSS ALL COINS")
    logger.info(f"{'='*60}")
    logger.info(f"Features with 0 importance in ALL coins: {len(common_zero)}")
    logger.info(f"Features important in at least one coin: {len(important_across_coins)}")
    
    # Save recommended features
    recommended = {
        'keep': list(important_across_coins),
        'remove': list(common_zero),
        'total_kept': len(important_across_coins),
        'total_removed': len(common_zero)
    }
    
    with open('configs/feature_selection.json', 'w') as f:
        json.dump(recommended, f, indent=2)
        
    logger.info(f"\nSaved feature selection to configs/feature_selection.json")
    logger.info(f"Recommended: Keep {len(important_across_coins)} features, remove {len(common_zero)}")
    
    return recommended


if __name__ == "__main__":
    get_important_features()
