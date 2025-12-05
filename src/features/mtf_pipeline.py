"""
Pipeline to add multi-timeframe features to existing processed data.
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.features.multi_timeframe import MultiTimeframeFetcher, calculate_htf_features
from src.ingestion.storage import DataStorage
from src.utils.config import load_config
import pandas as pd
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def add_mtf_features_to_coin(coin_id: str, storage: DataStorage, fetcher: MultiTimeframeFetcher) -> pd.DataFrame:
    """
    Add multi-timeframe features to a coin's processed data.
    
    1. Load existing 1H processed data
    2. Fetch 4H and Daily data
    3. Calculate HTF features
    4. Merge and save
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Adding MTF features to {coin_id}")
    logger.info(f"{'='*60}\n")
    
    # Load existing processed data
    df_1h = storage.load_processed(coin_id)
    if df_1h is None:
        raise ValueError(f"No processed data for {coin_id}")
        
    original_cols = len(df_1h.columns)
    logger.info(f"Loaded 1H data: {len(df_1h)} rows, {original_cols} columns")
    
    # Fetch higher timeframe data
    try:
        htf_data = fetcher.fetch_all_timeframes(coin_id, days=730)
        df_4h = htf_data['4h']
        df_1d = htf_data['1d']
        logger.info(f"Fetched 4H: {len(df_4h)} candles, Daily: {len(df_1d)} candles")
    except Exception as e:
        logger.error(f"Failed to fetch HTF data: {e}")
        raise
        
    # Calculate HTF features
    features_4h = calculate_htf_features(df_4h, '_4h')
    features_1d = calculate_htf_features(df_1d, '_1d')
    
    logger.info(f"4H features: {list(features_4h.columns)}")
    logger.info(f"Daily features: {list(features_1d.columns)}")
    
    # Ensure timestamp column exists and is datetime
    if 'timestamp' not in df_1h.columns:
        raise ValueError("1H data missing timestamp column")
        
    df_1h['timestamp'] = pd.to_datetime(df_1h['timestamp'])
    features_4h['timestamp'] = pd.to_datetime(features_4h['timestamp'])
    features_1d['timestamp'] = pd.to_datetime(features_1d['timestamp'])
    
    # Sort by timestamp for merge_asof
    df_1h = df_1h.sort_values('timestamp')
    features_4h = features_4h.sort_values('timestamp')
    features_1d = features_1d.sort_values('timestamp')
    
    # Merge 4H features (backward fill - use most recent 4H data)
    df_merged = pd.merge_asof(
        df_1h,
        features_4h,
        on='timestamp',
        direction='backward'
    )
    
    # Merge Daily features
    df_merged = pd.merge_asof(
        df_merged,
        features_1d,
        on='timestamp',
        direction='backward'
    )
    
    # Load feature selection to remove useless features
    try:
        with open('configs/feature_selection.json', 'r') as f:
            feature_selection = json.load(f)
        features_to_remove = feature_selection.get('remove', [])
        
        # Remove useless features (but keep if they're new MTF features)
        cols_to_drop = [col for col in features_to_remove if col in df_merged.columns 
                       and not col.endswith('_4h') and not col.endswith('_1d')]
        
        if cols_to_drop:
            df_merged = df_merged.drop(columns=cols_to_drop, errors='ignore')
            logger.info(f"Removed {len(cols_to_drop)} useless features")
    except FileNotFoundError:
        logger.warning("Feature selection file not found, keeping all features")
        
    new_cols = len(df_merged.columns)
    logger.info(f"Final data: {len(df_merged)} rows, {new_cols} columns")
    logger.info(f"Added {new_cols - original_cols + len(cols_to_drop) if 'cols_to_drop' in dir() else new_cols - original_cols} new features")
    
    # Save with MTF suffix
    output_path = Path(f"data/processed/{coin_id}_processed_mtf.parquet")
    df_merged.to_parquet(output_path, index=False)
    logger.info(f"Saved to {output_path}")
    
    return df_merged


def main():
    """Add MTF features to all coins."""
    config = load_config()
    storage = DataStorage()
    fetcher = MultiTimeframeFetcher()
    
    coin_ids = config['data']['symbols']
    
    results = {}
    
    for coin_id in coin_ids:
        try:
            df = add_mtf_features_to_coin(coin_id, storage, fetcher)
            results[coin_id] = {
                'rows': len(df),
                'columns': len(df.columns),
                'mtf_features': [col for col in df.columns if col.endswith('_4h') or col.endswith('_1d')]
            }
        except Exception as e:
            logger.error(f"Failed for {coin_id}: {e}")
            import traceback
            traceback.print_exc()
            
    # Save results summary
    with open('configs/mtf_features.json', 'w') as f:
        json.dump(results, f, indent=2)
        
    logger.info("\n" + "="*60)
    logger.info("MTF FEATURE PIPELINE COMPLETE")
    logger.info("="*60)
    for coin_id, info in results.items():
        logger.info(f"{coin_id}: {info['columns']} features, {len(info['mtf_features'])} MTF features")


if __name__ == "__main__":
    main()
