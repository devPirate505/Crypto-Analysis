# Quick Start Guide

## Prerequisites
- Python 3.10+
- pip

## Setup Steps

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Fetch data** (takes ~5-10 minutes)
   ```bash
   python src/ingestion/backfill.py
   ```

3. **Generate features** (takes ~2-3 minutes)
   ```bash
   python src/features/pipeline.py
   ```

4. **Train models** (takes ~5-10 minutes)
   ```bash
   python src/training/train.py
   ```

5. **Run backtest** (takes ~2-3 minutes)
   ```bash
   python src/backtest/run_backtest.py
   ```

6. **Launch dashboard**
   ```bash
   streamlit run app.py
   ```

## Expected Output

After step 2, you'll have:
- `data/raw/{coin}_*.parquet` files

After step 3:
- `data/processed/{coin}_processed.parquet` files
- `data/processed/{coin}_feature_manifest.json` files

After step 4:
- `models/{coin}_lightgbm.joblib` files
- `models/{coin}_random_forest.joblib` files
- `models/baseline_results.json`

After step 5:
- `data/processed/{coin}_backtest.parquet` files
- `models/backtest_results.json`

## Troubleshooting

### CoinGecko API Rate Limit
If you hit rate limits, increase `rate_limit_delay` in `configs/default.yaml`

### Out of Memory
Reduce `history_days` in `configs/default.yaml`

### Missing Dependencies
```bash
pip install --upgrade -r requirements.txt
```

## Next Steps

- Customize model parameters in `configs/default.yaml`
- Add more cryptocurrencies to track
- Experiment with different strategies in `src/backtest/strategy.py`
