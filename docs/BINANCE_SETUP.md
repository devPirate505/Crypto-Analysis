# Binance API Quick Start

## Installation

```bash
pip install python-binance
```

## Fetch Historical Data

**Get 2 years of hourly data (recommended):**

```bash
python src/ingestion/backfill_binance.py
```

This will:
- Fetch **730 days** (2 years) of hourly OHLCV data
- Download for Bitcoin, Ethereum, and Binance Coin
- Save to `data/raw/` in Parquet format
- No API key required!

## Configuration

Edit `configs/default.yaml` to customize:

```yaml
data:
  symbols:
    - bitcoin
    - ethereum
    - binancecoin
    - cardano      # Add more coins!
    - solana
  
  history_days: 730  # 2 years (can go up to 3+ years)
```

## Available Cryptocurrencies

- `bitcoin` → BTCUSDT
- `ethereum` → ETHUSDT
- `binancecoin` → BNBUSDT
- `cardano` → ADAUSDT
- `solana` → SOLUSDT
- `ripple` → XRPUSDT
- `polkadot` → DOTUSDT
- `dogecoin` → DOGEUSDT

## Timeframes Supported

- `1m` - 1 minute
- `5m` - 5 minutes
- `15m` - 15 minutes
- `1h` - 1 hour (default)
- `4h` - 4 hours
- `1d` - 1 day

To change interval, edit `backfill_binance.py`:

```python
interval = '1h'  # Change to '4h', '1d', etc.
```

## Benefits Over CoinGecko

✅ **More Data**: 3+ years vs 90 days  
✅ **No Rate Limits**: 1200 req/min vs 50/min  
✅ **Better Quality**: Direct exchange data  
✅ **Free**: No API key required  
✅ **Faster**: Parallel downloads supported  

## Next Steps

After fetching data:

1. **Generate Features**
   ```bash
   python src/features/pipeline.py
   ```

2. **Train Models**
   ```bash
   python src/training/train.py
   ```

3. **Run Backtest**
   ```bash
   python src/backtest/run_backtest.py
   ```

## Troubleshooting

**Error: "Unknown coin"**
- Check available coins with `BinanceFetcher.get_available_symbols()`
- Make sure coin name is lowercase

**Error: "Connection timeout"**
- Check internet connection
- Binance might be blocked in your country (use VPN)

**Too much data**
- Reduce `history_days` in config
- Binance can handle it, but processing takes time

## Advanced Usage

**Fetch specific date range:**

```python
from src.ingestion.binance_fetcher import BinanceFetcher

fetcher = BinanceFetcher()
df = fetcher.fetch_ohlcv(
    'bitcoin',
    interval='1h',
    days=365  # 1 year
)
```

**Multiple timeframes:**

```python
# Get both hourly and daily data
hourly = fetcher.fetch_ohlcv('bitcoin', interval='1h', days=730)
daily = fetcher.fetch_ohlcv('bitcoin', interval='1d', days=730)
```
