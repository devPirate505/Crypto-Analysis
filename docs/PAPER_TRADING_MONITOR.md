# 🤖 Paper Trading Monitor - Background Service

## Overview
The Paper Trading Monitor is a standalone Python script that runs in the background to automatically monitor your paper trading positions and close them when Stop Loss or Take Profit levels are hit.

## Features
- ✅ **Real-time monitoring** - Checks prices every 10 seconds
- ✅ **Auto SL/TP** - Closes positions automatically when triggered  
- ✅ **Persistent storage** - Saves all data to `models/paper_trading_state.json`
- ✅ **Independent** - Runs separately from Streamlit dashboard
- ✅ **Trade logging** - Records all auto-closed trades with reason

## How It Works

1. **Monitors Position File**: Continuously reads `models/paper_trading_state.json`
2. **Fetches Live Prices**: Gets current price from Binance every 10 seconds
3. **Checks SL/TP**: Compares price against Stop Loss and Take Profit levels
4. **Auto-Closes**: When triggered, closes position and updates the file
5. **Records Trade**: Saves trade history with `close_reason` and `auto_closed` flag

## Usage

### Step 1: Ensure Persistent Storage is Enabled
The dashboard must use the persistent storage system. When you enter a trade from **Model Predictions** page (which sets SL/TP automatically), it will be saved to `models/paper_trading_state.json`.

### Step 2: Start the Monitor
Open a **second terminal** and run:

```bash
python -m src.serving.paper_monitor
```

You'll see output like:
```
🚀 Paper Trading Monitor started
   Checking every 10 seconds
   Press Ctrl+C to stop

📊 Monitoring BITCOIN LONG @ $89,073
   Current: $89,129.60 | P&L: +0.06%
   🛑 SL: $88,200.00
   🎯 TP: $90,500.00
```

### Step 3: Trade Normally
- Open trades from the dashboard (Model Predictions page)
- The monitor will track them automatically
- When SL/TP is hit, you'll see:

```
🎯 Take Profit Hit! Closed LONG BITCOIN at $90,512.30
   P&L: +$1,215.45 (+1.36%)
💾 State saved. Position closed.
```

### Step 4: View Results
- Refresh your dashboard (F5)  
- Go to **Paper Trading** page
- You'll see the auto-closed trade in your history with the closure reason

## File Structure

```
models/
└── paper_trading_state.json    # Persistent state file
    ├── trades: []               # Trade history
    ├── balance: 10000.0         # Current balance
    ├── position: {...}          # Active position (or null)
    └── last_updated: "..."      # Timestamp
```

## Position Format

When a trade is active, the position looks like:
```json
{
  "coin": "bitcoin",
  "side": "long",
  "entry_price": 89073.0,
  "size": 0.0112,
  "entry_time": "2025-12-06 00:10:00",
  "predicted_return": 0.002,
  "stop_loss": 88200.0,
  "take_profit": 90500.0
}
```

## Trade Record Format

Auto-closed trades include:
```json
{
  "coin": "bitcoin",
  "side": "long",
  "entry_price": 89073.0,
  "exit_price": 90512.3,
  "pnl": 1215.45,
  "pnl_pct": 1.36,
  "close_reason": "Take Profit Hit",
  "auto_closed": true,
  "entry_time": "2025-12-06 00:10:00",
  "exit_time": "2025-12-06 00:15:30"
}
```

##Limitations

- **Single Position**: Only one position at a time
- **Requires SL/TP**: Position must have `stop_loss` or `take_profit` set
- **Live Data**: Requires Binance API access
- **Manual Closesstill work**: You can still close from dashboard

## Troubleshooting

### Monitor Not Finding Position
- Make sure you entered the trade from **Model Predictions** page (not manual entry)
- Check if `models/paper_trading_state.json` exists and has a position

### API Errors
- Check your `.env` file has valid Bin ance API keys
- Ensure you have internet connection

### Position Not Closing
- Verify SL/TP levels are actually being hit
- Check monitor terminal for error messages
- Make sure the monitor is running

## Running Both Together

**Terminal 1** (Dashboard):
```bash
streamlit run app.py
```

**Terminal 2** (Monitor):
```bash
python -m src.serving.paper_monitor
```

Keep both running for fully automated paper trading! 🚀
