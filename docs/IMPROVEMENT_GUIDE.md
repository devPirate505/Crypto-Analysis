# 🎯 Improving Model Performance for Real Trading

## Current Issues

**Performance Summary:**
- **Bitcoin:** 0% return (no trades - too conservative)
- **Ethereum:** -3.54% return, 14.3% win rate ❌
- **Binance Coin:** -9.40% return, 34.9% win rate ❌

**Root Causes:**
1. Limited training data (90 days only)
2. Simple binary classification approach
3. Technical indicators alone insufficient
4. No proper hyperparameter tuning
5. Market noise overwhelming signal
6. Transaction costs eating profits

---

## 🚀 Improvement Strategy

### 1. **Data Quality & Quantity** (CRITICAL)

#### Get More Historical Data

**Problem:** Only 90 days of hourly data from CoinGecko free tier

**Solutions:**

**Option A: Use Alternative Data Sources**
```python
# Free alternatives with more history:
# 1. Binance API (3 years+ of data)
# 2. Yahoo Finance for crypto (via yfinance)
# 3. Kraken API
# 4. Historical CSV downloads from CryptoDataDownload
```

**Option B: Upgrade to CoinGecko Pro**
- Get 365+ days of data
- Higher rate limits
- More granular data

**Option C: Combine Multiple Timeframes**
- Daily data for long-term trends
- Hourly for entry/exit timing
- Multi-timeframe features

**Recommended:** At least **1-2 years** of data for crypto

#### Add Alternative Data

**Sentiment Data:**
- Twitter/Reddit sentiment (free: snscrape, PRAW)
- Fear & Greed Index
- News sentiment (NewsAPI)

**On-Chain Metrics:**
- Active addresses
- Transaction volume
- Exchange inflows/outflows
- Network hash rate (Bitcoin)
- Gas prices (Ethereum)

**Market Structure:**
- Order book depth
- Liquidity metrics
- Funding rates (futures)
- Open interest

---

### 2. **Feature Engineering Enhancements**

#### A. Add Volatility Regime Detection

```python
def add_regime_features(df):
    # Identify market regimes
    df['volatility_regime'] = pd.qcut(
        df['rolling_std_24h'], 
        q=3, 
        labels=['low', 'medium', 'high']
    )
    
    # Trend strength
    df['trend_strength'] = abs(df['ema_20'] - df['ema_50']) / df['close']
    
    # Choppy vs trending
    df['adx_regime'] = df['adx_14'].apply(
        lambda x: 'trending' if x > 25 else 'choppy'
    )
    
    return df
```

#### B. Cross-Asset Features

```python
def add_correlation_features(btc_df, eth_df, sp500_df):
    # Bitcoin-Ethereum correlation
    btc_df['btc_eth_corr'] = btc_df['close'].rolling(24).corr(eth_df['close'])
    
    # Crypto-Stocks correlation
    btc_df['crypto_stocks_corr'] = btc_df['close'].rolling(24).corr(sp500_df['close'])
    
    return btc_df
```

#### C. Time-Based Features

```python
def add_temporal_features(df):
    # Day of week (crypto has weekly patterns)
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    
    # Hour of day (volatility patterns)
    df['hour'] = df['timestamp'].dt.hour
    
    # US market hours
    df['us_market_hours'] = df['hour'].between(14, 21)  # UTC
    
    # Weekend effect
    df['is_weekend'] = df['day_of_week'].isin([5, 6])
    
    return df
```

#### D. Advanced Price Patterns

```python
def add_pattern_features(df):
    # Support/Resistance levels
    df['distance_to_swing_high'] = df['close'] / df['high'].rolling(48).max()
    df['distance_to_swing_low'] = df['close'] / df['low'].rolling(48).min()
    
    # Volume profile
    df['volume_pct_rank'] = df['volume'].rolling(168).rank(pct=True)
    
    # Price momentum quality
    df['returns_sharpe'] = (
        df['return_1h'].rolling(24).mean() / 
        df['return_1h'].rolling(24).std()
    )
    
    return df
```

---

### 3. **Model Optimization**

#### A. Hyperparameter Tuning

**Use Optuna for Automated Tuning:**

```python
import optuna

def objective(trial):
    params = {
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 100),
        'lambda_l1': trial.suggest_float('lambda_l1', 0.0, 10.0),
        'lambda_l2': trial.suggest_float('lambda_l2', 0.0, 10.0),
    }
    
    model = lgb.train(params, train_data, valid_sets=[val_data])
    preds = model.predict(X_val)
    
    # Optimize for Sharpe ratio, not accuracy!
    sharpe = calculate_sharpe_from_predictions(preds, y_val, prices_val)
    return sharpe

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)
```

#### B. Change Prediction Target

**Problem:** Binary up/down is too simplistic

**Better Approaches:**

**1. Regression (Predict Return Magnitude)**
```python
# Predict exact return instead of direction
df['target'] = df['close'].pct_change(1).shift(-1)

# Focus on larger moves
df['target_filtered'] = df['target'].where(
    abs(df['target']) > 0.005,  # Only predict >0.5% moves
    0
)
```

**2. Multi-Class Classification**
```python
# Strong Up / Weak Up / Neutral / Weak Down / Strong Down
def create_multiclass_target(returns, thresholds=[-0.02, -0.005, 0.005, 0.02]):
    return pd.cut(returns, bins=[-np.inf] + thresholds + [np.inf], labels=[0,1,2,3,4])

df['target'] = create_multiclass_target(df['future_return'])
```

**3. Ranking / Learning to Rank**
```python
# Predict which asset will perform best (for portfolio construction)
# Use LightGBM LambdaRank objective
```

#### C. Ensemble Methods

**Combine Multiple Models:**

```python
class EnsembleModel:
    def __init__(self):
        self.models = {
            'lgb': LightGBMModel(),
            'rf': RandomForestModel(),
            'xgb': XGBoostModel(),
            'linear': LogisticRegression()
        }
        self.weights = {}
    
    def train(self, X_train, y_train, X_val, y_val):
        predictions = {}
        
        for name, model in self.models.items():
            model.train(X_train, y_train)
            predictions[name] = model.predict(X_val)
        
        # Optimize weights on validation set
        self.weights = self.optimize_weights(predictions, y_val)
    
    def predict(self, X):
        preds = {name: model.predict(X) for name, model in self.models.items()}
        return sum(preds[name] * self.weights[name] for name in preds)
```

---

### 4. **Better Training Methodology**

#### A. Proper Walk-Forward Validation

**Current:** Simple 70/15/15 split ❌

**Better:** Rolling window walk-forward

```python
class RollingWalkForward:
    def __init__(self, train_window=60, test_window=7):
        self.train_window = train_window  # days
        self.test_window = test_window
    
    def split(self, df):
        splits = []
        start = 0
        
        while start + self.train_window + self.test_window <= len(df):
            train_end = start + self.train_window
            test_end = train_end + self.test_window
            
            train_idx = df.index[start:train_end]
            test_idx = df.index[train_end:test_end]
            
            splits.append((train_idx, test_idx))
            start += self.test_window  # Slide forward
        
        return splits
```

#### B. Purging and Embargo

**Prevent Look-Ahead Bias:**

```python
def purge_and_embargo(train_idx, test_idx, embargo_days=1):
    """
    Remove overlapping samples and add embargo period
    """
    # Remove samples from train that overlap with test
    purge_mask = train_idx < test_idx[0]
    train_idx_purged = train_idx[purge_mask]
    
    # Remove embargo period from end of train
    embargo_samples = embargo_days * 24  # hourly data
    train_idx_final = train_idx_purged[:-embargo_samples]
    
    return train_idx_final, test_idx
```

---

### 5. **Smarter Trading Strategy**

#### A. Position Sizing

**Current:** All-in or all-out ❌

**Better:** Kelly Criterion or Volatility-Based Sizing

```python
class PositionSizer:
    def __init__(self, method='volatility'):
        self.method = method
    
    def calculate_size(self, prediction_prob, volatility, capital):
        if self.method == 'kelly':
            # Kelly Criterion
            win_prob = prediction_prob
            win_loss_ratio = 1.5  # From historical analysis
            kelly_fraction = (win_prob * win_loss_ratio - (1 - win_prob)) / win_loss_ratio
            kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Cap at 25%
            
        elif self.method == 'volatility':
            # Target volatility
            target_vol = 0.02  # 2% daily
            position_size = target_vol / volatility
            kelly_fraction = min(position_size, 1.0)
        
        return kelly_fraction * capital
```

#### B. Dynamic Thresholds

**Current:** Fixed 55% threshold ❌

**Better:** Adaptive based on market regime

```python
def get_threshold(volatility_regime, trend_strength):
    """
    Adjust prediction threshold based on market conditions
    """
    thresholds = {
        'low_vol_strong_trend': 0.52,    # Lower threshold OK
        'low_vol_weak_trend': 0.58,      # Higher threshold needed
        'high_vol_strong_trend': 0.55,   # Medium threshold
        'high_vol_weak_trend': 0.65,     # Very high threshold
    }
    
    key = f"{'low' if volatility < 0.02 else 'high'}_vol_"
    key += f"{'strong' if trend_strength > 0.01 else 'weak'}_trend"
    
    return thresholds.get(key, 0.60)
```

#### C. Multiple Time Horizon Strategy

```python
class MultiHorizonStrategy:
    """
    Combine predictions across different timeframes
    """
    def __init__(self):
        self.models = {
            '1h': LightGBMModel(),   # Short-term
            '4h': LightGBMModel(),   # Medium-term
            '1d': LightGBMModel(),   # Long-term
        }
    
    def get_signal(self, features):
        signals = {}
        for horizon, model in self.models.items():
            signals[horizon] = model.predict_proba(features)
        
        # Only trade if all timeframes agree
        if all(s > 0.55 for s in signals.values()):
            return 'BUY'
        elif all(s < 0.45 for s in signals.values()):
            return 'SELL'
        else:
            return 'HOLD'
```

---

### 6. **Risk Management & Execution**

#### A. Better Stop-Loss Strategy

```python
class AdaptiveStopLoss:
    def __init__(self):
        self.method = 'atr'  # or 'percentile', 'support_resistance'
    
    def calculate_stop(self, entry_price, atr, direction):
        if self.method == 'atr':
            # ATR-based stop (2x ATR)
            stop_distance = 2 * atr
        
        if direction == 'LONG':
            stop_loss = entry_price - stop_distance
        else:
            stop_loss = entry_price + stop_distance
        
        return stop_loss
```

#### B. Slippage & Execution

**Account for Real-World Costs:**

```python
execution_costs = {
    'maker_fee': 0.0001,      # 0.01% maker
    'taker_fee': 0.0004,      # 0.04% taker (more realistic)
    'slippage': 0.0005,       # 0.05% market impact
    'spread': 0.0002,         # Bid-ask spread
}

total_cost = sum(execution_costs.values())  # ~0.12% per trade
# Need >0.25% edge to profit after costs!
```

---

### 7. **Evaluation Metrics**

**Don't Optimize for Accuracy!** ❌

**Optimize for:**

1. **Sharpe Ratio** (risk-adjusted returns)
2. **Calmar Ratio** (return / max drawdown)
3. **Win Rate × Avg Win / Loss Ratio**
4. **Maximum Drawdown Duration**
5. **Profit Factor** (gross profit / gross loss)

```python
def calculate_trading_metrics(predictions, prices, signals):
    returns = calculate_strategy_returns(predictions, prices, signals)
    
    metrics = {
        'sharpe': returns.mean() / returns.std() * np.sqrt(252 * 24),
        'sortino': returns.mean() / returns[returns < 0].std() * np.sqrt(252 * 24),
        'max_drawdown': calculate_max_drawdown(returns.cumsum()),
        'calmar': returns.mean() * 252 * 24 / abs(calculate_max_drawdown(returns.cumsum())),
        'win_rate': (returns > 0).mean(),
        'profit_factor': returns[returns > 0].sum() / abs(returns[returns < 0].sum()),
    }
    
    return metrics
```

---

### 8. **Realistic Expectations**

> [!IMPORTANT]
> **Crypto is EXTREMELY challenging for ML/algo trading**

**Reality Check:**

✅ **Achievable (with lots of work):**
- Sharpe Ratio: 0.5 - 1.5
- Annual Return: 10-30% (after costs)
- Max Drawdown: 15-30%
- Win Rate: 45-55%

❌ **Not Realistic:**
- Consistent 100%+ annual returns
- 70%+ win rate
- <5% drawdowns
- Works on all coins equally

**Why Crypto is Hard:**
- Extremely high noise-to-signal ratio
- News/sentiment dominated
- Manipulated markets (whales, wash trading)
- High volatility
- Regime changes (bull/bear cycles)
- Regulatory uncertainty

---

## 📋 Prioritized Action Plan

### Phase 1: Quick Wins (1-2 days)

1. ✅ **Get more data** - Switch to Binance API for 1-2 years history
2. ✅ **Add temporal features** - Day of week, hour, weekend effects
3. ✅ **Hyperparameter tuning** - Use Optuna for LightGBM
4. ✅ **Change target** - Try regression instead of classification
5. ✅ **Adjust strategy** - Lower transaction costs, dynamic thresholds

**Expected Improvement:** -10% → Break-even

### Phase 2: Feature Engineering (3-5 days)

6. ✅ **Volatility regimes** - Market state detection
7. ✅ **Cross-asset features** - BTC-ETH correlation
8. ✅ **Advanced patterns** - Support/resistance, momentum quality
9. ✅ **Add sentiment** - Fear & Greed Index
10. ✅ **On-chain metrics** - Exchange flows, active addresses

**Expected Improvement:** 0% → +5-10% annually

### Phase 3: Advanced Models (1 week)

11. ✅ **Ensemble models** - Combine LightGBM + XGBoost + Linear
12. ✅ **Walk-forward validation** - Proper backtesting
13. ✅ **Position sizing** - Kelly Criterion
14. ✅ **Multi-timeframe** - 1h + 4h + 1d models
15. ✅ **Adaptive stops** - ATR-based risk management

**Expected Improvement:** +5-10% → +15-25% annually

### Phase 4: Production (Ongoing)

16. ✅ **Paper trading** - Test with fake money first!
17. ✅ **Monitoring dashboard** - Track live performance
18. ✅ **Auto-retraining** - Monthly model updates
19. ✅ **Risk limits** - Max drawdown caps, position limits
20. ✅ **Gradual scaling** - Start small, scale slowly

---

## 🎯 Recommended Next Steps

**For You Right Now:**

1. **Get More Data First** (CRITICAL)
   ```bash
   # Install Binance connector
   pip install python-binance
   
   # Fetch 2 years of hourly data
   python scripts/fetch_binance_data.py --symbols BTC ETH BNB --days 730
   ```

2. **Add Temporal Features** (Easy win)
   - Modify `src/features/transform.py`
   - Add hour, day of week, weekend indicators

3. **Tune Hyperparameters** (Medium effort, high impact)
   - Use Optuna as shown above
   - Optimize for Sharpe, not accuracy

4. **Test with Paper Trading** (Before real money!)
   - Forward test for 30 days minimum
   - Track all metrics

---

## ⚠️ Critical Warnings

> [!CAUTION]
> **Never trade real money until:**
> - ✅ 6+ months of positive paper trading
> - ✅ Sharpe ratio > 1.0 consistently
> - ✅ Tested across bull AND bear markets
> - ✅ Understand every line of code
> - ✅ Have proper risk management

> [!WARNING]
> **Common Pitfalls:**
> - Overfitting on limited data
> - Not accounting for transaction costs
> - Ignoring market regimes
> - Testing on same data used for training
> - Underestimating execution challenges

---

## 📚 Resources

**Learning:**
- "Advances in Financial Machine Learning" by Marcos López de Prado
- "Machine Learning for Algorithmic Trading" by Stefan Jansen
- "Quantitative Trading" by Ernest Chan

**Data Sources:**
- Binance API (free, 3 years+)
- Alternative.me (Fear & Greed Index)
- Glassnode (on-chain data - paid)
- CryptoCompare (historical + sentiment)

**Tools:**
- Backtesting.py (proper vectorized backtesting)
- PyAlgoTrade (event-driven backtesting)
- VectorBT (fast portfolio backtesting)
- QuantStats (performance analysis)

---

## 🎊 Summary

**Current State:** Baseline models with negative returns  
**Target State:** Profitable, risk-managed trading system  
**Estimated Effort:** 2-4 weeks of intensive work  
**Success Probability:** Medium (crypto is hard!)  

**Key Takeaways:**
1. More data > Better models
2. Feature engineering > Model complexity
3. Risk management > Prediction accuracy
4. Paper trade > Real trade
5. Patience > Greed

Start with Phase 1 quick wins, measure improvements, iterate!
