"""
Streamlit dashboard for crypto ML predictions.
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import json

from src.models.lightgbm_model import LightGBMModel
from src.ingestion.storage import DataStorage
from src.utils.config import load_config

# Page configuration
st.set_page_config(
    page_title="Crypto ML Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .prediction-buy {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        font-weight: bold;
        text-align: center;
    }
    .prediction-sell {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.5rem;
        font-weight: bold;
        text-align: center;
    }
    .prediction-hold {
        background-color: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 0.5rem;
        font-weight: bold;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_models_and_config():
    """Load MTF regression models and configuration."""
    config = load_config()
    storage = DataStorage()
    
    coin_ids = config['data']['symbols']
    models = {}
    model_type = {}
    
    for coin_id in coin_ids:
        # Try to load MTF model first, fallback to optimized, then standard
        if Path(f"models/{coin_id}_lightgbm_mtf.joblib").exists():
            model_path = f"models/{coin_id}_lightgbm_mtf.joblib"
            model_type[coin_id] = "MTF"
        elif Path(f"models/{coin_id}_lightgbm_optimized.joblib").exists():
            model_path = f"models/{coin_id}_lightgbm_optimized.joblib"
            model_type[coin_id] = "Optimized"
        else:
            model_path = f"models/{coin_id}_lightgbm.joblib"
            model_type[coin_id] = "Standard"
            
        if Path(model_path).exists():
            model = LightGBMModel(task_type='regression')
            model.load(model_path)
            models[coin_id] = model
            
    return config, storage, models, model_type


@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_coin_data(coin_id: str):
    """Load processed data for a coin (prefer MTF data)."""
    # Try MTF data first
    mtf_path = Path(f"data/processed/{coin_id}_processed_mtf.parquet")
    if mtf_path.exists():
        df = pd.read_parquet(mtf_path)
        return df, "MTF"
    
    # Fallback to standard processed data
    storage = DataStorage()
    df = storage.load_processed(coin_id)
    return df, "Standard"


def create_price_chart(df: pd.DataFrame, coin_id: str):
    """Create interactive price chart."""
    fig = go.Figure()
    
    fig.add_trace(go.Candlestick(
        x=df['timestamp'],
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name='OHLC'
    ))
    
    fig.update_layout(
        title=f'{coin_id.upper()} Price Chart',
        yaxis_title='Price (USD)',
        xaxis_title='Date',
        height=500,
        template='plotly_white'
    )
    
    return fig


def create_indicator_chart(df: pd.DataFrame, indicator: str, coin_id: str):
    """Create indicator chart."""
    fig = go.Figure()
    
    if indicator in df.columns:
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df[indicator],
            mode='lines',
            name=indicator
        ))
        
    fig.update_layout(
        title=f'{coin_id.upper()} - {indicator}',
        yaxis_title=indicator,
        xaxis_title='Date',
        height=400,
        template='plotly_white'
    )
    
    return fig


def main():
    """Main dashboard function."""
    
    # Header
    st.markdown('<div class="main-header">📈 Crypto ML Dashboard</div>', unsafe_allow_html=True)
    
    # Initialize session state for paper trading
    if 'paper_trades' not in st.session_state:
        st.session_state.paper_trades = []
    if 'paper_balance' not in st.session_state:
        st.session_state.paper_balance = 10000.0
    if 'paper_position' not in st.session_state:
        st.session_state.paper_position = None
    
    # Load models and config
    try:
        config, storage, models, model_type = load_models_and_config()
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.stop()
        
    # Sidebar
    st.sidebar.title("⚙️ Settings")
    
    # Coin selection
    coin_ids = config['data']['symbols']
    selected_coin = st.sidebar.selectbox(
        "Select Cryptocurrency",
        coin_ids,
        format_func=lambda x: x.upper()
    )
    
    # Page selection
    page = st.sidebar.radio(
        "Navigation",
        ["📊 Overview", "📈 Technical Indicators", "🤖 Model Predictions", "💰 Backtest Results", "📝 Paper Trading", "🎯 Feature Importance"]
    )
    
    # Load data
    try:
        df, data_type = load_coin_data(selected_coin)
        
        if df is None or len(df) == 0:
            st.error(f"No data available for {selected_coin}")
            st.stop()
        
        # Show data/model info in sidebar
        st.sidebar.markdown("---")
        st.sidebar.markdown(f"**Model:** {model_type.get(selected_coin, 'N/A')}")
        st.sidebar.markdown(f"**Data:** {data_type}")
        st.sidebar.markdown(f"**Features:** {len([c for c in df.columns if c not in ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'market_cap', 'coin_id']])}")
            
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()
        
    # Page: Overview
    if page == "📊 Overview":
        st.header(f"{selected_coin.upper()} Overview")
        
        # Current price metrics
        col1, col2, col3, col4 = st.columns(4)
        
        latest = df.iloc[-1]
        prev = df.iloc[-24] if len(df) > 24 else df.iloc[0]  # 24h ago
        
        with col1:
            st.metric("Current Price", f"${latest['close']:,.2f}", 
                     f"{((latest['close'] - prev['close']) / prev['close'] * 100):.2f}%")
        
        with col2:
            st.metric("24h High", f"${df.tail(24)['high'].max():,.2f}")
            
        with col3:
            st.metric("24h Low", f"${df.tail(24)['low'].min():,.2f}")
            
        with col4:
            if 'volume' in df.columns:
                st.metric("24h Volume", f"${df.tail(24)['volume'].sum():,.0f}")
            
        # Price chart
        # Price chart - show last 1 month
        st.plotly_chart(create_price_chart(df.tail(720), selected_coin), use_container_width=True)
        
        # Recent data
        st.subheader("Recent Data")
        st.dataframe(df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].tail(20), use_container_width=True)
        
    # Page: Technical Indicators
    elif page == "📈 Technical Indicators":
        st.header(f"{selected_coin.upper()} Technical Indicators")
        
        # Indicator selection
        available_indicators = [col for col in df.columns if any(x in col for x in ['rsi', 'macd', 'ema', 'bb', 'adx'])]
        
        selected_indicator = st.selectbox("Select Indicator", available_indicators)
        
        if selected_indicator:
            st.plotly_chart(create_indicator_chart(df.tail(720), selected_indicator, selected_coin), 
                          use_container_width=True)
            
        # Show multiple indicators
        col1, col2 = st.columns(2)
        
        with col1:
            if 'rsi_14' in df.columns:
                st.plotly_chart(create_indicator_chart(df.tail(300), 'rsi_14', selected_coin), 
                              use_container_width=True)
                
        with col2:
            if 'macd' in df.columns:
                st.plotly_chart(create_indicator_chart(df.tail(300), 'macd', selected_coin), 
                              use_container_width=True)
                
    # Page: Model Predictions (Regression)
    elif page == "🤖 Model Predictions":
        st.header(f"{selected_coin.upper()} Model Predictions")
        
        if selected_coin not in models:
            st.warning(f"No trained model available for {selected_coin}")
            st.stop()
            
        model = models[selected_coin]
        
        # Live Prediction Section
        st.subheader("Live Market Prediction")
        
        col1, col2 = st.columns([1, 2])
        with col1:
            predict_btn = st.button("🔮 Predict Future Price", use_container_width=True)
            
        if predict_btn:
            with st.spinner("Fetching live data and calculating features..."):
                try:
                    # Import here to avoid circular imports
                    from src.serving.live_fetcher import LivePriceFetcher, calculate_sl_tp
                    
                    # Fetch live data
                    fetcher = LivePriceFetcher()
                    live_data = fetcher.get_live_data_with_features(selected_coin)
                    
                    # Prepare features for model (using historical data structure)
                    # Note: In a real production system, we'd calculate features on live_data['klines']
                    # For now, we use the latest processed data features but with live price context
                    exclude_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'market_cap', 'coin_id', 'target', 'future_return']
                    feature_cols = [col for col in df.columns if col not in exclude_cols]
                    latest_features = df[feature_cols].iloc[-1:]
                    
                    # Make prediction
                    predicted_return = model.predict(latest_features)[0]
                    
                    # Calculate targets
                    current_price = live_data['current_price']
                    future_price = current_price * (1 + predicted_return)
                    
                    # Calculate SL/TP
                    atr = live_data['atr']
                    levels = calculate_sl_tp(current_price, atr, predicted_return)
                    
                    # Store in session state for paper trading
                    st.session_state.last_prediction = {
                        'coin': selected_coin,
                        'price': current_price,
                        'predicted_return': predicted_return,
                        'future_price': future_price,
                        'stop_loss': levels['stop_loss'],
                        'take_profit': levels['take_profit'],
                        'timestamp': datetime.now()
                    }
                    
                except Exception as e:
                    st.error(f"Prediction failed: {e}")
        
        # Display prediction results if available
        if 'last_prediction' in st.session_state and st.session_state.last_prediction['coin'] == selected_coin:
            pred = st.session_state.last_prediction
            
            # Signal
            pred_pct = pred['predicted_return'] * 100
            if pred_pct > 0.05:
                signal = "🟢 BUY"
                signal_color = "green"
            elif pred_pct < -0.05:
                signal = "🔴 SELL"
                signal_color = "red"
            else:
                signal = "🟡 HOLD"
                signal_color = "orange"
                
            # Main Metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Current Price", f"${pred['price']:,.2f}")
            with col2:
                st.metric("Predicted Price (1h)", f"${pred['future_price']:,.2f}", f"{pred_pct:.3f}%")
            with col3:
                st.markdown(f"### Signal: <span style='color:{signal_color}'>{signal}</span>", unsafe_allow_html=True)
                
            st.markdown("---")
            
            # Trading Plan
            st.subheader("Trading Plan")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("🛑 Stop Loss", f"${pred['stop_loss']:,.2f}", delta=f"-{(pred['price'] - pred['stop_loss']):.2f}", delta_color="inverse")
            with col2:
                st.metric("🎯 Take Profit", f"${pred['take_profit']:,.2f}", delta=f"+{(pred['take_profit'] - pred['price']):.2f}")
            with col3:
                risk_reward = (pred['take_profit'] - pred['price']) / (pred['price'] - pred['stop_loss'])
                st.metric("⚖️ Risk/Reward", f"1:{risk_reward:.1f}")
                
            # Action Buttons
            st.markdown("---")
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📝 Trade this on Paper", use_container_width=True):
                    # Set up paper trade
                    st.session_state.paper_position = {
                        'side': 'long' if pred_pct > 0 else 'short',
                        'entry_price': pred['price'],
                        'size': 1000 / pred['price'],  # Default $1000 size
                        'entry_time': pred['timestamp'],
                        'predicted_return': pred['predicted_return'],
                        'stop_loss': pred['stop_loss'],
                        'take_profit': pred['take_profit']
                    }
                    st.session_state.paper_balance -= 1000
                    st.success("Trade executed on Paper Trading account! Check the Paper Trading tab.")
                    
            with col2:
                if st.button("🔄 Refresh Prediction", use_container_width=True):
                    # Clear prediction to force refresh
                    del st.session_state.last_prediction
                    st.rerun()
            
    # Page: Backtest Results (Advanced)
    elif page == "💰 Backtest Results":
        st.header(f"{selected_coin.upper()} Backtest Results")
        
        # Load advanced backtest results (prefer advanced, fallback to optimized, then standard)
        backtest_path = None
        results_path = None
        
        # Try MTF first
        if Path(f"data/processed/{selected_coin}_backtest_mtf.parquet").exists():
            backtest_path = Path(f"data/processed/{selected_coin}_backtest_mtf.parquet")
            results_path = Path("models/backtest_results_mtf.json")
            st.info("📊 Showing **MTF** backtest results (Multi-Timeframe Model)")
        # Try advanced next
        elif Path(f"data/processed/{selected_coin}_backtest_advanced.parquet").exists():
            backtest_path = Path(f"data/processed/{selected_coin}_backtest_advanced.parquet")
            results_path = Path("models/backtest_results_advanced.json")
            st.info("📊 Showing **Advanced** backtest results (Ensemble + Stop-losses)")
        # Fall back to optimized
        elif Path(f"data/processed/{selected_coin}_backtest_optimized.parquet").exists():
            backtest_path = Path(f"data/processed/{selected_coin}_backtest_optimized.parquet")
            results_path = Path("models/backtest_results_optimized.json")
            st.info("📊 Showing **Optimized** backtest results (Regression model)")
        # Fall back to standard
        elif Path(f"data/processed/{selected_coin}_backtest.parquet").exists():
            backtest_path = Path(f"data/processed/{selected_coin}_backtest.parquet")
            results_path = Path("models/backtest_results.json")
            st.warning("⚠️ Showing **Baseline** backtest results (old model)")
        
        if backtest_path and backtest_path.exists() and results_path.exists():
            backtest_df = pd.read_parquet(backtest_path)
            
            with open(results_path, 'r') as f:
                results = json.load(f)
                
            if selected_coin in results:
                metrics = results[selected_coin]
                # Display metrics - handle both old and new format
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    ann_ret = metrics.get('annualized_return', metrics.get('total_return', 0))
                    st.write(f"**Annualized Return:** {ann_ret*100:.2f}%")
                    
                with col2:
                    vol = metrics.get('volatility', 0)
                    st.write(f"**Volatility:** {vol*100:.2f}%")
                    
                with col3:
                    pf = metrics.get('profit_factor', 0)
                    st.write(f"**Profit Factor:** {pf:.2f}" if pf else "**Profit Factor:** N/A")
                    
                with col4:
                    avg_win = metrics.get('avg_win', 0)
                    avg_loss = metrics.get('avg_loss', 0)
                    st.write(f"**Avg Win:** {avg_win*100:.2f}%")
                    st.write(f"**Avg Loss:** {avg_loss*100:.2f}%")
                    
                # Equity curve
                fig = go.Figure()
                
                # Handle different column names
                equity_col = 'equity_curve' if 'equity_curve' in backtest_df.columns else 'equity'
                
                fig.add_trace(go.Scatter(
                    x=backtest_df['timestamp'] if 'timestamp' in backtest_df.columns else backtest_df.index,
                    y=backtest_df[equity_col],
                    mode='lines',
                    name='Strategy Equity',
                    line=dict(color='blue', width=2)
                ))
                
                # Buy and hold (if cumulative_returns exists)
                if 'cumulative_returns' in backtest_df.columns:
                    buy_hold_equity = config['backtest']['initial_capital'] * (1 + backtest_df['cumulative_returns'])
                    fig.add_trace(go.Scatter(
                        x=backtest_df.index,
                        y=buy_hold_equity,
                        mode='lines',
                        name='Buy & Hold',
                        line=dict(color='gray', width=2, dash='dash')
                    ))
                
                fig.update_layout(
                    title='Equity Curve',
                    yaxis_title='Equity (USD)',
                    xaxis_title='Time',
                    height=500,
                    template='plotly_white'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Additional metrics
                st.subheader("Detailed Metrics")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    initial_cap = metrics.get('initial_capital', 10000)
                    final_eq = metrics.get('final_equity', initial_cap * (1 + metrics.get('total_return', 0)))
                    total_trades = metrics.get('total_trades', 0)
                    winning_trades = metrics.get('winning_trades', int(total_trades * metrics.get('win_rate', 0)))
                    
                    st.write(f"**Initial Capital:** ${initial_cap:,.2f}")
                    st.write(f"**Final Equity:** ${final_eq:,.2f}")
                    st.write(f"**Total Trades:** {total_trades}")
                    st.write(f"**Winning Trades:** {winning_trades}")
                    
                with col2:
                    ann_ret = metrics.get('annualized_return', metrics.get('total_return', 0) * 4)  # Rough estimate
                    vol = metrics.get('volatility', 0)
                    pf = metrics.get('profit_factor', 0)
                    avg_win = metrics.get('avg_win', 0)
                    avg_loss = metrics.get('avg_loss', 0)
                    
                    st.write(f"**Annualized Return:** {ann_ret*100:.2f}%")
                    st.write(f"**Volatility:** {vol*100:.2f}%")
                    st.write(f"**Profit Factor:** {pf:.2f}" if pf else "**Profit Factor:** N/A")
                    st.write(f"**Avg Win:** {avg_win*100:.2f}%")
                    st.write(f"**Avg Loss:** {avg_loss*100:.2f}%")
                    
        else:
            st.warning("No backtest results available. Run backtesting first.")
            
    # Page: Feature Importance
    elif page == "🎯 Feature Importance":
        st.header(f"{selected_coin.upper()} Feature Importance")
        
        if selected_coin not in models:
            st.warning(f"No trained model available for {selected_coin}")
            st.stop()
            
        model = models[selected_coin]
        
        # Get feature importance
        exclude_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'market_cap', 'coin_id']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        importance_df = model.get_feature_importance(feature_cols)
        
        # Top 20 features
        top_features = importance_df.head(20)
        
        fig = px.bar(
            top_features,
            x='importance',
            y='feature',
            orientation='h',
            title='Top 20 Important Features',
            labels={'importance': 'Importance Score', 'feature': 'Feature'}
        )
        
        fig.update_layout(height=600, template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)
        
        # Full table
        st.subheader("All Features")
        st.dataframe(importance_df, use_container_width=True, height=400)
    
    # Page: Paper Trading
    elif page == "📝 Paper Trading":
        st.header(f"{selected_coin.upper()} Paper Trading")
        
        st.info("📈 **Paper Trading Mode** - Practice trading with virtual money using live signals from the MTF model.")
            
        # Live Data Toggle
        use_live_data = st.checkbox("Use Live Binance Data", value=True)
        
        if selected_coin in models:
            model = models[selected_coin]
            
            try:
                # Get price and prediction
                if use_live_data:
                    from src.serving.live_fetcher import LivePriceFetcher
                    fetcher = LivePriceFetcher()
                    current_price = fetcher.get_current_price(selected_coin)
                else:
                    current_price = df['close'].iloc[-1]
                
                # Use last prediction if available, otherwise calculate
                if 'last_prediction' in st.session_state and st.session_state.last_prediction['coin'] == selected_coin:
                    predicted_return = st.session_state.last_prediction['predicted_return']
                else:
                    # Fallback to historical prediction
                    exclude_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'market_cap', 'coin_id', 'target', 'future_return']
                    feature_cols = [col for col in df.columns if col not in exclude_cols]
                    latest_features = df[feature_cols].iloc[-1:]
                    predicted_return = model.predict(latest_features)[0]
                
                # Signal logic
                if predicted_return > 0.0003:
                    signal = "🟢 BUY"
                    signal_color = "green"
                elif predicted_return < -0.0003:
                    signal = "🔴 SELL"
                    signal_color = "red"
                else:
                    signal = "🟡 HOLD"
                    signal_color = "orange"
                    
                # Display current signal
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Current Price", f"${current_price:,.2f}")
                with col2:
                    st.metric("Predicted Return", f"{predicted_return*100:.3f}%")
                with col3:
                    st.markdown(f"### Signal: <span style='color:{signal_color}'>{signal}</span>", unsafe_allow_html=True)
                    
                st.markdown("---")
                
                # Paper trading controls
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Paper Balance", f"${st.session_state.paper_balance:,.2f}")
                    
                with col2:
                    position_text = "None"
                    if st.session_state.paper_position:
                        pos = st.session_state.paper_position
                        # Calculate P&L based on current price
                        if pos['side'] == 'long':
                            pnl = (current_price - pos['entry_price']) / pos['entry_price'] * 100
                            pnl_val = (current_price - pos['entry_price']) * pos['size']
                        else:
                            pnl = (pos['entry_price'] - current_price) / pos['entry_price'] * 100
                            pnl_val = (pos['entry_price'] - current_price) * pos['size']
                            
                        position_text = f"{pos['size']:.2f} @ ${pos['entry_price']:,.0f}"
                        st.metric("Current Position", position_text, f"{pnl:+.2f}% (${pnl_val:+.2f})")
                    else:
                        st.metric("Current Position", "None")
                    
                with col3:
                    trade_amount = st.number_input("Trade Amount ($)", min_value=100.0, max_value=st.session_state.paper_balance, value=min(1000.0, st.session_state.paper_balance), step=100.0)
                    
                with col4:
                    st.write("")
                    
                # Trade buttons
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    # Determine button label and action based on signal
                    if predicted_return > 0:
                        btn_label = "🟢 ENTER LONG"
                        side = 'long'
                    else:
                        btn_label = "🔴 ENTER SHORT"
                        side = 'short'
                        
                    if st.button(btn_label, use_container_width=True, disabled=st.session_state.paper_position is not None):
                        if st.session_state.paper_position is None:
                            size = trade_amount / current_price
                            st.session_state.paper_position = {
                                'side': side,
                                'entry_price': current_price,
                                'size': size,
                                'entry_time': datetime.now(),
                                'predicted_return': predicted_return
                            }
                            st.session_state.paper_balance -= trade_amount
                            st.success(f"Entered {side.upper()} {size:.4f} {selected_coin.upper()} at ${current_price:,.2f}")
                            st.rerun()
                            
                with col2:
                    if st.button("❌ CLOSE POSITION", use_container_width=True, disabled=st.session_state.paper_position is None):
                        if st.session_state.paper_position:
                            pos = st.session_state.paper_position
                            
                            if pos['side'] == 'long':
                                exit_value = pos['size'] * current_price
                                pnl = exit_value - (pos['size'] * pos['entry_price'])
                                pnl_pct = (current_price - pos['entry_price']) / pos['entry_price'] * 100
                            else:
                                # Short P&L: (Entry - Exit) * Size
                                pnl = (pos['entry_price'] - current_price) * pos['size']
                                pnl_pct = (pos['entry_price'] - current_price) / pos['entry_price'] * 100
                                exit_value = (pos['size'] * pos['entry_price']) + pnl
                            
                            st.session_state.paper_balance += exit_value
                            
                            # Record trade
                            st.session_state.paper_trades.append({
                                'coin': selected_coin,
                                'side': pos['side'],
                                'entry_price': pos['entry_price'],
                                'exit_price': current_price,
                                'size': pos['size'],
                                'pnl': pnl,
                                'pnl_pct': pnl_pct,
                                'predicted': pos['predicted_return'] * 100,
                                'entry_time': str(pos['entry_time']),
                                'exit_time': str(datetime.now())
                            })
                            
                            st.session_state.paper_position = None
                            
                            if pnl >= 0:
                                st.success(f"Closed position for +${pnl:.2f} ({pnl_pct:+.2f}%)")
                            else:
                                st.error(f"Closed position for ${pnl:.2f} ({pnl_pct:+.2f}%)")
                            st.rerun()
                            
                with col3:
                    if st.button("🔄 Reset Paper Account", use_container_width=True):
                        st.session_state.paper_balance = 10000.0
                        st.session_state.paper_position = None
                        st.session_state.paper_trades = []
                        st.success("Paper account reset to $10,000")
                        st.rerun()
                        
                # Trade history
                st.markdown("---")
                st.subheader("Trade History")
                
                if st.session_state.paper_trades:
                    trades_df = pd.DataFrame(st.session_state.paper_trades)
                    
                    # Summary stats
                    total_pnl = trades_df['pnl'].sum()
                    win_rate = len(trades_df[trades_df['pnl'] > 0]) / len(trades_df) * 100
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Trades", len(trades_df))
                    with col2:
                        st.metric("Total P&L", f"${total_pnl:,.2f}")
                    with col3:
                        st.metric("Win Rate", f"{win_rate:.1f}%")
                        
                    st.dataframe(trades_df[['coin', 'side', 'entry_price', 'exit_price', 'pnl', 'pnl_pct', 'predicted']], use_container_width=True)
                else:
                    st.info("No trades yet. Use the buttons above to start paper trading!")
                    
            except Exception as e:
                st.error(f"Error generating signal: {e}")
        else:
            st.warning(f"No model available for {selected_coin}")
            
    # Page: Feature Importance
    elif page == "🎯 Feature Importance":
        st.header(f"{selected_coin.upper()} Feature Importance")
        
        if selected_coin not in models:
            st.warning(f"No trained model available for {selected_coin}")
            st.stop()
            
        model = models[selected_coin]
        
        # Get feature importance
        exclude_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'market_cap', 'coin_id']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        importance_df = model.get_feature_importance(feature_cols)
        
        # Top 20 features
        top_features = importance_df.head(20)
        
        fig = px.bar(
            top_features,
            x='importance',
            y='feature',
            orientation='h',
            title='Top 20 Important Features',
            labels={'importance': 'Importance Score', 'feature': 'Feature'}
        )
        
        fig.update_layout(height=600, template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)
        
        # Full table
        st.subheader("All Features")
        st.dataframe(importance_df, use_container_width=True, height=400)


if __name__ == "__main__":
    main()
