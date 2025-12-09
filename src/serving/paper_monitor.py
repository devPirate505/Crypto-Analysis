"""
Paper Trading Monitor - Background Service
Monitors active paper trading positions and auto-closes them when SL/TP is hit.
Run this script in the background while using the dashboard.
"""
import json
import time
from pathlib import Path
from datetime import datetime
import logging

from src.serving.live_fetcher import LivePriceFetcher
from src.serving.paper_storage import PaperTradingStorage

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PaperTradingMonitor:
    """Monitors paper trading positions and auto-closes when SL/TP hit."""
    
    def __init__(self, check_interval=10):
        """
        Args:
            check_interval: Seconds between price checks
        """
        self.storage = PaperTradingStorage()
        self.fetcher = LivePriceFetcher()
        self.check_interval = check_interval
        self.running = False
        
    def check_position(self, position, coin, current_price):
        """
        Check if position should be closed based on SL/TP.
        
        Returns:
            tuple: (should_close, reason) where reason is 'SL' or 'TP' or None
        """
        if not position or 'stop_loss' not in position and 'take_profit' not in position:
            return False, None
            
        side = position.get('side', 'long')
        
        if side == 'long':
            # Long position
            if 'stop_loss' in position and current_price <= position['stop_loss']:
                return True, 'Stop Loss Hit'
            elif 'take_profit' in position and current_price >= position['take_profit']:
                return True, 'Take Profit Hit'
        else:
            # Short position
            if 'stop_loss' in position and current_price >= position['stop_loss']:
                return True, 'Stop Loss Hit'
            elif 'take_profit' in position and current_price <= position['take_profit']:
                return True, 'Take Profit Hit'
                
        return False, None
    
    def close_position(self, state, coin, current_price, reason):
        """Close position and record trade."""
        pos = state['position']
        
        # Calculate P&L
        if pos['side'] == 'long':
            exit_value = pos['size'] * current_price
            pnl = exit_value - (pos['size'] * pos['entry_price'])
            pnl_pct = (current_price - pos['entry_price']) / pos['entry_price'] * 100
        else:
            pnl = (pos['entry_price'] - current_price) * pos['size']
            pnl_pct = (pos['entry_price'] - current_price) / pos['entry_price'] * 100
            exit_value = (pos['size'] * pos['entry_price']) + pnl
        
        # Update balance
        state['balance'] += exit_value
        
        # Record trade
        trade = {
            'coin': coin,
            'side': pos['side'],
            'entry_price': pos['entry_price'],
            'exit_price': current_price,
            'size': pos['size'],
            'pnl': pnl,
            'pnl_pct': pnl_pct,
           'predicted': pos.get('predicted_return', 0) * 100,
            'entry_time': str(pos.get('entry_time', datetime.now())),
            'exit_time': str(datetime.now()),
            'close_reason': reason,
            'auto_closed': True
        }
        state['trades'].append(trade)
        
        # Clear position
        state['position'] = None
        
        # Log closure
        logger.info(f"🎯 {reason}! Closed {pos['side'].upper()} {coin.upper()} at ${current_price:,.2f}")
        logger.info(f"   P&L: ${pnl:+.2f} ({pnl_pct:+.2f}%)")
        
        return state
    
    def monitor_loop(self):
        """Main monitoring loop."""
        logger.info("🚀 Paper Trading Monitor started")
        logger.info(f"   Checking every {self.check_interval} seconds")
        logger.info("   Press Ctrl+C to stop")
        
        self.running = True
        
        try:
            while self.running:
                # Load current state
                state = self.storage.load_state()
                
                if state['position']:
                    pos = state['position']
                    coin = pos.get('coin', 'bitcoin')  # Default to bitcoin if not specified
                    
                    try:
                        # Get current price
                        current_price = self.fetcher.get_current_price(coin)
                        
                        # Calculate current P&L
                        if pos['side'] == 'long':
                            pnl_pct = (current_price - pos['entry_price']) / pos['entry_price'] * 100
                        else:
                            pnl_pct = (pos['entry_price'] - current_price) / pos['entry_price'] * 100
                        
                        logger.info(f"📊 Monitoring {coin.upper()} {pos['side'].upper()} @ ${pos['entry_price']:,.0f}")
                        logger.info(f"   Current: ${current_price:,.2f} | P&L: {pnl_pct:+.2f}%")
                        
                        # Check SL/TP
                        if 'stop_loss' in pos:
                            logger.info(f"   🛑 SL: ${pos['stop_loss']:,.2f}")
                        if 'take_profit' in pos:
                            logger.info(f"   🎯 TP: ${pos['take_profit']:,.2f}")
                        
                        should_close, reason = self.check_position(pos, coin, current_price)
                        
                        if should_close:
                            # Close position and save
                            state = self.close_position(state, coin, current_price, reason)
                            self.storage.save_state(
                                state['trades'],
                                state['balance'],
                                state['position']
                            )
                            logger.info("💾 State saved. Position closed.")
                            logger.info("=" * 50)
                            
                    except Exception as e:
                        logger.error(f"Error checking {coin}: {e}")
                else:
                    logger.debug("No active position")
                
                # Wait before next check
                time.sleep(self.check_interval)
                
        except KeyboardInterrupt:
            logger.info("\n👋 Monitor stopped by user")
            self.running = False
        except Exception as e:
            logger.error(f"Monitor error: {e}")
            self.running = False


def main():
    """Run the paper trading monitor."""
    print("""
╔══════════════════════════════════════════════════════════╗
║        Paper Trading Monitor - Background Service        ║
╚══════════════════════════════════════════════════════════╝

This script will monitor your paper trading positions and
automatically close them when Stop Loss or Take Profit is hit.

Features:
  ✅ Real-time price monitoring every 10 seconds
  ✅ Automatic SL/TP detection
  ✅ Auto-closes positions when triggered
  ✅ Records trades with closure reason
  ✅ Runs independently of dashboard

Keep this running while paper trading!
""")
    
    monitor = PaperTradingMonitor(check_interval=10)
    monitor.monitor_loop()


if __name__ == "__main__":
    main()
