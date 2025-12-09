"""
Persistent storage for paper trading state.
"""
import json
from pathlib import Path
from datetime import datetime


class PaperTradingStorage:
    """Handles persistent storage of paper trading state."""
    
    def __init__(self, filepath="models/paper_trading_state.json"):
        self.filepath = Path(filepath)
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
    
    def load_state(self):
        """Load paper trading state from file."""
        if self.filepath.exists():
            try:
                with open(self.filepath, 'r') as f:
                    return json.load(f)
            except:
                return self._default_state()
        return self._default_state()
    
    def save_state(self, trades, balance, position):
        """Save paper trading state to file."""
        state = {
            'trades': trades,
            'balance': balance,
            'position': position,
            'last_updated': str(datetime.now())
        }
        with open(self.filepath, 'w') as f:
            json.dump(state, f, indent=2, default=str)
    
    def _default_state(self):
        """Return default state."""
        return {
            'trades': [],
            'balance': 10000.0,
            'position': None
        }
