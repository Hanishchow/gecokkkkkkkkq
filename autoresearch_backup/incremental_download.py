#!/usr/bin/env python3
"""
Incremental Data Downloader for GEOCK
Downloads BindingDB data in chunks while maintaining efficiency
"""
import os
import pickle
import time
import json

CACHE = 'CACHE_DIR / '
STATE_FILE = CACHE + 'download_state.json'

def load_state():
    """Load download state"""
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, 'r') as f:
            return json.load(f)
    return {'downloaded': 0, 'chunks': [], 'last_update': None}

def save_state(state):
    """Save download state"""
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2)

def get_current_count():
    """Get current data count"""
    try:
        with open(CACHE + 'lp_new_features_8k.pkl', 'rb') as f:
            data1 = pickle.load(f)
        with open(CACHE + 'geock_training_data.pkl', 'rb') as f:
            data2 = pickle.load(f)
        
        seen = set()
        for d in data1:
            seen.add(d['pdb_id'])
        for d in data2:
            seen.add(d['pdb_id'])
        return len(seen)
    except:
        return 0

def check_pending_downloads():
    """Check what's needed"""
    current = get_current_count()
    target = 100000
    needed = target - current
    
    state = load_state()
    state['current'] = current
    state['needed'] = needed
    state['progress'] = f"{current}/{target}"
    state['pct'] = f"{current/target*100:.1f}%"
    
    return state

if __name__ == '__main__':
    state = check_pending_downloads()
    print(f"=== Download Status ===")
    print(f"Current: {state['current']:,}")
    print(f"Target: 100,000")
    print(f"Needed: {state['needed']:,}")
    print(f"Progress: {state['pct']}")
    print(f"\n=== Pending Downloads ===")
    print("1. BindingDB TSV (~80k needed)")
    print("2. ChEMBL additional")
    print("3. Other sources")