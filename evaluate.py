"""
COMPLETE IMPLEMENTATION: Hadith-Inspired Features with Temporal Sequence Analysis
Full production-ready code with all features (26 Original + 16 Temporal = 42 Total)

Research: Multi-Axis Trust Modeling for Interpretable Account Hijacking Detection

DATA:
=====
clue_lds_logs.csv: CLUE-LDS subset used in this research, the logs span approximately 13 days
(2017-07-07 to 2017-07-21) and include 500{,}000 events from 77 unique users, covering 24
distinct event types.

USAGE:
======
Command Line:
    python evaluate.py --csv clue_lds_logs.csv --max-events 500000 --n-hijacks 30 --save-plots

Jupyter:
    
    results = run_comparison(csv='clue_lds_logs.csv', max_events=500000, save_plots=True)
"""

import os
import sys
import json
import gzip
import argparse
import warnings
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
from scipy.stats import entropy
from scipy.spatial.distance import jensenshannon

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score, accuracy_score,
    precision_score, recall_score, classification_report, roc_curve,
    precision_recall_curve
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

import matplotlib.pyplot as plt
import seaborn as sns

# Try to import torch for LSTM
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("WARNING: PyTorch not available. LSTM classifier will be skipped.")

warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

MAX_EVENTS = 1_000_000
MAX_USERS = 500
WINDOW_SIZE = 50
STEP_SIZE = 25
N_HIJACKS = 50
HIJACK_DURATION_HOURS = 8

# ============================================================
# ARGUMENT PARSING
# ============================================================

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Complete Hadith+Temporal Features Comparison',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--csv', type=str, help='Path to CSV file')
    parser.add_argument('--json-dir', type=str, help='Directory with JSON files')
    parser.add_argument('--json-file', type=str, help='Path to JSON file')
    parser.add_argument('--max-events', type=int, default=MAX_EVENTS)
    parser.add_argument('--max-users', type=int, default=MAX_USERS)
    parser.add_argument('--window-size', type=int, default=WINDOW_SIZE)
    parser.add_argument('--step-size', type=int, default=STEP_SIZE)
    parser.add_argument('--n-hijacks', type=int, default=N_HIJACKS)
    parser.add_argument('--hijack-duration', type=float, default=HIJACK_DURATION_HOURS)
    parser.add_argument('--output-dir', type=str, default='.')
    parser.add_argument('--save-plots', action='store_true')
    
    return parser.parse_args()


class Args:
    """Arguments class for notebook usage."""
    def __init__(self, csv=None, json_dir=None, json_file=None, 
                 max_events=MAX_EVENTS, max_users=MAX_USERS,
                 window_size=WINDOW_SIZE, step_size=STEP_SIZE,
                 n_hijacks=N_HIJACKS, hijack_duration=HIJACK_DURATION_HOURS,
                 output_dir='.', save_plots=False):
        self.csv = csv
        self.json_dir = json_dir
        self.json_file = json_file
        self.max_events = max_events
        self.max_users = max_users
        self.window_size = window_size
        self.step_size = step_size
        self.n_hijacks = n_hijacks
        self.hijack_duration = hijack_duration
        self.output_dir = output_dir
        self.save_plots = save_plots


def is_notebook():
    """Check if running in Jupyter."""
    try:
        from IPython import get_ipython
        return get_ipython() is not None
    except ImportError:
        return False


# ============================================================
# DATA LOADING - CSV
# ============================================================

def load_csv_dataset(csv_path: str, max_events: int, max_users: int) -> pd.DataFrame:
    """Load dataset from CSV with flexible column mapping."""
    print(f"Loading CSV: {csv_path}")
    
    if not os.path.exists(csv_path):
        print(f"ERROR: File not found: {csv_path}")
        return None
    
    try:
        df = pd.read_csv(csv_path, low_memory=False)
    except Exception as e:
        print(f"ERROR reading CSV: {e}")
        return None
    
    print(f"  Loaded {len(df):,} rows")
    
    # Column name mapping
    col_map = {
        'uid': 'user_id', 'userId': 'user_id', 'user': 'user_id',
        'time': 'timestamp', '@timestamp': 'timestamp', 'datetime': 'timestamp',
        'type': 'event_type', 'eventType': 'event_type', 'event': 'event_type',
        'file': 'path', 'resource': 'path',
        'ip': 'ip_address', 'sourceIp': 'ip_address'
    }
    
    df.columns = df.columns.str.strip()
    for old, new in col_map.items():
        for col in df.columns:
            if col.lower() == old.lower() and new not in df.columns:
                df = df.rename(columns={col: new})
    
    # Check required columns
    required = ['user_id', 'timestamp', 'event_type']
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"ERROR: Missing columns: {missing}")
        return None
    
    # Add optional columns
    if 'path' not in df.columns:
        df['path'] = ''
    if 'ip_address' not in df.columns:
        df['ip_address'] = ''
    
    # Type conversion
    df['user_id'] = df['user_id'].astype(str)
    df['event_type'] = df['event_type'].astype(str).fillna('unknown')
    df['path'] = df['path'].astype(str).fillna('')
    df['ip_address'] = df['ip_address'].astype(str).fillna('')
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.dropna(subset=['timestamp'])
    
    # Limit users
    user_counts = df['user_id'].value_counts()
    if len(user_counts) > max_users:
        top_users = user_counts.head(max_users).index.tolist()
        df = df[df['user_id'].isin(top_users)]
    
    # Limit events
    if len(df) > max_events:
        df = df.head(max_events)
    
    df = df.sort_values(['user_id', 'timestamp']).reset_index(drop=True)
    
    # Generate IPs if missing
    if df['ip_address'].isna().all() or (df['ip_address'] == '').all():
        df = generate_synthetic_ips(df)
    
    print(f"  Final: {len(df):,} events, {df['user_id'].nunique()} users")
    return df


def generate_synthetic_ips(df: pd.DataFrame) -> pd.DataFrame:
    """Generate realistic synthetic IP addresses."""
    df = df.copy()
    user_ips = {}
    
    for user_id in df['user_id'].unique():
        primary = f"192.168.{np.random.randint(1,255)}.{np.random.randint(1,255)}"
        user_ips[user_id] = primary
    
    def assign_ip(row):
        primary = user_ips[row['user_id']]
        if np.random.random() < 0.9:
            return primary
        parts = primary.split('.')
        parts[3] = str(np.random.randint(1, 255))
        return '.'.join(parts)
    
    df['ip_address'] = df.apply(assign_ip, axis=1)
    return df


def load_dataset(args) -> pd.DataFrame:
    """Load dataset based on arguments."""
    if args.csv:
        return load_csv_dataset(args.csv, args.max_events, args.max_users)
    
    print("ERROR: Please specify --csv option")
    return None


def print_dataset_summary(df: pd.DataFrame):
    """Print dataset statistics."""
    print("\n" + "="*70)
    print("DATASET SUMMARY")
    print("="*70)
    print(f"Total events:       {len(df):,}")
    print(f"Unique users:       {df['user_id'].nunique():,}")
    print(f"Event types:        {df['event_type'].nunique():,}")
    print(f"Date range:         {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Time span:          {(df['timestamp'].max()-df['timestamp'].min()).days} days")
    print("\nTop 10 Event Types:")
    print(df['event_type'].value_counts().head(10).to_string())


# ============================================================
# HIJACK INJECTION
# ============================================================

def inject_hijacks_clue(df: pd.DataFrame, n_hijacks: int, 
                         hijack_duration_hours: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Inject realistic account hijacks."""
    df = df.copy()
    hijack_intervals = []
    
    user_counts = df.groupby('user_id').size()
    eligible = user_counts[user_counts >= 50].index.tolist()
    np.random.shuffle(eligible)
    
    if not eligible:
        print("WARNING: No eligible users for hijack")
        return df, pd.DataFrame(columns=['user_id', 'start', 'end'])
    
    all_types = df['event_type'].unique().tolist()
    susp_keywords = ['admin', 'delete', 'export', 'download', 'permission']
    susp_events = [et for et in all_types if any(kw in et.lower() for kw in susp_keywords)]
    
    if not susp_events:
        counts = df['event_type'].value_counts()
        susp_events = counts[counts <= counts.quantile(0.2)].index.tolist()
    if not susp_events:
        susp_events = all_types[:5]
    
    print(f"\nInjecting hijacks using {len(susp_events)} suspicious event types")
    
    injected_rows = []
    actual_hijacks = 0
    
    for i in range(min(n_hijacks, len(eligible))):
        user_id = eligible[i]
        user_df = df[df['user_id'] == user_id]
        
        t_min, t_max = user_df['timestamp'].min(), user_df['timestamp'].max()
        time_range = (t_max - t_min).total_seconds()
        
        if time_range < 3600:
            continue
        
        start_offset = np.random.uniform(0.2, 0.6) * time_range
        hijack_start = t_min + timedelta(seconds=start_offset)
        hijack_end = min(hijack_start + timedelta(hours=hijack_duration_hours), t_max)
        
        hijack_intervals.append({
            'user_id': user_id,
            'start': hijack_start,
            'end': hijack_end
        })
        actual_hijacks += 1
        
        # Failed logins before hijack
        for j in range(np.random.randint(3, 8)):
            failed_time = hijack_start - timedelta(minutes=np.random.randint(1, 30))
            login_types = [et for et in all_types if 'login' in et.lower()]
            login_type = login_types[0] if login_types else 'login_attempt'
            
            injected_rows.append({
                'user_id': user_id,
                'timestamp': failed_time,
                'event_type': login_type,
                'path': '/auth/login',
                'ip_address': f"10.{np.random.randint(0,255)}.{np.random.randint(0,255)}.{np.random.randint(0,255)}"
            })
        
        # Suspicious events during hijack
        n_susp = max(5, int(0.2 * len(user_df)))
        for j in range(n_susp):
            t_offset = np.random.uniform(0, (hijack_end - hijack_start).total_seconds())
            event_time = hijack_start + timedelta(seconds=t_offset)
            
            if np.random.random() < 0.5:
                event_time = event_time.replace(hour=np.random.randint(1, 5))
            
            injected_rows.append({
                'user_id': user_id,
                'timestamp': event_time,
                'event_type': np.random.choice(susp_events),
                'path': f"/suspicious/{np.random.choice(['data','admin','export'])}",
                'ip_address': f"10.{np.random.randint(0,255)}.{np.random.randint(0,255)}.{np.random.randint(0,255)}"
            })
        
        # Change IPs during hijack
        mask = ((df['user_id'] == user_id) & 
                (df['timestamp'] >= hijack_start) & 
                (df['timestamp'] <= hijack_end))
        df.loc[mask, 'ip_address'] = df.loc[mask].apply(
            lambda x: f"10.{np.random.randint(0,255)}.{np.random.randint(0,255)}.{np.random.randint(0,255)}", 
            axis=1
        )
    
    if injected_rows:
        inj_df = pd.DataFrame(injected_rows)
        inj_df['timestamp'] = pd.to_datetime(inj_df['timestamp'])
        df = pd.concat([df, inj_df], ignore_index=True)
        df = df.sort_values(['user_id', 'timestamp']).reset_index(drop=True)
    
    hijack_df = pd.DataFrame(hijack_intervals)
    print(f"  Hijacked {actual_hijacks} users with {len(injected_rows)} injected events")
    
    return df, hijack_df


# ============================================================
# WINDOW CONSTRUCTION
# ============================================================

def construct_windows(df: pd.DataFrame, window_size: int, 
                     step_size: int) -> List[Dict]:
    """Create sliding windows."""
    windows = []
    
    for user_id, user_df in df.groupby('user_id'):
        user_df = user_df.sort_values('timestamp')
        indices = user_df.index.tolist()
        
        if len(indices) < window_size:
            continue
        
        for start in range(0, len(indices) - window_size + 1, step_size):
            window_indices = indices[start:start + window_size]
            w_df = user_df.loc[window_indices]
            
            windows.append({
                'user_id': user_id,
                'start_time': w_df['timestamp'].iloc[0],
                'end_time': w_df['timestamp'].iloc[-1],
                'indices': window_indices
            })
    
    return windows


def label_windows(windows: List[Dict], hijack_df: pd.DataFrame) -> np.ndarray:
    """Label windows as hijacked or normal."""
    labels = []
    
    for w in windows:
        uid = w['user_id']
        ws, we = w['start_time'], w['end_time']
        
        intervals = hijack_df[hijack_df['user_id'] == uid]
        is_hijacked = 0
        
        for _, row in intervals.iterrows():
            if not (we < row['start'] or ws > row['end']):
                is_hijacked = 1
                break
        
        labels.append(is_hijacked)
    
    return np.array(labels)


# ============================================================
# USER HISTORY & TRANSITION MATRICES
# ============================================================

def build_user_history(df: pd.DataFrame) -> Dict:
    """Build lookup of user histories."""
    history = {}
    for user_id, user_df in df.groupby('user_id'):
        history[user_id] = user_df.sort_values('timestamp')
    return history


def build_transition_matrices(df: pd.DataFrame, all_event_types: np.ndarray) -> Dict:
    """Build transition probability matrices for each user."""
    event_to_idx = {et: i for i, et in enumerate(all_event_types)}
    n_events = len(all_event_types)
    
    user_transitions = {}
    
    for user_id, user_df in df.groupby('user_id'):
        user_df = user_df.sort_values('timestamp')
        events = user_df['event_type'].values
        
        trans_counts = np.zeros((n_events, n_events))
        
        for i in range(len(events) - 1):
            curr = events[i]
            nxt = events[i + 1]
            
            if curr in event_to_idx and nxt in event_to_idx:
                curr_idx = event_to_idx[curr]
                next_idx = event_to_idx[nxt]
                trans_counts[curr_idx, next_idx] += 1
        
        # Normalize with smoothing
        trans_probs = trans_counts + 1e-9
        row_sums = trans_probs.sum(axis=1, keepdims=True)
        trans_probs = trans_probs / row_sums
        
        user_transitions[user_id] = trans_probs
    
    return user_transitions


# ============================================================
# ORIGINAL HADITH FEATURES (26 features)
# ============================================================

def build_hadith_features(win: Dict, df: pd.DataFrame, 
                         user_history: Dict, 
                         all_event_types: np.ndarray) -> np.ndarray:
    """
    Build original 26 Hadith-inspired features.
    
    Axes:
    - ADALAH (5): Integrity/character stability
    - DABT (7): Precision/behavioral regularity  
    - ISNAD (6): Continuity/network chain
    - REPUTATION (4): Long-term reliability
    - ANOMALY (4): Deviation from baseline
    """
    idx = win['indices']
    w_df = df.loc[idx]
    uid = win['user_id']
    u_df = user_history[uid]
    past = u_df[u_df['timestamp'] < win['start_time']]
    
    features = []
    
    # === ADALAH (5) ===
    if past.empty:
        features.extend([0, 0, 0, 0, 0])
    else:
        active_days = past['timestamp'].dt.date.nunique()
        total_events = len(past)
        account_age = (win['end_time'] - past['timestamp'].min()).days + 1
        daily_counts = past.groupby(past['timestamp'].dt.date).size()
        consistency = daily_counts.std() if len(daily_counts) > 1 else 0
        avg_events = total_events / max(1, active_days)
        features.extend([active_days, total_events, account_age, consistency, avg_events])
    
    # === DABT (7) ===
    is_attempt = w_df['event_type'].str.contains('login|auth', case=False, na=False)
    is_success = w_df['event_type'].str.contains('success|ok|complete', case=False, na=False)
    n_attempt, n_success = int(is_attempt.sum()), int(is_success.sum())
    login_success_rate = n_success / (n_attempt + 1e-9)
    
    if past.empty:
        baseline_fail = 0
    else:
        p_att = int(past['event_type'].str.contains('login|auth', case=False, na=False).sum())
        p_suc = int(past['event_type'].str.contains('success|ok|complete', case=False, na=False).sum())
        baseline_fail = (p_att - p_suc) / (p_att + 1e-9)
    
    current_fail = (n_attempt - n_success) / (n_attempt + 1e-9)
    delta_fail = current_fail - baseline_fail
    
    timestamps = w_df['timestamp'].sort_values()
    minutes = (timestamps.astype('int64') // (60 * 1_000_000_000))
    burstiness = float(pd.Series(minutes.values).value_counts().max()) if len(minutes) > 0 else 0
    
    if past.empty:
        frac_outside = 0
    else:
        hist_hours = past['timestamp'].dt.hour.values
        if len(hist_hours) > 0:
            low, high = np.percentile(hist_hours, 2.5), np.percentile(hist_hours, 97.5)
            cur_hours = w_df['timestamp'].dt.hour.values
            outside = ((cur_hours < low) | (cur_hours > high)).sum()
            frac_outside = outside / len(cur_hours) if len(cur_hours) > 0 else 0
        else:
            frac_outside = 0
    
    if len(timestamps) > 1:
        deltas = timestamps.diff().dropna().dt.total_seconds().values
        deltas = deltas[deltas > 0]
        if len(deltas) > 0:
            bins = np.logspace(-1, 4, 20)
            hist, _ = np.histogram(deltas, bins=bins)
            hist = hist / hist.sum() if hist.sum() > 0 else hist
            timing_entropy = -np.sum(hist * np.log(hist + 1e-9))
        else:
            timing_entropy = 0
    else:
        timing_entropy = 0
    
    if past.empty or 'path' not in w_df.columns:
        vocab_div = 0
    else:
        cur_paths = set(w_df['path'].dropna().unique())
        hist_paths = set(past['path'].dropna().unique())
        new_paths = cur_paths - hist_paths
        vocab_div = len(new_paths) / (len(cur_paths) + 1e-9) if cur_paths else 0
    
    sensitive_kw = ['admin', 'delete', 'export', 'config', 'permission', 'sensitive']
    sens_mask = w_df['event_type'].str.lower().str.contains('|'.join(sensitive_kw), na=False)
    if 'path' in w_df.columns:
        sens_mask |= w_df['path'].str.lower().str.contains('|'.join(sensitive_kw), na=False)
    sensitive_ratio = sens_mask.sum() / len(w_df) if len(w_df) > 0 else 0
    
    features.extend([login_success_rate, delta_fail, burstiness, frac_outside,
                    timing_entropy, vocab_div, sensitive_ratio])
    
    # === ISNAD (6) ===
    unique_ips = w_df['ip_address'].nunique()
    ip_consistency = 1.0 / (unique_ips + 1e-9)
    
    if past.empty or past['ip_address'].isna().all() or (past['ip_address'] == '').all():
        ip_match, subnet_match = 1.0, 1.0
    else:
        hist_ip_counts = past['ip_address'].value_counts()
        primary_ip = hist_ip_counts.index[0] if len(hist_ip_counts) > 0 else ''
        ip_match = (w_df['ip_address'] == primary_ip).mean()
        
        def get_subnet(ip):
            if pd.isna(ip) or ip == '':
                return ''
            parts = str(ip).split('.')
            return '.'.join(parts[:3]) if len(parts) >= 3 else ip
        
        primary_subnet = get_subnet(primary_ip)
        subnet_match = (w_df['ip_address'].apply(get_subnet) == primary_subnet).mean()
    
    ips = w_df.loc[timestamps.index, 'ip_address']
    geo_impossible = 0
    if len(timestamps) > 1:
        for i in range(1, len(timestamps)):
            if ips.iloc[i] != ips.iloc[i-1]:
                time_diff = (timestamps.iloc[i] - timestamps.iloc[i-1]).total_seconds()
                if time_diff < 300:
                    geo_impossible += 1
        geo_impossible /= len(timestamps)
    
    if len(timestamps) > 1:
        deltas = timestamps.diff().dropna().dt.total_seconds()
        session_discont = (deltas > 3600).sum() / len(deltas) if len(deltas) > 0 else 0
    else:
        session_discont = 0
    
    if past.empty:
        new_ip_rate = 0
    else:
        hist_ips = set(past['ip_address'].dropna().unique())
        cur_ips = set(w_df['ip_address'].dropna().unique())
        new_ip_rate = len(cur_ips - hist_ips) / (len(cur_ips) + 1e-9)
    
    features.extend([ip_consistency, ip_match, subnet_match, geo_impossible,
                    session_discont, new_ip_rate])
    
    # === REPUTATION (4) ===
    if past.empty:
        features.extend([0, 0, 0, 0])
    else:
        duration = (win['end_time'] - past['timestamp'].min()).days + 1
        success_kw = ['success', 'complete', 'ok', 'approved', 'granted']
        fail_kw = ['fail', 'error', 'denied', 'reject', 'violation', 'invalid']
        succ_cnt = past['event_type'].str.lower().str.contains('|'.join(success_kw), na=False).sum()
        fail_cnt = past['event_type'].str.lower().str.contains('|'.join(fail_kw), na=False).sum()
        trust = succ_cnt / (succ_cnt + fail_cnt + 1e-9)
        penalty = fail_cnt / (len(past) + 1e-9)
        
        mid_time = past['timestamp'].min() + (win['end_time'] - past['timestamp'].min()) / 2
        older = past[past['timestamp'] < mid_time]
        recent = past[past['timestamp'] >= mid_time]
        if len(older) > 0 and len(recent) > 0:
            old_fail = older['event_type'].str.lower().str.contains('|'.join(fail_kw), na=False).mean()
            rec_fail = recent['event_type'].str.lower().str.contains('|'.join(fail_kw), na=False).mean()
            trend = old_fail - rec_fail
        else:
            trend = 0
        features.extend([duration, trust, penalty, trend])
    
    # === ANOMALY (4) ===
    evt_counts = w_df['event_type'].value_counts(normalize=True)
    evt_vec = evt_counts.reindex(all_event_types, fill_value=0).values
    
    if past.empty:
        features.extend([0, 0, 0, 0])
    else:
        past_counts = past['event_type'].value_counts(normalize=True)
        past_vec = past_counts.reindex(all_event_types, fill_value=0).values
        
        eps = 1e-9
        evt_smooth = (evt_vec + eps) / (evt_vec + eps).sum()
        past_smooth = (past_vec + eps) / (past_vec + eps).sum()
        kl_div = np.sum(evt_smooth * np.log(evt_smooth / past_smooth))
        
        hist_hours = past['timestamp'].dt.hour.value_counts(normalize=True)
        cur_hours = w_df['timestamp'].dt.hour.value_counts(normalize=True)
        all_hours = range(24)
        hist_h = (hist_hours.reindex(all_hours, fill_value=0).values + eps)
        cur_h = (cur_hours.reindex(all_hours, fill_value=0).values + eps)
        hist_h /= hist_h.sum()
        cur_h /= cur_h.sum()
        hour_anom = np.sum(cur_h * np.log(cur_h / hist_h))
        
        hist_paths = set(past['path'].dropna().unique())
        cur_paths = set(w_df['path'].dropna().unique())
        path_anom = len(cur_paths - hist_paths) / len(cur_paths) if cur_paths else 0
        
        l2_dist = np.linalg.norm(evt_vec - past_vec)
        
        features.extend([kl_div, hour_anom, path_anom, l2_dist])
    
    return np.array(features, dtype=float)


# ============================================================
# TEMPORAL SEQUENCE FEATURES (16 features) - NEW
# ============================================================

def build_temporal_sequence_features(win: Dict, df: pd.DataFrame,
                                    user_history: Dict,
                                    all_event_types: np.ndarray,
                                    user_transitions: Dict) -> np.ndarray:
    """
    Build 16 temporal sequence features mapped to Hadith axes.
    
    Axes:
    - DABT (5): Event ordering, sequence transitions
    - ADALAH (5): Long-distance dependencies, behavioral drift
    - ISNAD (3): IP/device sequence stability
    - REPUTATION (3): Risk accumulation, trend analysis
    """
    idx = win['indices']
    w_df = df.loc[idx]
    uid = win['user_id']
    u_df = user_history[uid]
    past = u_df[u_df['timestamp'] < win['start_time']]
    
    event_to_idx = {et: i for i, et in enumerate(all_event_types)}
    n_event_types = len(all_event_types)
    
    features = []
    
    # === DABT: Event Ordering (5) ===
    
    # 1. KL Transition
    if uid in user_transitions and len(w_df) > 1:
        hist_trans = user_transitions[uid]
        curr_trans = np.zeros((n_event_types, n_event_types))
        events = w_df['event_type'].values
        
        for i in range(len(events) - 1):
            if events[i] in event_to_idx and events[i+1] in event_to_idx:
                curr_idx = event_to_idx[events[i]]
                next_idx = event_to_idx[events[i+1]]
                curr_trans[curr_idx, next_idx] += 1
        
        curr_trans = curr_trans + 1e-9
        curr_trans = curr_trans / curr_trans.sum(axis=1, keepdims=True)
        
        hist_flat = hist_trans.flatten() + 1e-9
        curr_flat = curr_trans.flatten() + 1e-9
        hist_flat = hist_flat / hist_flat.sum()
        curr_flat = curr_flat / curr_flat.sum()
        
        kl_transition = np.sum(curr_flat * np.log(curr_flat / hist_flat))
    else:
        kl_transition = 0
    
    # 2. Rare Transitions
    if uid in user_transitions and len(w_df) > 1:
        hist_trans = user_transitions[uid]
        events = w_df['event_type'].values
        
        rare_count = 0
        for i in range(len(events) - 1):
            if events[i] in event_to_idx and events[i+1] in event_to_idx:
                curr_idx = event_to_idx[events[i]]
                next_idx = event_to_idx[events[i+1]]
                prob = hist_trans[curr_idx, next_idx]
                if prob < 0.01:
                    rare_count += 1
        
        rare_transition_rate = rare_count / (len(events) - 1)
    else:
        rare_transition_rate = 0
    
    # 3. Sequence Entropy
    if len(w_df) > 1:
        events = w_df['event_type'].values
        bigrams = [f"{events[i]}_{events[i+1]}" for i in range(len(events) - 1)]
        
        if bigrams:
            bigram_counts = Counter(bigrams)
            bigram_probs = np.array(list(bigram_counts.values())) / len(bigrams)
            seq_entropy = entropy(bigram_probs)
        else:
            seq_entropy = 0
    else:
        seq_entropy = 0
    
    # 4. N-gram Anomaly
    if not past.empty and len(w_df) > 1:
        past_events = past['event_type'].values
        hist_bigrams = [f"{past_events[i]}_{past_events[i+1]}" 
                       for i in range(len(past_events) - 1)]
        hist_bigram_counts = Counter(hist_bigrams)
        
        curr_events = w_df['event_type'].values
        curr_bigrams = [f"{curr_events[i]}_{curr_events[i+1]}" 
                       for i in range(len(curr_events) - 1)]
        
        if curr_bigrams:
            unseen = sum(1 for bg in curr_bigrams if bg not in hist_bigram_counts)
            ngram_anomaly = unseen / len(curr_bigrams)
        else:
            ngram_anomaly = 0
    else:
        ngram_anomaly = 0
    
    # 5. Run-length Anomaly
    if len(w_df) > 1:
        events = w_df['event_type'].values
        run_lengths = []
        current_run = 1
        
        for i in range(1, len(events)):
            if events[i] == events[i-1]:
                current_run += 1
            else:
                run_lengths.append(current_run)
                current_run = 1
        run_lengths.append(current_run)
        
        max_run = max(run_lengths) if run_lengths else 1
        avg_run = np.mean(run_lengths) if run_lengths else 1
        run_anomaly = max_run / avg_run if avg_run > 0 else 1
    else:
        run_anomaly = 0
    
    features.extend([kl_transition, rare_transition_rate, seq_entropy, 
                    ngram_anomaly, run_anomaly])
    
    # === ADALAH: Long Dependencies (5) ===
    
    # 6. Behavior Drift
    if not past.empty:
        now = win['end_time']
        week_ago = now - timedelta(days=7)
        month_ago = now - timedelta(days=30)
        
        week_data = past[past['timestamp'] >= week_ago]
        month_data = past[past['timestamp'] >= month_ago]
        
        if len(week_data) > 0 and len(month_data) > 0:
            week_dist = week_data['event_type'].value_counts(normalize=True)
            month_dist = month_data['event_type'].value_counts(normalize=True)
            
            week_vec = week_dist.reindex(all_event_types, fill_value=0).values
            month_vec = month_dist.reindex(all_event_types, fill_value=0).values
            
            behavior_drift = np.linalg.norm(week_vec - month_vec)
        else:
            behavior_drift = 0
    else:
        behavior_drift = 0
    
    # 7. Temporal Autocorrelation
    if not past.empty and len(past) > 7:
        daily_counts = past.groupby(past['timestamp'].dt.date).size()
        
        if len(daily_counts) > 1:
            values = daily_counts.values
            mean_val = values.mean()
            
            numerator = sum((values[i] - mean_val) * (values[i-1] - mean_val) 
                           for i in range(1, len(values)))
            denominator = sum((v - mean_val)**2 for v in values)
            
            autocorr = numerator / denominator if denominator > 0 else 0
        else:
            autocorr = 0
    else:
        autocorr = 0
    
    # 8. Long-term Timing Shift
    if not past.empty:
        hist_hours = past['timestamp'].dt.hour
        curr_hours = w_df['timestamp'].dt.hour
        
        hist_median = hist_hours.median()
        curr_median = curr_hours.median()
        
        timing_shift = abs(curr_median - hist_median) / 24.0
    else:
        timing_shift = 0
    
    # 9. Day-of-Week Divergence
    if not past.empty and len(past) > 7:
        hist_dow = past['timestamp'].dt.dayofweek.value_counts(normalize=True)
        curr_dow = w_df['timestamp'].dt.dayofweek.value_counts(normalize=True)
        
        hist_dow_vec = hist_dow.reindex(range(7), fill_value=0).values + 1e-9
        curr_dow_vec = curr_dow.reindex(range(7), fill_value=0).values + 1e-9
        
        hist_dow_vec = hist_dow_vec / hist_dow_vec.sum()
        curr_dow_vec = curr_dow_vec / curr_dow_vec.sum()
        
        dow_divergence = jensenshannon(hist_dow_vec, curr_dow_vec)
    else:
        dow_divergence = 0
    
    # 10. Event Rate Drift
    if not past.empty:
        hist_days = (past['timestamp'].max() - past['timestamp'].min()).days + 1
        hist_rate = len(past) / max(1, hist_days)
        
        curr_days = (win['end_time'] - win['start_time']).days + 1
        curr_rate = len(w_df) / max(1, curr_days)
        
        rate_drift = abs(curr_rate - hist_rate) / (hist_rate + 1e-9)
    else:
        rate_drift = 0
    
    features.extend([behavior_drift, autocorr, timing_shift, dow_divergence, rate_drift])
    
    # === ISNAD: IP/Device Sequences (3) ===
    
    # 11. Slow-change IP Subnet Drift
    if not past.empty:
        def get_subnet(ip):
            if pd.isna(ip) or ip == '':
                return ''
            parts = str(ip).split('.')
            return '.'.join(parts[:3]) if len(parts) >= 3 else ip
        
        hist_subnets = past['ip_address'].apply(get_subnet).value_counts(normalize=True)
        curr_subnets = w_df['ip_address'].apply(get_subnet).value_counts(normalize=True)
        
        all_subnets = set(hist_subnets.index) | set(curr_subnets.index)
        hist_vec = np.array([hist_subnets.get(s, 0) for s in all_subnets])
        curr_vec = np.array([curr_subnets.get(s, 0) for s in all_subnets])
        
        subnet_drift = np.linalg.norm(hist_vec - curr_vec)
    else:
        subnet_drift = 0
    
    # 12. IP Switch Sequence Stability
    if len(w_df) > 1:
        ips = w_df['ip_address'].values
        ip_switches = sum(1 for i in range(1, len(ips)) if ips[i] != ips[i-1])
        ip_switch_rate = ip_switches / (len(ips) - 1)
    else:
        ip_switch_rate = 0
    
    # 13. Device Transition Pattern Anomaly
    if not past.empty and len(past) > 1:
        past_ips = past['ip_address'].values
        past_transitions = [f"{past_ips[i]}_{past_ips[i+1]}" 
                           for i in range(len(past_ips) - 1)]
        
        if past_transitions:
            hist_trans_counts = Counter(past_transitions)
            
            curr_ips = w_df['ip_address'].values
            if len(curr_ips) > 1:
                curr_transitions = [f"{curr_ips[i]}_{curr_ips[i+1]}" 
                                   for i in range(len(curr_ips) - 1)]
                
                unseen_trans = sum(1 for t in curr_transitions 
                                  if t not in hist_trans_counts)
                device_trans_anomaly = unseen_trans / len(curr_transitions)
            else:
                device_trans_anomaly = 0
        else:
            device_trans_anomaly = 0
    else:
        device_trans_anomaly = 0
    
    features.extend([subnet_drift, ip_switch_rate, device_trans_anomaly])
    
    # === REPUTATION: Risk Accumulation (3) ===
    
    # 14. Failure Rate Trend Curvature
    if not past.empty:
        now = win['end_time']
        
        last_30d = past[past['timestamp'] >= now - timedelta(days=30)]
        last_7d = past[past['timestamp'] >= now - timedelta(days=7)]
        last_2d = past[past['timestamp'] >= now - timedelta(days=2)]
        
        def get_failure_rate(data):
            if len(data) == 0:
                return 0
            fail_kw = ['fail', 'error', 'denied', 'reject', 'invalid']
            failures = data['event_type'].str.lower().str.contains('|'.join(fail_kw), na=False).sum()
            return failures / len(data)
        
        r_30d = get_failure_rate(last_30d)
        r_7d = get_failure_rate(last_7d)
        r_2d = get_failure_rate(last_2d)
        
        delta_1 = r_7d - r_30d
        delta_2 = r_2d - r_7d
        
        trend_curvature = delta_2 - delta_1
    else:
        trend_curvature = 0
    
    # 15. Cumulative Suspicious Action Index
    if not past.empty:
        suspicious_kw = ['admin', 'delete', 'export', 'config', 'permission', 
                        'sensitive', 'bulk', 'share']
        
        past_with_time = past.copy()
        past_with_time['days_ago'] = (win['end_time'] - past_with_time['timestamp']).dt.total_seconds() / 86400
        
        past_with_time['is_suspicious'] = past_with_time['event_type'].str.lower().str.contains(
            '|'.join(suspicious_kw), na=False).astype(int)
        
        past_with_time['weight'] = np.exp(-past_with_time['days_ago'] / 30)
        
        cumulative_suspicious = (past_with_time['is_suspicious'] * past_with_time['weight']).sum()
        cumulative_suspicious = cumulative_suspicious / (len(past_with_time) + 1e-9)
    else:
        cumulative_suspicious = 0
    
    # 16. Risk Acceleration
    if not past.empty:
        now = win['end_time']
        recent = past[past['timestamp'] >= now - timedelta(days=7)]
        older = past[past['timestamp'] < now - timedelta(days=7)]
        
        suspicious_kw = ['admin', 'delete', 'export', 'config', 'permission']
        
        def get_susp_rate(data):
            if len(data) == 0:
                return 0
            susp = data['event_type'].str.lower().str.contains(
                '|'.join(suspicious_kw), na=False).sum()
            return susp / len(data)
        
        recent_rate = get_susp_rate(recent)
        older_rate = get_susp_rate(older)
        
        risk_accel = recent_rate - older_rate
    else:
        risk_accel = 0
    
    features.extend([trend_curvature, cumulative_suspicious, risk_accel])
    
    return np.array(features, dtype=float)


# ============================================================
# OTHER FEATURE TYPES
# ============================================================

def build_raw_count_features(win: Dict, df: pd.DataFrame,
                             all_event_types: np.ndarray) -> np.ndarray:
    """Raw event counts only."""
    idx = win['indices']
    w_df = df.loc[idx]
    
    evt_counts = w_df['event_type'].value_counts()
    count_vec = evt_counts.reindex(all_event_types, fill_value=0).values
    
    return count_vec.astype(float)


def build_minimal_features(win: Dict, df: pd.DataFrame) -> np.ndarray:
    """Minimal statistical features."""
    idx = win['indices']
    w_df = df.loc[idx]
    
    n_events = len(w_df)
    n_unique_events = w_df['event_type'].nunique()
    n_unique_ips = w_df['ip_address'].nunique()
    n_unique_paths = w_df['path'].nunique() if 'path' in w_df.columns else 0
    
    time_span = (w_df['timestamp'].max() - w_df['timestamp'].min()).total_seconds()
    
    hours = w_df['timestamp'].dt.hour
    mean_hour = hours.mean()
    std_hour = hours.std() if len(hours) > 1 else 0
    
    return np.array([n_events, n_unique_events, n_unique_ips, n_unique_paths,
                    time_span, mean_hour, std_hour], dtype=float)


# ============================================================
# LSTM CLASSIFIER
# ============================================================

if TORCH_AVAILABLE:
    class LSTMClassifier(nn.Module):
        """LSTM for sequence classification."""
        def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.3):
            super(LSTMClassifier, self).__init__()
            
            self.hidden_dim = hidden_dim
            self.num_layers = num_layers
            
            self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, 
                               batch_first=True, dropout=dropout if num_layers > 1 else 0)
            self.dropout = nn.Dropout(dropout)
            self.fc = nn.Linear(hidden_dim, 1)
            self.sigmoid = nn.Sigmoid()
        
        def forward(self, x):
            lstm_out, (h_n, c_n) = self.lstm(x)
            out = h_n[-1]
            out = self.dropout(out)
            out = self.fc(out)
            return self.sigmoid(out).squeeze()
    
    
    def train_lstm(X_train, y_train, X_val, y_val, 
                   input_dim, epochs=50, batch_size=32, lr=0.001):
        """Train LSTM classifier."""
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        X_train_t = torch.FloatTensor(X_train).unsqueeze(1).to(device)
        y_train_t = torch.FloatTensor(y_train).to(device)
        X_val_t = torch.FloatTensor(X_val).unsqueeze(1).to(device)
        y_val_t = torch.FloatTensor(y_val).to(device)
        
        train_dataset = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        model = LSTMClassifier(input_dim).to(device)
        
        pos_weight = torch.tensor([len(y_train) / (y_train.sum() + 1e-9)]).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0
        
        for epoch in range(epochs):
            model.train()
            train_loss = 0
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                
                outputs_logits = torch.log(outputs / (1 - outputs + 1e-9))
                loss = criterion(outputs_logits, batch_y)
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_t)
                val_outputs_logits = torch.log(val_outputs / (1 - val_outputs + 1e-9))
                val_loss = criterion(val_outputs_logits, y_val_t).item()
            
            if epoch % 10 == 0:
                print(f"  Epoch {epoch}/{epochs}, Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = model.state_dict()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"  Early stopping at epoch {epoch}")
                    break
        
        model.load_state_dict(best_model_state)
        
        return model
    
    
    def predict_lstm(model, X_test):
        """Get predictions from LSTM."""
        device = next(model.parameters()).device
        model.eval()
        
        X_test_t = torch.FloatTensor(X_test).unsqueeze(1).to(device)
        
        with torch.no_grad():
            outputs = model(X_test_t)
        
        return outputs.cpu().numpy()


# ============================================================
# EVALUATION
# ============================================================

def evaluate(y_true: np.ndarray, scores: np.ndarray, name: str) -> Dict:
    """Compute comprehensive metrics."""
    if scores.max() - scores.min() > 1e-9:
        scores_norm = (scores - scores.min()) / (scores.max() - scores.min())
    else:
        scores_norm = np.zeros_like(scores) + 0.5
    
    try:
        roc = roc_auc_score(y_true, scores_norm)
    except ValueError:
        roc = 0.5
    
    try:
        pr = average_precision_score(y_true, scores_norm)
    except ValueError:
        pr = y_true.mean()
    
    best_f1, best_thr = 0, 0.5
    for thr in np.linspace(0, 1, 101):
        y_pred = (scores_norm >= thr).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, thr
    
    y_pred = (scores_norm >= best_thr).astype(int)
    
    return {
        'Approach': name,
        'ROC-AUC': roc,
        'PR-AUC': pr,
        'F1': best_f1,
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'Accuracy': accuracy_score(y_true, y_pred),
        'Threshold': best_thr,
    }


def cross_validate(X, y, model_class, model_params, n_folds=5):
    """Stratified K-fold cross-validation."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    
    metrics = {'roc_auc': [], 'pr_auc': [], 'f1': []}
    
    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_val_s = scaler.transform(X_val)
        
        model = model_class(**model_params)
        model.fit(X_train_s, y_train)
        
        if hasattr(model, 'predict_proba'):
            scores = model.predict_proba(X_val_s)[:, 1]
        else:
            scores = model.decision_function(X_val_s)
            scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
        
        try:
            metrics['roc_auc'].append(roc_auc_score(y_val, scores))
        except ValueError:
            metrics['roc_auc'].append(0.5)
        
        try:
            metrics['pr_auc'].append(average_precision_score(y_val, scores))
        except ValueError:
            metrics['pr_auc'].append(y_val.mean())
        
        best_f1 = 0
        for thr in np.linspace(0, 1, 51):
            f1 = f1_score(y_val, (scores >= thr).astype(int), zero_division=0)
            best_f1 = max(best_f1, f1)
        metrics['f1'].append(best_f1)
    
    return {k: (np.mean(v), np.std(v)) for k, v in metrics.items()}


# ============================================================
# VISUALIZATION
# ============================================================

def save_result_plots(results_df, importances, feature_names, 
                      orig_axis_ranges, temp_groups, output_dir,
                      y_test, scores_combined, scores_hadith, 
                      scores_temporal, scores_raw):
    """Save comprehensive visualizations."""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. ROC curves
    fig, ax = plt.subplots(figsize=(10, 8))
    
    fpr_c, tpr_c, _ = roc_curve(y_test, scores_combined)
    fpr_h, tpr_h, _ = roc_curve(y_test, scores_hadith)
    fpr_t, tpr_t, _ = roc_curve(y_test, scores_temporal)
    fpr_r, tpr_r, _ = roc_curve(y_test, scores_raw)
    
    ax.plot(fpr_c, tpr_c, label=f'Combined (AUC={roc_auc_score(y_test, scores_combined):.3f})', 
            linewidth=2.5, color='#2E86AB')
    ax.plot(fpr_h, tpr_h, label=f'Original (AUC={roc_auc_score(y_test, scores_hadith):.3f})', 
            linewidth=2, color='#A23B72')
    ax.plot(fpr_t, tpr_t, label=f'Temporal (AUC={roc_auc_score(y_test, scores_temporal):.3f})', 
            linewidth=2, color='#F18F01')
    ax.plot(fpr_r, tpr_r, label=f'Raw (AUC={roc_auc_score(y_test, scores_raw):.3f})', 
            linewidth=1.5, color='#98C1D9', linestyle='--')
    ax.plot([0, 1], [0, 1], '--', color='gray', linewidth=1)
    
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves: Impact of Temporal Features', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'roc_curves.png'), dpi=150)
    print(f"  Saved: {output_dir}/roc_curves.png")
    plt.close()
    
    # 2. Feature importance
    fig, ax = plt.subplots(figsize=(12, 8))
    
    sorted_idx = np.argsort(importances)[::-1][:20]
    top_features = [feature_names[i] for i in sorted_idx]
    top_importances = importances[sorted_idx]
    
    colors = ['#F18F01' if 'temp_' in f else '#2E86AB' for f in top_features]
    
    display_names = [f.replace('temp_', 'T:').replace('adalah_', 'AD:').replace('dabt_', 'DB:')
                    .replace('isnad_', 'IS:').replace('rep_', 'RP:').replace('anom_', 'AN:')
                    for f in top_features]
    
    y_pos = np.arange(len(display_names))
    bars = ax.barh(y_pos, top_importances, color=colors)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(display_names, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel('Feature Importance', fontsize=12, fontweight='bold')
    ax.set_title('Top 20 Features', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    for i, (bar, val) in enumerate(zip(bars, top_importances)):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
               f'{val:.4f}', va='center', fontsize=9)
    
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2E86AB', label='Original Hadith'),
        Patch(facecolor='#F18F01', label='Temporal Sequence')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'top_features.png'), dpi=150)
    print(f"  Saved: {output_dir}/top_features.png")
    plt.close()


# ============================================================
# MAIN COMPARISON PIPELINE
# ============================================================

def main(args=None):
    """Main comparison function."""
    
    if args is None:
        if is_notebook():
            print("ERROR: Running in Jupyter without arguments.")
            return None
        else:
            args = parse_arguments()
    
    print("="*70)
    print("COMPLETE HADITH+TEMPORAL FEATURES ANALYSIS")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Max events:  {args.max_events:,}")
    print(f"  Max users:   {args.max_users}")
    print(f"  Window size: {args.window_size}")
    print(f"  N hijacks:   {args.n_hijacks}")
    
    # Load data
    df = load_dataset(args)
    if df is None:
        return None
    
    print_dataset_summary(df)
    
    # Inject hijacks
    df, hijack_df = inject_hijacks_clue(df, args.n_hijacks, args.hijack_duration)
    if len(hijack_df) == 0:
        print("ERROR: No hijacks injected")
        return None
    
    # Build windows
    print("\nConstructing windows...")
    windows = construct_windows(df, args.window_size, args.step_size)
    print(f"  Created {len(windows)} windows")
    
    # Label windows
    y = label_windows(windows, hijack_df)
    print(f"  Positive: {y.sum()}, Negative: {len(y)-y.sum()}")
    
    # Prepare features
    user_history = build_user_history(df)
    all_event_types = df['event_type'].unique()
    
    print("\nBuilding transition matrices...")
    user_transitions = build_transition_matrices(df, all_event_types)
    
    print("\nBuilding features...")
    
    # Original Hadith (26)
    print("  Original Hadith (26)...")
    X_hadith = []
    for i, w in enumerate(windows):
        if i % 500 == 0:
            print(f"    {i}/{len(windows)}...")
        X_hadith.append(build_hadith_features(w, df, user_history, all_event_types))
    X_hadith = np.array(X_hadith)
    
    # Temporal (16)
    print("  Temporal Sequence (16)...")
    X_temporal = []
    for i, w in enumerate(windows):
        if i % 500 == 0:
            print(f"    {i}/{len(windows)}...")
        X_temporal.append(build_temporal_sequence_features(w, df, user_history, 
                                                           all_event_types, user_transitions))
    X_temporal = np.array(X_temporal)
    
    # Combined (42)
    X_combined = np.hstack([X_hadith, X_temporal])
    
    # Raw & Minimal
    print("  Other features...")
    X_raw = np.array([build_raw_count_features(w, df, all_event_types) for w in windows])
    X_minimal = np.array([build_minimal_features(w, df) for w in windows])
    
    print(f"\nFeature dimensions:")
    print(f"  Original:  {X_hadith.shape}")
    print(f"  Temporal:  {X_temporal.shape}")
    print(f"  Combined:  {X_combined.shape}")
    print(f"  Raw:       {X_raw.shape}")
    print(f"  Minimal:   {X_minimal.shape}")
    
    # Handle NaN
    X_hadith = np.nan_to_num(X_hadith, nan=0, posinf=0, neginf=0)
    X_temporal = np.nan_to_num(X_temporal, nan=0, posinf=0, neginf=0)
    X_combined = np.nan_to_num(X_combined, nan=0, posinf=0, neginf=0)
    X_raw = np.nan_to_num(X_raw, nan=0, posinf=0, neginf=0)
    X_minimal = np.nan_to_num(X_minimal, nan=0, posinf=0, neginf=0)
    
    # Split
    (X_h_train, X_h_test, X_t_train, X_t_test, X_c_train, X_c_test,
     X_r_train, X_r_test, X_m_train, X_m_test, y_train, y_test) = train_test_split(
        X_hadith, X_temporal, X_combined, X_raw, X_minimal, y,
        test_size=0.3, stratify=y if y.sum() >= 2 else None, random_state=RANDOM_STATE
    )
    
    print(f"\nTrain: {len(y_train)} (Pos: {y_train.sum()})")
    print(f"Test:  {len(y_test)} (Pos: {y_test.sum()})")
    
    # Train models
    results = []
    
    print("\nTraining models...")
    
    # Scale
    scaler_h = StandardScaler()
    X_h_tr = scaler_h.fit_transform(X_h_train)
    X_h_te = scaler_h.transform(X_h_test)
    
    scaler_t = StandardScaler()
    X_t_tr = scaler_t.fit_transform(X_t_train)
    X_t_te = scaler_t.transform(X_t_test)
    
    scaler_c = StandardScaler()
    X_c_tr = scaler_c.fit_transform(X_c_train)
    X_c_te = scaler_c.transform(X_c_test)
    
    scaler_r = StandardScaler()
    X_r_tr = scaler_r.fit_transform(X_r_train)
    X_r_te = scaler_r.transform(X_r_test)
    
    scaler_m = StandardScaler()
    X_m_tr = scaler_m.fit_transform(X_m_train)
    X_m_te = scaler_m.transform(X_m_test)
    
    # 1. Combined + RF
    print("  1. Combined + RF")
    rf_combined = RandomForestClassifier(n_estimators=100, class_weight='balanced', 
                                         random_state=RANDOM_STATE, n_jobs=-1)
    rf_combined.fit(X_c_tr, y_train)
    scores_combined_rf = rf_combined.predict_proba(X_c_te)[:, 1]
    results.append(evaluate(y_test, scores_combined_rf, "Combined + RF"))
    
    # 2. Combined + GB
    print("  2. Combined + GB")
    gb_combined = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=RANDOM_STATE)
    gb_combined.fit(X_c_tr, y_train)
    scores_combined_gb = gb_combined.predict_proba(X_c_te)[:, 1]
    results.append(evaluate(y_test, scores_combined_gb, "Combined + GB"))
    
    # 3. Original + RF
    print("  3. Original Hadith + RF")
    rf_hadith = RandomForestClassifier(n_estimators=100, class_weight='balanced', 
                                        random_state=RANDOM_STATE, n_jobs=-1)
    rf_hadith.fit(X_h_tr, y_train)
    scores_hadith_rf = rf_hadith.predict_proba(X_h_te)[:, 1]
    results.append(evaluate(y_test, scores_hadith_rf, "Original Hadith + RF"))
    
    # 4. Original + GB
    print("  4. Original Hadith + GB")
    gb_hadith = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=RANDOM_STATE)
    gb_hadith.fit(X_h_tr, y_train)
    scores_hadith_gb = gb_hadith.predict_proba(X_h_te)[:, 1]
    results.append(evaluate(y_test, scores_hadith_gb, "Original Hadith + GB"))
    
    # 5. Temporal + RF
    print("  5. Temporal Only + RF")
    rf_temporal = RandomForestClassifier(n_estimators=100, class_weight='balanced',
                                          random_state=RANDOM_STATE, n_jobs=-1)
    rf_temporal.fit(X_t_tr, y_train)
    scores_temporal_rf = rf_temporal.predict_proba(X_t_te)[:, 1]
    results.append(evaluate(y_test, scores_temporal_rf, "Temporal Only + RF"))
    
    # 6. Temporal + GB
    print("  6. Temporal Only + GB")
    gb_temporal = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=RANDOM_STATE)
    gb_temporal.fit(X_t_tr, y_train)
    scores_temporal_gb = gb_temporal.predict_proba(X_t_te)[:, 1]
    results.append(evaluate(y_test, scores_temporal_gb, "Temporal Only + GB"))
    
    # 7. LSTM (if available)
    if TORCH_AVAILABLE:
        print("  7. Combined + LSTM")
        lstm_model = train_lstm(X_c_tr, y_train, X_c_te, y_test, 
                               input_dim=X_combined.shape[1], epochs=50)
        scores_lstm = predict_lstm(lstm_model, X_c_te)
        results.append(evaluate(y_test, scores_lstm, "Combined + LSTM"))
    
    # 8. Raw + RF
    print("  8. Raw Counts + RF")
    rf_raw = RandomForestClassifier(n_estimators=100, class_weight='balanced',
                                     random_state=RANDOM_STATE, n_jobs=-1)
    rf_raw.fit(X_r_tr, y_train)
    scores_raw_rf = rf_raw.predict_proba(X_r_te)[:, 1]
    results.append(evaluate(y_test, scores_raw_rf, "Raw Counts + RF"))
    
    # 9. Minimal + RF
    print("  9. Minimal + RF")
    rf_min = RandomForestClassifier(n_estimators=100, class_weight='balanced',
                                     random_state=RANDOM_STATE, n_jobs=-1)
    rf_min.fit(X_m_tr, y_train)
    scores_min_rf = rf_min.predict_proba(X_m_te)[:, 1]
    results.append(evaluate(y_test, scores_min_rf, "Minimal + RF"))
    
    # 10. Random
    print("  10. Random Baseline")
    scores_random = np.random.random(len(y_test))
    results.append(evaluate(y_test, scores_random, "Random"))
    
    # Results
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('ROC-AUC', ascending=False)
    
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(results_df.to_string(index=False))
    
    # Ablation Analysis
    print("\n" + "="*80)
    print("ABLATION ANALYSIS")
    print("="*80)
    
    combined_roc = results_df[results_df['Approach'] == 'Combined + RF']['ROC-AUC'].values[0]
    orig_roc = results_df[results_df['Approach'] == 'Original Hadith + RF']['ROC-AUC'].values[0]
    temporal_roc = results_df[results_df['Approach'] == 'Temporal Only + RF']['ROC-AUC'].values[0]
    raw_roc = results_df[results_df['Approach'] == 'Raw Counts + RF']['ROC-AUC'].values[0]
    
    print(f"\nROC-AUC Comparison:")
    print(f"  Combined:  {combined_roc:.4f}")
    print(f"  Original:  {orig_roc:.4f}  (Δ = {combined_roc - orig_roc:+.4f})")
    print(f"  Temporal:  {temporal_roc:.4f}  (Δ = {combined_roc - temporal_roc:+.4f})")
    print(f"  Raw:       {raw_roc:.4f}  (Δ = {combined_roc - raw_roc:+.4f})")
    
    if combined_roc > orig_roc:
        improvement = (combined_roc - orig_roc) / orig_roc * 100
        print(f"\n✓ Temporal features improve ROC-AUC by {improvement:.1f}%")
    
    # Feature Importance
    print("\n" + "="*80)
    print("FEATURE IMPORTANCE")
    print("="*80)
    
    feature_names = [
        "adalah_active_days", "adalah_total_events", "adalah_account_age",
        "adalah_consistency", "adalah_avg_events",
        "dabt_login_success", "dabt_delta_fail", "dabt_burstiness",
        "dabt_outside_hours", "dabt_timing_entropy", "dabt_vocab_div",
        "dabt_sensitive_ratio",
        "isnad_ip_consistency", "isnad_ip_match", "isnad_subnet_match",
        "isnad_geo_impossible", "isnad_session_discont", "isnad_new_ip_rate",
        "rep_duration", "rep_trust", "rep_penalty", "rep_trend",
        "anom_kl_div", "anom_hour", "anom_path", "anom_l2_dist",
        "temp_kl_transition", "temp_rare_transitions", "temp_seq_entropy",
        "temp_ngram_anomaly", "temp_run_anomaly",
        "temp_behavior_drift", "temp_autocorr", "temp_timing_shift",
        "temp_dow_divergence", "temp_rate_drift",
        "temp_subnet_drift", "temp_ip_switch_rate", "temp_device_trans_anom",
        "temp_failure_trend", "temp_cumul_suspicious", "temp_risk_accel",
    ]
    
    print("\nTop 15 Features:")
    importances = rf_combined.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    
    for i in range(min(15, len(feature_names))):
        idx = sorted_idx[i]
        marker = "🔄" if "temp_" in feature_names[idx] else "📊"
        print(f"  {i+1:2d}. {marker} {feature_names[idx]:30s} {importances[idx]:.4f}")
    
    orig_total = importances[:26].sum()
    temp_total = importances[26:42].sum()
    
    print(f"\nTotal Importance:")
    print(f"  Original: {orig_total:.4f} ({orig_total/importances.sum()*100:.1f}%)")
    print(f"  Temporal: {temp_total:.4f} ({temp_total/importances.sum()*100:.1f}%)")
    
    # Cross-validation
    print("\n" + "="*80)
    print("CROSS-VALIDATION")
    print("="*80)
    
    print("\nCombined + RF:")
    cv_combined = cross_validate(X_combined, y, RandomForestClassifier,
                                 {'n_estimators': 100, 'class_weight': 'balanced',
                                  'random_state': RANDOM_STATE, 'n_jobs': -1})
    for m, (mean, std) in cv_combined.items():
        print(f"  {m}: {mean:.4f} ± {std:.4f}")
    
    print("\nOriginal + RF:")
    cv_hadith = cross_validate(X_hadith, y, RandomForestClassifier,
                               {'n_estimators': 100, 'class_weight': 'balanced',
                                'random_state': RANDOM_STATE, 'n_jobs': -1})
    for m, (mean, std) in cv_hadith.items():
        print(f"  {m}: {mean:.4f} ± {std:.4f}")
    
    print("\nTemporal + RF:")
    cv_temporal = cross_validate(X_temporal, y, RandomForestClassifier,
                                {'n_estimators': 100, 'class_weight': 'balanced',
                                 'random_state': RANDOM_STATE, 'n_jobs': -1})
    for m, (mean, std) in cv_temporal.items():
        print(f"  {m}: {mean:.4f} ± {std:.4f}")
    
    # Plots
    if args.save_plots:
        print("\nSaving plots...")
        orig_axis_ranges = {
            "Adalah": (0, 5), "Dabt": (5, 12), "Isnad": (12, 18),
            "Reputation": (18, 22), "Anomaly": (22, 26)
        }
        temp_groups = {
            "Dabt": (26, 31), "Adalah": (31, 36),
            "Isnad": (36, 39), "Reputation": (39, 42)
        }
        save_result_plots(results_df, importances, feature_names,
                         orig_axis_ranges, temp_groups, args.output_dir,
                         y_test, scores_combined_rf, scores_hadith_rf,
                         scores_temporal_rf, scores_raw_rf)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    
    return {
        'results_df': results_df,
        'cv_combined': cv_combined,
        'cv_hadith': cv_hadith,
        'cv_temporal': cv_temporal,
        'feature_importances': dict(zip(feature_names, importances)),
        'temporal_contribution': temp_total / importances.sum() if temp_total > 0 else 0,
    }


def run_comparison(args=None, **kwargs):
    """Wrapper for easy calling."""
    if args is None:
        args = Args(**kwargs)
    elif kwargs:
        for key, value in kwargs.items():
            if hasattr(args, key):
                setattr(args, key, value)
    return main(args)


if __name__ == "__main__":
    if not is_notebook():
        args = parse_arguments()
        results = main(args)
    else:
        print("Running in Jupyter. Use: run_comparison(csv='data.csv')")

    #from compare_features_temporal_FULL import run_comparison

   
    results = run_comparison(
        csv='data\clue_lds_logs.csv', #Change to your data path 
        max_events=500000,
        max_users=100,
        n_hijacks=30,
        window_size=50,
        step_size=25,
        save_plots=True
        )

    # Access results
    print(results['results_df'])  # Performance comparison
    print(f"Temporal contribution: {results['temporal_contribution']*100:.1f}%")