"""
Schema Migration: Add 'source' column to existing clean data.
Run this ONCE after data_cleanup.py and BEFORE running main.py.
"""
import os
import sys
import pandas as pd

if sys.platform == 'win32' and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "data", "processed")

# --- Migrate simulations_master.csv ---
master_path = os.path.join(PROCESSED_DIR, "simulations_master.csv")
if os.path.exists(master_path):
    df = pd.read_csv(master_path)
    if 'source' not in df.columns:
        df['source'] = 'interactive'  # All retained rows are from interactive sessions
        df.to_csv(master_path, index=False)
        print(f"[MIGRATED] simulations_master.csv: added 'source' column ({len(df)} rows)")
    else:
        print(f"[OK] simulations_master.csv already has 'source' column")

# --- Migrate simulation_metrics.csv ---  
metrics_path = os.path.join(PROCESSED_DIR, "simulation_metrics.csv")
if os.path.exists(metrics_path):
    df = pd.read_csv(metrics_path)
    if 'source' not in df.columns:
        df['source'] = 'interactive'
        df.to_csv(metrics_path, index=False)
        print(f"[MIGRATED] simulation_metrics.csv: added 'source' column ({len(df)} rows)")
    else:
        print(f"[OK] simulation_metrics.csv already has 'source' column")

# --- Also clean up the 'Error' and non-standard outcome rows ---
if os.path.exists(master_path):
    df = pd.read_csv(master_path)
    
    # Remove rows with non-standard outcomes that would confuse the ML model
    non_standard = df[~df['outcome'].isin(['Success', 'Failure', 'Pending'])]
    if len(non_standard) > 0:
        print(f"\n[INFO] Found {len(non_standard)} rows with non-standard outcomes:")
        for _, row in non_standard.iterrows():
            print(f"  - {row['target_url']}: outcome='{row['outcome']}', score={row['score']}")
        
        # Remap: 'Partial Success' -> 'Success', 'Ongoing' -> 'Pending', 'Error' -> remove
        remap = {'Partial Success': 'Success', 'Ongoing': 'Pending'}
        df_clean = df[df['outcome'] != 'Error'].copy()
        df_clean['outcome'] = df_clean['outcome'].replace(remap)
        
        removed = len(df) - len(df_clean)
        remapped = sum(df_clean['outcome'] != df.loc[df_clean.index, 'outcome']) if len(df_clean) > 0 else 0
        
        df_clean.to_csv(master_path, index=False)
        print(f"\n[CLEANED] Removed {removed} Error row(s), remapped non-standard outcomes")
        print(f"[FINAL] {len(df_clean)} rows in simulations_master.csv")
        print(f"  Outcomes: {df_clean['outcome'].value_counts().to_dict()}")

# --- Same cleanup for metrics ---
if os.path.exists(metrics_path):
    df_m = pd.read_csv(metrics_path)
    master_ids = set(pd.read_csv(master_path)['simulation_id'].tolist())
    df_m_clean = df_m[df_m['simulation_id'].isin(master_ids)].copy()
    
    # Remap outcome_label
    remap_m = {'Partial Success': 'Success', 'Ongoing': 'Pending'}
    df_m_clean['outcome_label'] = df_m_clean['outcome_label'].replace(remap_m)
    df_m_clean['outcome_binary'] = df_m_clean['outcome_label'].apply(lambda x: 1 if 'Success' in str(x) else 0)
    
    df_m_clean.to_csv(metrics_path, index=False)
    print(f"[FINAL] {len(df_m_clean)} rows in simulation_metrics.csv")
    print(f"  Outcomes: {df_m_clean['outcome_label'].value_counts().to_dict()}")

print("\n[DONE] Schema migration complete. Ready to run 'python main.py'.")
