"""
Data Cleanup Script for DeepMost Agentic SDR
=============================================

This script:
1. Archives all existing corrupted batch data
2. Preserves only high-quality rows (interactive simulations with real LLM analysis)
3. Cleans up the corresponding JSON conversation files
4. Provides a summary of what was kept vs archived

Run this ONCE before generating fresh batch data with the fixed pipeline.
"""

import os
import sys
import shutil
import pandas as pd
from datetime import datetime

# Handle Unicode output for Windows
if sys.platform == 'win32' and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# Paths
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
RAW_DIR = os.path.join(DATA_DIR, "raw")
CONVERSATIONS_DIR = os.path.join(RAW_DIR, "conversations")
ARCHIVE_DIR = os.path.join(DATA_DIR, "archived", f"pre_cleanup_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

# Files to clean
MASTER_CSV = os.path.join(PROCESSED_DIR, "simulations_master.csv")
METRICS_CSV = os.path.join(PROCESSED_DIR, "simulation_metrics.csv")
TURNS_CSV = os.path.join(PROCESSED_DIR, "conversation_turns.csv")
SALES_CSV = os.path.join(PROCESSED_DIR, "sales_dataset.csv")


def identify_bad_rows(df, source_name):
    """
    Identify rows that have corrupted/placeholder data from the old batch pipeline.
    
    Bad rows have ALL of these characteristics:
    - score == 3 (hardcoded)
    - outcome == 'Pending' (default fallback)  
    - key_objection == 'Unknown' (hardcoded)
    - feedback contains 'Auto-generated from batch pipeline'
    """
    if df.empty:
        return pd.Series(dtype=bool), pd.Series(dtype=bool)
    
    # For master CSV
    if 'feedback' in df.columns and 'outcome' in df.columns:
        is_bad = (
            (df['score'].astype(str) == '3') &
            (df['outcome'].str.strip() == 'Pending') &
            (df['key_objection'].str.strip() == 'Unknown') &
            (df['feedback'].str.contains('Auto-generated from batch pipeline', na=False))
        )
        is_good = ~is_bad
        return is_good, is_bad
    
    # For metrics CSV
    if 'outcome_label' in df.columns and 'objection_type' in df.columns:
        is_bad = (
            (df['score'].astype(str) == '3') &
            (df['outcome_label'].str.strip() == 'Pending') &
            (df['objection_type'].str.strip() == 'Unknown')
        )
        is_good = ~is_bad
        return is_good, is_bad
    
    return pd.Series([True] * len(df)), pd.Series([False] * len(df))


def cleanup():
    print("=" * 60)
    print("  DeepMost Agentic SDR — Data Cleanup")
    print("=" * 60)
    print()
    
    # --- Step 1: Create archive directory ---
    os.makedirs(ARCHIVE_DIR, exist_ok=True)
    os.makedirs(os.path.join(ARCHIVE_DIR, "conversations"), exist_ok=True)
    print(f"[1/5] Archive directory: {ARCHIVE_DIR}")
    
    # --- Step 2: Archive original files ---
    print(f"\n[2/5] Archiving original files...")
    for csv_file in [MASTER_CSV, METRICS_CSV, TURNS_CSV, SALES_CSV]:
        if os.path.exists(csv_file):
            dest = os.path.join(ARCHIVE_DIR, os.path.basename(csv_file))
            shutil.copy2(csv_file, dest)
            print(f"   Archived: {os.path.basename(csv_file)}")
    
    # --- Step 3: Clean Master CSV ---
    print(f"\n[3/5] Cleaning simulations_master.csv...")
    if os.path.exists(MASTER_CSV):
        df_master = pd.read_csv(MASTER_CSV, engine='python', on_bad_lines='warn')
        total_master = len(df_master)
        
        good_mask, bad_mask = identify_bad_rows(df_master, "master")
        df_good = df_master[good_mask].copy()
        df_bad = df_master[bad_mask].copy()
        
        # Archive bad rows separately for reference
        if len(df_bad) > 0:
            df_bad.to_csv(os.path.join(ARCHIVE_DIR, "removed_master_rows.csv"), index=False)
        
        # Get IDs of bad rows to clean up their JSON files
        bad_sim_ids = set(df_bad['simulation_id'].tolist()) if 'simulation_id' in df_bad.columns else set()
        good_sim_ids = set(df_good['simulation_id'].tolist()) if 'simulation_id' in df_good.columns else set()
        
        # Write clean data back
        df_good.to_csv(MASTER_CSV, index=False)
        
        print(f"   Total rows:    {total_master}")
        print(f"   Kept (good):   {len(df_good)}")
        print(f"   Removed (bad): {len(df_bad)}")
        
        if len(df_good) > 0:
            print(f"\n   --- Kept Rows Summary ---")
            if 'outcome' in df_good.columns:
                print(f"   Outcomes: {df_good['outcome'].value_counts().to_dict()}")
            if 'score' in df_good.columns:
                print(f"   Score range: {df_good['score'].min()} - {df_good['score'].max()} (avg: {df_good['score'].mean():.1f})")
            if 'key_objection' in df_good.columns:
                print(f"   Objections: {df_good['key_objection'].value_counts().to_dict()}")
    else:
        bad_sim_ids = set()
        good_sim_ids = set()
        print("   File not found, skipping.")
    
    # --- Step 4: Clean Metrics CSV ---
    print(f"\n[4/5] Cleaning simulation_metrics.csv...")
    if os.path.exists(METRICS_CSV):
        df_metrics = pd.read_csv(METRICS_CSV, engine='python', on_bad_lines='warn')
        total_metrics = len(df_metrics)
        
        # Filter by the same simulation IDs we kept
        if good_sim_ids:
            df_metrics_clean = df_metrics[df_metrics['simulation_id'].isin(good_sim_ids)].copy()
        else:
            good_mask_m, bad_mask_m = identify_bad_rows(df_metrics, "metrics")
            df_metrics_clean = df_metrics[good_mask_m].copy()
        
        df_metrics_clean.to_csv(METRICS_CSV, index=False)
        print(f"   Total rows:    {total_metrics}")
        print(f"   Kept (good):   {len(df_metrics_clean)}")
        print(f"   Removed (bad): {total_metrics - len(df_metrics_clean)}")
    else:
        print("   File not found, skipping.")
    
    # Clean Turns CSV
    if os.path.exists(TURNS_CSV):
        df_turns = pd.read_csv(TURNS_CSV, engine='python', on_bad_lines='warn')
        total_turns = len(df_turns)
        if good_sim_ids:
            df_turns_clean = df_turns[df_turns['simulation_id'].isin(good_sim_ids)].copy()
        else:
            df_turns_clean = df_turns.copy()
        df_turns_clean.to_csv(TURNS_CSV, index=False)
        print(f"\n   Turns CSV: {total_turns} -> {len(df_turns_clean)} rows")
    
    # --- Step 5: Archive bad conversation JSONs ---
    print(f"\n[5/5] Archiving removed conversation JSONs...")
    archived_json_count = 0
    if os.path.exists(CONVERSATIONS_DIR) and bad_sim_ids:
        for sim_id in bad_sim_ids:
            json_file = os.path.join(CONVERSATIONS_DIR, f"{sim_id}.json")
            if os.path.exists(json_file):
                dest = os.path.join(ARCHIVE_DIR, "conversations", f"{sim_id}.json")
                shutil.move(json_file, dest)
                archived_json_count += 1
    print(f"   Moved {archived_json_count} JSON files to archive")
    
    # --- Summary ---
    print("\n" + "=" * 60)
    print("  CLEANUP COMPLETE")
    print("=" * 60)
    print(f"\n  Archive location: {ARCHIVE_DIR}")
    print(f"  Good data points retained: {len(df_good) if 'df_good' in dir() else 0}")
    print(f"  Corrupted rows removed: {len(df_bad) if 'df_bad' in dir() else 0}")
    print(f"  JSON files archived: {archived_json_count}")
    print(f"\n  Next step: Run 'python main.py' to generate fresh, high-quality batch data.")
    print(f"  The new data will have real LLM-scored outcomes, scores, and objections.")
    print()


if __name__ == "__main__":
    cleanup()
