"""
Batch Processing Pipeline for DeepMost Agentic SDR
====================================================

Runs the sales simulation on multiple target sites and saves
all data for analysis. Works with any LLM provider (Groq/Gemini)
configured via LLM_PROVIDER in .env
"""

import os
import sys
import pandas as pd
import time
from src.scraper import simple_scraper
from src.agent_logic import generate_synthetic_call, analyze_call, get_provider_info
from src.data_manager import data_manager

# Handle Unicode output for Windows
if sys.platform == 'win32' and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# Ensure data directory exists
os.makedirs("data/processed", exist_ok=True)

# --- CONFIGURATION ---
TARGET_SITES = [
"https://www.zscaler.com", "https://www.akamai.com", "https://www.fortinet.com", "https://www.wrike.com", "https://squareup.com", "https://www.adp.com", "https://www.braze.com", "https://amplitude.com", "https://www.squarespace.com", "https://www.paychex.com"
]

# Dynamic inter-site delay based on provider
_provider_info = get_provider_info()
INTER_SITE_DELAY = 3 if _provider_info["provider"] == "groq" else 10
# ---------------------


def run_pipeline():
    """
    Run the batch processing pipeline on all target sites.
    Saves data using the data manager for comprehensive analysis.
    """
    dataset = []
    info = get_provider_info()
    print(f"[START] Agentic Pipeline | Provider: {info['provider'].upper()} | Model: {info['model']}", flush=True)
    print(f"[START] Processing {len(TARGET_SITES)} targets...", flush=True)

    for idx, site in enumerate(TARGET_SITES, 1):
        print(f"\n[{idx}/{len(TARGET_SITES)}] Processing: {site}", flush=True)
        
        # Step 1: Scrape
        context = simple_scraper(site)
        
        if context:
            # Step 2: Generate Conversation
            print(f"   ...Simulating Agents ({info['provider'].upper()})", flush=True)
            dialogue = generate_synthetic_call(context)
            
            if dialogue:
                # Step 3: Analyze
                sentiment, outcome = analyze_call(dialogue)
                
                # Parse dialogue into conversation history format
                conversation_history = parse_dialogue(dialogue)
                
                # Create analysis result in expected format
                analysis_result = f"""Score: {5 if outcome == 'Success' else 3}
Outcome: {outcome}
Key_Objection: Unknown
Feedback: Auto-generated from batch pipeline. Sentiment: {sentiment}"""
                
                # Save using data manager
                try:
                    sim_id = data_manager.save_simulation(
                        target_url=site,
                        company_context=context,
                        conversation_history=conversation_history,
                        analysis_result=analysis_result
                    )
                    print(f"   [SAVED] ID: {sim_id[:8]}...", flush=True)
                except Exception as e:
                    print(f"   [ERROR] Save failed: {e}", flush=True)
                
                dataset.append({
                    "Target_URL": site,
                    "Generated_Dialogue": dialogue,
                    "Sentiment": sentiment,
                    "Outcome": outcome
                })
                print(f"   [SUCCESS] {outcome} ({sentiment})", flush=True)
                
                # Rate Limit Safety
                if idx < len(TARGET_SITES):
                    print(f"   [WAIT] Pausing {INTER_SITE_DELAY}s before next site...", flush=True)
                    time.sleep(INTER_SITE_DELAY)
            else:
                print("   [FAILED] Failed to generate dialogue.", flush=True)
        else:
            print("   [FAILED] Failed to get context.", flush=True)

    # Save Results
    if dataset:
        df = pd.DataFrame(dataset)
        output_path = "data/processed/sales_dataset.csv"
        df.to_csv(output_path, index=False)
        print(f"\n[COMPLETE] Pipeline Complete! Data saved to: {output_path}", flush=True)
        print(df[['Target_URL', 'Outcome', 'Sentiment']].head())
        
        # Print summary
        stats = data_manager.get_summary_stats()
        print(f"\n--- Data Summary ---")
        print(f"Total Simulations: {stats.get('total_simulations', 0)}")
        print(f"Success Rate: {stats.get('success_rate', 0):.1f}%")
        print(f"Average Score: {stats.get('avg_score', 0):.2f}")
    else:
        print("\n[WARNING] No data generated. Check your API key or internet connection.", flush=True)


def parse_dialogue(dialogue: str) -> list:
    """
    Parse a dialogue string into a list of (speaker, message) tuples.
    Handles both 'SELLER:' and 'BUYER:' prefixed lines.
    """
    history = []
    lines = dialogue.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        if line.upper().startswith('SELLER:'):
            message = line.split(':', 1)[1].strip() if ':' in line else line
            history.append(('Seller', message))
        elif line.upper().startswith('BUYER:'):
            message = line.split(':', 1)[1].strip() if ':' in line else line
            history.append(('Buyer', message))
        elif line.upper().startswith('SDR:'):
            message = line.split(':', 1)[1].strip() if ':' in line else line
            history.append(('Seller', message))
        elif line.upper().startswith('CTO:') or line.upper().startswith('PROSPECT:'):
            message = line.split(':', 1)[1].strip() if ':' in line else line
            history.append(('Buyer', message))
    
    return history


if __name__ == "__main__":
    run_pipeline()
