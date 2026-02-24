"""
Dynamic Batch Processing Pipeline for DeepMost Agentic SDR
==========================================================

Fetches a dynamic number of new company websites using the LLM,
ensures NO DUPLICATES against the master database,
prints out the list of targeted websites,
and then runs the batch processing sales simulation.
"""

import os
import sys
import argparse
import pandas as pd
import time
import json
import re
import random

from src.scraper import simple_scraper
from src.agent_logic import generate_synthetic_call, analyze_call, get_provider_info, _generate_content
from src.data_manager import data_manager
from main import parse_dialogue

# Handle Unicode output for Windows
if sys.platform == 'win32' and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# Ensure data directory exists
os.makedirs("data/processed", exist_ok=True)

# Dynamic inter-site delay based on provider
_provider_info = get_provider_info()
INTER_SITE_DELAY = 3 if _provider_info["provider"] == "groq" else 10

def fetch_new_urls(count: int, existing_urls: set) -> list:
    """Uses the LLM to generate `count` new unique company URLs."""
    
    # Pass the existing URLs to give the LLM context of what to avoid.
    avoid_list = list(existing_urls)
    avoid_list_str = "\n- ".join(avoid_list)
    
    prompt = f"""Act as a B2B lead generation expert.
Generate a list of exactly {count + 10} unique URLs for relatively small, rising, or lesser-known B2B technology, SaaS, cloud, enterprise software, or industrial companies. Do NOT output giant tech conglomerates.
Make sure they are real companies with active websites, and include a diverse mix of large and mid-size companies.

Provide ONLY a valid JSON array of strings containing the URLs, and nothing else. No markdown formatting, no explanations!
Ensure all URLs start with 'https://www.'.

CRITICAL: Do NOT include any of the following URLs. I will fail the task if any of these are outputted:
- {avoid_list_str}
"""
    
    response = _generate_content(prompt)
    
    urls = []
    try:
        cleaned = response.replace('```json', '').replace('```', '').strip()
        parsed = json.loads(cleaned)
        if isinstance(parsed, list):
            urls = [str(u).strip() for u in parsed if str(u).startswith("http")]
    except Exception as e:
        print(f"   [WARNING] JSON parsing failed, using regex fallback...")
        pass
    
    if not urls:
        # Regex fallback to find URLs
        urls = re.findall(r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+', response)
    
    # Filter and deduplicate
    final_urls = []
    for u in urls:
        u_clean = u.lower().rstrip('/')
        # Check against existing (also stripped and lowered)
        if u_clean not in [ex.lower().rstrip('/') for ex in existing_urls]:
            if u_clean not in [f.lower().rstrip('/') for f in final_urls]:
                final_urls.append(u)
                
    return final_urls[:count]


def run_dynamic_pipeline(count: int):
    """
    Run the dynamic batch processing pipeline.
    """
    dataset = []
    info = get_provider_info()
    
    # Check for already processed URLs to avoid duplicates
    existing_urls = set()
    master_csv_path = "data/processed/simulations_master.csv"
    if os.path.exists(master_csv_path):
        try:
            df_existing = pd.read_csv(master_csv_path)
            if 'target_url' in df_existing.columns:
                existing_urls = set(df_existing['target_url'].dropna().tolist())
        except Exception as e:
            print(f"[WARNING] Could not read existing URLs: {e}")
            
    print(f"\n[INIT] Found {len(existing_urls)} existing URLs in master database.", flush=True)
    print(f"[FETCHING] Requesting {count} new websites from LLM...", flush=True)
    
    # We might need to call generating a few times if we don't get enough URLs
    new_urls = fetch_new_urls(count, existing_urls)
    
    if len(new_urls) < count:
        print(f"   [NOTE] Only got {len(new_urls)} valid new URLs from the first request. Trying one more time...", flush=True)
        additional_urls = fetch_new_urls(count - len(new_urls), existing_urls.union(set(new_urls)))
        new_urls.extend(additional_urls)
        
    if not new_urls:
        print("[ERROR] Failed to fetch valid URLs from LLM. Exiting.")
        return
        
    print(f"\n{'='*50}")
    print(f"[TARGETS] Acquired {len(new_urls)} new websites to process:")
    for idx, u in enumerate(new_urls, 1):
        print(f"  {idx}. {u}")
    print(f"{'='*50}\n", flush=True)
    
    print(f"[START] Agentic Pipeline | Provider: {info['provider'].upper()} | Model: {info['model']}", flush=True)

    for idx, site in enumerate(new_urls, 1):
        print(f"\n[{idx}/{len(new_urls)}] Processing: {site}", flush=True)
        
        # Step 1: Scrape
        context = simple_scraper(site)
        
        if context:
            # Step 2: Generate Conversation
            print(f"   ...Simulating Agents ({info['provider'].upper()})", flush=True)
            dialogue = generate_synthetic_call(context)
            
            if dialogue:
                # Step 3: Analyze
                sentiment, outcome, score, key_objection, feedback = analyze_call(dialogue)
                
                # Parse dialogue into conversation history format
                conversation_history = parse_dialogue(dialogue)
                
                # Create analysis result from LLM analysis
                analysis_result = f"Score: {score}\nOutcome: {outcome}\nKey_Objection: {key_objection}\nFeedback: {feedback} Sentiment: {sentiment}"
                
                # Save using data manager
                try:
                    sim_id = data_manager.save_simulation(
                        target_url=site,
                        company_context=context,
                        conversation_history=conversation_history,
                        analysis_result=analysis_result,
                        source="dynamic_batch"
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
                if idx < len(new_urls):
                    print(f"   [WAIT] Pausing {INTER_SITE_DELAY}s before next site...", flush=True)
                    time.sleep(INTER_SITE_DELAY)
            else:
                print("   [FAILED] Failed to generate dialogue.", flush=True)
        else:
            print("   [FAILED] Failed to get context.", flush=True)

    # Save Results
    if dataset:
        df = pd.DataFrame(dataset)
        output_path = "data/processed/dynamic_sales_dataset.csv"
        df.to_csv(output_path, index=False)
        print(f"\n[COMPLETE] Pipeline Complete! Batch data saved to: {output_path}", flush=True)
        
        # Print summary
        stats = data_manager.get_summary_stats()
        print(f"\n--- Master Data Summary ---")
        print(f"Total Simulations: {stats.get('total_simulations', 0)}")
        print(f"Success Rate: {stats.get('success_rate', 0):.1f}%")
        print(f"Average Score: {stats.get('avg_score', 0):.2f}")
    else:
        print("\n[WARNING] No data generated. Check your API key or internet connection.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch new websites and run SDR simulation batch processing.")
    parser.add_argument("--count", type=int, default=10, help="Number of new websites to fetch and process")
    args = parser.parse_args()
    
    run_dynamic_pipeline(args.count)
