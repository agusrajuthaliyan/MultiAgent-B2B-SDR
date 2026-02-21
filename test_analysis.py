"""Quick test: run 5 dialogues and check the outcome distribution."""
import sys
if sys.platform == 'win32' and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

from dotenv import load_dotenv
load_dotenv()

from src.agent_logic import generate_synthetic_call, analyze_call, get_provider_info
import time

info = get_provider_info()
print(f"Provider: {info['provider']} | Model: {info['model']}")
print("=" * 60)

context = "TechFlow Inc is a SaaS company with 150 employees building project management tools. They process large amounts of user analytics data using outdated ETL pipelines. Annual data spend is $80K."

results = []
for i in range(5):
    print(f"\n--- Test {i+1}/5 ---")
    dialogue = generate_synthetic_call(context)
    
    if dialogue:
        # Get last buyer line
        lines = [l.strip() for l in dialogue.strip().split('\n') if l.strip() and l.strip().startswith('BUYER:')]
        last_buyer = lines[-1] if lines else "N/A"
        print(f"Last BUYER: {last_buyer[:100]}...")
        
        sentiment, outcome, score, objection, feedback = analyze_call(dialogue)
        print(f"Result: {outcome} | Score: {score} | Objection: {objection} | Sentiment: {sentiment}")
        results.append(outcome)
    else:
        print("FAILED to generate")
        results.append("ERROR")
    
    time.sleep(1)

print(f"\n{'=' * 60}")
print(f"Results: {results}")
successes = results.count("Success")
failures = results.count("Failure")
pending = results.count("Pending")
print(f"Success: {successes} | Failure: {failures} | Pending: {pending}")
