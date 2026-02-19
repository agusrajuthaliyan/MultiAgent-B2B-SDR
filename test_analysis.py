
import os
import sys
import traceback

print("Script started.", flush=True)

try:
    from src.agent_logic import analyze_call, _generate_content
    print("Imports successful.", flush=True)
except Exception as e:
    print(f"Import failed: {e}", flush=True)
    traceback.print_exc()
    sys.exit(1)

sample_dialogue = """
SELLER: Hi, I'm calling from DeepData AI.
BUYER: I'm not interested.
SELLER: But we can save you money.
BUYER: Okay, tell me more.
SELLER: We integrate with your stack.
BUYER: Sounds good.
SELLER: Great, how is Tuesday?
"""

print("Testing analyze_call...", flush=True)
try:
    result = analyze_call(sample_dialogue)
    print(f"Result: {result}", flush=True)
except Exception as e:
    print(f"Analysis failed: {e}", flush=True)
    traceback.print_exc()

print("\n--- Raw LLM Check ---", flush=True)
prompt = f"""
Analyze this sales call transcript and provide:
1. Sentiment: Positive, Neutral, or Negative
2. Outcome: Success (meeting booked), Failure (rejected), or Pending (follow-up needed)

Transcript:
{sample_dialogue}

Respond in exactly this format (one word each):
SENTIMENT: [Positive/Neutral/Negative]
OUTCOME: [Success/Failure/Pending]
"""

try:
    raw_response = _generate_content(prompt)
    print(f"Raw Response:\n'{raw_response}'", flush=True) # Quotes to see empty string
except Exception as e:
    print(f"Raw Generation failed: {e}", flush=True)
    traceback.print_exc()
