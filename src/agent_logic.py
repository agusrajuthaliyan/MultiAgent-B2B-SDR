from google import genai
import os
import time
import random
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# Initialize the new GenAI client
client = genai.Client(api_key=api_key)

# ---------------------------------------------------------------
# MODEL SELECTION  (Free-Tier Optimized)
# ---------------------------------------------------------------
# gemini-2.5-flash      : 10 RPM | 20 RPD  (extremely limited)
# gemini-2.5-flash-lite : 15 RPM | 1000 RPD (50x more daily quota)
#
# We default to flash-lite for free-tier friendliness.
# Switch to 'gemini-2.5-flash' if you have a paid plan.
# ---------------------------------------------------------------
MODEL_ID = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")

# Rate-limit configuration (tuned for free tier)
_RATE_LIMIT_CONFIG = {
    "gemini-2.5-flash":      {"delay_between_calls": 7.0,  "rpm": 10},
    "gemini-2.5-flash-lite": {"delay_between_calls": 4.5,  "rpm": 15},
}
_cfg = _RATE_LIMIT_CONFIG.get(MODEL_ID, {"delay_between_calls": 5.0, "rpm": 10})
DELAY_BETWEEN_CALLS = _cfg["delay_between_calls"]

# Track the timestamp of the last API call for pacing
_last_call_time = 0.0

print(f"[CONFIG] Model: {MODEL_ID} | Delay: {DELAY_BETWEEN_CALLS}s between calls", flush=True)


def _generate_content(prompt: str, max_retries: int = 4) -> str:
    """
    Generate content with automatic rate-limit handling.
    
    Features:
    - Inter-call delay to stay under RPM
    - Exponential backoff on 429 / Resource Exhausted errors
    - Jitter to avoid thundering-herd with concurrent usage
    """
    global _last_call_time

    for attempt in range(max_retries):
        # ---- Pacing: respect inter-call delay ----
        elapsed = time.time() - _last_call_time
        if elapsed < DELAY_BETWEEN_CALLS:
            wait = DELAY_BETWEEN_CALLS - elapsed
            time.sleep(wait)

        try:
            _last_call_time = time.time()
            response = client.models.generate_content(
                model=MODEL_ID,
                contents=prompt
            )
            return response.text.strip()

        except Exception as e:
            error_str = str(e).lower()

            # Detect rate-limit / quota errors
            is_rate_limit = any(kw in error_str for kw in [
                "429", "resource exhausted", "rate limit",
                "quota", "too many requests", "resourceexhausted"
            ])

            if is_rate_limit and attempt < max_retries - 1:
                # Exponential backoff: 15s, 30s, 60s, ... + jitter
                backoff = min(15 * (2 ** attempt), 120) + random.uniform(1, 5)
                print(
                    f"   [RATE LIMIT] Hit rate limit (attempt {attempt + 1}/{max_retries}). "
                    f"Waiting {backoff:.0f}s before retry...",
                    flush=True,
                )
                time.sleep(backoff)
            else:
                print(f"   [Error] API call failed: {e}", flush=True)
                return ""

    print("   [Error] All retries exhausted.", flush=True)
    return ""


def generate_synthetic_call(context: str) -> str:
    """
    Generate a synthetic sales call dialogue based on company context.
    Used by main.py for batch processing.
    """
    prompt = f"""
    Generate a realistic B2B sales call dialogue between a Sales Development Representative (SDR) 
    and a skeptical CTO at a company with this context:
    
    {context[:2000]}
    
    The SDR is selling 'DeepData AI' - a data processing solution.
    
    Generate 4-6 turns of dialogue. Format each line as:
    SELLER: [message]
    BUYER: [message]
    
    Make the conversation realistic with objections and responses.
    """
    return _generate_content(prompt)


def analyze_call(dialogue: str) -> tuple:
    """
    Analyze a sales call dialogue and return sentiment and outcome.
    Used by main.py for batch processing.
    
    Returns:
        tuple: (sentiment, outcome)
    """
    prompt = f"""
    Analyze this sales call transcript and provide:
    1. Sentiment: Positive, Neutral, or Negative
    2. Outcome: Success (meeting booked), Failure (rejected), or Pending (follow-up needed)
    
    Transcript:
    {dialogue}
    
    Respond in exactly this format (one word each):
    SENTIMENT: [Positive/Neutral/Negative]
    OUTCOME: [Success/Failure/Pending]
    """
    result = _generate_content(prompt)
    
    # Parse the result
    sentiment = "Neutral"
    outcome = "Pending"
    
    try:
        for line in result.split('\n'):
            if 'SENTIMENT:' in line.upper():
                sentiment = line.split(':')[1].strip()
            elif 'OUTCOME:' in line.upper():
                outcome = line.split(':')[1].strip()
    except:
        pass
    
    return sentiment, outcome


class SalesSimulation:
    def __init__(self, company_context):
        self.context = company_context
        self.history = []
        self.max_turns = 6
        
    def _get_buyer_response(self, last_seller_msg):
        """The Buyer Agent: Skeptical, busy, protective of budget."""
        prompt = f"""
        ROLE: You are a skeptical CTO at this company: {self.context[:500]}...
        GOAL: Evaluate the seller's pitch. Be difficult but realistic.
        CURRENT CONVERSATION:
        {self.format_history()}
        Seller just said: "{last_seller_msg}"
        
        TASK: Respond in 1-2 sentences. Raise an objection (Price, Timing, Competitors) if not convinced.
        """
        return _generate_content(prompt)

    def _get_seller_response(self, last_buyer_msg):
        """The Seller Agent: Persistent, polite, uses value-based selling."""
        prompt = f"""
        ROLE: You are a top-tier SDR selling 'DeepData AI'.
        GOAL: Book a meeting. Overcome objections.
        CURRENT CONVERSATION:
        {self.format_history()}
        Buyer just said: "{last_buyer_msg}"
        
        TASK: Respond in 1-2 sentences. Address the objection and pivot back to value.
        """
        return _generate_content(prompt)

    def format_history(self):
        return "\n".join([f"{role}: {msg}" for role, msg in self.history])

    def run_turn(self, turn_number):
        """Executes ONE turn of conversation (Seller -> Buyer)."""
        
        # 1. Seller Speaks first or responds
        if turn_number == 0:
            seller_msg = "Hi, this is Alex from DeepData AI. I noticed your company is scaling fast\u2014are you currently struggling with data processing costs?"
        else:
            # Get last buyer message
            last_buyer = self.history[-1][1]
            seller_msg = self._get_seller_response(last_buyer)
            
        self.history.append(("Seller", seller_msg))
        
        # 2. Buyer Responds
        buyer_msg = self._get_buyer_response(seller_msg)
        self.history.append(("Buyer", buyer_msg))
        
        return seller_msg, buyer_msg

    def analyze_result(self):
        """Final LLM Judge step."""
        full_text = self.format_history()
        prompt = f"""
        Act as a Sales Coach. Analyze this transcript:
        {full_text}
        
        Return strictly in this format:
        Score: [1-10]
        Outcome: [Success/Failure]
        Key_Objection: [Price/Timing/Authority]
        Feedback: [1 sentence advice]
        """
        result = _generate_content(prompt)
        if result:
            return result
        return "Score: 0\nOutcome: Error"