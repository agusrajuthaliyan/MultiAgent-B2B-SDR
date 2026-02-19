"""
Agent Logic Module — Unified Multi-Provider Edition
=====================================================

Supports both Gemini and Groq as LLM backends.
Provider is selected via the LLM_PROVIDER env var in .env

Providers & Free Tier Limits:
  Groq   : 30 RPM | 14,400 RPD | 40,000 TPM  (recommended)
  Gemini : 15 RPM |  1,000 RPD

Usage:
  Set LLM_PROVIDER=groq   in .env for Groq   (default)
  Set LLM_PROVIDER=gemini in .env for Gemini
"""

import os
import time
import random
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------
# PROVIDER SELECTION
# ---------------------------------------------------------------
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower().strip()

# ---------------------------------------------------------------
# PROVIDER: GROQ
# ---------------------------------------------------------------
if LLM_PROVIDER == "groq":
    from groq import Groq

    _api_key = os.getenv("GROQ_API_KEY")
    _client = Groq(api_key=_api_key)

    # Model selection — free tier
    # llama-3.3-70b-versatile : Best quality, 30 RPM / 14,400 RPD
    # llama-3.1-8b-instant    : Ultra fast, lighter tasks
    # mixtral-8x7b-32768      : Good balance
    MODEL_ID = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    _RATE_LIMIT_CONFIG = {
        "llama-3.3-70b-versatile":  {"delay": 2.5, "rpm": 30},
        "llama-3.1-8b-instant":     {"delay": 2.0, "rpm": 30},
        "mixtral-8x7b-32768":       {"delay": 2.5, "rpm": 30},
    }
    _cfg = _RATE_LIMIT_CONFIG.get(MODEL_ID, {"delay": 2.5, "rpm": 30})
    DELAY_BETWEEN_CALLS = _cfg["delay"]

    def _call_llm(prompt: str) -> str:
        """Send a prompt to Groq and return the text response."""
        response = _client.chat.completions.create(
            model=MODEL_ID,
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant for sales simulation and analysis."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=2048,
        )
        return response.choices[0].message.content.strip()

# ---------------------------------------------------------------
# PROVIDER: GEMINI
# ---------------------------------------------------------------
elif LLM_PROVIDER == "gemini":
    from google import genai

    _api_key = os.getenv("GEMINI_API_KEY")
    _client = genai.Client(api_key=_api_key)

    # Model selection — free tier
    # gemini-2.5-flash      : 10 RPM | 20 RPD  (extremely limited)
    # gemini-2.5-flash-lite : 15 RPM | 1000 RPD (50x more daily quota)
    MODEL_ID = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")

    _RATE_LIMIT_CONFIG = {
        "gemini-2.5-flash":      {"delay": 7.0, "rpm": 10},
        "gemini-2.5-flash-lite": {"delay": 4.5, "rpm": 15},
    }
    _cfg = _RATE_LIMIT_CONFIG.get(MODEL_ID, {"delay": 5.0, "rpm": 10})
    DELAY_BETWEEN_CALLS = _cfg["delay"]

    def _call_llm(prompt: str) -> str:
        """Send a prompt to Gemini and return the text response."""
        response = _client.models.generate_content(
            model=MODEL_ID,
            contents=prompt
        )
        return response.text.strip()

else:
    raise ValueError(
        f"Unknown LLM_PROVIDER: '{LLM_PROVIDER}'. "
        f"Set LLM_PROVIDER to 'groq' or 'gemini' in your .env file."
    )

print(f"[CONFIG] Provider: {LLM_PROVIDER.upper()} | Model: {MODEL_ID} | Delay: {DELAY_BETWEEN_CALLS}s", flush=True)

# ---------------------------------------------------------------
# RATE-LIMITED CONTENT GENERATION (shared by both providers)
# ---------------------------------------------------------------
_last_call_time = 0.0


def _generate_content(prompt: str, max_retries: int = 4) -> str:
    """
    Generate content with automatic rate-limit handling.

    Features:
    - Inter-call delay to stay under RPM
    - Exponential backoff on 429 / rate-limit errors
    - Jitter to avoid thundering-herd with concurrent usage

    Works identically for both Groq and Gemini.
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
            return _call_llm(prompt)

        except Exception as e:
            error_str = str(e).lower()

            is_rate_limit = any(kw in error_str for kw in [
                "429", "resource exhausted", "rate limit",
                "quota", "too many requests", "rate_limit_exceeded",
                "resourceexhausted"
            ])

            if is_rate_limit and attempt < max_retries - 1:
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


# ---------------------------------------------------------------
# PUBLIC API  (same interface regardless of provider)
# ---------------------------------------------------------------

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
    
    Generate 5-7 turns of dialogue. Format each line as:
    SELLER: [message]
    BUYER: [message]
    
    IMPORTANT RULES:
    - Make the conversation realistic with objections (Price, Timing, Competitors, Authority, Integration).
    - The conversation MUST end with a CLEAR OUTCOME from the BUYER — either:
      (a) The buyer AGREES to book a demo/meeting (Success — this should be RARE, ~15-20% of calls), OR
      (b) The buyer firmly REJECTS/DECLINES the offer (Failure — this is the MOST COMMON outcome, ~80-85% of calls).
    - REALISTIC B2B SALES: Most cold outreach calls FAIL. The buyer usually says no.
      Common failure patterns: "We're not interested right now", "We already have a solution",
      "Send me an email and I'll review it" (polite brush-off), "This isn't a priority for us",
      "I don't have budget authority for this", "We're locked into a contract".
    - Success should only happen when the SDR handles objections exceptionally well AND 
      the buyer has a genuine need that aligns with the product.
    - Do NOT end the conversation with an open question or vague "let me think about it".
    - The final line must be from the BUYER with a definitive decision.
    """
    return _generate_content(prompt)


def analyze_call(dialogue: str) -> tuple:
    """
    Analyze a sales call dialogue and return detailed results.
    Used by main.py for batch processing.

    Returns:
        tuple: (sentiment, outcome, score, key_objection, feedback)
    """
    prompt = f"""
    Act as a Sales Coach. Analyze this sales call transcript carefully:
    
    {dialogue}
    
    Determine the outcome based on what ACTUALLY happened in the conversation:
    - Success: The buyer agreed to a meeting, demo, trial, or next step
    - Failure: The buyer rejected, declined, or ended the conversation negatively
    - Pending: ONLY if the conversation genuinely ended without any resolution
    
    Return strictly in this format (no extra text):
    Score: [1-10]
    Outcome: [Success/Failure/Pending]
    Key_Objection: [Price/Timing/Authority/Competitors/Integration/None]
    Sentiment: [Positive/Neutral/Negative]
    Feedback: [1 sentence of actionable advice for the seller]
    """
    result = _generate_content(prompt)

    sentiment = "Neutral"
    outcome = "Pending"
    score = 5
    key_objection = "Unknown"
    feedback = ""

    try:
        for line in result.split('\n'):
            line_stripped = line.strip()
            if line_stripped.upper().startswith('SENTIMENT:'):
                sentiment = line_stripped.split(':', 1)[1].strip()
            elif line_stripped.upper().startswith('OUTCOME:'):
                outcome = line_stripped.split(':', 1)[1].strip()
            elif line_stripped.upper().startswith('SCORE:'):
                score_str = line_stripped.split(':', 1)[1].strip()
                if '/' in score_str:
                    score_str = score_str.split('/')[0]
                score = int(score_str.strip())
            elif line_stripped.upper().startswith('KEY_OBJECTION:') or line_stripped.upper().startswith('KEY OBJECTION:'):
                key_objection = line_stripped.split(':', 1)[1].strip()
            elif line_stripped.upper().startswith('FEEDBACK:'):
                feedback = line_stripped.split(':', 1)[1].strip()
    except Exception:
        pass

    return sentiment, outcome, score, key_objection, feedback


class SalesSimulation:
    """Multi-turn sales simulation engine. Works with any LLM provider."""

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
        if turn_number == 0:
            seller_msg = "Hi, this is Alex from DeepData AI. I noticed your company is scaling fast\u2014are you currently struggling with data processing costs?"
        else:
            last_buyer = self.history[-1][1]
            seller_msg = self._get_seller_response(last_buyer)

        self.history.append(("Seller", seller_msg))

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


def get_provider_info() -> dict:
    """Return current provider configuration for display in UI."""
    return {
        "provider": LLM_PROVIDER,
        "model": MODEL_ID,
        "delay": DELAY_BETWEEN_CALLS,
    }