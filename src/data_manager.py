"""
Data Management Module for DeepMost Agentic SDR

This module handles all data persistence for the sales simulation system.
It saves data in multiple formats suitable for EDA and ML modeling.

Data Schema:
- simulation_id: UUID for each simulation run
- timestamp: When the simulation was run
- target_url: Company website URL
- company_context: Scraped company information
- conversation: Full turn-by-turn dialogue with metadata
- analysis: AI analysis of the call
- metrics: Extracted numerical metrics for modeling
"""

import os
import json
import csv
import uuid
import time as _time
from contextlib import contextmanager
from datetime import datetime
from typing import Dict, List, Any, Optional
import pandas as pd

# Data directories
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
CONVERSATIONS_DIR = os.path.join(RAW_DIR, "conversations")

# Ensure directories exist
for dir_path in [DATA_DIR, RAW_DIR, PROCESSED_DIR, CONVERSATIONS_DIR]:
    os.makedirs(dir_path, exist_ok=True)


@contextmanager
def _safe_open(filepath, mode='a', retries=5, delay=0.5, **kwargs):
    """
    Open a file with automatic retry on PermissionError.

    On Windows, CSV files can be temporarily locked by other processes
    (Excel, another Python reader, antivirus scanning, etc.). This wrapper
    retries the open() call a few times before giving up.
    """
    last_err = None
    for attempt in range(retries):
        try:
            f = open(filepath, mode, **kwargs)
            try:
                yield f
            finally:
                f.close()
            return
        except PermissionError as e:
            last_err = e
            if attempt < retries - 1:
                print(
                    f"   [FILE LOCK] {os.path.basename(filepath)} is locked "
                    f"(attempt {attempt + 1}/{retries}), retrying in {delay}s...",
                    flush=True,
                )
                _time.sleep(delay)
    raise PermissionError(
        f"Could not write to {filepath} after {retries} attempts. "
        f"Make sure the file is not open in Excel or another program. "
        f"Original error: {last_err}"
    )


class SimulationDataManager:
    """
    Manages data persistence for sales simulations.
    Saves data in multiple formats for comprehensive analysis.
    """
    
    def __init__(self):
        self.master_csv_path = os.path.join(PROCESSED_DIR, "simulations_master.csv")
        self.turns_csv_path = os.path.join(PROCESSED_DIR, "conversation_turns.csv")
        self.metrics_csv_path = os.path.join(PROCESSED_DIR, "simulation_metrics.csv")
        self._ensure_csv_headers()
    
    def _ensure_csv_headers(self):
        """Create CSV files with headers if they don't exist."""
        # Master simulations CSV
        if not os.path.exists(self.master_csv_path):
            with _safe_open(self.master_csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'simulation_id', 'timestamp', 'target_url', 'company_context_length',
                    'num_turns', 'total_seller_words', 'total_buyer_words',
                    'avg_seller_turn_length', 'avg_buyer_turn_length',
                    'score', 'outcome', 'key_objection', 'feedback',
                    'conversation_file', 'source'
                ])
        
        # Turns CSV (for turn-level analysis)
        if not os.path.exists(self.turns_csv_path):
            with _safe_open(self.turns_csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'simulation_id', 'turn_number', 'speaker', 'message',
                    'word_count', 'char_count', 'timestamp'
                ])
        
        # Metrics CSV (for ML-ready features)
        if not os.path.exists(self.metrics_csv_path):
            with _safe_open(self.metrics_csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'simulation_id', 'timestamp', 'target_url',
                    'context_length', 'num_turns',
                    'seller_total_words', 'buyer_total_words',
                    'seller_avg_words_per_turn', 'buyer_avg_words_per_turn',
                    'seller_max_words', 'buyer_max_words',
                    'seller_min_words', 'buyer_min_words',
                    'word_ratio_seller_buyer', 'total_conversation_length',
                    'score', 'outcome_binary', 'outcome_label', 'objection_type',
                    'source'
                ])
    
    def save_simulation(
        self,
        target_url: str,
        company_context: str,
        conversation_history: List[tuple],
        analysis_result: str,
        source: str = "interactive"
    ) -> str:
        """
        Save a complete simulation with all associated data.
        
        Args:
            target_url: The company URL that was scraped
            company_context: The scraped company information
            conversation_history: List of (speaker, message) tuples
            analysis_result: The AI analysis of the call
            source: Origin of simulation ('batch_v2', 'interactive', etc.)
            
        Returns:
            simulation_id: UUID of the saved simulation
        """
        simulation_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        # Parse the analysis result
        parsed_analysis = self._parse_analysis(analysis_result)
        
        # Calculate conversation metrics
        metrics = self._calculate_metrics(conversation_history)
        
        # Save full conversation as JSON
        conversation_file = self._save_conversation_json(
            simulation_id, timestamp, target_url, 
            company_context, conversation_history, 
            analysis_result, parsed_analysis, metrics
        )
        
        # Append to master CSV
        self._append_to_master_csv(
            simulation_id, timestamp, target_url, company_context,
            conversation_history, parsed_analysis, metrics, conversation_file,
            source
        )
        
        # Append conversation turns
        self._append_turns_to_csv(simulation_id, timestamp, conversation_history)
        
        # Append ML-ready metrics
        self._append_metrics_to_csv(
            simulation_id, timestamp, target_url, company_context,
            metrics, parsed_analysis, source
        )
        
        print(f"[DATA] Simulation {simulation_id[:8]}... saved successfully", flush=True)
        return simulation_id
    
    def _parse_analysis(self, analysis_result: str) -> Dict[str, Any]:
        """Parse the LLM analysis result into structured data."""
        parsed = {
            'score': 0,
            'outcome': 'Unknown',
            'key_objection': 'Unknown',
            'feedback': ''
        }
        
        try:
            for line in analysis_result.split('\n'):
                line = line.strip()
                if line.startswith('Score:'):
                    try:
                        score_str = line.split(':')[1].strip()
                        # Handle cases like "7/10" or "7"
                        if '/' in score_str:
                            score_str = score_str.split('/')[0]
                        parsed['score'] = int(score_str.strip())
                    except:
                        parsed['score'] = 0
                elif line.startswith('Outcome:'):
                    parsed['outcome'] = line.split(':')[1].strip()
                elif line.startswith('Key_Objection:') or line.startswith('Key Objection:'):
                    parsed['key_objection'] = line.split(':')[1].strip()
                elif line.startswith('Feedback:'):
                    parsed['feedback'] = line.split(':', 1)[1].strip()
        except Exception as e:
            print(f"[WARN] Failed to parse analysis: {e}", flush=True)
        
        return parsed
    
    def _calculate_metrics(self, conversation_history: List[tuple]) -> Dict[str, Any]:
        """Calculate various metrics from the conversation for ML features."""
        seller_messages = [msg for role, msg in conversation_history if role == 'Seller']
        buyer_messages = [msg for role, msg in conversation_history if role == 'Buyer']
        
        seller_word_counts = [len(msg.split()) for msg in seller_messages]
        buyer_word_counts = [len(msg.split()) for msg in buyer_messages]
        
        total_seller_words = sum(seller_word_counts)
        total_buyer_words = sum(buyer_word_counts)
        
        metrics = {
            'num_turns': len(conversation_history) // 2,
            'total_seller_words': total_seller_words,
            'total_buyer_words': total_buyer_words,
            'avg_seller_words_per_turn': total_seller_words / max(len(seller_messages), 1),
            'avg_buyer_words_per_turn': total_buyer_words / max(len(buyer_messages), 1),
            'seller_max_words': max(seller_word_counts) if seller_word_counts else 0,
            'buyer_max_words': max(buyer_word_counts) if buyer_word_counts else 0,
            'seller_min_words': min(seller_word_counts) if seller_word_counts else 0,
            'buyer_min_words': min(buyer_word_counts) if buyer_word_counts else 0,
            'word_ratio_seller_buyer': total_seller_words / max(total_buyer_words, 1),
            'total_conversation_length': total_seller_words + total_buyer_words,
            'seller_messages': seller_messages,
            'buyer_messages': buyer_messages,
            'seller_word_counts': seller_word_counts,
            'buyer_word_counts': buyer_word_counts
        }
        
        return metrics
    
    def _save_conversation_json(
        self, simulation_id: str, timestamp: str, target_url: str,
        company_context: str, conversation_history: List[tuple],
        analysis_result: str, parsed_analysis: Dict, metrics: Dict
    ) -> str:
        """Save the full conversation as a JSON file."""
        filename = f"{simulation_id}.json"
        filepath = os.path.join(CONVERSATIONS_DIR, filename)
        
        # Build structured conversation
        conversation_structured = []
        for i, (role, message) in enumerate(conversation_history):
            conversation_structured.append({
                'turn': i // 2 + 1,
                'speaker': role,
                'message': message,
                'word_count': len(message.split()),
                'char_count': len(message)
            })
        
        data = {
            'simulation_id': simulation_id,
            'timestamp': timestamp,
            'target_url': target_url,
            'company_context': company_context,
            'conversation': conversation_structured,
            'analysis': {
                'raw': analysis_result,
                'parsed': parsed_analysis
            },
            'metrics': {
                'num_turns': metrics['num_turns'],
                'total_seller_words': metrics['total_seller_words'],
                'total_buyer_words': metrics['total_buyer_words'],
                'avg_seller_words_per_turn': round(metrics['avg_seller_words_per_turn'], 2),
                'avg_buyer_words_per_turn': round(metrics['avg_buyer_words_per_turn'], 2),
                'word_ratio_seller_buyer': round(metrics['word_ratio_seller_buyer'], 2),
                'total_conversation_length': metrics['total_conversation_length']
            }
        }
        
        with _safe_open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        return filename
    
    def _append_to_master_csv(
        self, simulation_id: str, timestamp: str, target_url: str,
        company_context: str, conversation_history: List[tuple],
        parsed_analysis: Dict, metrics: Dict, conversation_file: str,
        source: str = "interactive"
    ):
        """Append a row to the master simulations CSV."""
        with _safe_open(self.master_csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                simulation_id,
                timestamp,
                target_url,
                len(company_context),
                metrics['num_turns'],
                metrics['total_seller_words'],
                metrics['total_buyer_words'],
                round(metrics['avg_seller_words_per_turn'], 2),
                round(metrics['avg_buyer_words_per_turn'], 2),
                parsed_analysis['score'],
                parsed_analysis['outcome'],
                parsed_analysis['key_objection'],
                parsed_analysis['feedback'],
                conversation_file,
                source
            ])
    
    def _append_turns_to_csv(
        self, simulation_id: str, timestamp: str, 
        conversation_history: List[tuple]
    ):
        """Append individual turns to the turns CSV."""
        with _safe_open(self.turns_csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            for i, (role, message) in enumerate(conversation_history):
                writer.writerow([
                    simulation_id,
                    i // 2 + 1,  # Turn number
                    role,
                    message,
                    len(message.split()),
                    len(message),
                    timestamp
                ])
    
    def _append_metrics_to_csv(
        self, simulation_id: str, timestamp: str, target_url: str,
        company_context: str, metrics: Dict, parsed_analysis: Dict,
        source: str = "interactive"
    ):
        """Append ML-ready metrics to the metrics CSV."""
        # Convert outcome to binary for classification
        outcome_label = parsed_analysis['outcome']
        outcome_binary = 1 if 'success' in outcome_label.lower() else 0
        
        with _safe_open(self.metrics_csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                simulation_id,
                timestamp,
                target_url,
                len(company_context),
                metrics['num_turns'],
                metrics['total_seller_words'],
                metrics['total_buyer_words'],
                round(metrics['avg_seller_words_per_turn'], 2),
                round(metrics['avg_buyer_words_per_turn'], 2),
                metrics['seller_max_words'],
                metrics['buyer_max_words'],
                metrics['seller_min_words'],
                metrics['buyer_min_words'],
                round(metrics['word_ratio_seller_buyer'], 2),
                metrics['total_conversation_length'],
                parsed_analysis['score'],
                outcome_binary,
                outcome_label,
                parsed_analysis['key_objection'],
                source
            ])
    
    def load_all_simulations(self) -> pd.DataFrame:
        """Load all simulations from the master CSV as a DataFrame."""
        if os.path.exists(self.master_csv_path):
            return pd.read_csv(self.master_csv_path)
        return pd.DataFrame()
    
    def load_all_turns(self) -> pd.DataFrame:
        """Load all conversation turns as a DataFrame."""
        if os.path.exists(self.turns_csv_path):
            return pd.read_csv(self.turns_csv_path)
        return pd.DataFrame()
    
    def load_all_metrics(self) -> pd.DataFrame:
        """Load ML-ready metrics as a DataFrame."""
        if os.path.exists(self.metrics_csv_path):
            return pd.read_csv(self.metrics_csv_path)
        return pd.DataFrame()
    
    def load_conversation(self, simulation_id: str) -> Optional[Dict]:
        """Load a specific conversation by its simulation ID."""
        filepath = os.path.join(CONVERSATIONS_DIR, f"{simulation_id}.json")
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics of all collected data."""
        df = self.load_all_metrics()
        if df.empty:
            return {'total_simulations': 0}
        
        return {
            'total_simulations': len(df),
            'success_rate': df['outcome_binary'].mean() * 100,
            'avg_score': df['score'].mean(),
            'avg_turns': df['num_turns'].mean(),
            'avg_conversation_length': df['total_conversation_length'].mean(),
            'objection_distribution': df['objection_type'].value_counts().to_dict(),
            'outcome_distribution': df['outcome_label'].value_counts().to_dict()
        }


# Singleton instance for easy import
data_manager = SimulationDataManager()
