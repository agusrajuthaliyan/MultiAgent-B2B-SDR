import gradio as gr
import pandas as pd
import time
import matplotlib.pyplot as plt
import os
from src.scraper import simple_scraper
from src.agent_logic import SalesSimulation
from src.data_manager import data_manager

# --- VERSION CHECK (For your sanity) ---
import sys

# Force unbuffered output for Windows
os.environ['PYTHONUNBUFFERED'] = '1'

# Reconfigure stdout/stderr for UTF-8 on Windows (without TextIOWrapper buffering issue)
if sys.platform == 'win32' and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass  # Fall back to default encoding if reconfigure fails

print(f"[LAUNCH] Launching App... (Gradio Version: {gr.__version__})", flush=True)

# Global Data Store (in-memory for session, persistent via data_manager)
SESSION_DATA = []


def run_simulation_generator(url):
    """Generator function that streams the conversation turn-by-turn."""
    
    # Initial State: Empty list
    chat_log = []
    current_url = url  # Store for later use
    current_context = None
    
    yield "[SCRAPING] Agent 1: Scraping Target Company...", chat_log, "", None, ""
    context = simple_scraper(url)
    
    if not context:
        yield "[ERROR] Scraping Failed", chat_log, "Check URL", None, ""
        return
    
    current_context = context

    # Initialize the Simulation Engine
    sim = SalesSimulation(context)
    
    yield "[INIT] Agents Initialized. Starting Negotiation...", chat_log, "", None, ""
    
    # THE LOOP (Turn-by-Turn)
    for i in range(4):  # 4 turns
        seller_text, buyer_text = sim.run_turn(i)
        
        # --- GRADIO 6.x FORMAT ---
        # Format: [{"role": "user"|"assistant", "content": "..."}]
        
        # 1. Seller Speaks (appears as user/right side)
        chat_log.append({"role": "user", "content": f"[SELLER]: {seller_text}"})
        yield f"[TURN {i+1}] Seller speaking...", chat_log, "", None, ""
        time.sleep(1.0)
        
        # 2. Buyer Speaks (appears as assistant/left side)
        chat_log.append({"role": "assistant", "content": f"[BUYER]: {buyer_text}"})
        yield f"[TURN {i+1}] Buyer responding...", chat_log, "", None, ""
        time.sleep(1.0)
        
    # Final Analysis
    yield "[ANALYSIS] Agents are Analyzing the call...", chat_log, "", None, ""
    analysis = sim.analyze_result()
    
    # Save Data using the data manager (for persistence)
    try:
        simulation_id = data_manager.save_simulation(
            target_url=current_url,
            company_context=current_context,
            conversation_history=sim.history,
            analysis_result=analysis
        )
        save_status = f"Data saved! ID: {simulation_id[:8]}..."
    except Exception as e:
        print(f"[ERROR] Failed to save simulation: {e}", flush=True)
        save_status = f"Save failed: {str(e)[:50]}"
    
    # Also keep in session memory for charts
    SESSION_DATA.append({
        "URL": current_url, 
        "Analysis": analysis,
        "History": sim.history
    })
    
    yield "[COMPLETE] Simulation Complete", chat_log, analysis, update_charts(), save_status


def update_charts():
    """Generates a Matplotlib chart for the Dashboard tab using persistent data."""
    # Try to load from persistent storage first
    try:
        df = data_manager.load_all_metrics()
        if not df.empty:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle('Sales Simulation Analytics Dashboard', fontsize=14, fontweight='bold')
            
            # 1. Outcome Distribution (Pie)
            ax1 = axes[0, 0]
            outcome_counts = df['outcome_label'].value_counts()
            colors = ['#2ecc71', '#e74c3c', '#f39c12', '#9b59b6'][:len(outcome_counts)]
            outcome_counts.plot(kind='pie', ax=ax1, autopct='%1.1f%%', colors=colors)
            ax1.set_title('Outcome Distribution')
            ax1.set_ylabel('')
            
            # 2. Score Distribution (Histogram)
            ax2 = axes[0, 1]
            df['score'].hist(ax=ax2, bins=10, color='#3498db', edgecolor='white')
            ax2.set_title('Score Distribution')
            ax2.set_xlabel('Score (1-10)')
            ax2.set_ylabel('Count')
            ax2.axvline(df['score'].mean(), color='red', linestyle='--', label=f'Mean: {df["score"].mean():.1f}')
            ax2.legend()
            
            # 3. Objection Types (Bar)
            ax3 = axes[1, 0]
            objection_counts = df['objection_type'].value_counts()
            objection_counts.plot(kind='bar', ax=ax3, color='#e67e22')
            ax3.set_title('Key Objection Types')
            ax3.set_xlabel('Objection')
            ax3.set_ylabel('Count')
            ax3.tick_params(axis='x', rotation=45)
            
            # 4. Conversation Length vs Score (Scatter)
            ax4 = axes[1, 1]
            colors_scatter = ['green' if x == 1 else 'red' for x in df['outcome_binary']]
            ax4.scatter(df['total_conversation_length'], df['score'], c=colors_scatter, alpha=0.6)
            ax4.set_title('Conversation Length vs Score')
            ax4.set_xlabel('Total Words')
            ax4.set_ylabel('Score')
            
            plt.tight_layout()
            return fig
    except Exception as e:
        print(f"[WARN] Chart error: {e}", flush=True)
    
    # Fallback to session data
    if not SESSION_DATA:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, 'No data yet.\nRun a simulation first!', 
                ha='center', va='center', fontsize=14, color='gray')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        return fig
    
    outcomes = []
    for d in SESSION_DATA:
        if "Success" in d['Analysis']:
            outcomes.append("Success")
        else:
            outcomes.append("Failure")
        
    fig, ax = plt.subplots()
    if outcomes:
        pd.Series(outcomes).value_counts().plot(kind='pie', ax=ax, autopct='%1.1f%%', colors=['#66b3ff', '#ff9999'])
        ax.set_title("Agent Success Rate")
    return fig


def get_data_summary():
    """Get summary statistics of collected data."""
    try:
        stats = data_manager.get_summary_stats()
        if stats['total_simulations'] == 0:
            return "No simulations recorded yet. Run some simulations to see statistics."
        
        summary = f"""
## Data Collection Summary

**Total Simulations:** {stats['total_simulations']}
**Success Rate:** {stats['success_rate']:.1f}%
**Average Score:** {stats['avg_score']:.2f}/10
**Avg Turns per Call:** {stats['avg_turns']:.1f}
**Avg Conversation Length:** {stats['avg_conversation_length']:.0f} words

### Outcome Distribution:
{chr(10).join([f"- {k}: {v}" for k, v in stats.get('outcome_distribution', {}).items()])}

### Objection Types:
{chr(10).join([f"- {k}: {v}" for k, v in stats.get('objection_distribution', {}).items()])}
        """
        return summary.strip()
    except Exception as e:
        return f"Error loading summary: {e}"


def export_data_info():
    """Return information about exported data files."""
    data_dir = os.path.join(os.path.dirname(__file__), "data", "processed")
    files_info = []
    
    for filename in ['simulations_master.csv', 'conversation_turns.csv', 'simulation_metrics.csv']:
        filepath = os.path.join(data_dir, filename)
        if os.path.exists(filepath):
            size = os.path.getsize(filepath) / 1024  # KB
            files_info.append(f"- **{filename}**: {size:.1f} KB")
        else:
            files_info.append(f"- **{filename}**: Not created yet")
    
    return f"""
## Data Export Files

Location: `data/processed/`

{chr(10).join(files_info)}

### How to load for EDA:

```python
import pandas as pd

# Load master simulations
df = pd.read_csv('data/processed/simulations_master.csv')

# Load turn-level data
turns = pd.read_csv('data/processed/conversation_turns.csv')

# Load ML-ready metrics
metrics = pd.read_csv('data/processed/simulation_metrics.csv')

# Load individual conversation
import json
with open('data/raw/conversations/<simulation_id>.json') as f:
    conversation = json.load(f)
```
    """


# --- UNIVERSAL UI BLOCK ---
with gr.Blocks() as demo:
    gr.Markdown("## DeepMost Agentic SDR (Multi-Agent System)")
    gr.Markdown("*AI-powered sales simulation with comprehensive data collection for EDA & ML*")
    
    with gr.Tabs():
        # TAB 1: The Simulation
        with gr.TabItem("Live Simulation"):
            with gr.Row():
                url_input = gr.Textbox(label="Target Company URL", value="https://deepmostai.com")
                run_btn = gr.Button("Initialize Agents", variant="primary")
            
            with gr.Row():
                chatbot = gr.Chatbot(label="Real-Time Agent Interaction", height=450)
                
                with gr.Column():
                    status = gr.Textbox(label="System Status")
                    analysis_box = gr.Textbox(label="Post-Call Analytics", lines=8)
                    save_status = gr.Textbox(label="Data Save Status", lines=1)

        # TAB 2: Analytics Dashboard
        with gr.TabItem("Analytics Dashboard"):
            gr.Markdown("### Aggregate Performance Metrics")
            with gr.Row():
                refresh_btn = gr.Button("Refresh Charts")
                summary_btn = gr.Button("Show Summary Stats")
            
            plot_output = gr.Plot(label="Analytics Charts")
            summary_output = gr.Markdown(label="Data Summary")
            
            refresh_btn.click(fn=update_charts, inputs=None, outputs=plot_output)
            summary_btn.click(fn=get_data_summary, inputs=None, outputs=summary_output)
        
        # TAB 3: Data Export Info
        with gr.TabItem("Data Export"):
            gr.Markdown("### Data Files for EDA & Modeling")
            export_info_btn = gr.Button("Show Export Info")
            export_info_output = gr.Markdown()
            
            export_info_btn.click(fn=export_data_info, inputs=None, outputs=export_info_output)
            
            gr.Markdown("""
### CSV Files Generated:

| File | Description | Use Case |
|------|-------------|----------|
| `simulations_master.csv` | One row per simulation with all metadata | High-level analysis |
| `conversation_turns.csv` | One row per message turn | Dialogue analysis, NLP |
| `simulation_metrics.csv` | ML-ready features with binary outcomes | Classification modeling |

### JSON Files:
- `data/raw/conversations/<id>.json` - Full conversation with context for each simulation
            """)

    # Event Logic
    run_btn.click(
        fn=run_simulation_generator,
        inputs=url_input,
        outputs=[status, chatbot, analysis_box, plot_output, save_status]
    )

if __name__ == "__main__":
    # Robust launch
    try:
        demo.queue().launch(theme=gr.themes.Soft())
    except:
        demo.launch()