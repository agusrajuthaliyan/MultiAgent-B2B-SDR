"""
DeepMost Agentic SDR - Premium Edition (Unified)
==================================================

AI-Powered Sales Development Representative with Advanced Analytics.
Features: Real-time coaching, ML predictions, interactive dashboards.

Works with any LLM provider (Groq/Gemini) configured via LLM_PROVIDER in .env
"""

import gradio as gr
import pandas as pd
import time
import os
import sys

# Configure for Windows
os.environ['PYTHONUNBUFFERED'] = '1'
if sys.platform == 'win32' and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# Core imports — unified agent logic (auto-selects provider from .env)
from src.scraper import simple_scraper
from src.agent_logic import SalesSimulation, get_provider_info
from src.data_manager import data_manager

# Analytics imports
from src.analytics_engine import (
    insights_generator, 
    real_time_coach, 
    conversation_analyzer,
    predictive_analytics
)

from src.dashboard_components import (
    create_win_rate_gauge,
    create_sentiment_trajectory,
    create_objection_radar,
    create_engagement_metrics,
    create_win_probability_funnel,
    create_score_distribution,
    create_performance_trend,
    create_feature_importance,
    create_comprehensive_dashboard,
    create_outcome_sunburst,
    generate_insights_html,
    create_empty_figure
)

# Provider info for display
_provider = get_provider_info()
_provider_label = f"{_provider['provider'].upper()} ({_provider['model']})"

print(f"[LAUNCH] DeepMost Agentic SDR Premium v2.0 | {_provider_label} (Gradio: {gr.__version__})", flush=True)

# ============================================================================
# CUSTOM CSS FOR PREMIUM LOOK
# ============================================================================
CUSTOM_CSS = """
/* Premium Dark Theme */
.gradio-container {
    background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%) !important;
    min-height: 100vh;
}

/* Header Styling */
.main-header {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.5rem !important;
    font-weight: 800 !important;
    text-align: center;
    margin-bottom: 0.5rem !important;
}

.sub-header {
    color: #8b9dc3 !important;
    text-align: center;
    font-size: 1.1rem !important;
    margin-bottom: 2rem !important;
}

/* Card Styling */
.card-container {
    background: rgba(30, 30, 46, 0.8) !important;
    border: 1px solid rgba(139, 92, 246, 0.2) !important;
    border-radius: 16px !important;
    padding: 20px !important;
    backdrop-filter: blur(10px) !important;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3) !important;
}

/* Button Styling */
.primary-btn {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
    padding: 12px 24px !important;
    border-radius: 10px !important;
    transition: all 0.3s ease !important;
}

.primary-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4) !important;
}

/* Chatbot Styling */
.chatbot-container {
    border-radius: 16px !important;
    overflow: hidden !important;
}

/* Tab Styling */
.tab-nav button {
    background: transparent !important;
    color: #8b9dc3 !important;
    border-bottom: 2px solid transparent !important;
    font-weight: 500 !important;
    transition: all 0.3s ease !important;
}

.tab-nav button.selected {
    color: #667eea !important;
    border-bottom: 2px solid #667eea !important;
}

/* Metric Cards */
.metric-card {
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.1), rgba(139, 92, 246, 0.1)) !important;
    border: 1px solid rgba(139, 92, 246, 0.3) !important;
    border-radius: 12px !important;
    padding: 16px !important;
    text-align: center !important;
}

.metric-value {
    font-size: 2rem !important;
    font-weight: 700 !important;
    color: #f8fafc !important;
}

.metric-label {
    font-size: 0.9rem !important;
    color: #64748b !important;
}

/* Status Badge */
.status-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 600;
}

.status-success { background: rgba(16, 185, 129, 0.2); color: #10b981; }
.status-warning { background: rgba(245, 158, 11, 0.2); color: #f59e0b; }
.status-danger { background: rgba(239, 68, 68, 0.2); color: #ef4444; }

/* Insights Panel */
.insights-panel {
    background: rgba(30, 30, 46, 0.9) !important;
    border-radius: 12px !important;
    padding: 20px !important;
    border: 1px solid rgba(99, 102, 241, 0.3) !important;
}

/* Provider Badge */
.provider-badge {
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.15), rgba(99, 102, 241, 0.15)) !important;
    border: 1px solid rgba(16, 185, 129, 0.3) !important;
    border-radius: 10px !important;
    padding: 8px 16px !important;
    text-align: center !important;
    font-size: 0.9rem !important;
}

/* Animations */
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
}

.loading { animation: pulse 1.5s infinite; }

/* Plot containers */
.plotly-graph-div {
    border-radius: 12px !important;
    overflow: hidden !important;
}
"""

# ============================================================================
# SESSION STATE
# ============================================================================
SESSION_DATA = []

# Dynamic sleep based on provider speed
_TURN_SLEEP = 0.5 if _provider["provider"] == "groq" else 1.0

# ============================================================================
# CORE SIMULATION FUNCTIONS
# ============================================================================

def run_simulation_with_analytics(url):
    """
    Run simulation with real-time analytics and coaching.
    Yields updates for progressive UI rendering.
    """
    chat_log = []
    insights_html = "<p style='color: #64748b;'>Waiting for simulation...</p>"
    
    # Phase 1: Scraping
    yield (
        "🔍 Agent 1: Researching Target Company...",
        chat_log,
        "",
        insights_html,
        create_empty_figure("Gathering data..."),
        create_empty_figure(),
        ""
    )
    
    context = simple_scraper(url)
    
    if not context:
        yield (
            "❌ Scraping Failed - Check URL",
            chat_log,
            "Could not retrieve company data.",
            "<p style='color: #ef4444;'>Failed to scrape target URL.</p>",
            create_empty_figure("Error"),
            create_empty_figure(),
            "Error: Scraping failed"
        )
        return
    
    # Phase 2: Initialize Simulation
    sim = SalesSimulation(context)
    
    yield (
        f"🤖 Agents Initialized ({_provider_label}) - Starting Negotiation...",
        chat_log,
        "",
        "<p style='color: #f59e0b;'>Simulation in progress...</p>",
        create_empty_figure("Initializing..."),
        create_empty_figure(),
        ""
    )
    
    time.sleep(0.5)
    
    # Phase 3: Run Conversation Turns
    for turn in range(4):
        seller_text, buyer_text = sim.run_turn(turn)
        
        # Add seller message
        chat_log.append({"role": "user", "content": f"**SELLER:** {seller_text}"})
        
        # Get real-time coaching
        coaching = real_time_coach.get_live_suggestions(sim.history, turn)
        coaching_html = f"""
        <div style='padding: 12px; background: rgba(99,102,241,0.1); border-radius: 8px; margin-bottom: 10px;'>
            <div style='font-weight: 600; color: #667eea; margin-bottom: 8px;'>
                🎯 Real-Time Coach (Turn {turn + 1})
            </div>
            <div style='color: #e2e8f0; font-size: 14px;'>
                {'<br>'.join(coaching.get('suggestions', []))}
            </div>
            <div style='margin-top: 8px; font-size: 12px; color: #64748b;'>
                Urgency: <span style='color: {"#ef4444" if coaching.get("urgency") == "high" else "#f59e0b" if coaching.get("urgency") == "medium" else "#10b981"};'>
                    {coaching.get('urgency', 'low').upper()}
                </span>
            </div>
        </div>
        """
        
        yield (
            f"💬 Turn {turn + 1}: Seller speaking...",
            chat_log,
            "",
            coaching_html,
            create_empty_figure(f"Turn {turn + 1}/4"),
            create_empty_figure(),
            ""
        )
        time.sleep(_TURN_SLEEP)
        
        # Add buyer message
        chat_log.append({"role": "assistant", "content": f"**BUYER:** {buyer_text}"})
        
        # Update sentiment trajectory
        sentiment = conversation_analyzer.analyze_sentiment_trajectory(sim.history)
        sentiment_fig = create_sentiment_trajectory(sentiment)
        
        yield (
            f"💬 Turn {turn + 1}: Buyer responding...",
            chat_log,
            "",
            coaching_html,
            sentiment_fig,
            create_empty_figure(),
            ""
        )
        time.sleep(_TURN_SLEEP)
    
    # Phase 4: Analysis
    yield (
        "📊 Analyzing Call Performance...",
        chat_log,
        "",
        "<p style='color: #f59e0b;'>Running AI analysis...</p>",
        sentiment_fig,
        create_empty_figure(),
        ""
    )
    
    analysis = sim.analyze_result()
    
    # Generate comprehensive insights
    full_insights = insights_generator.generate_simulation_insights(sim.history, analysis)
    final_insights_html = generate_insights_html(full_insights)
    
    # Create visualizations
    objection_fig = create_objection_radar(full_insights.get('objection_analysis', {}))
    
    # Save data
    try:
        simulation_id = data_manager.save_simulation(
            target_url=url,
            company_context=context,
            conversation_history=sim.history,
            analysis_result=analysis
        )
        save_status = f"✅ Saved: {simulation_id[:8]}..."
    except Exception as e:
        save_status = f"⚠️ Save failed: {str(e)[:30]}"
    
    # Store in session
    SESSION_DATA.append({
        "URL": url,
        "Analysis": analysis,
        "History": sim.history,
        "Insights": full_insights
    })
    
    yield (
        "✅ Simulation Complete!",
        chat_log,
        analysis,
        final_insights_html,
        sentiment_fig,
        objection_fig,
        save_status
    )


def refresh_analytics_dashboard():
    """Generate comprehensive analytics dashboard."""
    df = data_manager.load_all_metrics()
    
    if df.empty:
        return (
            create_empty_figure("No data yet - Run simulations first!"),
            create_empty_figure(),
            create_empty_figure(),
            create_empty_figure(),
            "No simulations recorded yet."
        )
    
    # Generate insights
    portfolio_insights = insights_generator.generate_portfolio_insights(df)
    
    # Create charts
    main_dashboard = create_comprehensive_dashboard(df, portfolio_insights)
    score_dist = create_score_distribution(df)
    trend_chart = create_performance_trend(df)
    
    # Feature importance from model training
    model_info = portfolio_insights.get('model_training', {})
    if 'feature_importance' in model_info:
        importance_chart = create_feature_importance(model_info['feature_importance'])
    else:
        importance_chart = create_empty_figure("Train model with more data")
    
    # Summary text
    summary = portfolio_insights.get('summary_metrics', {})
    summary_text = f"""
## 📊 Analytics Summary

**Total Simulations:** {summary.get('total_simulations', 0)}  
**Overall Win Rate:** {summary.get('overall_success_rate', 0):.1f}%  
**Average Score:** {summary.get('avg_score', 0):.1f}/10  
**Avg Conversation Length:** {summary.get('avg_conversation_length', 0):.0f} words

### 🎯 Objection Insights
{format_objection_insights(portfolio_insights.get('objection_patterns', {}))}

### 📈 Performance Trend
{format_trend_insights(portfolio_insights.get('performance_trends', {}))}
    """
    
    return main_dashboard, score_dist, trend_chart, importance_chart, summary_text


def format_objection_insights(obj_patterns: dict) -> str:
    """Format objection patterns as markdown."""
    if not obj_patterns or 'error' in obj_patterns:
        return "Not enough data for objection analysis."
    
    hardest = obj_patterns.get('hardest_objection', 'Unknown')
    easiest = obj_patterns.get('easiest_objection', 'Unknown')
    
    text = f"- **Hardest Objection:** {hardest}\n"
    text += f"- **Easiest Objection:** {easiest}\n"
    
    recs = obj_patterns.get('recommendations', [])
    if recs:
        text += "\n**Recommendations:**\n"
        for rec in recs[:3]:  # Top 3
            text += f"- {rec.get('objection', 'N/A')}: {rec.get('strategy', '')[:100]}...\n"
    
    return text


def format_trend_insights(trends: dict) -> str:
    """Format trend data as markdown."""
    trend_type = trends.get('trend', 'unknown')
    emojis = {'improving': '📈', 'declining': '📉', 'stable': '➡️'}
    
    return f"{emojis.get(trend_type, '❓')} Trend: **{trend_type.title()}**"


def get_quick_stats():
    """Get quick statistics for display."""
    stats = data_manager.get_summary_stats()
    
    if stats['total_simulations'] == 0:
        return "0", "0%", "0.0"
    
    return (
        str(stats['total_simulations']),
        f"{stats['success_rate']:.1f}%",
        f"{stats['avg_score']:.1f}"
    )


# ============================================================================
# BUILD THE UI
# ============================================================================

with gr.Blocks(
    css=CUSTOM_CSS,
    theme=gr.themes.Base(
        primary_hue="indigo",
        secondary_hue="purple",
        neutral_hue="slate",
        font=gr.themes.GoogleFont("Inter")
    ),
    title="DeepMost Agentic SDR"
) as demo:
    
    # Header
    gr.Markdown(
        f"""
        <h1 class='main-header'>🚀 DeepMost Agentic SDR</h1>
        <p class='sub-header'>AI-Powered Sales Simulation with Advanced Analytics & Real-Time Coaching</p>
        """,
        elem_classes=["header-container"]
    )
    
    # Quick Stats Row + Provider Badge
    with gr.Row():
        with gr.Column(scale=1):
            total_sims = gr.Textbox(
                value="0",
                label="Total Simulations",
                interactive=False,
                elem_classes=["metric-card"]
            )
        with gr.Column(scale=1):
            win_rate_box = gr.Textbox(
                value="0%",
                label="Win Rate",
                interactive=False,
                elem_classes=["metric-card"]
            )
        with gr.Column(scale=1):
            avg_score_box = gr.Textbox(
                value="0.0",
                label="Avg Score",
                interactive=False,
                elem_classes=["metric-card"]
            )
        with gr.Column(scale=1):
            gr.Textbox(
                value=_provider_label,
                label="⚡ LLM Provider",
                interactive=False,
                elem_classes=["provider-badge"]
            )
    
    # Main Tabs
    with gr.Tabs() as tabs:
        
        # =====================================================================
        # TAB 1: LIVE SIMULATION
        # =====================================================================
        with gr.TabItem("🎯 Live Simulation", id="simulation"):
            with gr.Row():
                # Left Column - Controls & Chat
                with gr.Column(scale=2):
                    with gr.Group(elem_classes=["card-container"]):
                        gr.Markdown("### 🎯 Target Company")
                        url_input = gr.Textbox(
                            label="Company URL",
                            value="https://deepmostai.com",
                            placeholder="Enter target company URL...",
                            elem_id="url-input"
                        )
                        run_btn = gr.Button(
                            "🚀 Start Simulation",
                            variant="primary",
                            elem_classes=["primary-btn"]
                        )
                        status_box = gr.Textbox(
                            label="Status",
                            interactive=False,
                            elem_id="status"
                        )
                    
                    chatbot = gr.Chatbot(
                        label="💬 Live Conversation",
                        height=400,
                        elem_classes=["chatbot-container"]
                    )
                    
                    with gr.Accordion("📝 Raw Analysis", open=False):
                        analysis_box = gr.Textbox(
                            label="AI Analysis",
                            lines=6,
                            interactive=False
                        )
                        save_status_box = gr.Textbox(
                            label="Save Status",
                            interactive=False
                        )
                
                # Right Column - Real-Time Analytics
                with gr.Column(scale=1):
                    with gr.Group(elem_classes=["card-container", "insights-panel"]):
                        gr.Markdown("### 💡 Real-Time Insights")
                        insights_html = gr.HTML(
                            value="<p style='color: #64748b;'>Start a simulation to see insights...</p>"
                        )
                    
                    with gr.Group(elem_classes=["card-container"]):
                        gr.Markdown("### 📈 Sentiment Trajectory")
                        sentiment_plot = gr.Plot(label="")
                    
                    with gr.Group(elem_classes=["card-container"]):
                        gr.Markdown("### 🎯 Objection Radar")
                        objection_plot = gr.Plot(label="")
        
        # =====================================================================
        # TAB 2: ANALYTICS DASHBOARD
        # =====================================================================
        with gr.TabItem("📊 Analytics Dashboard", id="analytics"):
            with gr.Row():
                refresh_btn = gr.Button(
                    "🔄 Refresh Analytics",
                    variant="primary",
                    elem_classes=["primary-btn"]
                )
            
            with gr.Row():
                with gr.Column(scale=2):
                    main_dashboard_plot = gr.Plot(label="Comprehensive Dashboard")
                with gr.Column(scale=1):
                    analytics_summary = gr.Markdown(
                        "Click 'Refresh Analytics' to load data..."
                    )
            
            with gr.Row():
                with gr.Column():
                    score_dist_plot = gr.Plot(label="Score Distribution")
                with gr.Column():
                    trend_plot = gr.Plot(label="Performance Trend")
            
            with gr.Row():
                importance_plot = gr.Plot(label="ML Feature Importance")
        
        # =====================================================================
        # TAB 3: DATA EXPORT
        # =====================================================================
        with gr.TabItem("📁 Data Export", id="export"):
            gr.Markdown("""
            ## 📊 Data Export & Integration
            
            Your simulation data is automatically saved in multiple formats for analysis.
            """)
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("""
                    ### 📂 Available Files
                    
                    | File | Location | Description |
                    |------|----------|-------------|
                    | `simulations_master.csv` | `data/processed/` | High-level simulation summary |
                    | `conversation_turns.csv` | `data/processed/` | Turn-by-turn dialogue data |
                    | `simulation_metrics.csv` | `data/processed/` | ML-ready features |
                    | `<uuid>.json` | `data/raw/conversations/` | Full conversation with context |
                    """)
                
                with gr.Column():
                    gr.Markdown("""
                    ### 🐍 Quick Load (Python)
                    
                    ```python
                    import pandas as pd
                    
                    # Load metrics for ML
                    df = pd.read_csv('data/processed/simulation_metrics.csv')
                    
                    # Load turns for NLP
                    turns = pd.read_csv('data/processed/conversation_turns.csv')
                    
                    # Load single conversation
                    import json
                    with open('data/raw/conversations/<id>.json') as f:
                        conv = json.load(f)
                    ```
                    """)
            
            with gr.Accordion("📈 Sample ML Model", open=False):
                gr.Markdown("""
                ```python
                from sklearn.ensemble import GradientBoostingClassifier
                from sklearn.model_selection import train_test_split
                import pandas as pd
                
                # Load data
                df = pd.read_csv('data/processed/simulation_metrics.csv')
                
                # Features
                feature_cols = ['total_conversation_length', 'word_ratio_seller_buyer',
                               'seller_avg_words_per_turn', 'buyer_avg_words_per_turn']
                
                X = df[feature_cols]
                y = df['outcome_binary']
                
                # Train
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
                
                model = GradientBoostingClassifier(n_estimators=100)
                model.fit(X_train, y_train)
                
                print(f"Accuracy: {model.score(X_test, y_test):.2%}")
                ```
                """)
    
    # =========================================================================
    # EVENT HANDLERS
    # =========================================================================
    
    # Simulation button
    run_btn.click(
        fn=run_simulation_with_analytics,
        inputs=[url_input],
        outputs=[
            status_box,
            chatbot,
            analysis_box,
            insights_html,
            sentiment_plot,
            objection_plot,
            save_status_box
        ]
    ).then(
        fn=get_quick_stats,
        outputs=[total_sims, win_rate_box, avg_score_box]
    )
    
    # Analytics refresh
    refresh_btn.click(
        fn=refresh_analytics_dashboard,
        outputs=[
            main_dashboard_plot,
            score_dist_plot,
            trend_plot,
            importance_plot,
            analytics_summary
        ]
    )
    
    # Load initial stats on page load
    demo.load(
        fn=get_quick_stats,
        outputs=[total_sims, win_rate_box, avg_score_box]
    )


# ============================================================================
# LAUNCH
# ============================================================================
if __name__ == "__main__":
    print(f"[INFO] Starting DeepMost Agentic SDR Premium v2.0 | {_provider_label}...", flush=True)
    print("[INFO] Open http://127.0.0.1:7860 in your browser", flush=True)
    
    try:
        demo.queue().launch(
            server_name="127.0.0.1",
            server_port=7860,
            share=False,
            show_error=True
        )
    except Exception as e:
        print(f"[ERROR] Launch failed: {e}", flush=True)
        demo.launch()