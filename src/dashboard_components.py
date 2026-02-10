"""
Dashboard Components for DeepMost Agentic SDR

Premium interactive visualizations using Plotly.
Designed for comprehensive sales analytics.
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional

# Premium color palette
COLORS = {
    'primary': '#6366f1',      # Indigo
    'secondary': '#8b5cf6',    # Purple
    'success': '#10b981',      # Green
    'danger': '#ef4444',       # Red
    'warning': '#f59e0b',      # Amber
    'info': '#3b82f6',         # Blue
    'dark': '#1e1e2e',         # Dark background
    'light': '#f8fafc',        # Light text
    'muted': '#64748b',        # Muted text
    'gradient_start': '#667eea',
    'gradient_end': '#764ba2'
}

GRADIENT_COLORS = ['#667eea', '#764ba2', '#f093fb', '#f5576c', '#4facfe']


def create_empty_figure(message: str = "No data available") -> go.Figure:
    """Create an empty figure with a message."""
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(size=18, color=COLORS['muted'])
    )
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=400
    )
    return fig


def create_win_rate_gauge(win_rate: float) -> go.Figure:
    """Create a premium gauge chart for win rate."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=win_rate,
        number={'suffix': '%', 'font': {'size': 48, 'color': COLORS['light']}},
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Win Rate", 'font': {'size': 24, 'color': COLORS['light']}},
        delta={'reference': 50, 'increasing': {'color': COLORS['success']}, 'decreasing': {'color': COLORS['danger']}},
        gauge={
            'axis': {'range': [0, 100], 'tickcolor': COLORS['muted']},
            'bar': {'color': COLORS['primary']},
            'bgcolor': COLORS['dark'],
            'borderwidth': 2,
            'bordercolor': COLORS['muted'],
            'steps': [
                {'range': [0, 30], 'color': 'rgba(239, 68, 68, 0.3)'},
                {'range': [30, 60], 'color': 'rgba(245, 158, 11, 0.3)'},
                {'range': [60, 100], 'color': 'rgba(16, 185, 129, 0.3)'}
            ],
            'threshold': {
                'line': {'color': COLORS['success'], 'width': 4},
                'thickness': 0.75,
                'value': 75
            }
        }
    ))
    
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=300,
        margin=dict(l=30, r=30, t=50, b=30)
    )
    
    return fig


def create_outcome_sunburst(df: pd.DataFrame) -> go.Figure:
    """Create a sunburst chart showing outcome by objection type."""
    if df.empty or 'outcome_label' not in df.columns:
        return create_empty_figure("No outcome data available")
    
    # Prepare data for sunburst
    sunburst_data = df.groupby(['outcome_label', 'objection_type']).size().reset_index(name='count')
    
    fig = go.Figure(go.Sunburst(
        labels=list(sunburst_data['outcome_label']) + list(sunburst_data['objection_type']),
        parents=[''] * len(sunburst_data['outcome_label']) + list(sunburst_data['outcome_label']),
        values=list(sunburst_data['count']) + list(sunburst_data['count']),
        branchvalues='total',
        marker=dict(
            colors=GRADIENT_COLORS[:len(sunburst_data)],
            line=dict(color=COLORS['dark'], width=2)
        ),
        textinfo='label+percent parent',
        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>%{percentParent:.1%} of parent<extra></extra>'
    ))
    
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        title=dict(text='Outcomes by Objection Type', font=dict(size=18, color=COLORS['light'])),
        height=400,
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    return fig


def create_score_distribution(df: pd.DataFrame) -> go.Figure:
    """Create a beautiful score distribution with violin plot."""
    if df.empty or 'score' not in df.columns:
        return create_empty_figure("No score data available")
    
    fig = go.Figure()
    
    # Add violin plot
    fig.add_trace(go.Violin(
        y=df['score'],
        box_visible=True,
        meanline_visible=True,
        fillcolor=COLORS['primary'],
        opacity=0.6,
        line_color=COLORS['secondary'],
        name='Score Distribution'
    ))
    
    # Add individual points
    fig.add_trace(go.Scatter(
        y=df['score'] + np.random.normal(0, 0.1, len(df)),
        x=np.random.normal(0, 0.05, len(df)),
        mode='markers',
        marker=dict(
            size=8,
            color=df['score'],
            colorscale='Viridis',
            opacity=0.7
        ),
        name='Individual Scores'
    ))
    
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        title=dict(text='Score Distribution', font=dict(size=18, color=COLORS['light'])),
        yaxis_title='Score (1-10)',
        showlegend=False,
        height=350,
        margin=dict(l=50, r=30, t=60, b=30)
    )
    
    return fig


def create_sentiment_trajectory(sentiment_data: Dict[str, Any]) -> go.Figure:
    """Create a sentiment trajectory line chart."""
    if not sentiment_data or 'turns' not in sentiment_data or not sentiment_data['turns']:
        return create_empty_figure("No sentiment data available")
    
    turns = sentiment_data['turns']
    
    fig = go.Figure()
    
    # Sentiment line
    x = [t['turn'] for t in turns]
    y = [t['score'] for t in turns]
    
    # Create gradient fill
    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        mode='lines+markers',
        name='Sentiment',
        line=dict(color=COLORS['primary'], width=3, shape='spline'),
        marker=dict(
            size=12,
            color=[COLORS['success'] if s > 0.2 else COLORS['danger'] if s < -0.2 else COLORS['warning'] for s in y],
            line=dict(color=COLORS['light'], width=2)
        ),
        fill='tozeroy',
        fillcolor='rgba(99, 102, 241, 0.2)'
    ))
    
    # Add zero line
    fig.add_hline(y=0, line_dash="dash", line_color=COLORS['muted'], opacity=0.5)
    
    # Add trajectory annotation
    trajectory_type = sentiment_data.get('trajectory_type', 'unknown')
    trajectory_emoji = {'improving': '📈', 'declining': '📉', 'recovery': '🔄', 'stable': '➡️'}.get(trajectory_type, '❓')
    
    fig.add_annotation(
        x=max(x), y=max(y),
        text=f"{trajectory_emoji} {trajectory_type.title()}",
        showarrow=False,
        font=dict(size=14, color=COLORS['light']),
        bgcolor=COLORS['dark'],
        bordercolor=COLORS['primary'],
        borderwidth=2,
        borderpad=4
    )
    
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        title=dict(text='Sentiment Trajectory', font=dict(size=18, color=COLORS['light'])),
        xaxis_title='Turn',
        yaxis_title='Sentiment Score',
        yaxis=dict(range=[-1.2, 1.2]),
        height=300,
        margin=dict(l=50, r=30, t=60, b=50)
    )
    
    return fig


def create_objection_radar(objections: Dict[str, Any]) -> go.Figure:
    """Create a radar chart for objection analysis."""
    if not objections or 'objections' not in objections or not objections['objections']:
        return create_empty_figure("No objections detected")
    
    categories = list(objections['objections'].keys())
    values = [objections['objections'][cat]['count'] for cat in categories]
    
    # Close the radar
    categories = categories + [categories[0]]
    values = values + [values[0]]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        fillcolor='rgba(139, 92, 246, 0.3)',
        line=dict(color=COLORS['secondary'], width=2),
        marker=dict(size=8, color=COLORS['secondary'])
    ))
    
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max(values) + 1],
                gridcolor=COLORS['muted'],
                tickfont=dict(color=COLORS['muted'])
            ),
            angularaxis=dict(
                gridcolor=COLORS['muted'],
                tickfont=dict(color=COLORS['light'], size=12)
            ),
            bgcolor='rgba(0,0,0,0)'
        ),
        title=dict(text='Objection Analysis', font=dict(size=18, color=COLORS['light'])),
        height=350,
        margin=dict(l=60, r=60, t=60, b=40)
    )
    
    return fig


def create_engagement_metrics(engagement: Dict[str, Any]) -> go.Figure:
    """Create a multi-metric engagement visualization."""
    if not engagement:
        return create_empty_figure("No engagement data available")
    
    fig = make_subplots(
        rows=1, cols=3,
        specs=[[{'type': 'indicator'}, {'type': 'indicator'}, {'type': 'indicator'}]],
        subplot_titles=['Talk Ratio', 'Seller Questions', 'Buyer Questions']
    )
    
    # Talk Ratio
    talk_ratio = engagement.get('talk_ratio', 1.0)
    fig.add_trace(go.Indicator(
        mode="number+delta",
        value=talk_ratio,
        number={'font': {'size': 36, 'color': COLORS['light']}, 'valueformat': '.2f'},
        delta={'reference': 1.0, 'relative': True},
        domain={'row': 0, 'column': 0}
    ), row=1, col=1)
    
    # Seller Questions
    fig.add_trace(go.Indicator(
        mode="number",
        value=engagement.get('seller_questions', 0),
        number={'font': {'size': 36, 'color': COLORS['primary']}}
    ), row=1, col=2)
    
    # Buyer Questions
    fig.add_trace(go.Indicator(
        mode="number",
        value=engagement.get('buyer_questions', 0),
        number={'font': {'size': 36, 'color': COLORS['secondary']}}
    ), row=1, col=3)
    
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=200,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig


def create_win_probability_funnel(win_prob: Dict[str, Any]) -> go.Figure:
    """Create a visual win probability indicator."""
    if not win_prob:
        return create_empty_figure("No prediction available")
    
    prob = win_prob.get('win_probability', 0.5) * 100
    confidence = win_prob.get('confidence', 'low')
    
    # Create a horizontal bar
    fig = go.Figure()
    
    # Background bar
    fig.add_trace(go.Bar(
        y=['Win Probability'],
        x=[100],
        orientation='h',
        marker=dict(color='rgba(100, 100, 100, 0.3)'),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    # Probability bar
    color = COLORS['success'] if prob > 65 else COLORS['warning'] if prob > 35 else COLORS['danger']
    
    fig.add_trace(go.Bar(
        y=['Win Probability'],
        x=[prob],
        orientation='h',
        marker=dict(
            color=color,
            line=dict(color=COLORS['light'], width=1)
        ),
        text=[f"{prob:.1f}%"],
        textposition='inside',
        textfont=dict(size=20, color=COLORS['light']),
        showlegend=False
    ))
    
    # Add confidence indicator
    confidence_emoji = {'high': '🎯', 'medium': '📊', 'low': '❓'}.get(confidence, '❓')
    
    fig.add_annotation(
        x=50, y=1.2,
        text=f"Confidence: {confidence_emoji} {confidence.title()}",
        showarrow=False,
        font=dict(size=14, color=COLORS['muted'])
    )
    
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        barmode='overlay',
        xaxis=dict(range=[0, 100], showticklabels=False, showgrid=False),
        yaxis=dict(showticklabels=False),
        height=120,
        margin=dict(l=20, r=20, t=40, b=20),
        title=dict(text='Win Probability', font=dict(size=16, color=COLORS['light']))
    )
    
    return fig


def create_performance_trend(df: pd.DataFrame) -> go.Figure:
    """Create a performance trend chart over time."""
    if df.empty or 'timestamp' not in df.columns:
        return create_empty_figure("No trend data available")
    
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')
    
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=['Success Rate (Rolling)', 'Average Score (Rolling)']
    )
    
    # Calculate rolling metrics
    window = min(5, len(df))
    if 'outcome_binary' in df.columns:
        df['rolling_success'] = df['outcome_binary'].rolling(window=window, min_periods=1).mean() * 100
        
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['rolling_success'],
            mode='lines+markers',
            name='Success Rate',
            line=dict(color=COLORS['success'], width=2),
            fill='tozeroy',
            fillcolor='rgba(16, 185, 129, 0.2)'
        ), row=1, col=1)
    
    if 'score' in df.columns:
        df['rolling_score'] = df['score'].rolling(window=window, min_periods=1).mean()
        
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['rolling_score'],
            mode='lines+markers',
            name='Avg Score',
            line=dict(color=COLORS['primary'], width=2),
            fill='tozeroy',
            fillcolor='rgba(99, 102, 241, 0.2)'
        ), row=2, col=1)
    
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=400,
        margin=dict(l=50, r=30, t=60, b=30),
        showlegend=False
    )
    
    fig.update_yaxes(title_text="Rate %", row=1, col=1)
    fig.update_yaxes(title_text="Score", row=2, col=1)
    
    return fig


def create_feature_importance(importance: Dict[str, float]) -> go.Figure:
    """Create a feature importance bar chart."""
    if not importance:
        return create_empty_figure("No feature importance data")
    
    # Sort by importance
    sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    features = [f[0] for f in sorted_features]
    values = [f[1] for f in sorted_features]
    
    fig = go.Figure(go.Bar(
        x=values,
        y=features,
        orientation='h',
        marker=dict(
            color=values,
            colorscale='Viridis',
            line=dict(color=COLORS['light'], width=1)
        ),
        text=[f"{v:.2%}" for v in values],
        textposition='outside',
        textfont=dict(color=COLORS['light'])
    ))
    
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        title=dict(text='Feature Importance (Win Prediction)', font=dict(size=18, color=COLORS['light'])),
        xaxis_title='Importance',
        height=300,
        margin=dict(l=150, r=50, t=60, b=40)
    )
    
    return fig


def create_comprehensive_dashboard(df: pd.DataFrame, insights: Dict[str, Any]) -> go.Figure:
    """Create a comprehensive multi-chart dashboard."""
    if df.empty:
        return create_empty_figure("Run simulations to see analytics")
    
    fig = make_subplots(
        rows=2, cols=2,
        specs=[
            [{'type': 'indicator'}, {'type': 'pie'}],
            [{'type': 'scatter'}, {'type': 'bar'}]
        ],
        subplot_titles=['Win Rate', 'Outcome Distribution', 'Score Trend', 'Objection Types']
    )
    
    # Win Rate Gauge
    win_rate = df['outcome_binary'].mean() * 100 if 'outcome_binary' in df.columns else 0
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=win_rate,
        number={'suffix': '%', 'font': {'size': 32}},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': COLORS['primary']},
            'steps': [
                {'range': [0, 40], 'color': 'rgba(239, 68, 68, 0.3)'},
                {'range': [40, 70], 'color': 'rgba(245, 158, 11, 0.3)'},
                {'range': [70, 100], 'color': 'rgba(16, 185, 129, 0.3)'}
            ]
        }
    ), row=1, col=1)
    
    # Outcome Distribution Pie
    if 'outcome_label' in df.columns:
        outcome_counts = df['outcome_label'].value_counts()
        fig.add_trace(go.Pie(
            labels=outcome_counts.index.tolist(),
            values=outcome_counts.values.tolist(),
            marker=dict(colors=[COLORS['success'], COLORS['danger'], COLORS['warning']]),
            textinfo='percent+label'
        ), row=1, col=2)
    
    # Score Trend
    if 'timestamp' in df.columns and 'score' in df.columns:
        df_sorted = df.sort_values('timestamp')
        fig.add_trace(go.Scatter(
            x=list(range(len(df_sorted))),
            y=df_sorted['score'],
            mode='lines+markers',
            line=dict(color=COLORS['primary']),
            marker=dict(size=8)
        ), row=2, col=1)
    
    # Objection Bar
    if 'objection_type' in df.columns:
        obj_counts = df['objection_type'].value_counts()
        fig.add_trace(go.Bar(
            x=obj_counts.index.tolist(),
            y=obj_counts.values.tolist(),
            marker=dict(color=GRADIENT_COLORS[:len(obj_counts)])
        ), row=2, col=2)
    
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=600,
        showlegend=False,
        margin=dict(l=40, r=40, t=80, b=40),
        title=dict(
            text='📊 Sales Performance Dashboard',
            font=dict(size=24, color=COLORS['light']),
            x=0.5
        )
    )
    
    return fig


def generate_insights_html(insights: Dict[str, Any]) -> str:
    """Generate HTML for insights display."""
    if not insights or 'recommendations' not in insights:
        return "<p style='color: #64748b;'>Run a simulation to see insights.</p>"
    
    html = """
    <div style='font-family: Inter, sans-serif; color: #f8fafc;'>
    """
    
    # Overall Score
    if 'overall_score' in insights:
        score = insights['overall_score']
        grade_colors = {'A': '#10b981', 'B': '#3b82f6', 'C': '#f59e0b', 'D': '#f97316', 'F': '#ef4444'}
        grade = score.get('grade', 'N/A')
        
        html += f"""
        <div style='text-align: center; margin-bottom: 20px; padding: 20px; 
                    background: linear-gradient(135deg, rgba(99,102,241,0.2), rgba(139,92,246,0.2));
                    border-radius: 12px; border: 1px solid rgba(139,92,246,0.3);'>
            <div style='font-size: 48px; font-weight: bold; color: {grade_colors.get(grade, "#fff")};'>
                {grade}
            </div>
            <div style='font-size: 24px; color: #f8fafc;'>Score: {score.get('score', 0)}/100</div>
        </div>
        """
    
    # Win Probability
    if 'win_probability' in insights:
        wp = insights['win_probability']
        prob = wp.get('win_probability', 0) * 100
        prob_color = '#10b981' if prob > 65 else '#f59e0b' if prob > 35 else '#ef4444'
        
        html += f"""
        <div style='margin-bottom: 20px; padding: 15px; background: rgba(30,30,46,0.8); 
                    border-radius: 8px; border-left: 4px solid {prob_color};'>
            <div style='font-size: 14px; color: #64748b;'>Win Probability</div>
            <div style='font-size: 28px; font-weight: bold; color: {prob_color};'>{prob:.1f}%</div>
            <div style='font-size: 12px; color: #64748b;'>Confidence: {wp.get('confidence', 'low').title()}</div>
        </div>
        """
    
    # Recommendations
    if 'recommendations' in insights:
        html += """
        <div style='margin-bottom: 15px;'>
            <div style='font-size: 16px; font-weight: 600; color: #f8fafc; margin-bottom: 10px;'>
                💡 Recommendations
            </div>
        """
        for rec in insights['recommendations']:
            html += f"""
            <div style='padding: 10px; margin-bottom: 8px; background: rgba(99,102,241,0.1); 
                        border-radius: 6px; font-size: 14px; color: #e2e8f0;'>
                {rec}
            </div>
            """
        html += "</div>"
    
    html += "</div>"
    return html
