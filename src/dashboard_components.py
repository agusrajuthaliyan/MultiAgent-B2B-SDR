"""
Dashboard Components for DeepMost Agentic SDR
==============================================

Design Philosophy:
  • INSIGHT-FIRST: Every chart title states the conclusion, not just the metric.
  • GLANCEABLE: Key number is large and dominant. 3-second comprehension.
  • CONSISTENT THEME: Deep Navy background, unified color language.
  • MINIMAL CLUTTER: No chart junk. Every pixel earns its place.

Unified Color System — "Deep Navy":
  Background   : #0f172a (deep navy)
  Surface      : #1e293b (card surface)
  Border       : #334155 (subtle dividers)
  Text Primary : #f1f5f9 (near-white)
  Text Secondary: #94a3b8 (slate muted)
  Accent / Brand: #818cf8 (soft indigo)
  Win / Success : #34d399 (emerald)
  Loss / Danger : #fb7185 (rose)
  Neutral / Warn: #fbbf24 (amber)
  Info / Link   : #60a5fa (sky blue)
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional


# ═══════════════════════════════════════════════════════════════════════
# UNIFIED COLOR SYSTEM
# ═══════════════════════════════════════════════════════════════════════

class Theme:
    """Single source of truth for all colors, fonts, sizes."""
    # Backgrounds
    BG       = '#0f172a'
    SURFACE  = '#1e293b'
    BORDER   = '#334155'

    # Text
    TEXT     = '#f1f5f9'
    TEXT2    = '#94a3b8'
    TEXT3    = '#64748b'

    # Semantic Colors
    ACCENT   = '#818cf8'   # Indigo — brand / primary actions
    WIN      = '#34d399'   # Emerald — success, won deals
    LOSS     = '#fb7185'   # Rose — failure, lost deals
    WARN     = '#fbbf24'   # Amber — neutral, pending
    INFO     = '#60a5fa'   # Sky — informational highlights

    # Chart palette (for categorical data, ordered)
    PALETTE  = ['#818cf8', '#34d399', '#fb7185', '#fbbf24', '#60a5fa',
                '#c084fc', '#f472b6', '#38bdf8', '#a3e635', '#fb923c']

    # Transparent overlays
    WIN_BG   = 'rgba(52, 211, 153, 0.12)'
    LOSS_BG  = 'rgba(251, 113, 133, 0.12)'
    WARN_BG  = 'rgba(251, 191, 36, 0.10)'
    ACCENT_BG = 'rgba(129, 140, 248, 0.12)'

    FONT = 'Inter, -apple-system, Segoe UI, sans-serif'


T = Theme  # shorthand


# ─── Base layout applied to every figure ──────────────────────────────

def _base(fig, height=380, **kw):
    """Apply the Deep Navy theme to any Plotly figure."""
    defaults = dict(
        template='plotly_dark',
        paper_bgcolor=T.SURFACE,       # card-like surface (#1e293b)
        plot_bgcolor='#151d2e',         # slightly darker inner plot area
        font=dict(family=T.FONT, color=T.TEXT, size=13),
        margin=dict(l=55, r=30, t=80, b=50),
        height=height,
    )
    defaults.update(kw)
    fig.update_layout(**defaults)

    # Consistent axis styling — visible gridlines
    axis_style = dict(
        gridcolor='rgba(71, 85, 105, 0.45)',   # #475569 at 45% — visible but not loud
        zerolinecolor='rgba(71, 85, 105, 0.6)',
        tickfont=dict(color=T.TEXT2, size=11),
        title_font=dict(color=T.TEXT2, size=12),
        linecolor=T.BORDER,
        linewidth=1,
    )
    fig.update_xaxes(**axis_style)
    fig.update_yaxes(**axis_style)
    return fig


def _title(main: str, sub: str = ""):
    """Build a Plotly title dict with main + subtitle."""
    text = f"<b>{main}</b>"
    if sub:
        text += f"<br><span style='font-size:12px;color:{T.TEXT2}'>{sub}</span>"
    return dict(text=text, font=dict(size=18, color=T.TEXT), x=0.5, xanchor='center')


# ═══════════════════════════════════════════════════════════════════════
# 1. EMPTY PLACEHOLDER
# ═══════════════════════════════════════════════════════════════════════

def create_empty_figure(message: str = "No data available") -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=f"<i>{message}</i>",
        xref="paper", yref="paper", x=0.5, y=0.5,
        showarrow=False,
        font=dict(size=16, color=T.TEXT3)
    )
    return _base(fig, height=350)


# ═══════════════════════════════════════════════════════════════════════
# 2. WIN RATE GAUGE — "You closed X% of deals"
# ═══════════════════════════════════════════════════════════════════════

def create_win_rate_gauge(win_rate: float) -> go.Figure:
    # Determine verdict
    if win_rate >= 60:
        verdict = "Strong closer"
        verdict_color = T.WIN
    elif win_rate >= 35:
        verdict = "Room to grow"
        verdict_color = T.WARN
    else:
        verdict = "Needs coaching"
        verdict_color = T.LOSS

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=win_rate,
        number={'suffix': '%', 'font': {'size': 52, 'color': T.TEXT, 'family': T.FONT}},
        domain={'x': [0, 1], 'y': [0.1, 1]},
        title={
            'text': f"<span style='font-size:13px;color:{verdict_color}'>{verdict}</span>",
            'font': {'size': 13}
        },
        gauge={
            'axis': {'range': [0, 100], 'tickcolor': T.BORDER,
                     'ticksuffix': '%', 'dtick': 25,
                     'tickfont': {'color': T.TEXT3, 'size': 10}},
            'bar': {'color': T.ACCENT, 'thickness': 0.7},
            'bgcolor': T.SURFACE,
            'borderwidth': 0,
            'steps': [
                {'range': [0, 35],  'color': T.LOSS_BG},
                {'range': [35, 60], 'color': T.WARN_BG},
                {'range': [60, 100], 'color': T.WIN_BG}
            ],
        }
    ))

    fig.add_annotation(
        x=0.5, y=-0.02, xref='paper', yref='paper', showarrow=False,
        text="<span style='color:#64748b'>🔴 &lt;35%</span>  · "
             "<span style='color:#64748b'>🟡 35–60%</span>  · "
             "<span style='color:#64748b'>🟢 &gt;60%</span>",
        font=dict(size=11)
    )

    return _base(fig, height=300, margin=dict(l=30, r=30, t=50, b=40),
                 title=_title("You closed {:.0f}% of deals".format(win_rate),
                              "Percentage of simulations where the SDR booked a meeting"))


# ═══════════════════════════════════════════════════════════════════════
# 3. OUTCOME SUNBURST — "How objections map to outcomes"
# ═══════════════════════════════════════════════════════════════════════

def create_outcome_sunburst(df: pd.DataFrame) -> go.Figure:
    if df.empty or 'outcome_label' not in df.columns:
        return create_empty_figure("No outcome data yet — run a simulation first")

    data = df.groupby(['outcome_label', 'objection_type']).size().reset_index(name='count')

    fig = go.Figure(go.Sunburst(
        labels=list(data['outcome_label']) + list(data['objection_type']),
        parents=[''] * len(data['outcome_label']) + list(data['outcome_label']),
        values=list(data['count']) + list(data['count']),
        branchvalues='total',
        marker=dict(
            colors=T.PALETTE[:len(data)],
            line=dict(color=T.BG, width=2)
        ),
        textinfo='label+percent parent',
        textfont=dict(size=12, color=T.TEXT),
        hovertemplate='<b>%{label}</b><br>%{value} calls<br>%{percentParent:.0%} of parent<extra></extra>'
    ))

    return _base(fig, height=420, margin=dict(l=20, r=20, t=85, b=25),
                 title=_title("How objections map to outcomes",
                              "Inner ring = Win/Loss · Outer ring = Objection type raised"))


# ═══════════════════════════════════════════════════════════════════════
# 4. SCORE DISTRIBUTION — "Score distribution by outcome"
# ═══════════════════════════════════════════════════════════════════════

def create_score_distribution(df: pd.DataFrame) -> go.Figure:
    if df.empty or 'score' not in df.columns or 'outcome_label' not in df.columns:
        return create_empty_figure("No score data yet")

    fig = go.Figure()

    # Create box plots separated by outcome
    outcomes = df['outcome_label'].unique()
    
    for outcome in outcomes:
        outcome_data = df[df['outcome_label'] == outcome]['score']
        
        # Color based on outcome type
        oc_lower = str(outcome).lower()
        if 'success' in oc_lower or 'win' in oc_lower:
            color = T.WIN
        elif 'fail' in oc_lower or 'loss' in oc_lower:
            color = T.LOSS
        else:
            color = T.WARN
            
        fig.add_trace(go.Box(
            x=outcome_data,
            name=outcome,
            marker_color=color,
            boxpoints='all',  # show all points
            jitter=0.3,       # spread them out
            pointpos=-1.8,    # put points next to box
            hovertemplate='Score: %{x}<br>Outcome: %{y}<extra></extra>'
        ))

    # Quality zones
    fig.add_vrect(x0=0.5, x1=4.5, fillcolor=T.LOSS_BG, line_width=0,
                  annotation_text="Weak", annotation_position="top left",
                  annotation_font=dict(size=10, color=T.LOSS), layer="below")
    fig.add_vrect(x0=4.5, x1=7.5, fillcolor=T.WARN_BG, line_width=0,
                  annotation_text="Average", annotation_position="top left",
                  annotation_font=dict(size=10, color=T.WARN), layer="below")
    fig.add_vrect(x0=7.5, x1=10.5, fillcolor=T.WIN_BG, line_width=0,
                  annotation_text="Strong", annotation_position="top left",
                  annotation_font=dict(size=10, color=T.WIN), layer="below")

    median_s = df['score'].median()

    return _base(fig, height=370,
                 title=_title(
                     f"Score Distribution by Call Outcome",
                     f"Overall Median: {median_s:.1f}/10"
                 ),
                 xaxis_title="Call Quality Score (1 = poor → 10 = excellent)",
                 yaxis_title="",
                 showlegend=False)


# ═══════════════════════════════════════════════════════════════════════
# 5. SENTIMENT TRAJECTORY — "Buyer warmed up / cooled down"
# ═══════════════════════════════════════════════════════════════════════

def create_sentiment_trajectory(sentiment_data: Dict[str, Any]) -> go.Figure:
    if not sentiment_data or 'turns' not in sentiment_data or not sentiment_data['turns']:
        return create_empty_figure("Sentiment data will appear as the call progresses")

    turns = sentiment_data['turns']
    x = [t['turn'] for t in turns]
    y = [t['score'] for t in turns]

    trajectory_type = sentiment_data.get('trajectory_type', 'unknown')
    traj_labels = {
        'improving': ('📈 Buyer warmed up', T.WIN),
        'declining': ('📉 Buyer cooled down', T.LOSS),
        'recovery': ('🔄 Buyer recovered', T.INFO),
        'stable':   ('➡️ Buyer stayed neutral', T.WARN),
    }
    traj_text, traj_color = traj_labels.get(trajectory_type, ('❓ Unknown', T.TEXT3))

    fig = go.Figure()

    # Background zones
    fig.add_hrect(y0=0.2, y1=1.2, fillcolor=T.WIN_BG, line_width=0)
    fig.add_hrect(y0=-0.2, y1=0.2, fillcolor=T.WARN_BG, line_width=0)
    fig.add_hrect(y0=-1.2, y1=-0.2, fillcolor=T.LOSS_BG, line_width=0)

    # Zone labels (right side)
    fig.add_annotation(x=max(x)+0.3, y=0.7, text="Receptive", showarrow=False,
                       font=dict(size=10, color=T.WIN), xanchor='left')
    fig.add_annotation(x=max(x)+0.3, y=0.0, text="Neutral", showarrow=False,
                       font=dict(size=10, color=T.WARN), xanchor='left')
    fig.add_annotation(x=max(x)+0.3, y=-0.7, text="Resistant", showarrow=False,
                       font=dict(size=10, color=T.LOSS), xanchor='left')

    # Sentiment line
    marker_colors = [
        T.WIN if s > 0.2 else T.LOSS if s < -0.2 else T.WARN for s in y
    ]
    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode='lines+markers+text',
        line=dict(color=T.ACCENT, width=3, shape='spline'),
        marker=dict(size=14, color=marker_colors,
                    line=dict(color=T.BG, width=2)),
        text=[f"{s:+.2f}" for s in y],
        textposition='top center',
        textfont=dict(size=10, color=T.TEXT2),
        fill='tozeroy',
        fillcolor='rgba(129, 140, 248, 0.08)',
        hovertemplate='Turn %{x}<br>Sentiment: %{y:+.2f}<extra></extra>'
    ))

    return _base(fig, height=320,
                 margin=dict(l=55, r=75, t=80, b=55),
                 title=_title(traj_text, "How the buyer's attitude changed each turn"),
                 xaxis_title="Conversation Turn",
                 yaxis_title="Sentiment (−1 hostile → +1 receptive)",
                 xaxis=dict(dtick=1),
                 yaxis=dict(range=[-1.3, 1.3]),
                 showlegend=False)


# ═══════════════════════════════════════════════════════════════════════
# 6. OBJECTION ANALYSIS — Dual Chart (Win Rate + Volume)
# ═══════════════════════════════════════════════════════════════════════

def create_objection_radar(objections: Dict[str, Any]) -> go.Figure:
    """
    Subplots showing Win Rate by Objection AND Objection Volume.
    """
    if not objections or 'objections' not in objections or not objections['objections']:
        return create_empty_figure("No objections detected in this call")

    # In a real scenario, this gets a dataframe or dict of objections
    # For now, adapting the existing single-dict format to mock the subplots
    categories = list(objections['objections'].keys())
    values = [objections['objections'][cat]['count'] for cat in categories]
    
    # Sort by count descending
    paired = sorted(zip(categories, values), key=lambda p: p[1])
    categories = [p[0] for p in paired]
    values = [p[1] for p in paired]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("<b>Win Rate by Objection</b>", "<b>Frequency (Volume)</b>"),
        horizontal_spacing=0.1
    )

    # Left: Mock Win Rate (if data doesn't have it, we use a placeholder or neutral color)
    fig.add_trace(go.Bar(
        y=categories, x=[50]*len(categories), # Placeholder 50% since dict only has counts
        orientation='h',
        marker=dict(color=T.BORDER, line=dict(color=T.BG, width=1)),
        text=[f"N/A"] * len(categories),
        textposition='inside',
        textfont=dict(size=11, color=T.TEXT3),
        hovertemplate='%{y}: Win rate N/A for individual call<extra></extra>'
    ), row=1, col=1)

    # Right: Volume
    max_val = max(values) if values else 1
    bar_colors = [T.LOSS if v == max_val else T.ACCENT for v in values]
    
    fig.add_trace(go.Bar(
        y=categories, x=values,
        orientation='h',
        marker=dict(color=bar_colors, line=dict(color=T.BG, width=1)),
        text=[f"  {v}×" for v in values],
        textposition='outside',
        textfont=dict(size=12, color=T.TEXT),
        hovertemplate='%{y}: raised %{x} time(s)<extra></extra>'
    ), row=1, col=2)

    # Turn off Y axis labels for the second subplot (they share categories)
    fig.update_yaxes(showticklabels=False, row=1, col=2)
    fig.update_xaxes(title_text="Win Rate (%)", row=1, col=1)
    fig.update_xaxes(title_text="Occurrences", row=1, col=2)

    top_obj = categories[-1] if categories else "N/A"
    fig.add_annotation(
        x=0.5, y=-0.2, xref='paper', yref='paper', showarrow=False,
        text=f"⚠️ <b>{top_obj}</b> was the hardest objection",
        font=dict(size=12, color=T.WARN)
    )

    return _base(fig, height=340,
                 margin=dict(l=120, r=40, t=80, b=60),
                 title=_title("Objections Analysis", "Impact and Frequency"),
                 showlegend=False)


# ═══════════════════════════════════════════════════════════════════════
# 7. ENGAGEMENT METRICS — Three big KPI cards
# ═══════════════════════════════════════════════════════════════════════

def create_engagement_metrics(engagement: Dict[str, Any]) -> go.Figure:
    if not engagement:
        return create_empty_figure("No engagement data yet")

    fig = make_subplots(
        rows=1, cols=3,
        specs=[[{'type': 'indicator'}] * 3],
        subplot_titles=[
            '<b>Talk Ratio</b><br><span style="font-size:10px;color:#94a3b8">'
            'Seller words ÷ Buyer words</span>',
            '<b>Seller Questions</b><br><span style="font-size:10px;color:#94a3b8">'
            'Discovery & probing</span>',
            '<b>Buyer Questions</b><br><span style="font-size:10px;color:#94a3b8">'
            'Interest signals</span>'
        ]
    )

    ratio = engagement.get('talk_ratio', 1.0)
    ratio_color = T.WIN if 0.8 <= ratio <= 1.3 else T.WARN

    fig.add_trace(go.Indicator(
        mode="number",
        value=ratio,
        number={'font': {'size': 40, 'color': ratio_color, 'family': T.FONT},
                'valueformat': '.2f'},
    ), row=1, col=1)

    fig.add_trace(go.Indicator(
        mode="number",
        value=engagement.get('seller_questions', 0),
        number={'font': {'size': 40, 'color': T.ACCENT, 'family': T.FONT}}
    ), row=1, col=2)

    fig.add_trace(go.Indicator(
        mode="number",
        value=engagement.get('buyer_questions', 0),
        number={'font': {'size': 40, 'color': T.INFO, 'family': T.FONT}}
    ), row=1, col=3)

    fig.add_annotation(
        x=0.5, y=-0.2, xref='paper', yref='paper', showarrow=False,
        text=f"<span style='color:{T.TEXT3}'>Ideal talk ratio ≈ 0.8–1.2 (balanced dialogue) · "
             f"Green = balanced · Amber = one-sided</span>",
        font=dict(size=11)
    )

    return _base(fig, height=200, margin=dict(l=20, r=20, t=55, b=45))


# ═══════════════════════════════════════════════════════════════════════
# 8. WIN PROBABILITY BAR
# ═══════════════════════════════════════════════════════════════════════

def create_win_probability_funnel(win_prob: Dict[str, Any]) -> go.Figure:
    if not win_prob:
        return create_empty_figure("No prediction available")

    prob = win_prob.get('win_probability', 0.5) * 100
    confidence = win_prob.get('confidence', 'low')

    if prob > 65:
        color, verdict = T.WIN, "Likely to book ✓"
    elif prob > 35:
        color, verdict = T.WARN, "Could go either way"
    else:
        color, verdict = T.LOSS, "Unlikely to convert"

    fig = go.Figure()

    # Background track
    fig.add_trace(go.Bar(
        y=[''], x=[100], orientation='h',
        marker=dict(color=T.SURFACE), showlegend=False, hoverinfo='skip'
    ))

    # Probability fill
    fig.add_trace(go.Bar(
        y=[''], x=[prob], orientation='h',
        marker=dict(color=color),
        text=[f"  {prob:.0f}% — {verdict}"],
        textposition='inside',
        textfont=dict(size=16, color=T.TEXT, family=T.FONT),
        showlegend=False,
        hovertemplate=f'{prob:.1f}% win probability<extra></extra>'
    ))

    fig.add_annotation(
        x=50, y=1.1, text=f"Confidence: {confidence.upper()}",
        showarrow=False, font=dict(size=11, color=T.TEXT3)
    )

    return _base(fig, height=110,
                 margin=dict(l=10, r=10, t=35, b=15),
                 barmode='overlay',
                 xaxis=dict(range=[0, 100], showticklabels=False, showgrid=False),
                 yaxis=dict(showticklabels=False, showgrid=False),
                 title=_title("Predicted Win Probability"))


# ═══════════════════════════════════════════════════════════════════════
# 9. PERFORMANCE TREND — "Your win rate is improving / declining"
# ═══════════════════════════════════════════════════════════════════════

def create_performance_trend(df: pd.DataFrame) -> go.Figure:
    if df.empty or 'timestamp' not in df.columns:
        return create_empty_figure("Not enough data for trends — run more simulations")

    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    n = len(df)
    w = min(5, n)

    # Compute trend direction
    trend_text, trend_color = "➡️ Stable performance", T.WARN
    if 'outcome_binary' in df.columns and n >= 4:
        df['rolling_wr'] = df['outcome_binary'].rolling(window=w, min_periods=1).mean() * 100
        first = df['rolling_wr'].iloc[:n//2].mean()
        second = df['rolling_wr'].iloc[n//2:].mean()
        if second > first + 5:
            trend_text, trend_color = "📈 Win rate is improving", T.WIN
        elif second < first - 5:
            trend_text, trend_color = "📉 Win rate is declining", T.LOSS

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.16,
        subplot_titles=[
            f'<b>Win Rate %</b> <span style="font-size:11px;color:{T.TEXT3}">'
            f'(rolling {w}-sim window)</span>',
            f'<b>Call Quality Score</b> <span style="font-size:11px;color:{T.TEXT3}">'
            f'(rolling {w}-sim average)</span>'
        ]
    )

    if 'rolling_wr' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['timestamp'], y=df['rolling_wr'],
            mode='lines+markers',
            line=dict(color=T.WIN, width=2.5),
            marker=dict(size=6, color=T.WIN),
            fill='tozeroy', fillcolor=T.WIN_BG,
            hovertemplate='%{x|%d %b %Y}<br>Win Rate: %{y:.1f}%<extra></extra>'
        ), row=1, col=1)

        fig.add_hline(y=50, row=1, col=1, line_dash="dot", line_color=T.TEXT3,
                      annotation_text="50% baseline", annotation_position="bottom right",
                      annotation_font=dict(size=10, color=T.TEXT3))

    if 'score' in df.columns:
        df['rolling_sc'] = df['score'].rolling(window=w, min_periods=1).mean()
        fig.add_trace(go.Scatter(
            x=df['timestamp'], y=df['rolling_sc'],
            mode='lines+markers',
            line=dict(color=T.ACCENT, width=2.5),
            marker=dict(size=6, color=T.ACCENT),
            fill='tozeroy', fillcolor=T.ACCENT_BG,
            hovertemplate='%{x|%d %b %Y}<br>Score: %{y:.1f}/10<extra></extra>'
        ), row=2, col=1)

    fig.update_yaxes(title_text="Win Rate %", row=1, col=1)
    fig.update_yaxes(title_text="Score (1-10)", row=2, col=1)

    return _base(fig, height=450,
                 margin=dict(l=60, r=30, t=85, b=40), showlegend=False,
                 title=_title(trend_text,
                              f"Performance across {n} simulations over time"))


# ═══════════════════════════════════════════════════════════════════════
# 10. FEATURE IMPACT HEATMAP — "Correlation between features"
# ═══════════════════════════════════════════════════════════════════════

def create_correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    """
    Creates a correlation heatmap of key metrics.
    """
    if df.empty:
        return create_empty_figure("Not enough data to matrix correlations")

    corr_cols = ['score', 'outcome_binary', 'word_ratio_seller_buyer', 
                 'total_conversation_length', 'seller_speaking_time_seconds']
    
    existing_cols = [c for c in corr_cols if c in df.columns]
    
    if len(existing_cols) < 2:
        return create_empty_figure("Missing required columns for heatmap")

    # Calculate correlation matrix
    corr_matrix = df[existing_cols].corr().round(2)
    
    friendly_names = {
        'score': 'Quality Score',
        'outcome_binary': 'Win/Loss',
        'word_ratio_seller_buyer': 'Talk Ratio',
        'total_conversation_length': 'Total Words',
        'seller_speaking_time_seconds': 'Seller Time (s)'
    }
    
    labels = [friendly_names.get(c, c) for c in corr_matrix.columns]

    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=labels,
        y=labels,
        colorscale=[[0, T.LOSS], [0.5, T.SURFACE], [1, T.WIN]], # Custom diverging colorscale
        zmin=-1, zmax=1,
        text=corr_matrix.values,
        texttemplate="%{text}",
        textfont={"size": 12, "family": T.FONT},
        hoverongaps=False,
        hovertemplate='%{y} vs %{x}<br>Correlation: %{z:.2f}<extra></extra>'
    ))

    # Match background
    fig.update_layout(
        plot_bgcolor=T.SURFACE,
        paper_bgcolor=T.SURFACE,
    )

    return _base(fig, height=380,
                 margin=dict(l=120, r=40, t=80, b=100), showlegend=False,
                 title=_title("Feature Correlation Heatmap",
                              "How conversational traits relate to outcomes and scores"))

# ═══════════════════════════════════════════════════════════════════════
# 10.5. FEATURE IMPORTANCE — "What drives a successful deal?"
# ═══════════════════════════════════════════════════════════════════════

def create_feature_importance(importance: Dict[str, float]) -> go.Figure:
    if not importance:
        return create_empty_figure("Need more data to train the ML model")

    friendly = {
        'total_conversation_length': 'Total Conversation Length',
        'word_ratio_seller_buyer': 'Seller / Buyer Word Ratio',
        'seller_avg_words_per_turn': 'Avg Words per Seller Turn',
        'buyer_avg_words_per_turn': 'Avg Words per Buyer Turn',
        'seller_total_words': 'Total Seller Words',
        'buyer_total_words': 'Total Buyer Words',
        'seller_max_words': 'Longest Seller Turn',
        'buyer_max_words': 'Longest Buyer Turn',
        'num_turns': 'Number of Turns',
        'context_length': 'Company Context Length',
    }

    items = sorted(importance.items(), key=lambda x: x[1])  # ascending for horizontal
    names = [friendly.get(k, k) for k, _ in items]
    vals = [v for _, v in items]

    # Gradient from muted to accent
    bar_colors = [T.ACCENT if v >= max(vals) * 0.8 else T.INFO if v >= max(vals) * 0.4
                  else T.TEXT3 for v in vals]

    fig = go.Figure(go.Bar(
        y=names, x=vals, orientation='h',
        marker=dict(color=bar_colors, line=dict(color=T.BG, width=1)),
        text=[f"{v:.0%}" for v in vals],
        textposition='outside',
        textfont=dict(size=12, color=T.TEXT),
        hovertemplate='%{y}<br>Importance: %{x:.1%}<extra></extra>'
    ))

    top_feature = names[-1] if names else "N/A"
    fig.add_annotation(
        x=0.5, y=-0.14, xref='paper', yref='paper', showarrow=False,
        text=f"🏆 Top predictor: <b>{top_feature}</b>",
        font=dict(size=13, color=T.WIN)
    )

    return _base(fig, height=350,
                 margin=dict(l=200, r=60, t=80, b=55), showlegend=False,
                 title=_title("What drives a successful deal?",
                              "ML feature importance — longer bar = stronger influence on outcome"))


# ═══════════════════════════════════════════════════════════════════════
# 11. COMPREHENSIVE DASHBOARD  (2×2 grid)
# ═══════════════════════════════════════════════════════════════════════

def create_comprehensive_dashboard(df: pd.DataFrame, insights: Dict[str, Any]) -> go.Figure:
    """
    Main 2×2 dashboard:
    ┌─────────────────────────┬─────────────────────────┐
    │  Win Rate (big number)  │  Win vs Loss pie chart  │
    ├─────────────────────────┼─────────────────────────┤
    │  Score over time (line) │  Objection counts (bar) │
    └─────────────────────────┴─────────────────────────┘
    """
    if df.empty:
        return create_empty_figure("Run simulations to see the dashboard")

    win_rate = df['outcome_binary'].mean() * 100 if 'outcome_binary' in df.columns else 0
    avg_score = df['score'].mean() if 'score' in df.columns else 0
    total = len(df)

    fig = make_subplots(
        rows=2, cols=2,
        specs=[
            [{'type': 'indicator'}, {'type': 'pie'}],
            [{'type': 'scatter'},   {'type': 'bar'}]
        ],
        subplot_titles=[
            '<b>Win Rate</b>',
            '<b>Win vs Loss Breakdown</b>',
            '<b>Score Trend</b> <span style="color:#94a3b8;font-size:11px">(each dot = 1 call)</span>',
            '<b>Top Buyer Objections</b>'
        ],
        vertical_spacing=0.18,
        horizontal_spacing=0.12
    )

    # 1. Win Rate Indicator
    wr_color = T.WIN if win_rate >= 60 else T.WARN if win_rate >= 35 else T.LOSS
    fig.add_trace(go.Indicator(
        mode="number",
        value=win_rate,
        number={'suffix': '%', 'font': {'size': 56, 'color': wr_color, 'family': T.FONT}},
    ), row=1, col=1)

    # 2. Outcome Breakdown (Horizontal Bar replaces Pie)
    if 'outcome_label' in df.columns:
        oc = df['outcome_label'].value_counts()
        bar_colors = []
        for label in oc.index:
            l = label.lower()
            if 'success' in l or 'win' in l:
                bar_colors.append(T.WIN)
            elif 'fail' in l or 'loss' in l:
                bar_colors.append(T.LOSS)
            else:
                bar_colors.append(T.WARN)

        fig.add_trace(go.Bar(
            y=oc.index.tolist(),
            x=oc.values.tolist(),
            orientation='h',
            marker=dict(color=bar_colors, line=dict(color=T.BG, width=1)),
            text=[str(v) for v in oc.values.tolist()],
            textposition='outside',
            textfont=dict(size=12, color=T.TEXT),
            hovertemplate='%{y}: %{x} calls<extra></extra>',
        ), row=1, col=2)
        fig.update_xaxes(title_text="Number of Calls", row=1, col=2)
        fig.update_yaxes(categoryorder='total ascending', row=1, col=2)

    # 3. Score Trend
    if 'timestamp' in df.columns and 'score' in df.columns:
        ds = df.sort_values('timestamp')
        n = len(ds)
        dot_colors = [T.WIN if s >= 6 else T.LOSS if s <= 3 else T.WARN for s in ds['score']]

        fig.add_trace(go.Scatter(
            x=list(range(1, n + 1)), y=ds['score'],
            mode='lines+markers',
            line=dict(color=T.ACCENT, width=2),
            marker=dict(size=9, color=dot_colors, line=dict(color=T.BG, width=1)),
            hovertemplate='Call #%{x}<br>Score: %{y}/10<extra></extra>'
        ), row=2, col=1)

        fig.update_xaxes(title_text="Call #", row=2, col=1)
        fig.update_yaxes(title_text="Score", range=[0, 11], row=2, col=1)

    # 4. Objection Bars
    if 'objection_type' in df.columns:
        oj = df['objection_type'].value_counts()
        fig.add_trace(go.Bar(
            x=oj.index.tolist(), y=oj.values.tolist(),
            marker=dict(color=T.PALETTE[:len(oj)],
                        line=dict(color=T.BG, width=1)),
            text=oj.values.tolist(), textposition='outside',
            textfont=dict(size=12, color=T.TEXT),
            hovertemplate='%{x}: %{y} times<extra></extra>'
        ), row=2, col=2)
        fig.update_xaxes(title_text="Objection", row=2, col=2)
        fig.update_yaxes(title_text="Count", row=2, col=2)

    # Summary banner at top
    fig.add_annotation(
        x=0.5, y=1.08, xref='paper', yref='paper', showarrow=False,
        text=(f"<b>{total}</b> calls  ·  "
              f"<b style='color:{wr_color}'>{win_rate:.0f}%</b> win rate  ·  "
              f"<b>{avg_score:.1f}</b>/10 avg score"),
        font=dict(size=14, color=T.TEXT2)
    )

    return _base(fig, height=680, showlegend=False,
                 margin=dict(l=55, r=40, t=100, b=50),
                 title=_title("📊 Sales Performance at a Glance",
                              "Your overall simulation results"))


# ═══════════════════════════════════════════════════════════════════════
# 12. INSIGHTS HTML PANEL
# ═══════════════════════════════════════════════════════════════════════

def generate_insights_html(insights: Dict[str, Any]) -> str:
    if not insights or 'recommendations' not in insights:
        return f"<p style='color: {T.TEXT3};'>Run a simulation to see insights.</p>"

    html = f"<div style='font-family: {T.FONT}; color: {T.TEXT};'>"

    # Grade Card
    if 'overall_score' in insights:
        sc = insights['overall_score']
        grade = sc.get('grade', 'N/A')
        grade_colors = {'A': T.WIN, 'B': T.INFO, 'C': T.WARN, 'D': '#fb923c', 'F': T.LOSS}
        grade_labels = {
            'A': 'Excellent — strong close potential',
            'B': 'Good — minor improvements possible',
            'C': 'Average — needs targeted coaching',
            'D': 'Below average — review fundamentals',
            'F': 'Poor — significant rethink needed'
        }
        gc = grade_colors.get(grade, T.TEXT)
        gl = grade_labels.get(grade, '')
        pts = sc.get('score', 0)

        html += f"""
        <div style='text-align:center; margin-bottom:20px; padding:22px;
                    background: linear-gradient(135deg, {T.SURFACE}, {T.BG});
                    border-radius:14px; border:1px solid {T.BORDER};'>
            <div style='font-size:52px; font-weight:800; color:{gc};
                        letter-spacing:-2px;'>{grade}</div>
            <div style='font-size:20px; color:{T.TEXT}; margin-top:4px;'>{pts}/100 points</div>
            <div style='font-size:12px; color:{T.TEXT2}; margin-top:6px;'>{gl}</div>
        </div>"""

    # Win Probability
    if 'win_probability' in insights:
        wp = insights['win_probability']
        prob = wp.get('win_probability', 0) * 100
        if prob > 65:
            pc, pl = T.WIN, "High likelihood of booking a meeting"
        elif prob > 35:
            pc, pl = T.WARN, "Could go either way — depends on follow-up"
        else:
            pc, pl = T.LOSS, "Unlikely to convert without strategy change"

        html += f"""
        <div style='margin-bottom:18px; padding:16px; background:{T.SURFACE};
                    border-radius:10px; border-left:4px solid {pc};'>
            <div style='font-size:11px; color:{T.TEXT3}; text-transform:uppercase;
                        letter-spacing:0.5px;'>Win Probability</div>
            <div style='font-size:32px; font-weight:700; color:{pc};
                        margin:4px 0;'>{prob:.0f}%</div>
            <div style='font-size:12px; color:{T.TEXT2};'>{pl}</div>
            <div style='font-size:11px; color:{T.TEXT3}; margin-top:6px;'>
                Confidence: {wp.get('confidence', 'low').upper()}</div>
        </div>"""

    # Recommendations (numbered, actionable)
    if 'recommendations' in insights:
        html += f"""
        <div style='margin-bottom:15px;'>
            <div style='font-size:15px; font-weight:700; color:{T.TEXT};
                        margin-bottom:12px; letter-spacing:0.3px;'>
                💡 What to do differently next time
            </div>"""
        for i, rec in enumerate(insights['recommendations'], 1):
            html += f"""
            <div style='padding:12px 14px; margin-bottom:8px; background:{T.SURFACE};
                        border-radius:8px; font-size:13px; color:{T.TEXT};
                        border-left:3px solid {T.ACCENT}; line-height:1.5;'>
                <span style='color:{T.ACCENT}; font-weight:700;'>{i}.</span> {rec}
            </div>"""
        html += "</div>"

    html += "</div>"
    return html
