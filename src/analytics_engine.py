"""
Advanced Analytics Engine for DeepMost Agentic SDR

This module provides ML-powered analytics, predictive modeling,
and actionable insights for sales simulation data.

Features:
- Win probability prediction
- Sentiment trajectory analysis
- Objection pattern clustering
- Conversation optimization recommendations
- Real-time performance scoring
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Optional ML imports with fallbacks
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import classification_report
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("[WARN] scikit-learn not available. Some analytics features disabled.")


class ConversationAnalyzer:
    """
    Analyzes individual conversations for patterns and insights.
    """
    
    def __init__(self):
        self.sentiment_keywords = {
            'positive': ['interested', 'great', 'love', 'perfect', 'amazing', 'yes', 'absolutely', 
                        'definitely', 'sounds good', 'let\'s do it', 'schedule', 'meeting', 'demo'],
            'negative': ['no', 'not interested', 'busy', 'expensive', 'budget', 'later', 'competitor',
                        'already have', 'not now', 'can\'t', 'won\'t', 'don\'t need'],
            'neutral': ['maybe', 'perhaps', 'tell me more', 'what about', 'how does', 'explain']
        }
        
        self.objection_patterns = {
            'Price': ['expensive', 'cost', 'budget', 'afford', 'price', 'money', 'cheap', 'roi'],
            'Timing': ['busy', 'later', 'not now', 'next quarter', 'timing', 'schedule', 'time'],
            'Authority': ['decide', 'boss', 'team', 'stakeholder', 'approval', 'committee', 'check with'],
            'Need': ['don\'t need', 'already have', 'solution', 'happy with', 'current'],
            'Competition': ['competitor', 'alternative', 'other vendor', 'comparing', 'evaluating'],
            'Trust': ['prove', 'case study', 'references', 'guarantee', 'risk']
        }
    
    def analyze_sentiment_trajectory(self, conversation: List[Tuple[str, str]]) -> Dict[str, Any]:
        """
        Analyze how sentiment changes throughout the conversation.
        Returns sentiment scores per turn and trajectory classification.
        """
        trajectory = []
        buyer_messages = [(i, msg) for i, (role, msg) in enumerate(conversation) if role == 'Buyer']
        
        for turn_idx, message in buyer_messages:
            score = self._calculate_sentiment_score(message)
            trajectory.append({
                'turn': turn_idx // 2 + 1,
                'score': score,
                'message_preview': message[:50] + '...' if len(message) > 50 else message
            })
        
        # Calculate trajectory type
        if len(trajectory) >= 2:
            start_score = trajectory[0]['score']
            end_score = trajectory[-1]['score']
            mid_scores = [t['score'] for t in trajectory[1:-1]] if len(trajectory) > 2 else []
            
            if end_score > start_score + 0.3:
                trajectory_type = 'improving'
            elif end_score < start_score - 0.3:
                trajectory_type = 'declining'
            elif mid_scores and min(mid_scores) < start_score - 0.3:
                trajectory_type = 'recovery'
            else:
                trajectory_type = 'stable'
        else:
            trajectory_type = 'insufficient_data'
        
        return {
            'turns': trajectory,
            'trajectory_type': trajectory_type,
            'start_sentiment': trajectory[0]['score'] if trajectory else 0,
            'end_sentiment': trajectory[-1]['score'] if trajectory else 0,
            'avg_sentiment': np.mean([t['score'] for t in trajectory]) if trajectory else 0,
            'volatility': np.std([t['score'] for t in trajectory]) if len(trajectory) > 1 else 0
        }
    
    def _calculate_sentiment_score(self, text: str) -> float:
        """Calculate sentiment score from -1 (negative) to 1 (positive)."""
        text_lower = text.lower()
        
        pos_count = sum(1 for word in self.sentiment_keywords['positive'] if word in text_lower)
        neg_count = sum(1 for word in self.sentiment_keywords['negative'] if word in text_lower)
        neutral_count = sum(1 for word in self.sentiment_keywords['neutral'] if word in text_lower)
        
        total = pos_count + neg_count + neutral_count
        if total == 0:
            return 0.0
        
        return (pos_count - neg_count) / total
    
    def detect_objections(self, conversation: List[Tuple[str, str]]) -> Dict[str, Any]:
        """
        Detect and classify objections raised during the conversation.
        """
        buyer_messages = [msg for role, msg in conversation if role == 'Buyer']
        full_text = ' '.join(buyer_messages).lower()
        
        detected_objections = {}
        for objection_type, keywords in self.objection_patterns.items():
            matches = [kw for kw in keywords if kw in full_text]
            if matches:
                detected_objections[objection_type] = {
                    'count': len(matches),
                    'keywords': matches,
                    'severity': 'high' if len(matches) >= 3 else 'medium' if len(matches) >= 2 else 'low'
                }
        
        # Determine primary objection
        primary_objection = max(detected_objections.items(), 
                                key=lambda x: x[1]['count'])[0] if detected_objections else 'None'
        
        return {
            'objections': detected_objections,
            'primary_objection': primary_objection,
            'total_objections': len(detected_objections),
            'objection_density': sum(o['count'] for o in detected_objections.values()) / max(len(buyer_messages), 1)
        }
    
    def calculate_engagement_metrics(self, conversation: List[Tuple[str, str]]) -> Dict[str, Any]:
        """
        Calculate engagement metrics from the conversation.
        """
        seller_msgs = [msg for role, msg in conversation if role == 'Seller']
        buyer_msgs = [msg for role, msg in conversation if role == 'Buyer']
        
        seller_words = [len(msg.split()) for msg in seller_msgs]
        buyer_words = [len(msg.split()) for msg in buyer_msgs]
        
        # Question count (engagement indicator)
        seller_questions = sum(msg.count('?') for msg in seller_msgs)
        buyer_questions = sum(msg.count('?') for msg in buyer_msgs)
        
        return {
            'talk_ratio': sum(seller_words) / max(sum(buyer_words), 1),
            'buyer_engagement_score': sum(buyer_words) / max(len(buyer_msgs), 1),
            'seller_questions': seller_questions,
            'buyer_questions': buyer_questions,
            'question_ratio': buyer_questions / max(seller_questions, 1),
            'avg_response_length': {
                'seller': np.mean(seller_words) if seller_words else 0,
                'buyer': np.mean(buyer_words) if buyer_words else 0
            },
            'response_length_trend': self._calculate_length_trend(buyer_words)
        }
    
    def _calculate_length_trend(self, word_counts: List[int]) -> str:
        """Determine if response lengths are increasing, decreasing, or stable."""
        if len(word_counts) < 2:
            return 'insufficient_data'
        
        first_half = np.mean(word_counts[:len(word_counts)//2])
        second_half = np.mean(word_counts[len(word_counts)//2:])
        
        if second_half > first_half * 1.2:
            return 'increasing'
        elif second_half < first_half * 0.8:
            return 'decreasing'
        return 'stable'


class PredictiveAnalytics:
    """
    ML-powered predictive analytics for sales outcomes.
    """
    
    def __init__(self):
        self.win_model = None
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        self.is_trained = False
        self.feature_importance = {}
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Prepare features for ML models from the metrics dataframe.
        """
        feature_cols = [
            'context_length', 'num_turns', 
            'seller_total_words', 'buyer_total_words',
            'seller_avg_words_per_turn', 'buyer_avg_words_per_turn',
            'word_ratio_seller_buyer', 'total_conversation_length'
        ]
        
        # Only use columns that exist
        available_cols = [col for col in feature_cols if col in df.columns]
        
        if not available_cols:
            return pd.DataFrame(), []
        
        X = df[available_cols].copy()
        
        # Handle missing values
        X = X.fillna(X.mean())
        
        return X, available_cols
    
    def train_win_predictor(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Train a model to predict win probability.
        """
        if not SKLEARN_AVAILABLE:
            return {'error': 'scikit-learn not available'}
        
        if len(df) < 10:
            return {'error': 'Insufficient data for training (need at least 10 samples)'}
        
        X, feature_cols = self.prepare_features(df)
        if X.empty:
            return {'error': 'No valid features available'}
        
        y = df['outcome_binary']
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.win_model = GradientBoostingClassifier(
            n_estimators=100, 
            max_depth=3, 
            random_state=42
        )
        
        # Cross-validation
        cv_scores = cross_val_score(self.win_model, X_scaled, y, cv=min(5, len(df)))
        
        # Fit on full data
        self.win_model.fit(X_scaled, y)
        self.is_trained = True
        
        # Feature importance
        self.feature_importance = dict(zip(feature_cols, self.win_model.feature_importances_))
        
        return {
            'status': 'trained',
            'cv_accuracy': np.mean(cv_scores),
            'cv_std': np.std(cv_scores),
            'feature_importance': self.feature_importance,
            'samples_used': len(df)
        }
    
    def predict_win_probability(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict win probability for a single conversation.
        """
        if not self.is_trained:
            # Return heuristic-based estimate
            return self._heuristic_win_probability(metrics)
        
        feature_cols = list(self.feature_importance.keys())
        X = pd.DataFrame([{col: metrics.get(col, 0) for col in feature_cols}])
        X_scaled = self.scaler.transform(X)
        
        prob = self.win_model.predict_proba(X_scaled)[0]
        
        return {
            'win_probability': float(prob[1]) if len(prob) > 1 else 0.5,
            'confidence': 'high' if abs(prob[1] - 0.5) > 0.3 else 'medium' if abs(prob[1] - 0.5) > 0.15 else 'low',
            'key_factors': self._get_key_factors(metrics),
            'prediction_method': 'ml_model'
        }
    
    def _heuristic_win_probability(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate win probability using heuristics when ML model isn't available.
        """
        score = 0.5  # Base probability
        
        # Talk ratio impact (optimal around 1.0-1.5)
        talk_ratio = metrics.get('word_ratio_seller_buyer', 1.0)
        if 0.8 <= talk_ratio <= 1.5:
            score += 0.1
        elif talk_ratio > 2.0:
            score -= 0.15  # Talking too much
        
        # Conversation length impact
        conv_length = metrics.get('total_conversation_length', 0)
        if 100 <= conv_length <= 400:
            score += 0.1
        elif conv_length < 50:
            score -= 0.1
        
        # Number of turns
        num_turns = metrics.get('num_turns', 0)
        if num_turns >= 3:
            score += 0.05
        
        # Buyer engagement
        buyer_words = metrics.get('buyer_total_words', 0)
        if buyer_words > 50:
            score += 0.1
        
        return {
            'win_probability': max(0, min(1, score)),
            'confidence': 'low',
            'key_factors': self._get_key_factors(metrics),
            'prediction_method': 'heuristic'
        }
    
    def _get_key_factors(self, metrics: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Identify key factors affecting the prediction.
        """
        factors = []
        
        talk_ratio = metrics.get('word_ratio_seller_buyer', 1.0)
        if talk_ratio > 2.0:
            factors.append({
                'factor': 'Talk Ratio',
                'impact': 'negative',
                'message': 'Seller is talking too much. Let the buyer speak more.'
            })
        elif talk_ratio < 0.5:
            factors.append({
                'factor': 'Talk Ratio',
                'impact': 'positive',
                'message': 'Good buyer engagement. They are actively participating.'
            })
        
        conv_length = metrics.get('total_conversation_length', 0)
        if conv_length < 50:
            factors.append({
                'factor': 'Conversation Length',
                'impact': 'negative',
                'message': 'Conversation too short. Need more discovery.'
            })
        
        num_turns = metrics.get('num_turns', 0)
        if num_turns >= 4:
            factors.append({
                'factor': 'Engagement',
                'impact': 'positive',
                'message': 'Multiple turns indicate buyer interest.'
            })
        
        return factors


class ObjectionClusterer:
    """
    Clusters and analyzes objection patterns across conversations.
    """
    
    def __init__(self):
        self.clusters = {}
        self.cluster_model = None
    
    def analyze_objection_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze objection patterns across all simulations.
        """
        if 'objection_type' not in df.columns or df.empty:
            return {'error': 'No objection data available'}
        
        objection_counts = df['objection_type'].value_counts()
        
        # Calculate success rate by objection type
        objection_success = df.groupby('objection_type')['outcome_binary'].agg(['mean', 'count'])
        objection_success.columns = ['success_rate', 'count']
        objection_success['success_rate'] = objection_success['success_rate'] * 100
        
        # Identify hardest objection to overcome
        hardest = objection_success['success_rate'].idxmin() if not objection_success.empty else 'Unknown'
        easiest = objection_success['success_rate'].idxmax() if not objection_success.empty else 'Unknown'
        
        return {
            'objection_distribution': objection_counts.to_dict(),
            'success_by_objection': objection_success.to_dict('index'),
            'hardest_objection': hardest,
            'easiest_objection': easiest,
            'recommendations': self._generate_objection_recommendations(objection_success)
        }
    
    def _generate_objection_recommendations(self, objection_stats: pd.DataFrame) -> List[Dict[str, str]]:
        """
        Generate recommendations for handling objections.
        """
        recommendations = []
        
        objection_strategies = {
            'Price': 'Focus on ROI and total cost of ownership. Use case studies showing 3x returns.',
            'Timing': 'Create urgency with limited-time offers or highlight cost of delay.',
            'Authority': 'Ask to loop in decision-makers early. Offer to present to the team.',
            'Need': 'Dig deeper into pain points. Share how similar companies benefited.',
            'Competition': 'Differentiate on unique features. Offer comparison demos.',
            'Trust': 'Provide references, case studies, and offer a pilot program.'
        }
        
        for objection_type in objection_stats.index:
            stats = objection_stats.loc[objection_type]
            if stats['success_rate'] < 40:
                recommendations.append({
                    'objection': objection_type,
                    'current_success_rate': f"{stats['success_rate']:.1f}%",
                    'strategy': objection_strategies.get(objection_type, 'Develop specific counter-arguments.')
                })
        
        return recommendations


class InsightsGenerator:
    """
    Generates actionable insights from analytics data.
    """
    
    def __init__(self):
        self.conversation_analyzer = ConversationAnalyzer()
        self.predictive = PredictiveAnalytics()
        self.objection_clusterer = ObjectionClusterer()
    
    def generate_simulation_insights(
        self, 
        conversation: List[Tuple[str, str]], 
        analysis_result: str
    ) -> Dict[str, Any]:
        """
        Generate comprehensive insights for a single simulation.
        """
        # Sentiment analysis
        sentiment = self.conversation_analyzer.analyze_sentiment_trajectory(conversation)
        
        # Objection detection
        objections = self.conversation_analyzer.detect_objections(conversation)
        
        # Engagement metrics
        engagement = self.conversation_analyzer.calculate_engagement_metrics(conversation)
        
        # Build metrics for prediction
        metrics = {
            'num_turns': len(conversation) // 2,
            'total_conversation_length': sum(len(msg.split()) for _, msg in conversation),
            'seller_total_words': sum(len(msg.split()) for role, msg in conversation if role == 'Seller'),
            'buyer_total_words': sum(len(msg.split()) for role, msg in conversation if role == 'Buyer'),
            'word_ratio_seller_buyer': engagement['talk_ratio']
        }
        
        # Win probability
        win_prob = self.predictive.predict_win_probability(metrics)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(sentiment, objections, engagement, win_prob)
        
        return {
            'sentiment_analysis': sentiment,
            'objection_analysis': objections,
            'engagement_metrics': engagement,
            'win_probability': win_prob,
            'recommendations': recommendations,
            'overall_score': self._calculate_overall_score(sentiment, objections, engagement)
        }
    
    def generate_portfolio_insights(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate insights across all simulations.
        """
        if df.empty:
            return {'error': 'No data available for analysis'}
        
        # Train predictive model if enough data
        model_status = self.predictive.train_win_predictor(df)
        
        # Objection patterns
        objection_patterns = self.objection_clusterer.analyze_objection_patterns(df)
        
        # Performance trends
        trends = self._calculate_trends(df)
        
        # Top performers analysis
        top_performers = self._analyze_top_performers(df)
        
        return {
            'model_training': model_status,
            'objection_patterns': objection_patterns,
            'performance_trends': trends,
            'top_performer_insights': top_performers,
            'summary_metrics': {
                'total_simulations': len(df),
                'overall_success_rate': df['outcome_binary'].mean() * 100 if 'outcome_binary' in df.columns else 0,
                'avg_score': df['score'].mean() if 'score' in df.columns else 0,
                'avg_conversation_length': df['total_conversation_length'].mean() if 'total_conversation_length' in df.columns else 0
            }
        }
    
    def _calculate_trends(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate performance trends over time."""
        if 'timestamp' not in df.columns or len(df) < 2:
            return {'trend': 'insufficient_data'}
        
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        # Calculate rolling success rate
        if 'outcome_binary' in df.columns:
            df['rolling_success'] = df['outcome_binary'].rolling(window=min(5, len(df)), min_periods=1).mean()
            
            first_half = df['outcome_binary'][:len(df)//2].mean()
            second_half = df['outcome_binary'][len(df)//2:].mean()
            
            if second_half > first_half + 0.1:
                trend = 'improving'
            elif second_half < first_half - 0.1:
                trend = 'declining'
            else:
                trend = 'stable'
        else:
            trend = 'unknown'
        
        return {
            'trend': trend,
            'first_half_success_rate': first_half * 100 if 'outcome_binary' in df.columns else 0,
            'second_half_success_rate': second_half * 100 if 'outcome_binary' in df.columns else 0
        }
    
    def _analyze_top_performers(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze characteristics of top-performing conversations."""
        if 'score' not in df.columns or df.empty:
            return {}
        
        top_quartile = df[df['score'] >= df['score'].quantile(0.75)]
        bottom_quartile = df[df['score'] <= df['score'].quantile(0.25)]
        
        if top_quartile.empty or bottom_quartile.empty:
            return {}
        
        insights = {}
        
        for col in ['total_conversation_length', 'word_ratio_seller_buyer', 'num_turns']:
            if col in df.columns:
                insights[col] = {
                    'top_performers_avg': top_quartile[col].mean(),
                    'bottom_performers_avg': bottom_quartile[col].mean(),
                    'recommendation': self._get_metric_recommendation(col, top_quartile[col].mean())
                }
        
        return insights
    
    def _get_metric_recommendation(self, metric: str, optimal_value: float) -> str:
        """Get recommendation based on optimal metric values."""
        recommendations = {
            'total_conversation_length': f'Aim for conversations around {optimal_value:.0f} words total.',
            'word_ratio_seller_buyer': f'Target a seller-to-buyer talk ratio of {optimal_value:.2f}.',
            'num_turns': f'Optimal conversations have around {optimal_value:.1f} turns.'
        }
        return recommendations.get(metric, '')
    
    def _generate_recommendations(
        self, 
        sentiment: Dict, 
        objections: Dict, 
        engagement: Dict,
        win_prob: Dict
    ) -> List[str]:
        """Generate actionable recommendations."""
        recs = []
        
        # Sentiment-based
        if sentiment['trajectory_type'] == 'declining':
            recs.append("⚠️ Buyer sentiment declined. Try asking more discovery questions early.")
        elif sentiment['trajectory_type'] == 'improving':
            recs.append("✅ Great momentum! Sentiment improved throughout the call.")
        
        # Objection-based
        if objections['total_objections'] >= 3:
            recs.append(f"🎯 Multiple objections raised. Focus on {objections['primary_objection']} first.")
        
        # Engagement-based
        if engagement['talk_ratio'] > 2.0:
            recs.append("🔇 You talked too much. Let the buyer share more.")
        if engagement['buyer_questions'] == 0:
            recs.append("❓ Buyer asked no questions. Try to spark curiosity.")
        
        # Win probability
        if win_prob['win_probability'] < 0.3:
            recs.append("📉 Low win probability. Consider pivoting approach or qualifying harder.")
        elif win_prob['win_probability'] > 0.7:
            recs.append("🚀 Strong signals! Push for the close.")
        
        if not recs:
            recs.append("📊 Solid conversation. Keep up the good work!")
        
        return recs
    
    def _calculate_overall_score(
        self, 
        sentiment: Dict, 
        objections: Dict, 
        engagement: Dict
    ) -> Dict[str, Any]:
        """Calculate an overall performance score."""
        score = 50  # Base score
        
        # Sentiment contribution (max +20)
        if sentiment['trajectory_type'] == 'improving':
            score += 20
        elif sentiment['trajectory_type'] == 'recovery':
            score += 10
        elif sentiment['trajectory_type'] == 'declining':
            score -= 15
        
        score += sentiment['avg_sentiment'] * 10
        
        # Objection handling (max +15)
        if objections['total_objections'] <= 1:
            score += 15
        elif objections['total_objections'] <= 2:
            score += 5
        else:
            score -= 5
        
        # Engagement contribution (max +15)
        if 0.8 <= engagement['talk_ratio'] <= 1.5:
            score += 15
        elif engagement['talk_ratio'] > 2.5:
            score -= 10
        
        if engagement['response_length_trend'] == 'increasing':
            score += 5
        
        # Normalize to 0-100
        score = max(0, min(100, score))
        
        return {
            'score': round(score),
            'grade': 'A' if score >= 85 else 'B' if score >= 70 else 'C' if score >= 55 else 'D' if score >= 40 else 'F',
            'breakdown': {
                'sentiment': 'positive' if sentiment['avg_sentiment'] > 0.2 else 'neutral' if sentiment['avg_sentiment'] > -0.2 else 'negative',
                'objection_handling': 'excellent' if objections['total_objections'] <= 1 else 'good' if objections['total_objections'] <= 2 else 'needs_work',
                'engagement': 'balanced' if 0.8 <= engagement['talk_ratio'] <= 1.5 else 'imbalanced'
            }
        }


class RealTimeCoach:
    """
    Provides real-time coaching suggestions during conversations.
    """
    
    def __init__(self):
        self.analyzer = ConversationAnalyzer()
    
    def get_live_suggestions(
        self, 
        conversation_so_far: List[Tuple[str, str]],
        current_turn: int
    ) -> Dict[str, Any]:
        """
        Get real-time coaching suggestions based on conversation progress.
        """
        if not conversation_so_far:
            return {
                'suggestion': 'Open with a personalized hook based on their company.',
                'urgency': 'low'
            }
        
        # Analyze current state
        sentiment = self.analyzer.analyze_sentiment_trajectory(conversation_so_far)
        objections = self.analyzer.detect_objections(conversation_so_far)
        engagement = self.analyzer.calculate_engagement_metrics(conversation_so_far)
        
        suggestions = []
        urgency = 'low'
        
        # Check if buyer sentiment is dropping
        if len(sentiment['turns']) >= 2:
            recent_sentiment = sentiment['turns'][-1]['score']
            if recent_sentiment < -0.3:
                suggestions.append("🔴 Buyer is disengaging. Pivot to a discovery question.")
                urgency = 'high'
        
        # Check for unaddressed objections
        if objections['primary_objection'] != 'None':
            suggestions.append(f"⚡ Address the {objections['primary_objection']} objection directly.")
            if urgency != 'high':
                urgency = 'medium'
        
        # Check talk ratio
        if engagement['talk_ratio'] > 1.8:
            suggestions.append("🎤 You've been talking a lot. Ask an open-ended question.")
        
        # Closing suggestion
        if current_turn >= 3 and sentiment['avg_sentiment'] > 0.2:
            suggestions.append("🎯 Good momentum! Consider suggesting next steps.")
        
        if not suggestions:
            suggestions.append("✅ Keep going, conversation is flowing well.")
        
        return {
            'suggestions': suggestions,
            'urgency': urgency,
            'current_sentiment': sentiment['turns'][-1]['score'] if sentiment['turns'] else 0,
            'win_probability_trend': 'up' if sentiment['trajectory_type'] == 'improving' else 'down' if sentiment['trajectory_type'] == 'declining' else 'stable'
        }


# Singleton instances for easy import
insights_generator = InsightsGenerator()
real_time_coach = RealTimeCoach()
conversation_analyzer = ConversationAnalyzer()
predictive_analytics = PredictiveAnalytics()
