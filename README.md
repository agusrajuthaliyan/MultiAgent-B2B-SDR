# DeepMost Agentic SDR
## AI-Powered Sales Development Representative Simulation System

**🚀 Now with Premium v2.0 - Advanced Analytics & ML-Powered Insights!**

This project uses Multi-Agent AI to simulate B2B sales conversations for training downstream models and analyzing sales strategies.

---

## ✨ What's New in v2.0

### 🧠 Advanced Analytics Engine
- **Win Probability Prediction** - ML model predicts deal success in real-time
- **Sentiment Trajectory Analysis** - Track buyer engagement throughout the call
- **Objection Pattern Detection** - Automatically categorize and analyze objections
- **Feature Importance** - Understand what drives successful outcomes

### 📊 Interactive Dashboards
- **Plotly Visualizations** - Beautiful, interactive charts
- **Real-time Insights Panel** - Live coaching during simulations
- **Performance Trends** - Track improvement over time
- **Comprehensive Analytics** - Gauge charts, sunbursts, and radar plots

### 🎯 Real-Time Coaching
- **Live Suggestions** - Get coaching tips during conversations
- **Urgency Indicators** - Know when to pivot your approach
- **Win Probability Updates** - See predictions update turn-by-turn

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure API Key
Create a `.env` file with your Gemini API key:
```
GEMINI_API_KEY="your-api-key-here"
```

### 3. Run the Application
```bash
# Interactive Web UI
python app.py

# Batch Processing Pipeline
python main.py
```

---

## 📊 Data Collection & Analysis

### Generated Data Files

| File | Location | Description |
|------|----------|-------------|
| `simulations_master.csv` | `data/processed/` | One row per simulation with all metadata |
| `conversation_turns.csv` | `data/processed/` | One row per message for NLP analysis |
| `simulation_metrics.csv` | `data/processed/` | ML-ready features with binary outcomes |
| `<uuid>.json` | `data/raw/conversations/` | Full conversation with context |

### Data Schema

#### simulations_master.csv
| Column | Type | Description |
|--------|------|-------------|
| simulation_id | UUID | Unique identifier |
| timestamp | ISO8601 | When simulation ran |
| target_url | String | Company website |
| num_turns | Int | Number of conversation turns |
| total_seller_words | Int | Total words by seller |
| total_buyer_words | Int | Total words by buyer |
| score | Int (1-10) | AI-assessed call quality |
| outcome | String | Success/Failure |
| key_objection | String | Price/Timing/Authority |
| feedback | String | AI coaching feedback |

#### simulation_metrics.csv (ML-Ready)
| Column | Type | Use Case |
|--------|------|----------|
| outcome_binary | 0/1 | Classification target |
| word_ratio_seller_buyer | Float | Feature engineering |
| total_conversation_length | Int | Regression feature |

---

## 📈 EDA & Modeling Guide

### Loading Data in Python

```python
import pandas as pd
import json

# Load aggregated data
df = pd.read_csv('data/processed/simulations_master.csv')

# Load turn-level data for NLP
turns = pd.read_csv('data/processed/conversation_turns.csv')

# Load ML-ready metrics
metrics = pd.read_csv('data/processed/simulation_metrics.csv')

# Load individual conversation
with open('data/raw/conversations/<simulation_id>.json') as f:
    conv = json.load(f)
```

### Sample EDA Queries

```python
# Success rate by objection type
metrics.groupby('objection_type')['outcome_binary'].mean()  

# Average conversation length by outcome
metrics.groupby('outcome_label')['total_conversation_length'].mean()

# Word ratio analysis
metrics[metrics['outcome_binary']==1]['word_ratio_seller_buyer'].describe()
```

### Sample ML Model

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Features
X = metrics[['total_conversation_length', 'word_ratio_seller_buyer', 
             'seller_avg_words_per_turn', 'buyer_avg_words_per_turn']]
y = metrics['outcome_binary']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)
print(f"Accuracy: {model.score(X_test, y_test):.2f}")
```

---

## 🏗️ Project Structure

```
DeepMost_Agentic_SDR/
├── app.py                      # Gradio Web UI
├── main.py                     # Batch processing pipeline
├── requirements.txt            # Dependencies
├── .env                        # API keys (not in git)
├── README.md                   # This file
├── src/
│   ├── __init__.py
│   ├── agent_logic.py          # Multi-agent simulation engine
│   ├── scraper.py              # Web scraping module
│   └── data_manager.py         # Data persistence layer
├── data/
│   ├── raw/
│   │   └── conversations/      # JSON conversation files
│   └── processed/
│       ├── simulations_master.csv
│       ├── conversation_turns.csv
│       └── simulation_metrics.csv
└── notebooks/                  # Jupyter notebooks for EDA
    └── eda_template.ipynb
```

---

## 🔬 Features

### Multi-Agent System
- **Seller Agent**: Trained on value-based selling techniques
- **Buyer Agent**: Skeptical CTO persona with realistic objections
- **Judge Agent**: Analyzes call quality and provides coaching

### Real-Time Streaming
- Watch conversations unfold in real-time in the web UI
- Turn-by-turn updates with status indicators

### Comprehensive Analytics
- Outcome distribution charts
- Score histograms
- Objection type analysis
- Conversation length correlation

### Data Science Ready
- Structured CSV exports for pandas
- JSON files for NLP/dialogue analysis
- ML-ready features with binary targets
- UUID tracking for reproducibility

---

## 📝 License

This project is for educational/internship purposes.

---

## 🤝 Contributing

Feel free to submit issues and enhancement requests!
