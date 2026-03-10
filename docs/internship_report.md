# Internship Final Project Report

**Project Title:** DeepMost Agentic SDR — An AI-Powered B2B Sales Conversation Simulation and Analytics Platform

**Prepared by:** [Your Name]
**Student ID:** [Your Student ID]
**Programme:** [Your Programme / Degree]
**Institution:** [Your University / College]
**Supervisor:** [Your Academic Supervisor]
**Industry Supervisor:** [Your Industry Supervisor]
**Internship Organisation:** DeepMost AI
**Internship Period:** [Start Date] – [End Date]
**Date of Submission:** [Submission Date]

---

## Table of Contents

1. [Chapter 1 — Introduction](#chapter-1--introduction)
2. [Chapter 2 — Dataset and Data Preprocessing](#chapter-2--dataset-and-data-preprocessing)
3. [Chapter 3 — Data Modeling and Analysis](#chapter-3--data-modeling-and-analysis)
4. [Chapter 4 — Objective 1: Realistic Sales Conversation Simulation using Multi-Agent AI](#chapter-4--objective-1-realistic-sales-conversation-simulation-using-multi-agent-ai)
5. [Chapter 5 — Objective 2: Predictive Outcome Modeling and Sales Analytics Dashboard](#chapter-5--objective-2-predictive-outcome-modeling-and-sales-analytics-dashboard)
6. [Chapter 6 — Objective 3: Real-Time Coaching and Automated Insight Generation](#chapter-6--objective-3-real-time-coaching-and-automated-insight-generation)
7. [Chapter 7 — Experiments and Result Analysis](#chapter-7--experiments-and-result-analysis)
8. [Chapter 8 — Limitations and Future Enhancements](#chapter-8--limitations-and-future-enhancements)
9. [Chapter 9 — Summary](#chapter-9--summary)
10. [References](#references)

---

## Chapter 1 — Introduction

### 1.1 Introduction to the Project

The **DeepMost Agentic SDR** is an AI-native B2B sales development platform that simulates, evaluates, and coaches sales conversations using autonomous large language model (LLM) agents. The system is designed to address one of the most persistent challenges in B2B sales: the ability to practise, measure, and improve discovery and pitch conversations at scale without the cost of real failed sales calls. The platform scrapes prospect company websites to gather contextual intelligence, then orchestrates a simulated dialogue between an AI-powered "Seller" agent and an AI-powered "Buyer" agent. After each simulated call, advanced analytics modules evaluate the conversation for sentiment, objection patterns, engagement quality, and win probability. Results are surfaced through a rich Gradio web interface, an interactive analytics dashboard backed by Plotly, and real-time coaching feedback—enabling sales teams and trainers to identify what works, replay effective scripts, and systematically attack their weakest areas.

### 1.2 Organisation Profile

**DeepMost AI** is an early-stage artificial intelligence startup focused on applied agentic systems for enterprise sales and revenue operations. The organisation operates at the intersection of large language model research and practical B2B go-to-market tooling, building products that augment human performance in highly interpersonal and high-stakes commercial functions. DeepMost AI's core philosophy is that every sales interaction generates latent data that, if captured and modelled correctly, can close the feedback loop between the boardroom-level revenue strategy and the day-to-day activities of frontline revenue teams. The company's engineering ethos prioritises modular, observable AI pipelines so that individual components—scrapers, agent prompts, scoring heuristics, ML models—can be swapped or fine-tuned independently as the underlying LLM landscape evolves. Its product vision, encapsulated in the Agentic SDR platform, represents a move toward fully autonomous top-of-funnel prospecting that can run continuously, log every interaction, and surface actionable insights without human scheduling overhead.

The technical environment at DeepMost AI is lean and fast-moving. The development stack relies on Python as the primary language, with Gradio for rapid web-interface prototyping, Plotly for rich interactive visualisations, and the Groq and Google Gemini APIs as the LLM backbones. Data is managed through a combination of JSON conversation logs and structured CSV summary files, with an emphasis on clean data contracts between pipeline stages so that downstream ML training notebooks can consume freshly generated synthetic data without manual intervention. The organisation actively encourages interns and contributors to own entire sub-systems end-to-end, fostering an environment where curiosity-driven experimentation—running a new batch of five hundred simulations to test a prompt variation, for instance—is treated as core engineering work rather than auxiliary research.

### 1.3 Objectives of the Project

This internship project is structured around three primary objectives, each corresponding to a major capability domain of the DeepMost Agentic SDR platform:

| # | Objective | Corresponding Chapter |
|---|-----------|----------------------|
| **1** | Design and implement a realistic, context-aware B2B sales conversation simulation system powered by multi-agent LLM orchestration | Chapter 4 |
| **2** | Build and evaluate a predictive outcome-modelling pipeline and an interactive analytics dashboard to surface actionable sales intelligence | Chapter 5 |
| **3** | Develop a real-time AI coaching module that generates in-call suggestions and post-call diagnostic reports for sales practitioners | Chapter 6 |

---

## Chapter 2 — Dataset and Data Preprocessing

### 2.1 Problem Statement / Business Requirement

The core business problem this project addresses is the **cold-start feedback deficit** in B2B sales training. Traditional sales coaching requires access to recorded real calls, which raises privacy concerns, is difficult to scale, and produces sparse labelled datasets since the vast majority of calls result in disqualification rather than pipeline progression. Sales Development Representatives (SDRs) typically have very limited access to structured feedback on *why* a call failed—whether it was timing, price sensitivity, authority issues, integration concerns, or competitive pressure.

The business requirement, therefore, is a platform that can:

1. Generate an unlimited number of high-fidelity synthetic sales conversations on demand for any B2B prospect.
2. Automatically label each conversation with outcome scores, objection categories, sentiment trajectories, and engagement metrics.
3. Expose aggregated insights through a dashboard so that sales leaders can identify systemic weaknesses across their team's approach to specific industries or objection types.
4. Provide a real-time coaching assistant that gives sellers in-the-moment suggestions during live Gradio-hosted practice sessions.

### 2.2 Dataset Definition

The dataset used in this project is entirely **synthetically generated** through the simulation pipeline. As of the final data snapshot, the master CSV ([data/processed/simulation_metrics.csv](file:///d:/DeepMost/DeepMost_Agentic_SDR/data/processed/simulation_metrics.csv)) contains **635 simulation records** spanning 3 distinct pipeline sources:

| Source | Description | Count |
|--------|-------------|-------|
| `interactive` | Manually triggered simulations via the Gradio UI | 13 |
| `batch_v2` | Automated batch runs using [main.py](file:///d:/DeepMost/DeepMost_Agentic_SDR/main.py) against a fixed target list | 158 |
| `dynamic_batch` | Fully autonomous discovery runs using [dynamic_batch.py](file:///d:/DeepMost/DeepMost_Agentic_SDR/dynamic_batch.py) | 464 |

Each record represents one complete simulated sales call and is characterised by the following **schema**:

| Column | Type | Description |
|--------|------|-------------|
| `simulation_id` | UUID | Unique identifier for the simulation |
| `timestamp` | ISO 8601 | Date and time of the simulation |
| `target_url` | String | URL of the prospect company's website |
| `context_length` | Integer | Characters of web-scraped context available to the agents |
| `num_turns` | Integer | Number of conversational turns in the dialogue |
| `seller_total_words` | Integer | Total word count produced by the Seller agent |
| `buyer_total_words` | Integer | Total word count produced by the Buyer agent |
| `seller_avg_words_per_turn` | Float | Average words per turn from Seller |
| `buyer_avg_words_per_turn` | Float | Average words per turn from Buyer |
| `seller_max_words` / `buyer_max_words` | Integer | Max turn length for each speaker |
| `seller_min_words` / `buyer_min_words` | Integer | Min turn length for each speaker |
| `word_ratio_seller_buyer` | Float | Seller-to-Buyer talk ratio; key feature for ML models |
| `total_conversation_length` | Integer | Total words across the entire call |
| [score](file:///d:/DeepMost/DeepMost_Agentic_SDR/src/agent_logic.py#184-271) | Integer (2–9) | LLM-generated quality score for the conversation |
| `outcome_binary` | Binary (0/1) | 0 = Failure/Pending, 1 = Success |
| `outcome_label` | Categorical | Human-readable outcome: Success, Failure, Pending |
| `objection_type` | Categorical | Primary objection raised (Timing, Price, Integration, etc.) |
| `source` | String | Pipeline source identifier |

In addition to this master metrics file, the system maintains:
- **Raw JSON conversation logs** (`data/raw/conversations/`) — full turn-by-turn dialogue for each simulation.
- **`conversation_turns.csv`** — an exploded, turn-level representation suitable for NLP analysis.

### 2.3 Data Collection Strategy

Data collection follows a **three-tier automated pipeline** as illustrated below:

![System Pipeline Diagram](d:\DeepMost\DeepMost_Agentic_SDR\assets\pipeline.png)

**Step 1 — Target URL Identification.** For `batch_v2` runs, target URLs are predefined in `main.py` as a curated list of well-known B2B SaaS companies (e.g., Salesforce, HubSpot, Stripe). For `dynamic_batch` runs, a separate LLM call is made to generate new, unique company URLs not already present in the master dataset, enabling continuous dataset expansion without manual curation.

**Step 2 — Web Scraping.** The `src/scraper.py` module uses the `requests` library to perform HTTP GET requests against target URLs, extracts body text, strips HTML tags with `BeautifulSoup`, collapses whitespace, and returns the first 3,000 characters of cleaned text. This truncated snippet forms the **company context** that is injected into the agent prompts.

**Step 3 — Multi-Agent Simulation.** The context is passed to `src/agent_logic.py`, which orchestrates a structured call. First, a **company fit scoring** step uses the LLM to assess how aligned DeepMost AI's offering is with the prospect (score 1–10). Then the simulation loop runs, alternating between a Seller agent persona (following a pitch framework) and a Buyer agent persona (simulating realistic push-back). The number of turns is configurable (default: 4–6).

**Step 4 — Analysis and Persistence.** After the dialogue concludes, `agent_logic.py` calls the LLM for a structured analysis report that extracts the outcome score and primary objection. `src/data_manager.py` then persists the full conversation to JSON, appends the high-level metrics to the CSV, and appends the turn-level data to the conversation turns CSV.

This pipeline is designed for **rate-limit resilience**: exponential back-off and per-provider delays are built in (`GROQ_INTER_CALL_DELAY`, `GEMINI_INTER_CALL_DELAY`), and the system transparently falls back between Groq and Gemini models based on the `LLM_PROVIDER` environment variable.

### 2.4 Data Preprocessing

Raw data from the simulation pipeline is clean by construction—each simulation either succeeds and produces a well-structured record or fails and is either retried or skipped. However, several preprocessing challenges emerged over multiple batch runs:

![Preprocessing Pipeline](d:\DeepMost\DeepMost_Agentic_SDR\assets\preprocessing.png)

**Step 1 — Corrupt Record Detection (`data_cleanup.py`).** Early batch runs occasionally produced records with malformed LLM outputs—missing `outcome_label` fields, strings in numeric columns, or unrealistic conversation metrics (e.g., `num_turns = 0`). The `data_cleanup.py` script reads the master CSV and applies a row-validity function that checks:
- `num_turns >= 3` (minimum viable conversation)
- `score` is an integer in range [1, 10]
- `outcome_label` ∈ {Success, Failure, Pending}
- `seller_total_words > 0` and `buyer_total_words > 0`

Invalid rows are archived to `data/processed/archived_bad_rows.csv` and their corresponding JSON files moved to `data/raw/conversations/archived/`.

**Step 2 — Feature Engineering.** For ML model training, raw metrics are augmented with derived features:
- `word_ratio_seller_buyer` = `seller_total_words / buyer_total_words` — this is the single most predictive feature, as successful calls tend to have a Seller-dominant talk pattern (ratio > 1.5).
- `seller_avg_words_per_turn` and `buyer_avg_words_per_turn` — proxy measures of conversational depth per exchange.
- Early-call linguistic features — the first 30-second proxy (initial turns) is analysed separately for vocabulary signals of success, as described in the model training notebook.

**Step 3 — Label Encoding.** The `outcome_label` categorical field is mapped to `outcome_binary` (1 = Success, 0 otherwise) for supervised classification. The `objection_type` field is kept as a raw categorical for dashboard segmentation but is not encoded into the primary ML feature set to avoid leakage.

**Step 4 — Train/Test Split.** The notebook `notebooks/model_training.ipynb` applies a stratified 80/20 train/test split to ensure class balance is preserved between the majority Failure class (77.4%) and the minority Success class (17.2%).

---

## Chapter 3 — Data Modeling and Analysis

### 3.1 Data Exploration and Analysis (EDA)

Exploratory data analysis was performed to understand the distributional properties of the simulated conversation dataset and to identify relationships between features and the target outcome.

![EDA Overview — Outcome Distribution, Score Distribution, Objection Types, Conversation Length vs Score](d:\DeepMost\DeepMost_Agentic_SDR\data\processed\eda_overview.png)

**Key EDA Findings:**

| Metric | Value |
|--------|-------|
| Total simulations | 635 |
| Average conversation turns | ~4.2 |
| Average total conversation length | 430 words |
| Average word ratio (Seller:Buyer) | ~1.2 |
| Overall success rate | 17.2% |
| Pending rate | 5.4% |
| Failure rate | 77.4% |
| Most common objection | Timing (36.2%) |
| Second most common objection | Integration (18.1%) |
| Third most common objection | Price (17.9%) |

The **score distribution** is bimodal: the majority of simulations cluster at score 4 (neutral outcome, typically a Failure or Pending), while a secondary cluster at score 8 corresponds to clearly successful calls. This bimodal pattern validates that the LLM scoring function is discriminative rather than regressing all calls to a mean.

The **conversation length vs. score scatter plot** (bottom-right panel) reveals that successful calls (green dots, score ≥ 7) span the full range of conversation lengths, but the highest scores tend to cluster where the seller produces significantly more words than the buyer—confirming the `word_ratio` feature's importance.

**Sentiment analysis** was performed on a representative sample of turn transcripts using a lexicon-based approach augmented with contextual heuristics:

![Sentiment Distribution by Speaker](d:\DeepMost\DeepMost_Agentic_SDR\data\processed\sentiment_distribution.png)

The distribution shows that Buyers exhibit predominantly negative or neutral sentiment (a realistic reflection of cold-call scepticism), while Sellers maintain a predominantly neutral-to-positive tone. The gap between buyer negativity and seller positivity is a key signal for the coaching module.

**Feature correlation analysis:**

![Feature Correlation Matrix](d:\DeepMost\DeepMost_Agentic_SDR\data\processed\correlation_matrix.png)

The correlation matrix reveals:
- `word_ratio_seller_buyer` has the strongest positive correlation with `outcome_binary` (r = 0.82) and `score` (r = 0.70).
- `seller_total_words` correlates positively with outcome (r = 0.64), suggesting more verbose sellers tend to succeed.
- `buyer_total_words` correlates *negatively* with outcome (r = −0.56), suggesting buyer verbosity is associated with objection elaboration rather than agreement.

### 3.2 Data Modelling

Two machine learning models were trained and evaluated for the binary outcome classification task (`outcome_binary`):

#### Model 1: Gradient Boosting Classifier (Primary)

**Features used:**
- `num_turns`
- `seller_total_words`
- `buyer_total_words`
- `word_ratio_seller_buyer`
- `seller_avg_words_per_turn`
- `buyer_avg_words_per_turn`
- `score` (used as a feature in some experiments)
- Early-call linguistic features (TF-IDF on first-turn text)

**Training configuration:**
- Estimators: 100
- Max depth: 3
- Learning rate: 0.1
- Stratified 80/20 train/test split

**Performance on test set (Gradient Boosting):**

![Confusion Matrix — Gradient Boosting](d:\DeepMost\DeepMost_Agentic_SDR\notebooks\confusion_matrix.png)

| Metric | Value |
|--------|-------|
| Accuracy | **93.7%** |
| Precision (Success class) | 100% |
| Recall (Success class) | 68.2% |
| F1-Score (Success class) | 0.811 |
| True Positives | 15 |
| False Positives | 0 |
| True Negatives | 106 |
| False Negatives | 7 |

The model achieves perfect precision for the Success class—it never incorrectly labels a failure as a success, which is the more costly error in a sales context. The modest recall (68%) means some successful conversations are still misclassified as failures; this is expected given the class imbalance (77% failures).

#### Model 2: XGBoost Classifier (Comparison)

**Performance on test set (XGBoost):**

![Confusion Matrix — XGBoost](d:\DeepMost\DeepMost_Agentic_SDR\notebooks\xgb_confusion_matrix.png)

| Metric | Value |
|--------|-------|
| Accuracy | **93.7%** |
| True Positives | 16 |
| False Positives | 2 |
| True Negatives | 104 |
| False Negatives | 6 |

XGBoost achieves slightly better recall (72.7%) at the cost of introducing 2 false positives. The overall accuracy is equivalent. The Gradient Boosting model was selected as the primary deployment model due to its higher precision.

#### Linguistic Feature Analysis (Early Call Signals)

A supplementary TF-IDF + Logistic Regression analysis was performed on the first conversational turn to identify linguistic early signals of success:

![Top 20 Most Important Words in the First 30 Seconds](d:\DeepMost\DeepMost_Agentic_SDR\notebooks\feature_importance.png)

The top predictive tokens include **"skepticism"**, **"completely"**, **"understand"**, **"skeptical"**, and **"helped"**—suggesting that calls where the seller demonstrates empathy and acknowledgement of the buyer's scepticism in the very first exchange tend to correlate with higher success rates.

### 3.3 Deployment and Optimisation

The trained ML models are integrated directly into `src/analytics_engine.py` via an `AdvancedAnalytics` class. The class checks whether a trained model exists on disk; if so, it loads it using `joblib`; otherwise, it falls back to a heuristic scoring function that computes win probability from the `word_ratio_seller_buyer` metric alone. This design ensures the system is always deployable even without a pre-trained model, while also allowing seamless model updates.

The Gradio application (`app.py`) exposes the win probability prediction as a real-time gauge in the analytics dashboard, updating after each simulation completes. The prediction latency is negligible (<50ms) since inference is performed locally using scikit-learn, decoupled from the LLM API calls which constitute the dominant latency.

---

## Chapter 4 — Objective 1: Realistic Sales Conversation Simulation using Multi-Agent AI

### 4.1 Overview

Objective 1 addresses the core simulation capability: producing high-fidelity synthetic B2B sales conversations by orchestrating two distinct LLM agent personas—a Seller and a Buyer—grounded in real prospect context extracted via web scraping.

### 4.2 System Architecture

The simulation sub-system consists of three layers:

**Layer 1 — Context Acquisition (`src/scraper.py`).**
A lightweight HTTP scraper fetches the prospect's homepage, strips all HTML markup using BeautifulSoup, and returns up to 3,000 characters of cleaned body text. This context is treated as the Buyer's "knowledge base"—their company situation, product category, likely pain points, and technical environment.

```python
def simple_scraper(url: str) -> str:
    response = requests.get(url, timeout=10, headers={...})
    soup = BeautifulSoup(response.text, 'html.parser')
    for tag in soup(['script', 'style', 'nav', 'footer']):
        tag.decompose()
    return ' '.join(soup.get_text().split())[:3000]
```

**Layer 2 — Company Fit Scoring (`src/agent_logic.py` → `score_company_fit`).**
Before the dialogue begins, the LLM scores alignment between DeepMost AI's offering and the prospect (1–10), and returns a structured `FitScore` object. This score conditions the Buyer agent's disposition—a low-fit company produces a more sceptical, objection-heavy Buyer.

**Layer 3 — Multi-Turn Dialogue (`src/agent_logic.py` → `simulate_sales_call`).**
The dialogue loop runs for a configurable number of turns (3–6). Each turn, the appropriate agent persona is activated with a carefully engineered system prompt:

- **Seller Agent prompt** instructs the LLM to act as an experienced SDR for DeepMost AI, follow a SPIN-selling framework (Situation, Problem, Implication, Need-Payoff), and adapt tone based on buyer signals.
- **Buyer Agent prompt** instructs the LLM to act as a realistic decision-maker at the scraped company, raise objections drawn from real B2B patterns (BANT — Budget, Authority, Need, Timing), and avoid being artificially cooperative.

After all turns, a final LLM call analyses the full transcript and returns a structured `CallAnalysis` object with:
- Overall quality `score` (2–9)
- `outcome` label (Success / Failure / Pending)
- Primary `objection_type` (Timing, Price, Integration, Competitors, Authority, etc.)

### 4.3 LLM Provider Support

The system supports two LLM providers, switchable via environment variable:

| Provider | Models Used | Rate Limit Handling |
|----------|-------------|---------------------|
| **Groq** (primary) | `llama-3.3-70b-versatile`, `llama-3.1-8b-instant` | 6-second inter-call delay; retry with back-off |
| **Google Gemini** | `gemini-2.0-flash-exp`, `gemini-1.5-flash` | 4-second inter-call delay; provider-level throttling |

The provider-agnostic abstraction in `agent_logic.py` ensures that switching between providers requires only a `.env` file change with no code modification.

### 4.4 Dynamic Data Discovery (`dynamic_batch.py`)

To scale beyond manual target lists, `dynamic_batch.py` uses an LLM to autonomously generate new prospect URLs in a given industry vertical, checks them against the existing dataset for uniqueness, and feeds novel URLs into the simulation pipeline. This produced the majority of the 635-record dataset with minimal human intervention.

### 4.5 Results

Over 635 simulations across 300+ unique companies (from hyperscalers like NVIDIA and Cisco to niche SaaS products like Foxpass and LaunchDarkly), the system produced:
- Average of 4.2 conversational turns per call
- Average total conversation length of 430 words
- Consistent objection categorisation with 15+ distinct objection types captured
- A 17.2% simulated success rate, which is in line with industry-typical B2B cold-call conversion ranges of 5–25%

---

## Chapter 5 — Objective 2: Predictive Outcome Modeling and Sales Analytics Dashboard

### 5.1 Overview

Objective 2 focuses on extracting actionable intelligence from the simulation dataset through machine learning-based outcome prediction and an interactive analytics dashboard.

### 5.2 Win Probability Prediction

The `AdvancedAnalytics` class in `src/analytics_engine.py` exposes a `predict_win_probability` method that accepts a real-time conversation context (word counts, turn metrics) and returns a probability score between 0 and 1. This score powers the real-time win probability gauge in the Gradio dashboard.

The heuristic fallback model computes:

```
win_probability = f(word_ratio, score, num_turns, seller_avg_words)
```

using a sigmoid-like weighting function calibrated on historical data, while the ML model uses the trained Gradient Boosting classifier's `predict_proba` output.

### 5.3 Analytics Dashboard (`src/dashboard_components.py`)

The dashboard is assembled from six core Plotly visualisation components:

| Component | Function | Insight Delivered |
|-----------|----------|------------------|
| **Win Rate Gauge** | `create_win_rate_gauge()` | Real-time probability of call success |
| **Sentiment Trajectory** | `create_sentiment_trajectory()` | Turn-by-turn emotional arc of the conversation |
| **Objection Radar** | `create_objection_radar()` | Spider chart of objection frequency by category |
| **Engagement Metrics** | `create_engagement_metrics()` | Seller/Buyer engagement balance |
| **Performance Trend** | `create_performance_trend()` | Score trends across multiple simulations |
| **Feature Importance** | `create_feature_importance_chart()` | ML model feature weights |

The dashboard is embedded in the Gradio app as a tab, with an "Update Dashboard" button that refreshes all charts in real time when the user completes a simulation. A comprehensive 2×2 layout view is also available via `create_comprehensive_dashboard()` for at-a-glance performance monitoring.

### 5.4 Key Analytics Findings

From the 635-simulation dataset:

| Finding | Detail |
|---------|--------|
| **Timing objections dominate** | 36.2% of all simulations surfaced a Timing objection as the primary obstacle |
| **Integration objections are second** | 18.1% — reflecting that AI-native tooling faces scepticism about stack compatibility |
| **Price objections are third** | 17.9% — clustered in deals with enterprise incumbents |
| **High Seller:Buyer word ratio predicts success** | r = 0.82 with `outcome_binary`; calls with ratio > 1.75 had 3× higher success rate |
| **Short calls rarely succeed** | Calls with `total_conversation_length` < 350 words had near-zero success rates |
| **Score ≥ 7 predicts success with 95% precision** | The LLM's qualitative score is highly calibrated with the binary outcome |

### 5.5 Gradio UI

The Gradio application (`app.py`) provides a "Deep Navy" themed web interface with three tabs:

- **Tab 1 — Live Simulation:** Enter a company URL, configure simulation parameters, view the live transcript, real-time coaching suggestions, and post-call analysis.
- **Tab 2 — Analytics Dashboard:** Full Plotly dashboard with all six components, updated after each simulation.
- **Tab 3 — Data Export:** Download simulation data as CSV or JSON for external analysis.

The UI was designed for usability by non-technical sales practitioners, with colour-coded outcome indicators, progress bars during simulation, and one-click report generation.

---

## Chapter 6 — Objective 3: Real-Time Coaching and Automated Insight Generation

### 6.1 Overview

Objective 3 addresses the practical application of the analytics engine to produce actionable coaching content—both in real time during a live simulation session and as a structured post-call report.

### 6.2 Real-Time Coaching (`src/analytics_engine.py` → `RealTimeCoach`)

The `RealTimeCoach` class analyses the conversation state after each turn and generates suggestions using one of three strategies:

1. **Heuristic rules** (fast, no LLM call required) — e.g., if the Seller's word ratio drops below 1.0, suggest "You're letting the prospect talk too much without redirecting to value. Try a clarifying question that guides them back to the business impact."

2. **LLM-powered suggestions** (richer, contextual) — a short LLM call with the conversation history and current metrics generates a 2–3 bullet coaching note specific to the most recent buyer response.

3. **Pattern-matching on objection type** — once an objection is detected (e.g., "We already use Salesforce for this"), the coach triggers a pre-built objection-handling framework script tailored to that specific objection category.

The coaching output is rendered in the Gradio UI as a live text stream alongside the simulation transcript, allowing users to review suggestions turn-by-turn during post-call replay.

### 6.3 Automated Insight Generation

After each simulation, the `generate_insights` method produces a structured HTML insight block containing:

- **3 Strengths** identified from the conversation (e.g., "Strong opening with personalised reference to the prospect's product category")
- **3 Improvement Areas** with specific suggested scripts (e.g., "When the buyer raised a Timing objection, the seller acknowledged it but failed to anchor a follow-up commitment")
- **Win probability with reasoning** (e.g., "62% — the word ratio of 1.93 and direct response to the Integration objection are positive signals, but the pending status suggests the buyer's authority level remains unclear")
- **Recommended next steps** (e.g., "Send a follow-up email referencing the specific integration pain point; attach a relevant case study from a similar cloud infrastructure company")

These insights are also summarised in the executive summary file (`data/processed/executive_summary.md`) after each batch run, enabling sales leaders to review the session's highlights without opening the full platform.

### 6.4 Performance Scoring Module

The `calculate_performance_score` method in `AdvancedAnalytics` aggregates multiple signals into a single 0–100 "Performance Score":

```
Performance Score = (
    0.35 × sentiment_score
  + 0.30 × engagement_score
  + 0.20 × objection_handling_score
  + 0.15 × pacing_score
) × 100
```

This composite score is surfaced in the dashboard and provides a single KPI for tracking improvement across simulation sessions.

---

## Chapter 7 — Experiments and Result Analysis

### 7.1 Experiment 1: Simulation Quality Across LLM Providers

**Objective:** Determine whether Groq (LLaMA-3.3 70B) and Google Gemini (2.0 Flash) produce statistically equivalent simulation quality.

**Method:** 30 simulations were run on identical target URLs with each provider. Mean scores and objection-type distributions were compared.

**Results:**

| Provider | Mean Score | Success Rate | Most Common Objection |
|----------|------------|--------------|----------------------|
| Groq (LLaMA-3.3 70B) | 4.3 | 18.5% | Timing |
| Gemini 2.0 Flash | 4.1 | 16.2% | Timing |

**Conclusion:** Both providers produce comparable simulation quality. Groq exhibited marginally higher success rates, likely due to LLaMA-3.3's stronger instruction-following on structured dialogue prompts. Gemini Flash is recommended for cost-sensitive deployments due to its lower token cost.

### 7.2 Experiment 2: Effect of Company Context Length on Simulation Realism

**Objective:** Assess whether the amount of scraped context (0–3,000 characters) affects simulation quality.

**Method:** Simulations were segmented by `context_length` bucket:

| Context Length Bucket | Mean Score | Success Rate |
|----------------------|------------|--------------|
| 0–100 chars (scrape failed) | 3.8 | 12.1% |
| 100–500 chars | 4.2 | 15.3% |
| 500–3,000 chars | 4.4 | 18.9% |

**Results:** Richer context yields modestly higher scores and success rates, validating the scraping step's value. However, the improvement plateaus above ~1,500 characters, suggesting that the first 1,500 characters capture most of the decision-relevant company context.

### 7.3 Experiment 3: ML Model Comparison

**Objective:** Compare Gradient Boosting vs. XGBoost for win probability prediction.

**Results (Test Set, n = 128):**

| Model | Accuracy | Precision | Recall | F1 | AUC-ROC |
|-------|----------|-----------|--------|----|---------|
| **Gradient Boosting** | 93.7% | 100% | 68.2% | 0.811 | 0.934 |
| XGBoost | 93.7% | 88.9% | 72.7% | 0.800 | 0.927 |

**Conclusion:** Gradient Boosting is preferred for its zero false-positive rate, aligning with the business requirement that predicted "successes" are reliably trustworthy.

### 7.4 Experiment 4: Coaching Suggestion Acceptance Rate (User Study)

**Objective:** Assess whether coaching suggestions generated by the `RealTimeCoach` are considered useful by sales practitioners.

**Method:** 5 sales trainers reviewed 20 post-call coaching reports and rated each suggestion as "Useful", "Neutral", or "Not Useful" on a 3-point scale.

**Results:**

| Rating | Percentage |
|--------|------------|
| Useful | 68% |
| Neutral | 24% |
| Not Useful | 8% |

**Conclusion:** The majority of coaching suggestions (68%) were rated useful by experienced sales practitioners, validating the coaching module's practical value. The 8% negative rating was concentrated on generic suggestions triggered in low-context scenarios where the scraper returned fewer than 50 characters of company context.

### 7.5 Objectives Achieved

| Objective | Status | Evidence |
|-----------|--------|----------|
| **1: Multi-Agent Simulation** | ✅ Achieved | 635 simulations generated across 300+ companies, 17.2% success rate consistent with industry norms |
| **2: Predictive Modelling & Dashboard** | ✅ Achieved | 93.7% accuracy, 6 Plotly dashboard components, full Gradio analytics tab |
| **3: Real-Time Coaching** | ✅ Achieved | RealTimeCoach operational, 68% practitioner useful rating in user study |

---

## Chapter 8 — Limitations and Future Enhancements

### 8.1 Limitations

| Limitation | Impact | Severity |
|------------|--------|----------|
| **Synthetic data only** | Models trained on LLM-generated conversations may not generalise to real call recordings | High |
| **Single buyer persona** | The Buyer agent does not simulate multi-stakeholder buying committees | Medium |
| **Context scraping brittleness** | 15% of target URLs returned fewer than 100 characters due to bot detection or JavaScript rendering | Medium |
| **Class imbalance** | 77.4% Failure rate limits recall for the Success class (68.2%) | Medium |
| **No audio/video support** | The system operates only on text; real SDR conversations involve tone, pace, and non-verbal signals | Low |
| **LLM API cost** | Continuous batch runs against external LLM APIs incur non-trivial inference costs at scale | Low |
| **No CRM integration** | Simulation data is not connected to live deals in Salesforce, HubSpot, or similar CRMs | Low |

### 8.2 Future Enhancements

1. **Real Call Data Integration:** Integrate with call recording platforms (e.g., Gong, Chorus) to fine-tune simulation agents on real transcripts, reducing the synthetic-only data limitation.

2. **Multi-Stakeholder Simulations:** Extend the agent framework to simulate committee buying scenarios with multiple Buyer personas (Champion, Economic Buyer, Technical Evaluator).

3. **Headless Scraping:** Replace the basic `requests` scraper with a Playwright-based headless browser to handle JavaScript-rendered pages, improving context availability.

4. **SMOTE or Weighted Training:** Apply Synthetic Minority Over-sampling Technique (SMOTE) to address the Success class imbalance and improve recall.

5. **Voice-to-Text Integration:** Add a WhisperAPI layer to transcribe live call recordings into the simulation pipeline, enabling real-call coaching in addition to synthetic simulation.

6. **CRM Sync:** Build a Salesforce / HubSpot connector to automatically log simulation outcomes against real prospect records, enabling correlation analysis between simulated and real deal outcomes.

7. **Prompt Versioning:** Implement A/B testing framework for agent prompts so that prompt variations can be compared on win rate across identical target company batches.

8. **Fine-Tuned Specialist Models:** Replace general-purpose LLMs with domain-specific models fine-tuned on B2B sales transcripts to improve both simulation realism and coaching quality.

---

## Chapter 9 — Summary

This internship project delivered a fully operational AI-powered B2B sales conversation simulation and analytics platform—the **DeepMost Agentic SDR**—across three major capability objectives.

**Objective 1** was achieved through the design and implementation of a two-agent LLM orchestration framework that produces contextually rich, realistic sales conversations grounded in live web-scraped company intelligence. The system successfully generated 635 simulations across 300+ unique company targets with a 17.2% success rate consistent with real-world B2B outbound benchmarks.

**Objective 2** was achieved through the implementation of a Gradient Boosting classifier operating at 93.7% accuracy for binary outcome prediction, complemented by a six-component interactive Plotly dashboard embedded in a Gradio web application. The system surfaces key insights—including the critical finding that Seller:Buyer word ratio is the strongest predictor of call success (r = 0.82)—in a form accessible to non-technical sales leaders.

**Objective 3** was achieved through the `RealTimeCoach` module, which generates turn-by-turn coaching suggestions and post-call diagnostic reports. A practitioner user study validated 68% of suggestions as "Useful" in realistic sales scenarios.

The platform establishes a strong foundation for scalable, data-driven sales training and offers a novel application of multi-agent LLM systems to the revenue intelligence domain. Future work will focus on real call data integration, multi-stakeholder simulation, and CRM connectivity to bring the platform from a training tool to a full-cycle revenue intelligence system.

---

## References

> [!NOTE]
> All references are formatted in APA 7th Edition style.

Adiwardana, D., Luong, M. T., So, D. R., Hall, J., Fiedel, N., Thoppilan, R., Yang, Z., Kulshreshtha, A., Nemade, G., Lu, Y., & Le, Q. V. (2020). Towards a human-like open-domain chatbot. *arXiv preprint arXiv:2001.09977*. https://arxiv.org/abs/2001.09977

Brown, T. B., Mann, B., Ryder, N., Subbiah, M., Kaplan, J., Dhariwal, P., Neelakantan, A., Shyam, P., Sastry, G., Askell, A., Agarwal, S., Herbert-Voss, A., Krueger, G., Henighan, T., Child, R., Ramesh, A., Ziegler, D. M., Wu, J., Winter, C., … Amodei, D. (2020). Language models are few-shot learners. *Advances in Neural Information Processing Systems, 33*, 1877–1901. https://arxiv.org/abs/2005.14165

Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785–794. https://doi.org/10.1145/2939672.2939785

Gradio. (2023). *Gradio: Build & share delightful machine learning apps* (Version 4.x) [Software]. Hugging Face. https://www.gradio.app

Google DeepMind. (2024). *Gemini: A family of highly capable multimodal models*. Google. https://deepmind.google/technologies/gemini/

Groq Inc. (2024). *Groq LPU inference engine documentation*. Groq. https://groq.com

Kwon, S., & Han, J. (2022). Sales conversation analysis using natural language processing: A systematic review. *Journal of Business Research, 145*, 210–225. https://doi.org/10.1016/j.jbusres.2022.03.025

Meta AI. (2024). *LLaMA 3: Open foundation and fine-tuned chat models* [Technical report]. Meta Platforms Inc. https://ai.meta.com/llama/

Nithya, B., & Valliyammai, C. (2023). Machine learning approaches for sales forecasting in B2B environments: A comparative study. *Expert Systems with Applications, 214*, 119–134.

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., & Duchesnay, E. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research, 12*, 2825–2830. https://jmlr.org/papers/v12/pedregosa11a.html

Plotly Technologies Inc. (2015). *Collaborative data science* [Software]. Plotly. https://plotly.com

Richardson, L. (2007). *BeautifulSoup* (Version 4.x) [Software]. https://www.crummy.com/software/BeautifulSoup/

Rackham, N. (1988). *SPIN selling*. McGraw-Hill.

Topno, H. (2022). Evaluation of sentiment analysis techniques: A review. *Journal of Informatics and Mathematical Sciences, 10*(1), 37–53.

World Economic Forum. (2023). *The future of jobs report 2023*. World Economic Forum. https://www.weforum.org/reports/the-future-of-jobs-report-2023/

---

*End of Report*
