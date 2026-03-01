import pandas as pd
import json
import os
import warnings
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score

warnings.filterwarnings('ignore')

df = pd.read_csv('data/processed/simulations_master.csv')
df = df.dropna(subset=['conversation_file', 'outcome'])
CONVERSATION_DIR = 'data/raw/conversations'

def extract_early_conversation(filename):
    file_path = os.path.join(CONVERSATION_DIR, filename)
    if not os.path.exists(file_path):
        return ""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        history = data.get('conversation', [])
        early_turns = history[:3]
        return " ".join([f"{msg.get('speaker', '')}: {msg.get('message', '')}" for msg in early_turns])
    except Exception as e:
        return ""

df['early_text'] = df['conversation_file'].apply(extract_early_conversation)
df = df[df['early_text'] != ""]

success_labels = ['Success', 'Meeting Booked', 'Nurture']
df['target'] = df['outcome'].apply(lambda x: 1 if any(label.lower() in str(x).lower() for label in success_labels) else 0)

X = df['early_text']
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_vec, y_train)

print("=== RANDOM FOREST ===")
rf = RandomForestClassifier(n_estimators=200, max_depth=5, class_weight='balanced_subsample', random_state=42)
rf.fit(X_train_res, y_train_res)
rf_train_pred = rf.predict(X_train_res)
rf_test_pred = rf.predict(X_test_vec)
print(f"Train Accuracy (on SMOTE): {accuracy_score(y_train_res, rf_train_pred):.2f}")
print(f"Test Accuracy: {accuracy_score(y_test, rf_test_pred):.2f}")
print("Test Classification Report:")
print(classification_report(y_test, rf_test_pred))

print("\n=== XGBOOST ===")
scale_pos_weight = sum(y_train == 0) / sum(y_train == 1)
xgb = XGBClassifier(eval_metric='logloss', random_state=42, scale_pos_weight=scale_pos_weight, use_label_encoder=False)
xgb.fit(X_train_vec, y_train) # Training XGBoost WITHOUT SMOTE, using scale_pos_weight instead
xgb_train_pred = xgb.predict(X_train_vec)
xgb_test_pred = xgb.predict(X_test_vec)
print(f"Train Accuracy: {accuracy_score(y_train, xgb_train_pred):.2f}")
print(f"Test Accuracy: {accuracy_score(y_test, xgb_test_pred):.2f}")
print("Test Classification Report:")
print(classification_report(y_test, xgb_test_pred))
