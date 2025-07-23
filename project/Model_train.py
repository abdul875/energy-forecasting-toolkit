import pandas as pd
import numpy as np
import os
import joblib
import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

from sklearn.ensemble import (
    RandomForestClassifier, VotingClassifier, BaggingClassifier, AdaBoostClassifier, StackingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC

# -------------------------------
# Step 1: Load and preprocess data
# -------------------------------
df = pd.read_csv("F:/JU/MY_Project/JU_project/weather_with_renewable_source.csv")

features = [
    'solarradiation', 'uvindex', 'cloudcover', 'windspeed',
    'windgust', 'precip', 'snow', 'snowdepth', 'sunrise', 'sunset'
]

df['sunrise'] = pd.to_datetime(df['sunrise'], errors='coerce')
df['sunset'] = pd.to_datetime(df['sunset'], errors='coerce')
df['day_length_minutes'] = (df['sunset'] - df['sunrise']).dt.total_seconds() / 60

features.remove('sunrise')
features.remove('sunset')
features.append('day_length_minutes')

df = df[features + ['renewable_source']].dropna()

le = LabelEncoder()
df['label'] = le.fit_transform(df['renewable_source'])

X = df[features]
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------------------
# Step 2: Define individual models
# -------------------------------
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "XGBoost": XGBClassifier(eval_metric='mlogloss'),
    "SVM": SVC(probability=True)
}

# -------------------------------
# Step 3: Add Ensemble Models
# -------------------------------
models["Voting Classifier"] = VotingClassifier(
    estimators=[
        ('rf', models["Random Forest"]),
        ('lr', models["Logistic Regression"]),
        ('xgb', models["XGBoost"])
    ],
    voting='soft'
)

models["Stacking Classifier"] = StackingClassifier(
    estimators=[
        ('rf', models["Random Forest"]),
        ('xgb', models["XGBoost"]),
        ('svc', models["SVM"])
    ],
    final_estimator=LogisticRegression(max_iter=1000)
)

models["Bagging Classifier"] = BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=50,
    random_state=42
)

models["AdaBoost Classifier"] = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=100,
    random_state=42
)

# -------------------------------
# Step 4: Train, Evaluate, Save
# -------------------------------
os.makedirs("saved_models", exist_ok=True)
print("🚀 Starting model training...\n")

best_model = None
best_score = 0

total_models = len(models)
for idx, (name, model) in enumerate(models.items(), 1):
    print(f"[{idx}/{total_models}] 🔄 Training {name}...")

    start = time.time()
    model.fit(X_train, y_train)
    end = time.time()

    y_pred = model.predict(X_test)
    score = accuracy_score(y_test, y_pred)

    print(f"✅ {name} trained in {end - start:.2f} seconds")
    print(f"📊 Accuracy: {score:.4f}")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # Save model immediately
    filename = f"saved_models/{name.replace(' ', '_').lower()}_model.pkl"
    joblib.dump(model, filename)
    print(f"💾 Model saved to: {filename}")
    print("-" * 60)

    # Track best model
    if score > best_score:
        best_score = score
        best_model = name

print(f"\n🎉 Training Complete!")
print(f"🏆 Best Model: {best_model} with Accuracy: {best_score:.4f}")
