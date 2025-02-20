# 📌 Import Required Libraries
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sentence_transformers import SentenceTransformer  # BERT Embeddings

# 📌 Load the Preprocessed Dataset
df = pd.read_csv("dataset/cleaned_resumes.csv")

# 📌 Handle Missing Values
df["Cleaned_Resume"] = df["Cleaned_Resume"].fillna("")  
df["Cleaned_Resume"] = df["Cleaned_Resume"].astype(str)

# 📌 Load Pretrained BERT Model
bert_model = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight, fast BERT model

# 📌 Convert Text into BERT Embeddings
print("⚡ Generating BERT Embeddings... (This may take a few minutes)")
X = bert_model.encode(df["Cleaned_Resume"], batch_size=32, show_progress_bar=True)

# 📌 Encode Job Categories
categories = df["Category"].unique()
category_mapping = {cat: idx for idx, cat in enumerate(categories)}
df["Category_Encoded"] = df["Category"].map(category_mapping)
y = df["Category_Encoded"]

# 📌 Split Data into Training & Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 📌 Train Multiple Models & Compare Accuracy
models = {
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=None, min_samples_split=5, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=200, random_state=42),
    "XGBoost": XGBClassifier(eval_metric="mlogloss")
}

results = {}

for model_name, model in models.items():
    print(f"🔥 Training {model_name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    results[model_name] = accuracy
    print(f"✅ {model_name} Accuracy: {accuracy:.2f}")

# 📌 Print the Best Model
best_model_name = max(results, key=results.get)
print(f"🏆 Best Model: {best_model_name} with Accuracy: {results[best_model_name]:.2f}")

# 📌 Save the Best Model
best_model = models[best_model_name]
with open("model/resume_classifier.pkl", "wb") as model_file:
    pickle.dump(best_model, model_file)

# 📌 Save the BERT Model (for future inference)
with open("model/bert_model.pkl", "wb") as bert_file:
    pickle.dump(bert_model, bert_file)

print("✅ Best Model and BERT Embeddings Saved Successfully!")
