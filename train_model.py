import pandas as pd
import numpy as np
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle  # For saving the trained model
import os

# Create the "model/" directory if it doesn't exist
os.makedirs("model", exist_ok=True)

# Load preprocessed resume data
df = pd.read_csv("dataset/cleaned_resumes.csv")

# Check if columns exist
if "Cleaned_Resume" not in df.columns or "Category" not in df.columns:
    raise ValueError("Columns 'Cleaned_Resume' or 'Category' not found in dataset.")


# **Fix missing values by filling empty resumes with an empty string**
df["Cleaned_Resume"] = df["Cleaned_Resume"].fillna("")


# Convert text into numerical features using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)  # Use top 5000 words
X = vectorizer.fit_transform(df["Cleaned_Resume"])

# Encode job categories as numbers
categories = df["Category"].unique()
category_mapping = {cat: idx for idx, cat in enumerate(categories)}
df["Category_Encoded"] = df["Category"].map(category_mapping)

# Define input (X) and output (y)
y = df["Category_Encoded"]

# Split data into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predict on the test set
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model Training Completed! Accuracy: {accuracy:.2f}")

# Show classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=categories))

# Save the trained model & vectorizer
with open("model/resume_classifier.pkl", "wb") as model_file:
    pickle.dump(clf, model_file)

with open("model/tfidf_vectorizer.pkl", "wb") as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)

print("✅ Model and vectorizer saved in 'model/' folder.")
