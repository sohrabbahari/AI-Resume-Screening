from fastapi import FastAPI
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

# Load the trained model and BERT embedding model
with open("model/resume_classifier.pkl", "rb") as model_file:
    clf = pickle.load(model_file)

with open("model/bert_model.pkl", "rb") as bert_file:
    bert_model = pickle.load(bert_file)

# Load job categories
import pandas as pd
df = pd.read_csv("dataset/cleaned_resumes.csv")
categories = df["Category"].unique()

# Initialize FastAPI app
app = FastAPI()

@app.get("/")
def home():
    return {"message": "BERT-powered Resume Screening API is running!"}

@app.post("/predict/")
def predict_category(resume_text: str):
    """Predict the job category for a given resume text using BERT embeddings."""
    # Convert input text into BERT embeddings
    embedding = bert_model.encode([resume_text])
    
    # Predict category
    prediction = clf.predict(embedding)[0]
    
    # Convert predicted category index to actual category name
    predicted_category = categories[prediction]
    
    return {"Predicted Job Category": predicted_category}
