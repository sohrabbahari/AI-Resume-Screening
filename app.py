from fastapi import FastAPI
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

# Load the trained model and vectorizer
with open("model/resume_classifier.pkl", "rb") as model_file:
    clf = pickle.load(model_file)

with open("model/tfidf_vectorizer.pkl", "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Load job categories
df = pd.read_csv("dataset/cleaned_resumes.csv")
categories = df["Category"].unique()

# Initialize FastAPI app
app = FastAPI()

@app.get("/")
def home():
    return {"message": "Resume Screening API is running!"}

@app.post("/predict/")
def predict_category(resume_text: str):
    """Predict the job category for a given resume text."""
    # Convert input text into numerical features
    transformed_text = vectorizer.transform([resume_text])
    
    # Predict category
    prediction = clf.predict(transformed_text)[0]
    
    # Convert predicted category index to actual category name
    predicted_category = categories[prediction]
    
    return {"Predicted Job Category": predicted_category}

