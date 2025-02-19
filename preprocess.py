import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import spacy
from bs4 import BeautifulSoup  # For removing HTML tags

# Load the dataset
df = pd.read_csv(r"C:\Users\sohra\Desktop\Projects\ResumeScreeningAI\dataset\Resume.csv")
# Load NLP model
nlp = spacy.load("en_core_web_sm")

# Download stopwords if not available
nltk.download("stopwords")
nltk.download("punkt")
stop_words = set(stopwords.words("english"))

def clean_text(text):
    """Preprocess resume text: remove HTML, special chars, stopwords, and tokenize."""
    # Remove HTML tags
    text = BeautifulSoup(text, "html.parser").get_text()
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove numbers and special characters
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\W+', ' ', text)
    
    # Tokenize text
    words = word_tokenize(text)
    
    # Remove stopwords
    words = [word for word in words if word not in stop_words]
    
    return " ".join(words)

# Apply preprocessing to all resumes
df["Cleaned_Resume"] = df["Resume_str"].apply(clean_text)

# Save cleaned data
df.to_csv("dataset/cleaned_resumes.csv", index=False)

print("✅ Resume Preprocessing Done! Cleaned data saved as 'cleaned_resumes.csv'.")
