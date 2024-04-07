import streamlit as st
from joblib import load
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import joblib

# Load the CountVectorizer and the trained model
cv = joblib.load("count_vectorizer.pkl")
loaded_logistic_model = joblib.load("Fake_News.pkl")

# Function to preprocess text
def preprocess_text(text):
    # Implement your preprocessing steps here
    return text

# Function to detect fake news
def detect_fake_news(text):
    # Preprocess the text and transform it using CountVectorizer
    count_vector = cv.transform([preprocess_text(text)]).reshape(1, -1)
    
    # Make predictions
    prediction = loaded_logistic_model.predict(count_vector)
    
    return prediction

def main():
    # Title and description
    st.title("Fake News Detector")
    st.write("Enter the text to check if it's fake news or not.")

    # Text input field
    text_input = st.text_area("Enter text here:")

    # Button to trigger fake news detection
    if st.button("Check"):
        # Perform fake news detection
        if text_input.strip() == "":
            st.error("Please enter some text.")
        else:
            fake_news = detect_fake_news(text_input)
            if fake_news == 1:
                st.error("This text is likely fake news.")
            else:
                st.success("This text seems to be genuine.")

if __name__ == "__main__":
    main()
