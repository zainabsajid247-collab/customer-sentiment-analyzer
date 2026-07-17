import streamlit as st
import pickle
import re

# 1. Load files
model = pickle.load(open("sentiment_model.pkl", "rb"))
tfidf = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

# 2. Clean text
def clean_text(text):
    text = str(text).lower()
    text = text.replace("won't", "will not")
    text = text.replace("can't", "can not")
    text = text.replace("n't", " not")
    text = re.sub(r'[^a-z\s]', '', text)
    return text

# 3. UI Design
st.title("Customer Review Sentiment Analyzer")
review = st.text_area("Write your review here...")

if st.button("Analyze Sentiment"):
    cleaned = clean_text(review)
    vectorized = tfidf.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    st.success(f"Predicted Sentiment: {prediction}")