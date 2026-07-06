import streamlit as st
import pickle
import re
import pandas as pd
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# ---------------- Load Models ---------------- #
with open("Models/model_xgb.pkl", "rb") as f:
    model = pickle.load(f)

with open("Models/countVectorizer.pkl", "rb") as f:
    cv = pickle.load(f)

with open("Models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

STOPWORDS = set(stopwords.words("english"))

# ---------------- Preprocessing ---------------- #
def preprocess_text(text):
    stemmer = PorterStemmer()

    text = re.sub("[^a-zA-Z]", " ", text)
    text = text.lower().split()

    text = [stemmer.stem(word) for word in text if word not in STOPWORDS]

    return " ".join(text)

# ---------------- Prediction ---------------- #
def predict_sentiment(text):
    processed = preprocess_text(text)

    vector = cv.transform([processed]).toarray()

    vector = scaler.transform(vector)

    prediction = model.predict(vector)[0]

    if prediction == 1:
        return "😊 Positive"
    else:
        return "😞 Negative"

# ---------------- UI ---------------- #

st.set_page_config(
    page_title="Sentiment Analysis",
    page_icon="💬",
    layout="centered"
)

st.title("💬 Amazon Alexa Sentiment Analysis")

st.write("Enter a review and predict its sentiment.")

review = st.text_area("Enter Review")

if st.button("Predict"):

    if review.strip() == "":
        st.warning("Please enter a review.")

    else:

        result = predict_sentiment(review)

        if "Positive" in result:
            st.success(result)
        else:
            st.error(result)