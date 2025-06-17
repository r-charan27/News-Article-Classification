
# 🧪 Streamlit App - Fake News Classifier

import streamlit as st
import joblib
import pandas as pd

# Load model and vectorizer
model = joblib.load("F:/fake news/news_model.pkl")
vectorizer = joblib.load("F:/fake news/tfidf_vectorizer.pkl")

st.set_page_config(page_title="Fake News Detector")
st.title("🧪 Fake News Classifier")
st.markdown("Enter a news article below, and the model will predict if it's **Real** or **Fake**.")

input_text = st.text_area("📝 Paste News Article Text:", height=250)

if st.button("🔍 Predict"):
    if input_text.strip() == "":
        st.warning("⚠️ Please enter some text.")
    else:
        input_vec = vectorizer.transform([input_text])
        prediction = model.predict(input_vec)[0]
        label = "✅ Real News" if prediction == 1 else "❌ Fake News"
        st.success(f"Prediction: {label}")

st.markdown("---")
st.markdown("<center>Built with 🧠 Logistic Regression + TF-IDF</center>", unsafe_allow_html=True)
