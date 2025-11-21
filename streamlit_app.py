import streamlit as st
import pickle

# Load model
model = pickle.load(open("simple_model.pkl", "rb"))

st.title("📰 Simple Fake News Detector")
st.write("Lightweight TF-IDF + Logistic Regression model")

# Inputs
text = st.text_area("Enter the news article text here:", height=200)

if st.button("Predict"):
    if text.strip() == "":
        st.warning("Please enter some text.")
    else:
        pred = model.predict([text])[0]
        label = "Real" if pred == 1 else "Fake"
        st.success(f"Prediction: **{label}**")
