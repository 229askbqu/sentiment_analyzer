import streamlit as st
import pickle

# Load model + vectorizer
model = pickle.load(open("model/model.pkl", "rb"))
vectorizer = pickle.load(open("model/vectorizer.pkl", "rb"))

st.title("🎬 Sentiment Analyzer")
user_input = st.text_area("Enter a movie review:")

if st.button("Analyze"):
    vec = vectorizer.transform([user_input])
    prediction = model.predict(vec)[0]
    st.write("Sentiment:", prediction)
