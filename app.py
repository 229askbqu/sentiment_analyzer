import streamlit as st
import pickle

# Sidebar instructions
with st.sidebar:
    st.header("📘 How to Use")
    st.write("""
    1. Type or paste a movie review in the box.
    2. Click **Analyze** to see if it's Good, Neutral, or Bad.
    3. Try different reviews to test the model!
    """)

# Page config
st.set_page_config(
    page_title="Sentiment Analyzer",
    page_icon="💬",
    layout="centered"
)

# App header
st.title("🎭 Movie Review Sentiment Analyzer")
st.markdown("Analyze the sentiment of any movie review — is it Good, Neutral, or Bad.")
st.markdown("---")

# Load model and vectorizer
model = pickle.load(open("model/model.pkl", "rb"))
vectorizer = pickle.load(open("model/vectorizer.pkl", "rb"))

# Input section
st.markdown("### 📝 Enter Your Review")
review = st.text_area("Your review here:", height=150)

# Analyze button
if st.button("Analyze"):
    if not review.strip():
        st.warning("⚠️ Please enter a review before analyzing.")
    else:
        try:
            X = vectorizer.transform([review])
            prediction = model.predict(X)[0]

            st.markdown("### 🔍 Sentiment Result")

            if prediction == "Positive":
                st.success("😊 Good")
            elif prediction == "Negative":
                st.error("😠 Bad")
            else:
                st.info("😐 Neutral")
        except Exception as e:
            st.error("Something went wrong while analyzing the review.")
