import streamlit as st
import pickle

# Sidebar instructions
with st.sidebar:
    st.header("📘 How to Use")
    st.write("""
    1. Type or paste a movie review in the box.
    2. Click **Analyze** to see the sentiment.
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
st.markdown("Analyze the sentiment of any movie review — is it Positive, Negative, or Neutral?")
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
            proba = model.predict_proba(X)[0]
            classes = model.classes_
            max_index = proba.argmax()
            prediction = classes[max_index]
            confidence = proba[max_index]

            st.markdown("### 🔍 Sentiment Result")

            if prediction == "Positive":
                st.success(f"😊 Positive ({confidence:.2%} confident)")
            elif prediction == "Negative":
                st.error(f"😠 Negative ({confidence:.2%} confident)")
            else:
                st.info(f"😐 Neutral ({confidence:.2%} confident)")
        except Exception as e:
            st.error(f"Something went wrong: {e}")
