import streamlit as st
import pickle

# sidebar 
with st.sidebar:
    st.header("📘 How to Use")
    st.write("""
    1. Type or paste a movie review in the box.
    2. Click **Analyze** to see the sentiment.
    3. Try different reviews to test the model!
    """)


# page config
st.set_page_config(
    page_title="Sentiment Analyzer",
    page_icon="💬",
    layout="centered"
)

# app header
st.title("🎭 Movie Review Sentiment Analyzer")
st.markdown("Analyze the sentiment of any movie review — is it Positive, Negative, or Neutral?")
st.markdown("---")


# Load model(pipeline with vectorizer inside)
model = pickle.load(open("model/model.pkl", "rb"))

# input section
st.markdown("### 📝 Enter Your Review")
review = st.text_area("Your review here:", height=150)

# analyze button
if st.button("Analyze"):
    if not review.strip():
        st.warning("⚠️ Please enter a review before analyzing.")
    else:
        try:
            vectorizer = pickle.load(open("model/vectorizer.pkl", "rb"))
            X = vectorizer.transform([review])
            proba = model.predict_proba(X)[0]
            prediction = model.predict(X)[0]
            confidence = max(proba)

            st.markdown("### 🔍 Sentiment Result")

            if prediction == "Positive":
                st.success(f"😊 Positive ({confidence:.2%} confident)")
            elif prediction == "Negative":
                st.error(f"😠 Negative ({confidence:.2%} confident)")
            else:
                st.info(f"😐 Neutral ({confidence:.2%} confident)")
        except Exception as e:
            st.error(f"Something went wrong: {e}")




    
        

