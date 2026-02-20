import streamlit as st
import pickle

# Load saved model
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

st.title("📰 Fake News Detector")

news = st.text_area("Enter News Text:")

if st.button("Check News"):
    if news.strip() != "":
        data = vectorizer.transform([news])
        prediction = model.predict(data)

        if prediction[0] == 1:
            st.success("✅ This is Real News")
        else:
            st.error("❌ This is Fake News")
    else:
        st.warning("Please enter some news text")