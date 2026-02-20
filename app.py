import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load dataset
df_fake = pd.read_csv("Fake.csv")
df_true = pd.read_csv("True.csv")

df_fake["label"] = 0
df_true["label"] = 1

# ✅ Reduced dataset size (important for Streamlit Cloud)
df = pd.concat([df_fake, df_true]).sample(n=3000)

# Prepare data
X = df["text"]
y = df["label"]

# Vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X = vectorizer.fit_transform(X)

# Train model
model = LogisticRegression(max_iter=100)
model.fit(X, y)

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