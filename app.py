import streamlit as st
from keybert import KeyBERT

# Initialize the model
kw_model = KeyBERT()

# Streamlit App UI
st.title("🔍 Niche Market Opportunity Detector")

user_input = st.text_area("Paste your content here (e.g. reviews, Reddit posts, blogs):")
num_keywords = st.slider("How many niche keywords do you want to extract?", 5, 20, 10)

if st.button("Detect Opportunities"):
    if user_input.strip():
        keywords = kw_model.extract_keywords(
            user_input,
            keyphrase_ngram_range=(1, 2),
            stop_words='english',
            top_n=num_keywords
        )
        st.subheader("📌 Niche Keyword Suggestions")
        for kw, score in keywords:
            st.write(f"• {kw} — Score: {score:.2f}")
    else:
        st.warning("Please enter some text to analyze.")
