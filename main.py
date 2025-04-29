import streamlit as st
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer

# Cache the model loading to avoid repeated downloads
@st.cache_resource
def load_model():
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    kw_model = KeyBERT(model)
    return kw_model

# Load the model
kw_model = load_model()

# Streamlit UI
st.title("🔍 Niche Market Opportunity Detector")
st.markdown("Enter a topic or idea below to detect related **niche market keywords** using AI.")

# Input text box
user_input = st.text_area("Enter a business idea, trend, or product concept:", height=200)

# Button to trigger keyword extraction
if st.button("Find Niche Opportunities"):
    if user_input.strip():
        keywords = kw_model.extract_keywords(user_input, keyphrase_ngram_range=(1, 2), stop_words='english', top_n=10)
        st.subheader("🧠 Suggested Niche Keywords:")
        for kw, score in keywords:
            st.markdown(f"✅ **{kw}** (Relevance Score: `{round(score, 2)}`)")
    else:
        st.warning("Please enter some text to analyze.")
