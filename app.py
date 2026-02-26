import re
import pickle
import streamlit as st

# ── Page Config ───────────────────────────────────────────
st.set_page_config(
    page_title="📰 Fake News Detector",
    page_icon="🔍",
    layout="centered"
)
# ── Cleaning ────────────────────────────────────────────
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-zA-Z ]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text

# ── Load Model ────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open('model_bundle.pkl', 'rb') as f:
        return pickle.load(f)

bundle = load_model()
vectorizer = bundle['vectorizer']
model      = bundle['model']

# ── Keyword Explanation ───────────────────────────────────
import numpy as np

def get_top_keywords(text: str, n: int = 5):
    cleaned = clean_text(text)
    st.write("Cleaned text:", cleaned)
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]
    feature_names = vectorizer.get_feature_names_out()
    coef = model.coef_[0]
    tfidf_arr = vec.toarray()[0]
    scores = tfidf_arr * coef
    top_idx = np.argsort(np.abs(scores))[::-1][:n]
    keywords = []
    for i in top_idx:
        if tfidf_arr[i] > 0:
            direction = '🔴 Fake signal' if coef[i] < 0 else '🟢 Real signal'
            keywords.append((feature_names[i], direction))
    return keywords

# ── UI ────────────────────────────────────────────────────
st.title("📰 Fake News Detector")
st.markdown("Paste a news article or headline below to check if it's **Real** or **Fake**.")

st.markdown("---")

example_real = "The Federal Reserve held interest rates steady citing continued progress on inflation while monitoring labor market conditions closely."
example_fake = "SHOCKING: Scientists have been secretly SUPPRESSING a cure for cancer discovered 30 years ago!! The truth they don't want you to know!!"

col1, col2 = st.columns(2)
with col1:
    if st.button("📋 Try Real Example"):
        st.session_state['input_text'] = example_real
with col2:
    if st.button("🚨 Try Fake Example"):
        st.session_state['input_text'] = example_fake

user_input = st.text_area(
    "📝 Enter news article text:",
    value=st.session_state.get('input_text', ''),
    height=180,
    placeholder="Paste your news article or headline here..."
)

if st.button("🔍 Analyze", type="primary", use_container_width=True):
    if not user_input.strip():
        st.warning("Please enter some text to analyze.")
    else:
        cleaned = clean_text(user_input)
        vec = vectorizer.transform([cleaned])
        pred = model.predict(vec)[0]
        proba = model.predict_proba(vec)[0]
        confidence = proba[pred] * 100

        st.markdown("---")

        if confidence < 70:
            st.warning("## ⚠️ Uncertain Prediction (Borderline case)")
        elif pred == 1:
            st.success("## ✅ REAL News")
        else:
            st.error("## ❌ FAKE News")

        # Probability bar
        st.markdown("### 📊 Probability Breakdown")
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("🟢 Real Probability", f"{proba[1]*100:.1f}%")
        with col_b:
            st.metric("🔴 Fake Probability", f"{proba[0]*100:.1f}%")

        # Keywords
        st.markdown("### 🔑 Key Signals Detected")
        keywords = get_top_keywords(user_input)
        if keywords:
            for kw, direction in keywords:
                st.markdown(f"- **`{kw}`** → {direction}")
        else:
            st.info("No strong keyword signals found.")

        # Tips
        st.markdown("---")
        st.markdown("### 💡 What to look for in Fake News")
        st.markdown("""
        - **ALL CAPS** words and excessive exclamation marks `!!!`
        - Clickbait phrases: *"You won't believe..."*, *"SHOCKING"*, *"They don't want you to know"*
        - Vague sources or no sources at all
        - Extreme emotional language designed to provoke outrage
        - Claims that contradict established scientific consensus
        """)

st.markdown("---")
st.caption("Built with Python · Scikit-learn · TF-IDF · Logistic Regression · Streamlit")
