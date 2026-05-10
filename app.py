import streamlit as st
import joblib
import re
import string
import matplotlib.pyplot as plt


# ==============================
# Load saved model files
# ==============================
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
suspicious_words = joblib.load("suspicious_words.pkl")


# ==============================
# Text cleaning function
# ==============================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"<.*?>+", "", text)
    text = re.sub(r"[%s]" % re.escape(string.punctuation), "", text)
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"\w*\d\w*", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ==============================
# Highlight suspicious words
# ==============================
def highlight_words(headline):
    words = headline.split()
    highlighted = []

    for word in words:
        cleaned_word = clean_text(word)

        if cleaned_word in suspicious_words:
            highlighted.append(
                f"<span style='background-color:#ffcccc; color:#b30000; padding:3px; border-radius:5px;'>{word}</span>"
            )
        else:
            highlighted.append(word)

    return " ".join(highlighted)


# ==============================
# Streamlit UI
# ==============================
st.set_page_config(
    page_title="Fake News Headline Detector",
    page_icon="📰",
    layout="centered"
)

st.title("📰 Fake News Headline Detector")
st.write("Enter a news headline to check whether it is likely **REAL** or **FAKE**.")

headline = st.text_area(
    "Enter News Headline:",
    placeholder="Example: Breaking news: Government announces new education policy"
)

if st.button("Check Headline"):

    if headline.strip() == "":
        st.warning("Please enter a headline first.")

    else:

        cleaned = clean_text(headline)

        vectorized = vectorizer.transform([cleaned])

        prediction = model.predict(vectorized)[0]

        probability = model.predict_proba(vectorized)[0]

        label = "FAKE" if prediction == 1 else "REAL"

        confidence = max(probability) * 100

        st.subheader("Prediction Result")

        if label == "FAKE":
            st.error(f"Verdict: {label}")
        else:
            st.success(f"Verdict: {label}")

        st.write(f"Confidence: **{confidence:.2f}%**")

        st.progress(int(confidence))


        # ==============================
        # Prediction Probability Pie Chart
        # ==============================

        fake_prob = probability[1] * 100
        real_prob = probability[0] * 100

        fig, ax = plt.subplots(figsize=(4,4))

        ax.pie(
            [real_prob, fake_prob],
            labels=["REAL", "FAKE"],
            autopct="%1.1f%%",
            startangle=90
        )

        ax.set_title("Prediction Probability")

        st.pyplot(fig)


        # ==============================
        # Suspicious Word Highlighting
        # ==============================

        st.subheader("Suspicious Word Highlighting")

        highlighted_headline = highlight_words(headline)

        st.markdown(
            highlighted_headline,
            unsafe_allow_html=True
        )


        # ==============================
        # Disclaimer
        # ==============================

        st.info(
            "Disclaimer: This tool provides automated predictions only. "
            "It cannot replace professional fact-checking by trained journalists or trusted news agencies."
        )
# ==============================
# Sidebar Section
# ==============================

with st.sidebar:

    st.sidebar.markdown("## 👨‍💻 Developer Info")

st.sidebar.markdown("""
### Md. Tareque Hasan

🎓 Industrial & Production Engineering Graduate from DUET

💡 Interests:
- Machine Learning
- NLP
- Data Analytics
- AI Applications

🚀 Focused on:
- Intelligent Systems
- Manufacturing Analytics
- Real-world AI Solutions

📍 Gazipur, Dhaka, Bangladesh
""")
    
st.markdown("---")

st.title("📰 Project Info")

st.markdown("""
    ### Fake News Headline Detector
    
    This project uses:
    
    ✅ TF-IDF Feature Extraction  
    ✅ Logistic Regression Model  
    ✅ Naive Bayes Comparison  
    ✅ Suspicious Word Highlighting  
    ✅ Streamlit GUI  
    """)

st.markdown("---")

st.title("📊 Model Performance")

st.markdown("""
    - Accuracy: **93.88%**
    - F1 Score: **94.13%**
    - AUC-ROC: **98.42%**
    """)

st.markdown("---")

st.info(
        "This tool assists users in identifying potentially fake news headlines using Machine Learning."
    )
    # ==============================
# Model Performance Chart
# ==============================

st.sidebar.markdown("---")

st.sidebar.title("📈 Performance Graph")

metrics = {
    "Accuracy": 93.88,
    "F1 Score": 94.13,
    "AUC-ROC": 98.42
}

st.sidebar.bar_chart(metrics)