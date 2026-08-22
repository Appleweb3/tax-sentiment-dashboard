import streamlit as st
import joblib
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = str(text)
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = text.lower()
    words = [lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words]
    return " ".join(words)

model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")
df = pd.read_csv("tax_reform_labeled.csv")
df["clean_text"] = df["Text"].apply(clean_text)
df["predicted"] = model.predict(vectorizer.transform(df["clean_text"]))

st.title("Tax Reform Sentiment Dashboard")
st.caption("Tracking public opinion on Tinubu's tax reform policy")

counts = df["predicted"].value_counts()
col1, col2, col3 = st.columns(3)
col1.metric("Positive", f"{counts.get('positive',0)}")
col2.metric("Negative", f"{counts.get('negative',0)}")
col3.metric("Neutral", f"{counts.get('neutral',0)}")

st.bar_chart(counts)

st.subheader("Try it yourself")
user_text = st.text_input("Type a sentence about the tax reform:")
if user_text:
    cleaned = clean_text(user_text)
    pred = model.predict(vectorizer.transform([cleaned]))[0]
    st.write(f"Predicted sentiment: **{pred}**")

st.subheader("Sample classified posts")
st.dataframe(df[["Text", "predicted"]].head(20))