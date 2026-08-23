import streamlit as st
import joblib
import pandas as pd
import re
import nltk
import numpy as np
import plotly.express as px
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

st.set_page_config(page_title="Tax Reform Sentiment Tracker", page_icon="🇳🇬", layout="wide")

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
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
df["clean_text"] = df["text"].apply(clean_text)
df["predicted"] = model.predict(vectorizer.transform(df["clean_text"]))

if "date" not in df.columns:
    df["date"] = pd.date_range(end=pd.Timestamp.today(), periods=len(df), freq="h")
else:
    df["date"] = pd.to_datetime(df["date"], format="mixed", dayfirst=True, errors="coerce")
    df = df.dropna(subset=["date"])

st.markdown("## 🇳🇬 Tax Reform Sentiment Tracker")
st.caption("Public sentiment on Bola Ahmed Tinubu's Tax Reform Policy — Nigeria Focus")

tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Trends", "Live Analyzer", "About"])

with tab1:
    col1, col2, col3 = st.columns([1.2, 1, 1])
    counts = df["predicted"].value_counts()
    total = len(df)

    with col1:
        st.markdown("**Overall Sentiment**")
        fig = px.pie(
            values=counts.values, names=counts.index,
            color=counts.index,
            color_discrete_map={"positive": "green", "negative": "red", "neutral": "grey"},
            hole=0.4
        )
        fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=280)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.metric("Total Posts Analyzed", f"{total:,}")
        neg_by_week = df.set_index("date")["predicted"].eq("negative").resample("W").mean()
        if len(neg_by_week) > 0:
            peak_week = neg_by_week.idxmax()
            st.metric("Negative Sentiment Peak", peak_week.strftime("%b %Y"))

    with col3:
        neg_share = round(counts.get("negative", 0) / total * 100, 1)
        st.metric("Negative Sentiment Share", f"{neg_share}%")
        pos_share = round(counts.get("positive", 0) / total * 100, 1)
        st.metric("Positive Sentiment Share", f"{pos_share}%")

with tab2:
    st.markdown("**Sentiment Trend Over Time**")
    trend = df.groupby([pd.Grouper(key="date", freq="W"), "predicted"]).size().unstack(fill_value=0)
    trend_pct = trend.div(trend.sum(axis=1), axis=0) * 100
    fig2 = px.line(trend_pct, x=trend_pct.index, y=trend_pct.columns,
                   color_discrete_map={"positive": "green", "negative": "red", "neutral": "grey"})
    fig2.update_layout(yaxis_title="Share of Posts (%)", xaxis_title="Week", height=400)
    st.plotly_chart(fig2, use_container_width=True)

with tab3:
    st.markdown("**Analyze a new post**")
    user_text = st.text_area("Type or paste a post here:")
    if st.button("Analyze"):
        if user_text:
            cleaned = clean_text(user_text)
            pred = model.predict(vectorizer.transform([cleaned]))[0]
            color = {"positive": "green", "negative": "red", "neutral": "grey"}[pred]
            st.markdown(f"Predicted sentiment: **:{color}[{pred.upper()}]**")

with tab4:
    st.markdown("""
    ### About this project
    This dashboard is part of a final year project titled **"A Sentiment Analysis Model for Predicting
    Public Opinion on President Bola Ahmed Tinubu's Tax Reform Policy Using Machine Learning Techniques."**

    - **Student:** Usman Hayatudeen (KASU/22/CSC/1127)
    - **Department:** Informatics, Faculty of Computing, Kaduna State University
    - **Supervisor:** Mrs. Amina Isa
    - **Model:** Support Vector Machine (SVM) trained on TF-IDF features
    """)
