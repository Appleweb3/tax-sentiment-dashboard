import streamlit as st
import joblib
import pandas as pd
import re
import nltk
import plotly.express as px
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ---- Page config (must be first Streamlit call) ----
st.set_page_config(
    page_title="Tax Reform Sentiment Dashboard",
    page_icon="🇳🇬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---- Light styling for KPI cards ----
st.markdown("""
<style>
[data-testid="stMetric"] {
    background-color: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    padding: 18px 18px 14px 18px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}
[data-testid="stMetricLabel"] {
    font-size: 13px;
    font-weight: 600;
    color: #6b7280;
    text-transform: uppercase;
    letter-spacing: 0.03em;
}
[data-testid="stMetricValue"] {
    font-size: 30px;
    font-weight: 700;
    color: #111827;
}
h1 {
    font-weight: 800 !important;
    letter-spacing: -0.02em;
}
.stTabs [data-baseweb="tab"] {
    font-size: 15px;
    font-weight: 600;
}
[data-testid="stSidebar"] {
    background-color: #f9fafb;
    border-right: 1px solid #e5e7eb;
}
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
    font-weight: 700;
}
</style>
""", unsafe_allow_html=True)

# ---- NLTK setup ----
nltk.download('stopwords')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
from collections import Counter
from sklearn.metrics import accuracy_score, classification_report


def clean_text(text):
    text = str(text)
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = text.lower()
    words = [lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words]
    return " ".join(words)


# ---- Load model + data (cached so it's fast after first load) ----
@st.cache_resource
def load_model():
    model = joblib.load("sentiment_model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    return model, vectorizer


@st.cache_data
def load_data():
    df = pd.read_csv("tax_reform_labeled.csv")
    df["clean_text"] = df["text"].apply(clean_text)
    return df


model, vectorizer = load_model()
df = load_data()
df["predicted"] = model.predict(vectorizer.transform(df["clean_text"]))

if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

COLOR_MAP = {"positive": "#2ecc71", "neutral": "#95a5a6", "negative": "#e74c3c"}

# ---- Sidebar ----
with st.sidebar:
    st.header("Filters")

    if "date" in df.columns and df["date"].notna().any():
        min_d, max_d = df["date"].min(), df["date"].max()
        date_range = st.date_input("Date range", [min_d, max_d])
    else:
        date_range = None

    keyword = st.text_input("Keyword contains")

    st.markdown("---")
    st.subheader("Analyze new post")
    user_text = st.text_area("Type or paste a post here...")
    if st.button("Analyze", use_container_width=True):
        if user_text.strip():
            cleaned = clean_text(user_text)
            vect = vectorizer.transform([cleaned])
            pred = model.predict(vect)[0]
            try:
                proba = model.predict_proba(vect)[0]
                confidence = max(proba) * 100
            except AttributeError:
                confidence = None

            badge_colors = {"positive": "#2ecc71", "negative": "#e74c3c", "neutral": "#95a5a6"}
            bg = badge_colors.get(pred, "#95a5a6")
            conf_text = f" ({confidence:.0f}% confidence)" if confidence is not None else ""
            st.markdown(
                f"""
                <div style='background-color:{bg}; color:white; padding:10px 14px;
                            border-radius:8px; text-align:center; font-weight:600;
                            margin-top:8px;'>
                    Predicted sentiment: {pred.upper()}{conf_text}
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.warning("Type something first.")

# ---- Apply filters ----
filtered = df.copy()
if date_range and len(date_range) == 2:
    filtered = filtered[(filtered["date"] >= pd.to_datetime(date_range[0])) &
                         (filtered["date"] <= pd.to_datetime(date_range[1]))]
if keyword:
    filtered = filtered[filtered["text"].str.contains(keyword, case=False, na=False)]

# ---- Header ----
header_col1, header_col2 = st.columns([0.06, 0.94])
with header_col1:
    st.image("https://flagcdn.com/w80/ng.png", width=50)
with header_col2:
    st.markdown("<h1 style='margin-bottom:0;'>Tax Reform Sentiment Dashboard</h1>", unsafe_allow_html=True)
    st.caption("Tracking public opinion on Tinubu's tax reform policy")

st.markdown(
    "<hr style='margin-top:10px; margin-bottom:20px; border:none; border-top:3px solid #2ecc71;'>",
    unsafe_allow_html=True
)

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "Trends", "Sample Posts", "Live Analyzer", "About"])

# =========================== OVERVIEW ===========================
with tab1:
    if filtered.empty:
        st.warning("No posts match the current filters.")
    else:
        counts = filtered["predicted"].value_counts()
        # Share of each sentiment as a percentage of the filtered posts
        pct = filtered["predicted"].value_counts(normalize=True) * 100

        col1, col2, col3 = st.columns(3)
        col1.metric("Positive", f"{pct.get('positive', 0):.1f}%")
        col2.metric("Negative", f"{pct.get('negative', 0):.1f}%")
        col3.metric("Neutral", f"{pct.get('neutral', 0):.1f}%")

        counts_df = counts.reset_index()
        counts_df.columns = ["sentiment", "count"]
        fig = px.pie(
            counts_df, values="count", names="sentiment",
            color="sentiment", color_discrete_map=COLOR_MAP, hole=0.45,
            title="Overall Sentiment"
        )
        fig.update_traces(textinfo="percent+label")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Top Keywords by Sentiment")
        kw_cols = st.columns(3)
        for kw_col, sentiment in zip(kw_cols, ["positive", "negative", "neutral"]):
            with kw_col:
                st.markdown(f"**{sentiment.capitalize()}**")
                subset = filtered[filtered["predicted"] == sentiment]["clean_text"]
                all_words = " ".join(subset).split()
                top_words = Counter(all_words).most_common(8)
                if top_words:
                    for word, count in top_words:
                        st.write(f"- {word} ({count})")
                else:
                    st.write("_No data_")

# ============================ TRENDS ============================
with tab2:
    if filtered.empty:
        st.warning("No posts match the current filters.")
    elif "date" in filtered.columns and filtered["date"].notna().any():
        trend = (
            filtered.groupby([pd.Grouper(key="date", freq="W"), "predicted"])
            .size().unstack(fill_value=0)
        )
        trend_pct = trend.div(trend.sum(axis=1), axis=0) * 100
        fig2 = px.line(
            trend_pct, x=trend_pct.index, y=trend_pct.columns,
            color_discrete_map=COLOR_MAP,
            labels={"value": "% of posts", "date": "Date", "variable": "Sentiment"},
            title="Sentiment Trend Over Time"
        )
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No usable date column found for a trend chart.")

# ========================= SAMPLE POSTS =========================
with tab3:
    st.subheader("Sample classified posts")
    if filtered.empty:
        st.warning("No posts match the current filters.")
    else:
        st.dataframe(filtered[["text", "predicted"]].head(50), use_container_width=True)
        csv_data = filtered[["text", "predicted"]].to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download filtered results as CSV",
            data=csv_data,
            file_name="tax_reform_sentiment_filtered.csv",
            mime="text/csv",
            use_container_width=True
        )

# ========================= LIVE ANALYZER ========================
with tab4:
    st.subheader("Live Sentiment Analyzer")
    st.write("Type or paste any text below to see its predicted sentiment in real time.")

    if "history" not in st.session_state:
        st.session_state.history = []

    live_text = st.text_area("Enter text to analyze", height=120, key="live_analyzer_input")
    if st.button("Analyze Sentiment", use_container_width=True, key="live_analyze_btn"):
        if live_text.strip():
            cleaned = clean_text(live_text)
            vect = vectorizer.transform([cleaned])
            pred = model.predict(vect)[0]
            try:
                proba = model.predict_proba(vect)[0]
                confidence = max(proba) * 100
            except AttributeError:
                confidence = None

            badge_colors = {"positive": "#2ecc71", "negative": "#e74c3c", "neutral": "#95a5a6"}
            bg = badge_colors.get(pred, "#95a5a6")
            conf_text = f" ({confidence:.0f}% confidence)" if confidence is not None else ""
            st.markdown(
                f"""
                <div style='background-color:{bg}; color:white; padding:14px 18px;
                            border-radius:10px; text-align:center; font-weight:700;
                            font-size:18px; margin-top:12px;'>
                    Predicted sentiment: {pred.upper()}{conf_text}
                </div>
                """,
                unsafe_allow_html=True
            )
            st.session_state.history.insert(0, {"text": live_text, "sentiment": pred})
        else:
            st.warning("Please type something first.")

    if st.session_state.history:
        st.markdown("#### Recent analyses (this session)")
        for item in st.session_state.history[:10]:
            color = {"positive": "green", "negative": "red", "neutral": "gray"}.get(item["sentiment"], "gray")
            st.markdown(f":{color}[**{item['sentiment'].upper()}**] — {item['text'][:100]}")

# ============================== ABOUT ===========================
with tab5:
    st.subheader("About this project")
    st.markdown(f"""
**Tax Reform Sentiment Dashboard** analyzes public sentiment toward Nigeria's
tax reform policy using posts collected from social media and other public sources.

**Data Source**
Posts were collected and manually/automatically labeled as positive, negative,
or neutral, saved in `tax_reform_labeled.csv`. The dataset contains {len(df)} posts.

**Methodology**
- Text is cleaned (URLs, mentions, hashtags, punctuation removed; lowercased; lemmatized)
- A TF-IDF vectorizer converts cleaned text into numerical features
- A trained Support Vector Machine (SVM) classifier predicts sentiment

**Tools & Libraries**
- Python, Streamlit, pandas, scikit-learn, NLTK, Plotly

**Limitations**
- Predictions are automated and may not always reflect nuanced human sentiment
- The dataset is small ({len(df)} posts), so results are indicative rather than conclusive
- Dataset size and labeling quality directly affect model accuracy

---

**Author**
Usman Hayatudeen

**Institution**
Kaduna State University — Faculty of Computing, Department of Informatics

**Supervisor**
Mrs. Amina Isa

This dashboard was developed as part of a final year project.
""")

    st.markdown("---")
    st.subheader("Model Performance")
    if "label" in df.columns:
        valid = df["label"].notna() & df["predicted"].notna()
        y_true = df.loc[valid, "label"].astype(str).str.strip().str.lower()
        y_pred = df.loc[valid, "predicted"].astype(str).str.strip().str.lower()

        acc = accuracy_score(y_true, y_pred)
        st.metric("Overall Accuracy", f"{acc * 100:.1f}%")

        report = classification_report(y_true, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose().round(2)
        st.dataframe(report_df, use_container_width=True)

        st.caption(
            "Accuracy, precision, recall, and F1-score computed by comparing "
            "the model's predictions against the labeled data in tax_reform_labeled.csv."
        )
    else:
        st.info("No ground-truth 'label' column found to compute evaluation metrics.")
