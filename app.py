# =============================================================================
# Tax Reform Sentiment Dashboard
# Author: Usman Hayatudeen
#
# This Streamlit app loads a trained SVM sentiment model, applies it to a
# labelled dataset of social media posts about Nigeria's tax reform policy,
# and shows the results as an interactive dashboard.
# =============================================================================

# ---- Libraries used in this app ----
import streamlit as st                       # Streamlit: builds the web dashboard
import joblib                                # joblib: loads the saved model files
import pandas as pd                          # pandas: handles the dataset (tables)
import re                                    # re: pattern matching for text cleaning
import nltk                                  # nltk: natural language toolkit
import plotly.express as px                  # plotly: interactive charts
from nltk.corpus import stopwords            # list of common words to remove
from nltk.stem import WordNetLemmatizer      # reduces words to their root form

# ---- Page config (must be the first Streamlit call) ----
# Sets the browser tab title, icon, wide layout and an open sidebar
st.set_page_config(
    page_title="Tax Reform Sentiment Dashboard",
    page_icon="🇳🇬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---- Custom styling (CSS) ----
# Gives the metric cards, headings, tabs and sidebar a cleaner look
st.markdown("""
<style>
/* Metric (KPI) cards: white box with a soft border and shadow */
[data-testid="stMetric"] {
    background-color: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    padding: 18px 18px 14px 18px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}
/* Small grey uppercase label above each metric value */
[data-testid="stMetricLabel"] {
    font-size: 13px;
    font-weight: 600;
    color: #6b7280;
    text-transform: uppercase;
    letter-spacing: 0.03em;
}
/* Large bold metric value */
[data-testid="stMetricValue"] {
    font-size: 30px;
    font-weight: 700;
    color: #111827;
}
/* Main page title */
h1 {
    font-weight: 800 !important;
    letter-spacing: -0.02em;
}
/* Tab labels */
.stTabs [data-baseweb="tab"] {
    font-size: 15px;
    font-weight: 600;
}
/* Sidebar background and border */
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
# Download the word lists NLTK needs (only downloads if not already present)
nltk.download('stopwords')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))   # common words with little meaning (the, is, and)
lemmatizer = WordNetLemmatizer()               # turns words into root form (e.g. "taxes" -> "tax")
from collections import Counter                # used later to count the most frequent words
from sklearn.metrics import accuracy_score, classification_report  # used for model performance


# ---- Text cleaning function ----
# Takes a raw post and returns a cleaned version. It must match the cleaning
# used when the model was trained, otherwise predictions will be unreliable.
def clean_text(text):
    text = str(text)                                  # make sure the input is text
    text = re.sub(r"http\S+|www\S+", "", text)        # remove web links
    text = re.sub(r"@\w+", "", text)                  # remove @mentions
    text = re.sub(r"#", "", text)                     # remove the # symbol, keep the word
    text = re.sub(r"[^a-zA-Z\s]", "", text)           # remove numbers, punctuation and emojis
    text = text.lower()                               # convert everything to lowercase
    # Split into words, drop stopwords, and reduce each remaining word to its root
    words = [lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words]
    return " ".join(words)                            # join the words back into one string


# ---- Load model + data (cached so it's fast after the first load) ----
# @st.cache_resource keeps the model in memory so it isn't reloaded on every click
@st.cache_resource
def load_model():
    model = joblib.load("sentiment_model.pkl")        # the trained SVM classifier
    vectorizer = joblib.load("vectorizer.pkl")        # the TF-IDF vectorizer (text -> numbers)
    return model, vectorizer


# @st.cache_data keeps the cleaned dataset in memory for the same reason
@st.cache_data
def load_data():
    df = pd.read_csv("tax_reform_labeled.csv")        # columns: text, label, date
    df["clean_text"] = df["text"].apply(clean_text)   # add a cleaned version of each post
    return df


model, vectorizer = load_model()
df = load_data()

# Predict the sentiment of every post: convert cleaned text to numbers with the
# vectorizer, then let the model classify them (positive / negative / neutral)
df["predicted"] = model.predict(vectorizer.transform(df["clean_text"]))

# Convert the date column to proper dates (invalid dates become blank instead of crashing)
if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

# Colours used for each sentiment in the charts
COLOR_MAP = {"positive": "#2ecc71", "neutral": "#95a5a6", "negative": "#e74c3c"}

# =============================================================================
# SIDEBAR: filters and the quick "Analyze new post" box
# =============================================================================
with st.sidebar:
    st.header("Filters")

    # Date range picker (only shown when the dataset has valid dates)
    if "date" in df.columns and df["date"].notna().any():
        min_d, max_d = df["date"].min(), df["date"].max()
        date_range = st.date_input("Date range", [min_d, max_d])
    else:
        date_range = None

    # Keyword filter: only keep posts containing this word
    keyword = st.text_input("Keyword contains")

    st.markdown("---")
    st.subheader("Analyze new post")
    user_text = st.text_area("Type or paste a post here...")

    # When the button is clicked, classify the typed post
    if st.button("Analyze", use_container_width=True):
        if user_text.strip():                          # only run if something was typed
            cleaned = clean_text(user_text)            # clean it the same way as the training data
            vect = vectorizer.transform([cleaned])     # convert to numbers
            pred = model.predict(vect)[0]              # predict the sentiment
            # Confidence is only available if the model supports probabilities
            try:
                proba = model.predict_proba(vect)[0]
                confidence = max(proba) * 100
            except AttributeError:
                confidence = None

            # Pick a badge colour that matches the predicted sentiment
            badge_colors = {"positive": "#2ecc71", "negative": "#e74c3c", "neutral": "#95a5a6"}
            bg = badge_colors.get(pred, "#95a5a6")
            conf_text = f" ({confidence:.0f}% confidence)" if confidence is not None else ""
            # Show the result as a coloured badge
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

# =============================================================================
# APPLY FILTERS
# "filtered" is the version of the data after the sidebar filters are applied.
# Every tab below uses it, so the whole dashboard updates with the filters.
# =============================================================================
filtered = df.copy()
if date_range and len(date_range) == 2:
    # Keep only posts between the chosen start and end dates
    filtered = filtered[(filtered["date"] >= pd.to_datetime(date_range[0])) &
                         (filtered["date"] <= pd.to_datetime(date_range[1]))]
if keyword:
    # Keep only posts whose text contains the keyword (ignoring capital letters)
    filtered = filtered[filtered["text"].str.contains(keyword, case=False, na=False)]

# =============================================================================
# HEADER: flag, title, subtitle and green divider line
# =============================================================================
header_col1, header_col2 = st.columns([0.06, 0.94])    # narrow column for the flag, wide one for the title
with header_col1:
    st.image("https://flagcdn.com/w80/ng.png", width=50)
with header_col2:
    st.markdown("<h1 style='margin-bottom:0;'>Tax Reform Sentiment Dashboard</h1>", unsafe_allow_html=True)
    st.caption("Tracking public opinion on Tinubu's tax reform policy")

st.markdown(
    "<hr style='margin-top:10px; margin-bottom:20px; border:none; border-top:3px solid #2ecc71;'>",
    unsafe_allow_html=True
)

# Create the five tabs shown under the header
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "Trends", "Sample Posts", "Live Analyzer", "About"])

# =============================================================================
# TAB 1: OVERVIEW - sentiment percentages, pie chart and top keywords
# =============================================================================
with tab1:
    if filtered.empty:
        st.warning("No posts match the current filters.")
    else:
        # Number of posts per sentiment (used for the pie chart)
        counts = filtered["predicted"].value_counts()
        # Share of each sentiment as a percentage of all filtered posts
        pct = filtered["predicted"].value_counts(normalize=True) * 100

        # Three metric cards side by side, each showing a percentage to 1 decimal place
        col1, col2, col3 = st.columns(3)
        col1.metric("Positive", f"{pct.get('positive', 0):.1f}%")
        col2.metric("Negative", f"{pct.get('negative', 0):.1f}%")
        col3.metric("Neutral", f"{pct.get('neutral', 0):.1f}%")

        # Prepare the counts as a small table, then draw a donut (pie) chart
        counts_df = counts.reset_index()
        counts_df.columns = ["sentiment", "count"]
        fig = px.pie(
            counts_df, values="count", names="sentiment",
            color="sentiment", color_discrete_map=COLOR_MAP, hole=0.45,
            title="Overall Sentiment"
        )
        fig.update_traces(textinfo="percent+label")    # show the percentage and name on each slice
        st.plotly_chart(fig, use_container_width=True)

        # Show the 8 most common words in each sentiment group
        st.markdown("#### Top Keywords by Sentiment")
        kw_cols = st.columns(3)
        for kw_col, sentiment in zip(kw_cols, ["positive", "negative", "neutral"]):
            with kw_col:
                st.markdown(f"**{sentiment.capitalize()}**")
                # Take the cleaned text of posts predicted as this sentiment
                subset = filtered[filtered["predicted"] == sentiment]["clean_text"]
                all_words = " ".join(subset).split()           # one big list of words
                top_words = Counter(all_words).most_common(8)   # count and keep the top 8
                if top_words:
                    for word, count in top_words:
                        st.write(f"- {word} ({count})")
                else:
                    st.write("_No data_")

# =============================================================================
# TAB 2: TRENDS - how sentiment changes over time
# =============================================================================
with tab2:
    if filtered.empty:
        st.warning("No posts match the current filters.")
    elif "date" in filtered.columns and filtered["date"].notna().any():
        # Group posts by week and sentiment, counting how many fall in each group
        trend = (
            filtered.groupby([pd.Grouper(key="date", freq="W"), "predicted"])
            .size().unstack(fill_value=0)
        )
        # Convert the counts to percentages of each week's posts
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

# =============================================================================
# TAB 3: SAMPLE POSTS - table of classified posts with a download button
# =============================================================================
with tab3:
    st.subheader("Sample classified posts")
    if filtered.empty:
        st.warning("No posts match the current filters.")
    else:
        # Show the first 50 filtered posts with their predicted sentiment
        st.dataframe(filtered[["text", "predicted"]].head(50), use_container_width=True)
        # Let the user download all filtered results as a CSV file
        csv_data = filtered[["text", "predicted"]].to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download filtered results as CSV",
            data=csv_data,
            file_name="tax_reform_sentiment_filtered.csv",
            mime="text/csv",
            use_container_width=True
        )

# =============================================================================
# TAB 4: LIVE ANALYZER - classify any text typed by the user
# =============================================================================
with tab4:
    st.subheader("Live Sentiment Analyzer")
    st.write("Type or paste any text below to see its predicted sentiment in real time.")

    # session_state remembers values between clicks; here it stores recent analyses
    if "history" not in st.session_state:
        st.session_state.history = []

    live_text = st.text_area("Enter text to analyze", height=120, key="live_analyzer_input")
    if st.button("Analyze Sentiment", use_container_width=True, key="live_analyze_btn"):
        if live_text.strip():
            cleaned = clean_text(live_text)             # clean the text
            vect = vectorizer.transform([cleaned])      # convert to numbers
            pred = model.predict(vect)[0]               # predict the sentiment
            # Confidence is only available if the model supports probabilities
            try:
                proba = model.predict_proba(vect)[0]
                confidence = max(proba) * 100
            except AttributeError:
                confidence = None

            badge_colors = {"positive": "#2ecc71", "negative": "#e74c3c", "neutral": "#95a5a6"}
            bg = badge_colors.get(pred, "#95a5a6")
            conf_text = f" ({confidence:.0f}% confidence)" if confidence is not None else ""
            # Show the result as a large coloured badge
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
            # Save this result at the top of the session history
            st.session_state.history.insert(0, {"text": live_text, "sentiment": pred})
        else:
            st.warning("Please type something first.")

    # List the 10 most recent analyses from this browsing session
    if st.session_state.history:
        st.markdown("#### Recent analyses (this session)")
        for item in st.session_state.history[:10]:
            color = {"positive": "green", "negative": "red", "neutral": "gray"}.get(item["sentiment"], "gray")
            st.markdown(f":{color}[**{item['sentiment'].upper()}**] — {item['text'][:100]}")

# =============================================================================
# TAB 5: ABOUT - project description and model performance
# =============================================================================
with tab5:
    st.subheader("About this project")
    # The f-string lets us insert the live dataset size, {len(df)}, into the text
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

    # ---- Model performance ----
    # Compares the model's predictions with the human labels in the CSV
    st.markdown("---")
    st.subheader("Model Performance")
    if "label" in df.columns:
        # Keep only rows that have both a human label and a prediction
        valid = df["label"].notna() & df["predicted"].notna()
        # Tidy both columns (text, no extra spaces, lowercase) so they compare fairly
        y_true = df.loc[valid, "label"].astype(str).str.strip().str.lower()
        y_pred = df.loc[valid, "predicted"].astype(str).str.strip().str.lower()

        # Overall accuracy: the share of posts the model labelled correctly
        acc = accuracy_score(y_true, y_pred)
        st.metric("Overall Accuracy", f"{acc * 100:.1f}%")

        # Precision, recall and F1-score for each sentiment class
        report = classification_report(y_true, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose().round(2)
        st.dataframe(report_df, use_container_width=True)

        st.caption(
            "Accuracy, precision, recall, and F1-score computed by comparing "
            "the model's predictions against the labeled data in tax_reform_labeled.csv."
        )
    else:
        st.info("No ground-truth 'label' column found to compute evaluation metrics.")
