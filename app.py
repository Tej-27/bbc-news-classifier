import streamlit as st
import joblib
import re


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="BBC News Classifier",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# NOTE: Only CSS is injected as HTML. The UI content itself
# uses native Streamlit components to avoid raw HTML rendering.
# ============================================================

st.markdown(
    """
    <style>
        .block-container {
            max-width: 1200px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        .hero-box {
            border: 1px solid rgba(128,128,128,0.22);
            border-radius: 18px;
            padding: 1.8rem 2rem;
            margin-bottom: 1.5rem;
            background: linear-gradient(
                135deg,
                rgba(49, 51, 63, 0.10),
                rgba(49, 51, 63, 0.025)
            );
        }

        .hero-subtitle {
            font-size: 1.05rem;
            line-height: 1.6;
            opacity: 0.75;
            margin-top: 0.4rem;
        }

        .prediction-box {
            border: 1px solid rgba(128,128,128,0.25);
            border-radius: 16px;
            padding: 1.4rem 1.5rem;
            margin-top: 1rem;
            margin-bottom: 1rem;
            background: rgba(128,128,128,0.035);
        }

        .prediction-label {
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 1.2px;
            opacity: 0.65;
        }

        .prediction-title {
            font-size: 2rem;
            font-weight: 700;
            margin-top: 0.35rem;
        }

        .prediction-description {
            font-size: 1rem;
            line-height: 1.55;
            opacity: 0.78;
            margin-top: 0.45rem;
        }

        .footer-line {
            margin-top: 3rem;
            padding-top: 1rem;
            border-top: 1px solid rgba(128,128,128,0.20);
            text-align: center;
            opacity: 0.55;
            font-size: 0.85rem;
        }

        @media (max-width: 768px) {
            .block-container {
                padding-left: 1rem;
                padding-right: 1rem;
                padding-top: 1rem;
            }

            .hero-box {
                padding: 1.25rem;
            }

            .hero-subtitle {
                font-size: 0.95rem;
            }

            .prediction-title {
                font-size: 1.6rem;
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_models():
    model = joblib.load("bbc_news_classifier.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    return model, vectorizer


try:
    model, vectorizer = load_models()
except FileNotFoundError:
    st.error(
        "Model files not found. Make sure these files are in the "
        "same folder as app.py:"
    )
    st.code(
        "bbc_news_classifier.pkl\n"
        "tfidf_vectorizer.pkl"
    )
    st.stop()


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):
    text = text.lower()

    # Remove HTML tags
    text = re.sub(r"<.*?>", " ", text)

    # Remove HTML entities such as &#163;
    text = re.sub(r"&[a-zA-Z0-9#]+;", " ", text)

    # Keep letters and spaces
    text = re.sub(r"[^a-z\s]", " ", text)

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ============================================================
# CATEGORY INFORMATION
# ============================================================

category_info = {
    "business": {
        "icon": "💼",
        "description": (
            "Companies, markets, finance, economics, investments, "
            "and the business world."
        ),
    },
    "entertainment": {
        "icon": "🎬",
        "description": (
            "Movies, music, celebrities, television, arts, culture, "
            "and entertainment events."
        ),
    },
    "politics": {
        "icon": "🏛️",
        "description": (
            "Governments, elections, political parties, public policy, "
            "and political affairs."
        ),
    },
    "sport": {
        "icon": "⚽",
        "description": (
            "Sports, athletes, competitions, teams, tournaments, "
            "and sporting events."
        ),
    },
    "tech": {
        "icon": "💻",
        "description": (
            "Technology, software, hardware, the internet, gadgets, "
            "and digital innovation."
        ),
    },
}


# ============================================================
# EXAMPLE ARTICLES
# ============================================================

examples = {
    "⚽ Sport": (
        "The football team secured a dramatic victory in the final "
        "match after scoring a late goal."
    ),
    "🏛️ Politics": (
        "The government announced a new policy regarding national "
        "elections and public administration."
    ),
    "💼 Business": (
        "The company reported strong profits and increased revenue "
        "this quarter as demand continued to grow."
    ),
    "🎬 Entertainment": (
        "The singer announced a new album that will be released "
        "next month along with a nationwide tour."
    ),
    "💻 Tech": (
        "The technology company unveiled a new smartphone with an "
        "advanced processor and improved camera."
    ),
}


# ============================================================
# SESSION STATE
# ============================================================

if "article_input" not in st.session_state:
    st.session_state.article_input = ""


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.title("📰 BBC News Classifier")
    st.caption("Machine Learning Text Classification")

    st.divider()

    st.subheader("📊 Model Information")

    st.markdown("**Algorithm**")
    st.write("Linear Support Vector Machine")

    st.markdown("**Text Representation**")
    st.write("TF-IDF")

    st.markdown("**Features**")
    st.write("Unigrams + Bigrams")

    st.markdown("**Training Samples**")
    st.write("1,701")

    st.markdown("**Test Samples**")
    st.write("426")

    st.markdown("**Categories**")
    st.write("5")

    st.divider()

    st.subheader("📈 Model Performance")

    metric_col1, metric_col2 = st.columns(2)

    with metric_col1:
        st.metric("Accuracy", "97.65%")

    with metric_col2:
        st.metric("F1 Score", "0.98")

    st.caption("Measured on the held-out test set.")

    st.divider()

    st.subheader("🧠 Models Evaluated")

    st.write("• Multinomial Naive Bayes — 96.48%")
    st.write("• Logistic Regression — 97.42%")
    st.write("• Linear SVM — 97.65%")

    st.divider()

    st.caption("Built with Python, Scikit-learn and Streamlit.")


# ============================================================
# HERO SECTION
# Native Streamlit components are used here instead of HTML.
# ============================================================

st.markdown('<div class="hero-box">', unsafe_allow_html=True)

st.title("📰 BBC News Category Classifier")

st.markdown(
    '<div class="hero-subtitle">'
    "An NLP machine learning application that analyzes news articles "
    "and classifies them into one of five categories using TF-IDF "
    "and a Linear SVM model."
    "</div>",
    unsafe_allow_html=True,
)

st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# ARTICLE INPUT
# ============================================================

st.subheader("📝 Enter a News Article")

st.write(
    "Paste the article text below and click **Classify Article** "
    "to predict its category."
)

st.markdown("#### 💡 Try an Example")

example_columns = st.columns(5)

for column, (label, example_text) in zip(
    example_columns,
    examples.items(),
):
    with column:
        if st.button(label, use_container_width=True):
            st.session_state.article_input = example_text


# ============================================================
# TEXT AREA
# ============================================================

st.text_area(
    "News Article",
    key="article_input",
    height=280,
    placeholder=(
        "Paste a news article here...\n\n"
        "For example: The company announced a new technology product..."
    ),
    label_visibility="collapsed",
)


# ============================================================
# ACTION BUTTONS
# ============================================================

button_col1, button_col2 = st.columns([3, 1])

with button_col1:
    classify_clicked = st.button(
        "🔍 Classify Article",
        type="primary",
        use_container_width=True,
    )

with button_col2:
    clear_clicked = st.button(
        "🧹 Clear",
        use_container_width=True,
    )


# ============================================================
# CLEAR BUTTON
# ============================================================

if clear_clicked:
    st.session_state.article_input = ""
    st.rerun()


# ============================================================
# CLASSIFICATION
# ============================================================

if classify_clicked:
    article = st.session_state.article_input.strip()

    if not article:
        st.warning(
            "⚠️ Please enter or paste a news article first."
        )

    elif len(article.split()) < 8:
        st.warning(
            "⚠️ The article is too short for reliable classification. "
            "Please provide at least 8 words of text."
        )

    else:
        cleaned_article = clean_text(article)

        if not cleaned_article:
            st.error(
                "Unable to process the provided text. "
                "Please enter a valid news article."
            )

        else:
            article_vector = vectorizer.transform([cleaned_article])
            prediction = model.predict(article_vector)[0]
            info = category_info[prediction]

            # Native Streamlit container avoids HTML rendering issues.
            with st.container(border=True):
                st.caption("PREDICTED CATEGORY")
                st.subheader(
                    f'{info["icon"]} {prediction.title()}'
                )
                st.write(info["description"])

            st.info(
                "Prediction generated using TF-IDF features and "
                "the trained Linear SVM classifier."
            )


# ============================================================
# SUPPORTED CATEGORIES
# ============================================================

st.subheader("📚 Supported Categories")

category_columns = st.columns(5)

for column, (category, info) in zip(
    category_columns,
    category_info.items(),
):
    with column:
        with st.container(border=True):
            st.markdown(
                f"### {info['icon']} {category.title()}"
            )
            st.caption(info["description"])


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    '<div class="footer-line">'
    "BBC News Category Classifier · TF-IDF + Linear SVM · "
    "Built with Python & Streamlit"
    "</div>",
    unsafe_allow_html=True,
)
