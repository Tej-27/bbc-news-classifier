# 📰 BBC News Category Classifier

A machine learning-based **NLP text classification application** that automatically categorizes news articles into one of five BBC News categories:

- 💼 Business
- 🎬 Entertainment
- 🏛️ Politics
- ⚽ Sport
- 💻 Tech

The project uses **TF-IDF** for text representation and a **Linear Support Vector Machine (Linear SVM)** for classification. The trained model is deployed as an interactive **Streamlit web application**.

## 🚀 Live Demo

👉 **[Open the BBC News Classifier](https://bbcnewsclassifier.streamlit.app/)**

You can paste a news article into the application and receive its predicted category instantly.

---

## 📌 Project Overview

News websites publish a large amount of content every day across different topics. Manually organizing these articles into categories can be time-consuming.

This project demonstrates how Natural Language Processing and supervised machine learning can be used to automatically classify news articles based on their textual content.

The complete workflow includes:

1. Dataset collection
2. Data loading and exploration
3. Duplicate detection and removal
4. Text preprocessing
5. Train-test splitting
6. TF-IDF feature extraction
7. Training multiple classification algorithms
8. Model evaluation
9. Error analysis
10. Model serialization
11. Streamlit application development
12. Cloud deployment

---

## 🎯 Problem Statement

Given the text of a news article, predict which category it belongs to.

### Input

A news article such as:

> "The football club secured a dramatic victory after scoring a late goal..."

### Output

```text
Predicted Category: Sport
```

The classifier supports five categories:

| Category | Description |
|---|---|
| 💼 Business | Companies, markets, finance, economics and corporate news |
| 🎬 Entertainment | Movies, music, celebrities, television and entertainment events |
| 🏛️ Politics | Government, elections, politicians, policies and political events |
| ⚽ Sport | Football, cricket, tennis and other sports-related news |
| 💻 Tech | Technology, software, hardware, internet and digital developments |

---

## 📊 Dataset

The project uses the **BBC News Summary dataset**, containing BBC news articles organized into five categories.

### Categories

- Business
- Entertainment
- Politics
- Sport
- Tech

### Dataset Statistics

| Metric | Value |
|---|---:|
| Original articles | 2,225 |
| Duplicate articles | 98 |
| Articles after duplicate removal | 2,127 |
| Training samples | 1,701 |
| Testing samples | 426 |
| Number of categories | 5 |

### Class Distribution After Duplicate Removal

| Category | Articles |
|---|---:|
| Sport | 511 |
| Business | 510 |
| Politics | 417 |
| Tech | 401 |
| Entertainment | 386 |

The dataset is relatively balanced across the five categories, although Sport and Business contain more articles than Entertainment.

---

## 🔍 Exploratory Data Analysis

Several aspects of the dataset were analyzed before model training.

### Article Length

The article lengths were analyzed using both character count and word count.

| Statistic | Word Count |
|---|---:|
| Mean | ~384 |
| Median | 332 |
| 25th percentile | 246 |
| 75th percentile | 471 |
| Minimum | 89 |
| Maximum | 4,432 |

### Average Article Length by Category

| Category | Approx. Average Words |
|---|---:|
| Tech | 503 |
| Politics | 454 |
| Entertainment | 331 |
| Sport | 329 |
| Business | 329 |

The exploratory analysis showed that Tech and Politics articles tended to be longer on average, while Business, Sport and Entertainment articles were generally shorter.

---

## 🧹 Text Preprocessing

Raw article text cannot be directly passed to most traditional machine learning algorithms. Therefore, the text was cleaned before feature extraction.

The preprocessing pipeline includes:

### 1. Convert text to lowercase

```python
text = text.lower()
```

This ensures that words such as:

```text
Technology
technology
TECHNOLOGY
```

are treated consistently.

### 2. Remove HTML tags

```python
text = re.sub(r"<.*?>", " ", text)
```

### 3. Remove HTML entities

```python
text = re.sub(r"&[a-zA-Z0-9#]+;", " ", text)
```

### 4. Remove non-alphabetic characters

```python
text = re.sub(r"[^a-z\s]", " ", text)
```

### 5. Normalize whitespace

```python
text = re.sub(r"\s+", " ", text).strip()
```

### Complete Cleaning Function

```python
import re

def clean_text(text):
    text = text.lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"&[a-zA-Z0-9#]+;", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
```

---

## ✂️ Train-Test Split

The dataset was divided into training and testing sets using an 80/20 split.

```python
from sklearn.model_selection import train_test_split

X = df["clean_text"]
y = df["category"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
```

### Split

- Training: **1,701 articles**
- Testing: **426 articles**

`stratify=y` was used to preserve the relative class distribution in both datasets.

---

# 🧠 Feature Engineering with TF-IDF

Machine learning algorithms cannot directly understand raw text. The cleaned articles were converted into numerical feature vectors using **TF-IDF (Term Frequency-Inverse Document Frequency)**.

```python
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True,
    stop_words="english"
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)
```

### TF-IDF Configuration

| Parameter | Value | Purpose |
|---|---|---|
| `max_features` | 10,000 | Limits the vocabulary size |
| `ngram_range` | `(1, 2)` | Uses unigrams and bigrams |
| `min_df` | 2 | Removes extremely rare terms |
| `max_df` | 0.95 | Removes very common terms |
| `sublinear_tf` | True | Applies logarithmic term-frequency scaling |
| `stop_words` | English | Removes common English stop words |

### Unigrams

A unigram contains one word:

```text
football
government
company
computer
```

### Bigrams

A bigram contains two consecutive words:

```text
stock market
prime minister
football match
software company
```

Using both unigrams and bigrams helps the classifier capture individual keywords as well as short phrases.

---

# 🤖 Machine Learning Models

Three classification algorithms were trained and evaluated.

## 1. Multinomial Naive Bayes

```python
from sklearn.naive_bayes import MultinomialNB

nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)
```

**Test Accuracy: 96.48%**

---

## 2. Logistic Regression

```python
from sklearn.linear_model import LogisticRegression

lr_model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

lr_model.fit(X_train_tfidf, y_train)
```

**Test Accuracy: 97.42%**

---

## 3. Linear Support Vector Machine

```python
from sklearn.svm import LinearSVC

svm_model = LinearSVC(random_state=42)
svm_model.fit(X_train_tfidf, y_train)
```

**Test Accuracy: 97.65%**

The Linear SVM was selected as the model used by the deployed application based on its performance on the held-out test set.

---

# 📈 Model Performance

| Model | Test Accuracy |
|---|---:|
| Multinomial Naive Bayes | 96.48% |
| Logistic Regression | 97.42% |
| Linear SVM | **97.65%** |

The final Linear SVM achieved:

- **Accuracy:** 97.65%
- **Weighted F1-score:** ~0.98
- **Correct predictions:** 416 / 426
- **Incorrect predictions:** 10 / 426

---

# 📊 Linear SVM Classification Report

| Category | Precision | Recall | F1-Score |
|---|---:|---:|---:|
| Business | 0.97 | 0.97 | 0.97 |
| Entertainment | 0.97 | 0.99 | 0.98 |
| Politics | 0.97 | 0.96 | 0.97 |
| Sport | 0.99 | 0.99 | 0.99 |
| Tech | 0.97 | 0.97 | 0.97 |

Overall:

```text
Accuracy:       97.65%
Macro F1:       ~0.98
Weighted F1:    ~0.98
```

---

# 🔲 Confusion Matrix

The Linear SVM produced the following confusion matrix on the 426 test articles.

Rows represent the **actual category**, while columns represent the **predicted category**.

```text
                 Predicted
             Bus  Ent  Pol  Spo  Tech

Business      98    0    1    1    1
Entertainment  0   73    1    0    0
Politics       2    0   78    0    1
Sport          1    0    0  100    0
Tech           0    2    0    0   67
```

The model correctly classified **416 out of 426** test articles.

---

# 🔎 Error Analysis

Although the classifier achieved high accuracy, a small number of articles were misclassified.

Examples of challenging cases included:

- Business articles containing strong technology-related vocabulary
- Technology articles involving entertainment events
- Politics articles discussing business or economic topics
- Sport articles containing business or ownership-related discussions
- Entertainment articles involving political or historical topics

These errors demonstrate that news categories can sometimes overlap semantically.

For example, an article about a technology company could contain many business-related terms, making it difficult to distinguish between **Business** and **Tech** using only textual features.

---

# 💾 Model Serialization

The trained TF-IDF vectorizer and Linear SVM model were saved using `joblib`.

```python
import joblib

joblib.dump(tfidf, "tfidf_vectorizer.pkl")
joblib.dump(svm_model, "bbc_news_classifier.pkl")
```

The application loads these saved objects rather than retraining the model every time it starts.

```python
loaded_tfidf = joblib.load("tfidf_vectorizer.pkl")
loaded_model = joblib.load("bbc_news_classifier.pkl")
```

This makes the deployed application faster and keeps the training process separate from inference.

---

# 🌐 Streamlit Web Application

The model is integrated into a Streamlit application.

The application allows users to:

- Enter or paste a news article
- Select from example articles
- Classify the article
- View the predicted category
- View model and dataset information
- Clear the input and try another article

### Application Workflow

```text
User enters article
        ↓
Text validation
        ↓
Text cleaning
        ↓
TF-IDF transformation
        ↓
Linear SVM prediction
        ↓
Predicted news category
```

---

# 🎨 Application Features

### 📝 Article Input

Users can paste a complete news article into the text area.

### ⚡ Instant Classification

The trained model predicts the category without requiring model retraining.

### 🧪 Example Articles

The application includes example inputs for each supported category so users can quickly test the classifier.

### 📊 Model Information

The sidebar provides information about:

- Algorithm
- TF-IDF representation
- Number of features
- Training samples
- Testing samples
- Number of categories
- Model comparison

### 🛡️ Input Validation

The application handles:

- Empty input
- Very short input
- Missing model files
- Invalid prediction scenarios

---

# 🛠️ Tech Stack

## Programming Language

- Python

## Machine Learning

- Scikit-learn
- Linear SVM
- Logistic Regression
- Multinomial Naive Bayes

## Natural Language Processing

- TF-IDF
- Text preprocessing
- Unigrams
- Bigrams
- Stop-word removal

## Data Processing

- Pandas
- NumPy

## Model Persistence

- Joblib

## Web Application

- Streamlit

## Deployment

- Streamlit Community Cloud
- GitHub

---

# 📁 Project Structure

```text
bbc-news-classifier/
│
├── app.py
├── bbc_news_classifier.pkl
├── tfidf_vectorizer.pkl
├── requirements.txt
├── .gitignore
└── README.md
```

### File Description

| File | Description |
|---|---|
| `app.py` | Streamlit web application |
| `bbc_news_classifier.pkl` | Trained Linear SVM model |
| `tfidf_vectorizer.pkl` | Fitted TF-IDF vectorizer |
| `requirements.txt` | Required Python packages |
| `.gitignore` | Files ignored by Git |
| `README.md` | Project documentation |

---

# ⚙️ Installation

## 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/bbc-news-classifier.git
cd bbc-news-classifier
```

Replace `YOUR_USERNAME` with your GitHub username.

## 2. Create a Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the Application Locally

Run:

```bash
python -m streamlit run app.py
```

The application will start locally and Streamlit will provide a local URL, usually:

```text
http://localhost:8501
```

Open the URL in your browser.

---

# 📦 Requirements

The application requires:

```text
streamlit
scikit-learn
joblib
```

For reproducible deployments, package versions can also be pinned in `requirements.txt`.

---

# ☁️ Deployment

The application is deployed using **Streamlit Community Cloud**.

Streamlit Community Cloud can deploy an application directly from a GitHub repository. Once the repository is connected, changes pushed to the repository can automatically be reflected in the deployed application.

Official documentation:

https://docs.streamlit.io/deploy/streamlit-community-cloud

### Deployment Configuration

```text
Repository: bbc-news-classifier
Branch: main
Main file: app.py
```

### Live Application

https://bbcnewsclassifier.streamlit.app/

---

# 🔄 Updating the Deployed Application

After making changes locally:

```bash
git add .
git commit -m "Update Streamlit application"
git push origin main
```

Streamlit Community Cloud monitors the connected GitHub repository and can automatically update the deployed application when changes are pushed.

---

# 🧪 Example Predictions

The application can classify articles such as:

### Sport

```text
The team secured a dramatic victory after scoring in the final minutes...
```

Expected category:

```text
Sport
```

### Politics

```text
The government announced a new policy following discussions in parliament...
```

Expected category:

```text
Politics
```

### Business

```text
The company reported higher revenue and increased profits during the quarter...
```

Expected category:

```text
Business
```

### Entertainment

```text
The singer announced a new album and revealed plans for a world tour...
```

Expected category:

```text
Entertainment
```

### Tech

```text
The company unveiled a new smartphone featuring a faster processor and improved camera...
```

Expected category:

```text
Tech
```

---

# 🧠 Key Machine Learning Concepts Demonstrated

This project demonstrates practical understanding of:

- Natural Language Processing
- Text preprocessing
- Exploratory Data Analysis
- Duplicate detection
- Train-test splitting
- Stratified sampling
- TF-IDF vectorization
- N-gram feature extraction
- Stop-word removal
- Supervised classification
- Multinomial Naive Bayes
- Logistic Regression
- Linear SVM
- Model evaluation
- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrices
- Error analysis
- Model serialization
- Streamlit application development
- Git and GitHub
- Cloud deployment

---

# 📚 What I Learned

Through this project, I worked through the complete lifecycle of an NLP machine learning application:

```text
Raw Dataset
     ↓
Data Exploration
     ↓
Data Cleaning
     ↓
Duplicate Removal
     ↓
Text Preprocessing
     ↓
Train/Test Split
     ↓
TF-IDF Feature Extraction
     ↓
Model Training
     ↓
Model Comparison
     ↓
Evaluation
     ↓
Error Analysis
     ↓
Model Serialization
     ↓
Streamlit App
     ↓
Cloud Deployment
```

This project helped bridge the gap between building a machine learning model in a notebook and deploying that model as a usable application.

---

# 🚧 Limitations

Although the model performs well on the test dataset, there are several limitations:

1. **Dataset Size**  
   The dataset contains 2,127 unique articles after duplicate removal, which is relatively small compared with large-scale modern NLP datasets.

2. **Domain Dependence**  
   The model was trained on BBC news articles and may perform differently on articles from other publishers.

3. **Category Overlap**  
   Some articles naturally contain information from multiple categories, such as business articles discussing technology companies.

4. **Traditional NLP Approach**  
   The model uses TF-IDF rather than contextual transformer-based embeddings, so it does not fully capture deep semantic relationships between words.

5. **Language Limitation**  
   The preprocessing pipeline is designed for English-language news articles.

6. **Out-of-Domain Text**  
   Performance may decrease when the input is very different from the training data, such as social media posts, informal text, advertisements or very short headlines.

---

# 🔮 Future Improvements

Potential improvements include:

- [ ] Add more news datasets
- [ ] Increase the size and diversity of the training data
- [ ] Experiment with word and character n-grams
- [ ] Perform hyperparameter tuning
- [ ] Add cross-validation
- [ ] Compare additional classifiers
- [ ] Experiment with Word2Vec or GloVe embeddings
- [ ] Experiment with transformer models such as BERT
- [ ] Add calibrated prediction confidence
- [ ] Add batch CSV classification
- [ ] Add prediction history
- [ ] Add article statistics and visualization
- [ ] Add automated model retraining
- [ ] Add unit tests
- [ ] Add CI/CD using GitHub Actions

---

# 📌 Why Linear SVM?

Linear SVM is particularly suitable for high-dimensional sparse text representations such as TF-IDF.

In this project, it produced strong results while remaining relatively lightweight compared with large transformer-based models.

The final deployed pipeline is therefore:

```text
Text
 ↓
Cleaning
 ↓
TF-IDF
 ↓
Linear SVM
 ↓
Category
```

---

# 🔐 Reproducibility

The project stores the fitted TF-IDF vectorizer and trained classifier as `.pkl` files.

This means the Streamlit application can use the exact trained preprocessing and model pipeline without requiring the original dataset or retraining during deployment.

For reproducibility, keep the following files together:

```text
app.py
bbc_news_classifier.pkl
tfidf_vectorizer.pkl
requirements.txt
```

---

# 📈 Project Highlights

- ✅ 2,127 unique news articles
- ✅ 5 news categories
- ✅ TF-IDF with unigrams and bigrams
- ✅ 3 machine learning models evaluated
- ✅ 97.65% test accuracy with Linear SVM
- ✅ ~0.98 weighted F1-score
- ✅ Confusion matrix and error analysis
- ✅ Serialized production model
- ✅ Interactive Streamlit interface
- ✅ Deployed online
- ✅ GitHub-ready project structure

---

# 👨‍💻 Author

**Surya**

This project was developed as part of a machine learning and NLP portfolio to demonstrate the complete process of building, evaluating and deploying a text classification application.

---

## ⭐ If You Found This Project Useful

Feel free to explore the repository, try the live application, and experiment with your own news articles.

### 🔗 Live Demo

**https://bbcnewsclassifier.streamlit.app/**

---

## 📄 License

This project is intended for educational and portfolio purposes.

If you reuse or modify the project, please provide appropriate attribution.
