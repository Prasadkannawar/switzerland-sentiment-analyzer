import streamlit as st
import re
import nltk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from textblob import TextBlob
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from wordcloud import WordCloud
import wikipedia
from sklearn.metrics import classification_report

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

# Step 1 & 2
wiki_text = wikipedia.page("Switzerland").content

# Step 3 - Clean text
def clean_text(text):
    text = re.sub(r'\s+', ' ', text)  # remove extra whitespace
    text = re.sub(r'\[[0-9]*\]', '', text)  # remove reference numbers
    text = re.sub(r'[^A-Za-z. ]+', '', text)  # keep only letters and periods
    return text

cleaned_text = clean_text(wiki_text)
# Step 4
sentences = sent_tokenize(cleaned_text)

# Step 5
sentiments = [TextBlob(sentence).sentiment.polarity for sentence in sentences]
df = pd.DataFrame({'Sentence': sentences, 'Sentiment': sentiments})
df['Label'] = df['Sentiment'].apply(lambda x: 'positive' if x > 0 else ('negative' if x < 0 else 'neutral'))
words = word_tokenize(cleaned_text.lower())
stop_words = set(stopwords.words('english'))
filtered_words = [word for word in words if word not in stop_words and word.isalpha()]
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(filtered_words))
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()
from collections import Counter

word_freq = Counter(filtered_words)
freq_df = pd.DataFrame(word_freq.items(), columns=['Word', 'Frequency']).sort_values(by='Frequency', ascending=False)
print(freq_df.head(10))
# Using only sentences and ignoring neutral
filtered_df = df[df['Label'] != 'neutral']
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(filtered_df['Sentence'])
y = filtered_df['Label'].map({'positive': 1, 'negative': 0})
smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(X, y)
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)

models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "Naive Bayes": MultinomialNB(),
    "K-Nearest Neighbors": KNeighborsClassifier()
}

for name, model in models.items():
    print(f"\nModel: {name}")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))


@st.cache_resource
def load_and_clean_wiki():
    text = wikipedia.page("Switzerland").content
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\[[0-9]*\]', '', text)
    text = re.sub(r'[^A-Za-z. ]+', '', text)
    sentences = sent_tokenize(text)
    sentiments = [TextBlob(s).sentiment.polarity for s in sentences]
    df = pd.DataFrame({'Sentence': sentences, 'Sentiment': sentiments})
    df['Label'] = df['Sentiment'].apply(lambda x: 'positive' if x > 0 else ('negative' if x < 0 else 'neutral'))
    df = df[df['Label'] != 'neutral']
    return df


@st.cache_resource
def train_all_models():
    df = load_and_clean_wiki()
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['Sentence'])
    y = df['Label'].map({'positive': 1, 'negative': 0})

    smote = SMOTE()
    X_resampled, y_resampled = smote.fit_resample(X, y)
    X_train, _, y_train, _ = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

    models = {
        "Logistic Regression": LogisticRegression(),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "Gradient Boosting": GradientBoostingClassifier(),
        "Naive Bayes": MultinomialNB(),
        "K-Nearest Neighbors": KNeighborsClassifier()
    }

    trained = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained[name] = model

    return trained, vectorizer


def preprocess_input(text):
    text = re.sub(r'[^A-Za-z ]+', '', text)
    return text.strip()


def generate_wordcloud(text):
    stop_words = set(stopwords.words('english'))
    words = nltk.word_tokenize(text.lower())
    filtered = [w for w in words if w not in stop_words and w.isalpha()]
    wc = WordCloud(width=600, height=400, background_color='white').generate(' '.join(filtered))
    return wc

st.set_page_config(page_title="Switzerland Sentiment Analyzer", layout="wide")
st.title("🇨🇭 Switzerland Sentiment Analyzer with ML & Visualizations")
st.write("Enter a sentence and select a model to see sentiment prediction and related visualizations.")

# Sentence input
sentence_input = st.text_area("📝 Enter a sentence to analyze:")

# Model choice
model_choice = st.selectbox("🤖 Choose a model:", [
    "Logistic Regression", "Decision Tree", "Random Forest", "Naive Bayes", "K-Nearest Neighbors","Gradient Boosting"
])

# Predict on input
if sentence_input and model_choice:
    models, vectorizer = train_all_models()
    model = models[model_choice]

    cleaned_input = preprocess_input(sentence_input)
    input_vector = vectorizer.transform([cleaned_input])
    prediction = model.predict(input_vector)[0]
    proba = model.predict_proba(input_vector)[0]

    sentiment = "Positive 😊" if prediction == 1 else "Negative 😞"
    st.success(f"🔍 **Predicted Sentiment**: `{sentiment}` using `{model_choice}`")

    # Layout with columns
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Prediction Probabilities")
        fig, ax = plt.subplots()
        ax.bar(['Negative', 'Positive'], proba, color=['red', 'green'])
        ax.set_ylabel("Probability")
        st.pyplot(fig)

    with col2:
        st.subheader("☁️ WordCloud of Input")
        wc = generate_wordcloud(cleaned_input)
        plt.figure(figsize=(6, 4))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        st.pyplot(plt)


