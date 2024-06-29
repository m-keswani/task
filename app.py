import json
import pandas as pd
import streamlit as st
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Initialize Vader sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Load the dataset
def load_data(file):
    data = json.load(file)
    conversations = []
    for convo in data:
        convo_text = " ".join([turn['value'] for turn in convo if turn['from'] == 'human'])
        conversations.append(convo_text.strip())
    return conversations

# Function for sentiment analysis using Vader
def analyze_sentiment_vader(conversations):
    sentiments = []
    for conv in conversations:
        # Get compound sentiment score
        score = analyzer.polarity_scores(conv)["compound"]
        
        # Classify sentiment based on score
        if score >= 0.05:
            sentiments.append("positive")
        elif score <= -0.05:
            sentiments.append("negative")
        else:
            sentiments.append("neutral")
    
    return sentiments

# Function for clustering conversations based on TF-IDF vectors
def cluster_conversations(conversations, n_clusters=5):
    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(conversations)
    
    # K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(tfidf_matrix)
    
    return clusters, kmeans, vectorizer

# Streamlit app
def main():
    st.title("Conversation Analysis")

    uploaded_file = st.file_uploader("Upload a JSON file", type="json")
    
    if uploaded_file:
        conversations = load_data(uploaded_file)

        # Perform sentiment analysis using Vader
        sentiments = analyze_sentiment_vader(conversations)

        # Cluster conversations
        clusters, kmeans, vectorizer = cluster_conversations(conversations)

        # Prepare data for display
        session_data = pd.DataFrame({
            "Conversation No": range(1, len(conversations) + 1),
            "Conversation": conversations,
            "Sentiment": sentiments,
            "Topic": clusters
        })

        # Screen 1: Counts
        st.header("Counts")

        # Count conversations by topic
        topic_counts = session_data['Topic'].value_counts().sort_index()
        st.subheader("Topic Counts")
        st.table(topic_counts)

        # Count conversations by sentiment
        sentiment_counts = session_data['Sentiment'].value_counts()
        st.subheader("Sentiment Counts")
        st.table(sentiment_counts)

        # Screen 2: Sessions
        st.header("Sessions")
        st.dataframe(session_data)

if __name__ == "__main__":
    main()
