"""
preprocess_nlp.py - NLP Feature Engineering (News & Sentiment)
Author: [Member 3 Name]
Course: [Course Name]
Description: Processes raw news data, applies NLP algorithms (VADER, NER, TF-IDF), and saves NLP features.
"""

import pandas as pd
import numpy as np
import os
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
warnings.filterwarnings('ignore')

# Download NLP resources
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("⚠️ Spacy model not found. Run: python -m spacy download en_core_web_sm")
    nlp = None

sia = SentimentIntensityAnalyzer()

def get_sentiment_score(text):
    """NLP Algorithm 1: VADER Sentiment Analysis"""
    if pd.isna(text):
        return 0
    scores = sia.polarity_scores(str(text))
    return scores['compound']

def get_entities(text):
    """NLP Algorithm 2: Named Entity Recognition (NER)"""
    if nlp is None:
        return 0
    doc = nlp(str(text))
    return len([ent for ent in doc.ents if ent.label_ in ['ORG', 'MONEY', 'PERCENT', 'DATE']])

def get_tfidf_score(texts):
    """NLP Algorithm 3: TF-IDF Keyword Extraction"""
    if len(texts) == 0:
        return np.array([0])
    vectorizer = TfidfVectorizer(stop_words='english', max_features=100)
    tfidf = vectorizer.fit_transform(texts)
    return tfidf.mean(axis=1).A1

def preprocess_nlp_features():
    """Main NLP preprocessing pipeline"""
    print("="*70)
    print("🏗️  NLP PREPROCESSING PIPELINE (Member 3)")
    print("="*70)
    
    os.makedirs('data/processed', exist_ok=True)
    
    # 1. Load Raw News
    print("\n📥 Loading raw news data...")
    df_news = pd.read_csv('data/processed/news_aligned.csv')
    print(f"   Raw news samples: {len(df_news):,}")
    
    # 2. Preprocess Dates
    df_news['Date'] = pd.to_datetime(df_news['Date']).dt.date
    
    # 3. NLP Pipeline (40% of Project)
    print("\n📰 Running NLP Pipeline...")
    
    # NLP Algorithm 1: VADER Sentiment
    print("   1. Applying VADER Sentiment Analysis...")
    df_news['Sentiment_Score'] = df_news['Headline'].apply(get_sentiment_score)
    
    # NLP Algorithm 2: NER
    print("   2. Applying Named Entity Recognition (NER)...")
    df_news['Entity_Count'] = df_news['Headline'].apply(get_entities)
    
    # NLP Algorithm 3: TF-IDF
    print("   3. Applying TF-IDF Keyword Extraction...")
    df_news['TFIDF_Score'] = get_tfidf_score(df_news['Headline'].astype(str).tolist())
    
    # 4. Aggregate News by Date
    print("📰 Aggregating news by date...")
    df_nlp = df_news.groupby('Date').agg(
        News_Volume=('Headline', 'size'),
        News_Sentiment=('Sentiment_Score', 'mean'),
        NER_Score=('Entity_Count', 'mean'),
        TFIDF_Score=('TFIDF_Score', 'mean')
    ).reset_index()
    
    print(f"   Daily news records: {len(df_nlp)}")
    print(f"   NLP Features: News_Volume, News_Sentiment, NER_Score, TFIDF_Score")
    
    # 5. Save NLP Features
    output_path = 'data/processed/nlp_features.csv'
    df_nlp.to_csv(output_path, index=False)
    print(f"💾 NLP Features saved to: {output_path}")
    
    return df_nlp

if __name__ == "__main__":
    preprocess_nlp_features()