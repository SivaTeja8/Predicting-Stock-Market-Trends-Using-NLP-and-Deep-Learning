"""
align_data.py - Merge ML & NLP Features and Create Target Variable
Author: [Member 1 & 4 Name]
Course: [Course Name]
Description: Merges ML and NLP features by date, creates target variable, and saves final dataset.
"""

import pandas as pd
import numpy as np
import os
from datetime import timedelta

def align_ml_nlp_features():
    """Main alignment function"""
    print("="*70)
    print("🏗️  DATA ALIGNMENT PIPELINE (Member 1 & 4)")
    print("="*70)
    
    os.makedirs('data/processed', exist_ok=True)
    
    # 1. Load ML Features
    print("\n📥 Loading ML features...")
    df_ml = pd.read_csv('data/processed/ml_features.csv')
    print(f"   ML samples: {len(df_ml):,}")
    
    # 2. Load NLP Features
    print("📥 Loading NLP features...")
    df_nlp = pd.read_csv('data/processed/nlp_features.csv')
    print(f"   NLP samples: {len(df_nlp):,}")
    
    # 3. Merge: News from Day T → Price Movement on Day T+1
    print("🔗 Merging ML and NLP features (avoiding look-ahead bias)...")
    
    # Shift news forward by 1 day
    df_nlp['Date_Next'] = df_nlp['Date'] + timedelta(days=1)
    
    # Merge prices with news: Price Date = News Date + 1
    df_aligned = pd.merge(
        df_ml,
        df_nlp[['Date', 'Date_Next', 'News_Volume', 'News_Sentiment', 'NER_Score', 'TFIDF_Score']],
        left_on='Date',
        right_on='Date_Next',
        how='left'
    )
    
    df_aligned = df_aligned.rename(columns={'Date_Next': 'News_Date', 'Date': 'Price_Date'})
    
    # 4. Create Target Variable
    print("📈 Creating target variable (price movement)...")
    df_aligned = df_aligned.sort_values(['Symbol', 'Price_Date']).reset_index(drop=True)
    df_aligned['Next_Close'] = df_aligned.groupby('Symbol')['Close'].shift(-1)
    df_aligned['Return'] = (df_aligned['Next_Close'] - df_aligned['Close']) / df_aligned['Close']
    df_aligned['Label'] = (df_aligned['Return'] > 0).astype(int)
    
    # 5. Handle Missing Values
    df_aligned = df_aligned.dropna(subset=['Label', 'News_Volume', 'Close'])
    df_aligned = df_aligned.drop(['Next_Close', 'Return', 'News_Date'], axis=1, errors='ignore')
    
    # 6. Feature Selection
    feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'News_Volume', 'News_Sentiment', 'NER_Score', 'TFIDF_Score']
    available_features = [col for col in feature_cols if col in df_aligned.columns]
    
    keep_cols = ['Symbol', 'Price_Date'] + available_features + ['Label']
    df_final = df_aligned[keep_cols].copy()
    
    # 7. Final Verification
    print(f"\n✅ Aligned Samples: {len(df_final):,} (Required: ≥50,000)")
    print(f"📅 Date Range: {df_final['Price_Date'].min()} to {df_final['Price_Date'].max()}")
    print(f"📊 Stocks: {df_final['Symbol'].nunique()}")
    print(f"📋 Features: {available_features}")
    print(f"🎯 Label Distribution:\n{df_final['Label'].value_counts()}")
    
    if len(df_final) >= 50000:
        print("✅ PASS: Meets ≥50,000 samples requirement (Guideline 2.3)")
    else:
        print("❌ FAIL: Below 50,000 samples.")
    
    # 8. Save Final Dataset
    output_path = 'data/processed/final_dataset.csv'
    df_final.to_csv(output_path, index=False)
    print(f"💾 Saved: {output_path}")
    
    return df_final

if __name__ == "__main__":
    df_final = align_ml_nlp_features()
    if df_final is not None:
        print(f"\n📊 Final Dataset Shape: {df_final.shape}")