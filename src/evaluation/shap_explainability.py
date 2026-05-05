"""
shap_explainability.py - Model explainability using SHAP
Author: [Member 3 Name]
Course: [Course Name]
"""

import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def generate_shap_explanations():
    """Generate SHAP explainability visualizations"""
    
    print("🔍 Generating SHAP explanations...")
    
    df = pd.read_csv('data/processed/final_dataset.csv')
    
    feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'News_Volume', 'News_Sentiment', 'NER_Score', 'TFIDF_Score']
    available_features = [col for col in feature_cols if col in df.columns]
    X = df[available_features].values
    y = df['Label'].values
    
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=False)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    
    os.makedirs('src/evaluation/results', exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values[1], X_test, feature_names=available_features, show=False, plot_type="bar")
    plt.title('Feature Importance (SHAP Values)', fontweight='bold')
    plt.savefig('src/evaluation/results/shap_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("💾 SHAP summary saved to src/evaluation/results/shap_summary.png")
    
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values[1], X_test, feature_names=available_features, show=False)
    plt.title('SHAP Value Distribution', fontweight='bold')
    plt.savefig('src/evaluation/results/shap_beeswarm.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("💾 SHAP beeswarm saved to src/evaluation/results/shap_beeswarm.png")
    
    print("\n✅ SHAP explainability complete!")
    
    return shap_values, available_features

if __name__ == "__main__":
    shap_values, features = generate_shap_explanations()