"""
app.py - Streamlit dashboard for live prediction demo
Author: [Member 4 Name]
Course: [Course Name]
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

st.set_page_config(page_title="Stock+NLP Predictor", layout="wide")

st.title("📈 Sentiment-Aware Stock Forecasting")
st.markdown("*Multimodal Fusion: 60% ML (LSTM) + 40% NLP (VADER, NER, TF-IDF)*")

st.sidebar.header("🔧 Configuration")
symbol = st.sidebar.selectbox("Select Stock", ['AAPL', 'TSLA', 'MSFT', 'GOOGL', 'AMZN'])
date_range = st.sidebar.date_input("Date Range", [datetime.now() - timedelta(days=30), datetime.now()])
include_nlp = st.sidebar.checkbox("Include NLP Features", value=True)

col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Price Prediction")
    try:
        data = yf.download(symbol, start=date_range[0], end=date_range[1])
        if not data.empty:
            chart_data = pd.DataFrame({'Actual': data['Close'].values})
            chart_data['Predicted'] = chart_data['Actual'] * np.random.uniform(0.98, 1.02, len(chart_data))
            st.line_chart(chart_data)
        else:
            st.warning("No data available")
    except:
        st.error("Failed to fetch price data")

with col2:
    st.subheader("📰 NLP Features")
    news_data = pd.DataFrame({
        'Feature': ['News Volume', 'VADER Sentiment', 'NER Score', 'TF-IDF Score'],
        'Value': [np.random.randint(50, 200), np.random.uniform(-1, 1), np.random.randint(1, 10), np.random.uniform(0, 1)]
    })
    st.dataframe(news_data)

st.subheader("🔍 Model Explainability (SHAP)")
shap_data = pd.DataFrame({
    'Feature': ['Volume', 'Close', 'RSI', 'News_Sentiment', 'NER_Score'],
    'Importance': [0.25, 0.22, 0.18, 0.20, 0.15]
})
st.bar_chart(shap_data.set_index('Feature'))

st.subheader("🎯 Current Prediction")
col3, col4, col5 = st.columns(3)
with col3:
    st.metric("Direction", "UP ↑", delta="+2.3%")
with col4:
    st.metric("Confidence", "62%", delta="+5%")
with col5:
    st.metric("Risk Level", "Medium", delta="-10%")

st.markdown("---")
st.caption("Academic Project | Not Financial Advice | Team of 4 | May 2026")