# Sentiment-Aware Stock Forecasting with Multimodal Fusion

## 👥 Team Information
| Name | Student ID | Role | Email |
|------|-----------|------|-------|
| [Member 1] | [ID] | Data Engineer | [email] |
| [Member 2] | [ID] | ML Specialist | [email] |
| [Member 3] | [ID] | NLP Specialist | [email] |
| [Member 4] | [ID] | Deployment Lead | [email] |

## 📖 Introduction
This project predicts stock market movements by fusing numerical price data (LSTM) with financial news sentiment (NLP). We compare a baseline model against a hybrid fusion architecture to quantify the value of NLP in finance.

**Project Composition:** 60% Machine Learning (LSTM, Fusion, SHAP) + 40% NLP (VADER, NER, TF-IDF, FinBERT)

## 🚀 Quick Start
```bash
# Install dependencies
pip install -r requirements.txt

# Download NLP resources
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('vader_lexicon')"

# Pull data programmatically
python data/scripts/fetch_prices.py
python data/scripts/fetch_news.py

# Run preprocessing with NLP pipeline
python src/preprocessing/align_data.py

# Train models
python src/models/baseline_lstm.py
python src/models/hybrid_fusion.py

# Compare models
python src/evaluation/compare_models.py

# Generate SHAP explainability
python src/evaluation/shap_explainability.py

# Launch demo
streamlit run demo/app.py