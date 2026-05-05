import os
import warnings

# Hide TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# Hide Scikit-Learn feature name warnings
warnings.filterwarnings("ignore", category=UserWarning)

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from datetime import timedelta

# --- Page Setup ---
st.set_page_config(page_title="AI Equity Terminal", layout="wide", initial_sidebar_state="expanded")

# --- High-Contrast & Theme-Aware Styling ---
st.markdown("""
    <style>
    [data-testid="stMetricValue"] { 
        font-size: 28px; 
        font-weight: 800;
        color: #1d4ed8; 
    }
    [data-testid="stMetricLabel"] { 
        font-size: 15px; 
        font-weight: 700;
        opacity: 0.9;
    }
    .stMetric {
        background-color: rgba(128, 128, 128, 0.05);
        border: 1px solid rgba(128, 128, 128, 0.2);
        padding: 15px;
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Global Official Metrics (Synced to your table) ---
GLOBAL_METRICS = {
    "Baseline LSTM": {"Accuracy": 0.5265, "Precision": 0.5265, "Recall": 1.0000, "F1-Score": 0.6898, "AUC-ROC": 0.4777},
    "Hybrid NLP-Fusion": {"Accuracy": 0.5686, "Precision": 0.5783, "Recall": 0.6674, "F1-Score": 0.6197, "AUC-ROC": 0.5935},
    "NLP Tree Ensemble": {"Accuracy": 0.6284, "Precision": 0.6280, "Recall": 0.7216, "F1-Score": 0.6716, "AUC-ROC": 0.6756},
    "Multi-Model Ensemble": {"Accuracy": 0.6252, "Precision": 0.6223, "Recall": 0.7331, "F1-Score": 0.6732, "AUC-ROC": 0.6716}
}

# --- Data Loading ---
MODEL_DIR = 'saved_models'

@st.cache_data
def load_all():
    df = pd.read_csv(os.path.join(MODEL_DIR, 'df_final.csv'))
    date_col = 'Date' if 'Date' in df.columns else 'Price_Date'
    df[date_col] = pd.to_datetime(df[date_col])
    
    ps = joblib.load(os.path.join(MODEL_DIR, 'price_scaler.pkl'))
    ns = joblib.load(os.path.join(MODEL_DIR, 'nlp_scaler.pkl'))
    m_lstm = joblib.load(os.path.join(MODEL_DIR, 'model_a_lstm.pkl'))
    m_hybrid = joblib.load(os.path.join(MODEL_DIR, 'model_b_hybrid.pkl'))
    m_tree = joblib.load(os.path.join(MODEL_DIR, 'model_c_ensemble.pkl'))
    
    return df, ps, ns, m_lstm, m_hybrid, m_tree, date_col

df_full, ps, ns, m_lstm, m_hybrid, m_tree, DATE_COL = load_all()

# --- SIDEBAR: NAVIGATION ---
st.sidebar.title("📊 Terminal Navigation")
page = st.sidebar.radio("Select View", ["Market Dashboard", "Historical Data Audit", "Model Comparison"])

st.sidebar.divider()
st.sidebar.title("🔍 Strategy Controls")

ticker_col = next((c for c in ['Ticker', 'Symbol', 'stock'] if c in df_full.columns), None)
if ticker_col:
    selected_stock = st.sidebar.selectbox("Select Asset", sorted(df_full[ticker_col].unique()))
    df = df_full[df_full[ticker_col] == selected_stock].copy().sort_values(DATE_COL)
else:
    df = df_full.copy().sort_values(DATE_COL)

selected_model_name = st.sidebar.selectbox("Analysis Engine", list(GLOBAL_METRICS.keys()))



# --- Inference Engine ---
seq_len = 10
X_p = ps.transform(df[['Open', 'High', 'Low', 'Close', 'Volume']])
X_n = ns.transform(df[['News_Volume', 'News_Sentiment', 'NER_Score', 'TFIDF_Score']])

p_seq, n_feat, c_feat = [], [], []
for i in range(seq_len, len(X_p)):
    p_seq.append(X_p[i-seq_len:i])
    n_feat.append(X_n[i])
    c_feat.append(np.hstack([X_p[i-seq_len:i].flatten(), X_n[i]]))

p_seq, n_feat, c_feat = np.array(p_seq), np.array(n_feat), np.array(c_feat)
df_res = df.iloc[seq_len:].copy()

# Probabilities
prob_lstm = m_lstm.predict(p_seq).flatten()
prob_hybrid = m_hybrid.predict([p_seq, n_feat]).flatten()
prob_tree = m_tree.predict_proba(c_feat)[:, 1]
prob_ens = (0.2 * prob_lstm) + (0.3 * prob_hybrid) + (0.5 * prob_tree)

all_probs = {
    "Baseline LSTM": prob_lstm, "Hybrid NLP-Fusion": prob_hybrid,
    "NLP Tree Ensemble": prob_tree, "Multi-Model Ensemble": prob_ens
}

y_prob = all_probs[selected_model_name]
y_pred = (y_prob > 0.5).astype(int)
df_res['Prediction'] = y_pred

# --- MAIN CONTENT ---

if page == "Market Dashboard":
    st.title(f"🚀 {selected_model_name} Accuracy Analysis")
    
    # Header Metrics focused on Model Performance
    m_stats = GLOBAL_METRICS[selected_model_name]
    col1, col2, col3, col4 = st.columns(4)
    
    col1.metric("Overall Accuracy", f"{m_stats['Accuracy']:.1%}")
    col2.metric("Precision", f"{m_stats['Precision']:.1%}")
    col3.metric("F1-Score", f"{m_stats['F1-Score']:.2f}")
    col4.metric("Model AUC-ROC", f"{m_stats['AUC-ROC']:.2f}")

    st.divider()

    # --- CHART: Price vs Predictions (150-Day Clarity View) ---
    st.subheader(f"📊 {selected_stock} Test Set: Accuracy vs. Predictions")
    
    n_plot = 150 
    df_plot = df_res.tail(n_plot).copy()
    
    fig = go.Figure()

    # Black line for Actual Price Trend
    fig.add_trace(go.Scatter(
        x=df_plot[DATE_COL], y=df_plot['Close'],
        mode='lines', name='Actual Close Price',
        line=dict(color='black', width=1.8),
        hovertemplate="Price: $%{y:.2f}<extra></extra>"
    ))

    # Green markers for Predicted UP (#27ae60)
    df_up = df_plot[df_plot['Prediction'] == 1]
    fig.add_trace(go.Scatter(
        x=df_up[DATE_COL], y=df_up['Close'],
        mode='markers', name='Predicted UP',
        marker=dict(color='#27ae60', size=10, symbol='triangle-up', line=dict(width=1, color='white'))
    ))

    # Red markers for Predicted DOWN (#c0392b)
    df_down = df_plot[df_plot['Prediction'] == 0]
    fig.add_trace(go.Scatter(
        x=df_down[DATE_COL], y=df_down['Close'],
        mode='markers', name='Predicted DOWN',
        marker=dict(color='#c0392b', size=10, symbol='triangle-down', line=dict(width=1, color='white'))
    ))

    fig.update_layout(
        height=500,
        hovermode="x unified",
        plot_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.1)', title="Date"),
        yaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.1)', title="Price ($)")
    )
    
    st.plotly_chart(fig, use_container_width=True)

    # Detailed Accuracy breakdown table
    st.subheader("🏁 Performance Audit (Last 5 Trading Days)")
    last_5 = df_res.tail(5)
    audit_df = pd.DataFrame({
        "Date": last_5[DATE_COL].dt.date,
        "Actual Direction": ["📈 UP" if l == 1 else "📉 DOWN" for l in last_5['Label']],
        "AI Prediction": ["📈 UP" if p == 1 else "📉 DOWN" for p in y_pred[-5:]],
        "Result": np.where(last_5['Label'] == y_pred[-5:], "✅ HIT", "❌ MISS")
    })
    st.table(audit_df)

elif page == "Model Comparison":
    st.title("🏆 Strategy Benchmarking")
    comparison_df = pd.DataFrame(GLOBAL_METRICS).T
    st.dataframe(comparison_df.style.highlight_max(axis=0, color='#1d4ed8'), use_container_width=True)

    st.subheader("📊 Performance Visualization")
    plot_df = comparison_df.reset_index().rename(columns={'index': 'Model'})
    fig_comp = px.bar(
        plot_df, x='Model', y=['Accuracy', 'AUC-ROC', 'F1-Score'],
        barmode='group',
        color_discrete_sequence=['#1d4ed8', '#10b981', '#f59e0b'],
        height=500
    )
    st.plotly_chart(fig_comp, use_container_width=True)

elif page == "Historical Data Audit":
    st.title("🔢 Historical Strategy Audit")
    
    # --- Date Filter (Now in Main Page) ---
    st.markdown("### 📅 Filter Audit Period")
    
    # Create columns to make the date input more compact
    col_a, col_b = st.columns([1, 2])
    with col_a:
        min_d = df_res[DATE_COL].min().to_pydatetime()
        max_d = df_res[DATE_COL].max().to_pydatetime()
        
        date_range = st.date_input(
            "Select Range",
            value=(max_d - timedelta(days=60), max_d),
            min_value=min_d,
            max_value=max_d,
            label_visibility="collapsed" # Hides the label for a cleaner look
        )

    # --- Filtering Logic ---
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
        mask = (df_res[DATE_COL].dt.date >= start_date) & (df_res[DATE_COL].dt.date <= end_date)
        filtered_df = df_res.loc[mask].copy()
    else:
        filtered_df = df_res.copy()

    # --- Table Preparation ---
    audit_table = pd.DataFrame({
        'Date': filtered_df[DATE_COL].dt.date,
        'Close Price': filtered_df['Close'].map("${:,.2f}".format),
        'Actual Move': filtered_df['Label'].map({1: '📈 UP', 0: '📉 DOWN'}),
        'AI Prediction': filtered_df['Prediction'].map({1: '📈 UP', 0: '📉 DOWN'}),
        'Outcome': np.where(filtered_df['Label'] == filtered_df['Prediction'], '✅ HIT', '❌ MISS')
    })

    # --- Display ---
    st.write(f"Showing **{len(filtered_df)}** results for **{selected_stock}**")
    
    st.dataframe(
        audit_table.sort_values('Date', ascending=False), 
        use_container_width=True, 
        hide_index=True
    )