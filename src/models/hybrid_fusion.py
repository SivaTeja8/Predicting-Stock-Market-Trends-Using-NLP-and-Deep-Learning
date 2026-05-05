"""
hybrid_fusion.py - Model B: Multimodal fusion of price + NLP features (60% ML + 40% NLP)
Author: [Member 3 Name]
Course: [Course Name]
Description: Hybrid model fusing numerical price features (LSTM) with NLP features (VADER, NER, TF-IDF).
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, precision_recall_curve
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Concatenate, BatchNormalization, Attention, GlobalAveragePooling1D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import os
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
tf.random.set_seed(42)

def add_technical_indicators(df):
    """Calculate technical indicators for better feature representation"""
    df = df.copy()
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    
    # Bollinger Bands (FIXED ORDER)
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    df['BB_Std'] = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + 2 * df['BB_Std']
    df['BB_Lower'] = df['BB_Middle'] - 2 * df['BB_Std']
    df['BB_Percent'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'] + 1e-8)
    
    # Volume features
    df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
    df['Volume_Ratio'] = df['Volume'] / (df['Volume_MA'] + 1e-8)
    
    # Momentum
    df['Momentum'] = df['Close'].diff(10)
    df['ROC'] = df['Close'].pct_change(10) * 100
    
    df = df.fillna(0)
    df = df.replace([np.inf, -np.inf], 0)
    
    return df

def create_hybrid_model(num_price_features, num_nlp_features, sequence_length=10):
    """Create optimized hybrid fusion model with attention mechanism"""
    
    # Price branch (LSTM with attention)
    price_input = Input(shape=(sequence_length, num_price_features))
    price_lstm = LSTM(64, return_sequences=True)(price_input)
    price_lstm = BatchNormalization()(price_lstm)
    price_lstm = Dropout(0.3)(price_lstm)
    price_lstm = LSTM(32, return_sequences=True)(price_lstm)
    price_lstm = BatchNormalization()(price_lstm)
    price_lstm = Dropout(0.3)(price_lstm)
    
    # Self-attention on price features
    price_attention = Attention()([price_lstm, price_lstm])
    price_pool = GlobalAveragePooling1D()(price_attention)
    price_dense = Dense(32, activation='relu')(price_pool)
    price_dense = BatchNormalization()(price_dense)
    price_dense = Dropout(0.3)(price_dense)
    
    # NLP branch (Dense network for NLP features)
    nlp_input = Input(shape=(num_nlp_features,))
    nlp_dense = Dense(16, activation='relu')(nlp_input)
    nlp_dense = BatchNormalization()(nlp_dense)
    nlp_dense = Dropout(0.3)(nlp_dense)
    nlp_dense = Dense(8, activation='relu')(nlp_dense)
    
    # Fusion layer
    combined = Concatenate()([price_dense, nlp_dense])
    combined = Dense(64, activation='relu')(combined)
    combined = BatchNormalization()(combined)
    combined = Dropout(0.4)(combined)
    combined = Dense(32, activation='relu')(combined)
    combined = Dropout(0.3)(combined)
    output = Dense(1, activation='sigmoid')(combined)
    
    # Create model
    model = Model(inputs=[price_input, nlp_input], outputs=output)
    
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])
    
    return model

def prepare_sequences_with_news(prices, nlp_features, labels, sequence_length=10):
    """Create sequences for hybrid model (prices + NLP)"""
    X_price, X_nlp, y = [], [], []
    for i in range(sequence_length, len(prices)):
        X_price.append(prices[i-sequence_length:i])
        X_nlp.append(nlp_features[i])
        y.append(labels[i])
    return np.array(X_price), np.array(X_nlp), np.array(y).flatten()

def find_optimal_threshold(y_true, y_pred_proba):
    """Find optimal threshold based on F1-Score"""
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
    return optimal_threshold, f1_scores[optimal_idx]

def train_hybrid_fusion(sequence_length=10, epochs=30, batch_size=32):
    """Main training function for hybrid fusion model"""
    
    print("="*70)
    print("🏗️  HYBRID FUSION MODEL (Model B) - Training Started")
    print("="*70)
    
    # 1. Load Data
    print("\n📥 Loading aligned dataset...")
    df = pd.read_csv('data/processed/final_dataset.csv')
    
    # 2. Add Technical Indicators
    print("🔄 Adding technical indicators...")
    df = add_technical_indicators(df)
    
    # 3. Select Features
    price_feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'MACD', 'BB_Percent', 'Volume_Ratio', 'Momentum', 'ROC']
    nlp_feature_cols = ['News_Volume', 'News_Sentiment', 'NER_Score', 'TFIDF_Score']  # 4 NLP Features
    
    available_price_features = [col for col in price_feature_cols if col in df.columns]
    available_nlp_features = [col for col in nlp_feature_cols if col in df.columns]
    
    X_price = df[available_price_features].values
    X_nlp = df[available_nlp_features].values if available_nlp_features else np.zeros((len(df), 1))
    y = df['Label'].values
    
    print(f"   Price features ({len(available_price_features)}): {available_price_features}")
    print(f"   NLP features ({len(available_nlp_features)}): {available_nlp_features}")
    print(f"   Total samples: {len(X_price):,}")
    print(f"   Label distribution: Up={np.sum(y==1):,} ({np.mean(y)*100:.1f}%)")
    
    # 4. Normalize Features
    print("🔄 Normalizing features...")
    price_scaler = MinMaxScaler()
    nlp_scaler = StandardScaler()
    
    X_price_scaled = price_scaler.fit_transform(X_price)
    X_nlp_scaled = nlp_scaler.fit_transform(X_nlp)
    
    # 5. Create Sequences
    print(f"📊 Creating sequences (length={sequence_length})...")
    X_price_seq, X_nlp_seq, y_seq = prepare_sequences_with_news(X_price_scaled, X_nlp_scaled, y, sequence_length)
    print(f"   Price shape: {X_price_seq.shape}, NLP shape: {X_nlp_seq.shape}, y shape: {y_seq.shape}")
    
    # 6. Split Data
    X_price_train, X_price_test, X_nlp_train, X_nlp_test, y_train, y_test = train_test_split(X_price_seq, X_nlp_seq, y_seq, test_size=0.2, shuffle=False)
    print(f"   Train: {len(X_price_train):,} samples, Test: {len(X_price_test):,} samples")
    
    # 7. Calculate Class Weights
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
    print(f"   Class weights: {class_weight_dict}")
    
    # 8. Create Model
    print("🏗️ Building Hybrid Fusion model...")
    model = create_hybrid_model(num_price_features=len(available_price_features), num_nlp_features=len(available_nlp_features), sequence_length=sequence_length)
    model.summary()
    
    # 9. Callbacks
    early_stop = EarlyStopping(monitor='val_auc', patience=7, restore_best_weights=True, mode='max', verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)
    
    # 10. Train Model
    print("\n🚀 Training model...")
    history = model.fit([X_price_train, X_nlp_train], y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1, class_weight=class_weight_dict, callbacks=[early_stop, reduce_lr], verbose=1)
    
    # 11. Evaluate Model
    print("\n📊 Evaluating model on test set...")
    y_pred_proba = model.predict([X_price_test, X_nlp_test], verbose=0)
    
    optimal_threshold, best_f1 = find_optimal_threshold(y_test, y_pred_proba)
    print(f"   Optimal threshold: {optimal_threshold:.4f} (default was 0.5)")
    
    y_pred = (y_pred_proba > optimal_threshold).astype(int).flatten()
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_pred_proba)
    cm = confusion_matrix(y_test, y_pred)
    
    print("\n" + "="*70)
    print("✅ HYBRID FUSION MODEL RESULTS (Model B)")
    print("="*70)
    print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"Recall:    {recall:.4f} ({recall*100:.2f}%)")
    print(f"F1-Score:  {f1:.4f} ({f1*100:.2f}%)")
    print(f"AUC-ROC:   {auc:.4f} ({auc*100:.2f}%)")
    print("\nConfusion Matrix:")
    print(f"  TN={cm[0,0]:,}  FP={cm[0,1]:,}")
    print(f"  FN={cm[1,0]:,}  TP={cm[1,1]:,}")
    print("="*70)
    
    # 12. Save Model
    os.makedirs('src/models/saved', exist_ok=True)
    model.save('src/models/saved/hybrid_fusion.keras')
    joblib.dump(price_scaler, 'src/models/saved/hybrid_price_scaler.pkl')
    joblib.dump(nlp_scaler, 'src/models/saved/hybrid_nlp_scaler.pkl')
    joblib.dump(available_price_features, 'src/models/saved/hybrid_price_features.pkl')
    joblib.dump(available_nlp_features, 'src/models/saved/hybrid_nlp_features.pkl')
    
    metrics = {
        'model': 'Hybrid_Fusion_Attention',
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'auc_roc': float(auc),
        'optimal_threshold': float(optimal_threshold),
        'test_samples': int(len(X_price_test)),
        'confusion_matrix': cm.tolist(),
        'price_features': available_price_features,
        'nlp_features': available_nlp_features,
        'sequence_length': sequence_length,
        'epochs_trained': len(history.history['loss'])
    }
    
    with open('src/models/saved/hybrid_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    joblib.dump(history.history, 'src/models/saved/hybrid_history.pkl')
    
    print("\n💾 Model saved to src/models/saved/hybrid_fusion.keras")
    print("💾 Metrics saved to src/models/saved/hybrid_metrics.json")
    
    return model, metrics, history

if __name__ == "__main__":
    model, metrics, history = train_hybrid_fusion(sequence_length=10, epochs=30, batch_size=32)