"""
baseline_lstm.py - Model A: LSTM on numerical price features (60% ML)
Author: [Member 2 Name]
Course: [Course Name]
Description: Baseline LSTM model using only numerical price features with technical indicators.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, precision_recall_curve
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, BatchNormalization
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
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
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

def create_lstm_model(input_shape, units1=64, units2=32, dropout=0.3, lr=0.001):
    """Create optimized LSTM model architecture"""
    model = Sequential([
        Input(shape=input_shape),
        LSTM(units1, return_sequences=True),
        BatchNormalization(),
        Dropout(dropout),
        LSTM(units2, return_sequences=False),
        BatchNormalization(),
        Dropout(dropout),
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dropout(dropout/2),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    optimizer = Adam(learning_rate=lr)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])
    return model

def prepare_sequences(data, labels, sequence_length=10):
    """Create sequences for LSTM input (labels remain single values)"""
    X, y = [], []
    for i in range(sequence_length, len(data)):
        X.append(data[i-sequence_length:i])
        y.append(labels[i])
    return np.array(X), np.array(y).flatten()

def find_optimal_threshold(y_true, y_pred_proba):
    """Find optimal threshold based on F1-Score"""
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
    return optimal_threshold, f1_scores[optimal_idx]

def train_baseline_lstm(sequence_length=10, epochs=30, batch_size=32):
    """Main training function for baseline LSTM model"""
    
    print("="*70)
    print("🏗️  BASELINE LSTM MODEL (Model A) - Training Started")
    print("="*70)
    
    # 1. Load Data
    print("\n📥 Loading aligned dataset...")
    df = pd.read_csv('data/processed/final_dataset.csv')
    
    # 2. Add Technical Indicators
    print("🔄 Adding technical indicators...")
    df = add_technical_indicators(df)
    
    # 3. Select Features (ML Features Only - No NLP)
    feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'MACD', 'BB_Percent', 'Volume_Ratio', 'Momentum', 'ROC']
    available_features = [col for col in feature_cols if col in df.columns]
    X = df[available_features].values
    y = df['Label'].values
    
    print(f"   Features ({len(available_features)}): {available_features}")
    print(f"   Total samples: {len(X):,}")
    print(f"   Label distribution: Up={np.sum(y==1):,} ({np.mean(y)*100:.1f}%)")
    
    # 4. Normalize Features
    print("🔄 Normalizing features...")
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 5. Create Sequences
    print(f"📊 Creating sequences (length={sequence_length})...")
    X_seq, y_seq = prepare_sequences(X_scaled, y, sequence_length)
    print(f"   X shape: {X_seq.shape}, y shape: {y_seq.shape}")
    
    # 6. Split Data
    X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, shuffle=False)
    print(f"   Train: {len(X_train):,} samples, Test: {len(X_test):,} samples")
    
    # 7. Calculate Class Weights
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
    print(f"   Class weights: {class_weight_dict}")
    
    # 8. Create Model
    print("🏗️ Building LSTM model...")
    model = create_lstm_model(input_shape=(sequence_length, len(available_features)), units1=64, units2=32, dropout=0.3, lr=0.001)
    model.summary()
    
    # 9. Callbacks
    early_stop = EarlyStopping(monitor='val_auc', patience=7, restore_best_weights=True, mode='max', verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)
    
    # 10. Train Model
    print("\n🚀 Training model...")
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1, class_weight=class_weight_dict, callbacks=[early_stop, reduce_lr], verbose=1)
    
    # 11. Evaluate Model
    print("\n📊 Evaluating model on test set...")
    y_pred_proba = model.predict(X_test, verbose=0)
    
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
    print("✅ BASELINE LSTM MODEL RESULTS (Model A)")
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
    model.save('src/models/saved/baseline_lstm.keras')
    joblib.dump(scaler, 'src/models/saved/baseline_scaler.pkl')
    joblib.dump(available_features, 'src/models/saved/baseline_features.pkl')
    
    metrics = {
        'model': 'Baseline_LSTM_Technical',
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'auc_roc': float(auc),
        'optimal_threshold': float(optimal_threshold),
        'test_samples': int(len(X_test)),
        'confusion_matrix': cm.tolist(),
        'features': available_features,
        'sequence_length': sequence_length,
        'epochs_trained': len(history.history['loss'])
    }
    
    with open('src/models/saved/baseline_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    joblib.dump(history.history, 'src/models/saved/baseline_history.pkl')
    
    print("\n💾 Model saved to src/models/saved/baseline_lstm.keras")
    print("💾 Metrics saved to src/models/saved/baseline_metrics.json")
    
    return model, metrics, history

if __name__ == "__main__":
    model, metrics, history = train_baseline_lstm(sequence_length=10, epochs=30, batch_size=32)