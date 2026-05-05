"""
preprocess_ml.py - ML Feature Engineering (Prices & Technical Indicators)
Author: [Member 2 Name]
Course: [Course Name]
Description: Processes raw price data, adds technical indicators, and saves ML features.
"""

import pandas as pd
import numpy as np
import os

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

def preprocess_ml_features():
    """Main ML preprocessing pipeline"""
    print("="*70)
    print("🏗️  ML PREPROCESSING PIPELINE (Member 2)")
    print("="*70)
    
    os.makedirs('data/processed', exist_ok=True)
    
    # 1. Load Raw Prices
    print("\n📥 Loading raw price data...")
    df_prices = pd.read_csv('data/processed/prices_raw.csv')
    print(f"   Raw price samples: {len(df_prices):,}")
    
    # 2. Reshape from Wide to Long Format
    print("🔄 Reshaping price data from wide to long format...")
    price_cols = [col for col in df_prices.columns if col != 'Date']
    symbols = list(set([col.split('_')[-1] for col in price_cols]))
    symbols = [s for s in symbols if s and s not in ['Open', 'High', 'Low', 'Close', 'Volume'] and len(s) >= 2 and s.isupper()]
    
    features = ['Close', 'Open', 'High', 'Low', 'Volume']
    df_long = []
    
    for symbol in symbols:
        df_stock = pd.DataFrame()
        df_stock['Date'] = pd.to_datetime(df_prices['Date']).dt.date
        df_stock['Symbol'] = symbol
        
        for feature in features:
            col_name = f'{feature}_{symbol}'
            if col_name in df_prices.columns:
                df_stock[feature] = df_prices[col_name]
            else:
                df_stock[feature] = np.nan
        df_long.append(df_stock)
    
    df_prices_long = pd.concat(df_long, ignore_index=True)
    df_prices_long = df_prices_long.dropna(subset=['Close', 'Open', 'High', 'Low', 'Volume'])
    print(f"   Reshaped price samples: {len(df_prices_long):,}")
    
    # 3. Add Technical Indicators
    print("🔄 Adding technical indicators...")
    df_ml = add_technical_indicators(df_prices_long)
    
    # 4. Save ML Features
    output_path = 'data/processed/ml_features.csv'
    df_ml.to_csv(output_path, index=False)
    print(f"💾 ML Features saved to: {output_path}")
    print(f"   Columns: {df_ml.columns.tolist()}")
    
    return df_ml

if __name__ == "__main__":
    preprocess_ml_features()