"""
fetch_prices.py - Pull stock prices from Yahoo Finance via yfinance
Author: [Member 1 Name]
Course: [Course Name]
"""

import os
import pandas as pd
import yfinance as yf
import time

def fetch_stock_prices(tickers=None, start_date='2021-01-01', end_date='2026-04-01'):
    """Pull price data and save to processed folder"""
    
    # Create output directory if not exists
    os.makedirs('data/processed', exist_ok=True)
    
    # EXPANDED: Top 50 S&P 500 stocks (guarantees ≥50k samples)
    if tickers is None:
        tickers = [
            # Technology
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'ADBE', 'CRM', 'ORCL',
            'INTC', 'AMD', 'QCOM', 'TXN', 'AVGO', 'CSCO', 'IBM', 'NOW', 'INTU', 'MU',
            # Finance
            'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'BLK', 'SCHW', 'AXP', 'USB',
            # Healthcare
            'JNJ', 'UNH', 'PFE', 'MRK', 'ABBV', 'TMO', 'ABT', 'DHR', 'BMY', 'LLY',
            # Consumer
            'WMT', 'PG', 'KO', 'PEP', 'COST', 'HD', 'MCD', 'NKE', 'SBUX', 'TGT',
            # Energy & Industrial
            'XOM', 'CVX', 'CAT', 'BA', 'HON', 'UPS', 'GE', 'MMM', 'LMT', 'RTX'
        ]
    
    print(f"📥 Pulling Stock Prices from Yahoo Finance API for {len(tickers)} stocks...")
    
    # Download with rate limiting to avoid 'database is locked' errors
    all_data = []
    failed_tickers = []
    
    for i, ticker in enumerate(tickers):
        try:
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if not data.empty:
                data['Symbol'] = ticker
                data.reset_index(inplace=True)
                all_data.append(data)
                print(f"   ✓ {ticker} ({len(data)} rows)")
            else:
                failed_tickers.append(ticker)
                print(f"   ✗ {ticker} (empty data)")
            
            # Rate limiting: wait 0.5 seconds between requests
            if i % 10 == 0 and i > 0:
                time.sleep(1)
                
        except Exception as e:
            failed_tickers.append(ticker)
            print(f"   ✗ {ticker} (Error: {e})")
            time.sleep(1)  # Wait before retry
    
    # Combine all data
    if all_data:
        df_prices = pd.concat(all_data, ignore_index=True)
        
        # Clean multi-index columns if present
        if isinstance(df_prices.columns, pd.MultiIndex):
            df_prices.columns = ['_'.join(col).strip() if col[1] else col[0] 
                                 for col in df_prices.columns.values]
        
        # Calculate total samples
        total_samples = len(df_prices)
        print(f"\n✅ Total Price Samples: {total_samples:,} (Required: ≥50,000)")
        print(f"📅 Date Range: {df_prices['Date'].min()} to {df_prices['Date'].max()}")
        print(f"📊 Stocks Successfully Downloaded: {len(all_data)}/{len(tickers)}")
        
        if failed_tickers:
            print(f"⚠️ Failed Downloads: {failed_tickers}")
        
        # Verify requirement
        if total_samples >= 50000:
            print("✅ PASS: Meets ≥50,000 samples requirement (Guideline 2.3)")
        else:
            print("❌ FAIL: Below 50,000 samples. Add more stocks or extend date range.")
        
        # Save for alignment phase
        output_path = 'data/processed/prices_raw.csv'
        df_prices.to_csv(output_path, index=False)
        print(f"💾 Saved: {output_path}")
        
        return df_prices
    else:
        print("❌ ERROR: No data downloaded. Check internet connection.")
        return None

if __name__ == "__main__":
    fetch_stock_prices()