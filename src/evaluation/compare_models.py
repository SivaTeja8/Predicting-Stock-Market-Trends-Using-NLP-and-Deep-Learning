"""
compare_models.py - Compare Model A vs Model B and select best algorithm
Author: [Member 4 Name]
Course: [Course Name]
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def compare_models():
    """Load and compare metrics from both models"""
    
    print("="*70)
    print("📊 MODEL COMPARISON & BEST ALGORITHM SELECTION")
    print("="*70)
    
    with open('src/models/saved/baseline_metrics.json', 'r') as f:
        baseline = json.load(f)
    
    with open('src/models/saved/hybrid_metrics.json', 'r') as f:
        hybrid = json.load(f)
    
    comparison = pd.DataFrame({
        'Model': ['Baseline LSTM (A)', 'Hybrid Fusion (B)'],
        'Accuracy': [baseline['accuracy'], hybrid['accuracy']],
        'Precision': [baseline['precision'], hybrid['precision']],
        'Recall': [baseline['recall'], hybrid['recall']],
        'F1-Score': [baseline['f1_score'], hybrid['f1_score']],
        'AUC-ROC': [baseline['auc_roc'], hybrid['auc_roc']],
        'Test Samples': [baseline['test_samples'], hybrid['test_samples']]
    })
    
    print("\n📋 MODEL COMPARISON TABLE (For IEEE Report Section 6)")
    print("="*70)
    print(comparison.to_string(index=False))
    print("="*70)
    
    metrics_to_compare = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']
    improvement = {}
    for metric in metrics_to_compare:
        if baseline[metric] > 0:
            improvement[metric] = (hybrid[metric] - baseline[metric]) / baseline[metric] * 100
        else:
            improvement[metric] = 0
    
    print("\n📈 Improvement (Hybrid vs Baseline):")
    for metric, value in improvement.items():
        symbol = "↑" if value > 0 else "↓"
        print(f"   {metric.capitalize():12s}: {value:+.2f}% {symbol}")
    
    best_model = 'Hybrid Fusion (B)' if hybrid['f1_score'] >= baseline['f1_score'] else 'Baseline LSTM (A)'
    best_f1 = max(hybrid['f1_score'], baseline['f1_score'])
    
    print("\n" + "="*70)
    print(f"🏆 BEST ALGORITHM: {best_model}")
    print(f"   Best F1-Score: {best_f1:.4f}")
    print("="*70)
    
    os.makedirs('src/evaluation/results', exist_ok=True)
    comparison.to_csv('src/evaluation/results/model_comparison.csv', index=False)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']
    titles = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']
    colors = ['#3498db', '#e74c3c']
    
    for idx, (metric, title) in enumerate(zip(metrics_to_plot, titles)):
        ax = axes[idx // 3, idx % 3]
        ax.bar(['Baseline', 'Hybrid'], [baseline[metric], hybrid[metric]], color=colors, alpha=0.7, edgecolor='black')
        ax.set_ylabel('Score', fontsize=11)
        ax.set_title(f'{title} Comparison', fontsize=12, fontweight='bold')
        ax.set_ylim(0, 1)
        ax.grid(axis='y', alpha=0.3)
        
        for i, v in enumerate([baseline[metric], hybrid[metric]]):
            ax.text(i, v + 0.02, f'{v:.4f}', ha='center', fontweight='bold', fontsize=10)
    
    axes[1, 2].remove()
    plt.tight_layout()
    plt.savefig('src/evaluation/results/model_comparison.png', dpi=300, bbox_inches='tight')
    print("\n💾 Comparison chart saved to src/evaluation/results/model_comparison.png")
    
    best_model_info = {
        'best_model': best_model,
        'best_f1_score': best_f1,
        'baseline_f1': baseline['f1_score'],
        'hybrid_f1': hybrid['f1_score'],
        'recommendation': 'Use Hybrid Fusion for production if F1-Score is higher; otherwise use Baseline for simplicity'
    }
    
    with open('src/evaluation/results/best_model.json', 'w') as f:
        json.dump(best_model_info, f, indent=2)
    
    return comparison, improvement, best_model

if __name__ == "__main__":
    comparison, improvement, best_model = compare_models()