"""
Generate submission using HierarchicalMP v6.0 with GPU acceleration.
"""

import sys
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
import time
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from src.models.hierarchical_mp_v6 import HierarchicalMPPredictorV6, FAISS_GPU

def load_all_data():
    """Load all data sources."""
    print("Loading data sources...")
    
    kaggle = pd.read_csv('data/raw/train.csv')[['SMILES', 'Tm']].copy()
    kaggle['source'] = 'kaggle'
    print(f"  Kaggle: {len(kaggle)}")
    
    smp = pd.read_csv('data/raw/smiles_melting_point.csv')
    smp = smp[['SMILES', 'Melting Point {measured, converted}']].copy()
    smp.columns = ['SMILES', 'Tm']
    smp = smp.dropna()
    smp['Tm'] = pd.to_numeric(smp['Tm'], errors='coerce')
    smp = smp.dropna()
    smp['source'] = 'smp'
    print(f"  SMP: {len(smp)}")
    
    try:
        bradley = pd.read_excel('data/raw/BradleyMeltingPointDataset.xlsx')
        smiles_col = [c for c in bradley.columns if 'smiles' in c.lower()][0]
        tm_col = [c for c in bradley.columns if 'mp' in c.lower()][0]
        bradley = bradley[[smiles_col, tm_col]].copy()
        bradley.columns = ['SMILES', 'Tm']
        bradley = bradley.dropna()
        bradley['Tm'] = pd.to_numeric(bradley['Tm'], errors='coerce')
        if bradley['Tm'].mean() < 200:
            bradley['Tm'] = bradley['Tm'] + 273.15
        bradley['source'] = 'bradley'
        print(f"  Bradley: {len(bradley)}")
    except Exception as e:
        print(f"  Bradley failed: {e}")
        bradley = pd.DataFrame(columns=['SMILES', 'Tm', 'source'])
    
    return pd.concat([kaggle, smp, bradley], ignore_index=True)

def main():
    print("=" * 70)
    print("HierarchicalMP v6.0 Submission Generator")
    print(f"FAISS GPU available: {FAISS_GPU}")
    print("=" * 70)
    
    # Load data
    print("\n[1/5] Loading data...")
    all_data = load_all_data()
    test_df = pd.read_csv('data/raw/test.csv')
    print(f"  Total: {len(all_data)}, Test: {len(test_df)}")
    
    # Build predictor
    print("\n[2/5] Building v6.0 predictor...")
    predictor = HierarchicalMPPredictorV6(
        exact_threshold=0.95,
        similarity_threshold=0.7,
        n_neighbors=50,
        top_k=10,
        nprobe=32,
        use_gpu=True,
        use_ivfpq=False,
    )
    
    start = time.time()
    predictor.build_index(
        all_data['SMILES'].tolist(),
        all_data['Tm'].values,
        all_data['source'].tolist()
    )
    build_time = time.time() - start
    print(f"  Build time: {build_time:.1f}s")
    
    # Config
    config = predictor.get_config()
    print(f"\n  Configuration:")
    for k, v in config.items():
        print(f"    {k}: {v}")
    
    # Predict
    print("\n[3/5] Running predictions...")
    predictor.metrics.reset()
    start = time.time()
    results = predictor.predict_batch(test_df['SMILES'].tolist())
    pred_time = time.time() - start
    print(f"  Time: {pred_time:.2f}s ({len(test_df)/pred_time:.0f} mol/s)")
    
    # Metrics
    metrics = predictor.get_metrics()
    print(f"\n  Pipeline breakdown:")
    for stage, pct in metrics['breakdown'].items():
        print(f"    {stage}: {pct:.1f}%")
    print(f"  GPU enabled: {metrics['using_gpu']}")
    
    # Method distribution
    method_counts = results['method'].value_counts()
    print(f"\n[4/5] Method distribution:")
    for method, count in method_counts.items():
        pct = count / len(results) * 100
        print(f"    {method}: {count} ({pct:.1f}%)")
    
    # Coverage
    exact_pct = method_counts.get('exact', 0) / len(results) * 100
    print(f"\n  Exact match coverage: {exact_pct:.1f}%")
    
    # Interval stats
    print(f"\n  Interval statistics:")
    print(f"    Mean width: {results['interval_width'].mean():.1f} K")
    print(f"    Median width: {results['interval_width'].median():.1f} K")
    
    # Create submission
    print("\n[5/5] Creating submission...")
    submission = pd.DataFrame({
        'id': test_df['id'],
        'Tm': results['Tm_pred']
    })
    
    Path('submissions').mkdir(exist_ok=True)
    submission_path = 'submissions/submission_v6.csv'
    submission.to_csv(submission_path, index=False)
    print(f"  Saved: {submission_path}")
    
    # Stats
    print(f"\n=== Submission Summary ===")
    print(f"  Rows: {len(submission)}")
    print(f"  Tm range: [{submission['Tm'].min():.1f}, {submission['Tm'].max():.1f}] K")
    print(f"  Tm mean: {submission['Tm'].mean():.1f} K")
    
    # Detailed
    results['id'] = test_df['id'].values
    results.to_csv('submissions/submission_v6_detailed.csv', index=False)
    
    # Save predictor
    predictor.save('models/hierarchical_mp_v6')
    
    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)

if __name__ == "__main__":
    main()
