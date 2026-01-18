"""
Generate submission using HierarchicalMP v7.0.
Demonstrates proper separation of training, calibration, and prediction.
"""

import sys
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
import time
from pathlib import Path
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

from src.models.hierarchical_mp_v7 import HierarchicalMPPredictorV7


def load_all_data():
    """Load all data sources."""
    print("Loading data sources...")
    
    # Kaggle
    kaggle = pd.read_csv('data/raw/train.csv')[['SMILES', 'Tm']].copy()
    kaggle['source'] = 'kaggle'
    print(f"  Kaggle: {len(kaggle)}")
    
    # SMP
    smp = pd.read_csv('data/raw/smiles_melting_point.csv')
    smp = smp[['SMILES', 'Melting Point {measured, converted}']].copy()
    smp.columns = ['SMILES', 'Tm']
    smp = smp.dropna()
    smp['Tm'] = pd.to_numeric(smp['Tm'], errors='coerce')
    smp = smp.dropna()
    smp['source'] = 'smp'
    print(f"  SMP: {len(smp)}")
    
    # Bradley
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
    print("HierarchicalMP v7.0 Submission Generator")
    print("=" * 70)
    
    # Load data
    print("\n[1/6] Loading data...")
    all_data = load_all_data()
    test_df = pd.read_csv('data/raw/test.csv')
    print(f"  Total: {len(all_data)}, Test: {len(test_df)}")
    
    # PROPER SPLIT: Train (90%) / Calibration (10%)
    print("\n[2/6] Splitting into train/calibration...")
    train_data, calib_data = train_test_split(
        all_data, test_size=0.1, random_state=42
    )
    print(f"  Train: {len(train_data)}, Calibration: {len(calib_data)}")
    
    # Create predictor
    predictor = HierarchicalMPPredictorV7(
        exact_threshold=0.95,
        similarity_threshold=0.7,
        n_neighbors=50,
        top_k=10,
        nprobe=32,
        alpha=0.1
    )
    
    # FIT INDEX (on training data only)
    print("\n[3/6] Fitting index on training data...")
    start = time.time()
    predictor.fit_index(
        train_data['SMILES'].tolist(),
        train_data['Tm'].values
    )
    build_time = time.time() - start
    print(f"  Build time: {build_time:.1f}s")
    
    # FIT CALIBRATION (on held-out calibration set)
    print("\n[4/6] Calibrating intervals on held-out set...")
    start = time.time()
    predictor.fit_calibration(
        calib_data['SMILES'].tolist(),
        calib_data['Tm'].values
    )
    calib_time = time.time() - start
    print(f"  Calibration time: {calib_time:.1f}s")
    
    # Show calibration results
    print("\n  Conformal corrections:")
    for method, corr in predictor.conformal.corrections.items():
        n = predictor.conformal.n_samples.get(method, 0)
        print(f"    {method}: ±{corr:.1f}K (n={n})")
    
    # PREDICT
    print("\n[5/6] Running predictions on test set...")
    start = time.time()
    results = predictor.predict_batch(test_df['SMILES'].tolist())
    pred_time = time.time() - start
    print(f"  Time: {pred_time:.2f}s ({len(test_df)/pred_time:.0f} mol/s)")
    
    # Method distribution
    method_counts = results['method'].value_counts()
    print(f"\n  Method distribution:")
    for method, count in method_counts.items():
        pct = count / len(results) * 100
        print(f"    {method}: {count} ({pct:.1f}%)")
    
    # Exact SMILES vs near-exact breakdown
    exact_smiles_pct = method_counts.get('exact_smiles', 0) / len(results) * 100
    near_exact_pct = method_counts.get('near_exact', 0) / len(results) * 100
    print(f"\n  True exact SMILES matches: {exact_smiles_pct:.1f}%")
    print(f"  Near-exact (Tanimoto>=0.95): {near_exact_pct:.1f}%")
    print(f"  Total high-quality: {exact_smiles_pct + near_exact_pct:.1f}%")
    
    # Cache utilization
    cache_hits = results['from_cache'].sum()
    print(f"\n  Cache hits (FAISS bypassed): {cache_hits} ({cache_hits/len(results)*100:.1f}%)")
    
    # Interval stats
    print(f"\n  Interval statistics:")
    print(f"    Mean width: {results['interval_width'].mean():.1f}K")
    print(f"    Median width: {results['interval_width'].median():.1f}K")
    
    # Create submission
    print("\n[6/6] Creating submission...")
    submission = pd.DataFrame({
        'id': test_df['id'],
        'Tm': results['Tm_pred']
    })
    
    Path('submissions').mkdir(exist_ok=True)
    submission.to_csv('submissions/submission_v7.csv', index=False)
    print("  Saved: submissions/submission_v7.csv")
    
    # Summary
    print(f"\n=== Submission Summary ===")
    print(f"  Rows: {len(submission)}")
    print(f"  Tm range: [{submission['Tm'].min():.1f}, {submission['Tm'].max():.1f}] K")
    print(f"  Tm mean: {submission['Tm'].mean():.1f} K")
    
    # Detailed
    results['id'] = test_df['id'].values
    results.to_csv('submissions/submission_v7_detailed.csv', index=False)
    
    # Save predictor
    predictor.save('models/hierarchical_mp_v7')
    
    print("\n" + "=" * 70)
    print("v7.0 Key Improvements:")
    print("  ✓ True uint64 popcount Tanimoto (scalable)")
    print("  ✓ Exact SMILES dict short-circuit (bypasses FAISS)")
    print("  ✓ Proper train/calibration split (no leakage)")
    print("  ✓ Split conformal absolute error (correctly named)")
    print("  ✓ Minimal state persistence (no class pickling)")
    print("=" * 70)


if __name__ == "__main__":
    main()
