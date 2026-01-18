"""
Generate submission CSV using HierarchicalMP v4.0

This script:
1. Loads training data
2. Builds HierarchicalMP v4.0 predictor
3. Predicts on test set
4. Saves submission CSV
"""

import sys
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
import time
from pathlib import Path

# Import v4.0 predictor
from src.models.hierarchical_mp_v4 import HierarchicalMPPredictorV4

def main():
    print("=" * 60)
    print("HierarchicalMP v4.0 Submission Generator")
    print("=" * 60)
    
    # Load data
    print("\n[1/4] Loading data...")
    train_df = pd.read_csv('data/raw/train.csv')
    test_df = pd.read_csv('data/raw/test.csv')
    print(f"  Train: {len(train_df)} molecules")
    print(f"  Test: {len(test_df)} molecules")
    
    # Build predictor
    print("\n[2/4] Building HierarchicalMP v4.0 predictor...")
    predictor = HierarchicalMPPredictorV4(
        exact_threshold=0.95,
        similarity_threshold=0.7,
        n_neighbors=50,
        top_k=10,
        nprobe=16,
        n_workers=4,
        use_rar=True
    )
    
    start = time.time()
    predictor.build_index(
        train_df['SMILES'].tolist(),
        train_df['Tm'].values,
        train_rar=True,
        train_fallback=True
    )
    build_time = time.time() - start
    print(f"  Build time: {build_time:.1f}s")
    
    # Print config
    config = predictor.get_config()
    print(f"\n  Configuration:")
    print(f"    - Molecules indexed: {config['n_molecules']}")
    print(f"    - Packed FP size: {config['packed_fp_bytes'] / 1024:.1f} KB")
    print(f"    - Train mean: {config['train_mean']:.1f} K")
    
    # Predict
    print("\n[3/4] Running batch prediction...")
    start = time.time()
    results = predictor.predict_batch(test_df['SMILES'].tolist())
    pred_time = time.time() - start
    print(f"  Prediction time: {pred_time:.2f}s")
    print(f"  Throughput: {len(test_df)/pred_time:.0f} mol/s")
    
    # Method distribution
    method_counts = results['method'].value_counts()
    print(f"\n  Method distribution:")
    for method, count in method_counts.items():
        pct = count / len(results) * 100
        print(f"    {method}: {count} ({pct:.1f}%)")
    
    # Create submission
    print("\n[4/4] Creating submission...")
    submission = pd.DataFrame({
        'id': test_df['id'],
        'Tm': results['Tm_pred']
    })
    
    # Save
    Path('submissions').mkdir(exist_ok=True)
    submission_path = 'submissions/submission_hierarchical_v4.csv'
    submission.to_csv(submission_path, index=False)
    print(f"  Saved: {submission_path}")
    
    # Summary statistics
    print(f"\n=== Submission Summary ===")
    print(f"  Rows: {len(submission)}")
    print(f"  Tm min: {submission['Tm'].min():.1f} K")
    print(f"  Tm max: {submission['Tm'].max():.1f} K")
    print(f"  Tm mean: {submission['Tm'].mean():.1f} K")
    print(f"  Tm std: {submission['Tm'].std():.1f} K")
    
    # Save predictor for future use
    predictor.save('models/hierarchical_mp_v4')
    
    # Save detailed results
    detailed_path = 'submissions/submission_hierarchical_v4_detailed.csv'
    results_with_id = results.copy()
    results_with_id['id'] = test_df['id'].values
    results_with_id.to_csv(detailed_path, index=False)
    print(f"  Detailed results: {detailed_path}")
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
