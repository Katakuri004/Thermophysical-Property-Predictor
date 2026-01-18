"""
Generate submission using HierarchicalMP v5.0 with external data.
"""

import sys
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
import time
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from src.models.hierarchical_mp_v5 import HierarchicalMPPredictorV5

def load_all_data():
    """Load and combine all data sources."""
    print("Loading data sources...")
    
    # Kaggle train
    kaggle = pd.read_csv('data/raw/train.csv')
    kaggle = kaggle[['SMILES', 'Tm']].copy()
    kaggle['source'] = 'kaggle'
    print(f"  Kaggle: {len(kaggle)}")
    
    # SMP external
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
        # Find columns
        smiles_col = [c for c in bradley.columns if 'smiles' in c.lower()][0]
        tm_col = [c for c in bradley.columns if 'mp' in c.lower() or 'melt' in c.lower()][0]
        bradley = bradley[[smiles_col, tm_col]].copy()
        bradley.columns = ['SMILES', 'Tm']
        bradley = bradley.dropna()
        bradley['Tm'] = pd.to_numeric(bradley['Tm'], errors='coerce')
        # Convert to K if needed
        if bradley['Tm'].mean() < 200:
            bradley['Tm'] = bradley['Tm'] + 273.15
        bradley['source'] = 'bradley'
        print(f"  Bradley: {len(bradley)}")
    except Exception as e:
        print(f"  Bradley failed: {e}")
        bradley = pd.DataFrame(columns=['SMILES', 'Tm', 'source'])
    
    # Combine
    all_data = pd.concat([kaggle, smp, bradley], ignore_index=True)
    print(f"  Total: {len(all_data)}")
    
    return all_data

def main():
    print("=" * 60)
    print("HierarchicalMP v5.0 Submission Generator")
    print("=" * 60)
    
    # Load data
    print("\n[1/4] Loading data...")
    all_data = load_all_data()
    test_df = pd.read_csv('data/raw/test.csv')
    print(f"  Test: {len(test_df)}")
    
    # Build predictor
    print("\n[2/4] Building v5.0 predictor...")
    predictor = HierarchicalMPPredictorV5(
        exact_threshold=0.95,
        similarity_threshold=0.7,
        n_neighbors=50,
        top_k=10,
        nprobe=32,
        use_binary_index=True,
        alpha=0.1
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
    print(f"\n  Config:")
    for k, v in config.items():
        print(f"    {k}: {v}")
    
    # Predict
    print("\n[3/4] Running predictions...")
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
    
    # Pipeline breakdown
    breakdown = predictor.get_pipeline_breakdown()
    print(f"\n  Pipeline timing:")
    for stage, pct in breakdown.items():
        print(f"    {stage}: {pct:.1f}%")
    
    # Coverage
    exact_pct = method_counts.get('exact', 0) / len(results) * 100
    rar_pct = method_counts.get('rar', 0) / len(results) * 100
    print(f"\n  High-quality coverage: {exact_pct + rar_pct:.1f}%")
    
    # Create submission
    print("\n[4/4] Creating submission...")
    submission = pd.DataFrame({
        'id': test_df['id'],
        'Tm': results['Tm_pred']
    })
    
    Path('submissions').mkdir(exist_ok=True)
    submission_path = 'submissions/submission_v5.csv'
    submission.to_csv(submission_path, index=False)
    print(f"  Saved: {submission_path}")
    
    # Stats
    print(f"\n=== Submission Summary ===")
    print(f"  Rows: {len(submission)}")
    print(f"  Tm range: [{submission['Tm'].min():.1f}, {submission['Tm'].max():.1f}] K")
    print(f"  Tm mean: {submission['Tm'].mean():.1f} K")
    print(f"  Tm std: {submission['Tm'].std():.1f} K")
    
    # Intervals summary
    print(f"\n=== Interval Summary ===")
    print(f"  Mean width: {results['interval_width'].mean():.1f} K")
    print(f"  Median width: {results['interval_width'].median():.1f} K")
    
    # Save detailed
    detailed_path = 'submissions/submission_v5_detailed.csv'
    results['id'] = test_df['id'].values
    results.to_csv(detailed_path, index=False)
    print(f"  Detailed: {detailed_path}")
    
    # Save predictor
    predictor.save('models/hierarchical_mp_v5')
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)

if __name__ == "__main__":
    main()
