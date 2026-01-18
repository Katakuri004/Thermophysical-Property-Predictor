"""
Generate submission CSV using HierarchicalMP v4.0 with EXPANDED DATA

Uses 278k external molecules (same as GODMODE) for high coverage.
"""

import sys
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
import time
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import v4.0 predictor
from src.models.hierarchical_mp_v4 import HierarchicalMPPredictorV4

def load_external_data():
    """Load and combine all data sources."""
    print("  Loading Kaggle train...")
    kaggle_train = pd.read_csv('data/raw/train.csv')
    kaggle_train = kaggle_train[['SMILES', 'Tm']].rename(columns={'Tm': 'Tm_K'})
    kaggle_train['source'] = 'kaggle'
    print(f"    Kaggle train: {len(kaggle_train)} molecules")
    
    print("  Loading external SMP data...")
    smp_df = pd.read_csv('data/raw/smiles_melting_point.csv')
    
    # Use the Kelvin column
    smp_df = smp_df[['SMILES', 'Melting Point {measured, converted}']].copy()
    smp_df.columns = ['SMILES', 'Tm_K']
    smp_df['source'] = 'smp'
    
    # Drop missing values
    smp_df = smp_df.dropna(subset=['SMILES', 'Tm_K'])
    
    # Convert Tm to float if needed
    smp_df['Tm_K'] = pd.to_numeric(smp_df['Tm_K'], errors='coerce')
    smp_df = smp_df.dropna(subset=['Tm_K'])
    
    print(f"    SMP external: {len(smp_df)} molecules")
    
    # Try loading Bradley if available
    try:
        print("  Loading Bradley data...")
        bradley_df = pd.read_excel('data/raw/BradleyMeltingPointDataset.xlsx')
        
        # Find the right columns
        if 'smiles' in bradley_df.columns:
            smiles_col = 'smiles'
        elif 'SMILES' in bradley_df.columns:
            smiles_col = 'SMILES'
        else:
            smiles_col = bradley_df.columns[0]
        
        # Find Tm column (usually in Kelvin or Celsius)
        tm_col = None
        for col in bradley_df.columns:
            if 'mp' in col.lower() or 'melt' in col.lower() or 'tm' in col.lower():
                tm_col = col
                break
        
        if tm_col:
            bradley_df = bradley_df[[smiles_col, tm_col]].copy()
            bradley_df.columns = ['SMILES', 'Tm_K']
            bradley_df['source'] = 'bradley'
            bradley_df = bradley_df.dropna()
            
            # Convert to Kelvin if values look like Celsius
            if bradley_df['Tm_K'].mean() < 200:  # Probably Celsius
                bradley_df['Tm_K'] = bradley_df['Tm_K'] + 273.15
            
            print(f"    Bradley: {len(bradley_df)} molecules")
        else:
            bradley_df = pd.DataFrame()
    except Exception as e:
        print(f"    Bradley loading failed: {e}")
        bradley_df = pd.DataFrame()
    
    # Combine all sources
    all_dfs = [df for df in [kaggle_train, smp_df, bradley_df] if len(df) > 0]
    combined = pd.concat(all_dfs, ignore_index=True)
    
    # Deduplicate by SMILES, prioritizing Kaggle data
    combined['priority'] = combined['source'].map({'kaggle': 0, 'bradley': 1, 'smp': 2})
    combined = combined.sort_values('priority').drop_duplicates(subset='SMILES', keep='first')
    combined = combined.drop(columns=['priority'])
    
    print(f"  Combined (deduplicated): {len(combined)} molecules")
    
    return combined

def main():
    print("=" * 60)
    print("HierarchicalMP v4.0 Submission (EXPANDED DATA)")
    print("=" * 60)
    
    # Load expanded data
    print("\n[1/4] Loading expanded data sources...")
    expanded_df = load_external_data()
    
    # Load test set
    test_df = pd.read_csv('data/raw/test.csv')
    print(f"\n  Test: {len(test_df)} molecules")
    
    # Build predictor with expanded data
    print(f"\n[2/4] Building HierarchicalMP v4.0 predictor with {len(expanded_df)} molecules...")
    predictor = HierarchicalMPPredictorV4(
        exact_threshold=0.95,
        similarity_threshold=0.7,
        n_neighbors=50,
        top_k=10,
        nprobe=32,  # Higher nprobe for larger dataset
        n_workers=4,
        use_rar=True
    )
    
    start = time.time()
    predictor.build_index(
        expanded_df['SMILES'].tolist(),
        expanded_df['Tm_K'].values,
        train_rar=True,
        train_fallback=True
    )
    build_time = time.time() - start
    print(f"  Build time: {build_time:.1f}s")
    
    # Print config
    config = predictor.get_config()
    print(f"\n  Configuration:")
    print(f"    - Molecules indexed: {config['n_molecules']}")
    print(f"    - Packed FP size: {config['packed_fp_bytes'] / 1024 / 1024:.1f} MB")
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
    
    # Coverage analysis
    exact_pct = method_counts.get('exact', 0) / len(results) * 100
    rar_pct = method_counts.get('rar', 0) / len(results) * 100
    print(f"\n  HIGH QUALITY COVERAGE: {exact_pct + rar_pct:.1f}% (exact + rar)")
    
    # Create submission
    print("\n[4/4] Creating submission...")
    submission = pd.DataFrame({
        'id': test_df['id'],
        'Tm': results['Tm_pred']
    })
    
    # Save
    Path('submissions').mkdir(exist_ok=True)
    submission_path = 'submissions/submission_hierarchical_v4_expanded.csv'
    submission.to_csv(submission_path, index=False)
    print(f"  Saved: {submission_path}")
    
    # Summary statistics
    print(f"\n=== Submission Summary ===")
    print(f"  Rows: {len(submission)}")
    print(f"  Tm min: {submission['Tm'].min():.1f} K")
    print(f"  Tm max: {submission['Tm'].max():.1f} K")
    print(f"  Tm mean: {submission['Tm'].mean():.1f} K")
    print(f"  Tm std: {submission['Tm'].std():.1f} K")
    
    # Save predictor
    predictor.save('models/hierarchical_mp_v4_expanded')
    
    # Save detailed results
    detailed_path = 'submissions/submission_hierarchical_v4_expanded_detailed.csv'
    results_with_id = results.copy()
    results_with_id['id'] = test_df['id'].values
    results_with_id.to_csv(detailed_path, index=False)
    print(f"  Detailed results: {detailed_path}")
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
