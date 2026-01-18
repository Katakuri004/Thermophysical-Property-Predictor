"""
Scaffold Split Evaluation Framework
Provides rigorous evaluation for novel chemotype generalization.
"""

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import GroupKFold
from typing import List, Tuple, Dict, Optional
from collections import defaultdict


def get_murcko_scaffold(smiles: str, generic: bool = False) -> Optional[str]:
    """
    Extract Murcko scaffold from SMILES.
    
    Args:
        smiles: Input SMILES string
        generic: If True, return generic scaffold (all atoms → C, all bonds → single)
    
    Returns:
        Scaffold SMILES or None if failed
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        
        if generic:
            scaffold = MurckoScaffold.MakeScaffoldGeneric(scaffold)
        
        return Chem.MolToSmiles(scaffold, canonical=True)
    except:
        return None


def get_scaffold_groups(smiles_list: List[str], generic: bool = False) -> List[str]:
    """
    Get scaffold groups for a list of molecules.
    
    Returns:
        List of scaffold SMILES (same length as input)
    """
    scaffolds = []
    for smi in smiles_list:
        scaffold = get_murcko_scaffold(smi, generic=generic)
        if scaffold is None or scaffold == '':
            scaffold = smi  # Use original SMILES as unique group
        scaffolds.append(scaffold)
    return scaffolds


class ScaffoldSplitter:
    """
    Scaffold-based train/test splitter for rigorous evaluation.
    Ensures test set contains scaffolds NOT seen in training.
    """
    
    def __init__(self, generic_scaffold: bool = False):
        self.generic_scaffold = generic_scaffold
        self.scaffold_to_idx = {}
        
    def split(self, smiles_list: List[str], test_size: float = 0.2, 
              random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
        """
        Split data by scaffolds.
        
        Returns:
            (train_indices, test_indices)
        """
        np.random.seed(random_state)
        
        # Group molecules by scaffold
        scaffold_to_indices = defaultdict(list)
        for idx, smi in enumerate(smiles_list):
            scaffold = get_murcko_scaffold(smi, generic=self.generic_scaffold)
            if scaffold is None:
                scaffold = f"unique_{idx}"
            scaffold_to_indices[scaffold].append(idx)
        
        # Sort scaffolds by size (largest first for more deterministic split)
        scaffolds = list(scaffold_to_indices.keys())
        scaffolds.sort(key=lambda s: len(scaffold_to_indices[s]), reverse=True)
        
        # Assign scaffolds to train/test
        n_total = len(smiles_list)
        n_test = int(n_total * test_size)
        
        test_indices = []
        train_indices = []
        
        # Shuffle scaffolds
        np.random.shuffle(scaffolds)
        
        for scaffold in scaffolds:
            indices = scaffold_to_indices[scaffold]
            if len(test_indices) < n_test:
                test_indices.extend(indices)
            else:
                train_indices.extend(indices)
        
        return np.array(train_indices), np.array(test_indices)
    
    def get_scaffold_cv(self, smiles_list: List[str], n_splits: int = 5) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate scaffold-based cross-validation splits.
        
        Returns:
            List of (train_indices, test_indices) tuples
        """
        scaffolds = get_scaffold_groups(smiles_list, generic=self.generic_scaffold)
        
        # Create unique scaffold IDs
        unique_scaffolds = list(set(scaffolds))
        scaffold_to_id = {s: i for i, s in enumerate(unique_scaffolds)}
        groups = [scaffold_to_id[s] for s in scaffolds]
        
        # Use GroupKFold
        gkf = GroupKFold(n_splits=n_splits)
        splits = []
        
        X = np.arange(len(smiles_list))
        for train_idx, test_idx in gkf.split(X, groups=groups):
            splits.append((train_idx, test_idx))
        
        return splits


class ScaffoldBenchmark:
    """
    Comprehensive benchmarking with multiple split strategies.
    """
    
    def __init__(self, smiles: List[str], targets: np.ndarray):
        self.smiles = smiles
        self.targets = targets
        self.results = {}
        
    def evaluate(self, model_fn, split_types: List[str] = None) -> pd.DataFrame:
        """
        Evaluate model on multiple split types.
        
        Args:
            model_fn: Function that takes (X_train_smiles, y_train, X_test_smiles) 
                     and returns predictions
            split_types: List of split types to evaluate
        
        Returns:
            DataFrame with results
        """
        if split_types is None:
            split_types = ['random', 'scaffold', 'generic_scaffold']
        
        results = []
        
        for split_type in split_types:
            print(f"\nEvaluating {split_type} split...")
            
            if split_type == 'random':
                # Random 80/20 split
                np.random.seed(42)
                indices = np.random.permutation(len(self.smiles))
                split_point = int(0.8 * len(indices))
                train_idx, test_idx = indices[:split_point], indices[split_point:]
                
            elif split_type == 'scaffold':
                splitter = ScaffoldSplitter(generic_scaffold=False)
                train_idx, test_idx = splitter.split(self.smiles, test_size=0.2)
                
            elif split_type == 'generic_scaffold':
                splitter = ScaffoldSplitter(generic_scaffold=True)
                train_idx, test_idx = splitter.split(self.smiles, test_size=0.2)
            
            # Get data
            train_smiles = [self.smiles[i] for i in train_idx]
            test_smiles = [self.smiles[i] for i in test_idx]
            y_train = self.targets[train_idx]
            y_test = self.targets[test_idx]
            
            # Get predictions
            y_pred = model_fn(train_smiles, y_train, test_smiles)
            
            # Calculate metrics
            mae = np.mean(np.abs(y_test - y_pred))
            rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
            r2 = 1 - np.sum((y_test - y_pred) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2)
            
            results.append({
                'split_type': split_type,
                'train_size': len(train_idx),
                'test_size': len(test_idx),
                'MAE': mae,
                'RMSE': rmse,
                'R2': r2
            })
            
            print(f"  Train: {len(train_idx)}, Test: {len(test_idx)}")
            print(f"  MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.3f}")
        
        return pd.DataFrame(results)
    
    def analyze_scaffold_distribution(self) -> Dict:
        """Analyze scaffold distribution in the dataset."""
        scaffolds = get_scaffold_groups(self.smiles)
        
        scaffold_counts = defaultdict(int)
        for s in scaffolds:
            scaffold_counts[s] += 1
        
        counts = list(scaffold_counts.values())
        
        return {
            'n_molecules': len(self.smiles),
            'n_unique_scaffolds': len(scaffold_counts),
            'scaffold_coverage': len(scaffold_counts) / len(self.smiles),
            'singleton_scaffolds': sum(1 for c in counts if c == 1),
            'largest_scaffold_size': max(counts),
            'mean_scaffold_size': np.mean(counts),
            'median_scaffold_size': np.median(counts)
        }


def compute_scaffold_similarity(smiles1: str, smiles2: str) -> float:
    """
    Compute Tanimoto similarity between scaffolds.
    
    Returns:
        Similarity score [0, 1] or 0 if failed
    """
    try:
        from rdkit.Chem import AllChem, DataStructs
        
        scaffold1 = get_murcko_scaffold(smiles1)
        scaffold2 = get_murcko_scaffold(smiles2)
        
        if scaffold1 is None or scaffold2 is None:
            return 0.0
        
        mol1 = Chem.MolFromSmiles(scaffold1)
        mol2 = Chem.MolFromSmiles(scaffold2)
        
        if mol1 is None or mol2 is None:
            return 0.0
        
        fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits=1024)
        fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, nBits=1024)
        
        return DataStructs.TanimotoSimilarity(fp1, fp2)
    except:
        return 0.0


def classify_test_difficulty(train_smiles: List[str], test_smiles: List[str]) -> pd.DataFrame:
    """
    Classify test molecules by difficulty based on scaffold novelty.
    
    Returns:
        DataFrame with test SMILES and difficulty classification
    """
    train_scaffolds = set(get_scaffold_groups(train_smiles))
    
    results = []
    for smi in test_smiles:
        scaffold = get_murcko_scaffold(smi)
        
        if scaffold in train_scaffolds:
            difficulty = 'easy'  # Scaffold seen in training
        else:
            # Check similarity to nearest scaffold
            max_sim = 0
            for train_smi in train_smiles[:1000]:  # Sample for speed
                sim = compute_scaffold_similarity(smi, train_smi)
                max_sim = max(max_sim, sim)
            
            if max_sim > 0.7:
                difficulty = 'medium'  # Similar scaffold exists
            else:
                difficulty = 'hard'  # Novel scaffold
        
        results.append({
            'SMILES': smi,
            'scaffold': scaffold,
            'difficulty': difficulty
        })
    
    return pd.DataFrame(results)


if __name__ == "__main__":
    # Demo
    print("Scaffold Split Demo")
    print("=" * 50)
    
    # Sample data
    demo_smiles = [
        'c1ccccc1',      # Benzene
        'c1ccccc1C',     # Toluene
        'c1ccccc1CC',    # Ethylbenzene
        'c1ccc(O)cc1',   # Phenol
        'c1ccc(N)cc1',   # Aniline
        'CCO',           # Ethanol
        'CCCO',          # Propanol
        'CCCCO',         # Butanol
    ]
    demo_targets = np.array([278.7, 178.0, 178.2, 316.0, 267.0, 159.0, 147.0, 183.0])
    
    # Analyze scaffolds
    benchmark = ScaffoldBenchmark(demo_smiles, demo_targets)
    stats = benchmark.analyze_scaffold_distribution()
    
    print(f"\nScaffold Statistics:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    
    # Show scaffolds
    print(f"\nScaffolds:")
    for smi in demo_smiles:
        scaffold = get_murcko_scaffold(smi)
        print(f"  {smi} → {scaffold}")
