"""
Memory-Optimized Comprehensive Molecular Featurizer
Processes data in batches and uses efficient dtypes to handle 300k+ molecules.
"""

import pandas as pd
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors, Crippen, Lipinski, Fragments, AllChem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.AllChem import ComputeGasteigerCharges
from rdkit.Chem.MACCSkeys import GenMACCSKeys
from rdkit.Chem.EState import AtomTypes as EAtomTypes
from rdkit.Chem.Scaffolds import MurckoScaffold
from typing import Dict, Any, List, Optional
from tqdm import tqdm
import gc

# Silence RDKit warnings
RDLogger.DisableLog('rdApp.*')

MORGAN_BITS = 256  # Reduced from 512 to save memory
MORGAN_RADIUS = 2

class ComprehensiveFeaturizer:
    """
    Memory-optimized featurizer that processes data in batches.
    """
    
    def __init__(self, batch_size: int = 5000):
        self.descriptor_list = Descriptors.descList
        self.batch_size = batch_size
    
    def _safe(self, f, default=None):
        def wrap(*args, **kwargs):
            try:
                return f(*args, **kwargs)
            except Exception:
                return default
        return wrap

    def _count_atoms(self, m, symbols):
        s = set(symbols)
        return sum(1 for a in m.GetAtoms() if a.GetSymbol() in s)

    def _largest_ring_size(self, m):
        ri = m.GetRingInfo()
        return max((len(r) for r in ri.AtomRings()), default=0)

    def _ring_size_hist(self, m):
        ri = m.GetRingInfo()
        sizes = [len(r) for r in ri.AtomRings()]
        out = {5:0, 6:0, 7:0, 8:0}
        for s in sizes:
            if s in out: out[s] += 1
        return out, len(sizes)

    def _ring_systems_count(self, m):
        ri = m.GetRingInfo()
        rings = [set(r) for r in ri.AtomRings()]
        if not rings: return 0
        seen = set()
        sys_count = 0
        for i in range(len(rings)):
            if i in seen: continue
            sys_count += 1
            stack = [i]
            seen.add(i)
            while stack:
                j = stack.pop()
                for k in range(len(rings)):
                    if k in seen: continue
                    if rings[j] & rings[k]:
                        seen.add(k)
                        stack.append(k)
        return sys_count

    def _murcko_stats(self, m):
        try:
            scaf = MurckoScaffold.GetScaffoldForMol(m)
            if scaf is None or scaf.GetNumAtoms() == 0:
                return {"MurckoAtoms":0, "MurckoRings":0, "SideChainAtoms":m.GetNumAtoms()}
            return {
                "MurckoAtoms": scaf.GetNumAtoms(),
                "MurckoRings": rdMolDescriptors.CalcNumRings(scaf),
                "SideChainAtoms": m.GetNumAtoms() - scaf.GetNumAtoms(),
            }
        except Exception:
            return {"MurckoAtoms":0, "MurckoRings":0, "SideChainAtoms": 0}

    def _smiles_morphology(self, smi: str):
        if not smi: 
            return {"SMI_len":0, "SMI_branches":0, "SMI_ringDigits":0}
        return {
            "SMI_len": len(smi),
            "SMI_branches": smi.count("("),
            "SMI_ringDigits": sum(1 for ch in smi if ch.isdigit()),
        }

    def gasteiger_stats(self, m):
        try:
            m_copy = Chem.AddHs(m)
            ComputeGasteigerCharges(m_copy)
            vals = []
            for a in m_copy.GetAtoms():
                v = a.GetDoubleProp('_GasteigerCharge') if a.HasProp('_GasteigerCharge') else 0.0
                if pd.isna(v) or v == float('inf') or v == float('-inf'):
                    v = 0.0
                vals.append(v)
            arr = np.array(vals, dtype=np.float32)
            if len(arr) == 0:
                return {"Gast_sum": 0.0, "Gast_abs": 0.0, "Gast_std": 0.0}
            return {
                "Gast_sum": float(arr.sum()),
                "Gast_abs": float(np.abs(arr).sum()),
                "Gast_std": float(arr.std()),
            }
        except Exception:
            return {"Gast_sum": 0.0, "Gast_abs": 0.0, "Gast_std": 0.0}

    def _generate_row_features(self, m, smi: str) -> Dict[str, Any]:
        """Generate features for a single molecule."""
        row = {}
        
        if m is None:
            return row

        # 1. RDKit Descriptors (all ~200)
        for name, func in self.descriptor_list:
            row[name] = self._safe(func)(m)

        # 2. Fragment Counts
        for attr in dir(Fragments):
            if attr.startswith("fr_"):
                fn = getattr(Fragments, attr)
                if callable(fn):
                    row[attr] = self._safe(fn)(m)

        # 3. Morgan Fingerprints (reduced to 256 bits)
        try:
            mgen = rdFingerprintGenerator.GetMorganGenerator(radius=MORGAN_RADIUS, fpSize=MORGAN_BITS)
            mfp = mgen.GetFingerprint(m)
            for i in range(MORGAN_BITS):
                row[f"Morg_{i}"] = int(mfp[i])
        except Exception:
            for i in range(MORGAN_BITS):
                row[f"Morg_{i}"] = 0

        # 4. MACCS Keys (167 bits)
        try:
            maccs = GenMACCSKeys(m)
            for i in range(len(maccs)):
                row[f"MACCS_{i}"] = int(maccs[i])
        except Exception:
            for i in range(167):
                row[f"MACCS_{i}"] = 0

        # 5. VSA Binnings
        for vsa_name, vsa_fn in [
            ("SlogP_VSA", getattr(rdMolDescriptors, "SlogP_VSA_", None)),
            ("SMR_VSA", getattr(rdMolDescriptors, "SMR_VSA_", None)),
        ]:
            if vsa_fn:
                try:
                    bins = vsa_fn(m)
                    for i, val in enumerate(bins):
                        row[f"{vsa_name}{i}"] = val
                except Exception:
                    pass

        # 6. Gasteiger
        row.update(self.gasteiger_stats(m))

        # 7. Murcko
        row.update(self._murcko_stats(m))

        # 8. SMILES Morphology
        row.update(self._smiles_morphology(smi))

        # 9. Bond Fractions
        bonds = list(m.GetBonds())
        nb = max(len(bonds), 1)
        n_arom = sum(1 for b in bonds if b.GetIsAromatic())
        row["FracArom"] = n_arom / nb

        # 10. Ring Stats
        hist, n_rings = self._ring_size_hist(m)
        row["Rings5"] = hist[5]
        row["Rings6"] = hist[6]
        row["RingSystems"] = self._ring_systems_count(m)

        # 11. Composite Features
        mw = row.get("MolWt", 1.0) or 1.0
        tpsa = row.get("TPSA", 0.0) or 0.0
        mollogp = row.get("MolLogP", 0.0) or 0.0
        nrot = row.get("NumRotatableBonds", 0.0) or 0.0
        narm = row.get("NumAromaticRings", 0.0) or 0.0

        row["LogP_div_TPSA"] = mollogp / (tpsa + 1.0)
        row["Flex_Score"] = nrot / (mw + 1.0)
        row["Rigid_Score"] = narm / (nrot + 1.0)

        return row

    def _reduce_memory(self, df: pd.DataFrame) -> pd.DataFrame:
        """Reduce memory usage by converting to efficient dtypes."""
        for col in df.columns:
            col_type = df[col].dtype
            if col_type == np.float64:
                df[col] = df[col].astype(np.float32)
            elif col_type == np.int64:
                c_min, c_max = df[col].min(), df[col].max()
                if c_min >= 0 and c_max <= 1:
                    df[col] = df[col].astype(np.int8)
                elif c_min >= -128 and c_max <= 127:
                    df[col] = df[col].astype(np.int8)
                elif c_min >= -32768 and c_max <= 32767:
                    df[col] = df[col].astype(np.int16)
                else:
                    df[col] = df[col].astype(np.int32)
        return df

    def generate_features(self, df: pd.DataFrame, smiles_col: str = 'SMILES', save_path: Optional[str] = None) -> pd.DataFrame:
        """
        Generate features in batches to manage memory.
        
        Args:
            df: DataFrame with SMILES column.
            smiles_col: Name of SMILES column.
            save_path: Optional path to save intermediate results.
            
        Returns:
            DataFrame with features.
        """
        all_feats = []
        n_batches = (len(df) + self.batch_size - 1) // self.batch_size
        
        for batch_idx in tqdm(range(n_batches), desc="Processing batches"):
            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, len(df))
            batch_df = df.iloc[start_idx:end_idx]
            
            batch_feats = []
            for _, row_data in batch_df.iterrows():
                smi = row_data[smiles_col]
                if pd.isna(smi):
                    batch_feats.append({})
                    continue
                m = Chem.MolFromSmiles(smi)
                batch_feats.append(self._generate_row_features(m, smi))
            
            batch_feats_df = pd.DataFrame(batch_feats)
            batch_feats_df = batch_feats_df.fillna(0)
            batch_feats_df = self._reduce_memory(batch_feats_df)
            
            # Combine with original batch data
            batch_result = pd.concat([batch_df.reset_index(drop=True), batch_feats_df.reset_index(drop=True)], axis=1)
            all_feats.append(batch_result)
            
            # Force garbage collection
            del batch_feats, batch_feats_df, batch_result
            gc.collect()
            
            # Optional: save intermediate results
            if save_path and (batch_idx + 1) % 10 == 0:
                temp_df = pd.concat(all_feats, ignore_index=True)
                temp_df.to_parquet(save_path, index=False)
                print(f"Saved checkpoint at batch {batch_idx + 1}")
        
        result = pd.concat(all_feats, ignore_index=True)
        
        # Final memory reduction
        result = self._reduce_memory(result)
        
        # Drop constant columns
        nunique = result.nunique(dropna=False)
        constant_cols = nunique[nunique <= 1].index.tolist()
        result = result.drop(columns=constant_cols, errors='ignore')
        
        print(f"Final shape: {result.shape}, Memory: {result.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        
        return result
