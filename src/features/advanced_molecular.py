import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, Crippen, Lipinski, Fragments, AllChem
from rdkit.Chem.AllChem import ComputeGasteigerCharges
from rdkit.Chem.MACCSkeys import GenMACCSKeys
from rdkit.Chem.EState import AtomTypes as EAtomTypes
from rdkit.Chem.Scaffolds import MurckoScaffold
from typing import Dict, Any, List
from tqdm import tqdm

class AdvancedMolecularFeaturizer:
    def __init__(self):
        pass

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

    def _count_explicit_h(self, m):
        mH = Chem.AddHs(m)
        return sum(1 for a in mH.GetAtoms() if a.GetSymbol() == "H")

    def _bond_order(self, b):
        if b.GetIsAromatic():
            return 1.5
        t = b.GetBondType()
        if t == Chem.BondType.SINGLE: return 1
        if t == Chem.BondType.DOUBLE: return 2
        if t == Chem.BondType.TRIPLE: return 3
        return 0

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
        sys = 0
        for i in range(len(rings)):
            if i in seen: continue
            sys += 1
            stack = [i]
            seen.add(i)
            while stack:
                j = stack.pop()
                for k in range(len(rings)):
                    if k in seen: continue
                    if rings[j] & rings[k]:
                        seen.add(k); stack.append(k)
        return sys

    def _murcko_stats(self, m):
        try:
            scaf = MurckoScaffold.GetScaffoldForMol(m)
            if scaf is None or scaf.GetNumAtoms() == 0:
                return {"MurckoAtoms":0, "MurckoRings":0, "MurckoRingSystems":0, "SideChainAtoms":m.GetNumAtoms()}
            msys = self._ring_systems_count(scaf)
            return {
                "MurckoAtoms": scaf.GetNumAtoms(),
                "MurckoRings": rdMolDescriptors.CalcNumRings(scaf),
                "MurckoRingSystems": msys,
                "SideChainAtoms": m.GetNumAtoms() - scaf.GetNumAtoms(),
            }
        except Exception:
            return {"MurckoAtoms":0, "MurckoRings":0, "MurckoRingSystems":0, "SideChainAtoms":m.GetNumAtoms()}

    def _estate_stats(self, m):
        try:
            vals = EAtomTypes.EStateIndices(m)
            if not vals: return {"EState_sum":0.0,"EState_mean":0.0,"EState_max":0.0,"EState_min":0.0,"EState_std":0.0}
            arr = np.asarray(vals, dtype=float)
            return {
                "EState_sum": float(arr.sum()),
                "EState_mean": float(arr.mean()),
                "EState_max": float(arr.max()),
                "EState_min": float(arr.min()),
                "EState_std": float(arr.std(ddof=0)),
            }
        except Exception:
            return {"EState_sum":0.0,"EState_mean":0.0,"EState_max":0.0,"EState_min":0.0,"EState_std":0.0}

    def _smiles_morphology(self, smi: str):
        if not smi: 
            return {"SMI_len":0, "SMI_branches":0, "SMI_ringDigits":0, "SMI_stereoAt":0, "SMI_ezSlashes":0}
        return {
            "SMI_len": len(smi),
            "SMI_branches": smi.count("("),
            "SMI_ringDigits": sum(1 for ch in smi if ch.isdigit()),
            "SMI_stereoAt": smi.count("@"),
            "SMI_ezSlashes": smi.count("/") + smi.count("\\"),
        }

    def gasteiger_stats(self, m):
        try:
            m = Chem.AddHs(m)
            ComputeGasteigerCharges(m)
            vals = []
            for a in m.GetAtoms():
                v = a.GetDoubleProp('_GasteigerCharge') if a.HasProp('_GasteigerCharge') else 0.0
                if pd.isna(v) or v == float('inf') or v == float('-inf'):
                    v = 0.0
                vals.append(v)
            arr = np.array(vals, dtype=float)
            if len(arr) == 0:
                 return {
                    "Gasteiger_q_sum": 0.0,
                    "Gasteiger_q_abs_sum": 0.0,
                    "Gasteiger_q_min": 0.0,
                    "Gasteiger_q_max": 0.0,
                    "Gasteiger_q_std": 0.0,
                }
            return {
                "Gasteiger_q_sum": float(arr.sum()),
                "Gasteiger_q_abs_sum": float(np.abs(arr).sum()),
                "Gasteiger_q_min": float(arr.min()),
                "Gasteiger_q_max": float(arr.max()),
                "Gasteiger_q_std": float(arr.std(ddof=0)),
            }
        except Exception:
            return {
                "Gasteiger_q_sum": 0.0,
                "Gasteiger_q_abs_sum": 0.0,
                "Gasteiger_q_min": 0.0,
                "Gasteiger_q_max": 0.0,
                "Gasteiger_q_std": 0.0,
            }

    def augment_extra_cheaps(self, row, m):
        row.update(self._estate_stats(m))

        bonds = list(m.GetBonds())
        nb = max(len(bonds), 1)
        n_single = sum(1 for b in bonds if b.GetBondType() == Chem.BondType.SINGLE and not b.GetIsAromatic())
        n_double = sum(1 for b in bonds if b.GetBondType() == Chem.BondType.DOUBLE)
        n_triple = sum(1 for b in bonds if b.GetBondType() == Chem.BondType.TRIPLE)
        n_arom   = sum(1 for b in bonds if b.GetIsAromatic())
        row["FracSingle"]  = n_single / nb
        row["FracDouble"]  = n_double / nb
        row["FracTriple"]  = n_triple / nb
        row["FracAromatic"]= n_arom   / nb
        row["MeanBondOrder"]= (sum(self._bond_order(b) for b in bonds) / nb) if nb>0 else 0.0
        row["UnsatBondCount"] = n_double + n_triple + n_arom

        hist, n_rings = self._ring_size_hist(m)
        row["Rings5"] = hist[5]; row["Rings6"] = hist[6]
        row["Rings7"] = hist[7]; row["Rings8"] = hist[8]
        row["RingSystems"] = self._ring_systems_count(m)
        row["Rings56_frac"] = (hist[5] + hist[6]) / (n_rings if n_rings>0 else 1)

        row.update(self._murcko_stats(m))

        tot_charge = sum(a.GetFormalCharge() for a in m.GetAtoms())
        has_pos = any(a.GetFormalCharge() > 0 for a in m.GetAtoms())
        has_neg = any(a.GetFormalCharge() < 0 for a in m.GetAtoms())
        row["FormalCharge"] = int(tot_charge)
        row["IsZwitterion"] = int(has_pos and has_neg)

        try:
            smi = Chem.MolToSmiles(m, canonical=True)
        except Exception:
            smi = ""
        row.update(self._smiles_morphology(smi))
        return row

    def generate_features(self, df: pd.DataFrame, smiles_col: str = 'SMILES') -> pd.DataFrame:
        mols = df[smiles_col].apply(lambda s: Chem.MolFromSmiles(s) if pd.notna(s) else None)
        
        feats = []
        for m in tqdm(mols, desc="Advanced Featurization"):
            row = {}
            if m is None:
                feats.append(row)
                continue

            # Standard Descriptors
            row["MolLogP"] = self._safe(Crippen.MolLogP)(m)
            row["MolMR"] = self._safe(Crippen.MolMR)(m)
            row["TPSA"] = self._safe(rdMolDescriptors.CalcTPSA)(m)
            row["MolWt"] = self._safe(Descriptors.MolWt)(m)
            row["NumRotatableBonds"] = self._safe(rdMolDescriptors.CalcNumRotatableBonds)(m)
            row["NumAromaticRings"] = self._safe(rdMolDescriptors.CalcNumAromaticRings)(m)
            row["NumRings"] = self._safe(rdMolDescriptors.CalcNumRings)(m)
            row["NumHAcceptors"] = self._safe(Lipinski.NumHAcceptors)(m)
            row["NumHDonors"] = self._safe(Lipinski.NumHDonors)(m)
            row["BertzCT"] = self._safe(Descriptors.BertzCT)(m)
            
            # Gasteiger
            row.update(self.gasteiger_stats(m))

            # Composite Features from Reference
            tpsa = row.get("TPSA", 0.0) or 0.0
            mollogp = row.get("MolLogP", 0.0) or 0.0
            mw = row.get("MolWt", 0.0) or 1.0
            nrot = row.get("NumRotatableBonds", 0.0) or 0.0
            narm = row.get("NumAromaticRings", 0.0) or 0.0
            bertz = row.get("BertzCT", 0.0) or 0.0

            row["LogP_div_TPSA"] = mollogp / (tpsa + 1.0)
            row["LogP_x_TPSA"] = mollogp * tpsa
            row["Flexibility_Score"] = nrot / (mw + 1.0)
            row["MolWt_x_AromaticRings"] = mw * narm
            row["Complexity_per_MW"] = bertz / (mw + 1.0)
            row["Rigidity_Score"] = narm / (nrot + 1.0)

            # Augment Extra Cheaps (Morphology, Estates, etc)
            row = self.augment_extra_cheaps(row, m)
            
            feats.append(row)

        feats_df = pd.DataFrame(feats)
        feats_df = feats_df.fillna(0)
        
        # Combine
        return pd.concat([df.reset_index(drop=True), feats_df.reset_index(drop=True)], axis=1)
