"""
HierarchicalMP Predictor v8.0 - Calibrated Neighborhood Uncertainty (CNU)

Theoretical Framework:
- First-principles uncertainty from neighborhood geometry
- Learned nonnegative weights via NNLS
- Percentile-based regime calibration
- Risk-aware routing with formal coverage guarantees

Based on v7.0 with CNU integration.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional, Union
from dataclasses import dataclass
from pathlib import Path
import json
import time
import warnings
import hashlib

warnings.filterwarnings('ignore')

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, Descriptors, Crippen
from rdkit.Chem import rdMolDescriptors

try:
    from lightgbm import LGBMRegressor
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False

# Import CNU components
from ..calibration import (
    NeighborhoodUncertainty,
    NeighborhoodFeatures,
    CNUCalibrator,
    CalibrationResult,
    learn_weights
)

# Import v7 utilities
from .hierarchical_mp_v7 import (
    fp_to_uint64_blocks,
    popcount_u64,
    fast_tanimoto_u64,
    tanimoto_single_u64,
    ResidualFallbackModel,
    HAS_BIT_COUNT,
    _POPCOUNT_TABLE
)


# ============================================================================
# PREDICTION RESULT (extended for CNU)
# ============================================================================

@dataclass
class PredictionResultV8:
    smiles: str
    tm_pred: float
    tm_low: float
    tm_high: float
    method: str
    confidence: float
    regime: str  # CNU regime assignment
    u_score: float  # Uncertainty score
    top_similarity: float = 0.0
    interval_width: float = 0.0
    from_cache: bool = False


# ============================================================================
# HIERARCHICAL MP v8.0 (CNU-enabled)
# ============================================================================

class HierarchicalMPPredictorV8:
    """
    HierarchicalMP v8.0 - Calibrated Neighborhood Uncertainty
    
    Theoretical contributions:
    1. First-principles uncertainty decomposition (epistemic/aleatoric/ambiguity)
    2. Learned uncertainty functional u(x) from neighborhood geometry
    3. Percentile-based regime calibration with finite-sample guarantees
    4. Risk-aware routing based on calibrated quantiles
    """
    
    MAX_SMILES_LENGTH = 5000
    
    def __init__(self,
                 fp_radius: int = 2,
                 fp_bits: int = 2048,
                 n_neighbors: int = 50,
                 top_k: int = 10,
                 nprobe: int = 32,
                 alpha: float = 0.1,
                 n_regimes: int = 5):
        
        self.fp_radius = fp_radius
        self.fp_bits = fp_bits
        self.n_neighbors = n_neighbors
        self.top_k = top_k
        self.nprobe = nprobe
        self.alpha = alpha
        self.n_regimes = n_regimes
        
        # Exact SMILES lookup
        self.exact_lookup: Dict[str, float] = {}
        
        # Retrieval index
        self.smiles_list: List[str] = []
        self.canonical_list: List[str] = []
        self.tm_values: np.ndarray = None
        self.fps_u64: np.ndarray = None
        self.faiss_index = None
        
        # CNU components
        self.uncertainty = NeighborhoodUncertainty()
        self.cnu_calibrator = CNUCalibrator(
            alpha=alpha, 
            n_regimes=n_regimes
        )
        
        # Fallback model
        self.fallback = ResidualFallbackModel(sim_threshold=0.7)
        
        # Stats
        self.train_mean = 300.0
        self.train_std = 50.0
        
        # State flags
        self.index_fitted = False
        self.calibration_fitted = False
        
        # Provenance
        self.provenance = {}
    
    def _canonicalize(self, smiles: str) -> Optional[str]:
        if len(smiles) > self.MAX_SMILES_LENGTH:
            return None
        try:
            mol = Chem.MolFromSmiles(smiles)
            return Chem.MolToSmiles(mol, canonical=True) if mol else None
        except:
            return None
    
    def _mol_to_u64_fp(self, mol) -> np.ndarray:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, self.fp_radius, nBits=self.fp_bits)
        return fp_to_uint64_blocks(fp)
    
    def _mol_to_numpy_fp(self, mol) -> np.ndarray:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, self.fp_radius, nBits=self.fp_bits)
        arr = np.zeros(self.fp_bits, dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr
    
    # ========================================================================
    # FIT INDEX
    # ========================================================================
    
    def fit_index(self, smiles_list: List[str], tm_values: np.ndarray):
        """Build retrieval index from training data."""
        print(f"fit_index: Building index for {len(smiles_list)} molecules...")
        start = time.time()
        
        # Deduplicate and canonicalize
        canonical_map = {}
        for smi, tm in zip(smiles_list, tm_values):
            can = self._canonicalize(smi)
            if can and can not in canonical_map:
                canonical_map[can] = (tm, smi)
        
        self.exact_lookup = {can: tm for can, (tm, _) in canonical_map.items()}
        
        # Build indexed lists
        valid_smiles, valid_canonical, valid_tms = [], [], []
        u64_fps, numpy_fps = [], []
        
        for can, (tm, orig) in canonical_map.items():
            mol = Chem.MolFromSmiles(can)
            if mol is None:
                continue
            valid_smiles.append(orig)
            valid_canonical.append(can)
            valid_tms.append(tm)
            u64_fps.append(self._mol_to_u64_fp(mol))
            numpy_fps.append(self._mol_to_numpy_fp(mol))
        
        self.smiles_list = valid_smiles
        self.canonical_list = valid_canonical
        self.tm_values = np.array(valid_tms, dtype=np.float32)
        self.fps_u64 = np.vstack(u64_fps)
        numpy_fps = np.vstack(numpy_fps).astype(np.float32)
        
        self.train_mean = float(np.mean(self.tm_values))
        self.train_std = float(np.std(self.tm_values))
        
        # Build FAISS index
        faiss.normalize_L2(numpy_fps)
        n = len(numpy_fps)
        n_clusters = min(100, n // 10)
        
        if n_clusters < 2:
            self.faiss_index = faiss.IndexFlatIP(self.fp_bits)
            self.faiss_index.add(numpy_fps)
        else:
            quantizer = faiss.IndexFlatIP(self.fp_bits)
            self.faiss_index = faiss.IndexIVFFlat(
                quantizer, self.fp_bits, n_clusters, faiss.METRIC_INNER_PRODUCT
            )
            self.faiss_index.train(numpy_fps)
            self.faiss_index.add(numpy_fps)
            self.faiss_index.nprobe = self.nprobe
        
        self.index_fitted = True
        elapsed = time.time() - start
        
        self.provenance = {
            'version': '8.0',
            'n_molecules': len(self.smiles_list),
            'n_exact_lookup': len(self.exact_lookup),
            'train_mean': self.train_mean,
            'build_time_s': elapsed,
        }
        
        print(f"  Index built: {len(self.smiles_list)} molecules in {elapsed:.1f}s")
        print(f"  Exact lookup entries: {len(self.exact_lookup)}")
    
    # ========================================================================
    # FIT CNU CALIBRATION
    # ========================================================================
    
    def fit_calibration(self, calib_smiles: List[str], calib_tm: np.ndarray) -> CalibrationResult:
        """
        Calibrate CNU using held-out calibration set.
        
        This computes:
        1. Neighborhood features for each calibration point
        2. Learned nonnegative weights for u(x)
        3. Percentile-based regime boundaries
        4. Per-regime conformal quantiles
        """
        if not self.index_fitted:
            raise RuntimeError("Call fit_index() first")
        
        print(f"fit_calibration: Calibrating CNU on {len(calib_smiles)} molecules...")
        start = time.time()
        
        # Run pipeline to get predictions and features
        predictions = []
        features_list = []
        neighbor_values_list = []
        
        for smi in calib_smiles:
            pred, feat, neighbor_vals = self._predict_with_features(smi)
            predictions.append(pred)
            features_list.append(feat)
            neighbor_values_list.append(neighbor_vals)
        
        predictions = np.array(predictions)
        actuals = np.array(calib_tm)
        
        # Fit CNU calibrator
        result = self.cnu_calibrator.fit(
            predictions, actuals, features_list, neighbor_values_list
        )
        
        # Update uncertainty weights
        self.uncertainty.set_weights(result.weights)
        
        self.calibration_fitted = True
        elapsed = time.time() - start
        
        # Report
        print(f"  Calibration time: {elapsed:.1f}s")
        print(f"  Learned weights: {result.weights}")
        print(f"  Regime quantiles:")
        for regime, q in result.regime_quantiles.items():
            n = result.regime_counts.get(regime, 0)
            cov = result.coverage_achieved.get(regime, 0)
            print(f"    {regime}: ±{q:.1f}K (n={n}, cov={cov:.1%})")
        
        return result
    
    # ========================================================================
    # PREDICT WITH FEATURES (internal)
    # ========================================================================
    
    def _predict_with_features(self, smiles: str) -> Tuple[float, NeighborhoodFeatures, np.ndarray]:
        """Get prediction + neighborhood features for calibration."""
        
        # Default features for invalid inputs
        default_feat = NeighborhoodFeatures(
            s1=0.0, delta_s=0.0, sigma_w=50.0,
            k_eff=1.0, ambiguity=np.log(101),
            z=np.array([1.0, 50.0, 1.0, np.log(101)])
        )
        empty_neighbors = np.array([self.train_mean])
        
        can = self._canonicalize(smiles)
        if can is None:
            return self.train_mean, default_feat, empty_neighbors
        
        # Exact lookup
        if can in self.exact_lookup:
            tm = self.exact_lookup[can]
            feat = NeighborhoodFeatures(
                s1=1.0, delta_s=1.0, sigma_w=0.0,
                k_eff=1.0, ambiguity=0.0,
                z=np.array([0.0, 0.0, 1.0, 0.0])
            )
            return tm, feat, np.array([tm])
        
        # FAISS retrieval
        mol = Chem.MolFromSmiles(can)
        if mol is None:
            return self.train_mean, default_feat, empty_neighbors
        
        query_numpy = self._mol_to_numpy_fp(mol).reshape(1, -1)
        faiss.normalize_L2(query_numpy)
        query_u64 = self._mol_to_u64_fp(mol)
        
        _, indices = self.faiss_index.search(query_numpy, self.n_neighbors)
        indices = indices[0]
        valid_idx = indices[(indices >= 0) & (indices < len(self.fps_u64))]
        
        if len(valid_idx) == 0:
            return self.train_mean, default_feat, empty_neighbors
        
        # Rerank with true Tanimoto
        sims = fast_tanimoto_u64(query_u64, self.fps_u64[valid_idx])
        order = np.argsort(sims)[::-1][:self.top_k]
        reranked_idx = valid_idx[order]
        reranked_sims = sims[order]
        neighbor_vals = self.tm_values[reranked_idx]
        
        # Compute neighborhood features
        feat = self.uncertainty.compute_features(reranked_sims, neighbor_vals)
        
        # Prediction based on similarity
        top_sim = feat.s1
        if top_sim >= 0.95:
            tm_pred = float(neighbor_vals[0])
        elif top_sim >= 0.7:
            weights = reranked_sims ** 2
            tm_pred = float(np.sum(neighbor_vals * weights) / np.sum(weights))
        else:
            neighbor_mean = float(np.mean(neighbor_vals[:5]))
            tm_pred = self.fallback.predict(smiles, neighbor_mean)
        
        return tm_pred, feat, neighbor_vals
    
    # ========================================================================
    # PREDICT (inference)
    # ========================================================================
    
    def predict(self, smiles: str) -> PredictionResultV8:
        """Full prediction with CNU-calibrated intervals."""
        if not self.index_fitted:
            raise RuntimeError("Call fit_index() first")
        
        tm_pred, feat, _ = self._predict_with_features(smiles)
        u_score = self.uncertainty.score(feat)
        
        # Determine method
        if feat.s1 >= 0.999:
            method = 'exact_smiles'
        elif feat.s1 >= 0.95:
            method = 'near_exact'
        elif feat.s1 >= 0.7:
            method = 'retrieval'
        else:
            method = 'fallback'
        
        # Get calibrated interval
        if self.calibration_fitted:
            tm_low, tm_high, regime = self.cnu_calibrator.get_interval(feat, tm_pred)
        else:
            tm_low, tm_high, regime = tm_pred - 50, tm_pred + 50, 'uncalibrated'
        
        return PredictionResultV8(
            smiles=smiles,
            tm_pred=tm_pred,
            tm_low=tm_low,
            tm_high=tm_high,
            method=method,
            confidence=feat.s1,
            regime=regime,
            u_score=u_score,
            top_similarity=feat.s1,
            interval_width=tm_high - tm_low,
            from_cache=(method == 'exact_smiles')
        )
    
    def predict_batch(self, smiles_list: List[str]) -> pd.DataFrame:
        """Batch prediction."""
        results = [self.predict(s) for s in smiles_list]
        return pd.DataFrame([{
            'SMILES': r.smiles,
            'Tm_pred': r.tm_pred,
            'Tm_low': r.tm_low,
            'Tm_high': r.tm_high,
            'method': r.method,
            'confidence': r.confidence,
            'regime': r.regime,
            'u_score': r.u_score,
            'top_similarity': r.top_similarity,
            'interval_width': r.interval_width,
            'from_cache': r.from_cache,
        } for r in results])
    
    # ========================================================================
    # ABLATION: Compute contribution of each uncertainty primitive
    # ========================================================================
    
    def compute_ablation(self, calib_smiles: List[str], calib_tm: np.ndarray) -> pd.DataFrame:
        """
        Ablation study: contribution of each uncertainty primitive.
        
        Returns DataFrame with coverage/width for each weight configuration.
        """
        if not self.index_fitted:
            raise RuntimeError("Call fit_index() first")
        
        # Get features and predictions
        predictions, features_list = [], []
        for smi in calib_smiles:
            pred, feat, _ = self._predict_with_features(smi)
            predictions.append(pred)
            features_list.append(feat)
        
        predictions = np.array(predictions)
        actuals = np.array(calib_tm)
        residuals = np.abs(actuals - predictions)
        
        # Weight configurations for ablation
        configs = {
            'full': [1, 1, 1, 1],
            'no_sigma': [1, 0, 1, 1],
            'no_ambiguity': [1, 1, 1, 0],
            'no_k_eff': [1, 1, 0, 1],
            'only_s1': [1, 0, 0, 0],
        }
        
        results = []
        for name, weights in configs.items():
            # Learn weights constrained to this config
            mask = np.array(weights) > 0
            Z = np.array([f.z for f in features_list])[:, mask]
            from scipy.optimize import nnls
            w_sub, _ = nnls(Z, residuals)
            
            # Compute u(x) with this config
            w_full = np.zeros(4)
            w_full[mask] = w_sub
            u_scores = np.array([np.dot(w_full, f.z) for f in features_list])
            
            # Calibrate with percentiles
            boundaries = np.percentile(u_scores, np.linspace(0, 100, 6))
            regime_quantiles = {}
            for r in range(5):
                mask_r = (u_scores >= boundaries[r]) & (u_scores < boundaries[r+1])
                if mask_r.sum() >= 10:
                    q = np.percentile(residuals[mask_r], 90)
                    regime_quantiles[r] = q
            
            # Compute mean width and coverage
            widths = []
            covered = 0
            for i, u in enumerate(u_scores):
                r = np.digitize([u], boundaries[1:-1])[0]
                r = min(max(r, 0), 4)
                q = regime_quantiles.get(r, np.percentile(residuals, 90))
                widths.append(2 * q)
                if residuals[i] <= q:
                    covered += 1
            
            results.append({
                'config': name,
                'mean_width': np.mean(widths),
                'coverage': covered / len(residuals),
                'weights': w_full.tolist()
            })
        
        return pd.DataFrame(results)
    
    # ========================================================================
    # MONOTONICITY VALIDATION
    # ========================================================================
    
    def validate_monotonicity(self, calib_smiles: List[str], calib_tm: np.ndarray) -> pd.DataFrame:
        """
        Validate that u(x) ranks risk correctly (monotonicity check).
        
        Returns DataFrame with MAE by u(x) decile.
        """
        predictions, features_list = [], []
        for smi in calib_smiles:
            pred, feat, _ = self._predict_with_features(smi)
            predictions.append(pred)
            features_list.append(feat)
        
        predictions = np.array(predictions)
        actuals = np.array(calib_tm)
        residuals = np.abs(actuals - predictions)
        u_scores = np.array([self.uncertainty.score(f) for f in features_list])
        
        # Compute MAE by decile
        decile_edges = np.percentile(u_scores, np.arange(0, 101, 10))
        results = []
        
        for i in range(10):
            mask = (u_scores >= decile_edges[i]) & (u_scores < decile_edges[i+1])
            if i == 9:
                mask = u_scores >= decile_edges[i]
            
            if mask.sum() > 0:
                results.append({
                    'decile': i + 1,
                    'u_range': f"[{decile_edges[i]:.2f}, {decile_edges[i+1]:.2f})",
                    'n': int(mask.sum()),
                    'mae': float(np.mean(residuals[mask])),
                    'median_error': float(np.median(residuals[mask]))
                })
        
        return pd.DataFrame(results)
    
    # ========================================================================
    # PERSISTENCE
    # ========================================================================
    
    def save(self, path: Union[str, Path]):
        """Save predictor with minimal state."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        faiss.write_index(self.faiss_index, str(path / 'faiss.index'))
        np.save(path / 'tm_values.npy', self.tm_values)
        np.save(path / 'fps_u64.npy', self.fps_u64)
        
        with open(path / 'smiles.json', 'w') as f:
            json.dump(self.smiles_list, f)
        with open(path / 'exact_lookup.json', 'w') as f:
            json.dump(self.exact_lookup, f)
        
        # CNU state
        cnu_state = {
            'weights': self.uncertainty.weights.tolist(),
            'global_sigma': self.uncertainty.global_sigma_fallback,
            'regime_quantiles': self.cnu_calibrator.regime_quantiles,
            'boundaries': self.cnu_calibrator.boundaries.tolist() if self.cnu_calibrator.boundaries is not None else None,
            'global_quantile': self.cnu_calibrator.global_quantile,
            'n_regimes': self.n_regimes,
            'alpha': self.alpha,
        }
        with open(path / 'cnu_state.json', 'w') as f:
            json.dump(cnu_state, f, indent=2)
        
        config = {
            'version': '8.0',
            'fp_radius': self.fp_radius,
            'fp_bits': self.fp_bits,
            'n_neighbors': self.n_neighbors,
            'train_mean': self.train_mean,
            'train_std': self.train_std,
            'provenance': self.provenance,
        }
        with open(path / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Saved CNU v8.0 to {path}")
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'HierarchicalMPPredictorV8':
        """Load predictor from saved state."""
        path = Path(path)
        
        with open(path / 'config.json') as f:
            config = json.load(f)
        
        with open(path / 'cnu_state.json') as f:
            cnu_state = json.load(f)
        
        predictor = cls(
            fp_radius=config['fp_radius'],
            fp_bits=config['fp_bits'],
            n_neighbors=config['n_neighbors'],
            alpha=cnu_state['alpha'],
            n_regimes=cnu_state['n_regimes'],
        )
        
        predictor.faiss_index = faiss.read_index(str(path / 'faiss.index'))
        predictor.tm_values = np.load(path / 'tm_values.npy')
        predictor.fps_u64 = np.load(path / 'fps_u64.npy')
        
        with open(path / 'smiles.json') as f:
            predictor.smiles_list = json.load(f)
        with open(path / 'exact_lookup.json') as f:
            predictor.exact_lookup = json.load(f)
        
        # Restore CNU state
        predictor.uncertainty.set_weights(np.array(cnu_state['weights']))
        predictor.uncertainty.set_global_sigma(cnu_state['global_sigma'])
        predictor.cnu_calibrator.regime_quantiles = cnu_state['regime_quantiles']
        predictor.cnu_calibrator.boundaries = np.array(cnu_state['boundaries']) if cnu_state['boundaries'] else None
        predictor.cnu_calibrator.global_quantile = cnu_state['global_quantile']
        predictor.cnu_calibrator.is_fitted = True
        
        predictor.train_mean = config['train_mean']
        predictor.train_std = config['train_std']
        predictor.provenance = config.get('provenance', {})
        predictor.index_fitted = True
        predictor.calibration_fitted = True
        
        return predictor


if __name__ == "__main__":
    print("HierarchicalMP v8.0 (CNU) Demo")
    print("=" * 50)
    
    # Demo would require full data - see generate_v8_submission.py
    print("Run scripts/generate_v8_submission.py for full demo")
