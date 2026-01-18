"""
Calibrated Neighborhood Uncertainty (CNU) - Uncertainty Functional

First-principles uncertainty computation from retrieval neighborhood geometry.

Uncertainty Sources (Decomposition):
- Epistemic: Coverage/density around query (1-s₁, 1/k_eff)
- Aleatoric: Inherent noise/polymorphs (σ_w)
- Ambiguity: Identifiability of nearest neighbor (1/Δs)

Reference: CNU Framework Implementation Plan
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class NeighborhoodFeatures:
    """Retrieval geometry features for uncertainty computation."""
    s1: float            # Top similarity
    delta_s: float       # Similarity gap (s1 - s2)
    sigma_w: float       # Weighted neighbor variance
    k_eff: float         # Effective sample size
    ambiguity: float     # Log-scaled ambiguity term
    z: np.ndarray        # Feature vector for u(x) computation


class NeighborhoodUncertainty:
    """
    First-principles uncertainty functional from retrieval geometry.
    
    u(x) = w₁(1-s₁) + w₂σ_w + w₃/k_eff + w₄·ambiguity
    
    where weights w ≥ 0 are learned via nonnegative least squares.
    """
    
    EPS = 1e-6
    MIN_VALID_SIMILARITY = 0.5  # For σ_w computation
    
    def __init__(self, 
                 weights: Optional[np.ndarray] = None,
                 global_sigma_fallback: float = None):
        """
        Initialize uncertainty functional.
        
        Args:
            weights: Nonnegative weights [w1, w2, w3, w4]. If None, uniform.
            global_sigma_fallback: Default σ_w when insufficient neighbors.
                                   If None, will be set from calibration data.
        """
        self.weights = weights if weights is not None else np.ones(4)
        self.global_sigma_fallback = global_sigma_fallback
        
    def compute_features(self, 
                         similarities: np.ndarray, 
                         values: np.ndarray) -> NeighborhoodFeatures:
        """
        Compute neighborhood geometry features.
        
        Args:
            similarities: Sorted (descending) Tanimoto similarities to k neighbors
            values: Corresponding property values (Tm) of neighbors
            
        Returns:
            NeighborhoodFeatures with all primitives and z vector
        """
        if len(similarities) == 0:
            return self._fallback_features()
        
        # Top similarity
        s1 = float(similarities[0])
        
        # Similarity gap (ambiguity signal)
        delta_s = float(similarities[0] - similarities[1]) if len(similarities) > 1 else 1.0
        
        # Compute weighted statistics with numerical safety
        weights = similarities ** 2
        weight_sum = weights.sum() + self.EPS
        weights_norm = weights / weight_sum
        
        # σ_w: Use only "valid" neighbors (similarity >= threshold)
        valid_mask = similarities >= self.MIN_VALID_SIMILARITY
        if valid_mask.sum() >= 2:
            valid_weights = weights[valid_mask]
            valid_values = values[valid_mask]
            valid_weights_norm = valid_weights / (valid_weights.sum() + self.EPS)
            mu = np.dot(valid_weights_norm, valid_values)
            sigma_w = float(np.sqrt(np.dot(valid_weights_norm, (valid_values - mu) ** 2)))
        else:
            # Fallback: use data-derived global sigma
            sigma_w = self.global_sigma_fallback if self.global_sigma_fallback else 30.0
        
        # Effective sample size
        k_eff = float(1.0 / (np.sum(weights_norm ** 2) + self.EPS))
        
        # Log-scaled ambiguity (stable, bounded)
        ambiguity = float(np.log(1 + 1.0 / (delta_s + self.EPS)))
        
        # Feature vector z for u(x) computation
        z = np.array([
            1 - s1,           # Epistemic: distance to nearest
            sigma_w,          # Aleatoric: neighbor disagreement
            1.0 / k_eff,      # Epistemic: sparse neighborhood
            ambiguity         # Ambiguity: identifiability
        ])
        
        return NeighborhoodFeatures(
            s1=s1,
            delta_s=delta_s,
            sigma_w=sigma_w,
            k_eff=k_eff,
            ambiguity=ambiguity,
            z=z
        )
    
    def _fallback_features(self) -> NeighborhoodFeatures:
        """Features when no neighbors found (maximum uncertainty)."""
        sigma = self.global_sigma_fallback if self.global_sigma_fallback else 50.0
        z = np.array([1.0, sigma, 1.0, np.log(1 + 100)])
        return NeighborhoodFeatures(
            s1=0.0, delta_s=0.0, sigma_w=sigma,
            k_eff=1.0, ambiguity=np.log(101), z=z
        )
    
    def score(self, features: NeighborhoodFeatures) -> float:
        """
        Compute uncertainty score u(x) = w⊤z.
        
        Higher score = higher uncertainty = wider intervals.
        """
        return float(np.dot(self.weights, features.z))
    
    def set_weights(self, weights: np.ndarray):
        """Set learned nonnegative weights."""
        assert len(weights) == 4, "Need 4 weights"
        assert np.all(weights >= 0), "Weights must be nonnegative"
        self.weights = weights.copy()
    
    def set_global_sigma(self, sigma: float):
        """Set data-derived global sigma fallback."""
        self.global_sigma_fallback = sigma


def learn_weights(features_list: List[NeighborhoodFeatures], 
                  residuals: np.ndarray) -> np.ndarray:
    """
    Learn nonnegative weights via NNLS.
    
    Solves: min_w ||r - Zw||² s.t. w ≥ 0
    
    Args:
        features_list: List of NeighborhoodFeatures from calibration
        residuals: Absolute residuals |y - ŷ| for calibration points
        
    Returns:
        Learned weights w ≥ 0 (shape: (4,))
    """
    from scipy.optimize import nnls
    
    Z = np.array([f.z for f in features_list])  # (n, 4)
    r = np.array(residuals)
    
    # Nonnegative least squares
    w, _ = nnls(Z, r)
    
    # Ensure some minimum weight to avoid degenerate cases
    w = np.maximum(w, 0.01)
    
    return w


def compute_global_sigma(values_by_query: List[np.ndarray]) -> float:
    """
    Compute global σ fallback from all neighbor value distributions.
    
    Uses median absolute deviation (MAD) for robustness.
    """
    all_stds = []
    for values in values_by_query:
        if len(values) >= 2:
            all_stds.append(np.std(values))
    
    if len(all_stds) > 0:
        return float(np.median(all_stds))
    return 30.0  # Conservative default
