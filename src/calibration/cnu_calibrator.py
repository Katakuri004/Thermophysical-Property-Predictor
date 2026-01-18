"""
Calibrated Neighborhood Uncertainty (CNU) - Calibrator

Regime-based calibration with percentile-derived boundaries and 
post-routing calibration for global coverage guarantees.

Theoretical Basis:
- Theorem 2: Regime-conditional coverage via split conformal
- Proposition: Risk-aware routing via quantile comparison
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass

from .uncertainty_functional import (
    NeighborhoodUncertainty, 
    NeighborhoodFeatures,
    learn_weights,
    compute_global_sigma
)


@dataclass
class CalibrationResult:
    """Result of CNU calibration."""
    regime_quantiles: Dict[str, float]
    regime_counts: Dict[str, int]
    global_quantile: float
    boundaries: np.ndarray
    weights: np.ndarray
    coverage_achieved: Dict[str, float]


class CNUCalibrator:
    """
    Calibrated Neighborhood Uncertainty Calibrator.
    
    Provides:
    1. Learned uncertainty weights from calibration residuals
    2. Percentile-based regime boundaries (data-driven)
    3. Per-regime conformal quantiles
    4. Post-routing global quantile for conservative coverage
    """
    
    def __init__(self, 
                 alpha: float = 0.10,
                 n_regimes: int = 5,
                 min_regime_size: int = 10):
        """
        Initialize calibrator.
        
        Args:
            alpha: Miscoverage rate (1-alpha = target coverage)
            n_regimes: Number of regimes (will evaluate 3/5/8/10 if None)
            min_regime_size: Minimum samples per regime before merging
        """
        self.alpha = alpha
        self.n_regimes = n_regimes
        self.min_regime_size = min_regime_size
        
        self.uncertainty = NeighborhoodUncertainty()
        self.regime_quantiles: Dict[str, float] = {}
        self.boundaries: np.ndarray = None
        self.global_quantile: float = None
        self.is_fitted = False
        
    def fit(self,
            predictions: np.ndarray,
            actuals: np.ndarray,
            features_list: List[NeighborhoodFeatures],
            neighbor_values_list: Optional[List[np.ndarray]] = None) -> CalibrationResult:
        """
        Fit CNU calibrator on calibration data.
        
        Args:
            predictions: Predicted values
            actuals: Actual values
            features_list: Neighborhood features for each calibration point
            neighbor_values_list: Optional neighbor values for global sigma
            
        Returns:
            CalibrationResult with all calibration statistics
        """
        residuals = np.abs(actuals - predictions)
        n = len(residuals)
        
        # Step 1: Set global sigma fallback (data-derived)
        if neighbor_values_list:
            global_sigma = compute_global_sigma(neighbor_values_list)
        else:
            global_sigma = float(np.median([f.sigma_w for f in features_list]))
        self.uncertainty.set_global_sigma(global_sigma)
        
        # Step 2: Learn nonnegative weights
        weights = learn_weights(features_list, residuals)
        self.uncertainty.set_weights(weights)
        
        # Step 3: Compute u(x) scores
        u_scores = np.array([self.uncertainty.score(f) for f in features_list])
        
        # Step 4: Select optimal number of regimes (if needed)
        if self.n_regimes is None:
            self.n_regimes = self._select_n_regimes(u_scores, residuals)
        
        # Step 5: Define regime boundaries from percentiles
        percentiles = np.linspace(0, 100, self.n_regimes + 1)
        self.boundaries = np.percentile(u_scores, percentiles)
        
        # Step 6: Assign regimes and calibrate per-regime quantiles
        regimes = self._assign_regimes(u_scores)
        self.regime_quantiles, regime_counts = self._calibrate_regimes(
            residuals, regimes
        )
        
        # Step 7: Post-routing global quantile (for conservative global coverage)
        self.global_quantile = self._compute_conformal_quantile(residuals)
        
        # Step 8: Compute achieved coverage per regime (validation)
        coverage_achieved = self._compute_regime_coverage(
            residuals, regimes, self.regime_quantiles
        )
        
        self.is_fitted = True
        
        return CalibrationResult(
            regime_quantiles=self.regime_quantiles,
            regime_counts=regime_counts,
            global_quantile=self.global_quantile,
            boundaries=self.boundaries,
            weights=weights,
            coverage_achieved=coverage_achieved
        )
    
    def _select_n_regimes(self, 
                          u_scores: np.ndarray, 
                          residuals: np.ndarray) -> int:
        """Select optimal regime count via coverage stability."""
        candidates = [3, 5, 8, 10]
        best_n = 5
        best_score = float('inf')
        
        for n in candidates:
            percentiles = np.linspace(0, 100, n + 1)
            boundaries = np.percentile(u_scores, percentiles)
            regimes = np.digitize(u_scores, boundaries[1:-1])
            
            # Check minimum regime sizes
            counts = np.bincount(regimes, minlength=n)
            if np.min(counts) < self.min_regime_size:
                continue
            
            # Score: balance between interval width and coverage variance
            quantiles = []
            for r in range(n):
                mask = regimes == r
                if mask.sum() >= self.min_regime_size:
                    q = self._compute_conformal_quantile(residuals[mask])
                    quantiles.append(q)
            
            if len(quantiles) >= 2:
                # Prefer lower mean width with stable coverage
                score = np.mean(quantiles) + 0.5 * np.std(quantiles)
                if score < best_score:
                    best_score = score
                    best_n = n
        
        return best_n
    
    def _assign_regimes(self, u_scores: np.ndarray) -> np.ndarray:
        """Assign regime indices based on u(x) and boundaries."""
        # np.digitize gives bin index; clip to valid range
        regimes = np.digitize(u_scores, self.boundaries[1:-1])
        return np.clip(regimes, 0, self.n_regimes - 1)
    
    def _calibrate_regimes(self, 
                           residuals: np.ndarray,
                           regimes: np.ndarray) -> Tuple[Dict[str, float], Dict[str, int]]:
        """Calibrate per-regime quantiles with merging for small regimes."""
        regime_quantiles = {}
        regime_counts = {}
        
        for r in range(self.n_regimes):
            regime_name = f'regime_{r}'
            mask = regimes == r
            count = mask.sum()
            regime_counts[regime_name] = count
            
            if count >= self.min_regime_size:
                regime_quantiles[regime_name] = self._compute_conformal_quantile(
                    residuals[mask]
                )
            else:
                # Merge with adjacent regime (use global as fallback)
                regime_quantiles[regime_name] = None  # Will use global
        
        # Fill None with global quantile
        global_q = self._compute_conformal_quantile(residuals)
        for k in regime_quantiles:
            if regime_quantiles[k] is None:
                regime_quantiles[k] = global_q
                
        return regime_quantiles, regime_counts
    
    def _compute_conformal_quantile(self, residuals: np.ndarray) -> float:
        """Finite-sample split-conformal quantile."""
        n = len(residuals)
        if n == 0:
            return float('inf')
        
        # Standard conformal quantile: ceil((n+1)(1-α)) / n
        idx = int(np.ceil((n + 1) * (1 - self.alpha)))
        idx = min(idx, n) - 1  # 0-indexed
        idx = max(idx, 0)
        
        sorted_res = np.sort(residuals)
        return float(sorted_res[idx])
    
    def _compute_regime_coverage(self,
                                  residuals: np.ndarray,
                                  regimes: np.ndarray,
                                  quantiles: Dict[str, float]) -> Dict[str, float]:
        """Compute empirical coverage per regime."""
        coverage = {}
        for r in range(self.n_regimes):
            regime_name = f'regime_{r}'
            mask = regimes == r
            if mask.sum() > 0:
                q = quantiles.get(regime_name, self.global_quantile)
                covered = (residuals[mask] <= q).mean()
                coverage[regime_name] = float(covered)
        return coverage
    
    def get_interval(self, 
                     features: NeighborhoodFeatures,
                     prediction: float) -> Tuple[float, float, str]:
        """
        Get prediction interval for a query.
        
        Returns:
            (lower, upper, regime_name)
        """
        assert self.is_fitted, "Calibrator not fitted"
        
        u_score = self.uncertainty.score(features)
        regime_idx = np.digitize([u_score], self.boundaries[1:-1])[0]
        regime_idx = min(max(regime_idx, 0), self.n_regimes - 1)
        regime_name = f'regime_{regime_idx}'
        
        # Use per-regime quantile, with global as conservative fallback
        q_regime = self.regime_quantiles.get(regime_name, self.global_quantile)
        q = max(q_regime, 0)  # Safety
        
        return (prediction - q, prediction + q, regime_name)
    
    def get_conservative_interval(self,
                                   features: NeighborhoodFeatures,
                                   prediction: float) -> Tuple[float, float]:
        """
        Get conservative interval using max(regime, global) quantile.
        
        Use this for guaranteed global coverage on routed systems.
        """
        low, high, regime = self.get_interval(features, prediction)
        q_regime = (high - low) / 2
        q_conservative = max(q_regime, self.global_quantile)
        return (prediction - q_conservative, prediction + q_conservative)
    
    def should_use_retrieval(self,
                              features: NeighborhoodFeatures,
                              fallback_quantile: Optional[float] = None) -> bool:
        """
        Risk-aware routing decision.
        
        Returns True if retrieval has narrower calibrated intervals than fallback.
        """
        u_score = self.uncertainty.score(features)
        regime_idx = np.digitize([u_score], self.boundaries[1:-1])[0]
        regime_idx = min(max(regime_idx, 0), self.n_regimes - 1)
        regime_name = f'regime_{regime_idx}'
        
        q_retrieval = self.regime_quantiles.get(regime_name, self.global_quantile)
        q_fallback = fallback_quantile or self.global_quantile
        
        return q_retrieval <= q_fallback
