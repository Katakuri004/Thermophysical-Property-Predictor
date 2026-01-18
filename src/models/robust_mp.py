"""
Noise-Robust Learning v2.0 - Optimized Implementation

Fixes applied:
- Conformal prediction for calibrated intervals
- Grouped conformal by source/risk
- Efficient ensemble inference (single pass)
- Quantile LightGBM for heteroskedastic intervals
- Active source calibration
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings('ignore')

try:
    from lightgbm import LGBMRegressor
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False


class ConformalCalibrator:
    """
    Split conformal prediction for calibrated prediction intervals.
    Provides coverage guarantees under mild exchangeability assumptions.
    """
    
    def __init__(self, alpha: float = 0.1):
        """
        Args:
            alpha: Miscoverage rate (0.1 = 90% coverage target)
        """
        self.alpha = alpha
        self.q = None
        self.group_q = {}
        
    def calibrate(self, y_true: np.ndarray, y_pred: np.ndarray, 
                  groups: List[str] = None) -> float:
        """
        Calibrate conformal quantile from residuals.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            groups: Optional group labels for grouped conformal
        
        Returns:
            Global conformity quantile
        """
        residuals = np.abs(y_true - y_pred)
        n = len(residuals)
        
        # Global quantile with finite-sample correction
        q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        q_level = min(q_level, 1.0)
        self.q = float(np.quantile(residuals, q_level))
        
        # Grouped quantiles if provided
        if groups is not None:
            unique_groups = set(groups)
            for group in unique_groups:
                mask = np.array([g == group for g in groups])
                if mask.sum() >= 10:  # Minimum samples for reliable quantile
                    group_residuals = residuals[mask]
                    n_g = len(group_residuals)
                    q_level_g = np.ceil((n_g + 1) * (1 - self.alpha)) / n_g
                    q_level_g = min(q_level_g, 1.0)
                    self.group_q[group] = float(np.quantile(group_residuals, q_level_g))
                else:
                    self.group_q[group] = self.q
        
        return self.q
    
    def predict_interval(self, y_pred: np.ndarray, 
                        groups: List[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute prediction intervals.
        
        Returns:
            (lower_bounds, upper_bounds)
        """
        if self.q is None:
            raise RuntimeError("Calibrator not fitted. Call calibrate() first.")
        
        if groups is not None and self.group_q:
            # Use group-specific quantiles
            q_values = np.array([
                self.group_q.get(g, self.q) for g in groups
            ])
            return y_pred - q_values, y_pred + q_values
        
        return y_pred - self.q, y_pred + self.q
    
    def get_coverage(self, y_true: np.ndarray, y_pred: np.ndarray,
                    groups: List[str] = None) -> float:
        """Compute empirical coverage on held-out data."""
        lower, upper = self.predict_interval(y_pred, groups)
        in_interval = (y_true >= lower) & (y_true <= upper)
        return float(np.mean(in_interval))


class QuantileEnsemble:
    """
    Quantile regression ensemble for heteroskedastic prediction intervals.
    Trains separate models for different quantiles.
    """
    
    def __init__(self, quantiles: List[float] = None):
        """
        Args:
            quantiles: List of quantiles to predict (default: 0.05, 0.5, 0.95)
        """
        self.quantiles = quantiles or [0.05, 0.5, 0.95]
        self.models = {}
        self.is_fitted = False
        
    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray = None):
        """Train quantile models."""
        if not LGBM_AVAILABLE:
            raise RuntimeError("LightGBM not available")
        
        for q in self.quantiles:
            model = LGBMRegressor(
                objective='quantile',
                alpha=q,
                n_estimators=300,
                learning_rate=0.1,
                num_leaves=31,
                random_state=42,
                verbose=-1
            )
            model.fit(X, y, sample_weight=sample_weight)
            self.models[q] = model
            print(f"  Trained quantile {q} model")
        
        self.is_fitted = True
    
    def predict(self, X: np.ndarray) -> Dict[float, np.ndarray]:
        """Predict all quantiles."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")
        
        return {q: model.predict(X) for q, model in self.models.items()}
    
    def predict_interval(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict median with adaptive intervals.
        
        Returns:
            (lower, median, upper)
        """
        preds = self.predict(X)
        return preds[0.05], preds[0.5], preds[0.95]


class SourceCalibrator:
    """
    Per-source quality calibration with learned factors.
    """
    
    # Default source quality weights (higher = more reliable)
    DEFAULT_WEIGHTS = {
        'kaggle': 1.0,
        'bradley': 0.95,
        'bradley_plus': 0.97,
        'smp': 0.80,
        'pubchem': 0.75,
        'unknown': 0.85
    }
    
    def __init__(self, source_weights: Dict[str, float] = None):
        self.source_weights = source_weights or self.DEFAULT_WEIGHTS.copy()
        self.learned_factors = {}
        self.is_calibrated = False
        
    def get_weight(self, source: str) -> float:
        """Get quality weight for a data source."""
        return self.source_weights.get(source.lower(), self.source_weights['unknown'])
    
    def calibrate_from_cv(self, sources: List[str], predictions: np.ndarray, 
                         targets: np.ndarray) -> Dict[str, float]:
        """
        Learn calibration factors from cross-validation residuals.
        
        Factors > 1 indicate higher-than-average error for that source.
        """
        residuals = np.abs(predictions - targets)
        overall_mae = np.mean(residuals)
        
        # Group by source
        source_residuals = {}
        for src, res in zip(sources, residuals):
            src_lower = src.lower() if isinstance(src, str) else 'unknown'
            if src_lower not in source_residuals:
                source_residuals[src_lower] = []
            source_residuals[src_lower].append(res)
        
        # Compute calibration factor per source
        for src, res_list in source_residuals.items():
            src_mae = np.mean(res_list)
            factor = src_mae / overall_mae if overall_mae > 0 else 1.0
            self.learned_factors[src] = factor
            print(f"  Source '{src}': n={len(res_list)}, MAE={src_mae:.2f}, factor={factor:.2f}")
        
        self.is_calibrated = True
        return self.learned_factors
    
    def get_interval_scaling(self, source: str) -> float:
        """
        Get interval scaling factor for a source.
        Higher factor = wider intervals (less reliable source).
        """
        if self.is_calibrated and source.lower() in self.learned_factors:
            return self.learned_factors[source.lower()]
        
        # Fall back to inverse of default weight
        weight = self.get_weight(source)
        return 1.0 / weight if weight > 0 else 1.5


class PolymorphismRiskDetector:
    """
    Detect molecules with high polymorphism risk.
    High-risk molecules need wider prediction intervals.
    """
    
    def __init__(self, threshold: float = 0.7):
        self.threshold = threshold
        self.risk_residuals = {}  # Learned from validation
        
    def compute_risk_score(self, mol_features: Dict[str, float]) -> float:
        """
        Compute polymorphism risk score based on molecular features.
        Returns score [0, 1] where higher = more likely polymorphic.
        """
        score = 0.0
        
        # Rotatable bonds: high flexibility → higher risk
        rot_bonds = mol_features.get('NumRotatableBonds', 0)
        if rot_bonds > 5:
            score += 0.3
        elif rot_bonds > 2:
            score += 0.15
        
        # H-bond capacity: more H-bonds → more packing options
        hbd = mol_features.get('NumHBD', 0)
        hba = mol_features.get('NumHBA', 0)
        hbond_capacity = hbd + hba
        if hbond_capacity > 4:
            score += 0.25
        elif hbond_capacity > 2:
            score += 0.1
        
        # Aromatic rings: planar regions can pack differently
        arom = mol_features.get('NumAromaticRings', 0)
        if arom >= 2:
            score += 0.2
        elif arom == 1:
            score += 0.1
        
        # CSP3 fraction: low = more rigid/planar
        csp3 = mol_features.get('FractionCSP3', 0.5)
        if csp3 < 0.3:
            score += 0.15
        
        return min(score, 1.0)
    
    def classify_risk(self, mol_features: Dict[str, float]) -> str:
        """Classify molecule into risk category."""
        score = self.compute_risk_score(mol_features)
        
        if score > self.threshold:
            return 'high'
        elif score > self.threshold * 0.5:
            return 'medium'
        else:
            return 'low'
    
    def get_interval_scaling(self, risk_class: str) -> float:
        """Get interval scaling factor for risk class."""
        scaling = {'low': 1.0, 'medium': 1.3, 'high': 1.7}
        return scaling.get(risk_class, 1.0)


class EfficientEnsemble:
    """
    Memory-efficient ensemble with single-pass prediction.
    """
    
    def __init__(self, n_estimators: int = 10, subsample_ratio: float = 0.8):
        self.n_estimators = n_estimators
        self.subsample_ratio = subsample_ratio
        self.models = []
        
    def fit(self, X: np.ndarray, y: np.ndarray, sample_weights: np.ndarray = None):
        """Train ensemble with bootstrap sampling."""
        if not LGBM_AVAILABLE:
            raise RuntimeError("LightGBM not available")
        
        n_samples = len(X)
        n_subsample = int(n_samples * self.subsample_ratio)
        
        self.models = []
        for i in range(self.n_estimators):
            indices = np.random.choice(n_samples, n_subsample, replace=True)
            X_sub = X[indices]
            y_sub = y[indices]
            weights = sample_weights[indices] if sample_weights is not None else None
            
            model = LGBMRegressor(
                n_estimators=200,
                learning_rate=0.1,
                num_leaves=31,
                objective='regression_l1',
                random_state=42 + i,
                verbose=-1
            )
            model.fit(X_sub, y_sub, sample_weight=weights)
            self.models.append(model)
        
        print(f"Trained {len(self.models)} ensemble members")
    
    def predict_full(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Single-pass prediction returning all statistics.
        
        Returns:
            (mean, std, lower_5pct, upper_95pct)
        """
        # Single pass through all models
        all_preds = np.vstack([m.predict(X) for m in self.models])  # (M, N)
        
        mean = all_preds.mean(axis=0)
        std = all_preds.std(axis=0)
        lower = np.percentile(all_preds, 5, axis=0)
        upper = np.percentile(all_preds, 95, axis=0)
        
        return mean, std, lower, upper


class RobustMPPredictor:
    """
    Robust Melting Point Predictor v2.0 with:
    - Conformal calibrated prediction intervals
    - Per-source calibration
    - Polymorphism risk detection
    - Efficient ensemble inference
    """
    
    def __init__(self, 
                 use_quantile: bool = True,
                 use_conformal: bool = True,
                 n_ensemble: int = 10,
                 alpha: float = 0.1):
        
        self.use_quantile = use_quantile
        self.use_conformal = use_conformal
        
        self.calibrator = SourceCalibrator()
        self.conformal = ConformalCalibrator(alpha=alpha)
        self.risk_detector = PolymorphismRiskDetector()
        
        if use_quantile:
            self.predictor = QuantileEnsemble()
        else:
            self.predictor = EfficientEnsemble(n_estimators=n_ensemble)
        
        self.is_fitted = False
        
    def fit(self, X: np.ndarray, y: np.ndarray, 
            sources: List[str] = None,
            calibration_split: float = 0.2):
        """
        Fit with integrated calibration.
        
        Args:
            X: Feature matrix
            y: Target values
            sources: Data source labels (optional)
            calibration_split: Fraction for conformal calibration
        """
        n = len(X)
        n_cal = int(n * calibration_split)
        
        # Random split for calibration
        indices = np.random.permutation(n)
        train_idx = indices[n_cal:]
        cal_idx = indices[:n_cal]
        
        X_train, y_train = X[train_idx], y[train_idx]
        X_cal, y_cal = X[cal_idx], y[cal_idx]
        
        # Get sample weights from sources
        if sources is not None:
            train_sources = [sources[i] for i in train_idx]
            cal_sources = [sources[i] for i in cal_idx]
            weights = np.array([self.calibrator.get_weight(s) for s in train_sources])
        else:
            train_sources = None
            cal_sources = None
            weights = None
        
        print(f"Training on {len(X_train)} samples, calibrating on {len(X_cal)} samples")
        
        # Fit predictor
        if self.use_quantile:
            self.predictor.fit(X_train, y_train, sample_weight=weights)
        else:
            self.predictor.fit(X_train, y_train, sample_weights=weights)
        
        # Get calibration predictions
        if self.use_quantile:
            preds = self.predictor.predict(X_cal)
            y_pred_cal = preds[0.5]  # Median
        else:
            y_pred_cal, _, _, _ = self.predictor.predict_full(X_cal)
        
        # Calibrate conformal intervals
        if self.use_conformal:
            self.conformal.calibrate(y_cal, y_pred_cal, groups=cal_sources)
            coverage = self.conformal.get_coverage(y_cal, y_pred_cal, cal_sources)
            print(f"Conformal calibration complete. Empirical coverage: {coverage:.1%}")
        
        # Calibrate source factors
        if sources is not None:
            print("\nSource calibration:")
            self.calibrator.calibrate_from_cv(cal_sources, y_pred_cal, y_cal)
        
        self.is_fitted = True
        
    def predict(self, X: np.ndarray, 
                sources: List[str] = None,
                risk_classes: List[str] = None) -> pd.DataFrame:
        """
        Predict with calibrated intervals.
        
        Returns:
            DataFrame with: Tm_pred, Tm_lower, Tm_upper, uncertainty
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")
        
        # Get predictions
        if self.use_quantile:
            lower_q, median, upper_q = self.predictor.predict_interval(X)
            pred = median
            uncertainty = (upper_q - lower_q) / 2
        else:
            pred, std, lower_q, upper_q = self.predictor.predict_full(X)
            uncertainty = std
        
        # Apply conformal calibration if enabled
        if self.use_conformal:
            lower_c, upper_c = self.conformal.predict_interval(pred, sources)
            
            # Use wider of quantile or conformal intervals
            lower = np.minimum(lower_q, lower_c)
            upper = np.maximum(upper_q, upper_c)
        else:
            lower, upper = lower_q, upper_q
        
        # Apply risk-based interval scaling
        if risk_classes is not None:
            for i, risk in enumerate(risk_classes):
                scale = self.risk_detector.get_interval_scaling(risk)
                half_width = (upper[i] - lower[i]) / 2
                lower[i] = pred[i] - half_width * scale
                upper[i] = pred[i] + half_width * scale
        
        return pd.DataFrame({
            'Tm_pred': pred,
            'Tm_lower': lower,
            'Tm_upper': upper,
            'uncertainty': uncertainty 
        })


if __name__ == "__main__":
    print("Noise-Robust Learning v2.0 Demo")
    print("=" * 50)
    
    if not LGBM_AVAILABLE:
        print("Install LightGBM to run demo")
    else:
        # Demo data
        np.random.seed(42)
        n = 500
        X = np.random.randn(n, 10)
        y = X[:, 0] * 10 + 300 + np.random.randn(n) * 5
        sources = ['kaggle'] * 150 + ['bradley'] * 150 + ['smp'] * 200
        
        # Fit robust predictor
        predictor = RobustMPPredictor(use_quantile=True, use_conformal=True)
        predictor.fit(X, y, sources)
        
        # Predict
        X_test = np.random.randn(20, 10)
        test_sources = ['kaggle'] * 10 + ['smp'] * 10
        results = predictor.predict(X_test, sources=test_sources)
        
        print("\nPredictions with calibrated intervals:")
        print(results.head(10))
        
        # Compute interval widths
        widths = results['Tm_upper'] - results['Tm_lower']
        print(f"\nInterval widths: mean={widths.mean():.1f}K, std={widths.std():.1f}K")
