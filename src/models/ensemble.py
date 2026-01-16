import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Union
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.model_selection import KFold
from .base import BaseModel

class StackingEnsemble(BaseModel):
    def __init__(self, base_models: Dict[str, BaseModel], meta_model: Optional[BaseModel] = None, n_folds: int = 5):
        """
        Stacking Ensemble.
        
        Args:
            base_models: Dictionary of base models {name: model_instance}.
            meta_model: The meta-model to learn from base model predictions. Defaults to Ridge.
            n_folds: Number of folds for generating OOF predictions for training the meta-model.
        """
        super().__init__("StackingEnsemble")
        self.base_models = base_models
        self.meta_model = meta_model if meta_model else Ridge(alpha=1.0)
        self.n_folds = n_folds
        
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fit the stacking ensemble.
        1. Generate OOF predictions for all base models.
        2. Train meta-model on OOF predictions.
        3. Retrain all base models on full dataset (if not already trained).
        """
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        
        # DataFrame to store OOF predictions
        oof_preds = pd.DataFrame(index=X.index, columns=self.base_models.keys())
        
        print(f"Generating OOF predictions with {self.n_folds} folds...")
        
        # Generate OOF predictions
        for name, model in self.base_models.items():
            print(f"  Processing {name}...")
            
            for train_idx, val_idx in kf.split(X, y):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                # Clone model to avoid modifying the original instance structure if needed
                # For simplicity here, we assume standard sklearn-like 'fit' resets or overwrites
                # However, our wrappers track state. We should ideally create fresh instances.
                # Here we will re-instantiate if possible, or just call fit.
                # Assuming our wrappers work like sklearn (fit resets).
                
                # Note: To do this properly, we should ideally clone. 
                # Since we passed instances, let's assume they are fresh or fit resets them.
                # BUT, since we loop folds, we can't reuse the same instance 'model' for each fold simply
                # without overwriting. That's fine for OOF generation.
                
                model.fit(X_train, y_train, X_val, y_val)
                oof_preds.loc[val_idx, name] = model.predict(X_val)
                
        # Train meta-model
        print("Training meta-model...")
        self.meta_model.fit(oof_preds, y)
        
        # Retrain base models on full data for final prediction
        print("Retraining base models on full dataset...")
        for name, model in self.base_models.items():
            # For tree models, we might struggle without X_val for early stopping if not careful.
            # Our wrappers now handle this by ignoring early stopping if X_val is None.
            model.fit(X, y)
            
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions.
        1. Predict with all base models.
        2. Feed base predictions to meta-model.
        """
        base_preds = pd.DataFrame(index=X.index, columns=self.base_models.keys())
        
        for name, model in self.base_models.items():
            base_preds[name] = model.predict(X)
            
        return self.meta_model.predict(base_preds)

class BlendingEnsemble(BaseModel):
    def __init__(self, models: Dict[str, BaseModel], weights: Optional[Dict[str, float]] = None):
        """
        Simple Weighted Blending Ensemble.
        
        Args:
            models: Dictionary of trained models.
            weights: Dictionary of weights {model_name: weight}. If None, use equal weights.
        """
        super().__init__("BlendingEnsemble")
        self.models = models
        if weights:
            self.weights = weights
            # Normalize
            total = sum(weights.values())
            self.weights = {k: v/total for k, v in weights.items()}
        else:
            self.weights = {k: 1.0/len(models) for k in models.keys()}
            
    def fit(self, X: pd.DataFrame, y: pd.Series):
        # Blending usually assumes models are already trained or we just train them here.
        # We'll train them here to be consistent.
        for name, model in self.models.items():
            model.fit(X, y)
            
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        final_preds = np.zeros(len(X))
        
        for name, model in self.models.items():
            preds = model.predict(X)
            final_preds += preds * self.weights[name]
            
        return final_preds
