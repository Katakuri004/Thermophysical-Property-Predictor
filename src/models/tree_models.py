from .base import BaseModel
import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from typing import Dict, Any, Optional

class XGBoostModel(BaseModel):
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        default_params = {
            'objective': 'reg:absoluteerror',
            'n_estimators': 1000,
            'learning_rate': 0.05,
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1,
            'early_stopping_rounds': 50
        }
        if params:
            default_params.update(params)
            
        # Extract early_stopping_rounds to handle it dynamically
        self.early_stopping_rounds = default_params.pop('early_stopping_rounds', None)
        
        super().__init__("XGBoost", default_params)
        
    def fit(self, X: pd.DataFrame, y: pd.Series, X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None):
        # Prepare params, injecting early_stopping_rounds only if validation data is available
        fit_params = self.params.copy()
        eval_set = None
        
        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]
            if self.early_stopping_rounds is not None:
                fit_params['early_stopping_rounds'] = self.early_stopping_rounds
        
        # Initialize model with appropriate params
        self.model = xgb.XGBRegressor(**fit_params)
            
        self.model.fit(
            X, y,
            eval_set=eval_set,
            verbose=False
        )
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        return self.model.predict(X)

class LightGBMModel(BaseModel):
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        default_params = {
            'objective': 'regression_l1',
            'metric': 'mae',
            'n_estimators': 1000,
            'learning_rate': 0.05,
            'num_leaves': 31,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1,
            'early_stopping_rounds': 50
        }
        if params:
            default_params.update(params)
        
        self.early_stopping_rounds = default_params.pop('early_stopping_rounds', 50)
        
        super().__init__("LightGBM", default_params)
        
    def fit(self, X: pd.DataFrame, y: pd.Series, X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None):
        self.model = lgb.LGBMRegressor(**self.params)
        
        callbacks = []
        eval_set = None
        
        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]
            if self.early_stopping_rounds is not None:
                callbacks.append(lgb.early_stopping(stopping_rounds=self.early_stopping_rounds))
            
        self.model.fit(
            X, y,
            eval_set=eval_set,
            eval_metric='mae',
            callbacks=callbacks
        )
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        return self.model.predict(X)

class CatBoostModel(BaseModel):
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        default_params = {
            'loss_function': 'MAE',
            'iterations': 1000,
            'learning_rate': 0.05,
            'depth': 6,
            'l2_leaf_reg': 3,
            'random_state': 42,
            'verbose': False,
            'allow_writing_files': False,
            'early_stopping_rounds': 50
        }
        if params:
            default_params.update(params)
            
        self.early_stopping_rounds = default_params.pop('early_stopping_rounds', 50)
            
        super().__init__("CatBoost", default_params)
        
    def fit(self, X: pd.DataFrame, y: pd.Series, X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None):
        self.model = CatBoostRegressor(**self.params)
        
        eval_set = None
        fit_params = {}
        
        if X_val is not None and y_val is not None:
            eval_set = (X_val, y_val)
            if self.early_stopping_rounds is not None:
                fit_params['early_stopping_rounds'] = self.early_stopping_rounds
            
        self.model.fit(
            X, y,
            eval_set=eval_set,
            verbose=False,
            **fit_params
        )
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        return self.model.predict(X)
