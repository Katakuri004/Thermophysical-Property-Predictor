from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import joblib
from typing import Any, Dict, Optional

class BaseModel(ABC):
    """
    Abstract base class for all melting point prediction models.
    Enforces a consistent API for training, prediction, and persistence.
    """
    
    def __init__(self, name: str, params: Optional[Dict[str, Any]] = None):
        self.name = name
        self.params = params or {}
        self.model = None

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Train the model."""
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        pass
    
    def save(self, filepath: str):
        """Save the model to disk."""
        if self.model is not None:
            joblib.dump(self.model, filepath)
        else:
            raise ValueError("Model has not been trained yet.")
            
    def load(self, filepath: str):
        """Load the model from disk."""
        self.model = joblib.load(filepath)
