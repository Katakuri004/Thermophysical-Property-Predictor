import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from sklearn.preprocessing import StandardScaler
from .base import BaseModel

class MeltingPointMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims=[512, 256, 128, 64], dropout_rate=0.2):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

class NeuralNetworkModel(BaseModel):
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        default_params = {
            'hidden_dims': [512, 256, 128, 64],
            'dropout_rate': 0.2,
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 100,
            'patience': 10,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'
        }
        if params:
            default_params.update(params)
        super().__init__("NeuralNetwork", default_params)
        
        self.scaler = StandardScaler()
        self.best_model_state = None
        
    def fit(self, X: pd.DataFrame, y: pd.Series, X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None):
        # Scale Data
        X_scaled = self.scaler.fit_transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.params['device'])
        y_tensor = torch.FloatTensor(y.values).reshape(-1, 1).to(self.params['device'])
        
        val_loader = None
        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            X_val_tensor = torch.FloatTensor(X_val_scaled).to(self.params['device'])
            y_val_tensor = torch.FloatTensor(y_val.values).reshape(-1, 1).to(self.params['device'])
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
            val_loader = DataLoader(val_dataset, batch_size=self.params['batch_size'], shuffle=False)
            
        # Initialize Model
        input_dim = X.shape[1]
        self.model = MeltingPointMLP(
            input_dim, 
            self.params['hidden_dims'], 
            self.params['dropout_rate']
        ).to(self.params['device'])
        
        criterion = nn.L1Loss()  # MAE Loss
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.params['learning_rate'])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        
        # Training Loop
        train_dataset = TensorDataset(X_tensor, y_tensor)
        train_loader = DataLoader(train_dataset, batch_size=self.params['batch_size'], shuffle=True)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.params['epochs']):
            self.model.train()
            train_loss = 0
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                preds = self.model(batch_X)
                loss = criterion(preds, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * batch_X.size(0)
                
            train_loss /= len(train_dataset)
            
            # Validation
            val_loss = 0
            if val_loader:
                self.model.eval()
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        preds = self.model(batch_X)
                        loss = criterion(preds, batch_y)
                        val_loss += loss.item() * batch_X.size(0)
                val_loss /= len(val_loader.dataset)
                
                scheduler.step(val_loss)
                
                # Early Stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.best_model_state = self.model.state_dict()
                    patience_counter = 0
                else:
                    patience_counter += 1
                    
                if patience_counter >= self.params['patience']:
                    print(f"Early stopping at epoch {epoch}")
                    break
        
        # Restore best model
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
            
        self.model.eval()
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.params['device'])
        
        with torch.no_grad():
            preds = self.model(X_tensor)
            
        return preds.cpu().numpy().flatten()
