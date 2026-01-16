import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.loader import DataLoader
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from ..features.graph import GraphFeaturizer
from .base import BaseModel

class GCN(nn.Module):
    def __init__(self, num_node_features, hidden_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = nn.Linear(hidden_channels, 1)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings 
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        return x

class GNNModel(BaseModel):
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        default_params = {
            'hidden_channels': 64,
            'learning_rate': 0.01,
            'batch_size': 32,
            'epochs': 50,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'
        }
        if params:
            default_params.update(params)
        super().__init__("GNN", default_params)
        
        self.featurizer = GraphFeaturizer()
        
    def fit(self, X: pd.DataFrame, y: pd.Series, X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None):
        # Convert DataFrames to Graph Data Lists
        print("Featurizing training graphs...")
        train_data = self.featurizer.process_dataframe(X, target_col=y.name if isinstance(y, pd.Series) else 'Tm')
        # Manually attach y to data objects if process_dataframe didn't (it tries to if column exists)
        # But here X doesn't have 'Tm'. processing needs X and y together or we modify logic.
        # My featurizer logic took 'df' and looked for 'target_col'.
        # So I should pass a combined df or manually set y.
        
        # Fixing featurizer interaction:
        # process_dataframe iterates rows. X doesn't have y. y is separate.
        # We need to assign y.
        for i, data in enumerate(train_data):
            data.y = torch.tensor([y.iloc[i]], dtype=torch.float)
            
        train_loader = DataLoader(train_data, batch_size=self.params['batch_size'], shuffle=True)
        
        val_loader = None
        if X_val is not None and y_val is not None:
            print("Featurizing validation graphs...")
            val_data = self.featurizer.process_dataframe(X_val)
            for i, data in enumerate(val_data):
                data.y = torch.tensor([y_val.iloc[i]], dtype=torch.float)
            val_loader = DataLoader(val_data, batch_size=self.params['batch_size'], shuffle=False)
            
        # Model Init
        num_features = 5 # As defined in GraphFeaturizer
        self.model = GCN(num_features, self.params['hidden_channels']).to(self.params['device'])
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.params['learning_rate'])
        criterion = nn.L1Loss() # MAE

        # Training Loop
        print(f"Training GNN on {self.params['device']}...")
        self.model.train()
        for epoch in range(self.params['epochs']):
            total_loss = 0
            for batch in train_loader:
                batch = batch.to(self.params['device'])
                optimizer.zero_grad()
                out = self.model(batch.x, batch.edge_index, batch.batch)
                loss = criterion(out.flatten(), batch.y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * batch.num_graphs
            
            avg_loss = total_loss / len(train_loader.dataset)
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch {epoch+1}, Loss: {avg_loss:.4f}')

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self.model.eval()
        data_list = self.featurizer.process_dataframe(X)
        loader = DataLoader(data_list, batch_size=self.params['batch_size'], shuffle=False)
        
        preds = []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.params['device'])
                out = self.model(batch.x, batch.edge_index, batch.batch)
                preds.append(out.cpu().numpy())
                
        return np.concatenate(preds).flatten()
