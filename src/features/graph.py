import torch
import numpy as np
from rdkit import Chem
from torch_geometric.data import Data
from typing import List, Optional

class GraphFeaturizer:
    """
    Converts SMILES strings to PyTorch Geometric Data objects.
    """
    def __init__(self):
        self.atom_types = [1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53] # Common organic elements (H, B, C, N, O, F, Si, P, S, Cl, Br, I)

    def _get_atom_features(self, atom):
        """
        Get atom features:
        - Atomic Number (mapped to index in self.atom_types, else 'other')
        - Degree
        - Formal Charge
        - Hybridization
        - Aromaticity
        """
        # Atomic Number Encoding
        atomic_num = atom.GetAtomicNum()
        if atomic_num in self.atom_types:
            type_idx = self.atom_types.index(atomic_num)
        else:
            type_idx = len(self.atom_types) # 'Other' category
            
        return [
            type_idx,
            atom.GetTotalDegree(),
            atom.GetFormalCharge(),
            int(atom.GetHybridization()),
            int(atom.GetIsAromatic())
        ]

    def smilies_to_graph(self, smiles: str) -> Optional[Data]:
        """
        Convert SMILES to Graph Data object.
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        # Node Features
        node_feats = []
        for atom in mol.GetAtoms():
            node_feats.append(self._get_atom_features(atom))
            
        x = torch.tensor(node_feats, dtype=torch.float)
        
        # Edge Index
        edge_indices = []
        edge_attrs = []
        
        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            
            # Bidirectional edges for undirected graph
            edge_indices.append([start, end])
            edge_indices.append([end, start])
            
            # Simple bond type feature (Single=1, Double=2, Triple=3, Aromatic=1.5)
            btype = bond.GetBondTypeAsDouble()
            edge_attrs.append([btype])
            edge_attrs.append([btype])
            
        if len(edge_indices) == 0:
            # Handle single ion/atom case
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 1), dtype=torch.float)
        else:
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
            
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    def process_dataframe(self, df, smiles_col='SMILES', target_col='Tm'):
        """
        Convert a DataFrame to a list of Data objects.
        """
        data_list = []
        for idx, row in df.iterrows():
            graph = self.smilies_to_graph(row[smiles_col])
            if graph:
                if target_col in row:
                    graph.y = torch.tensor([row[target_col]], dtype=torch.float)
                data_list.append(graph)
        return data_list
