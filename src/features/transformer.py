import torch
from transformers import AutoTokenizer, AutoModel
from typing import List, Union
import pandas as pd
import numpy as np
from tqdm import tqdm

class ChemBERTaFeaturizer:
    def __init__(self, model_name: str = "seyonec/ChemBERTa-zinc-base-v1", device: str = "cuda"):
        """
        Initializes the ChemBERTa feature extractor.
        
        Args:
            model_name (str): HuggingFace model identifier.
            device (str): 'cuda' or 'cpu'.
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        print(f"Loading ChemBERTa model: {model_name} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def transform(self, smiles_list: List[str], batch_size: int = 32) -> pd.DataFrame:
        """
        Transforms a list of SMILES strings into ChemBERTa embeddings.
        
        Args:
            smiles_list (List[str]): List of SMILES strings.
            batch_size (int): Batch size for inference.
            
        Returns:
            pd.DataFrame: DataFrame containing the embeddings columns.
        """
        all_embeddings = []
        
        # Process in batches
        for i in tqdm(range(0, len(smiles_list), batch_size), desc="Generating Embeddings"):
            batch_smiles = smiles_list[i : i + batch_size]
            
            # Tokenize
            inputs = self.tokenizer(
                batch_smiles, 
                padding=True, 
                truncation=True, 
                return_tensors="pt", 
                max_length=512
            )
            
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Inference
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Use the embedding of the [CLS] token (first token)
            # Shape: [batch_size, hidden_size]
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            all_embeddings.append(cls_embeddings)
            
        # Concatenate all batches
        embeddings_matrix = np.vstack(all_embeddings)
        
        # Create column names
        cols = [f"ChemBERTa_{i}" for i in range(embeddings_matrix.shape[1])]
        
        return pd.DataFrame(embeddings_matrix, columns=cols)

    def calculate_transformer_features(self, df: pd.DataFrame, smiles_col: str = 'SMILES') -> pd.DataFrame:
        """
        Takes a DataFrame with a SMILES column, generates embeddings, and returns the enhanced DataFrame.
        """
        smiles = df[smiles_col].tolist()
        emb_df = self.transform(smiles)
        
        # Concatenate original (typically just ID/Target) with embeddings
        # We assume the index matches
        result_df = pd.concat([df.reset_index(drop=True), emb_df], axis=1)
        return result_df
