import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, MACCSkeys
from typing import List, Tuple, Dict, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MolecularFeaturizer:
    """
    A class to generate molecular descriptors and fingerprints from SMILES strings.
    """
    
    def __init__(self, smiles_col: str = 'SMILES'):
        """
        Initialize the featurizer.
        
        Args:
            smiles_col (str): The name of the column containing SMILES strings.
        """
        self.smiles_col = smiles_col
        self.descriptor_funcs = Descriptors.descList
        
    def _get_mol(self, smiles: str):
        """Helper to get RDKit Mol object from SMILES."""
        try:
            mol = Chem.MolFromSmiles(smiles)
            return mol
        except Exception:
            return None

    def generate_descriptors(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate RDKit physicochemical descriptors.
        
        Args:
            df (pd.DataFrame): Input dataframe.
            
        Returns:
            pd.DataFrame: Dataframe with added descriptor columns.
        """
        logger.info(f"Generating {len(self.descriptor_funcs)} RDKit descriptors for {len(df)} molecules...")
        
        # Pre-calculate Mols to avoid repetitive parsing
        mols = df[self.smiles_col].apply(self._get_mol)
        
        # Valid mols mask
        valid_mask = mols.notna()
        valid_mols = mols[valid_mask]
        
        if not valid_mask.all():
            logger.warning(f"Failed to parse {(~valid_mask).sum()} SMILES strings.")
            
        # Generate descriptors
        desc_data = {}
        for desc_name, func in self.descriptor_funcs:
            desc_values = [func(mol) if mol else np.nan for mol in mols]
            desc_data[desc_name] = desc_values
            
        desc_df = pd.DataFrame(desc_data, index=df.index)
        
        # Concatenate with original dataframe
        result_df = pd.concat([df, desc_df], axis=1)
        return result_df

    def generate_morgan_fingerprints(self, df: pd.DataFrame, radius: int = 2, n_bits: int = 2048, prefix: str = 'Morgan') -> pd.DataFrame:
        """
        Generate Morgan (ECFP) fingerprints.
        
        Args:
            df (pd.DataFrame): Input dataframe.
            radius (int): Radius for Morgan fingerprints.
            n_bits (int): Number of bits.
            prefix (str): Prefix for column names.
            
        Returns:
            pd.DataFrame: Dataframe with fingerprint columns.
        """
        logger.info(f"Generating Morgan fingerprints (r={radius}, bits={n_bits})...")
        
        mols = df[self.smiles_col].apply(self._get_mol)
        
        fp_list = []
        for mol in mols:
            if mol:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
                arr = np.zeros((1,))
                AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
                fp_list.append(arr)
            else:
                fp_list.append(np.zeros(n_bits) * np.nan)
                
        fp_matrix = np.vstack(fp_list)
        col_names = [f'{prefix}_{i}' for i in range(n_bits)]
        
        fp_df = pd.DataFrame(fp_matrix, columns=col_names, index=df.index)
        
        return pd.concat([df, fp_df], axis=1)

    def generate_maccs_keys(self, df: pd.DataFrame, prefix: str = 'MACCS') -> pd.DataFrame:
        """
        Generate MACCS Keys fingerprints.
        
        Args:
            df (pd.DataFrame): Input dataframe.
            prefix (str): Prefix for column names.
            
        Returns:
            pd.DataFrame: Dataframe with MACCS columns.
        """
        logger.info("Generating MACCS keys...")
        
        mols = df[self.smiles_col].apply(self._get_mol)
        
        fp_list = []
        for mol in mols:
            if mol:
                fp = MACCSkeys.GenMACCSKeys(mol)
                # MACCS keys are length 167 (index 0 is usually 0)
                arr = np.zeros((1,))
                AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
                fp_list.append(arr)
            else:
                fp_list.append(np.zeros(167) * np.nan)
                
        fp_matrix = np.vstack(fp_list)
        # MACCS keys usually have 166 bits of interest, but RDKit returns 167. 
        # We'll stick to what RDKit returns.
        col_names = [f'{prefix}_{i}' for i in range(167)]
        
        fp_df = pd.DataFrame(fp_matrix, columns=col_names, index=df.index)
        
        return pd.concat([df, fp_df], axis=1)
