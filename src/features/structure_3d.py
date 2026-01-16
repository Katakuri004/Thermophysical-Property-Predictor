import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors3D
from typing import List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Conformer3DFeaturizer:
    """
    Generates 3D conformers and calculates 3D descriptors.
    """
    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed

    def _generate_conformer(self, mol: Chem.Mol) -> Optional[Chem.Mol]:
        """
        Embed molecule in 3D space using ETKDG method.
        """
        mol_3d = Chem.AddHs(mol) # Add hydrogens for correct 3D geometry
        params = AllChem.ETKDGv3()
        params.randomSeed = self.random_seed
        
        res = AllChem.EmbedMolecule(mol_3d, params)
        if res == 0:
            # Optimize geometry
            AllChem.MMFFOptimizeMolecule(mol_3d)
            return mol_3d
        else:
            # Embedding failed
            return None

    def calculate_3d_descriptors(self, df: pd.DataFrame, smiles_col: str = 'SMILES') -> pd.DataFrame:
        """
        Calculate 3D descriptors for a dataframe of SMILES.
        """
        logger.info(f"Generating 3D conformers and descriptors for {len(df)} molecules...")
        
        descriptors = []
        for smiles in df[smiles_col]:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                mol_3d = self._generate_conformer(mol)
                if mol_3d:
                    # Calculate descriptors
                    try:
                        desc = {
                            'Asphericity': Descriptors3D.Asphericity(mol_3d),
                            'Eccentricity': Descriptors3D.Eccentricity(mol_3d),
                            'InertialShapeFactor': Descriptors3D.InertialShapeFactor(mol_3d),
                            'NPR1': Descriptors3D.NPR1(mol_3d),
                            'NPR2': Descriptors3D.NPR2(mol_3d),
                            'PMI1': Descriptors3D.PMI1(mol_3d),
                            'PMI2': Descriptors3D.PMI2(mol_3d),
                            'PMI3': Descriptors3D.PMI3(mol_3d),
                            'RadiusOfGyration': Descriptors3D.RadiusOfGyration(mol_3d),
                            'SpherocityIndex': Descriptors3D.SpherocityIndex(mol_3d)
                        }
                    except Exception as e:
                        # Sometimes calculation fails on weird geometries
                        desc = {k: np.nan for k in ['Asphericity', 'Eccentricity', 'InertialShapeFactor', 'NPR1', 'NPR2', 'PMI1', 'PMI2', 'PMI3', 'RadiusOfGyration', 'SpherocityIndex']}
                else:
                     desc = {k: np.nan for k in ['Asphericity', 'Eccentricity', 'InertialShapeFactor', 'NPR1', 'NPR2', 'PMI1', 'PMI2', 'PMI3', 'RadiusOfGyration', 'SpherocityIndex']}
            else:
                 desc = {k: np.nan for k in ['Asphericity', 'Eccentricity', 'InertialShapeFactor', 'NPR1', 'NPR2', 'PMI1', 'PMI2', 'PMI3', 'RadiusOfGyration', 'SpherocityIndex']}
            
            descriptors.append(desc)
            
        desc_df = pd.DataFrame(descriptors, index=df.index)
        
        # Add '3D_' prefix
        desc_df.columns = ['3D_' + c for c in desc_df.columns]
        
        return pd.concat([df, desc_df], axis=1)
