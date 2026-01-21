"""
PubChem Melting Point Data Fetcher

This script downloads melting point data from PubChem for integration
with HierarchicalMP predictor.

Usage:
    python scripts/fetch_pubchem_data.py --output data/raw/pubchem_melting_points.csv
"""

import argparse
import requests
import pandas as pd
import numpy as np
from pathlib import Path
import time
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from rdkit import Chem
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False
    print("Warning: RDKit not available. SMILES canonicalization will be skipped.")


def canonicalize_smiles(smiles):
    """Convert SMILES to canonical form."""
    if not HAS_RDKIT:
        return smiles
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            return Chem.MolToSmiles(mol, canonical=True)
    except:
        pass
    return None


def fetch_pubchem_property_cids(property_name="Melting Point", max_cids=100000):
    """
    Fetch CIDs of compounds that have a given property in PubChem.
    
    Note: This is a simplified approach. Full extraction requires
    downloading PubChem bulk data files.
    """
    print(f"Fetching compound CIDs with property: {property_name}")
    
    # PubChem search URL for compounds with melting point
    # This is limited - for full data, use bulk download from FTP
    base_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
    
    # Get list of CIDs (limited by API)
    cids = []
    
    # Use classification/property approach
    # Note: Full implementation requires downloading from:
    # https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/Extras/
    
    return cids


def fetch_compound_batch(cids, batch_size=100):
    """
    Fetch SMILES and properties for a batch of CIDs.
    """
    results = []
    
    for i in range(0, len(cids), batch_size):
        batch = cids[i:i+batch_size]
        cid_str = ','.join(map(str, batch))
        
        # Fetch canonical SMILES
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid_str}/property/CanonicalSMILES/JSON"
        
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                for prop in data.get('PropertyTable', {}).get('Properties', []):
                    results.append({
                        'CID': prop.get('CID'),
                        'SMILES': prop.get('CanonicalSMILES')
                    })
        except Exception as e:
            print(f"Error fetching batch: {e}")
        
        time.sleep(0.2)  # Rate limiting
    
    return results


def fetch_melting_point_for_cid(cid):
    """
    Fetch melting point for a single CID from PubChem.
    
    Returns: (CID, SMILES, melting_point_kelvin) or None
    """
    try:
        # Get compound properties
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{cid}/JSON"
        resp = requests.get(url, timeout=10)
        
        if resp.status_code != 200:
            return None
        
        data = resp.json()
        
        # Parse response to find melting point
        smiles = None
        melting_point = None
        
        # Navigate the JSON structure
        record = data.get('Record', {})
        
        for section in record.get('Section', []):
            if section.get('TOCHeading') == 'Names and Identifiers':
                for subsection in section.get('Section', []):
                    if subsection.get('TOCHeading') == 'Computed Descriptors':
                        for item in subsection.get('Section', []):
                            if item.get('TOCHeading') == 'Canonical SMILES':
                                info = item.get('Information', [{}])[0]
                                smiles = info.get('Value', {}).get('StringWithMarkup', [{}])[0].get('String')
            
            elif section.get('TOCHeading') == 'Chemical and Physical Properties':
                for subsection in section.get('Section', []):
                    if subsection.get('TOCHeading') == 'Experimental Properties':
                        for item in subsection.get('Section', []):
                            if 'Melting Point' in item.get('TOCHeading', ''):
                                for info in item.get('Information', []):
                                    value = info.get('Value', {})
                                    if 'Number' in value:
                                        mp_value = value['Number'][0]
                                        unit = value.get('Unit', 'K')
                                        
                                        # Convert to Kelvin
                                        if unit in ['°C', 'C', 'Celsius']:
                                            melting_point = mp_value + 273.15
                                        elif unit in ['°F', 'F', 'Fahrenheit']:
                                            melting_point = (mp_value - 32) * 5/9 + 273.15
                                        else:  # Assume Kelvin
                                            melting_point = mp_value
                                        break
        
        if smiles and melting_point:
            return (cid, smiles, melting_point)
    
    except Exception as e:
        pass
    
    return None


def download_pubchem_bulk_melting_points(output_path, max_compounds=50000):
    """
    Alternative approach: Download from known melting point datasets on Kaggle
    that were originally sourced from PubChem.
    """
    print("Note: Full PubChem extraction requires bulk data download.")
    print("For comprehensive coverage, download from:")
    print("  - https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/Extras/")
    print("  - https://www.kaggle.com/datasets/ (search 'melting point')")
    print()
    
    # Provide some sample CIDs with known melting points for testing
    sample_compounds = [
        # (SMILES, Tm in Kelvin, Name)
        ("CCO", 159.05, "Ethanol"),
        ("CC(=O)O", 289.75, "Acetic acid"),
        ("c1ccccc1", 278.65, "Benzene"),
        ("CC(C)O", 184.65, "Isopropanol"),
        ("CCCCCC", 178.15, "Hexane"),
        ("c1ccc2ccccc2c1", 353.45, "Naphthalene"),
        ("CC(=O)OC1=CC=CC=C1C(=O)O", 408.15, "Aspirin"),
        ("CN1C=NC2=C1C(=O)N(C(=O)N2C)C", 508.15, "Caffeine"),
    ]
    
    df = pd.DataFrame(sample_compounds, columns=['SMILES', 'Tm', 'Name'])
    
    if HAS_RDKIT:
        df['canonical'] = df['SMILES'].apply(canonicalize_smiles)
    else:
        df['canonical'] = df['SMILES']
    
    return df


def main():
    parser = argparse.ArgumentParser(description='Fetch PubChem melting point data')
    parser.add_argument('--output', type=str, default='data/raw/pubchem_melting_points.csv',
                        help='Output CSV path')
    parser.add_argument('--max-compounds', type=int, default=10000,
                        help='Maximum compounds to fetch')
    parser.add_argument('--sample', action='store_true',
                        help='Create sample dataset only (fast)')
    args = parser.parse_args()
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if args.sample:
        print("Creating sample PubChem dataset...")
        df = download_pubchem_bulk_melting_points(output_path, args.max_compounds)
    else:
        print("Full PubChem download not implemented in this script.")
        print("For production use, download bulk data from PubChem FTP site.")
        print()
        print("Creating sample dataset instead...")
        df = download_pubchem_bulk_melting_points(output_path, args.max_compounds)
    
    # Save
    df.to_csv(output_path, index=False)
    print(f"\nSaved {len(df)} compounds to {output_path}")
    
    # Show summary
    print(f"\nDataset summary:")
    print(f"  Total compounds: {len(df)}")
    print(f"  Tm range: {df['Tm'].min():.1f} - {df['Tm'].max():.1f} K")
    print(f"  Mean Tm: {df['Tm'].mean():.1f} K")


if __name__ == '__main__':
    main()
