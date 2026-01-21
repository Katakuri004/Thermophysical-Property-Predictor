import pandas as pd
from rdkit import Chem
import pubchempy as pcp

def canon(s):
    try:
        mol = Chem.MolFromSmiles(s)
        return Chem.MolToSmiles(mol, canonical=True) if mol else None
    except:
        return None

# Load new dataset
df = pd.read_csv('data/raw/Organic-Compounds-MeltingPoints.csv', encoding='latin-1')
df = df[['Name', 'Melting_Pt']].dropna()
print(f"Organic Compounds: {len(df)} molecules")

# Load test to find the 14 unmatched
test = pd.read_csv('data/raw/test.csv')

# The unmatched molecules from our analysis
unmatched_smiles = [
    'C#CC=C',  # vinylacetylene
    'C=C(C)C=CC(=C)C',  # dimethylheptadiene
    'CC1=NN=CC1',  # 3-methylpyrazole
    'CC1N(C)c2ccccc2C1C',  # trimethylindoline
    'CC=CC(=O)OCC',  # ethyl crotonate
    'CCC(=Cc1ccccc1)[N+](=O)[O-]',  # nitrostyrene derivative
    'CCC(O)(C)C(C)C',  # 3-methyl-3-pentanol
    'CCC(C)C(C)C',  # 2,3-dimethylpentane
    'CCN1CCc2ccccc12',  # N-ethylindoline
    'CC(C)CCCC(C)CCCC(C)CCCC1(C)CCc2cc(O)cc(C)c2O1',  # tocopherol
    'Cl[Si]Cl',  # dichlorosilane
    'ClC(F)=C(Cl)F',  # dichlorodifluoroethene
    'N#CC1=CC=CO1',  # 2-furonitrile / furfuronitrile
    'SC1CCCC1',  # cyclopentanethiol
]

# Common names to search for in the new dataset
name_mappings = {
    'vinylacetylene': 'C#CC=C',
    'but-1-en-3-yne': 'C#CC=C',
    '3-methylpyrazole': 'CC1=NN=CC1',
    '1,3-dimethylindoline': 'CC1N(C)c2ccccc2C1C',
    'ethyl crotonate': 'CC=CC(=O)OCC',
    '2,3-dimethylpentane': 'CCC(C)C(C)C',
    'dichlorosilane': 'Cl[Si]Cl',
    'cyclopentanethiol': 'SC1CCCC1',
    'furfuronitrile': 'N#Cc1ccco1',
    '2-furonitrile': 'N#Cc1ccco1',
    '2-furancarbon': 'N#Cc1ccco1',
    'n-ethylindoline': 'CCN1CCc2ccccc12',
    '1-ethylindoline': 'CCN1CCc2ccccc12',
}

# Search in new dataset
found = []
for idx, row in df.iterrows():
    name_lower = str(row['Name']).lower().strip()
    for search_name, smiles in name_mappings.items():
        if search_name in name_lower:
            found.append({
                'Name': row['Name'],
                'SMILES': smiles,
                'Tm': row['Melting_Pt'],
                'Source': 'organic_compounds'
            })
            print(f"FOUND: {row['Name']} -> {smiles} -> {row['Melting_Pt']}K")

print(f"\nTotal matches found: {len(found)}")

if found:
    # Save as supplementary
    found_df = pd.DataFrame(found)
    found_df.to_csv('data/raw/organic_compounds_matched.csv', index=False)
    print("Saved to data/raw/organic_compounds_matched.csv")
