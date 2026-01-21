import pandas as pd
from rdkit import Chem

def canon(s):
    try:
        mol = Chem.MolFromSmiles(s)
        return Chem.MolToSmiles(mol, canonical=True) if mol else None
    except:
        return None

# Load ALL sources and check for Tm variance on SAME molecules
smp = pd.read_csv('data/raw/smiles_melting_point.csv', on_bad_lines='skip')
smp = smp.rename(columns={'Melting Point {measured, converted}': 'Tm'})[['SMILES', 'Tm']]
smp['Tm'] = pd.to_numeric(smp['Tm'], errors='coerce')
smp['can'] = smp['SMILES'].apply(canon)
smp['source'] = 'syracuse'

try:
    b1 = pd.read_excel('data/raw/BradleyMeltingPointDataset.xlsx')
    b1['Tm'] = b1['mpC'] + 273.15
    b1['SMILES'] = b1['smiles']
    b1['can'] = b1['SMILES'].apply(canon)
    b1['source'] = 'bradley'
    bradley = b1[['can', 'Tm', 'source']].dropna()
except:
    bradley = pd.DataFrame()

# Find molecules that appear in BOTH sources
all_data = pd.concat([smp[['can', 'Tm', 'source']], bradley])
all_data = all_data.dropna(subset=['can', 'Tm'])

# Group by canonical SMILES and check variance
grouped = all_data.groupby('can').agg({'Tm': ['mean', 'std', 'count']}).reset_index()
grouped.columns = ['can', 'tm_mean', 'tm_std', 'count']
multi_source = grouped[grouped['count'] > 1]
multi_source = multi_source.dropna(subset=['tm_std'])
multi_source = multi_source[multi_source['tm_std'] > 0]

print(f"Molecules with multiple measurements: {len(multi_source)}")
print(f"Mean Tm standard deviation: {multi_source['tm_std'].mean():.1f}K")
print(f"Max Tm standard deviation: {multi_source['tm_std'].max():.1f}K")
print()
print("Top 10 molecules with highest variance:")
print(multi_source.nlargest(10, 'tm_std')[['can', 'tm_mean', 'tm_std', 'count']].to_string())
