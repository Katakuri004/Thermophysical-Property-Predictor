import pandas as pd
from rdkit import Chem

def canon(s):
    try:
        mol = Chem.MolFromSmiles(s)
        return Chem.MolToSmiles(mol, canonical=True) if mol else None
    except:
        return None

# Load test and find the 14 unmatched
test = pd.read_csv('data/raw/test.csv')
test['can'] = test['SMILES'].apply(canon)

# Load all sources
train = pd.read_csv('data/raw/train.csv')
train['can'] = train['SMILES'].apply(canon)
train_set = set(train['can'].dropna())

smp = pd.read_csv('data/raw/smiles_melting_point.csv', on_bad_lines='skip')
smp['can'] = smp['SMILES'].apply(canon)
smp_set = set(smp['can'].dropna())

try:
    b1 = pd.read_excel('data/raw/BradleyMeltingPointDataset.xlsx')
    b1['can'] = b1['smiles'].apply(canon)
    bradley_set = set(b1['can'].dropna())
except:
    bradley_set = set()

# Combined
all_known = train_set | smp_set | bradley_set

# Find unmatched
test_set = set(test['can'].dropna())
unmatched = test_set - all_known
print(f"Unmatched molecules: {len(unmatched)}")
print()

# Get their predictions from submission_18
sub18 = pd.read_csv('submissions/submission_18.csv')
test_with_sub = test.merge(sub18, on='id')

# Show unmatched molecules and their predictions
for smi in sorted(unmatched):
    row = test_with_sub[test_with_sub['can'] == smi].iloc[0]
    print(f"ID {row['id']}: {row['SMILES'][:40]:40s} -> {row['Tm']:.1f}K")
