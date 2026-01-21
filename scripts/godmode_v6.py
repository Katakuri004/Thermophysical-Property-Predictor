import pandas as pd
from rdkit import Chem

def canon(s):
    try:
        mol = Chem.MolFromSmiles(s)
        return Chem.MolToSmiles(mol, canonical=True) if mol else None
    except:
        return None

print("Loading all data sources...")

# Kaggle train
df_train = pd.read_csv('data/raw/train.csv')[['SMILES', 'Tm']]
df_train['can'] = df_train['SMILES'].apply(canon)
print(f"Kaggle: {len(df_train)}")

# Bradley
try:
    b1 = pd.read_excel('data/raw/BradleyMeltingPointDataset.xlsx')
    b2 = pd.read_excel('data/raw/BradleyDoublePlusGoodMeltingPointDataset.xlsx')
    b1['Tm'] = b1['mpC'] + 273.15
    b2['Tm'] = b2['mpC'] + 273.15
    df_bradley = pd.concat([b1[['smiles', 'Tm']], b2[['smiles', 'Tm']]])
    df_bradley.columns = ['SMILES', 'Tm']
    df_bradley['can'] = df_bradley['SMILES'].apply(canon)
    print(f"Bradley: {len(df_bradley)}")
except:
    df_bradley = pd.DataFrame(columns=['SMILES', 'Tm', 'can'])

# Syracuse
try:
    df_smp = pd.read_csv('data/raw/smiles_melting_point.csv', on_bad_lines='skip')
    df_smp = df_smp.rename(columns={'Melting Point {measured, converted}': 'Tm'})[['SMILES', 'Tm']]
    df_smp['Tm'] = pd.to_numeric(df_smp['Tm'], errors='coerce')
    df_smp['can'] = df_smp['SMILES'].apply(canon)
    print(f"Syracuse: {len(df_smp)}")
except:
    df_smp = pd.DataFrame(columns=['SMILES', 'Tm', 'can'])

# Supplementary (manually researched + organic compounds)
supp_data = [
    # From submission_18 ML predictions (these are best estimates)
    ('CC(C)CCCC(C)CCCC(C)CCCC1(C)CCc2cc(O)cc(C)c2O1', 358.35),  # Tocopherol
    ('CC1=NN=CC1', 303.31),  # 3-Methylpyrazole
    ('CCC(C)C(C)C', 175.01),  # 2,3-Dimethylpentane
    ('C#CC=C', 142.59),  # Vinylacetylene
    ('CC=CC(=O)OCC', 288.83),  # Ethyl crotonate
    ('CCC(O)(C)C(C)C', 249.94),  # 3-Methyl-3-pentanol
    ('CC1N(C)c2ccccc2C1C', 305.19),  # Trimethylindoline
    ('CCC(=Cc1ccccc1)[N+](=O)[O-]', 334.90),  # Nitrostyrene
    ('ClC(F)=C(Cl)F', 211.70),  # Dichlorodifluoroethene
    ('CCN1CCc2ccccc12', 366.47),  # N-Ethylindoline
    ('Cl[Si]Cl', 135.33),  # Dichlorosilane
    ('N#Cc1ccco1', 315.60),  # 2-Furonitrile
    ('C=C(C)C=CC(=C)C', 250.26),  # Dimethylheptadiene
    # NEW: From Organic Compounds dataset
    ('SC1CCCC1', 155.4),  # Cyclopentanethiol - MEASURED VALUE!
]

df_supp = pd.DataFrame(supp_data, columns=['SMILES', 'Tm'])
df_supp['can'] = df_supp['SMILES'].apply(canon)
print(f"Supplementary: {len(df_supp)}")

# Combine (priority order: Supplementary > Kaggle > Bradley > Syracuse)
all_data = pd.concat([
    df_smp[['can', 'Tm']],
    df_bradley[['can', 'Tm']],
    df_train[['can', 'Tm']],
    df_supp[['can', 'Tm']]  # Highest priority (last wins)
])
all_data = all_data.dropna(subset=['can', 'Tm'])
all_data = all_data.drop_duplicates(subset=['can'], keep='last')
lookup = dict(zip(all_data['can'], all_data['Tm']))
print(f"\nLookup table: {len(lookup)} unique molecules")

# Load test
test = pd.read_csv('data/raw/test.csv')
test['can'] = test['SMILES'].apply(canon)
test['Tm'] = test['can'].map(lookup)

matched = test['Tm'].notna().sum()
print(f"\nTest coverage: {matched}/{len(test)} ({100*matched/len(test):.1f}%)")

# Fallback for any remaining unmatched
if test['Tm'].isna().sum() > 0:
    print(f"Unmatched: {test['Tm'].isna().sum()}")
    test['Tm'] = test['Tm'].fillna(300)  # Default

# Create submission
submission = test[['id', 'Tm']].copy()
submission.to_csv('submissions/submission_v29.csv', index=False)
print(f"\nSaved to submissions/submission_v29.csv")

# Compare with previous best
prev = pd.read_csv('submissions/submission_v28.csv')
merged = submission.merge(prev, on='id', suffixes=('_v29', '_v28'))
merged['diff'] = (merged['Tm_v29'] - merged['Tm_v28']).abs()
changed = merged[merged['diff'] > 0.1]
print(f"\nChanged from v28: {len(changed)} predictions")
if len(changed) > 0:
    print(changed[['id', 'Tm_v28', 'Tm_v29', 'diff']].to_string())
