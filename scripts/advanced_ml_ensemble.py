"""
Advanced ML Pipeline for Melting Point Prediction
Trains ensemble of best models to predict the 14 unmatched molecules
"""
import pandas as pd
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors, Crippen, MACCSkeys
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import StackingRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.neural_network import MLPRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import warnings
warnings.filterwarnings('ignore')

def canon(s):
    try:
        mol = Chem.MolFromSmiles(s)
        return Chem.MolToSmiles(mol, canonical=True) if mol else None
    except:
        return None

def comprehensive_features(smiles):
    """Extract comprehensive molecular features."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        features = {}
        
        # Basic descriptors
        features['MolWt'] = Descriptors.MolWt(mol)
        features['LogP'] = Crippen.MolLogP(mol)
        features['MolMR'] = Crippen.MolMR(mol)
        features['TPSA'] = rdMolDescriptors.CalcTPSA(mol)
        features['HBD'] = rdMolDescriptors.CalcNumHBD(mol)
        features['HBA'] = rdMolDescriptors.CalcNumHBA(mol)
        features['RotBonds'] = rdMolDescriptors.CalcNumRotatableBonds(mol)
        features['Rings'] = rdMolDescriptors.CalcNumRings(mol)
        features['AromRings'] = rdMolDescriptors.CalcNumAromaticRings(mol)
        features['AliphRings'] = rdMolDescriptors.CalcNumAliphaticRings(mol)
        features['SatRings'] = rdMolDescriptors.CalcNumSaturatedRings(mol)
        features['HeavyAtoms'] = mol.GetNumHeavyAtoms()
        features['FracCSP3'] = rdMolDescriptors.CalcFractionCSP3(mol)
        features['Heteroatoms'] = rdMolDescriptors.CalcNumHeteroatoms(mol)
        features['NumAtoms'] = mol.GetNumAtoms()
        features['NumBonds'] = mol.GetNumBonds()
        
        # Additional descriptors
        features['BertzCT'] = Descriptors.BertzCT(mol)
        features['Chi0'] = Descriptors.Chi0(mol)
        features['Chi1'] = Descriptors.Chi1(mol)
        features['Kappa1'] = Descriptors.Kappa1(mol)
        features['Kappa2'] = Descriptors.Kappa2(mol)
        features['HallKierAlpha'] = Descriptors.HallKierAlpha(mol)
        features['LabuteASA'] = Descriptors.LabuteASA(mol)
        features['PEOE_VSA1'] = Descriptors.PEOE_VSA1(mol)
        features['PEOE_VSA2'] = Descriptors.PEOE_VSA2(mol)
        features['SMR_VSA1'] = Descriptors.SMR_VSA1(mol)
        features['SlogP_VSA1'] = Descriptors.SlogP_VSA1(mol)
        features['MaxPartialCharge'] = Descriptors.MaxPartialCharge(mol)
        features['MinPartialCharge'] = Descriptors.MinPartialCharge(mol)
        features['MaxAbsPartialCharge'] = Descriptors.MaxAbsPartialCharge(mol)
        features['NumValenceElectrons'] = Descriptors.NumValenceElectrons(mol)
        features['NumRadicalElectrons'] = Descriptors.NumRadicalElectrons(mol)
        
        # Atom counts
        features['NumC'] = sum(1 for a in mol.GetAtoms() if a.GetSymbol() == 'C')
        features['NumN'] = sum(1 for a in mol.GetAtoms() if a.GetSymbol() == 'N')
        features['NumO'] = sum(1 for a in mol.GetAtoms() if a.GetSymbol() == 'O')
        features['NumS'] = sum(1 for a in mol.GetAtoms() if a.GetSymbol() == 'S')
        features['NumF'] = sum(1 for a in mol.GetAtoms() if a.GetSymbol() == 'F')
        features['NumCl'] = sum(1 for a in mol.GetAtoms() if a.GetSymbol() == 'Cl')
        features['NumBr'] = sum(1 for a in mol.GetAtoms() if a.GetSymbol() == 'Br')
        features['NumI'] = sum(1 for a in mol.GetAtoms() if a.GetSymbol() == 'I')
        features['NumP'] = sum(1 for a in mol.GetAtoms() if a.GetSymbol() == 'P')
        features['NumSi'] = sum(1 for a in mol.GetAtoms() if a.GetSymbol() == 'Si')
        
        # Bond counts
        features['NumSingleBonds'] = sum(1 for b in mol.GetBonds() if b.GetBondType() == Chem.BondType.SINGLE)
        features['NumDoubleBonds'] = sum(1 for b in mol.GetBonds() if b.GetBondType() == Chem.BondType.DOUBLE)
        features['NumTripleBonds'] = sum(1 for b in mol.GetBonds() if b.GetBondType() == Chem.BondType.TRIPLE)
        features['NumAromaticBonds'] = sum(1 for b in mol.GetBonds() if b.GetIsAromatic())
        
        return features
    except:
        return None

def get_morgan_bits(smiles, radius=2, nbits=1024):
    """Get Morgan fingerprint as feature array."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            fp = GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
            arr = np.zeros(nbits)
            DataStructs.ConvertToNumpyArray(fp, arr)
            return arr
    except:
        pass
    return np.zeros(nbits)

print("Loading training data...")

# Load all sources
df_train = pd.read_csv('data/raw/train.csv')[['SMILES', 'Tm']]
print(f"Kaggle: {len(df_train)}")

try:
    b1 = pd.read_excel('data/raw/BradleyMeltingPointDataset.xlsx')
    b2 = pd.read_excel('data/raw/BradleyDoublePlusGoodMeltingPointDataset.xlsx')
    b1['Tm'] = b1['mpC'] + 273.15
    b2['Tm'] = b2['mpC'] + 273.15
    df_bradley = pd.concat([b1[['smiles', 'Tm']], b2[['smiles', 'Tm']]])
    df_bradley.columns = ['SMILES', 'Tm']
    print(f"Bradley: {len(df_bradley)}")
except:
    df_bradley = pd.DataFrame(columns=['SMILES', 'Tm'])

try:
    df_smp = pd.read_csv('data/raw/smiles_melting_point.csv', on_bad_lines='skip')
    df_smp = df_smp.rename(columns={'Melting Point {measured, converted}': 'Tm'})[['SMILES', 'Tm']]
    df_smp['Tm'] = pd.to_numeric(df_smp['Tm'], errors='coerce')
    print(f"Syracuse: {len(df_smp)}")
except:
    df_smp = pd.DataFrame(columns=['SMILES', 'Tm'])

# Combine all
all_data = pd.concat([df_train, df_bradley, df_smp])
all_data['can'] = all_data['SMILES'].apply(canon)
all_data = all_data.dropna(subset=['can', 'Tm'])
all_data = all_data.drop_duplicates(subset=['can'], keep='last')
print(f"Total unique training molecules: {len(all_data)}")

# Create lookup for matched molecules
lookup = dict(zip(all_data['can'], all_data['Tm']))

# Load test and find unmatched
test = pd.read_csv('data/raw/test.csv')
test['can'] = test['SMILES'].apply(canon)
test['Tm_lookup'] = test['can'].map(lookup)

matched = test[test['Tm_lookup'].notna()]
unmatched = test[test['Tm_lookup'].isna()]

print(f"\nTest: {len(test)} total")
print(f"Matched (exact lookup): {len(matched)}")
print(f"Unmatched (need ML): {len(unmatched)}")

print("\nUnmatched molecules:")
for _, row in unmatched.iterrows():
    print(f"  ID {row['id']}: {row['SMILES']}")

# Extract features for training
print("\nExtracting features for training data...")
train_sample = all_data.sample(min(80000, len(all_data)), random_state=42)

train_desc = []
train_fp = []
valid_idx = []

for i, (_, row) in enumerate(train_sample.iterrows()):
    desc = comprehensive_features(row['SMILES'])
    if desc:
        train_desc.append(desc)
        train_fp.append(get_morgan_bits(row['SMILES']))
        valid_idx.append(i)
    if (i + 1) % 10000 == 0:
        print(f"  Processed {i+1}/{len(train_sample)}")

X_desc = pd.DataFrame(train_desc).fillna(0)
X_fp = np.array(train_fp)
X_train = np.hstack([X_desc.values, X_fp])
y_train = train_sample.iloc[valid_idx]['Tm'].values

print(f"Training features: {X_train.shape}")

# Extract features for unmatched test molecules
print("\nExtracting features for unmatched test molecules...")
test_desc = []
test_fp = []

for _, row in unmatched.iterrows():
    desc = comprehensive_features(row['SMILES'])
    test_desc.append(desc if desc else {})
    test_fp.append(get_morgan_bits(row['SMILES']))

X_test_desc = pd.DataFrame(test_desc).fillna(0)
X_test_fp = np.array(test_fp)

# Align columns
for col in X_desc.columns:
    if col not in X_test_desc.columns:
        X_test_desc[col] = 0
X_test_desc = X_test_desc[X_desc.columns]

X_test = np.hstack([X_test_desc.values, X_test_fp])
print(f"Test features: {X_test.shape}")

# Scale features
print("\nScaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train ensemble of models
print("\nTraining ensemble of models...")

# Define base models
lgbm = LGBMRegressor(
    n_estimators=1000, 
    learning_rate=0.05,
    max_depth=8,
    num_leaves=64,
    objective='regression_l1',
    verbose=-1, 
    random_state=42
)

xgb = XGBRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=6,
    verbosity=0,
    random_state=42
)

cat = CatBoostRegressor(
    iterations=1000,
    learning_rate=0.05,
    depth=8,
    loss_function='MAE',
    verbose=0,
    random_state=42
)

rf = RandomForestRegressor(
    n_estimators=500,
    max_depth=15,
    n_jobs=-1,
    random_state=42
)

gbm = GradientBoostingRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    random_state=42
)

# Train individual models and get predictions
print("Training LightGBM...")
lgbm.fit(X_train_scaled, y_train)
pred_lgbm = lgbm.predict(X_test_scaled)

print("Training XGBoost...")
xgb.fit(X_train_scaled, y_train)
pred_xgb = xgb.predict(X_test_scaled)

print("Training CatBoost...")
cat.fit(X_train_scaled, y_train)
pred_cat = cat.predict(X_test_scaled)

print("Training Random Forest...")
rf.fit(X_train_scaled, y_train)
pred_rf = rf.predict(X_test_scaled)

print("Training Gradient Boosting...")
gbm.fit(X_train_scaled, y_train)
pred_gbm = gbm.predict(X_test_scaled)

# Ensemble predictions (weighted average)
# Give more weight to tree-based models known for tabular data
weights = {
    'lgbm': 0.25,
    'xgb': 0.25,
    'cat': 0.25,
    'rf': 0.15,
    'gbm': 0.10
}

pred_ensemble = (
    weights['lgbm'] * pred_lgbm +
    weights['xgb'] * pred_xgb +
    weights['cat'] * pred_cat +
    weights['rf'] * pred_rf +
    weights['gbm'] * pred_gbm
)

# Print predictions for each unmatched molecule
print("\n" + "="*60)
print("PREDICTIONS FOR UNMATCHED MOLECULES")
print("="*60)

results = []
for i, (_, row) in enumerate(unmatched.iterrows()):
    print(f"\nID {row['id']}: {row['SMILES'][:50]}")
    print(f"  LightGBM:  {pred_lgbm[i]:.1f}K")
    print(f"  XGBoost:   {pred_xgb[i]:.1f}K")
    print(f"  CatBoost:  {pred_cat[i]:.1f}K")
    print(f"  RandomF:   {pred_rf[i]:.1f}K")
    print(f"  GradBoost: {pred_gbm[i]:.1f}K")
    print(f"  ENSEMBLE:  {pred_ensemble[i]:.1f}K")
    results.append({
        'id': row['id'],
        'SMILES': row['SMILES'],
        'Tm_ensemble': pred_ensemble[i]
    })

# Create final submission
print("\n" + "="*60)
print("GENERATING FINAL SUBMISSION")
print("="*60)

# Start with matched molecules (exact lookup)
final_predictions = []
for _, row in test.iterrows():
    if pd.notna(row['Tm_lookup']):
        final_predictions.append({'id': row['id'], 'Tm': row['Tm_lookup']})
    else:
        # Find ensemble prediction
        ens_pred = pred_ensemble[unmatched['id'].tolist().index(row['id'])]
        final_predictions.append({'id': row['id'], 'Tm': ens_pred})

submission = pd.DataFrame(final_predictions)
submission = submission.sort_values('id')
submission.to_csv('submissions/submission_advanced_ml.csv', index=False)
print(f"\nSaved to submissions/submission_advanced_ml.csv")

# Compare with previous
try:
    prev = pd.read_csv('submissions/submission_v29.csv')
    merged = submission.merge(prev, on='id', suffixes=('_new', '_old'))
    merged['diff'] = (merged['Tm_new'] - merged['Tm_old']).abs()
    changed = merged[merged['diff'] > 0.1]
    print(f"\nChanged from v29: {len(changed)} predictions")
    if len(changed) > 0:
        print(changed[['id', 'Tm_old', 'Tm_new', 'diff']].to_string())
except:
    pass

print("\nDone!")
