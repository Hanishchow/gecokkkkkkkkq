import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
import warnings
warnings.filterwarnings('ignore')

# Load data to analyze patterns
df = pd.read_csv('.cache/geock_autoresearch/LP_PDBBind.csv')

# Parse Kd values
def parse_kd(val):
    if pd.isna(val):
        return np.nan
    val = str(val).strip()
    try:
        if '=' in val:
            parts = val.split('=')
            num = float(parts[1].replace('nM','').replace('uM','*1000').replace('mM','*1000000').replace('M','*1000000000'))
            return eval(parts[1]) if '*' in parts[1] else num
        return float(val)
    except:
        return np.nan

df['kd_nM'] = df['value'].apply(parse_kd)
df['pKD'] = -np.log10(df['kd_nM'].values * 1e-9 + 1e-15)

# Extract molecular properties from SMILES
def calc_mol_props(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return [np.nan]*8
        return [
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            Descriptors.TPSA(mol),
            Descriptors.NumHDonors(mol),
            Descriptors.NumHAcceptors(mol),
            Descriptors.NumRotatableBonds(mol),
            Descriptors.NumAromaticRings(mol),
            Descriptors.FractionCSP3(mol)
        ]
    except:
        return [np.nan]*8

print('Calculating molecular properties...')
props = df['smiles'].apply(calc_mol_props)
props_df = pd.DataFrame(props.tolist(), columns=['MW', 'LogP', 'TPSA', 'HDonors', 'HAcceptors', 'RotBonds', 'AromRings', 'FractionCSP3'])
df = pd.concat([df, props_df], axis=1)

# Drop NaN
df = df.dropna(subset=['pKD', 'MW', 'LogP'])

print('=== MOLECULAR PROPERTY ANALYSIS BY AFFINITY ===')
# Correlation with pKD
corrs = df[['pKD', 'MW', 'LogP', 'TPSA', 'HDonors', 'HAcceptors', 'RotBonds', 'AromRings', 'FractionCSP3']].corr()['pKD'].drop('pKD')
print('Correlation with pKD (binding affinity):')
for col, corr in corrs.sort_values(key=abs, ascending=False).items():
    print(f'  {col}: {corr:.4f}')

# Bin by affinity and analyze
df['affinity_bin'] = pd.cut(df['pKD'], bins=[0, 7, 8, 9, 15], labels=['weak', 'moderate', 'strong', 'very_strong'])
print('\\n=== PROPERTY MEANS BY AFFINITY BIN ===')
props_cols = ['MW', 'LogP', 'TPSA', 'HDonors', 'HAcceptors', 'RotBonds']
for col in props_cols:
    means = df.groupby('affinity_bin')[col].mean()
    print(f'{col}: ', end='')
    for m in means:
        print(f'{m:.1f} ', end='')
    print()

# Resolution correlation
print('\\n=== RESOLUTION VS pKD ===')
print('Correlation:', df[['resolution', 'pKD']].corr().iloc[0,1])
df['res_bin'] = pd.cut(df['resolution'], bins=[0, 2, 2.5, 3, 10], labels=['high', 'med', 'low', 'very_low'])
print('pKD by resolution:')
for res, pkd in df.groupby('res_bin')['pKD'].mean().items():
    print(f'  {res}: {pkd:.2f}')