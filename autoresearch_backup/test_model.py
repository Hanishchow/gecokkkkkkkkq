import pickle, numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

# Load model
with open('geock_deep_trees_final.pkl','rb') as f:
    d = pickle.load(f)

model = d['model']
scaler = d['scaler']
selector = d['selector']
print(f'Model loaded: CV R = {d["cv_r"]:.4f}')

# Test predictions
smiles_list = [
    'CCO',                           # ethanol
    'CC(=O)Oc1ccccc1C(=O)O',         # aspirin
    'c1ccccc1',                       # benzene
    'CC(C)Cc1ccc(CC(N)C(=O)O)cc1',   # leucine
]

for smi in smiles_list:
    mol = Chem.MolFromSmiles(smi)
    fp = np.array(AllChem.GetMorganFingerprintAsBitVect(mol,2,nBits=512), dtype=np.float32).reshape(1,-1)
    X = scaler.transform(fp)
    X = selector.transform(X)
    pKd = model.predict(X)[0]
    Kd_nM = 10**(-pKd) * 1e9
    print(f'  {smi:40s}  pKd={pKd:.2f}  Kd={Kd_nM:.1f} nM')
