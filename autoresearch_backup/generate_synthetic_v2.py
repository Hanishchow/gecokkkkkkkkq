#!/usr/bin/env python3
"""GEOCK v2 - Synthetic Data Generation using NeMo Data Designer"""
from nemo_microservices.essentials import (
    CategorySamplerParams,
    DataDesignerConfigBuilder,
    InferenceParameters,
    LLMTextColumnConfig,
    ModelConfig,
    NeMoDataDesignerClient,
    SamplerColumnConfig,
    SamplerType,
    SubcategorySamplerParams,
    UniformSamplerParams,
)
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import os

api_key = os.environ.get("NVIDIA_API_KEY") or "nvapi-ixro4Tz9ggN90olY2AThqtV-4eACNChsjz1o7izqY8fflA"

client = NeMoDataDesignerClient(base_url="https://ai.nvidia.com/gt")

model_configs = [
    ModelConfig(
        alias="geock-gen",
        model="nvidia/nemo/llama-3.3-nemotron-super-49b-v1.5",
        inference_parameters=InferenceParameters(temperature=0.7, top_p=0.95, max_tokens=2048)
    )
]

cfg = DataDesignerConfigBuilder(model_configs)

cfg.add_column(SamplerColumnConfig(
    name="kinase_family", sampler_type=SamplerType.CATEGORY,
    params=CategorySamplerParams(values=["TK","PKA","CMGC","STE","CK1","AGC"], weights=[3,2,2,1,1,1])
)

cfg.add_column(SamplerColumnConfig(
    name="kinase_subfamily", sampler_type=SamplerType.SUBCATEGORY,
    params=SubcategorySamplerParams(category="kinase_family", values={
        "TK":["Src","Abl","EGFR","Her2","Met","Kit","Flt3","Ret"],
        "PKA":["PRKACA","PRKACB","PRKACG"],
        "CMGC":["CDK2","CDK4","MAPK1","MAPK14","GSK3B","CK2A1"],
        "STE":["MAP2K1","MAP2K2","MINK1"],
        "CK1":["CK1A","CK1D"],
        "AGC":["AKT1","AKT2","PKG1","PKC_alpha"],
    })
)

cfg.add_column(SamplerColumnConfig(
    name="molecule_type", sampler_type=SamplerType.CATEGORY,
    params=CategorySamplerParams(values=["Type I - ATP-competitive","Type II - Allosteric","Type III - Substrate-competitive","Covalent inhibitor","Multi-target inhibitor"])
)

cfg.add_column(SamplerColumnConfig(
    name="scaffold_type", sampler_type=SamplerType.CATEGORY,
    params=CategorySamplerParams(values=["Aminopyrimidine","Pyrazolo[3,4-d]pyrimidine","Quinazoline","Indolinone","Phenylamino-pyrimidine","Urea-based","Nitrogen heterocycle"])
)

cfg.add_column(SamplerColumnConfig(
    name="pKd", sampler_type=SamplerType.UNIFORM,
    params=UniformSamplerParams(low=4.0, high=12.0), convert_to="float")
)

cfg.add_column(SamplerColumnConfig(
    name="mw_class", sampler_type=SamplerType.CATEGORY,
    params=CategorySamplerParams(values=["<300","300-500","500-700",">700"])
)

cfg.add_column(SamplerColumnConfig(
    name="logp_class", sampler_type=SamplerType.CATEGORY,
    params=CategorySamplerParams(values=["<2","2-4","4-6",">6"])
)

cfg.add_column(SamplerColumnConfig(
    name="hbd_class", sampler_type=SamplerType.CATEGORY,
    params=CategorySamplerParams(values=["0-1","2-3","4-5",">5"])
)

cfg.add_column(LLMTextColumnConfig(
    name="smiles",
    prompt=("Generate a valid SMILES for a kinase inhibitor with family {{kinase_family}}, type {{molecule_type}}, scaffold {{scaffold_type}}, MW {{mw_class}}, LogP {{logp_class}}, HBD {{hbd_class}}. Reply ONLY the SMILES."),
    system_prompt="Expert medicinal chemist. Output only SMILES string, nothing else.",
    model_alias="geock-gen")
)

print("Generating synthetic data via NeMo Data Designer...")
preview = client.preview(cfg, num_records=500)
df = preview.dataset
print(f"Generated {len(df)} records. Columns: {list(df.columns)}")

print("\nEngineering ECFP fingerprints...")
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    def ecfp(smi, r=2, bits=512):
        mol = Chem.MolFromSmiles(smi) if smi else None
        return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, r, nBits=bits)) if mol else np.zeros(bits)
    ecfp_arr = np.array([ecfp(s) for s in df['smiles']])
except ImportError:
    ecfp_arr = np.zeros((len(df), 512))

data = []
for i, row in df.iterrows():
    data.append({'smiles': row.get('smiles',''), 'affinity': row.get('pKd', 7.0), 'ecfp': ecfp_arr[i],
               'kinase_family': row.get('kinase_family',''), 'kinase_subfamily': row.get('kinase_subfamily',''),
               'molecule_type': row.get('molecule_type',''), 'scaffold_type': row.get('scaffold_type',''),
               'pdb_id': f"SYN{i:04d}", 'physics': np.zeros(4)})

out = Path.home() / ".cache" / "geock_autoresearch" / "synthetic_500.pkl"
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, "wb") as f:
    pickle.dump(data, f)
df.to_csv(out.with_suffix(".csv"), index=False)
print(f"Saved {len(data)} records to {out}")