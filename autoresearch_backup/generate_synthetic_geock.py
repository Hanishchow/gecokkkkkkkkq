#!/usr/bin/env python3
"""
GEOCK v2 - Synthetic Data Generation using NeMo Data Designer
Generates synthetic kinase inhibitor binding data
"""
from nemo_microservices.data_designer.essentials import (
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

# Initialize client
data_designer_client = NeMoDataDesignerClient(
    base_url="https://ai.phi.api.nvidia.com/v1/gt/12",  # Public endpoint
    # Or use the default if you have API key:
    # base_ai.api.nvidia.com/v1/nemo/dd",
    # default_headers={"Authorization": f"Bearer {os.environ.get('NVIDIA_API_KEY')}"}
)

# Model configuration
model_configs = [
    ModelConfig(
        alias="geock-gen",
        model="nvidia/llama-3.3-nemotron-super-49b-v1.5",  # Good for structured generation
        inference_parameters=InferenceParameters(
            temperature=0.7,
            top_p=0.95,
            max_tokens=2048,
        )
    )
]

config_builder = DataDesignerConfigBuilder(model_configs)

# ====== KINASE FAMILY SAMPLER ======
config_builder.add_column(
    SamplerColumnConfig(
        name="kinase_family",
        sampler_type=SamplerType.CATEGORY,
        params=CategorySamplerParams(
            values=[
                "TK",      # Tyrosine Kinase
                "PKA",     # Protein Kinase A
                "CMGC",    # CK2, MAPK, GSK3, CDK
                "STE",     # STE family
                "CK1",     # Casein Kinase 1
                "AGC",     # PKG, PKC, PKN
            ],
            weights=[3, 2, 2, 1, 1, 1],  # TK most common in training data
        ),
    )
)

# ====== KINASE SUBFAMILY ======
config_builder.add_column(
    SamplerColumnConfig(
        name="kinase_subfamily",
        sampler_type=SamplerType.SUBCATEGORY,
        params=SubcategorySamplerParams(
            category="kinase_family",
            values={
                "TK": ["Src", "Abl", "EGFR", "Her2", "Met", "Kit", "Flt3", "Ret"],
                "PKA": ["PRKACA", "PRKACB", "PRKACG"],
                "CMGC": ["CDK2", "CDK4", "MAPK1", "MAPK14", "GSK3B", "CK2A1"],
                "STE": ["MAP2K1", "MAP2K2", "MINK1"],
                "CK1": ["CK1A", "CK1D"],
                "AGC": ["AKT1", "AKT2", "PKG1", "PKC_alpha"],
            },
        ),
    )
)

# ====== MOLECULE TYPE ======
config_builder.add_column(
    SamplerColumnConfig(
        name="molecule_type",
        sampler_type=SamplerType.CATEGORY,
        params=CategorySamplerParams(
            values=[
                "Type I - ATP-competitive",
                "Type II - Allosteric",
                "Type III - Substrate-competitive",
                "Covalent inhibitor",
                "Multi-target inhibitor",
            ],
        ),
    )
)

# ====== SCAFFOLD TYPE ======
config_builder.add_column(
    SamplerColumnConfig(
        name="scaffold_type",
        sampler_type=SamplerType.CATEGORY,
        params=CategorySamplerParams(
            values=[
                "Aminopyrimidine",
                "Pyrazolo[3,4-d]pyrimidine",
                "Quinazoline",
                "Indolinone",
                "Phenylamino-pyrimidine",
                "Urea-based",
                "Nitrogen heterocycle",
            ],
        ),
    )
)

# ====== BINDING AFFINITY (TARGET) ======
config_builder.add_column(
    SamplerColumnConfig(
        name="pKd",
        sampler_type=SamplerType.UNIFORM,
        params=UniformSamplerParams(low=4.0, high=12.0),  # pKd range
        convert_to="float",
    )
)

# ====== LIGAND EFFICIENCY ======
config_builder.add_column(
    SamplerColumnConfig(
        name="ligand_efficiency",
        sampler_type=SamplerType.UNIFORM,
        params=UniformSamplerParams(low=0.2, high=1.5),  # kcal/mol per heavy atom
        convert_to="float",
    )
)

# ====== LIPINSKI PROPERTIES ======
config_builder.add_column(
    SamplerColumnConfig(
        name="mw_class",
        sampler_type=SamplerType.CATEGORY,
        params=CategorySamplerParams(
            values=["<300", "300-500", "500-700", ">700"],
        ),
    )
)

config_builder.add_column(
    SamplerColumnConfig(
        name="logp_class",
        sampler_type=SamplerType.CATEGORY,
        params=CategorySamplerParams(
            values=["<2", "2-4", "4-6", ">6"],
        ),
    )
)

config_builder.add_column(
    SamplerColumnConfig(
        name="hbd_class",
        sampler_type=SamplerType.CATEGORY,
        params=CategorySamplerParams(
            values=["0-1", "2-3", "4-5", ">5"],
        ),
    )
)

# ====== SMILES GENERATION (LLM TEXT) ======
config_builder.add_column(
    LLMTextColumnConfig(
        name="smiles",
        prompt=(
            "Generate a valid SMILES string for a small molecule kinase inhibitor with the following properties:\n"
            "- Kinase family: {{ kinase_family }}\n"
            "- Molecule type: {{ molecule_type }}\n"
            "- Scaffold: {{ scaffold_type }}\n"
            "- Molecular weight class: {{ mw_class }}\n"
            "- LogP class: {{ logp_class }}\n"
            "- H-bond donor class: {{ hbd_class }}\n"
            "Respond with ONLY the SMILES string, no explanations or quotes."
        ),
        system_prompt=(
            "You are an expert medicinal chemist. Generate valid SMILES for kinase inhibitors. "
            "Only output the SMILES string, nothing else. No quotes around the SMILES."
        ),
        model_alias="geock-gen",
    )
)

# ====== BINDING MODE DESCRIPTION ======
config_builder.add_column(
    LLMTextColumnConfig(
        name="binding_mode",
        prompt=(
            "Describe the binding mode of a {{ molecule_type }} for {{ kinase_subfamily }} "
            "in {{ mw_class }} molecular weight range. "
            "Include key interactions (hydrogen bonds, hydrophobic, salt bridges). "
            "Be concise, 2-3 sentences."
        ),
        system_prompt=(
            "You are a computational medicinal chemist. Describe binding modes concisely."
        ),
        model_alias="geock-gen",
    )
)

# ====== GENERATE DATA ======
print("="*60)
print("GEOCK v2 - Synthetic Data Generation")
print("="*60)

print("\n[1] Generating synthetic kinase inhibitor data...")
print("    This will create 500 records...")

preview = data_designer_client.preview(config_builder, num_records=500)
print(f"\n[2] Generated {len(preview.dataset)} records")

# Display sample
print("\n[3] Sample record:")
preview.display_sample_record()

# Save to dataframe
df = preview.dataset
print(f"\n[4] Dataset columns: {list(df.columns)}")
print(f"    Shape: {df.shape}")

# ====== FEATURE ENGINEERING FOR GEOCK ======
print("\n[5] Engineering ECFP features...")

def smiles_to_ecfp(smiles, radius=2, bits=512):
    """Convert SMILES to ECFP fingerprint (simplified)"""
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(bits)
        
        # Generate ECFP
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=bits)
        return np.array(fp)
    except:
        return np.zeros(bits)

# Generate ECFP for each SMILES
ecfp_list = []
for smiles in df['smiles']:
    if pd.notna(smiles):
        ecfp = smiles_to_ecfp(str(smiles))
    else:
        ecfp = np.zeros(512)
    ecfp_list.append(ecfp)

ecfp_array = np.array(ecfp_list)

# Create GEOCK format data
geock_data = []
for i, row in df.iterrows():
    geock_data.append({
        'smiles': row.get('smiles', ''),
        'affinity': row['pKd'],
        'ecfp': ecfp_array[i],
        'kinase_family': row['kinase_family'],
        'kinase_subfamily': row['kinase_subfamily'],
        'molecule_type': row['molecule_type'],
        'scaffold_type': row['scaffold_type'],
        'pdb_id': f"SYN{i:04d}",  # Synthetic PDB ID
        'physics': np.array([0.0, 0.0, 0.0, 0.0]),  # Placeholder physics features
    })

# Save
output_path = Path.home() / '.cache' / 'geock_autoresearch' / 'synthetic_geock_500.pkl'
output_path.parent.mkdir(parents=True, exist_ok=True)

with open(output_path, 'wb') as f:
    pickle.dump(geock_data, f)

print(f"\n[6] Saved to: {output_path}")
print(f"    Records: {len(geock_data)}")

# Save DataFrame too for inspection
csv_path = output_path.with_suffix('.csv')
df.to_csv(csv_path, index=False)
print(f"    CSV: {csv_path}")

print("\n" + "="*60)
print("SYNTHETIC DATA GENERATION COMPLETE")
print("="*60)