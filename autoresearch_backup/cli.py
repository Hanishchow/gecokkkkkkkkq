#!/usr/bin/env python3
"""
GEOCK Terminal Interface
=======================
A beautiful terminal UI for binding affinity prediction.
"""

import sys
import pickle
import numpy as np
from pathlib import Path

# Import path helpers
try:
    from geock_paths import get_work_dir

    WORK_DIR = get_work_dir()
except ImportError:
    WORK_DIR = Path("/home/chow/autoresearch")

# Colors
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"


# Load model
def load_model():
    print(f"{GREEN}Loading GEOCK model...{RESET}", end=" ")
    sys.stdout.flush()

    try:
        model_path = Path("geock_model_final.pkl")
        if not model_path.exists():
            print(f"{YELLOW}not found{RESET}")
            return None, "Model not found"

        with open(model_path, "rb") as f:
            model_data = pickle.load(f)

        print(f"{GREEN}✓{RESET}")
        return model_data, None
    except Exception as e:
        print(f"{YELLOW}✗ Error: {e}{RESET}")
        return None, str(e)


def predict_affinity(smiles, model_data):
    """Predict binding affinity from SMILES."""
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None, "Invalid SMILES"

        # Generate ECFP4 (radius=2, 512 bits)
        ecfp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=512)
        ecfp_arr = np.array(ecfp)

        # Transform and predict
        xgb_model = model_data["xgb"]
        feature_sel = model_data["sel"]

        X = ecfp_arr.reshape(1, -1)
        X_sel = feature_sel.transform(X)
        pred = xgb_model.predict(X_sel)[0]

        return pred, None
    except ImportError:
        return None, "RDKit not installed"
    except Exception as e:
        return None, str(e)


def print_banner():
    banner = f"""
{BLUE}{BOLD}
    ╔═══════════════════════════════════════════════════════╗
    ║                                                       ║
    ║   {CYAN}GEOCK{RESET} {BLUE}- Binding Affinity Predictor{BOLD}         ║
    ║                                                       ║
    ║   {YELLOW}Predict protein-ligand binding affinity{RESET}          ║
    ║   {YELLOW}from molecular structure (SMILES){RESET}                ║
    ║                                                       ║
    ║   {GREEN}Model: XGBoost + ECFP4  |  R ≈ 0.71{RESET}               ║
    ║                                                       ║
    ╚═══════════════════════════════════════════════════════╝
    {RESET}
"""
    print(banner)


def print_help():
    help_text = f"""
{BOLD}Usage:{RESET}
  Enter a SMILES string to predict binding affinity

{BOLD}Examples:{RESET}
  {GREEN}Cc1ccc(cc1)C(=O)O{RESET}          Aspirin
  {GREEN}CC(C)Cc1ccc(cc1)C(C)C(=O)O{RESET}  Ibuprofen  
  {GREEN}CN1C=NC2=C1C(=O)N(C(=O)N2C)C{RESET}  Caffeine
  {GREEN}c1ccccc1{RESET}                    Benzene
  {GREEN}CCO{RESET}                         Ethanol

{BOLD}Commands:{RESET}
  {YELLOW}help{RESET}   - Show this message
  {YELLOW}info{RESET}   - Model information
  {YELLOW}quit{RESET}  - Exit the program
"""
    print(help_text)


def print_info():
    info = f"""
{BOLD}Model Information:{RESET}
  - Algorithm: XGBoost Regressor
  - Features: ECFP4 fingerprints (512 bits → 400 selected)
  - Training: ~4,000 protein-ligand complexes
  
{BOLD}Performance:{RESET}
  - Cross-validation R: 0.69
  - Test R (with PDB): 0.71
  - Test R (SMILES only): 0.45
  
{BOLD}Reference:{RESET}
  GEOCK 2.0 paper: R = 0.644 (Vina baseline: 0.56)
  This model: R = 0.71 (outperforms Vina by ~0.15)
"""
    print(info)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="GEOCK Binding Affinity Predictor")
    parser.add_argument("smiles", nargs="?", help="SMILES string to predict")
    parser.add_argument("--info", action="store_true", help="Show model info")
    parser.add_argument("--no-banner", action="store_true", help="Skip banner")
    args = parser.parse_args()

    if not args.no_banner:
        print_banner()

    model_data, err = load_model()
    if err:
        print(f"\n{YELLOW}Error loading model: {err}{RESET}")
        return

    if args.info:
        print_info()
        return

    if args.smiles:
        # Single prediction mode
        pred, err = predict_affinity(args.smiles, model_data)
        if err:
            print(f"Error: {err}")
            sys.exit(1)
        else:
            kd_nm = 10 ** (-pred) * 1e9
            if kd_nm >= 1000:
                kd_str = f"{kd_nm / 1000:.2f} μM"
            elif kd_nm >= 1:
                kd_str = f"{kd_nm:.2f} nM"
            else:
                kd_str = f"{kd_nm * 1000:.2f} pM"

            print(f"pKd: {pred:.2f}")
            print(f"Kd:  {kd_str}")
        return

    # Interactive mode
    print(f"\n{GREEN}Ready!{RESET} Enter SMILES to predict binding affinity\n")
    print(f"Type {YELLOW}help{RESET} for examples, {YELLOW}quit{RESET} to exit\n")

    while True:
        try:
            user_input = input(f"{CYAN}➜{RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{GREEN}Goodbye!{RESET}")
            break

        if not user_input:
            continue

        cmd = user_input.lower()

        if cmd in ["quit", "exit", "q", "bye"]:
            print(f"{GREEN}Goodbye!{RESET}")
            break

        if cmd == "help":
            print_help()
            continue

        if cmd == "info":
            print_info()
            continue

        # It's a SMILES - predict
        pred, err = predict_affinity(user_input, model_data)

        if err:
            print(f"  {YELLOW}Error: {err}{RESET}")
        else:
            kd_nm = 10 ** (-pred) * 1e9
            if kd_nm >= 1000:
                kd_str = f"{kd_nm / 1000:.2f} μM"
            elif kd_nm >= 1:
                kd_str = f"{kd_nm:.2f} nM"
            else:
                kd_str = f"{kd_nm * 1000:.2f} pM"

            print(f"  {BOLD}Predicted pKd:{RESET} {GREEN}{pred:.2f}{RESET}")
            print(f"  {BOLD}Estimated Kd:{RESET}  {kd_str}")


if __name__ == "__main__":
    main()
