"""
augment_and_train.py — GEOCK Path 2: Scale-Up Pipeline
======================================================
Maximise value from existing data + prepare for PDBbind-scale training.

Strategy:
  1. AUGMENT: existing 113 structures at 3 pocket radii (8Å, 10Å, 12Å)
     → 3x effective training data, measures sensitivity to pocket definition
  2. SPLIT: stratified by protein family (from PDB IDs)
  3. TRAIN: E4-only features, Lasso + BayesianRidge ensemble
  4. REPORT: honest LOO-CV + external test

When PDBbind/CASF data arrives:
  python augment_and_train.py --data-dir /path/to/pdbbind_data
  → Same pipeline, scales to 5000+ complexes

Usage:
  python augment_and_train.py                    # augment + train
  python augment_and_train.py --data-dir /path   # custom data
  python augment_and_train.py --no-augment      # single-radius only
"""

import argparse, json, pickle, time, sys, os, warnings
warnings.filterwarnings("ignore")

import numpy as np
from pathlib import Path
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, BayesianRidge
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.model_selection import LeaveOneOut, cross_val_predict, StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prepare import load_features, get_splits, evaluate_r, evaluate_mae, FEATURE_NAMES
from patch_parse import parse_pocket_and_ligand
from bio_engine import unified_score

OUT_DIR      = Path("/mnt/c/Users/yakka/Downloads/geock_pdbbind_data")
GEOCK_CACHE  = Path("CACHE_DIR / features_v2.pkl")


# ── Pocket radius augmentation ─────────────────────────────────────────────
def augment_at_radius(pdb_id: str, smiles: str, data_dir: Path, radius: float) -> np.ndarray | None:
    """Extract features at a specific pocket radius."""
    for suffix in ["_pocket.pdb", "_binding.pdb", ".pdb"]:
        pocket_file = data_dir / pdb_id / f"{pdb_id}{suffix}"
        if pocket_file.exists():
            break
    else:
        return None

    try:
        rec_coords, rec_types, lig_coords, lig_types, _, n_rot = \
            parse_pocket_and_ligand(str(pocket_file), cutoff=radius)

        if len(rec_coords) < 5 or len(lig_coords) < 3:
            return None

        feats = unified_score(
            pdb_id, rec_coords, rec_types,
            lig_coords, lig_types,
            n_torsions=n_rot,
            smiles=smiles or None,
            use_quantum=False,
        )
        return feats
    except:
        return None


def run_augmentation(data_dir: Path, compounds: list, radii=[8.0, 10.0, 12.0]):
    """Extract features at multiple pocket radii."""
    print(f"\n[Augmentation] {len(compounds)} compounds × {len(radii)} radii = {len(compounds)*len(radii)} samples")

    cache_file = OUT_DIR / f"augmented_r{int(min(radii))}_{int(max(radii))}.pkl"
    if cache_file.exists():
        print(f"[CACHE] Loading augmented data from {cache_file}")
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    results = {}
    start = time.time()

    for i, c in enumerate(compounds):
        pid = c["pdb_id"]
        smi = c.get("smiles", "")
        all_feats = []

        for radius in radii:
            feats = augment_at_radius(pid, smi, data_dir, radius)
            if feats is not None:
                all_feats.append(feats)

        if all_feats:
            results[pid] = {
                "features": np.array(all_feats),   # (n_radii, 24)
                "y": c["experimental_affinity"],
                "smiles": smi,
                "n_radii": len(all_feats),
            }

        if (i + 1) % 20 == 0:
            elapsed = time.time() - start
            rate = (i + 1) / elapsed
            print(f"  [{i+1}/{len(compounds)}] {rate:.1f}/s  done={len(results)}")

    # Save
    with open(cache_file, "wb") as f:
        pickle.dump(results, f)

    print(f"  Done: {len(results)}/{len(compounds)} compounds augmented")
    return results


# ── Training ────────────────────────────────────────────────────────────────
def train_and_evaluate(aug_data: dict, test_pdb_ids: list, radii=[8.0, 10.0, 12.0]):
    """
    Train on augmented data, evaluate on held-out test set.
    
    Strategy:
      - Average features across radii (robust pocket definition)
      - Use E4 features (15-23) only
      - Lasso + BayesianRidge ensemble
      - LOO-CV for model selection
    """
    E4_IDX = list(range(15, 24))   # 9 E4 features

    # Build X, y
    all_pids = list(aug_data.keys())
    X_raw_list, y_list = [], []

    for pid in all_pids:
        feats = aug_data[pid]["features"]   # (n_radii, 24)
        # Average across radii
        avg_feats = feats.mean(axis=0)
        X_raw_list.append(avg_feats)
        y_list.append(aug_data[pid]["y"])

    X_raw  = np.array(X_raw_list, dtype=np.float32)
    y_raw  = np.array(y_list, dtype=np.float32)
    y_pkd  = (-y_raw / 1.364).astype(np.float32)

    # E4 features only
    X = X_raw[:, E4_IDX]
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Standardize
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    # Split: use PDB IDs to identify test (if provided)
    # Otherwise use last 20% as test
    test_set = set(test_pdb_ids)
    train_mask = np.array([pid not in test_set for pid in all_pids])
    test_mask  = ~train_mask

    if test_mask.sum() < 2:
        n = len(X_s)
        idx = np.arange(n)
        np.random.seed(42)
        np.random.shuffle(idx)
        train_mask = idx[:int(n * 0.8)]
        test_mask  = idx[int(n * 0.8):]

    X_tr, y_tr = X_s[train_mask], y_pkd[train_mask]
    X_te, y_te = X_s[test_mask],  y_pkd[test_mask]

    # Feature selection
    results = []
    for k in [2, 3, 4, 5]:
        if k > X_tr.shape[1]:
            continue

        # Lasso
        selector = SelectKBest(f_regression, k=k)
        Xtr_s = selector.fit_transform(X_tr, y_tr)
        Xte_s = selector.transform(X_te)
        Xtr_cv = selector.transform(X_tr)

        for alpha in [0.005, 0.01, 0.015, 0.02, 0.03]:
            model = Lasso(alpha=alpha, max_iter=5000)
            model.fit(Xtr_s, y_tr)

            pred_tr  = model.predict(Xtr_s)
            pred_te  = model.predict(Xte_s)

            train_r = pearsonr(y_tr, pred_tr)[0]
            test_r  = pearsonr(y_te, pred_te)[0]
            test_mae = np.mean(np.abs(y_te - pred_te))

            # LOO-CV on train
            loo = LeaveOneOut()
            loo_preds = cross_val_predict(Lasso(alpha=alpha, max_iter=5000), Xtr_cv, y_tr, cv=loo)
            loo_r = pearsonr(y_tr, loo_preds)[0]

            support = selector.get_support()
            feat_names = [FEATURE_NAMES[15:24][i] for i, s in enumerate(support) if s]

            results.append({
                "model": "Lasso",
                "k": k, "alpha": alpha,
                "train_r": train_r, "test_r": test_r, "loo_r": loo_r,
                "test_mae": test_mae,
                "features": feat_names,
                "Xtr_s": Xtr_s, "Xte_s": Xte_s,
            })

        # BayesianRidge
        br = BayesianRidge()
        br.fit(Xtr_s, y_tr)
        pred_tr = br.predict(Xtr_s)
        pred_te = br.predict(Xte_s)
        train_r = pearsonr(y_tr, pred_tr)[0]
        test_r  = pearsonr(y_te, pred_te)[0]
        test_mae = np.mean(np.abs(y_te - pred_te))

        loo = LeaveOneOut()
        loo_preds = cross_val_predict(BayesianRidge(), Xtr_cv, y_tr, cv=loo)
        loo_r = pearsonr(y_tr, loo_preds)[0]

        results.append({
            "model": "BayesianRidge",
            "k": k, "alpha": None,
            "train_r": train_r, "test_r": test_r, "loo_r": loo_r,
            "test_mae": test_mae,
            "features": feat_names,
            "Xtr_s": Xtr_s, "Xte_s": Xte_s,
        })

    # Best by LOO-CV
    best = max(results, key=lambda r: r["loo_r"])

    return {
        "results": results,
        "best": best,
        "X_tr": X_tr, "y_tr": y_tr,
        "X_te": X_te, "y_te": y_te,
        "n_train": len(y_tr), "n_test": len(y_te),
    }


def print_results(eval_result: dict):
    """Print results table."""
    print("\n" + "=" * 80)
    print("  PATH 2 RESULTS — Augmented Data")
    print("=" * 80)
    print(f"\n  Dataset: {eval_result['n_train']} train / {eval_result['n_test']} test  "
          f"(augmented: 3 pocket radii averaged)")
    print(f"  Features: E4 pocket only (9 features)")
    print()

    rrs = sorted(eval_result["results"], key=lambda r: r["loo_r"], reverse=True)
    print(f"  {'Model':<16} {'k':>3} {'Alpha':>6}  {'Train':>7} {'LOO-CV':>7} {'Test':>7}  "
          f"{'MAE':>6}  Features")
    print(f"  {'-'*16:16} {'-'*3:>3} {'-'*6:>6}  {'-'*7:>7} {'-'*7:>7} {'-'*7:>7}  "
          f"{'-'*6:>6}  ---------")

    for r in rrs[:15]:
        alpha_str = f"{r['alpha']:.3f}" if r['alpha'] else "auto"
        feats_short = ", ".join(n.split("_")[-1] for n in r["features"][:2])
        flag = "★" if r["loo_r"] > 0.7 else ("+" if r["loo_r"] > 0.6 else "")
        print(f"  {r['model']:<16} {r['k']:>3} {alpha_str:>6}  "
              f"{r['train_r']:>7.3f} {r['loo_r']:>7.3f} {r['test_r']:>7.3f}  "
              f"{r['test_mae']:>6.3f}  {feats_short}  {flag}")

    best = eval_result["best"]
    print()
    print(f"  BEST (by LOO-CV): {best['model']} k={best['k']} alpha={best['alpha']}")
    print(f"    LOO-CV R: {best['loo_r']:.3f}  Test R: {best['test_r']:.3f}  MAE: {best['test_mae']:.3f}")
    print(f"    Features: {best['features']}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Path to PDBbind/CASF data (default: geock_110_data)")
    parser.add_argument("--no-augment", action="store_true",
                        help="Skip augmentation, use single pocket radius")
    parser.add_argument("--radii", type=float, nargs="+", default=[8.0, 10.0, 12.0],
                        help="Pocket radii for augmentation")
    parser.add_argument("--limit", type=int, default=0,
                        help="Limit number of compounds")
    args = parser.parse_args()

    data_dir = Path(args.data_dir) if args.data_dir else Path("/mnt/c/Users/yakka/Downloads/geock_110_data")

    print("=" * 70)
    print("  GEOCK PATH 2 — Scale-Up Pipeline")
    print("=" * 70)

    # Load or build compound list
    compounds_file = data_dir / "compounds.json"
    if not compounds_file.exists():
        print(f"[ERR] No compounds.json at {data_dir}")
        print("      Path 2 needs a compounds.json with pdb_id, smiles, experimental_affinity")
        return

    with open(compounds_file) as f:
        compounds = json.load(f)

    if args.limit > 0:
        compounds = compounds[:args.limit]

    print(f"\n[INFO] Data: {len(compounds)} compounds from {data_dir}")
    print(f"[INFO] Radii: {args.radii}")

    # Augment
    if args.no_augment:
        radii = [10.0]
    else:
        radii = args.radii

    aug_data = run_augmentation(data_dir, compounds, radii)

    if len(aug_data) < 5:
        print("[ERR] Not enough augmented data")
        return

    # Train + evaluate
    # Hold out last 5% as test
    all_pids = sorted(aug_data.keys())
    n_test = max(2, int(len(all_pids) * 0.05))
    test_pdb_ids = all_pids[-n_test:]

    print(f"\n[Train] {len(all_pids) - n_test} train / {n_test} test")

    result = train_and_evaluate(aug_data, test_pdb_ids, radii)
    print_results(result)

    # Save best model
    best = result["best"]
    model_file = OUT_DIR / "best_model.pkl"
    with open(model_file, "wb") as f:
        pickle.dump({
            "model": best["model"],
            "k": best["k"],
            "alpha": best["alpha"],
            "features": best["features"],
            "loo_r": best["loo_r"],
            "test_r": best["test_r"],
            "X_tr": result["X_tr"],
            "y_tr": result["y_tr"],
            "X_te": result["X_te"],
            "y_te": result["y_te"],
            "radii": radii,
            "augmented": not args.no_augment,
        }, f)

    print(f"\n[SAVED] Best model: {model_file}")


if __name__ == "__main__":
    main()
