#!/usr/bin/env python3
"""
GEOCK Complete Pipeline
========================
End-to-end: acquire → train → evaluate → save

Usage:
    python run_pipeline.py              # Full pipeline
    python run_pipeline.py --acquire   # Data only
    python run_pipeline.py --train    # Training only
    python run_pipeline.py --model xgboost
    python run_pipeline.py --limit 50000
"""

import os
import sys
import pickle
import argparse
import json
import subprocess
from pathlib import Path
from datetime import datetime


# Use cross-platform paths
def _get_cache_dir():
    linux = Path("/home/chow/.cache/geock_autoresearch")
    if linux.exists():
        return linux
    win = Path(os.path.expanduser("~/OneDrive/.cache/geock_autoresearch"))
    if win.exists():
        return win
    return Path("./cache")


def _get_work_dir():
    linux = Path("/home/chow/autoresearch")
    if linux.exists():
        return linux
    win = Path(os.path.expanduser("~/OneDrive/autoresearch"))
    if win.exists():
        return win
    return Path(".")


CACHE_DIR = _get_cache_dir()
WORK_DIR = _get_work_dir()


def run_command(cmd, description):
    """Run a shell command with output."""
    print(f"\n[{description}]")
    print(f"  $ {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ⚠ Error: {result.stderr}")
        return False
    print(f"  ✅ Done")
    return True


def main():
    parser = argparse.ArgumentParser(description="GEOCK Complete Pipeline")
    parser.add_argument("--acquire", action="store_true", help="Acquire data only")
    parser.add_argument(
        "--train", action="store_true", help="Train only (skip acquisition)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="xgboost",
        choices=["xgboost", "neural", "both"],
        help="Model type",
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="Epochs for neural network"
    )
    parser.add_argument("--folds", type=int, default=5, help="CV folds")
    parser.add_argument(
        "--limit", type=int, default=None, help="Limit training samples"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Show what would run without executing"
    )
    parser.add_argument("--output", type=str, default=None, help="Output model name")

    args = parser.parse_args()

    print("=" * 60)
    print("GEOCK Complete Pipeline")
    print("=" * 60)

    if args.dry_run:
        print("\n⚠ DRY RUN MODE - No changes will be made")

    # ===== STEP 1: Data Acquisition =====
    if not args.train and not args.acquire:
        do_acquire = True
    else:
        do_acquire = args.acquire

    if do_acquire:
        print("\n" + "=" * 60)
        print("STEP 1: Data Acquisition")
        print("=" * 60)

        # Check existing data
        data_files = [
            WORK_DIR / "geock_merged_data.pkl",
            CACHE_DIR / "merged_39k.pkl",
            CACHE_DIR / "lp_new_features_8k_no2016.pkl",
        ]

        existing_data = None
        for f in data_files:
            if f.exists():
                print(f"Found: {f.name}")
                with open(f, "rb") as fp:
                    existing_data = pickle.load(fp)
                print(f"  Entries: {len(existing_data)}")
                break

        if existing_data is None:
            print("No existing data found")
            if not args.dry_run:
                # Run acquire script
                import acquire_data

                existing_data = acquire_data.run_pipeline(
                    output_name="geock_merged_data.pkl"
                )

        # Check if we need more data
        n_current = len(existing_data) if existing_data else 0
        target = args.limit or 50000

        if n_current < target:
            print(f"\n⚠ Have {n_current}, target {target}")
            print("Run manually: python acquire_data.py --source all")
        else:
            print(f"\n✅ Data ready: {n_current} entries")

    # ===== STEP 2: Training =====
    if not args.acquire:
        do_train = True
    else:
        do_train = False

    if do_train:
        print("\n" + "=" * 60)
        print("STEP 2: Training")
        print("=" * 60)

        # Determine data file
        data_files = [
            WORK_DIR / "geock_merged_data.pkl",
            CACHE_DIR / "merged_39k.pkl",
        ]

        data_file = None
        for f in data_files:
            if f.exists():
                data_file = f
                break

        if not data_file:
            print("⚠ No training data found!")
            return

        print(f"Training data: {data_file.name}")

        if not args.dry_run:
            # Import and run training
            import train_pipeline

            # Set up args
            class Args:
                model = args.model
                data = data_file.name
                folds = args.folds
                epochs = args.epochs
                output = args.output
                features = 512

            result = train_pipeline.main()

            if result:
                best_r2 = max(r["cv_r2"] for r in result.values())
                print(f"\n🎉 Training complete!")
                print(f"  Best CV R²: {best_r2:.4f}")

                # Update AUTOPLAN with results
                autoplan_file = WORK_DIR / "docs/AUTOPLAN.md"
                if autoplan_file.exists():
                    with open(autoplan_file, "a") as f:
                        f.write(f"\n### Last Run: {datetime.now().isoformat()}\n")
                        for name, res in result.items():
                            f.write(f"- {name}: R²={res['cv_r2']:.4f}\n")

    # ===== STEP 3: Summary =====
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Work dir: {WORK_DIR}")

    # List output files
    print("\nOutput files:")
    for f in sorted(WORK_DIR.glob("geock_model*.pkl")):
        print(f"  {f.name}")
    for f in sorted(WORK_DIR.glob("geock_merged_data*")):
        print(f"  {f.name}")

    print("\nNext steps:")
    print("  python evaluate.py              # Evaluate model")
    print("  python predict.py --smiles CCO # Make prediction")
    print("  python web/app.py               # Start web UI")


if __name__ == "__main__":
    main()
