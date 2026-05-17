#!/usr/bin/env python3
"""
GEOCK Model Training Pipeline
============================
Train and evaluate binding affinity prediction models.

MODELS:
  - Ridge Regression (baseline, interpretable)
  - XGBoost (gradient boosting)
  - Ensemble (XGBoost + Ridge blend)

USAGE:
  python pipeline_train.py --model xgboost      # Train XGBoost
  python pipeline_train.py --model ensemble     # Train ensemble
  python pipeline_train.py --evaluate           # Evaluate model
  python pipeline_train.py --all               # Full pipeline
"""

import os
import sys
import json
import pickle
import logging
import argparse
import warnings
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import Ridge
import xgboost as xgb

warnings.filterwarnings("ignore")

# Import path helpers
try:
    from geock_paths import get_cache_dir, get_work_dir

    WORK_DIR = get_work_dir()
    CACHE_DIR = get_cache_dir()
except ImportError:
    # Fallback for systems without geock_paths.py
    WORK_DIR = Path("/home/chow/autoresearch")
    CACHE_DIR = Path("/home/chow/.cache/geock_autoresearch")


@dataclass
class TrainingResult:
    """Results from model training."""

    model_type: str
    train_r: float
    val_r: float
    test_r: float
    train_mae: float
    val_mae: float
    test_mae: float
    gap: float
    cv_r: float
    cv_std: float
    config: Dict
    n_train: int
    n_val: int
    n_test: int
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict:
        return asdict(self)


class DataLoader:
    """Load and prepare training data."""

    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir

    def load(
        self, min_affinity: float = None, max_affinity: float = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Load training data from cache."""
        paths = [
            self.cache_dir / "lp_new_features_8k.pkl",
            self.cache_dir / "geock_training_data.pkl",
        ]

        all_features = []
        all_affinities = []

        for path in paths:
            if path.exists():
                try:
                    with open(path, "rb") as f:
                        data = pickle.load(f)

                    if isinstance(data, list):
                        for record in data:
                            if "ecfp" in record and "affinity" in record:
                                aff = record["affinity"]
                                if min_affinity and aff < min_affinity:
                                    continue
                                if max_affinity and aff > max_affinity:
                                    continue
                                all_features.append(record["ecfp"])
                                all_affinities.append(aff)

                    log.info(f"Loaded {len(data)} records from {path.name}")
                except Exception as e:
                    log.warning(f"Failed to load {path}: {e}")

        X = np.array(all_features) if all_features else None
        y = np.array(all_affinities) if all_affinities else None

        log.info(f"Total: {X.shape if X is not None else 'None'} samples")
        return X, y

    def split_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.2,
        val_size: float = 0.1,
        seed: int = 42,
    ) -> Tuple:
        """Split data into train/val/test sets."""
        np.random.seed(seed)
        n = len(X)

        indices = np.random.permutation(n)
        n_test = int(n * test_size)
        n_val = int(n * val_size)

        test_idx = indices[:n_test]
        val_idx = indices[n_test : n_test + n_val]
        train_idx = indices[n_test + n_val :]

        return (
            X[train_idx],
            X[val_idx],
            X[test_idx],
            y[train_idx],
            y[val_idx],
            y[test_idx],
        )


class ModelTrainer:
    """Train and evaluate models."""

    def __init__(self, work_dir: Path):
        self.work_dir = work_dir

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Calculate regression metrics."""
        try:
            pearson_r, _ = pearsonr(y_true, y_pred)
            spearman_r, _ = spearmanr(y_true, y_pred)
        except:
            pearson_r, spearman_r = 0.0, 0.0

        mae = np.mean(np.abs(y_true - y_pred))
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

        return {
            "pearson_r": pearson_r,
            "spearman_r": spearman_r,
            "mae": mae,
            "rmse": rmse,
        }

    def train_ridge(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        config: Dict,
    ) -> Tuple:
        """Train Ridge regression model."""
        k = config.get("k", 500)
        alpha = config.get("alpha", 100.0)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)

        selector = SelectKBest(f_regression, k=min(k, X_train.shape[1]))
        X_train_sel = selector.fit_transform(X_train_scaled, y_train)
        X_val_sel = selector.transform(X_val_scaled)
        X_test_sel = selector.transform(X_test_scaled)

        model = Ridge(alpha=alpha)
        model.fit(X_train_sel, y_train)

        train_pred = model.predict(X_train_sel)
        val_pred = model.predict(X_val_sel)
        test_pred = model.predict(X_test_sel)

        train_metrics = self.evaluate(y_train, train_pred)
        val_metrics = self.evaluate(y_val, val_pred)
        test_metrics = self.evaluate(y_test, test_pred)

        artifacts = {"model": model, "scaler": scaler, "selector": selector}

        return train_metrics, val_metrics, test_metrics, artifacts

    def train_xgboost(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        config: Dict,
    ) -> Tuple:
        """Train XGBoost model."""
        k = config.get("k", 500)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)

        selector = SelectKBest(f_regression, k=min(k, X_train.shape[1]))
        X_train_sel = selector.fit_transform(X_train_scaled, y_train)
        X_val_sel = selector.transform(X_val_scaled)
        X_test_sel = selector.transform(X_test_scaled)

        model = xgb.XGBRegressor(
            n_estimators=config.get("n_estimators", 200),
            max_depth=config.get("max_depth", 6),
            learning_rate=config.get("learning_rate", 0.05),
            reg_alpha=config.get("reg_alpha", 1.0),
            reg_lambda=config.get("reg_lambda", 5.0),
            subsample=config.get("subsample", 0.8),
            colsample_bytree=config.get("colsample_bytree", 0.8),
            random_state=config.get("seed", 42),
            n_jobs=-1,
            verbosity=0,
        )

        model.fit(X_train_sel, y_train, eval_set=[(X_val_sel, y_val)], verbose=False)

        train_pred = model.predict(X_train_sel)
        val_pred = model.predict(X_val_sel)
        test_pred = model.predict(X_test_sel)

        train_metrics = self.evaluate(y_train, train_pred)
        val_metrics = self.evaluate(y_val, val_pred)
        test_metrics = self.evaluate(y_test, test_pred)

        artifacts = {"model": model, "scaler": scaler, "selector": selector}

        return train_metrics, val_metrics, test_metrics, artifacts

    def train_ensemble(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        config: Dict,
    ) -> Tuple:
        """Train XGBoost + Ridge ensemble."""
        xgb_weight = config.get("ensemble_weight", 0.8)

        xgb_train, xgb_val, xgb_test, xgb_artifacts = self.train_xgboost(
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
            {k: v for k, v in config.items() if k != "ensemble_weight"},
        )

        ridge_config = {
            "k": config.get("k", 500),
            "alpha": config.get("ridge_alpha", 100),
        }
        ridge_train, ridge_val, ridge_test, ridge_artifacts = self.train_ridge(
            X_train, y_train, X_val, y_val, X_test, y_test, ridge_config
        )

        train_pred = xgb_weight * xgb_artifacts["model"].predict(
            xgb_artifacts["selector"].transform(
                xgb_artifacts["scaler"].transform(X_train)
            )
        ) + (1 - xgb_weight) * ridge_artifacts["model"].predict(
            ridge_artifacts["selector"].transform(
                ridge_artifacts["scaler"].transform(X_train)
            )
        )
        val_pred = xgb_weight * xgb_artifacts["model"].predict(
            xgb_artifacts["selector"].transform(
                xgb_artifacts["scaler"].transform(X_val)
            )
        ) + (1 - xgb_weight) * ridge_artifacts["model"].predict(
            ridge_artifacts["selector"].transform(
                ridge_artifacts["scaler"].transform(X_val)
            )
        )
        test_pred = xgb_weight * xgb_artifacts["model"].predict(
            xgb_artifacts["selector"].transform(
                xgb_artifacts["scaler"].transform(X_test)
            )
        ) + (1 - xgb_weight) * ridge_artifacts["model"].predict(
            ridge_artifacts["selector"].transform(
                ridge_artifacts["scaler"].transform(X_test)
            )
        )

        train_metrics = self.evaluate(y_train, train_pred)
        val_metrics = self.evaluate(y_val, val_pred)
        test_metrics = self.evaluate(y_test, test_pred)

        artifacts = {
            "xgb_model": xgb_artifacts["model"],
            "ridge_model": ridge_artifacts["model"],
            "xgb_scaler": xgb_artifacts["scaler"],
            "ridge_scaler": ridge_artifacts["scaler"],
            "xgb_selector": xgb_artifacts["selector"],
            "ridge_selector": ridge_artifacts["selector"],
            "ensemble_weight": xgb_weight,
        }

        return train_metrics, val_metrics, test_metrics, artifacts

    def cross_validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_type: str,
        config: Dict,
        n_folds: int = 5,
    ) -> Tuple:
        """Perform cross-validation."""
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

        fold_rs = []
        for train_idx, val_idx in kf.split(X):
            X_tr, X_vl = X[train_idx], X[val_idx]
            y_tr, y_vl = y[train_idx], y[val_idx]

            if model_type == "ridge":
                _, _, test_m, _ = self.train_ridge(
                    X_tr, y_tr, X_vl, y_vl, X_vl, y_vl, config
                )
            elif model_type == "xgboost":
                _, _, test_m, _ = self.train_xgboost(
                    X_tr, y_tr, X_vl, y_vl, X_vl, y_vl, config
                )
            else:
                _, _, test_m, _ = self.train_ensemble(
                    X_tr, y_tr, X_vl, y_vl, X_vl, y_vl, config
                )

            fold_rs.append(test_m["pearson_r"])

        return np.mean(fold_rs), np.std(fold_rs)


class TrainingPipeline:
    """Main training pipeline orchestrator."""

    def __init__(self, work_dir: Path = WORK_DIR, cache_dir: Path = CACHE_DIR):
        self.work_dir = work_dir
        self.cache_dir = cache_dir
        self.data_loader = DataLoader(cache_dir)
        self.trainer = ModelTrainer(work_dir)

    def train(
        self, model_type: str = "ensemble", config: Dict = None
    ) -> TrainingResult:
        """Train a model."""
        if config is None:
            config = self._default_config(model_type)

        log.info(f"Loading data for {model_type} training...")
        X, y = self.data_loader.load()

        if X is None or y is None:
            raise ValueError("No training data found")

        X_train, X_val, X_test, y_train, y_val, y_test = self.data_loader.split_data(
            X, y
        )

        log.info(f"Split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

        log.info(f"Cross-validating {model_type}...")
        cv_r, cv_std = self.trainer.cross_validate(X, y, model_type, config)
        log.info(f"CV R: {cv_r:.4f} ± {cv_std:.4f}")

        log.info(f"Training {model_type}...")
        if model_type == "ridge":
            train_m, val_m, test_m, artifacts = self.trainer.train_ridge(
                X_train, y_train, X_val, y_val, X_test, y_test, config
            )
        elif model_type == "xgboost":
            train_m, val_m, test_m, artifacts = self.trainer.train_xgboost(
                X_train, y_train, X_val, y_val, X_test, y_test, config
            )
        else:
            train_m, val_m, test_m, artifacts = self.trainer.train_ensemble(
                X_train, y_train, X_val, y_val, X_test, y_test, config
            )

        log.info(f"Train R: {train_m['pearson_r']:.4f}, MAE: {train_m['mae']:.2f}")
        log.info(f"Val R: {val_m['pearson_r']:.4f}, MAE: {val_m['mae']:.2f}")
        log.info(f"Test R: {test_m['pearson_r']:.4f}, MAE: {test_m['mae']:.2f}")

        result = TrainingResult(
            model_type=model_type,
            train_r=train_m["pearson_r"],
            val_r=val_m["pearson_r"],
            test_r=test_m["pearson_r"],
            train_mae=train_m["mae"],
            val_mae=val_m["mae"],
            test_mae=test_m["mae"],
            gap=train_m["pearson_r"] - test_m["pearson_r"],
            cv_r=cv_r,
            cv_std=cv_std,
            config=config,
            n_train=len(X_train),
            n_val=len(X_val),
            n_test=len(X_test),
        )

        self.save_model(artifacts, model_type, result)

        return result

    def _default_config(self, model_type: str) -> Dict:
        """Get default config for model type."""
        base = {
            "k": 500,
            "n_estimators": 200,
            "max_depth": 6,
            "learning_rate": 0.05,
            "reg_alpha": 1.0,
            "reg_lambda": 5.0,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "seed": 42,
        }

        if model_type == "ridge":
            return {"k": 500, "alpha": 100.0}
        elif model_type == "ensemble":
            base["ensemble_weight"] = 0.8
            base["ridge_alpha"] = 100.0
            return base
        else:
            return base

    def save_model(self, artifacts: Dict, model_type: str, result: TrainingResult):
        """Save trained model to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"geock_{model_type}_{timestamp}.pkl"
        path = self.work_dir / filename

        model_data = {
            **artifacts,
            "model_type": model_type,
            "config": result.config,
            "cv_r": result.cv_r,
            "test_r": result.test_r,
            "train_r": result.train_r,
            "gap": result.gap,
            "mae": result.test_mae,
            "date": timestamp,
        }

        with open(path, "wb") as f:
            pickle.dump(model_data, f)

        results_path = self.work_dir / f"training_results_{timestamp}.json"
        with open(results_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)

        latest_path = self.work_dir / f"geock_{model_type}_latest.pkl"
        with open(latest_path, "wb") as f:
            pickle.dump(model_data, f)

        log.info(f"Model saved: {path}")
        log.info(f"Results saved: {results_path}")

    def compare_models(self) -> List[TrainingResult]:
        """Train and compare multiple model types."""
        results = []

        log.info("=" * 60)
        log.info("Comparing models...")
        log.info("=" * 60)

        for model_type in ["ridge", "xgboost", "ensemble"]:
            try:
                result = self.train(model_type)
                results.append(result)
                log.info(
                    f"{model_type}: Test R = {result.test_r:.4f}, Gap = {result.gap:.3f}"
                )
            except Exception as e:
                log.error(f"Failed to train {model_type}: {e}")

        best = max(results, key=lambda r: r.test_r)
        log.info(f"\nBest model: {best.model_type} (Test R = {best.test_r:.4f})")

        return results


def main():
    parser = argparse.ArgumentParser(description="GEOCK Model Training Pipeline")
    parser.add_argument(
        "--model",
        type=str,
        default="ensemble",
        choices=["ridge", "xgboost", "ensemble", "compare"],
        help="Model type to train",
    )
    parser.add_argument(
        "--config", type=str, default=None, help="Path to config JSON file"
    )
    parser.add_argument("--all", action="store_true", help="Run full pipeline")

    args = parser.parse_args()

    pipeline = TrainingPipeline()

    if args.config:
        with open(args.config) as f:
            config = json.load(f)
    else:
        config = None

    if args.model == "compare" or args.all:
        results = pipeline.compare_models()
    else:
        result = pipeline.train(args.model, config)
        print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    main()
