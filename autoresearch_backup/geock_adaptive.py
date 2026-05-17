"""
GEOCK Adaptive Computation Layer
Inspired by Neural ODEs: trade precision for speed based on input difficulty
"""

import numpy as np
import pickle
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler

class AdaptiveGEOCK:
    """
    Two-tier model with adaptive computation:
    - Light model: Fast, for well-represented chemical space
    - Deep model: More compute, for novel scaffolds/targets
    """
    
    def __init__(self, light_model=None, deep_model=None, 
                 confidence_threshold=0.85, similarity_db=None):
        self.light_model = light_model
        self.deep_model = deep_model
        self.confidence_threshold = confidence_threshold
        self.similarity_db = similarity_db or {}
        
    def compute_similarity_context(self, ligand_fp):
        """Determine if ligand is in known chemical space."""
        if not self.similarity_db:
            return 1.0  # Full confidence - use light model
        
        # Compare to known training clusters
        max_sim = 0.0
        for cluster_fp in self.similarity_db.values():
            sim = 1 - np.sum(np.abs(ligand_fp.astype(bool) ^ cluster_fp.astype(bool))) / len(ligand_fp)
            max_sim = max(max_sim, sim)
            
        return max_sim
    
    def predict(self, ligand_fp, return_confidence=False):
        """
        Adaptive prediction:
        - Similar to training data → use light model
        - Novel scaffold → use deep model
        """
        context = self.compute_similarity_context(ligand_fp)
        
        # Prepare features
        X = ligand_fp.reshape(1, -1)
        if hasattr(self.light_model, 'scaler'):
            X = self.light_model.scaler.transform(X)
        if hasattr(self.light_model, 'selector'):
            X = self.light_model.selector.transform(X)
        
        # Adaptive selection
        if context >= self.confidence_threshold:
            # Light model sufficient
            pred = self.light_model.model.predict(X)[0]
            confidence = context
        else:
            # Novel region - use deep model
            pred = self.deep_model.model.predict(X)[0]
            confidence = context
            
        if return_confidence:
            return pred, confidence
        return pred
    
    def build_similarity_db(self, training_fps, n_clusters=50):
        """
        Build cluster centroids for chemical space coverage.
        Each cluster = representative fingerprint.
        """
        from sklearn.cluster import KMeans
        
        # Cluster training fingerprints
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(training_fps)
        
        # Store cluster centroids
        self.similarity_db = {}
        for i, center in enumerate(kmeans.cluster_centers_):
            self.similarity_db[f'cluster_{i}'] = center
            
        print(f"Built similarity DB with {n_clusters} cluster centroids")
        return self.similarity_db


class CascadeGEOCK:
    """
    Cascade approach: cheap model first, expensive if uncertain
    """
    
    def __init__(self, light_model, heavy_model, uncertainty_threshold=1.5):
        self.light = light_model
        self.heavy = heavy_model
        self.threshold = uncertainty_threshold
        
    def predict(self, fp):
        # Fast prediction
        light_pred, light_std = self._predict_with_uncertainty(self.light, fp)
        
        # Only use heavy if light is uncertain
        if light_std > self.threshold:
            heavy_pred, _ = self._predict_with_uncertainty(self.heavy, fp)
            return heavy_pred, 'heavy'
        return light_pred, 'light'
    
    def _predict_with_uncertainty(self, model, fp):
        # Using tree variance as uncertainty proxy
        # XGBoost can return leaf predictions for variance estimation
        X = fp.reshape(1, -1)
        if model.scaler:
            X = model.scaler.transform(X)
        if model.selector:
            X = model.selector.transform(X)
            
        pred = model.model.predict(X)[0]
        
        # Simple std estimate based on prediction magnitude
        # (real implementation would use ensemble variance)
        std = abs(pred - 7.0) / 3.0  #.normalize around pKd=7
        
        return pred, std


# Simple utility: check if ligand is novel
def is_novel_scaffold(test_fp, train_fps, novelty_threshold=0.7):
    """
    Quick check if ligand is novel compared to training set.
    
    Args:
        test_fp: 512-bit fingerprint
        train_fps: array of training fingerprints
        novelty_threshold: below this = novel
        
    Returns:
        bool: is_novel
    """
    # Compute max similarity to any training sample
    max_sim = 0
    for train_fp in train_fps:
        sim = 1 - np.sum(np.abs(test_fp.astype(bool) ^ train_fp.astype(bool))) / len(test_fp)
        max_sim = max(max_sim, sim)
        
    return max_sim < novelty_threshold


if __name__ == "__main__":
    # Quick test
    print("=== Adaptive GEOCK Demo ===")
    print("Concepts:")
    print("1. Light model + Deep model cascade")
    print("2. Chemical similarity DB for coverage")
    print("3. Novelty detection for compute allocation")
    print("\nUsage:")
    print("  model = CascadeGEOCK(light_model, heavy_model)")
    print("  pred, model_type = model.predict(fingerprint)")