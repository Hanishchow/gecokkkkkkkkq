#!/usr/bin/env python3
"""
GEOCK Deep Analysis and Ensemble Implementation
Phase 1: Data Foundation Analysis
"""

import pickle
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("PHASE 1: DATA FOUNDATION ANALYSIS")
print("="*70)

# Load training data
print("\n1. Loading training data...")
with open('CACHE_DIR / geock_training_data_no2016.pkl', 'rb') as f:
    train_list = pickle.load(f)

X_train = np.array([t['ecfp'] for t in train_list])
y_train = np.array([t['affinity'] for t in train_list])
train_pdb_ids = [t['pdb_id'] for t in train_list]
print(f"   Training: {X_train.shape[0]} samples, {X_train.shape[1]} features")
print(f"   Affinity range: {y_train.min():.1f} - {y_train.max():.1f}")

# Load CASF-2016 test
print("\n2. Loading CASF-2016 test data...")
with open('casf2016_enhanced_features.pkl', 'rb') as f:
    test_data = pickle.load(f)

X_test = test_data['X'][:, :512]  # Just ECFP4 for now
y_test = test_data['y']
test_pdb_ids = [cx['pdb_id'] for cx in test_data['complexes']]
print(f"   Test: {X_test.shape[0]} samples")

# Compute similarity matrix
print("\n3. Computing similarity between CASF-2016 and training...")
print("   This may take a moment...")

sims = cosine_similarity(X_test, X_train)
max_sim_per_test = sims.max(axis=1)
mean_top5_sim = np.sort(sims, axis=1)[:, -5:].mean(axis=1)

print(f"\n   Similarity Statistics:")
print(f"   Max similarity per test sample:")
print(f"     Mean: {max_sim_per_test.mean():.3f}")
print(f"     Median: {np.median(max_sim_per_test):.3f}")
print(f"     Min: {max_sim_per_test.min():.3f}")
print(f"     Max: {max_sim_per_test.max():.3f}")

# Categorize by similarity
low_sim = max_sim_per_test < 0.4
med_sim = (max_sim_per_test >= 0.4) & (max_sim_per_test < 0.6)
high_sim = max_sim_per_test >= 0.6

print(f"\n   Similarity Distribution:")
print(f"     Low (< 0.4): {low_sim.sum()} samples ({low_sim.mean()*100:.1f}%)")
print(f"     Medium (0.4-0.6): {med_sim.sum()} samples ({med_sim.mean()*100:.1f}%)")
print(f"     High (> 0.6): {high_sim.sum()} samples ({high_sim.mean()*100:.1f}%)")

# Analysis by binding strength
print("\n4. Binding strength distribution...")
weak = y_test < 5.0
medium = (y_test >= 5.0) & (y_test <= 9.0)
strong = y_test > 9.0

print(f"   Weak binders (pKd < 5): {weak.sum()}")
print(f"   Medium binders (5-9): {medium.sum()}")
print(f"   Strong binders (pKd > 9): {strong.sum()}")

# Save analysis
analysis = {
    'max_sim_per_test': max_sim_per_test,
    'mean_top5_sim': mean_top5_sim,
    'low_sim_mask': low_sim,
    'med_sim_mask': med_sim,
    'high_sim_mask': high_sim,
    'weak_binders': weak,
    'medium_binders': medium,
    'strong_binders': strong,
    'y_test': y_test,
    'sims_matrix': sims,
    'train_pdb_ids': train_pdb_ids,
    'test_pdb_ids': test_pdb_ids,
}

with open('phase1_analysis.pkl', 'wb') as f:
    pickle.dump(analysis, f)

print("\n5. Analysis saved to phase1_analysis.pkl")

# Key insight
print("\n" + "="*70)
print("KEY INSIGHTS")
print("="*70)
print(f"""
1. CASF-2016 has MEAN max similarity of {max_sim_per_test.mean():.2f} to training
   - This is MODERATE overlap (not clean split)
   - Literature shows R~0.59-0.65 is realistic for this level
   
2. Only {low_sim.sum()} samples are truly novel (sim < 0.4)
   - These will be hardest to predict
   - Need protein features for these

3. {high_sim.sum()} samples have high similarity (> 0.6)  
   - Should predict well with ECFP4 only
   - Risk of memorization
   
4. Strategy: Adaptive ensemble based on similarity
""")

print("Phase 1 complete. Ready for Phase 2.")