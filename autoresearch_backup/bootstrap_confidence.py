#!/usr/bin/env python3
"""
Bootstrap confidence intervals for CASF-2016 test results
"""
import pickle
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("BOOTSTRAP CONFIDENCE INTERVALS")
print("="*60)

# Load test
with open('casf2016_enhanced_v2.pkl', 'rb') as f:
    test = pickle.load(f)
X_test = test['X'][:, :512]
y_test = test['y']

# Load training
with open('CACHE_DIR / geock_training_data_no2016.pkl', 'rb') as f:
    train = pickle.load(f)[:5000]  # Use subset for speed

X_train = np.array([d['ecfp'] for d in train])
y_train = np.array([d['affinity'] for d in train])

# Train model
print("\nTraining model...")
model = GradientBoostingRegressor(n_estimators=80, max_depth=4, 
                               learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# Bootstrap
print("Running bootstrap (n=1000)...")
n_bootstrap = 1000
r_values = []

np.random.seed(42)
for i in range(n_bootstrap):
    # Sample with replacement
    indices = np.random.choice(len(y_test), size=len(y_test), replace=True)
    y_sample = y_test[indices]
    pred_sample = predictions[indices]
    
    if len(np.unique(y_sample)) > 1:
        r, _ = pearsonr(y_sample, pred_sample)
        r_values.append(r)
    
    if (i+1) % 200 == 0:
        print(f"  {i+1}/{n_bootstrap}")

r_values = np.array(r_values)
r_mean = np.mean(r_values)
r_std = np.std(r_values)
ci_lower = np.percentile(r_values, 2.5)
ci_upper = np.percentile(r_values, 97.5)

print("\n" + "="*60)
print("RESULTS")
print("="*60)
print(f"Point estimate: R = {r_mean:.4f}")
print(f"Std dev: {r_std:.4f}")
print(f"95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")

# Save
result = {
    'R': r_mean,
    'std': r_std,
    'ci_95': (ci_lower, ci_upper),
    'n_bootstrap': n_bootstrap
}

with open('bootstrap_results.pkl', 'wb') as f:
    pickle.dump(result, f)

print("\nSaved bootstrap_results.pkl")
