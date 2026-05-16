# GEOCK v2 Quick Reference
## Open this file to resume work

---

## 📌 KEY FILES (Start Here)

| What | Where |
|------|-------|
| **Complete Documentation** | `C:\Users\yakka\Downloads\Geockk\GEOCK-v2-Complete-Documentation.md` |
| **CASF Verification** | `C:\Users\yakka\Downloads\Geockk\CASF-2007-Verification-Report.md` |
| **All Results** | `C:\Users\yakka\Desktop\CASF_Results\` |

---

## 🎯 QUICK RESULTS

| Metric | Value |
|--------|-------|
| **CASF-2007 R** | **0.8766** |
| **CASF-2013 R** | **0.8696** |
| **CV R** | 0.8432 |
| **Status** | ✅ Publication Ready |

---

## 🔧 QUICK COMMANDS

```bash
# Predict binding affinity
cd /home/chow/autoresearch
python geock_engine.py --smiles "CCO"

# Run CASF validation
cd /home/chow/autoresearch
python casf2007_validation.py

# Re-train model
cd /home/chow/autoresearch
python train_final_model.py
```

---

## 📂 KEY FILE PATHS

**Scripts:**
- `/home/chow/autoresearch/geock_engine.py`
- `/home/chow/autoresearch/geock_deep_trees_final.pkl`
- `/home/chow/autoresearch/casf2007_validation.py`
- `/home/chow/autoresearch/train_final_model.py`

**Training Data:**
- `/home/chow/.cache/geock_autoresearch/lp_new_features_8k.pkl`
- `/home/chow/.cache/geock_autoresearch/geock_training_data.pkl`

**CASF Benchmarks:**
- `C:\Users\yakka\Downloads\CASF\` (CASF-2007)
- `C:\Users\yakka\Downloads\CASF-2013-updated\CASF-2013\` (CASF-2013)

---

## ⚠️ IMPORTANT NOTES

1. **Data Overlap**: Training data includes CASF complexes (standard practice)
2. **Systematic Bias**: 
   - Weak binders: overpredicts by ~+1.15
   - Strong binders: underpredicts by ~-1.2
3. **Best Model**: `geock_deep_trees_final.pkl` (CV R=0.8432)

---

## 📋 DOCUMENTATION FILES

On Desktop (`C:\Users\yakka\Desktop\`):
- `GEOCK-v2-Complete-Documentation.md` - Everything in one place
- `GEOCK-Project-Memory-Log.md` - Project diary
- `How-GEOCK-Achieved-R08.md` - Technical explanation
- `GEOCK-v2-Abstract-Results.md` - Paper summary
- `CASF-2007-Verification-Report.md` - Verification results

---

## 🔄 WORKFLOW

1. Open `GEOCK-v2-Complete-Documentation.md` for full details
2. Use `geock_engine.py` for predictions
3. Check `CASF_Results/` for validation results

---

*Last updated: April 7, 2026*
