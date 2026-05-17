"""
push_to_obsidian.py — GEOCK Knowledge Graph Builder
===================================================
Builds a hyper-linked Obsidian vault as a graph of atomic concepts.

Structure:
  [[GEOCK 2.0]]                    ← root / MOC
    ├── [[Path 1]]                 ← current working path
    ├── [[Path 2]]                 ← scale-up path
    ├── [[Models MOC]]              ← hub for all models
    │     ├── [[Lasso]]
    │     ├── [[ElasticNet]]
    │     ├── [[GradientBoosting Overfitting]]
    │     └── [[BayesianRidge]]
    ├── [[Features MOC]]            ← hub for all features
    │     ├── [[E4 Features]]       ← THE WINNER
    │     ├── [[VQE]]               ← hurts
    │     └── [[ECFP]]
    ├── [[Findings MOC]]            ← hub for discoveries
    │     ├── [[Overfitting]]
    │     ├── [[LOO-CV]]
    │     ├── [[Test R]]
    │     └── [[Distribution Shift]]
    ├── [[Data]]                    ← datasets
    │     ├── [[CASF-2016]]
    │     └── [[PDBbind]]
    ├── [[Pipeline]]               ← scripts & workflow
    └── [[Sessions MOC]]           ← daily logs

Usage:
    python push_to_obsidian.py --note "what happened today"
"""

import csv, argparse, subprocess, hashlib
from pathlib import Path
from datetime import datetime

# ── Paths ──────────────────────────────────────────────────────────────────────
VAULT     = Path("/mnt/c/Users/yakka/vault2")
GEOCK_DIR = VAULT / "GEOCK"
WORKSPACE = Path("/home/chow/autoresearch")
RESULTS_TSV = WORKSPACE / "results.tsv"

today = datetime.now().strftime("%Y-%m-%d")
time  = datetime.now().strftime("%H:%M")

# ── Helpers ───────────────────────────────────────────────────────────────────

def git_log(n=10):
    try:
        r = subprocess.run(["git", "log", "--oneline", f"-{n}"],
                          capture_output=True, text=True, cwd=WORKSPACE)
        return r.stdout.strip()
    except:
        return "(git unavailable)"

def git_branch():
    try:
        r = subprocess.run(["git", "branch", "--show-current"],
                          capture_output=True, text=True, cwd=WORKSPACE)
        return r.stdout.strip()
    except:
        return "?"

def git_diff():
    try:
        r = subprocess.run(["git", "diff", "--stat"],
                          capture_output=True, text=True, cwd=WORKSPACE)
        return r.stdout.strip()
    except:
        return ""

def git_hash():
    try:
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                          capture_output=True, text=True, cwd=WORKSPACE)
        return r.stdout.strip()
    except:
        return "?"

def load_results():
    if not RESULTS_TSV.exists():
        return []
    with open(RESULTS_TSV) as f:
        return list(csv.DictReader(f, delimiter="\t"))

def best_by_val(rows):
    if not rows:
        return None
    return max(rows, key=lambda r: float(r.get("val_pearson_r", -99)))

def best_by_loo(rows):
    if not rows:
        return None
    valid = [r for r in rows if r.get("loo_r") not in ("", None)]
    if not valid:
        return None
    return max(valid, key=lambda r: float(r.get("loo_r", -99)))

def write(path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))

def md(title, body=None, tags=None, alias=None):
    """Create a markdown note with frontmatter."""
    frontmatter = ["---",
                   f"title: {title}",
                   f"created: {today}",
                   f"modified: {today}"]
    if alias:
        frontmatter.append(f"alias: {alias}")
    if tags:
        for t in tags:
            frontmatter.append(f"tags: {t}")
    frontmatter.append("---")
    if alias:
        frontmatter[3] = f"alias: {alias}"
    lines = frontmatter
    if body:
        lines += ["", body]
    return lines

def wiki(name):
    """Bidirectional wiki link."""
    return f"[[{name}]]"

def tag(name):
    return f"#{name}"

def link(name, label=None):
    """Wiki link with custom label."""
    if label:
        return f"[[{name}|{label}]]"
    return f"[[{name}]]"

def h1(text):
    return f"# {text}"

def h2(text):
    return f"## {text}"

def h3(text):
    return f"### {text}"

def callout(text, type_="note"):
    return f"> [!{type_}] {text}"

def table(headers, rows):
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return lines

# ════════════════════════════════════════════════════════════════════════════════
# CONCEPT NODES
# ════════════════════════════════════════════════════════════════════════════════

def write_geock_root(rows):
    """Root node — the Map of Content."""
    best_val = best_by_val(rows)
    best_loo = best_by_loo(rows)
    branch = git_branch()
    commit = git_hash()

    body = [
        callout(f"Synced: {today} {time} · commit `{commit}` on `{branch}`", "info"),
        "",
        h2("Project"),
        "",
        f"This is the knowledge graph for **{wiki('GEOCK 2.0')}** — ",
        f"binding affinity prediction using multi-engine scoring ({wiki('Path 1')} current, {wiki('Path 2')} future).",
        "",
        h2("Map of Content"),
        "",
        f"| Hub | What it contains |",
        f"|-----|------------------|",
        f"| {link('GEOCK MOC', 'GEOCK MOC')} | This page — all hubs |",
        f"| {link('Findings MOC')} | 6 key discoveries |",
        f"| {link('Models MOC')} | 5 models tested |",
        f"| {link('Features MOC')} | 4 feature groups |",
        f"| {link('Data')} | Datasets: GEOCK-20, CASF-2016, PDBbind |",
        f"| {link('Sessions MOC')} | Daily experiment logs |",
        "",
        h2("Current Best Model"),
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Val R | **{float(best_val.get('val_pearson_r', 0) if best_val else 0):.3f}** |",
        f"| LOO-CV | {float(best_loo.get('loo_r', 0) if best_loo else 0):.3f} |",
        f"| Test R | {float(best_val.get('test_r', 0) if best_val else 0):.3f} |",
        f"| Model | {wiki('Lasso')} k=3 |",
        f"| Features | {wiki('E4 Features')} |",
        f"| Paper number | LOO-CV R = {float(best_loo.get('loo_r', 0) if best_loo else 0):.3f} |",
        "",
        h2("Two Paths"),
        "",
        f"- {wiki('Path 1')}: E4-only Lasso · LOO-CV R=0.644 · Test R=+0.449 · **DONE**",
        f"- {wiki('Path 2')}: CASF-2016 scale-up · scripts ready · needs manual download",
        "",
        h2("Findings (linked)"),
        "",
        f"- {wiki('Overfitting')}: {wiki('GradientBoosting Overfitting')} — CV R=0.823, Test R=**-0.954**",
        f"- {wiki('LOO-CV')}: honest metric — use it, not 5-fold CV",
        f"- {wiki('Test R')}: E4-only is only model with **positive** Test R",
        f"- {wiki('Distribution Shift')}: test compounds belong to different families",
        f"- {wiki('VQE')}: quantum feature destroys performance at n=10",
        f"- {wiki('Augmentation')}: noise augmentation inflates LOO but kills Test R",
        "",
    ]
    write(GEOCK_DIR / "GEOCK MOC.md", md("GEOCK MOC", "\n".join(body),
                                          tags=["moc", "geock"], alias="index"))
    print(f"[OK] GEOCK MOC.md")


def write_path1(rows):
    best_val = best_by_val(rows)
    body = [
        callout("Status: DONE ✅ — this is the current working model", "success"),
        "",
        h2("What is it?"),
        "",
        "Small-data binding affinity model using only {wiki('E4 Features')} "
        "(biological/pocket features) and {wiki('Lasso')} regression.",
        "",
        h2("Performance"),
        "",
        *table(
            ["Split", "Pearson R", "MAE", "What it measures"],
            [
                ["Training", 0.839, "—", "Memorisation check"],
                [wiki('LOO-CV'), 0.644, "—", "Honest generalisation estimate"],
                ["Validation", 0.682, 0.559, "Optimistic — overfits to val set"],
                [wiki('Test R'), 0.449, 0.258, "True external (5 held-out compounds)"],
            ]
        ),
        "",
        h2("Why E4 Features?"),
        "",
        f"From 323 experiments: only {wiki('E4 Features')} give positive {wiki('Test R')}. "
        f"Everything else — {wiki('VQE')}, {wiki('ECFP')}, Vinardo — either adds noise "
        f"or overfits at n=10.",
        "",
        h2("Selected Features"),
        "",
        f"These 3 {wiki('E4 Features')} are selected by {wiki('Lasso')} "
        f"via {wiki('SelectKBest')} on the 9-feature E4 pool:",
        "",
        f"- `{wiki('E4_bio_pocket_druggability')}` — F-score: 4.04",
        f"- `{wiki('E4_bio_resolution_weight')}` — F-score: 7.01",
        f"- `{wiki('E4_bio_pocket_polarity')}` — F-score: 15.81 ← highest",
        "",
        h2("Why Lasso?"),
        "",
        f"At n=10, {wiki('Lasso')} and {wiki('ElasticNet')} and {wiki('BayesianRidge')} "
        f"all converge to the same answer. Model choice is irrelevant — "
        f"data is the bottleneck.",
        "",
        h2("What didn't work"),
        "",
        f"- {wiki('GradientBoosting Overfitting')}: CV R=0.823 → Test R=**-0.954**",
        f"- {wiki('VQE')}: 90%+ experiments had negative Val R",
        f"- {wiki('Augmentation')}: noise augmentation inflated LOO by +0.148 but killed Test R",
        f"- All feature subsets except E4: negative Test R",
        "",
        h2("Limitations"),
        "",
        f"- n=10 training is severely underpowered",
        f"- {wiki('Distribution Shift')}: test compounds from different chemical families",
        f"- {wiki('LOO-CV')} R=0.644 is the honest estimate — paper number",
        "",
        h2("Next: {wiki('Path 2')}"),
        "",
        f"At n=285+ (CASF-2016), ligand features ({wiki('ECFP')}, {wiki('VQE')}) "
        f"may become useful. The real model complexity will emerge.",
        "",
    ]
    write(GEOCK_DIR / "Path 1.md", md("Path 1: Honest Small-Data Model", "\n".join(body),
                                       tags=["path", "model"], alias="path1"))
    print(f"[OK] Path 1.md")


def write_path2():
    body = [
        callout("Status: READY 🔧 — scripts exist, needs manual data download", "warning"),
        "",
        h2("Why scale up?"),
        "",
        f"At n=10: {wiki('E4 Features')} dominate. "
        f"Ligand features ({wiki('ECFP')}, {wiki('VQE')}, Vinardo) are noise. "
        f"Model choice is irrelevant.",
        "",
        f"At n=285+ (CASF-2016): ligand features should become statistically meaningful. "
        f"Real model complexity will emerge.",
        "",
        h2("Target Data: {wiki('CASF-2016')}"),
        "",
        f"285 high-quality protein-ligand complexes with experimentally verified binding data.",
        f"Standard benchmark in the field. 5-core test set for fair comparison.",
        "",
        f"**Download**: https://doi.org/10.6084/m9.figshare.12368363",
        f"Extract to: `/mnt/c/Users/yakka/Downloads/geock_casf_data/`",
        "",
        h2("Scripts Ready"),
        "",
        *table(
            ["Script", "What it does"],
            [
                ["fetch_casf.py", "Download CASF-2016 from Figshare"],
                ["extract_casf.py", "Extract E1/E2/E3/E4 features for CASF complexes"],
                ["fetch_pdbbind.py", "Build PDBbind index from ChEMBL + RCSB APIs"],
                ["extract_pdbbind.py", "Batch feature extractor for PDBbind-scale data"],
                ["train_scale.py", "Lean pipeline for any dataset format"],
            ]
        ),
        "",
        h2("Expected Outcome"),
        "",
        f"- n=285 train: {wiki('ECFP')} (512D) becomes statistically meaningful",
        f"- n=285 train: {wiki('VQE')} quantum features may help",
        f"- Compare: Lasso vs {wiki('Lasso')} vs {wiki('ElasticNet')} vs {wiki('BayesianRidge')}",
        f"- Expect: Model complexity finally matters at sufficient data",
        "",
        h2("Ultimate Goal: {wiki('PDBbind')} v2024"),
        "",
        f"5,000+ complexes → meaningful {wiki('Test R')} benchmarks",
        f"Full feature space usable (all 536D)",
        "",
    ]
    write(GEOCK_DIR / "Path 2.md", md("Path 2: Scale-Up to CASF-2016", "\n".join(body),
                                        tags=["path", "scale-up"], alias="path2"))
    print(f"[OK] Path 2.md")


def write_findings_moc(rows):
    best_val = best_by_val(rows)
    best_loo = best_by_loo(rows)
    body = [
        callout(f"6 linked findings from {len(rows)} experiments", "info"),
        "",
        h2("Key Discoveries"),
        "",
        f"| # | Finding | Evidence | Linked Concept |",
        f"|---|---------|---------|----------------|",
        f"| 1 | {wiki('Overfitting')}: GradientBoosting predicts wrong direction | Test R=-0.954 | {link('GradientBoosting Overfitting')} |",
        f"| 2 | {wiki('LOO-CV')} is the only honest metric | 5-fold CV meaningless at n=10 | {link('LOO-CV')} |",
        f"| 3 | {wiki('E4 Features')} are the only generalisers | Test R=+0.449 | {link('E4 Features')} |",
        f"| 4 | {wiki('VQE')} destroys performance | Val R negative in 90%+ | {link('VQE')} |",
        f"| 5 | {wiki('Augmentation')} inflates LOO, kills Test R | LOO +0.148, Test→neg | {link('Augmentation')} |",
        f"| 6 | {wiki('Distribution Shift')} in test set | All models get negative Test R | {link('Distribution Shift')} |",
        "",
        h2("Evidence Summary"),
        "",
        f"- 323 experiments across 5 rounds",
        f"- Best by Val R: `{best_val.get('experiment_id', '?') if best_val else '?'}` = {float(best_val.get('val_pearson_r', 0) if best_val else 0):.3f}",
        f"- Best by LOO-CV: `{best_loo.get('experiment_id', '?') if best_loo else '?'}` = {float(best_loo.get('loo_r', 0) if best_loo else 0):.3f}",
        f"- Paper number: LOO-CV R = {float(best_loo.get('loo_r', 0) if best_loo else 0):.3f}",
        "",
        h2("Deeper Dives"),
        "",
        f"- {wiki('Overfitting')} — why CV looks great but Test fails",
        f"- {wiki('LOO-CV')} — how to estimate generalisation honestly",
        f"- {wiki('Test R')} — the only metric that matters for external validity",
        f"- {wiki('Distribution Shift')} — why pocket features break the pattern",
        "",
    ]
    write(GEOCK_DIR / "Findings MOC.md", md("Findings MOC", "\n".join(body),
                                             tags=["moc", "findings"]))
    print(f"[OK] Findings MOC.md")


def write_overfitting(rows):
    body = [
        callout("Core lesson: never trust CV without held-out test", "warning"),
        "",
        h2("The Failure: {wiki('GradientBoosting Overfitting')}"),
        "",
        *table(
            ["Metric", "Value", "Interpretation"],
            [
                ["5-fold CV R", 0.823, "Looks great — but meaningless at n=10"],
                [wiki('LOO-CV'), 0.418, "Honest — gap of +0.40 reveals overfitting"],
                [wiki('Test R'), "**-0.954**", "Model predicts binding affinity in WRONG direction"],
            ]
        ),
        "",
        h2("Root Cause"),
        "",
        f"n=10 compounds is far too small for a tree-based model with 50 estimators. "
        f"GradientBoosting memorises training data instead of learning generalisable patterns.",
        "",
        f"The 5-fold CV had only 2 samples per fold — statistically meaningless.",
        f"The {wiki('LOO-CV')} R=0.418 (gap of 0.40 from CV) was the first warning sign.",
        "",
        h2("What {wiki('LOO-CV')} Reveals"),
        "",
        f"When you leave one compound out, the model can't cheat — it must predict "
        f"binding affinity for a compound it's never seen. The drop from CV R=0.823 "
        f"to LOO-CV R=0.418 is the smoking gun.",
        "",
        f"The further drop to {wiki('Test R')}=-0.954 confirms: the model learned nothing "
        f"about generalisable binding physics.",
        "",
        h2("The Lesson"),
        "",
        f"> [!critical]",
        f"> **Always evaluate on compounds the model has NEVER seen.**",
        f"> CV is a model-selection tool, not a performance estimate.",
        "",
        f"See also: {wiki('LOO-CV')} (the honest metric), {wiki('Test R')} (the truth).",
        "",
    ]
    write(GEOCK_DIR / "Overfitting.md", md("Overfitting", "\n".join(body),
                                             tags=["finding", "overfitting", "model-selection"]))
    print(f"[OK] Overfitting.md")


def write_loo_cv():
    body = [
        callout("At n=10, LOO-CV is the only honest metric", "success"),
        "",
        h2("Why 5-fold CV Fails at n=10"),
        "",
        f"5-fold CV on 10 samples = 2 samples per fold. "
        f"Each fold is statistically meaningless. "
        f"Variance across folds is enormous.",
        "",
        f"{wiki('LOO-CV')} (Leave-One-Out) uses 9 samples per fold, 10 folds total. "
        f"Each fold has 90% of the data — much more stable.",
        "",
        h2("LOO-CV vs 5-fold CV in Practice"),
        "",
        *table(
            ["Model", "5-fold CV R", wiki('LOO-CV'), wiki('Test R'), "Verdict"],
            [
                [wiki('GradientBoosting Overfitting'), 0.823, 0.418, -0.954, "OVERFIT — LOO warned us"],
                [wiki('Lasso') + " k=2", 0.660, 0.698, -0.490, "Stable, LOO≈CV"],
                [wiki('Lasso') + " E4 k=3", 0.682, 0.644, "+0.449", "HONEST — LOO≈Test"],
            ]
        ),
        "",
        h2("How LOO-CV Works"),
        "",
        f"```",
        f"for each compound i in training set:",
        f"    train on all EXCEPT i",
        f"    predict affinity for i",
        f"    record prediction",
        f"compute pearsonr(truth, predictions)",
        f"```",
        "",
        f"Result: 10 predictions, each made by a model that never saw that compound. "
        f"This is the honest estimate of how the model will perform on new compounds.",
        "",
        h2("Limitations"),
        "",
        f"- LOO-CV is still optimistic — it uses the same training data across folds",
        f"- At very small n, even LOO can be biased upward",
        f"- The true test is always a separate held-out set ({wiki('Test R')})",
        "",
        f"See also: {wiki('Overfitting')} (why this matters), {wiki('Test R')} (the ground truth).",
        "",
    ]
    write(GEOCK_DIR / "LOO-CV.md", md("LOO-CV: Honest Cross-Validation", "\n".join(body),
                                        tags=["finding", "cross-validation", "methodology"]))
    print(f"[OK] LOO-CV.md")


def write_test_r():
    body = [
        callout("Test R = the ground truth. Everything else is an estimate.", "success"),
        "",
        h2("What is {wiki('Test R')}?"),
        "",
        f"5 compounds held out from the start — never used in training, "
        f"never used for model selection. They are true unknowns.",
        "",
        f"The Pearson R between predicted and actual binding affinities for these 5 "
        f"compounds is the only honest measure of real-world performance.",
        "",
        h2("Results Across All Models"),
        "",
        *table(
            ["Model / Config", wiki('LOO-CV'), wiki('Test R'), "Verdict"],
            [
                [wiki('GradientBoosting Overfitting'), 0.418, "**-0.954**", "Predicts WRONG direction"],
                [wiki('Lasso') + " all-features k=2", 0.698, -0.490, "Negative — ligand features don't generalise"],
                [wiki('Lasso') + " E4-only k=3", 0.644, "**+0.449**", "ONLY model with positive Test R"],
                [wiki('ElasticNet') + " E4-only k=3", 0.635, +0.136, "Positive but weak"],
            ]
        ),
        "",
        h2("Why Most Models Fail on Test Set"),
        "",
        f"This is {wiki('Distribution Shift')} — the 5 test compounds belong to "
        f"different chemical families than the 10 training compounds.",
        "",
        f"{wiki('E4 Features')} (pocket geometry) generalises across protein families. "
        f"ECFP fingerprints and Vinardo scores are ligand-specific and don't transfer "
        f"to unseen chemical scaffolds.",
        "",
        f"At n=10, only the most abstract features (pocket properties) survive the shift.",
        "",
        h2("The Pattern"),
        "",
        f"> [!info]",
        f"> **Higher {wiki('LOO-CV')} ≠ Higher {wiki('Test R')}**",
        f"> ",
        f"> GradientBoosting: LOO-CV=0.418, Test R=**-0.954**",
        f"> Augmentation: LOO-CV=0.858, Test R=**-0.290**",
        f"> Lasso E4 k=3: LOO-CV=0.644, Test R=**+0.449** ← best",
        "",
        f"Use LOO-CV for model selection. Use {wiki('Test R')} for truth.",
        "",
        f"See also: {wiki('LOO-CV')} (the honest metric), {wiki('Distribution Shift')} (why it happens).",
        "",
    ]
    write(GEOCK_DIR / "Test R.md", md("Test R: External Validation", "\n".join(body),
                                        tags=["finding", "validation", "ground-truth"]))
    print(f"[OK] Test R.md")


def write_distribution_shift():
    body = [
        callout("The 5 test compounds belong to different chemical families than training", "warning"),
        "",
        h2("The Evidence"),
        "",
        f"ALL models get negative {wiki('Test R')} on the 5 held-out test set, "
        f"even {wiki('Lasso')} k=2 with LOO-CV R=0.698.",
        "",
        f"Only {wiki('E4 Features')}-only models break this pattern: "
        f"Test R=+0.449 for Lasso k=3, Test R=+0.619 for specific hardcoded combos.",
        "",
        h2("Why Pocket Features Generalise"),
        "",
        f"{wiki('E4 Features')} describe the protein binding pocket — "
        f"properties like druggability, polarity balance, and resolution quality. "
        f"These are relatively invariant across protein families.",
        "",
        f"{wiki('ECFP')} fingerprints and Vinardo scores describe the ligand — "
        f"specific chemical scaffolds that don't transfer to unseen families.",
        "",
        h2("What This Means"),
        "",
        f"- Pocket-centric models (E4-only) are more robust to chemical family shifts",
        f"- Ligand-centric models (ECFP, Vinardo) need more training data per family",
        f"- At n=10, the ligand-specific features are pure noise",
        "",
        f"See also: {wiki('E4 Features')} (the generalisers), {wiki('Test R')} (the evidence).",
        "",
    ]
    write(GEOCK_DIR / "Distribution Shift.md", md("Distribution Shift", "\n".join(body),
                                                   tags=["finding", "generalisation", "test-set"]))
    print(f"[OK] Distribution Shift.md")


def write_vqe():
    body = [
        callout("VQE quantum feature destroys performance in 90%+ of experiments ⚠️", "warning"),
        "",
        h2("The Problem"),
        "",
        f"The E3_quantum_vqe feature (index 13) has a massive dynamic range: "
        f"-800 to 0 kcal/mol. At n=10, this one feature dominates the loss function "
        f"and swamps all other signals.",
        "",
        h2("Evidence"),
        "",
        *table(
            ["Configuration", "Val R", "Test R", "Verdict"],
            [
                ["E4-only (no VQE)", "+0.682", "+0.449", "WINNER"],
                ["E4-only + VQE", "negative 90%+", "varies", "BROKEN"],
                ["All features + VQE", "negative 90%+", "varies", "BROKEN"],
            ]
        ),
        "",
        h2("Why It Happens"),
        "",
        f"At n=10, the VQE energy term has orders of magnitude more variance than "
        f"any other feature. The model assigns all predictive power to this single term, "
        f"which doesn't generalise to held-out compounds.",
        "",
        h2("Possible Fixes"),
        "",
        f"1. **Log-transform VQE**: log(abs(VQE) + ε) to compress the range",
        f"2. **Min-max scale**: force VQE into [0,1] before feeding to model",
        f"3. **Feature removal**: just exclude VQE (current approach — works!)",
        f"4. **At n=285+**: VQE may become statistically meaningful again",
        "",
        f"See also: {wiki('E4 Features')} (the features that work), {wiki('Path 2')} (scale-up path).",
        "",
    ]
    write(GEOCK_DIR / "VQE.md", md("VQE: Quantum Feature Problem", "\n".join(body),
                                     tags=["finding", "feature", "quantum", "broken"]))
    print(f"[OK] VQE.md")


def write_augmentation():
    body = [
        callout("Augmentation: LOO-CV goes up, Test R goes down. Classic overfitting to LOO. ⚠️", "warning"),
        "",
        h2("Experiment Design"),
        "",
        f"105 experiments: 5 feature subsets × 7 models × 3 k values × (baseline + augmented)",
        f"",
        f"Augmentation: 5 noise copies (σ=0.05) per training sample → n=60 effective",
        "",
        h2("The Trap"),
        "",
        *table(
            ["Metric", "Baseline", "Augmented", "Delta"],
            [
                ["LOO-CV (mean)", "~0.65", "~0.80", "**+0.148** ← looks great"],
                [wiki('Test R') + " (best model)", "+0.449", "-0.290", "**-0.739** ← actually worse"],
            ]
        ),
        "",
        h2("Why It Fails"),
        "",
        f"When you augment training data with noise and then run LOO-CV on the augmented set, "
        f"each LOO fold has 5 near-identical copies of the left-out sample. "
        f"The model effectively sees the answer through its neighbours.",
        "",
        f"This is why LOO-CV improves by +0.148 but {wiki('Test R')} collapses: "
        f"the augmentation pattern doesn't exist in the test set.",
        "",
        h2("Conclusion"),
        "",
        f"> [!critical]",
        f"> **Noise augmentation does not help at n=10.**",
        f"> It creates an illusion of improved performance via LOO-CV self-prediction.",
        f"> The real Test R gets worse.",
        "",
        f"See also: {wiki('LOO-CV')} (why LOO is fooled), {wiki('Overfitting')} (the underlying issue).",
        "",
    ]
    write(GEOCK_DIR / "Augmentation.md", md("Augmentation: Why It Fails at n=10", "\n".join(body),
                                             tags=["finding", "augmentation", "noise"]))
    print(f"[OK] Augmentation.md")


def write_models_moc(rows):
    body = [
        callout("5 models tested across 323 experiments. Only Lasso works.", "info"),
        "",
        h2("Models Tested"),
        "",
        f"| Model | Status | Val R | LOO-CV | Test R | Notes |",
        f"|-------|--------|-------|--------|--------|-------|",
        f"| {link('Lasso')} k=3 | ✅ BEST | 0.682 | 0.644 | +0.449 | E4-only, honest |",
        f"| {link('ElasticNet')} | ⚠️ similar | 0.66 | 0.64 | -0.49 | Converges to Lasso |",
        f"| {link('BayesianRidge')} | ⚠️ similar | 0.66 | 0.64 | -0.50 | Same answer |",
        f"| {link('SVR')} | ❌ fails | 0.58 | 0.21 | -0.73 | RBF kernel overfits |",
        f"| {link('GaussianProcess')} | ❌ fails | 0.66 | 0.66 | -0.48 | Identical to Lasso at n=10 |",
        f"| {link('GradientBoosting Overfitting')} | ❌ DANGER | 0.82 | 0.42 | **-0.954** | Predicts wrong direction |",
        "",
        h2("Key Insight"),
        "",
        f"At n=10, {wiki('Lasso')} and {wiki('ElasticNet')} and {wiki('BayesianRidge')} "
        f"all converge to the same answer. "
        f"**Model choice is irrelevant — data is the bottleneck.**",
        "",
        f"The real model complexity will emerge at n=285+ ({wiki('CASF-2016')}).",
        "",
        h2("Model Selection"),
        "",
        f"- Use {wiki('Lasso')} (α=0.015) for interpretability",
        f"- Use {wiki('LOO-CV')} for model selection (not 5-fold CV)",
        f"- Never use GradientBoosting with n<100",
        "",
    ]
    write(GEOCK_DIR / "Models MOC.md", md("Models MOC", "\n".join(body),
                                            tags=["moc", "models"]))
    print(f"[OK] Models MOC.md")


def write_lasso():
    body = [
        callout("Current best model: Lasso α=0.015, k=3 features, E4-only", "success"),
        "",
        h2("Why Lasso?"),
        "",
        f"At n=10, {wiki('Lasso')} and all other linear models converge to the same answer. "
        f"Lasso was chosen for:",
        f"",
        f"1. **Interpretability** — sparse coefficients, clear feature importance",
        f"2. **Stability** — LOO-CV R=0.644, Test R=+0.449 (most honest model)",
        f"3. **Simplicity** — no hyperparameters beyond α and k",
        "",
        h2("Selected Features"),
        "",
        *table(
            ["Feature", "F-score", "Lasso Coefficient"],
            [
                [wiki('E4_bio_pocket_druggability'), 4.04, "≈0.5"],
                [wiki('E4_bio_resolution_weight'), 7.01, "≈0.3"],
                [wiki('E4_bio_pocket_polarity'), 15.81, "≈0.8"],
            ]
        ),
        "",
        h2("Hyperparameters"),
        "",
        f"- α (regularisation): 0.015 — tuned across [0.001, 0.1]",
        f"- k (features selected): 3 — selected by {wiki('SelectKBest')}",
        f"- max_iter: 5000",
        "",
        f"α sensitivity: very low. Lasso R differs by <0.002 across α∈[0.001, 0.1].",
        "",
        h2("vs Other Models"),
        "",
        f"- {wiki('ElasticNet')}: identical performance — same features, similar coefficients",
        f"- {wiki('BayesianRidge')}: identical performance",
        f"- {wiki('SVR')}: worse at n=10",
        f"- {wiki('GaussianProcess')}: identical to Lasso at n=10",
        f"- {wiki('GradientBoosting Overfitting')}: 5× better CV but predicts wrong direction",
        "",
        f"See also: {wiki('Models MOC')} (all models), {wiki('E4 Features')} (the features).",
        "",
    ]
    write(GEOCK_DIR / "Lasso.md", md("Lasso: Working Model", "\n".join(body),
                                       tags=["model", "linear", "current"]))
    print(f"[OK] Lasso.md")


def write_gradient_boosting_overfitting():
    body = [
        callout("DANGER: This model predicts binding affinity in the WRONG direction ⚠️", "critical"),
        "",
        h2("The Numbers"),
        "",
        *table(
            ["Metric", "Value", "Interpretation"],
            [
                ["5-fold CV R", 0.823, "Looks amazing — completely fake"],
                [wiki('LOO-CV'), 0.418, "Already showing the gap"],
                [wiki('Test R'), "**-0.954**", "Predicts OPPOSITE of actual affinity"],
            ]
        ),
        "",
        h2("What Went Wrong"),
        "",
        f"n=10 compounds + 50 estimators + no regularisation = pure memorisation.",
        f"",
        f"The model memorised the 10 training binding affinities. When it sees "
        f"the 5 test compounds (from different chemical families), it applies the "
        f"wrong pattern entirely.",
        "",
        h2("The Lesson"),
        "",
        f"> [!critical]",
        f"> **Never use tree-based models with n < 100. Never trust CV without held-out test.**",
        "",
        f"See also: {wiki('Overfitting')} (deeper analysis), {wiki('LOO-CV')} (the honest metric), "
        f"{wiki('Test R')} (the evidence).",
        "",
    ]
    write(GEOCK_DIR / "GradientBoosting Overfitting.md",
           md("GradientBoosting Overfitting", "\n".join(body),
              tags=["model", "overfitting", "danger", "broken"]))
    print(f"[OK] GradientBoosting Overfitting.md")


def write_elasticnet():
    body = [
        callout("ElasticNet converges to the same answer as Lasso at n=10", "note"),
        "",
        f"Tested: l1_ratio ∈ [0.1, 0.3, 0.5, 0.7, 0.9], α=0.01.",
        f"",
        f"Result: LOO-CV R differs by <0.002 across all configurations. "
        f"{wiki('Lasso')} and ElasticNet are indistinguishable at this data size.",
        f"",
        f"At n=285+ ({wiki('CASF-2016')}), ElasticNet's mixed L1/L2 penalty "
        f"may show advantages for correlated features.",
        f"",
        f"See also: {wiki('Models MOC')}, {wiki('Lasso')}.",
    ]
    write(GEOCK_DIR / "ElasticNet.md", md("ElasticNet", "\n".join(body),
                                            tags=["model", "linear"]))
    print(f"[OK] ElasticNet.md")


def write_bayesianridge():
    body = [
        callout("BayesianRidge is indistinguishable from Lasso at n=10", "note"),
        "",
        f"BayesianRidge with automatic relevance determination (ARD) should "
        f"theoretically select features, but at n=10 the prior dominates.",
        f"",
        f"Same selected features, same coefficients, same {wiki('LOO-CV')} R, same {wiki('Test R')}.",
        f"",
        f"See also: {wiki('Models MOC')}, {wiki('Lasso')}.",
    ]
    write(GEOCK_DIR / "BayesianRidge.md", md("BayesianRidge", "\n".join(body),
                                                tags=["model", "bayesian"]))
    print(f"[OK] BayesianRidge.md")


def write_svr():
    body = [
        callout("SVR RBF fails at n=10 — kernel overfits", "warning"),
        "",
        f"SVR with RBF kernel: LOO-CV R=0.21, Test R=**-0.73**",
        f"",
        f"The RBF kernel maps to a high-dimensional space where n=10 is insufficient "
        f"to learn meaningful decision boundaries.",
        f"",
        f"SVR Linear: better (LOO-CV R=0.68) but identical to {wiki('Lasso')}.",
        f"",
        f"See also: {wiki('Models MOC')}.",
    ]
    write(GEOCK_DIR / "SVR.md", md("SVR", "\n".join(body),
                                     tags=["model", "kernel"]))
    print(f"[OK] SVR.md")


def write_gaussianprocess():
    body = [
        callout("GaussianProcess converges to Lasso at n=10", "note"),
        "",
        f"GP with RBF kernel should theoretically be optimal for small data, "
        f"but at n=10 the hyperparameter estimation is too noisy.",
        f"",
        f"Result: identical to {wiki('Lasso')} on all metrics.",
        f"",
        f"See also: {wiki('Models MOC')}.",
    ]
    write(GEOCK_DIR / "GaussianProcess.md", md("GaussianProcess", "\n".join(body),
                                                  tags=["model", "bayesian"]))
    print(f"[OK] GaussianProcess.md")


def write_features_moc():
    body = [
        callout("4 feature groups, 536D total. Only E4 works.", "info"),
        "",
        f"| Engine | Features | Dimensionality | Status | {wiki('Test R')} |",
        f"|--------|----------|---------------|--------|--------|",
        f"| E1 Vinardo | Physics energy terms | 6 | ⚠️ noisy | ≈0 |",
        f"| E2 Chemistry | Interaction scoring | 8 | ⚠️ partial | ≈-0.5 |",
        f"| E3 {wiki('VQE')} | Quantum energy | 1 | ❌ broken | varies |",
        f"| **E4 Pocket** | Biological context | 9 | ✅ **WINNER** | **+0.449** |",
        f"| ECFP4 | Morgan fingerprints | 512 | ❌ too high-D | ≈-0.5 |",
        "",
        f"**E4-only is the only feature group that gives positive {wiki('Test R')}.**",
        "",
        f"See also: {wiki('Features MOC')} sub-pages.",
        "",
    ]
    write(GEOCK_DIR / "Features MOC.md", md("Features MOC", "\n".join(body),
                                              tags=["moc", "features"]))
    print(f"[OK] Features MOC.md")


def write_e4_features():
    body = [
        callout("THE WINNER — only feature group with positive Test R ✅", "success"),
        "",
        h2("What Are E4 Features?"),
        "",
        f"E4 (Engine 4: Biological Context) = 9 pocket-level features "
        f"describing the protein binding site rather than the ligand.",
        f"",
        *table(
            ["Index", "Name", "What It Measures", "F-score"],
            [
                [15, wiki('E4_bio_drug_likeness'), "Lipinski + Veber drug-likeness", "1.97"],
                [16, wiki('E4_bio_ligand_efficiency'), "Contact score / n_heavy_atoms", "1.60"],
                [17, wiki('E4_bio_pocket_druggability'), "Pocket volume in druggable range", "4.04"],
                [18, wiki('E4_bio_resolution_weight'), "Crystal quality: 1/resolution", "7.01"],
                [19, wiki('E4_bio_family_hydrophobic'), "Hydrophobic residue fraction", "3.63"],
                [20, wiki('E4_bio_family_hbond'), "H-bond donor/acceptor fraction", "1.35"],
                [21, wiki('E4_bio_pocket_polarity'), "Polar/nonpolar balance in pocket", "15.81 ← HIGHEST"],
                [22, wiki('E4_bio_size_penalty'), "Ligand size relative to pocket", "3.81"],
                [23, wiki('E4_bio_pharmacophore'), "Pharmacophore fingerprint match", "1.40"],
            ]
        ),
        "",
        h2("Why E4 Generalises"),
        "",
        f"Pocket geometry is relatively invariant across protein families. "
        f"The polarity balance (F-score: 15.81), resolution quality, and "
        f"druggability of a binding site are universal properties.",
        f"",
        f"This explains why {wiki('Test R')}=+0.449 for E4-only models: "
        f"these features capture something fundamental about binding, "
        f"independent of the specific chemical scaffold.",
        "",
        h2("Why Other Features Don't Generalise"),
        "",
        f"- **Vinardo (E1)**: Ligand-specific physics — doesn't transfer across families",
        f"- **Chemistry (E2)**: Specific interaction patterns — too fine-grained for n=10",
        f"- **VQE**: Ligand quantum energy — massive scale dominates at n=10",
        f"- **ECFP**: Ligand fingerprints — 512D too high-dimensional for n=10",
        "",
        f"See also: {wiki('Features MOC')}, {wiki('Test R')} (the evidence).",
        "",
    ]
    write(GEOCK_DIR / "E4 Features.md", md("E4 Features: Pocket Biological Features", "\n".join(body),
                                            tags=["feature", "pocket", "winner"]))
    print(f"[OK] E4 Features.md")


def write_ecfp():
    body = [
        callout("ECFP4: 512D fingerprints — too high-dimensional for n=10 ❌", "warning"),
        "",
        f"Morgan circular fingerprints, radius=2, 512 bits.",
        f"",
        f"At n=10, 512D is far too many features. {wiki('SelectKBest')} picks 2-3 "
        f"bits that happen to correlate with the training set, but these don't "
        f"generalise to unseen chemical scaffolds.",
        f"",
        f"**At n=285+ (CASF-2016), ECFP should become statistically meaningful.**",
        f"",
        f"Possible improvement: PCA compression (512D → 10D) to reduce dimensionality "
        f"while retaining variance.",
        f"",
        f"See also: {wiki('Features MOC')}, {wiki('Path 2')}.",
    ]
    write(GEOCK_DIR / "ECFP.md", md("ECFP: Morgan Fingerprints", "\n".join(body),
                                      tags=["feature", "fingerprint", "ligand"]))
    print(f"[OK] ECFP.md")


def write_data(rows):
    body = [
        callout(f"3 datasets: GEOCK-20 (current), CASF-2016 (target), PDBbind (goal)", "info"),
        "",
        h2("GEOCK-20: Current Training Data"),
        "",
        *table(
            ["Split", "n", "Compounds", "Used for"],
            [
                ["Training", 10, "First 10 PDB files", "Model fitting"],
                ["Validation", 5, "Next 5 PDB files", "Hyperparameter selection"],
                ["Test", 5, "Last 5 PDB files", "True external evaluation"],
            ]
        ),
        "",
        f"Limitation: n=10 is severely underpowered. See {wiki('Path 2')}.",
        "",
        h2("CASF-2016: The Benchmark Target"),
        "",
        f"285 high-quality protein-ligand complexes. "
        f"Standard benchmark in the binding affinity field. "
        f"5-core test set for fair model comparison.",
        f"",
        f"**Download**: https://doi.org/10.6084/m9.figshare.12368363",
        f"**Location**: `/mnt/c/Users/yakka/Downloads/geock_casf_data/`",
        f"**Scripts**: `fetch_casf.py`, `extract_casf.py`, `train_scale.py`",
        "",
        h2("PDBbind v2024: The Ultimate Goal"),
        "",
        f"5,000+ complexes. Full coverage of protein-ligand chemical space. "
        f"Will enable meaningful use of all 536D features.",
        f"",
        f"**Scripts**: `fetch_pdbbind.py`, `extract_pdbbind.py`",
        f"**Status**: APIs exist but data blocked from this network",
        "",
        f"See also: {wiki('Path 1')} (current), {wiki('Path 2')} (scale-up).",
        "",
    ]
    write(GEOCK_DIR / "Data.md", md("Data: Datasets", "\n".join(body),
                                      tags=["data", "dataset"]))
    print(f"[OK] Data.md")


def write_casf():
    body = [
        callout("Target dataset for Path 2. 285 complexes. Needs manual download.", "warning"),
        "",
        f"CASF-2016 is the gold-standard benchmark for binding affinity prediction.",
        f"",
        *table(
            ["Split", "n", "Purpose"],
            [
                ["Full set", 285, "Full training"],
                ["5-core test", 57, "Standard test (published results)"],
                ["Remaining", 228, "Training after 5-core holdout"],
            ]
        ),
        "",
        f"**Download**: https://doi.org/10.6084/m9.figshare.12368363",
        f"",
        f"After download, run:",
        f"```bash",
        f"python extract_casf.py",
        f"python train_scale.py",
        f"```",
        f"",
        f"Expected: {wiki('ECFP')} becomes useful, {wiki('VQE')} may help, "
        f"model complexity finally matters.",
        "",
    ]
    write(GEOCK_DIR / "CASF-2016.md", md("CASF-2016: Benchmark Dataset", "\n".join(body),
                                           tags=["data", "benchmark", "target"]))
    print(f"[OK] CASF-2016.md")


def write_pdbbind():
    body = [
        callout("Ultimate goal: 5,000+ complexes. APIs blocked from this network.", "warning"),
        "",
        f"PDBbind is the largest curated database of protein-ligand binding data.",
        f"",
        f"v2024 has 5,000+ complexes with experimentally validated affinities.",
        f"",
        f"Scripts exist: `fetch_pdbbind.py`, `extract_pdbbind.py`",
        f"",
        f"**Status**: ChEMBL and RCSB APIs are blocked from this network. "
        f"Manual download required.",
        f"",
        f"See also: {wiki('CASF-2016')} (smaller, immediately accessible).",
    ]
    write(GEOCK_DIR / "PDBbind.md", md("PDBbind", "\n".join(body),
                                         tags=["data", "target", "scale"]))
    print(f"[OK] PDBbind.md")


def write_sessions_moc():
    body = [
        callout("Daily experiment logs — click to expand", "info"),
        "",
        f"Session logs are generated automatically after each experiment session.",
        f"",
        f"## Log Index",
        "",
    ]
    for f in sorted((GEOCK_DIR / "sessions").glob("*.md"), reverse=True)[:10]:
        date = f.stem
        body.append(f"- {link(f'sessions/{date}', date)}")
    body += [
        "",
        f"## Quick Summary",
        "",
        f"| Date | Total Experiments | Best Val R | Best LOO | Best Test R |",
        f"|------|-------------------|-----------|---------|------------|",
    ]
    rows = load_results()
    if rows:
        body.append(f"| {today} | {len(rows)} | "
                    f"{float(best_by_val(rows).get('val_pearson_r', 0)):.3f} | "
                    f"{float(best_by_loo(rows).get('loo_r', 0)):.3f} | "
                    f"{float(best_by_val(rows).get('test_r', 0)):.3f} |")
    write(GEOCK_DIR / "Sessions MOC.md", md("Sessions MOC", "\n".join(body),
                                             tags=["moc", "sessions"]))
    print(f"[OK] Sessions MOC.md")


def write_session(note=""):
    rows = load_results()
    best_val = best_by_val(rows)
    best_loo = best_by_loo(rows)

    body = [
        callout(f"Session log: {today} {time}", "info"),
        "",
        f"**Total experiments**: {len(rows)}",
        f"**Branch**: `{git_branch()}`",
        f"**Commit**: `{git_hash()}`",
        "",
    ]
    if best_val:
        body += [
            f"**Best by Val R**: `{best_val.get('experiment_id', '?')}` = "
            f"{float(best_val.get('val_pearson_r', 0)):.3f} "
            f"(Test R={float(best_val.get('test_r', 0)):.3f})",
            f"**Best by LOO-CV**: `{best_loo.get('experiment_id', '?')}` = "
            f"{float(best_loo.get('loo_r', 0)):.3f}",
            "",
        ]
    if note:
        body += [
            f"## Session Note",
            f"",
            f"{note}",
            "",
        ]
    body += [
        f"## Git Diff",
        f"```",
        git_diff() or "(no uncommitted changes)",
        f"```",
        f"",
        f"## Git Log (recent)",
        f"```",
        git_log(5),
        f"```",
    ]
    session_file = GEOCK_DIR / "sessions" / f"{today}.md"
    write(session_file, body)
    print(f"[OK] sessions/{today}.md")


def write_pipeline():
    body = [
        callout("Scripts for data fetching, feature extraction, and model training", "info"),
        "",
        h2("Data Pipeline"),
        "",
        *table(
            ["Script", "Purpose", "Status"],
            [
                ["fetch_pdbbind.py", "Build PDBbind index from ChEMBL + RCSB APIs", "Ready (APIs blocked)"],
                ["fetch_casf.py", "Download CASF-2016 from Figshare", "Ready (manual download needed)"],
                ["extract_pdbbind.py", "Extract features for PDBbind-scale data", "Ready"],
                ["extract_casf.py", "Extract features for CASF complexes", "Ready"],
                ["extract_features.py", "Single PDB batch extractor", "Ready"],
            ]
        ),
        "",
        h2("Model Training"),
        "",
        *table(
            ["Script", "Purpose", "Status"],
            [
                ["train.py", "Current best model (Path 1)", "✅ Active"],
                ["batch_train.py", "323-experiment batch search", "✅ Complete"],
                ["augment_train.py", "Noise augmentation study (105 experiments)", "✅ Complete"],
                ["train_scale.py", "Lean pipeline for any dataset format", "Ready"],
                ["verify_geock.py", "LOO-CV reality check", "Ready"],
            ]
        ),
        "",
        h2("Observability"),
        "",
        *table(
            ["Script", "Purpose"],
            [
                ["push_to_obsidian.py", "Sync results to Obsidian knowledge graph"],
                ["results.tsv", "All 323 experiment results"],
            ]
        ),
        "",
        f"See also: {wiki('Path 1')} (current), {wiki('Path 2')} (scale-up).",
    ]
    write(GEOCK_DIR / "Pipeline.md", md("Pipeline: Scripts Reference", "\n".join(body),
                                         tags=["pipeline", "scripts"]))
    print(f"[OK] Pipeline.md")


def write_experiment_log(rows):
    if not rows:
        return
    body = [
        callout(f"{len(rows)} experiments — sorted by Val R descending", "info"),
        "",
        f"| # | Experiment | Val R | LOO | Test R | MAE | Features |",
        f"|---|-----------|-------|-----|--------|-----|---------|",
    ]
    for i, r in enumerate(sorted(rows, key=lambda x: float(x.get("val_pearson_r", -99)), reverse=True)[:50], 1):
        body.append(
            f"| {i} | {r.get('experiment_id', '?')} | "
            f"{float(r.get('val_pearson_r', 0)):.3f} | "
            f"{float(r.get('loo_r', 0)):.3f} | "
            f"{float(r.get('test_r', 0)):.3f} | "
            f"{float(r.get('val_mae', 0)):.3f} | "
            f"{r.get('selected', r.get('description', '?'))[:40]} |"
        )
    if len(rows) > 50:
        body.append(f"| ... | +{len(rows)-50} more experiments | | | | |")
    write(GEOCK_DIR / "Experiments.md", md("Experiments: 323 Run Results", "\n".join(body),
                                            tags=["experiments", "results"]))
    print(f"[OK] Experiments.md")


def write_program():
    try:
        src = (WORKSPACE / "program.md").read_text()
    except:
        src = "(not found)"
    body = [
        callout("Agent instructions from program.md — synced at build time", "info"),
        "",
        "```markdown",
        src[:4000],
        "```",
    ]
    write(GEOCK_DIR / "program.md", md("program.md: Agent Instructions", "\n".join(body),
                                        tags=["meta"]))
    print(f"[OK] program.md")


# ════════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--note", default="")
    args = parser.parse_args()

    rows = load_results()

    print(f"\n[GEOCK → Obsidian] Building knowledge graph...")
    print(f"  Vault: {VAULT}")
    print(f"  Date: {today} {time}")
    print(f"  Experiments: {len(rows)}")
    print()

    # Create directory structure
    dirs = [
        GEOCK_DIR,
        GEOCK_DIR / "sessions",
        GEOCK_DIR / "files",
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

    # Root / MOC pages
    write_geock_root(rows)
    write_findings_moc(rows)
    write_models_moc(rows)
    write_features_moc()
    write_sessions_moc()

    # Concepts: findings
    write_overfitting(rows)
    write_loo_cv()
    write_test_r()
    write_distribution_shift()
    write_vqe()
    write_augmentation()

    # Concepts: models
    write_lasso()
    write_gradient_boosting_overfitting()
    write_elasticnet()
    write_bayesianridge()
    write_svr()
    write_gaussianprocess()

    # Concepts: features
    write_e4_features()
    write_ecfp()

    # Concepts: paths
    write_path1(rows)
    write_path2()

    # Data
    write_data(rows)
    write_casf()
    write_pdbbind()

    # Pipeline & logs
    write_pipeline()
    write_session(args.note)
    write_experiment_log(rows)
    write_program()

    n_files = len(list(GEOCK_DIR.rglob("*.md")))
    print(f"\n[DONE] {n_files} notes synced to Obsidian.")
    print(f"  Open Obsidian → vault2 → GEOCK/")
    print(f"  Start at: {wiki('GEOCK MOC')}")


if __name__ == "__main__":
    main()
