---
tags: [setup, wsl, linux, windows, environment]
type: setup
status: active
related: [[AMD GPU ROCm]], [[Tools Installed]]
---

# 🐧 WSL Setup

> [!abstract] Setup
> Windows Subsystem for Linux. Running **Kali Linux** on Windows. This is my daily driver.

---

## My Environment

| Item | Value |
|------|-------|
| Host OS | Windows |
| WSL distro | Kali Linux |
| Shell path | `/mnt/c/Users/yakka` |
| Python | Miniconda3 — `/home/chow/miniconda3` |
| Python version | 3.13 |

---

## Key Paths

| What | Where |
|------|-------|
| Windows C drive | `/mnt/c/` |
| My Windows user folder | `/mnt/c/Users/yakka` |
| Conda base | `/home/chow/miniconda3` |
| Conda envs | `/home/chow/miniconda3/envs/` |
| WSL files from Windows Explorer | `\\wsl$\kali-linux\home\chow\` |

---

## Common Commands

```bash
# Conda
conda activate myenv
conda deactivate
conda env list
conda create -n myenv python=3.11   # create new env

# Check Python
python --version && which python

# Access Windows files
ls /mnt/c/Users/yakka/
```

---

## ⚠️ The Bash vs Python Mistake

> [!bug] Classic Error
> Running Python code directly in bash gives this:
> ```bash
> $ from rdkit import Chem
> bash: syntax error near unexpected token `('
> ```
>
> **Fix:** always run Python inside a script or interpreter:
> ```bash
> python3 myscript.py
> python3 -c "from rdkit import Chem; print('ok')"
> python3   # REPL
> ```

---

## pip Gotchas on Python 3.13

```bash
# General fix for build failures
pip install --upgrade pip setuptools wheel

# oddt-specific fix
pip install six && pip install oddt

# conda for C-extension packages
conda install -c conda-forge rdkit   # NOT pip install rdkit
```

> [!tip] Rule: conda for heavy C packages, pip for pure Python
