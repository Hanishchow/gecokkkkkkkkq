#!/usr/bin/env python3
"""
GEOCK Cross-Platform Path Helpers
================================
Reusable path utilities for Linux/Windows compatibility.

Usage:
    from geock_paths import get_cache_dir, get_work_dir

    cache_dir = get_cache_dir()
    work_dir = get_work_dir()
"""

import os
from pathlib import Path

# Known Windows OneDrive paths (user's machine)
ONEDRIVE_BASE = os.path.expanduser("~/OneDrive")
ONEDRIVE_DESKTOP = os.path.join(ONEDRIVE_BASE, "Desktop", "lki")


def get_cache_dir():
    """Get cache directory - works on Linux and Windows.

    Returns:
        Path: Cache directory with geock data
    """
    # Try Linux path first
    linux_cache = Path("/home/chow/.cache/geock_autoresearch")
    if linux_cache.exists():
        return linux_cache

    # Try Windows OneDrive path (e.g., ~/OneDrive/.cache/geock_autoresearch)
    win_cache = Path(os.path.join(ONEDRIVE_BASE, ".cache", "geock_autoresearch"))
    if win_cache.exists():
        return win_cache

    # Try Windows OneDrive Desktop path (C:\Users\yakka\OneDrive\Desktop\lki)
    win_desktop = Path(os.path.join(ONEDRIVE_DESKTOP, ".cache", "geock_autoresearch"))
    if win_desktop.exists():
        return win_desktop

    # Fallback to current directory
    cache_dir = Path("./cache")
    cache_dir.mkdir(exist_ok=True)
    return cache_dir


def get_work_dir():
    """Get work directory - works on Linux and Windows.

    Returns:
        Path: Work directory with models
    """
    # Try Linux path
    linux = Path("/home/chow/autoresearch")
    if linux.exists():
        return linux

    # Try Windows OneDrive path (e.g., ~/OneDrive/autoresearch)
    win = Path(os.path.join(ONEDRIVE_BASE, "autoresearch"))
    if win.exists():
        return win

    # Try Windows OneDrive Desktop path (C:\Users\yakka\OneDrive\Desktop\lki)
    win_desktop = Path(ONEDRIVE_DESKTOP)
    if win_desktop.exists():
        return win_desktop

    # Fallback to current directory
    return Path(".")


def get_data_dir():
    """Get additional data directory.

    Returns:
        Path: Additional data (PDB files, etc)
    """
    cache = get_cache_dir()
    data_dir = cache / "lp_pdb_files"
    if data_dir.exists():
        return data_dir

    # Alternative locations
    alt1 = Path("CACHE_DIR / lp_pdb_files")
    if alt1.exists():
        return alt1

    alt2 = Path(
        os.path.join(ONEDRIVE_DESKTOP, ".cache", "geock_autoresearch", "lp_pdb_files")
    )
    if alt2.exists():
        return alt2

    return cache


# Convenience
CACHE_DIR = get_cache_dir()
WORK_DIR = get_work_dir()
DATA_DIR = get_data_dir()


def get_model_path(model_name: str) -> Path:
    """Get path to a saved model file.

    Args:
        model_name: Name of model file (e.g., 'geock_deep_trees_final.pkl')

    Returns:
        Path: Full path to model file
    """
    return WORK_DIR / model_name


def get_data_path(data_name: str) -> Path:
    """Get path to a data file in cache.

    Args:
        data_name: Name of data file (e.g., 'merged_39k.pkl')

    Returns:
        Path: Full path to data file
    """
    return CACHE_DIR / data_name
