"""
Utility functions for CyVerse Discovery Environment (VICE) support.

This module provides functions to detect if code is running in a VICE environment
and manage paths to the CyVerse Data Store.

Usage:
    from scripts.vice_utils import is_vice_environment, get_datastore_path, resolve_data_path
    
    if is_vice_environment():
        datastore_path = get_datastore_path()
        data_path = resolve_data_path("data/raw/lr.nc")
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


def is_vice_environment() -> bool:
    """
    Detect if code is running in a CyVerse VICE environment.
    
    VICE environments typically have:
    - A ~/data-store/ directory mounted via CSI Driver
    - Environment variables indicating VICE/DE
    - Jupyter Lab or other VICE apps running
    
    Returns:
        True if running in VICE, False otherwise.
    """
    # Check for data-store directory (primary indicator)
    home = Path.home()
    data_store_dir = home / "data-store"
    
    if data_store_dir.exists() and data_store_dir.is_dir():
        # Additional check: verify structure
        home_subdir = data_store_dir / "home"
        if home_subdir.exists():
            return True
    
    # Check for VICE-related environment variables
    vice_env_vars = [
        "VICE_APP_NAME",
        "CYVERSE_USERNAME",
        "DE_APP_NAME",
        "VICE_USER",
    ]
    
    if any(os.environ.get(var) for var in vice_env_vars):
        return True
    
    # Check for Jupyter Lab in typical VICE location
    jupyter_config = home / ".jupyter" / "jupyter_lab_config.py"
    if jupyter_config.exists():
        # Additional check: look for VICE-specific paths in env
        python_path = os.environ.get("PYTHONPATH", "")
        if "vice" in python_path.lower() or "cyverse" in python_path.lower():
            return True
    
    return False


def get_datastore_path(username: Optional[str] = None) -> Optional[Path]:
    """
    Get the path to the CyVerse Data Store for the current user.
    
    Args:
        username: Optional username. If not provided, attempts to detect from
                  environment variables or uses 'USER' or 'USERNAME'.
    
    Returns:
        Path to ~/data-store/home/<username>/ if in VICE, None otherwise.
    """
    if not is_vice_environment():
        return None
    
    # Get username
    if username is None:
        # Try environment variables first
        username = (
            os.environ.get("CYVERSE_USERNAME") or
            os.environ.get("VICE_USER") or
            os.environ.get("USER") or
            os.environ.get("USERNAME") or
            "default"
        )
    
    home = Path.home()
    datastore_path = home / "data-store" / "home" / username
    
    return datastore_path if datastore_path.exists() else None


def resolve_data_path(
    relative_path: str | Path,
    prefer_local: bool = True,
    username: Optional[str] = None,
) -> Path:
    """
    Resolve a relative data path according to the environment.
    
    In VICE:
    - If prefer_local=True: Resolves to ~/climate_data/<relative_path> (fast local disk)
    - If prefer_local=False: Resolves to ~/data-store/home/<username>/<relative_path> (persistent but slow)
    
    Outside VICE:
    - Always resolves relative to current working directory or project root.
    
    Args:
        relative_path: Relative path to data file/directory.
        prefer_local: If True (default), prefer local disk in VICE for performance.
                     If False, use Data Store (slower but persistent).
        username: Optional username for Data Store. Auto-detected if not provided.
    
    Returns:
        Resolved absolute Path.
    """
    relative_path = Path(relative_path)
    
    # If already absolute, return as-is
    if relative_path.is_absolute():
        return relative_path
    
    if is_vice_environment() and prefer_local:
        # In VICE, prefer local disk (~/) for better I/O performance
        home = Path.home()
        # Try to detect project root (look for climate_data or similar)
        project_root = home / "climate_data"
        if not project_root.exists():
            # Fallback to current directory
            project_root = Path.cwd()
        
        resolved = project_root / relative_path
        return resolved
    
    elif is_vice_environment() and not prefer_local:
        # In VICE, use Data Store (persistent but slower)
        datastore_path = get_datastore_path(username)
        if datastore_path is None:
            # Fallback to local if Data Store not available
            home = Path.home()
            project_root = home / "climate_data"
            if not project_root.exists():
                project_root = Path.cwd()
            return project_root / relative_path
        
        # Resolve relative to Data Store
        resolved = datastore_path / relative_path
        return resolved
    
    else:
        # Outside VICE, resolve relative to current working directory
        # Try to find project root (look for src/ or config/ directories)
        current = Path.cwd()
        
        # Check if we're in project root or subdirectory
        if (current / "src").exists() or (current / "config").exists():
            return current / relative_path
        
        # Walk up to find project root
        for parent in current.parents:
            if (parent / "src").exists() or (parent / "config").exists():
                return parent / relative_path
        
        # Fallback to current directory
        return current / relative_path


def recommend_local_copy(file_size_mb: Optional[float] = None) -> bool:
    """
    Recommend whether to copy data locally in VICE based on file size.
    
    In VICE, files accessed via CSI Driver (~/data-store/) are slow for large files.
    This function recommends copying to local disk (~/) if beneficial.
    
    Args:
        file_size_mb: Size of file in MB. If None, always recommends True in VICE.
    
    Returns:
        True if local copy is recommended, False otherwise.
    """
    if not is_vice_environment():
        return False  # Not applicable outside VICE
    
    # For large files (>100MB), local copy is strongly recommended
    if file_size_mb is not None and file_size_mb > 100:
        return True
    
    # For smaller files or unknown size, still recommend local copy
    # as it's generally faster and doesn't hurt
    return True


def get_project_root() -> Path:
    """
    Get the project root directory.
    
    Returns:
        Path to project root (directory containing src/ or config/).
    """
    current = Path.cwd()
    
    # Check if we're in project root
    if (current / "src").exists() or (current / "config").exists():
        return current
    
    # Walk up to find project root
    for parent in current.parents:
        if (parent / "src").exists() or (parent / "config").exists():
            return parent
    
    # Fallback to current directory
    return current


def ensure_directory(path: Path, create_if_missing: bool = True) -> Path:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        path: Path to directory.
        create_if_missing: If True, create directory if it doesn't exist.
    
    Returns:
        Path object (same as input).
    
    Raises:
        OSError: If directory creation fails.
    """
    if create_if_missing and not path.exists():
        path.mkdir(parents=True, exist_ok=True)
    return path


