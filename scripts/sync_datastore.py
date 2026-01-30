"""
Script for synchronizing data between local disk and CyVerse Data Store.

This script provides utilities to:
- Copy data from Data Store to local disk (for better I/O performance)
- Save results from local disk to Data Store (for persistence)
- List files in Data Store
- Check disk space

Usage:
    # Copy from Data Store to local
    python scripts/sync_datastore.py --copy-from-datastore \
        ~/data-store/home/username/data/raw/*.nc \
        ~/climate_data/data/raw/
    
    # Save to Data Store from local
    python scripts/sync_datastore.py --save-to-datastore \
        ~/climate_data/models/*.pt \
        ~/data-store/home/username/st-cdgm/models/
    
    # List files in Data Store
    python scripts/sync_datastore.py --list-datastore \
        ~/data-store/home/username/data/
    
    # Dry run (simulate without copying)
    python scripts/sync_datastore.py --copy-from-datastore \
        --dry-run ~/data-store/home/username/data/raw/*.nc ~/climate_data/data/raw/
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import List, Optional

# Add project to path
project_root = Path(__file__).parent.parent
if str(project_root / "src") not in sys.path:
    sys.path.insert(0, str(project_root / "src"))
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from scripts.vice_utils import (
    get_datastore_path,
    is_vice_environment,
    recommend_local_copy,
)


def format_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


def get_disk_usage(path: Path) -> dict:
    """Get disk usage statistics for a path."""
    try:
        stat = shutil.disk_usage(path)
        return {
            "total": stat.total,
            "used": stat.used,
            "free": stat.free,
            "percent_used": (stat.used / stat.total) * 100,
        }
    except OSError:
        return None


def copy_files(
    source: Path | str,
    destination: Path | str,
    dry_run: bool = False,
    verbose: bool = True,
) -> int:
    """
    Copy files from source to destination.
    
    Args:
        source: Source path (file or directory).
        destination: Destination path (file or directory).
        dry_run: If True, simulate without copying.
        verbose: If True, print progress.
    
    Returns:
        Number of files copied.
    """
    source = Path(source).expanduser().resolve()
    destination = Path(destination).expanduser().resolve()
    
    if not source.exists():
        print(f"Error: Source path does not exist: {source}")
        return 0
    
    # Ensure destination directory exists
    if source.is_file():
        destination.parent.mkdir(parents=True, exist_ok=True)
    else:
        destination.mkdir(parents=True, exist_ok=True)
    
    copied_count = 0
    
    if source.is_file():
        # Copy single file
        if dry_run:
            if verbose:
                size = source.stat().st_size
                print(f"[DRY RUN] Would copy: {source} -> {destination} ({format_size(size)})")
            copied_count = 1
        else:
            if verbose:
                size = source.stat().st_size
                print(f"Copying: {source.name} ({format_size(size)})...")
            shutil.copy2(source, destination)
            copied_count = 1
            if verbose:
                print(f"  ✓ Copied to: {destination}")
    
    elif source.is_dir():
        # Copy directory tree
        if dry_run:
            if verbose:
                print(f"[DRY RUN] Would copy directory: {source} -> {destination}")
            # Count files recursively
            for file_path in source.rglob("*"):
                if file_path.is_file():
                    copied_count += 1
                    if verbose:
                        size = file_path.stat().st_size
                        rel_path = file_path.relative_to(source)
                        print(f"  Would copy: {rel_path} ({format_size(size)})")
        else:
            if verbose:
                print(f"Copying directory: {source} -> {destination}")
            
            for file_path in source.rglob("*"):
                if file_path.is_file():
                    rel_path = file_path.relative_to(source)
                    dest_path = destination / rel_path
                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    if verbose:
                        size = file_path.stat().st_size
                        print(f"  Copying: {rel_path} ({format_size(size)})...")
                    
                    shutil.copy2(file_path, dest_path)
                    copied_count += 1
            
            if verbose:
                print(f"  ✓ Copied {copied_count} files")
    
    return copied_count


def list_datastore(path: Path | str, recursive: bool = False) -> None:
    """List files in Data Store directory."""
    path = Path(path).expanduser().resolve()
    
    if not path.exists():
        print(f"Error: Path does not exist: {path}")
        return
    
    if not path.is_dir():
        print(f"Error: Path is not a directory: {path}")
        return
    
    print(f"Listing files in: {path}")
    print("=" * 80)
    
    if recursive:
        files = list(path.rglob("*"))
    else:
        files = list(path.iterdir())
    
    # Separate files and directories
    dirs = [f for f in files if f.is_dir()]
    files_only = [f for f in files if f.is_file()]
    
    # Print directories first
    if dirs:
        print("\nDirectories:")
        for d in sorted(dirs):
            rel_path = d.relative_to(path)
            print(f"  📁 {rel_path}/")
    
    # Print files
    if files_only:
        print("\nFiles:")
        total_size = 0
        for f in sorted(files_only):
            rel_path = f.relative_to(path)
            size = f.stat().st_size
            total_size += size
            print(f"  📄 {rel_path} ({format_size(size)})")
        
        print(f"\nTotal: {len(files_only)} files, {format_size(total_size)}")


def copy_from_datastore(
    source_pattern: str,
    destination: str,
    dry_run: bool = False,
    verbose: bool = True,
) -> None:
    """Copy files from Data Store to local disk."""
    if not is_vice_environment():
        print("Warning: Not running in VICE environment. Data Store may not be available.")
    
    # Expand user paths
    source = Path(source_pattern).expanduser()
    destination = Path(destination).expanduser()
    
    # Check if source exists
    if not source.exists():
        print(f"Error: Source does not exist: {source}")
        print("\nTip: Check that your Data Store path is correct.")
        print(f"     Example: ~/data-store/home/<username>/data/raw/")
        return
    
    # Check disk space
    dest_usage = get_disk_usage(destination.parent if destination.is_file() else destination)
    if dest_usage and verbose:
        print(f"Destination disk usage: {dest_usage['percent_used']:.1f}% used")
        print(f"  Free space: {format_size(dest_usage['free'])}")
    
    # Recommend local copy
    if verbose and recommend_local_copy():
        print("\nℹ️  Recommendation: Copying to local disk for better I/O performance.")
        print("   Data Store access is slower for large files.\n")
    
    # Copy files
    count = copy_files(source, destination, dry_run=dry_run, verbose=verbose)
    
    if verbose:
        if dry_run:
            print(f"\n[DRY RUN] Would copy {count} file(s)")
        else:
            print(f"\n✓ Successfully copied {count} file(s)")


def save_to_datastore(
    source: str,
    destination_pattern: str,
    dry_run: bool = False,
    verbose: bool = True,
) -> None:
    """Save files from local disk to Data Store."""
    if not is_vice_environment():
        print("Warning: Not running in VICE environment. Data Store may not be available.")
    
    # Expand user paths
    source = Path(source).expanduser()
    destination = Path(destination_pattern).expanduser()
    
    # Check if source exists
    if not source.exists():
        print(f"Error: Source does not exist: {source}")
        return
    
    # Check disk space in Data Store
    dest_usage = get_disk_usage(destination.parent if destination.is_file() else destination)
    if dest_usage and verbose:
        print(f"Data Store disk usage: {dest_usage['percent_used']:.1f}% used")
        print(f"  Free space: {format_size(dest_usage['free'])}")
    
    if verbose:
        print("\nℹ️  Saving to Data Store for persistence.")
        print("   Files in Data Store persist after VICE session ends.\n")
    
    # Copy files
    count = copy_files(source, destination, dry_run=dry_run, verbose=verbose)
    
    if verbose:
        if dry_run:
            print(f"\n[DRY RUN] Would save {count} file(s) to Data Store")
        else:
            print(f"\n✓ Successfully saved {count} file(s) to Data Store")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Synchronize data between local disk and CyVerse Data Store",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Copy data from Data Store to local
  python scripts/sync_datastore.py --copy-from-datastore \\
      ~/data-store/home/username/data/raw/*.nc \\
      ~/climate_data/data/raw/
  
  # Save results to Data Store
  python scripts/sync_datastore.py --save-to-datastore \\
      ~/climate_data/models/*.pt \\
      ~/data-store/home/username/st-cdgm/models/
  
  # List files in Data Store
  python scripts/sync_datastore.py --list-datastore \\
      ~/data-store/home/username/data/
  
  # Dry run (simulate)
  python scripts/sync_datastore.py --copy-from-datastore --dry-run \\
      ~/data-store/home/username/data/raw/*.nc \\
      ~/climate_data/data/raw/
        """,
    )
    
    parser.add_argument(
        "--copy-from-datastore",
        nargs=2,
        metavar=("SOURCE", "DEST"),
        help="Copy files from Data Store to local disk",
    )
    parser.add_argument(
        "--save-to-datastore",
        nargs=2,
        metavar=("SOURCE", "DEST"),
        help="Save files from local disk to Data Store",
    )
    parser.add_argument(
        "--list-datastore",
        nargs=1,
        metavar="PATH",
        help="List files in Data Store directory",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate operations without copying files",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress output",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="List files recursively (with --list-datastore)",
    )
    
    args = parser.parse_args()
    
    verbose = not args.quiet
    
    # Check if any action is specified
    if not any([args.copy_from_datastore, args.save_to_datastore, args.list_datastore]):
        parser.print_help()
        return 1
    
    # Handle list datastore
    if args.list_datastore:
        list_datastore(args.list_datastore[0], recursive=args.recursive)
        return 0
    
    # Handle copy from datastore
    if args.copy_from_datastore:
        copy_from_datastore(
            args.copy_from_datastore[0],
            args.copy_from_datastore[1],
            dry_run=args.dry_run,
            verbose=verbose,
        )
        return 0
    
    # Handle save to datastore
    if args.save_to_datastore:
        save_to_datastore(
            args.save_to_datastore[0],
            args.save_to_datastore[1],
            dry_run=args.dry_run,
            verbose=verbose,
        )
        return 0
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
