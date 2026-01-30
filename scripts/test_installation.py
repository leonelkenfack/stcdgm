"""
Test script to verify ST-CDGM installation and dependencies.

This script checks:
- Python version
- PyTorch installation and CUDA availability
- Required dependencies
- Module imports
- GPU availability and basic operations
- VICE environment detection (if in CyVerse)
- Data Store access (if in VICE)

Usage:
    python scripts/test_installation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add workspace to path
workspace_path = Path(__file__).parent.parent
if str(workspace_path / "src") not in sys.path:
    sys.path.insert(0, str(workspace_path / "src"))
if str(workspace_path) not in sys.path:
    sys.path.insert(0, str(workspace_path))

# Import VICE utilities if available
try:
    from scripts.vice_utils import (
        get_datastore_path,
        is_vice_environment,
        recommend_local_copy,
    )
    HAS_VICE_UTILS = True
except ImportError:
    HAS_VICE_UTILS = False


def test_python_version():
    """Test Python version."""
    print("Testing Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"  [FAIL] Python {version.major}.{version.minor} (requires >= 3.8)")
        return False
    print(f"  [OK] Python {version.major}.{version.minor}.{version.micro}")
    return True


def test_pytorch():
    """Test PyTorch installation and CUDA."""
    print("\nTesting PyTorch...")
    try:
        import torch
        print(f"  [OK] PyTorch {torch.__version__}")
        
        if torch.cuda.is_available():
            print(f"  [OK] CUDA available: {torch.version.cuda}")
            print(f"  [OK] GPU devices: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"    - Device {i}: {torch.cuda.get_device_name(i)}")
            
            # Test basic GPU operation
            x = torch.randn(10, 10).cuda()
            y = torch.matmul(x, x)
            print(f"  [OK] GPU computation test passed")
        else:
            print(f"  [WARN] CUDA not available (CPU mode only)")
        
        return True
    except ImportError as e:
        print(f"  [FAIL] PyTorch not installed: {e}")
        return False


def test_dependencies():
    """Test required dependencies."""
    print("\nTesting dependencies...")
    dependencies = [
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("xarray", "xarray"),
        ("torch_geometric", "torch_geometric"),
        ("diffusers", "diffusers"),
        ("hydra", "hydra.core"),
        ("omegaconf", "omegaconf"),
        ("zarr", "zarr"),
        ("matplotlib", "matplotlib"),
    ]
    
    all_ok = True
    for name, module in dependencies:
        try:
            __import__(module)
            print(f"  [OK] {name}")
        except ImportError:
            print(f"  [FAIL] {name} not installed")
            all_ok = False
    
    return all_ok


def test_st_cdgm_imports():
    """Test ST-CDGM module imports."""
    print("\nTesting ST-CDGM module imports...")
    try:
        from st_cdgm import (
            NetCDFDataPipeline,
            HeteroGraphBuilder,
            IntelligibleVariableEncoder,
            RCNCell,
            RCNSequenceRunner,
            CausalDiffusionDecoder,
        )
        print("  [OK] All core modules imported successfully")
        return True
    except ImportError as e:
        print(f"  [FAIL] Import error: {e}")
        return False


def test_basic_operations():
    """Test basic PyTorch operations."""
    print("\nTesting basic operations...")
    try:
        import torch
        from st_cdgm import RCNCell
        
        # Test RCNCell creation
        rcn = RCNCell(
            num_vars=3,
            hidden_dim=64,
            driver_dim=8,
            reconstruction_dim=8,
        )
        print("  [OK] RCNCell creation successful")
        
        # Test forward pass (if GPU available)
        if torch.cuda.is_available():
            rcn = rcn.cuda()
            H = torch.randn(3, 10, 64).cuda()
            driver = torch.randn(10, 8).cuda()
            H_next, _, _ = rcn(H, driver)
            print(f"  [OK] GPU forward pass successful (output shape: {H_next.shape})")
        else:
            H = torch.randn(3, 10, 64)
            driver = torch.randn(10, 8)
            H_next, _, _ = rcn(H, driver)
            print(f"  [OK] CPU forward pass successful (output shape: {H_next.shape})")
        
        return True
    except Exception as e:
        print(f"  [FAIL] Operation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_vice_environment():
    """Test VICE environment detection and Data Store access."""
    print("\nTesting VICE environment...")
    
    if not HAS_VICE_UTILS:
        print("  [SKIP] VICE utilities not available (scripts/vice_utils.py not found)")
        return True  # Not a failure, just not available
    
    try:
        # Check if running in VICE
        is_vice = is_vice_environment()
        
        if is_vice:
            print("  [INFO] Running in CyVerse VICE environment")
            
            # Check Data Store access
            datastore_path = get_datastore_path()
            if datastore_path:
                print(f"  [OK] Data Store accessible: {datastore_path}")
                
                # Check if Data Store is writable
                try:
                    test_file = datastore_path / ".test_write"
                    test_file.touch()
                    test_file.unlink()
                    print("  [OK] Data Store is writable")
                except (OSError, PermissionError):
                    print("  [WARN] Data Store may not be writable (read-only access)")
            else:
                print("  [WARN] Data Store path not found")
            
            # Recommendation for local copy
            if recommend_local_copy():
                print("  [INFO] Recommendation: Copy data to local disk (~/) for better I/O performance")
                print("         Use: python scripts/sync_datastore.py --copy-from-datastore ...")
            
        else:
            print("  [INFO] Not running in VICE environment (local execution)")
        
        return True  # Always pass, this is informational
        
    except Exception as e:
        print(f"  [WARN] VICE detection failed: {e}")
        return True  # Not a critical failure


def main():
    """Run all tests."""
    print("=" * 80)
    print("ST-CDGM Installation Test")
    print("=" * 80)
    
    tests = [
        ("Python Version", test_python_version),
        ("PyTorch", test_pytorch),
        ("Dependencies", test_dependencies),
        ("ST-CDGM Imports", test_st_cdgm_imports),
        ("Basic Operations", test_basic_operations),
        ("VICE Environment", test_vice_environment),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n  [FAIL] {name} test crashed: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)
    all_passed = True
    for name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"  {status}: {name}")
        if not result:
            all_passed = False
    
    print("=" * 80)
    if all_passed:
        print("[SUCCESS] All tests passed!")
        return 0
    else:
        print("[FAIL] Some tests failed. Please check the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

