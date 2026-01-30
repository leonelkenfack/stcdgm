#!/usr/bin/env python
"""
Script de vérification de l'installation ST-CDGM.
Teste que toutes les dépendances critiques sont installées correctement.
"""

import sys
from typing import List, Tuple

def check_package(package_name: str, import_name: str = None) -> Tuple[bool, str]:
    """
    Vérifie si un package est installé et retourne sa version.
    
    Args:
        package_name: Nom du package à afficher
        import_name: Nom pour l'import (si différent)
    
    Returns:
        (success, version_string)
    """
    if import_name is None:
        import_name = package_name.replace("-", "_")
    
    try:
        module = __import__(import_name)
        version = getattr(module, "__version__", "unknown")
        return True, version
    except ImportError as e:
        return False, str(e)

def main():
    print("=" * 80)
    print("🔍 ST-CDGM Installation Verification")
    print("=" * 80)
    print()
    
    # Python version
    print(f"🐍 Python: {sys.version}")
    print(f"   Executable: {sys.executable}")
    print()
    
    # Liste des packages à vérifier
    packages = [
        ("numpy", None),
        ("pandas", None),
        ("xarray", None),
        ("torch", None),
        ("torchvision", None),
        ("torch_geometric", None),
        ("torch_scatter", None),
        ("torch_sparse", None),
        ("diffusers", None),
        ("accelerate", None),
        ("netCDF4", "netCDF4"),
        ("h5netcdf", None),
        ("xbatcher", None),
        ("dask", None),
        ("matplotlib", None),
        ("seaborn", None),
        ("hydra", None),
        ("omegaconf", None),
        ("pytest", None),
    ]
    
    results = []
    max_name_len = max(len(p[0]) for p in packages)
    
    print("📦 Checking Packages:")
    print("-" * 80)
    
    for package_name, import_name in packages:
        success, info = check_package(package_name, import_name)
        results.append((package_name, success, info))
        
        status = "✅" if success else "❌"
        padding = " " * (max_name_len - len(package_name))
        
        if success:
            print(f"{status} {package_name}{padding} : {info}")
        else:
            print(f"{status} {package_name}{padding} : NOT INSTALLED")
    
    print()
    
    # PyTorch specific checks
    print("🔥 PyTorch Configuration:")
    print("-" * 80)
    try:
        import torch
        print(f"   PyTorch Version: {torch.__version__}")
        print(f"   CUDA Available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"   CUDA Version: {torch.version.cuda}")
            print(f"   cuDNN Version: {torch.backends.cudnn.version()}")
            print(f"   Number of GPUs: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
                props = torch.cuda.get_device_properties(i)
                print(f"           Memory: {props.total_memory / 1024**3:.1f} GB")
        else:
            print("   ⚠️  Running on CPU only")
    except Exception as e:
        print(f"   ❌ Error checking PyTorch: {e}")
    
    print()
    
    # Summary
    print("=" * 80)
    successful = sum(1 for _, success, _ in results if success)
    total = len(results)
    
    if successful == total:
        print(f"✅ SUCCESS: All {total} packages installed correctly!")
        print()
        print("🚀 You're ready to use ST-CDGM!")
        print("   Next steps:")
        print("   1. Open the notebook: jupyter notebook st_cdgm_training_evaluation.ipynb")
        print("   2. Or run the training script: python ops/train_st_cdgm.py")
        return 0
    else:
        print(f"⚠️  WARNING: {successful}/{total} packages installed")
        print()
        print("Missing packages:")
        for name, success, info in results:
            if not success:
                print(f"   ❌ {name}")
        print()
        print("📝 Installation instructions:")
        print("   pip install -r requirements.txt")
        print()
        print("   Or see INSTALLATION.md for detailed instructions")
        return 1

if __name__ == "__main__":
    sys.exit(main())

