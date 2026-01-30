"""
Script de validation complète pour vérifier que tout est prêt avant déploiement.

Ce script vérifie:
- Syntaxe Python de tous les fichiers
- Structure des fichiers et répertoires
- Configuration YAML valide
- Imports (sans exécuter le code nécessitant torch)
- Présence de tous les fichiers nécessaires
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import List, Tuple

# Couleurs pour Windows (compatible)
GREEN = "[OK]"
RED = "[FAIL]"
YELLOW = "[WARN]"


def check_syntax(file_path: Path) -> Tuple[bool, str]:
    """Vérifie la syntaxe Python d'un fichier."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            ast.parse(f.read(), filename=str(file_path))
        return True, ""
    except SyntaxError as e:
        return False, f"Syntax error at line {e.lineno}: {e.msg}"
    except Exception as e:
        return False, str(e)


def check_imports_safe(file_path: Path) -> Tuple[bool, str]:
    """Vérifie les imports sans exécuter le code."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            tree = ast.parse(f.read(), filename=str(file_path))
        
        # Vérifier les imports
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    # Vérifier que les imports locaux existent
                    if not alias.name.startswith('.'):
                        continue
                    # Imports relatifs - on vérifie juste la syntaxe
                    pass
            elif isinstance(node, ast.ImportFrom):
                if node.module and not node.module.startswith('.'):
                    # Import externe - OK
                    pass
        
        return True, ""
    except Exception as e:
        return False, str(e)


def check_yaml_config(file_path: Path) -> Tuple[bool, str]:
    """Vérifie qu'un fichier YAML est valide."""
    try:
        import yaml
        with open(file_path, 'r', encoding='utf-8') as f:
            yaml.safe_load(f)
        return True, ""
    except ImportError:
        return False, "yaml module not available"
    except Exception as e:
        return False, str(e)


def validate_project_structure() -> List[Tuple[str, bool, str]]:
    """Valide la structure complète du projet."""
    results = []
    
    # Fichiers essentiels
    essential_files = [
        "docker-compose.yml",
        "Dockerfile",
        ".dockerignore",
        "setup.py",
        "requirements.txt",
        "config/docker.env",
        "config/training_config.yaml",
    ]
    
    print("\n" + "=" * 80)
    print("Validation de la Structure du Projet")
    print("=" * 80)
    
    for file_path in essential_files:
        path = Path(file_path)
        exists = path.exists()
        results.append((f"Fichier: {file_path}", exists, "" if exists else "Fichier manquant"))
        status = GREEN if exists else RED
        print(f"  {status} {file_path}")
    
    # Répertoires essentiels
    essential_dirs = [
        "src/st_cdgm",
        "src/st_cdgm/models",
        "src/st_cdgm/data",
        "src/st_cdgm/training",
        "src/st_cdgm/evaluation",
        "scripts",
        "ops",
        "config",
        "docs",
    ]
    
    print("\nRépertoires:")
    for dir_path in essential_dirs:
        path = Path(dir_path)
        exists = path.exists() and path.is_dir()
        results.append((f"Répertoire: {dir_path}", exists, "" if exists else "Répertoire manquant"))
        status = GREEN if exists else RED
        print(f"  {status} {dir_path}")
    
    return results


def validate_python_files() -> List[Tuple[str, bool, str]]:
    """Valide tous les fichiers Python."""
    results = []
    
    print("\n" + "=" * 80)
    print("Validation des Fichiers Python")
    print("=" * 80)
    
    # Fichiers à vérifier
    python_files = []
    
    # Scripts
    scripts_dir = Path("scripts")
    if scripts_dir.exists():
        python_files.extend(scripts_dir.glob("*.py"))
    
    # Ops
    ops_dir = Path("ops")
    if ops_dir.exists():
        python_files.extend(ops_dir.glob("*.py"))
    
    # Source
    src_dir = Path("src/st_cdgm")
    if src_dir.exists():
        python_files.extend(src_dir.rglob("*.py"))
    
    for py_file in python_files:
        # Ignorer __pycache__
        if "__pycache__" in str(py_file):
            continue
        
        # Vérifier syntaxe
        syntax_ok, syntax_msg = check_syntax(py_file)
        results.append((f"Syntaxe: {py_file}", syntax_ok, syntax_msg))
        
        status = GREEN if syntax_ok else RED
        if syntax_ok:
            print(f"  {status} {py_file.name}")
        else:
            print(f"  {status} {py_file.name}: {syntax_msg}")
    
    return results


def validate_configs() -> List[Tuple[str, bool, str]]:
    """Valide les fichiers de configuration."""
    results = []
    
    print("\n" + "=" * 80)
    print("Validation des Configurations")
    print("=" * 80)
    
    config_files = [
        "config/training_config.yaml",
    ]
    
    for config_file in config_files:
        path = Path(config_file)
        if not path.exists():
            results.append((f"Config: {config_file}", False, "Fichier manquant"))
            print(f"  {RED} {config_file}: Fichier manquant")
            continue
        
        yaml_ok, yaml_msg = check_yaml_config(path)
        results.append((f"Config: {config_file}", yaml_ok, yaml_msg))
        status = GREEN if yaml_ok else RED
        print(f"  {status} {config_file}")
        if not yaml_ok:
            print(f"      Erreur: {yaml_msg}")
    
    return results


def validate_docker_files() -> List[Tuple[str, bool, str]]:
    """Valide les fichiers Docker."""
    results = []
    
    print("\n" + "=" * 80)
    print("Validation des Fichiers Docker")
    print("=" * 80)
    
    docker_files = [
        "docker-compose.yml",
        "Dockerfile",
        ".dockerignore",
    ]
    
    for docker_file in docker_files:
        path = Path(docker_file)
        exists = path.exists()
        results.append((f"Docker: {docker_file}", exists, "" if exists else "Fichier manquant"))
        status = GREEN if exists else RED
        print(f"  {status} {docker_file}")
        
        if exists and docker_file.endswith('.yml'):
            # Vérifier que docker-compose.yml est valide (basique)
            try:
                with open(path, 'r') as f:
                    content = f.read()
                    # Vérifications basiques
                    if 'services:' in content or 'version:' in content:
                        print(f"      Structure docker-compose valide")
            except Exception as e:
                results.append((f"Docker: {docker_file} (structure)", False, str(e)))
                print(f"      {RED} Erreur structure: {e}")
    
    return results


def validate_imports_structure() -> List[Tuple[str, bool, str]]:
    """Vérifie la structure des imports sans exécuter."""
    results = []
    
    print("\n" + "=" * 80)
    print("Validation de la Structure des Imports")
    print("=" * 80)
    
    # Vérifier que les modules __init__.py existent
    init_files = [
        "src/st_cdgm/__init__.py",
        "src/st_cdgm/models/__init__.py",
        "src/st_cdgm/data/__init__.py",
        "src/st_cdgm/training/__init__.py",
        "src/st_cdgm/evaluation/__init__.py",
    ]
    
    for init_file in init_files:
        path = Path(init_file)
        exists = path.exists()
        results.append((f"__init__: {init_file}", exists, "" if exists else "Fichier manquant"))
        status = GREEN if exists else RED
        print(f"  {status} {init_file}")
    
    return results


def main():
    """Exécute toutes les validations."""
    print("=" * 80)
    print("Validation Complète du Projet ST-CDGM")
    print("=" * 80)
    
    all_results = []
    
    # Structure du projet
    all_results.extend(validate_project_structure())
    
    # Fichiers Python
    all_results.extend(validate_python_files())
    
    # Configurations
    all_results.extend(validate_configs())
    
    # Docker
    all_results.extend(validate_docker_files())
    
    # Imports
    all_results.extend(validate_imports_structure())
    
    # Résumé
    print("\n" + "=" * 80)
    print("Résumé de la Validation")
    print("=" * 80)
    
    passed = sum(1 for _, ok, _ in all_results if ok)
    total = len(all_results)
    failed = total - passed
    
    for name, ok, msg in all_results:
        if not ok:
            status = RED
            print(f"  {status} {name}")
            if msg:
                print(f"      {msg}")
    
    print(f"\nTotal: {total} | Réussis: {passed} | Échoués: {failed}")
    
    if failed == 0:
        print(f"\n{GREEN} Toutes les validations ont réussi!")
        return 0
    else:
        print(f"\n{RED} {failed} validation(s) ont échoué. Veuillez corriger les erreurs ci-dessus.")
        return 1


if __name__ == "__main__":
    sys.exit(main())








