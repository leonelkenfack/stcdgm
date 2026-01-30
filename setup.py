"""
Setup script for ST-CDGM package.
Allows installation in development mode with: pip install -e .
"""

from pathlib import Path
from setuptools import setup, find_packages

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
if requirements_path.exists():
    with open(requirements_path, encoding="utf-8") as f:
        install_requires = [
            line.strip()
            for line in f
            if line.strip() and not line.startswith("#") and not line.startswith("-")
        ]
else:
    install_requires = []

setup(
    name="st-cdgm",
    version="0.1.0",
    description="Spatio-Temporal Causal Diffusion Generative Model",
    author="ST-CDGM Team",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=install_requires,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)

