"""
Script de pré-traitement pour convertir des données NetCDF en format Zarr optimisé.

Ce script applique toutes les transformations nécessaires (normalisation, baseline,
transformations) et écrit les données en format Zarr avec chunks optimisés pour
l'entraînement ST-CDGM.

Usage:
    python ops/preprocess_to_zarr.py \
        --lr_path data/raw/lr.nc \
        --hr_path data/raw/hr.nc \
        --output_dir data/zarr/ \
        --seq_len 10 \
        --baseline_strategy hr_smoothing \
        --normalize
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import xarray as xr
import zarr

from st_cdgm import NetCDFDataPipeline


def convert_netcdf_to_zarr(
    lr_path: Path,
    hr_path: Path,
    output_dir: Path,
    *,
    static_path: Optional[Path] = None,
    seq_len: int = 10,
    baseline_strategy: str = "hr_smoothing",
    baseline_factor: int = 4,
    normalize: bool = False,
    target_transform: Optional[str] = None,
    lr_variables: Optional[Sequence[str]] = None,
    hr_variables: Optional[Sequence[str]] = None,
    static_variables: Optional[Sequence[str]] = None,
    means_path: Optional[Path] = None,
    stds_path: Optional[Path] = None,
    chunk_size_time: Optional[int] = None,
    chunk_size_lat: Optional[int] = None,
    chunk_size_lon: Optional[int] = None,
    compressor: Optional[zarr.codec.Codec] = None,
) -> None:
    """
    Convertit des données NetCDF en format Zarr optimisé.

    Parameters
    ----------
    lr_path, hr_path, static_path :
        Chemins vers les fichiers NetCDF d'entrée.
    output_dir :
        Répertoire de sortie pour les magasins Zarr.
    seq_len :
        Longueur de séquence pour l'entraînement (utilisée pour optimiser les chunks).
    baseline_strategy, baseline_factor :
        Stratégie de calcul du baseline.
    normalize :
        Activer la normalisation LR.
    target_transform :
        Transformation à appliquer (None, "log", "log1p").
    lr_variables, hr_variables, static_variables :
        Variables à sélectionner.
    means_path, stds_path :
        Chemins vers les statistiques de normalisation pré-calculées.
    chunk_size_time, chunk_size_lat, chunk_size_lon :
        Tailles de chunks personnalisées. Si None, calculées automatiquement.
    compressor :
        Compresseur Zarr (par défaut: Blosc avec compression LZ4).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Compresseur par défaut (compression rapide, bon ratio)
    if compressor is None:
        compressor = zarr.Blosc(
            cname="lz4",  # Compression rapide
            clevel=3,  # Niveau de compression modéré
            shuffle=zarr.Blosc.BITSHUFFLE,  # Bon pour données numériques
        )

    print("=" * 80)
    print("🔄 CONVERSION NETCDF → ZARR")
    print("=" * 80)
    print(f"📂 Répertoire de sortie: {output_dir}")
    print(f"📊 Longueur de séquence: {seq_len}")
    print(f"⚙️  Stratégie baseline: {baseline_strategy}")
    print(f"📈 Normalisation: {normalize}")
    print()

    # Étape 1 : Créer le pipeline NetCDF pour appliquer toutes les transformations
    print("📥 Chargement et préparation des données NetCDF...")
    pipeline = NetCDFDataPipeline(
        lr_path=lr_path,
        hr_path=hr_path,
        static_path=static_path,
        seq_len=seq_len,
        baseline_strategy=baseline_strategy,
        baseline_factor=baseline_factor,
        target_transform=target_transform,
        normalize=normalize,
        lr_variables=lr_variables,
        hr_variables=hr_variables,
        static_variables=static_variables,
        means_path=means_path,
        stds_path=stds_path,
    )

    # Étape 2 : Récupérer les datasets préparés
    lr_dataset = pipeline.lr_dataset
    hr_dataset = pipeline.hr_dataset
    baseline_dataset = pipeline.baseline_prepared
    residual_dataset = pipeline.residual_dataset
    static_dataset = pipeline.static_dataset

    dims = pipeline.dims

    # Étape 3 : Déterminer les tailles de chunks optimales
    print("\n🔧 Configuration des chunks Zarr...")
    
    # Chunks temporels : multiple de seq_len pour optimiser l'accès
    time_dim_size = lr_dataset.dims[dims.time]
    if chunk_size_time is None:
        # Choisir un multiple de seq_len qui donne des chunks raisonnables
        # Objectif: chunks de ~100-500 pas de temps
        chunk_size_time = min(max(seq_len * 10, 100), time_dim_size // 4, 500)
        # S'assurer que c'est un multiple de seq_len
        chunk_size_time = (chunk_size_time // seq_len) * seq_len
    
    # Chunks spatiaux : taille raisonnable (64-128 pixels typiquement)
    lr_lat_size = lr_dataset.dims[dims.lr_lat]
    lr_lon_size = lr_dataset.dims[dims.lr_lon]
    hr_lat_size = hr_dataset.dims[dims.hr_lat]
    hr_lon_size = hr_dataset.dims[dims.hr_lon]
    
    if chunk_size_lat is None:
        chunk_size_lat_lr = min(64, lr_lat_size)
        chunk_size_lat_hr = min(64, hr_lat_size)
    else:
        chunk_size_lat_lr = min(chunk_size_lat, lr_lat_size)
        chunk_size_lat_hr = min(chunk_size_lat, hr_lat_size)
    
    if chunk_size_lon is None:
        chunk_size_lon_lr = min(64, lr_lon_size)
        chunk_size_lon_hr = min(64, hr_lon_size)
    else:
        chunk_size_lon_lr = min(chunk_size_lon, lr_lon_size)
        chunk_size_lon_hr = min(chunk_size_lon, hr_lon_size)

    print(f"   LR chunks: ({chunk_size_time}, {chunk_size_lat_lr}, {chunk_size_lon_lr})")
    print(f"   HR chunks: ({chunk_size_time}, {chunk_size_lat_hr}, {chunk_size_lon_hr})")

    # Étape 4 : Convertir et sauvegarder en Zarr
    print("\n💾 Écriture en format Zarr...")

    # LR dataset
    lr_zarr_path = output_dir / "lr.zarr"
    print(f"   LR dataset → {lr_zarr_path}")
    lr_dataset.to_zarr(
        lr_zarr_path,
        mode="w",
        encoding={
            var: {
                "chunks": (chunk_size_time, lr_lat_size, lr_lon_size),
                "compressor": compressor,
            }
            for var in lr_dataset.data_vars
        },
    )

    # HR dataset
    hr_zarr_path = output_dir / "hr.zarr"
    print(f"   HR dataset → {hr_zarr_path}")
    hr_dataset.to_zarr(
        hr_zarr_path,
        mode="w",
        encoding={
            var: {
                "chunks": (chunk_size_time, hr_lat_size, hr_lon_size),
                "compressor": compressor,
            }
            for var in hr_dataset.data_vars
        },
    )

    # Baseline dataset
    baseline_zarr_path = output_dir / "baseline.zarr"
    print(f"   Baseline dataset → {baseline_zarr_path}")
    baseline_dataset.to_zarr(
        baseline_zarr_path,
        mode="w",
        encoding={
            var: {
                "chunks": (chunk_size_time, hr_lat_size, hr_lon_size),
                "compressor": compressor,
            }
            for var in baseline_dataset.data_vars
        },
    )

    # Residual dataset
    residual_zarr_path = output_dir / "residual.zarr"
    print(f"   Residual dataset → {residual_zarr_path}")
    residual_dataset.to_zarr(
        residual_zarr_path,
        mode="w",
        encoding={
            var: {
                "chunks": (chunk_size_time, hr_lat_size, hr_lon_size),
                "compressor": compressor,
            }
            for var in residual_dataset.data_vars
        },
    )

    # Static dataset (si présent)
    if static_dataset is not None:
        static_zarr_path = output_dir / "static.zarr"
        print(f"   Static dataset → {static_zarr_path}")
        static_dataset.to_zarr(
            static_zarr_path,
            mode="w",
            encoding={
                var: {
                    "chunks": (hr_lat_size, hr_lon_size),
                    "compressor": compressor,
                }
                for var in static_dataset.data_vars
            },
        )

    # Étape 5 : Sauvegarder les statistiques de normalisation (si disponibles)
    if normalize and pipeline.lr_stats:
        stats_dir = output_dir / "stats"
        stats_dir.mkdir(exist_ok=True)
        
        stats = pipeline.lr_stats
        if "mean" in stats or len(stats) > 0:
            # Save means if available
            mean_ds = stats.get("mean")
            if mean_ds is None and len(stats) > 0:
                # If stats dict has datasets, save the first one as mean
                mean_ds = list(stats.values())[0]
            if mean_ds is not None:
                mean_path = stats_dir / "mean.zarr"
                mean_ds.to_zarr(mean_path, mode="w")
                print(f"   LR mean stats → {mean_path}")
        
        if "std" in stats or len(stats) > 1:
            # Save stds if available
            std_ds = stats.get("std")
            if std_ds is None and len(stats) > 1:
                # If stats dict has multiple datasets, save the second one as std
                std_ds = list(stats.values())[1]
            if std_ds is not None:
                std_path = stats_dir / "stds.zarr"
                std_ds.to_zarr(std_path, mode="w")
                print(f"   LR std stats → {std_path}")

    # Étape 6 : Sauvegarder les métadonnées
    metadata = {
        "seq_len": seq_len,
        "baseline_strategy": baseline_strategy,
        "baseline_factor": baseline_factor,
        "normalize": normalize,
        "target_transform": target_transform,
        "dims": {
            "time": dims.time,
            "lr_lat": dims.lr_lat,
            "lr_lon": dims.lr_lon,
            "hr_lat": dims.hr_lat,
            "hr_lon": dims.hr_lon,
        },
        "chunk_sizes": {
            "time": chunk_size_time,
            "lr_lat": chunk_size_lat_lr,
            "lr_lon": chunk_size_lon_lr,
            "hr_lat": chunk_size_lat_hr,
            "hr_lon": chunk_size_lon_hr,
        },
    }

    import json

    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\n📋 Métadonnées → {metadata_path}")

    print("\n" + "=" * 80)
    print("✅ CONVERSION TERMINÉE")
    print("=" * 80)
    print(f"📁 Données Zarr disponibles dans: {output_dir}")
    print("\n💡 Pour utiliser ces données, utilisez ZarrDataPipeline au lieu de NetCDFDataPipeline")


def main():
    parser = argparse.ArgumentParser(
        description="Convertir des données NetCDF en format Zarr optimisé pour ST-CDGM"
    )
    parser.add_argument("--lr_path", type=Path, required=True, help="Chemin vers le dataset LR NetCDF")
    parser.add_argument("--hr_path", type=Path, required=True, help="Chemin vers le dataset HR NetCDF")
    parser.add_argument("--output_dir", type=Path, required=True, help="Répertoire de sortie pour les données Zarr")
    parser.add_argument("--static_path", type=Path, default=None, help="Chemin vers le dataset statique NetCDF (optionnel)")
    parser.add_argument("--seq_len", type=int, default=10, help="Longueur de séquence (pour optimiser les chunks)")
    parser.add_argument("--baseline_strategy", type=str, default="hr_smoothing", choices=["hr_smoothing", "lr_interp"], help="Stratégie de baseline")
    parser.add_argument("--baseline_factor", type=int, default=4, help="Facteur de coarsening pour hr_smoothing")
    parser.add_argument("--normalize", action="store_true", help="Activer la normalisation LR")
    parser.add_argument("--target_transform", type=str, default=None, choices=[None, "log", "log1p"], help="Transformation à appliquer")
    parser.add_argument("--lr_variables", type=str, nargs="+", default=None, help="Variables LR à inclure")
    parser.add_argument("--hr_variables", type=str, nargs="+", default=None, help="Variables HR à inclure")
    parser.add_argument("--static_variables", type=str, nargs="+", default=None, help="Variables statiques à inclure")
    parser.add_argument("--means_path", type=Path, default=None, help="Chemin vers les moyennes pré-calculées")
    parser.add_argument("--stds_path", type=Path, default=None, help="Chemin vers les écarts-types pré-calculés")
    parser.add_argument("--chunk_size_time", type=int, default=None, help="Taille de chunk temporelle (auto si None)")
    parser.add_argument("--chunk_size_lat", type=int, default=None, help="Taille de chunk latitude (auto si None)")
    parser.add_argument("--chunk_size_lon", type=int, default=None, help="Taille de chunk longitude (auto si None)")

    args = parser.parse_args()

    convert_netcdf_to_zarr(
        lr_path=args.lr_path,
        hr_path=args.hr_path,
        output_dir=args.output_dir,
        static_path=args.static_path,
        seq_len=args.seq_len,
        baseline_strategy=args.baseline_strategy,
        baseline_factor=args.baseline_factor,
        normalize=args.normalize,
        target_transform=args.target_transform,
        lr_variables=args.lr_variables,
        hr_variables=args.hr_variables,
        static_variables=args.static_variables,
        means_path=args.means_path,
        stds_path=args.stds_path,
        chunk_size_time=args.chunk_size_time,
        chunk_size_lat=args.chunk_size_lat,
        chunk_size_lon=args.chunk_size_lon,
    )


if __name__ == "__main__":
    main()

