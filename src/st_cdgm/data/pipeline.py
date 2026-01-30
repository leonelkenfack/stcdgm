"""
Module 1 - Data pipeline utilities for the ST-CDGM architecture.

This module prepares climate NetCDF datasets for the ST-CDGM pipeline:
  * loading and aligning LR/HR/static datasets,
  * optional normalisation and target-domain transforms,
  * construction of deterministic baselines and residual targets,
  * creation of streaming IterableDataset objects that yield
    ResDiff-style sequences (LR inputs, baselines, residuals, HR truth).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional, Sequence, Tuple
import json

import numpy as np
import pandas as pd
import xarray as xr

from .netcdf_utils import NetCDFToDataFrame

try:
    import cftime
    HAS_CFTIME = True
except ImportError:
    HAS_CFTIME = False

try:  # Optional torch dependency (required for IterableDataset)
    import torch
    from torch import Tensor
    from torch.utils.data import IterableDataset
except ImportError:  # pragma: no cover
    torch = None
    Tensor = None
    IterableDataset = None

try:
    import xbatcher
except ImportError:  # pragma: no cover
    xbatcher = None

try:
    import zarr
    HAS_ZARR = True
except ImportError:  # pragma: no cover
    HAS_ZARR = False
    zarr = None

try:
    import webdataset as wds
    HAS_WEBDATASET = True
except ImportError:  # pragma: no cover
    HAS_WEBDATASET = False
    wds = None

ArrayLike = np.ndarray
TransformFn = Callable[[xr.Dataset], xr.Dataset]


@dataclass
class GridMetadata:
    """Container describing the main dimension names used across datasets."""

    time: str
    lr_lat: str
    lr_lon: str
    hr_lat: str
    hr_lon: str


def _infer_dim(dataset: xr.Dataset, keyword: str) -> str:
    """Infer a dimension/coordinate name containing ``keyword`` (case-insensitive)."""
    keyword = keyword.lower()
    for name in dataset.dims:
        if keyword in name.lower():
            return name
    for name in dataset.coords:
        if keyword in name.lower():
            return name
    raise ValueError(f"Unable to infer dimension for '{keyword}' in dataset {list(dataset.dims)}")


def _ensure_callable_transform(
    transform: Optional[TransformFn | str],
    epsilon: float,
) -> Optional[TransformFn]:
    """Normalise transform specifications to callables."""
    if transform is None:
        return None
    if isinstance(transform, str):
        key = transform.lower()
        if key in {"log", "logarithm"}:
            return lambda ds: xr.apply_ufunc(
                lambda x: np.log(x + epsilon),
                ds,
                keep_attrs=True,
            )
        if key in {"log1p"}:
            return lambda ds: xr.apply_ufunc(
                lambda x: np.log1p(x),
                ds,
                keep_attrs=True,
            )
        raise ValueError(f"Unknown transform identifier '{transform}'.")
    if callable(transform):
        return transform
    raise TypeError("target_transform must be callable or string identifier.")


def _dataset_to_numpy(
    dataset: xr.Dataset,
    time_dim: str,
    lat_dim: str,
    lon_dim: str,
    spatial_shape: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """Convert an ``xr.Dataset`` to ``np.ndarray`` with shape (time, channel, lat, lon)."""
    # Convert to array (stacks variables into 'channel' dimension)
    array = dataset.to_array(dim="channel")
    
    # Get available dimensions after to_array
    available_dims = list(array.dims)
    
    # Check if we have the expected dimensions
    missing_dims = []
    if time_dim not in available_dims:
        missing_dims.append(time_dim)
    if lat_dim not in available_dims:
        missing_dims.append(lat_dim)
    if lon_dim not in available_dims:
        missing_dims.append(lon_dim)
    
    if missing_dims:
        # Try to find alternative dimension names
        # Sometimes dimensions might be in coordinates but not in dims
        # Or xbatcher might have renamed them
        
        # Check if dimensions exist as coordinates
        coord_dims = list(dataset.coords.keys())
        
        # Try to infer spatial dimensions from available dims
        spatial_candidates = [d for d in available_dims if d not in [time_dim, "channel"]]
        
        # If we have exactly 2 spatial candidates, use them
        if len(spatial_candidates) == 2:
            # Assume they are lat and lon (order might matter)
            inferred_lat = spatial_candidates[0]
            inferred_lon = spatial_candidates[1]
            lat_dim = inferred_lat
            lon_dim = inferred_lon
        elif "sample" in available_dims and spatial_shape is not None:
            # xbatcher has flattened spatial dims into 'sample'
            # It may have flattened lat, lon, and potentially other dims (like lev)
            lat_size, lon_size = spatial_shape
            
            # Get the current shape and find sample dimension index
            sample_idx = available_dims.index("sample")
            values = array.values
            
            # Calculate expected sample size (just lat * lon)
            expected_sample_size = lat_size * lon_size
            actual_sample_size = values.shape[sample_idx]
            
            # Check if there are additional dimensions that were flattened
            # (e.g., lev, level, depth, etc.)
            extra_dims_in_sample = actual_sample_size // expected_sample_size
            
            if extra_dims_in_sample == 1:
                # Simple case: only lat and lon were flattened
                new_shape = list(values.shape)
                new_shape[sample_idx] = lat_size
                new_shape.insert(sample_idx + 1, lon_size)
                values = values.reshape(new_shape)
                
                # Create new dimension names
                new_dims = list(available_dims)
                new_dims[sample_idx] = lat_dim
                new_dims.insert(sample_idx + 1, lon_dim)
            elif extra_dims_in_sample > 1 and actual_sample_size % expected_sample_size == 0:
                # Complex case: additional dimensions were flattened (e.g., lev)
                # We need to find what extra dimensions exist in the original dataset
                original_dims = set(dataset.dims)
                expected_dims = {time_dim, lat_dim, lon_dim}
                extra_dims_original = original_dims - expected_dims
                
                if extra_dims_original:
                    # Try to find the extra dimension that was flattened
                    # Usually it's 'lev', 'level', 'depth', etc.
                    extra_dim_name = None
                    extra_dim_size = None
                    for dim_name in ['lev', 'level', 'depth', 'z']:
                        if dim_name in dataset.dims:
                            extra_dim_name = dim_name
                            extra_dim_size = dataset.dims[dim_name]
                            break
                    
                    if extra_dim_name and extra_dim_size == extra_dims_in_sample:
                        # Reshape: sample -> extra_dim, lat, lon
                        new_shape = list(values.shape)
                        new_shape[sample_idx] = extra_dim_size
                        new_shape.insert(sample_idx + 1, lat_size)
                        new_shape.insert(sample_idx + 2, lon_size)
                        values = values.reshape(new_shape)
                        
                        # Create new dimension names
                        new_dims = list(available_dims)
                        new_dims[sample_idx] = extra_dim_name
                        new_dims.insert(sample_idx + 1, lat_dim)
                        new_dims.insert(sample_idx + 2, lon_dim)
                        
                        # After reshape, we need to average or select a level
                        # For now, let's average across the extra dimension
                        # (or we could select the first level)
                        extra_dim_idx = new_dims.index(extra_dim_name)
                        values = values.mean(axis=extra_dim_idx)
                        new_dims.pop(extra_dim_idx)
                    else:
                        # Fallback: just reshape assuming the extra dimension
                        new_shape = list(values.shape)
                        new_shape[sample_idx] = extra_dims_in_sample
                        new_shape.insert(sample_idx + 1, lat_size)
                        new_shape.insert(sample_idx + 2, lon_size)
                        values = values.reshape(new_shape)
                        
                        # Average across the first extra dimension
                        values = values.mean(axis=sample_idx)
                        new_shape.pop(sample_idx)
                        
                        # Create new dimension names
                        new_dims = list(available_dims)
                        new_dims[sample_idx] = lat_dim
                        new_dims.insert(sample_idx + 1, lon_dim)
                else:
                    raise ValueError(
                        f"Cannot determine extra dimensions. "
                        f"Expected sample size {expected_sample_size}, got {actual_sample_size}. "
                        f"Ratio: {extra_dims_in_sample}, but no extra dimensions found in dataset."
                    )
            else:
                raise ValueError(
                    f"Cannot reshape 'sample' dimension: expected size {expected_sample_size} "
                    f"(lat={lat_size} * lon={lon_size}), but got {actual_sample_size}. "
                    f"The size is not a multiple of the expected size."
                )
            
            # Recreate the DataArray with new dimensions
            # Build coordinates carefully - only use coords that match the new dimensions exactly
            new_coords = {}
            for dim in new_dims:
                # For time and channel, use coordinates from array if they match
                if dim in array.coords:
                    coord = array.coords[dim]
                    # Only use if it has the correct single dimension
                    if hasattr(coord, 'dims') and coord.dims == (dim,):
                        new_coords[dim] = coord
                # For lat and lon that were reshaped, create simple index-based coordinates
                # Don't try to use dataset.coords as they may have wrong dimensions
                elif dim == lat_dim:
                    new_coords[dim] = np.arange(lat_size)
                elif dim == lon_dim:
                    new_coords[dim] = np.arange(lon_size)
            
            # Create DataArray - xarray will validate coordinates match dimensions
            array = xr.DataArray(values, dims=new_dims, coords=new_coords if new_coords else None)
            # Update available_dims for transpose
            available_dims = new_dims
        elif "sample" in available_dims:
            raise ValueError(
                f"xbatcher has flattened spatial dimensions into 'sample', "
                f"but spatial_shape was not provided. "
                f"Available dimensions: {available_dims}."
            )
        else:
            raise ValueError(
                f"Missing dimensions: {missing_dims}. "
                f"Available dimensions: {available_dims}. "
                f"Dataset coordinates: {coord_dims}. "
                f"Cannot infer spatial dimensions automatically."
            )
    
    # Now transpose to the expected order
    try:
        array = array.transpose(time_dim, "channel", lat_dim, lon_dim)
    except ValueError as e:
        raise ValueError(
            f"Failed to transpose array. "
            f"Available dims: {available_dims}, "
            f"Requested order: ({time_dim}, channel, {lat_dim}, {lon_dim}). "
            f"Error: {e}"
        )
    
    return array.values.astype(np.float32)


class NetCDFDataPipeline:
    """
    High-level data preparation pipeline for ST-CDGM training.

    Parameters
    ----------
    lr_path :
        Path to the low-resolution (LR) dataset (predictors).
    hr_path :
        Path to the high-resolution (HR) ground-truth dataset.
    static_path :
        Optional static fields at HR resolution (topography, land-use, ...).
    seq_len :
        Sequence length (number of time steps) for iterable datasets.
    baseline_strategy :
        Strategy to build deterministic baselines. Options: ``"hr_smoothing"`` (default),
        ``"lr_interp"`` (bilinear upsampling of LR to HR grid).
    baseline_factor :
        Coarsening factor used with ``hr_smoothing`` strategy.
    target_transform :
        Optional transform applied to HR/baseline datasets (callable or "log"/"log1p").
    target_inverse_transform :
        Optional inverse transform callable (used for evaluation/export).
    normalize :
        Whether to normalise LR predictors using per-variable mean/std.
    lr_variables / hr_variables / static_variables :
        Optional variable subsets to select from the respective datasets.
    means_path / stds_path :
        Optional pre-computed statistics for LR normalisation.
    chunks :
        Optional chunk sizes passed to ``xr.open_dataset``.
    """

    def __init__(
        self,
        lr_path: str | Path,
        hr_path: str | Path,
        static_path: Optional[str | Path] = None,
        *,
        seq_len: int = 10,
        baseline_strategy: str = "hr_smoothing",
        baseline_factor: int = 4,
        target_transform: Optional[TransformFn | str] = None,
        target_inverse_transform: Optional[TransformFn] = None,
        normalize: bool = False,
        nan_fill_strategy: str = "zero",
        precipitation_delta: float = 0.01,
        lr_variables: Optional[Sequence[str]] = None,
        hr_variables: Optional[Sequence[str]] = None,
        static_variables: Optional[Sequence[str]] = None,
        means_path: Optional[str | Path] = None,
        stds_path: Optional[str | Path] = None,
        chunks: Optional[Dict[str, int]] = None,
        transform_epsilon: float = 1e-6,
    ) -> None:
        if xbatcher is None:
            raise ImportError("xbatcher is required for ST-CDGM data streaming. Install it via `pip install xbatcher`.")

        self.seq_len = seq_len
        self.baseline_strategy = baseline_strategy
        self.baseline_factor = max(1, baseline_factor)
        self.normalize = normalize
        self.nan_fill_strategy = nan_fill_strategy
        self.precipitation_delta = precipitation_delta
        self.lr_path = Path(lr_path)
        self.hr_path = Path(hr_path)
        self.static_path = Path(static_path) if static_path else None
        self.means_path = Path(means_path) if means_path else None
        self.stds_path = Path(stds_path) if stds_path else None
        self.transform_epsilon = transform_epsilon
        self._chunks = chunks

        self._target_transform = _ensure_callable_transform(target_transform, transform_epsilon)
        self._target_inverse_transform = target_inverse_transform

        # ------------------------------------------------------------------
        # Load datasets (kept both raw + working copies)
        # ------------------------------------------------------------------
        self.lr_dataset_raw = self._open_dataset(self.lr_path)
        self.hr_dataset_raw = self._open_dataset(self.hr_path)
        self.static_dataset = self._open_dataset(self.static_path) if self.static_path else None

        if self.hr_dataset_raw is None:
            raise ValueError("High-resolution dataset is required for ST-CDGM training.")

        # Select variables if requested
        if lr_variables:
            missing = set(lr_variables) - set(self.lr_dataset_raw.data_vars)
            if missing:
                raise KeyError(f"LR variables not found: {missing}")
            self.lr_dataset_raw = self.lr_dataset_raw[lr_variables]
        if hr_variables:
            missing = set(hr_variables) - set(self.hr_dataset_raw.data_vars)
            if missing:
                raise KeyError(f"HR variables not found: {missing}")
            self.hr_dataset_raw = self.hr_dataset_raw[hr_variables]
        if self.static_dataset is not None and static_variables:
            missing = set(static_variables) - set(self.static_dataset.data_vars)
            if missing:
                raise KeyError(f"Static variables not found: {missing}")
            self.static_dataset = self.static_dataset[static_variables]

        # Infer shared dimension names
        self.dims = GridMetadata(
            time=_infer_dim(self.lr_dataset_raw, "time"),
            lr_lat=_infer_dim(self.lr_dataset_raw, "lat"),
            lr_lon=_infer_dim(self.lr_dataset_raw, "lon"),
            hr_lat=_infer_dim(self.hr_dataset_raw, "lat"),
            hr_lon=_infer_dim(self.hr_dataset_raw, "lon"),
        )

        # Align datasets along the shared temporal axis
        self.lr_dataset_raw, self.hr_dataset_raw = self._align_time(self.lr_dataset_raw, self.hr_dataset_raw)
        
        # Clean NaN values from datasets
        self.lr_dataset_raw = self._clean_nan_values(self.lr_dataset_raw, self.nan_fill_strategy)
        self.hr_dataset_raw = self._clean_nan_values(self.hr_dataset_raw, self.nan_fill_strategy)
        
        if self.static_dataset is not None:
            self.static_dataset = self.static_dataset.load()

        # Normalise LR predictors if requested
        if self.normalize:
            self.lr_dataset_normalised, self.lr_stats = self._normalise_lr_dataset(self.lr_dataset_raw)
        else:
            self.lr_dataset_normalised = self.lr_dataset_raw
            self.lr_stats: Dict[str, xr.Dataset] = {}

        # Prepare deterministic baselines and residual ground-truth
        self.baseline_raw = self._compute_baseline()
        self.hr_prepared = self._apply_target_transform(self.hr_dataset_raw)
        self.baseline_prepared = self._apply_target_transform(self.baseline_raw)
        self.residual_dataset = self.hr_prepared - self.baseline_prepared

        # Convenience handles used by downstream modules
        self.lr_dataset = self.lr_dataset_normalised
        self.hr_dataset = self.hr_prepared

        self.static_tensor_np: Optional[np.ndarray] = self._prepare_static_tensor()
        self.static_tensor_torch: Optional[Tensor] = (
            torch.from_numpy(self.static_tensor_np) if (torch is not None and self.static_tensor_np is not None) else None
        )

    # ------------------------------------------------------------------
    # Dataset opening & alignment helpers
    # ------------------------------------------------------------------
    def _open_dataset(self, path: Optional[Path]) -> Optional[xr.Dataset]:
        if path is None:
            return None
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")
        kwargs = {"chunks": self._chunks} if self._chunks else {}
        return xr.open_dataset(path, **kwargs)

    def _convert_cftime_to_datetime(self, time_values):
        """
        Convert cftime objects to pandas datetime, handling various calendar types.
        
        Args:
            time_values: Array of time values (could be cftime, datetime64, or other)
            
        Returns:
            pandas Index of datetime values
        """
        # Check if we have cftime objects
        if HAS_CFTIME and len(time_values) > 0 and isinstance(time_values[0], cftime.datetime):
            # Convert cftime to pandas datetime
            # Use the num2date approach for consistency
            datetime_list = []
            for t in time_values:
                # Convert cftime to standard datetime
                # cftime objects have year, month, day, hour, minute, second attributes
                try:
                    dt = pd.Timestamp(
                        year=t.year,
                        month=t.month,
                        day=t.day,
                        hour=t.hour,
                        minute=t.minute,
                        second=t.second
                    )
                    datetime_list.append(dt)
                except (ValueError, AttributeError):
                    # Fallback: convert to string and let pandas parse it
                    datetime_list.append(pd.Timestamp(str(t)))
            return pd.Index(datetime_list)
        else:
            # Standard datetime conversion
            try:
                return pd.Index(pd.to_datetime(time_values))
            except Exception:
                # Fallback: convert to string first
                return pd.Index(pd.to_datetime([str(t) for t in time_values]))

    def _align_time(self, lr_ds: xr.Dataset, hr_ds: xr.Dataset) -> Tuple[xr.Dataset, xr.Dataset]:
        # Convert time coordinates to comparable datetime format
        lr_times = self._convert_cftime_to_datetime(lr_ds[self.dims.time].values)
        hr_times = self._convert_cftime_to_datetime(hr_ds[self.dims.time].values)
        
        # Find common times
        common_times = lr_times.intersection(hr_times)
        if common_times.empty:
            raise ValueError("No overlapping timestamps between LR and HR datasets.")
        
        # Get indices of common times in original datasets
        lr_indices = [i for i, t in enumerate(lr_times) if t in common_times]
        hr_indices = [i for i, t in enumerate(hr_times) if t in common_times]
        
        # Select by integer index instead of coordinate value
        lr_aligned = lr_ds.isel({self.dims.time: lr_indices})
        hr_aligned = hr_ds.isel({self.dims.time: hr_indices})
        
        return lr_aligned, hr_aligned

    def _clean_nan_values(self, dataset: xr.Dataset, strategy: str = "zero") -> xr.Dataset:
        """
        Nettoie les valeurs NaN du dataset selon la stratégie spécifiée.
        
        Parameters
        ----------
        dataset : xr.Dataset
            Dataset à nettoyer
        strategy : str
            Stratégie de nettoyage : "zero" (remplacer par 0), "mean" (moyenne spatiale),
            ou "interpolate" (interpolation temporelle)
        
        Returns
        -------
        xr.Dataset
            Dataset nettoyé
        """
        if strategy == "zero":
            return dataset.fillna(0.0)
        elif strategy == "mean":
            # Remplacer par la moyenne spatiale pour chaque timestep
            return dataset.fillna(dataset.mean(dim=[self.grid_metadata.hr_lat, self.grid_metadata.hr_lon]))
        elif strategy == "interpolate":
            return dataset.interpolate_na(dim=self.grid_metadata.time, method="linear")
        else:
            raise ValueError(f"Unknown nan_fill_strategy: {strategy}. Must be 'zero', 'mean', or 'interpolate'")

    def _normalise_lr_dataset(self, dataset: xr.Dataset) -> Tuple[xr.Dataset, Dict[str, xr.Dataset]]:
        if self.means_path and self.stds_path:
            means = xr.open_dataset(self.means_path)
            stds = xr.open_dataset(self.stds_path)
        else:
            # Utiliser skipna=True pour ignorer les NaN dans le calcul des statistiques
            means = dataset.mean(dim=self.dims.time, skipna=True, keep_attrs=True)
            stds = dataset.std(dim=self.dims.time, skipna=True, keep_attrs=True)
        
        # Protection contre division par zéro avec epsilon
        epsilon = 1e-6
        stds = stds.where(stds > epsilon, other=1.0)
        normalised = (dataset - means) / stds
        
        # Remplacer les NaN résiduels par 0 (après normalisation)
        normalised = normalised.fillna(0.0)
        
        return normalised, {"mean": means, "std": stds}

    def _compute_baseline(self) -> xr.Dataset:
        if self.baseline_strategy == "lr_interp":
            mapping = {
                self.dims.lr_lat: self.hr_dataset_raw[self.dims.hr_lat],
                self.dims.lr_lon: self.hr_dataset_raw[self.dims.hr_lon],
            }
            baseline = self.lr_dataset_raw.interp(mapping, method="linear")
            baseline = baseline.rename({self.dims.lr_lat: self.dims.hr_lat, self.dims.lr_lon: self.dims.hr_lon})
            return baseline

        if self.baseline_strategy == "hr_smoothing":
            coarsen_kwargs = {
                self.dims.hr_lat: self.baseline_factor,
                self.dims.hr_lon: self.baseline_factor,
            }
            smoothed = self.hr_dataset_raw.coarsen(coarsen_kwargs, boundary="trim").mean(keep_attrs=True)
            baseline = smoothed.interp(
                {
                    self.dims.hr_lat: self.hr_dataset_raw[self.dims.hr_lat],
                    self.dims.hr_lon: self.hr_dataset_raw[self.dims.hr_lon],
                },
                method="linear",
            )
            return baseline

        raise ValueError(f"Unsupported baseline_strategy '{self.baseline_strategy}'.")

    def _detect_precipitation_vars(self, dataset: xr.Dataset) -> list[str]:
        """
        Détecte les variables de précipitation dans le dataset.
        
        Parameters
        ----------
        dataset : xr.Dataset
            Dataset à analyser
        
        Returns
        -------
        list[str]
            Liste des noms de variables de précipitation détectées
        """
        precip_keywords = ['pr', 'precip', 'precipitation', 'rain']
        return [var for var in dataset.data_vars 
                if any(kw in var.lower() for kw in precip_keywords)]

    def _apply_target_transform(self, dataset: xr.Dataset) -> xr.Dataset:
        """
        Applique la transformation cible avec détection automatique des précipitations.
        
        Si target_transform est None et que des variables de précipitation sont détectées,
        applique automatiquement log1p avec precipitation_delta.
        """
        if self._target_transform is not None:
            transformed = self._target_transform(dataset)
            if not isinstance(transformed, xr.Dataset):
                raise TypeError("target_transform must return an xarray.Dataset.")
            return transformed
        
        # Auto-détection : si précipitations et pas de transform explicite, utiliser log1p
        precip_vars = self._detect_precipitation_vars(dataset)
        if precip_vars:
            # Appliquer log1p uniquement aux variables de précipitation
            result = dataset.copy()
            for var in precip_vars:
                result[var] = xr.apply_ufunc(
                    lambda x: np.log1p(x + self.precipitation_delta),
                    dataset[var],
                    keep_attrs=True
                )
            return result
        
        return dataset

    def _prepare_static_tensor(self) -> Optional[np.ndarray]:
        if self.static_dataset is None:
            return None

        # Create a working copy
        ds = self.static_dataset.copy()

        # 1. Drop coordinate bounds if present as variables (common in climate data)
        # Find variables that look like bounds (containing 'bnds' or 'bounds')
        drop_vars = [v for v in ds.data_vars if "bnds" in str(v) or "bounds" in str(v)]
        if drop_vars:
            ds = ds.drop_vars(drop_vars)

        # 2. Squeeze singleton dimensions (e.g. depth=1, time=1)
        ds = ds.squeeze(drop=True)

        # 3. Handle potential remaining extra dimensions (e.g. depth > 1)
        # We expect only (lat, lon) to remain for static 2D fields
        expected_dims = {self.dims.hr_lat, self.dims.hr_lon}
        # Note: ds.dims might include 'time' if it wasn't squeezed out (unlikely for static)
        
        # Identify extra dimensions that are NOT lat/lon
        extra_dims = set(ds.dims) - expected_dims
        
        if extra_dims:
            # If we still have extra dims, selecting the first index is a reasonable default 
            # for "static predictors" which are typically 2D maps.
            # (e.g. selecting surface level if depth is present)
            isel_kwargs = {dim: 0 for dim in extra_dims}
            ds = ds.isel(**isel_kwargs, drop=True)

        # 4. Convert to array (stacks variables into 'channel' dimension)
        static_array = ds.to_array(dim="channel")
        
        # 5. Transpose to ensure (channel, lat, lon) order
        # We use ... to handle any edge cases, but at this point we should have 3 dims
        try:
            static_array = static_array.transpose("channel", self.dims.hr_lat, self.dims.hr_lon)
        except ValueError:
            # Fallback if strict transpose fails (should be covered by logic above, but for safety)
             static_array = static_array.transpose("channel", ...)

        return static_array.values.astype(np.float32)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get_lr_dataset(self) -> xr.Dataset:
        return self.lr_dataset

    def get_lr_stats(self) -> Dict[str, xr.Dataset]:
        return self.lr_stats

    def get_hr_dataset(self) -> xr.Dataset:
        return self.hr_dataset

    def get_baseline_dataset(self) -> xr.Dataset:
        return self.baseline_prepared

    def get_residual_dataset(self) -> xr.Dataset:
        return self.residual_dataset

    def get_static_dataset(self) -> Optional[xr.Dataset]:
        return self.static_dataset

    def get_target_inverse_transform(self) -> Optional[TransformFn]:
        return self._target_inverse_transform

    # ------------------------------------------------------------------
    # Metadata export helpers
    # ------------------------------------------------------------------
    def export_lr_metadata_to_json(self, output_file: Optional[str | Path] = None) -> str:
        converter = NetCDFToDataFrame(self.lr_path)
        return converter.export_metadata_to_json(str(output_file) if output_file else None)

    def export_lr_metadata_to_csv(self, output_file: Optional[str | Path] = None) -> str:
        converter = NetCDFToDataFrame(self.lr_path)
        return converter.export_metadata_to_csv(str(output_file) if output_file else None)

    def export_hr_metadata_to_json(self, output_file: Optional[str | Path] = None) -> str:
        converter = NetCDFToDataFrame(self.hr_path)
        return converter.export_metadata_to_json(str(output_file) if output_file else None)

    def export_hr_metadata_to_csv(self, output_file: Optional[str | Path] = None) -> str:
        converter = NetCDFToDataFrame(self.hr_path)
        return converter.export_metadata_to_csv(str(output_file) if output_file else None)

    # ------------------------------------------------------------------
    # IterableDataset factory
    # ------------------------------------------------------------------
    def build_sequence_dataset(
        self,
        *,
        seq_len: Optional[int] = None,
        stride: int = 1,
        drop_last: bool = True,
        as_torch: bool = True,
    ) -> "ResDiffIterableDataset":
        seq_len = seq_len or self.seq_len
        if as_torch and torch is None:
            raise ImportError("Torch is required to obtain PyTorch tensors. Install it via `pip install torch`.")
        if IterableDataset is None:
            raise ImportError("Torch IterableDataset is required. Install PyTorch to continue.")
        return ResDiffIterableDataset(
            lr_dataset=self.lr_dataset,
            baseline_dataset=self.baseline_prepared,
            residual_dataset=self.residual_dataset,
            hr_dataset=self.hr_dataset,
            static_tensor_np=self.static_tensor_np,
            static_tensor_torch=self.static_tensor_torch,
            dims=self.dims,
            seq_len=seq_len,
            stride=max(1, stride),
            drop_last=drop_last,
            as_torch=as_torch,
        )

class ResDiffIterableDataset(IterableDataset):
    """
    Streaming dataset yielding ResDiff-style batches for ST-CDGM training.

    Each yielded sample is a dictionary containing:
        * ``lr``        : (seq_len, channels_lr, lat_lr, lon_lr)
        * ``baseline``  : (seq_len, channels_hr, lat_hr, lon_hr)
        * ``residual``  : (seq_len, channels_hr, lat_hr, lon_hr)
        * ``hr``        : (seq_len, channels_hr, lat_hr, lon_hr)
        * ``static``    : (channels_static, lat_hr, lon_hr)  (optional)
        * ``time``      : sequence of timestamps
    """

    def __init__(
        self,
        *,
        lr_dataset: xr.Dataset,
        baseline_dataset: xr.Dataset,
        residual_dataset: xr.Dataset,
        hr_dataset: xr.Dataset,
        static_tensor_np: Optional[np.ndarray],
        static_tensor_torch: Optional[Tensor],
        dims: GridMetadata,
        seq_len: int,
        stride: int,
        drop_last: bool,
        as_torch: bool,
    ) -> None:
        if IterableDataset is None:
            raise ImportError("PyTorch IterableDataset unavailable. Install torch to use ResDiffIterableDataset.")

        self.seq_len = seq_len
        self.stride = max(1, stride)
        self.drop_last = drop_last
        self.as_torch = as_torch
        self.dims = dims
        self.static_tensor_np = static_tensor_np
        self.static_tensor_torch = static_tensor_torch

        overlap = max(seq_len - self.stride, 0)
        batch_kwargs = dict(
            input_dims={dims.time: seq_len},
            input_overlap={dims.time: overlap},
            preload_batch=False,
        )
        # Store original dataset shapes for potential reshaping if xbatcher flattens spatial dims
        self.lr_spatial_shape = (lr_dataset.dims[dims.lr_lat], lr_dataset.dims[dims.lr_lon])
        self.hr_spatial_shape = (hr_dataset.dims[dims.hr_lat], hr_dataset.dims[dims.hr_lon])
        
        self.lr_gen = xbatcher.BatchGenerator(lr_dataset, **batch_kwargs)
        self.baseline_gen = xbatcher.BatchGenerator(baseline_dataset, **batch_kwargs)
        self.residual_gen = xbatcher.BatchGenerator(residual_dataset, **batch_kwargs)
        self.hr_gen = xbatcher.BatchGenerator(hr_dataset, **batch_kwargs)

    def __iter__(self) -> Iterable[Dict[str, object]]:
        for lr_window, baseline_window, residual_window, hr_window in zip(
            self.lr_gen, self.baseline_gen, self.residual_gen, self.hr_gen
        ):
            if not self._window_has_required_length(lr_window):
                if self.drop_last:
                    continue
            yield self._format_sample(lr_window, baseline_window, residual_window, hr_window)

    # ------------------------------------------------------------------
    # Helper utilities
    # ------------------------------------------------------------------
    def _window_has_required_length(self, window: xr.Dataset) -> bool:
        return window.dims.get(self.dims.time, 0) == self.seq_len

    def _format_sample(
        self,
        lr_window: xr.Dataset,
        baseline_window: xr.Dataset,
        residual_window: xr.Dataset,
        hr_window: xr.Dataset,
    ) -> Dict[str, object]:
        lr_np = _dataset_to_numpy(
            lr_window, self.dims.time, self.dims.lr_lat, self.dims.lr_lon,
            spatial_shape=self.lr_spatial_shape
        )
        baseline_np = _dataset_to_numpy(
            baseline_window, self.dims.time, self.dims.hr_lat, self.dims.hr_lon,
            spatial_shape=self.hr_spatial_shape
        )
        residual_np = _dataset_to_numpy(
            residual_window, self.dims.time, self.dims.hr_lat, self.dims.hr_lon,
            spatial_shape=self.hr_spatial_shape
        )
        hr_np = _dataset_to_numpy(
            hr_window, self.dims.time, self.dims.hr_lat, self.dims.hr_lon,
            spatial_shape=self.hr_spatial_shape
        )

        if self.as_torch and torch is not None:
            sample = {
                "lr": torch.from_numpy(lr_np),
                "baseline": torch.from_numpy(baseline_np),
                "residual": torch.from_numpy(residual_np),
                "hr": torch.from_numpy(hr_np),
                "time": lr_window[self.dims.time].values,
            }
            if self.static_tensor_torch is not None:
                sample["static"] = self.static_tensor_torch
        else:
            sample = {
                "lr": lr_np,
                "baseline": baseline_np,
                "residual": residual_np,
                "hr": hr_np,
                "time": lr_window[self.dims.time].values,
            }
            if self.static_tensor_np is not None:
                sample["static"] = self.static_tensor_np
        return sample


class ZarrDataPipeline:
    """
    High-level data preparation pipeline for ST-CDGM training using pre-processed Zarr data.
    
    This class reads pre-processed Zarr datasets that have already been transformed
    (normalized, baseline computed, residuals calculated) and creates IterableDatasets
    for training.
    
    Parameters
    ----------
    zarr_dir :
        Directory containing the pre-processed Zarr datasets (lr.zarr, hr.zarr,
        baseline.zarr, residual.zarr, static.zarr, metadata.json).
    """
    
    def __init__(
        self,
        zarr_dir: str | Path,
    ) -> None:
        if not HAS_ZARR:
            raise ImportError(
                "Zarr support is not available. Install zarr via `pip install zarr`."
            )
        
        zarr_dir = Path(zarr_dir)
        if not zarr_dir.exists():
            raise ValueError(f"Zarr directory does not exist: {zarr_dir}")
        
        # Load metadata
        metadata_path = zarr_dir / "metadata.json"
        if not metadata_path.exists():
            raise ValueError(f"Metadata file not found: {metadata_path}")
        
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        
        self.zarr_dir = zarr_dir
        self.seq_len = metadata.get("seq_len", 10)
        
        # Load dimension metadata
        dims_dict = metadata.get("dims", {})
        self.dims = GridMetadata(
            time=dims_dict.get("time", "time"),
            lr_lat=dims_dict.get("lr_lat", "lat"),
            lr_lon=dims_dict.get("lr_lon", "lon"),
            hr_lat=dims_dict.get("hr_lat", "lat"),
            hr_lon=dims_dict.get("hr_lon", "lon"),
        )
        
        # Load Zarr datasets
        self.lr_dataset = xr.open_zarr(zarr_dir / "lr.zarr")
        self.hr_dataset = xr.open_zarr(zarr_dir / "hr.zarr")
        self.baseline_prepared = xr.open_zarr(zarr_dir / "baseline.zarr")
        self.residual_dataset = xr.open_zarr(zarr_dir / "residual.zarr")
        
        static_zarr_path = zarr_dir / "static.zarr"
        self.static_dataset = xr.open_zarr(static_zarr_path) if static_zarr_path.exists() else None
        
        # Prepare static tensor
        self.static_tensor_np: Optional[np.ndarray] = self._prepare_static_tensor()
        self.static_tensor_torch: Optional[Tensor] = (
            torch.from_numpy(self.static_tensor_np) if (torch is not None and self.static_tensor_np is not None) else None
        )
    
    def _prepare_static_tensor(self) -> Optional[np.ndarray]:
        """Prepare static tensor from static dataset (same logic as NetCDFDataPipeline)."""
        if self.static_dataset is None:
            return None
        
        # Create a working copy
        ds = self.static_dataset.copy()
        
        # 1. Drop coordinate bounds if present as variables
        drop_vars = [v for v in ds.data_vars if "bnds" in str(v) or "bounds" in str(v)]
        if drop_vars:
            ds = ds.drop_vars(drop_vars)
        
        # 2. Squeeze singleton dimensions
        ds = ds.squeeze(drop=True)
        
        # 3. Handle extra dimensions (expect only lat, lon)
        expected_dims = {self.dims.hr_lat, self.dims.hr_lon}
        extra_dims = set(ds.dims) - expected_dims
        
        if extra_dims:
            isel_kwargs = {dim: 0 for dim in extra_dims}
            ds = ds.isel(**isel_kwargs, drop=True)
        
        # 4. Convert to array and transpose
        static_array = ds.to_array(dim="channel")
        try:
            static_array = static_array.transpose("channel", self.dims.hr_lat, self.dims.hr_lon)
        except ValueError:
            static_array = static_array.transpose("channel", ...)
        
        return static_array.values.astype(np.float32)
    
    def build_sequence_dataset(
        self,
        *,
        seq_len: Optional[int] = None,
        stride: int = 1,
        drop_last: bool = True,
        as_torch: bool = True,
    ) -> "ResDiffIterableDataset":
        """
        Build an IterableDataset for training.
        
        Parameters
        ----------
        seq_len :
            Sequence length (defaults to the value from metadata).
        stride :
            Stride for sequence generation.
        drop_last :
            Whether to drop the last incomplete sequence.
        as_torch :
            Whether to return PyTorch tensors.
        
        Returns
        -------
        ResDiffIterableDataset
            IterableDataset yielding ResDiff-style batches.
        """
        seq_len = seq_len or self.seq_len
        if as_torch and torch is None:
            raise ImportError("Torch is required to obtain PyTorch tensors. Install it via `pip install torch`.")
        if IterableDataset is None:
            raise ImportError("Torch IterableDataset is required. Install PyTorch to continue.")
        return ResDiffIterableDataset(
            lr_dataset=self.lr_dataset,
            baseline_dataset=self.baseline_prepared,
            residual_dataset=self.residual_dataset,
            hr_dataset=self.hr_dataset,
            static_tensor_np=self.static_tensor_np,
            static_tensor_torch=self.static_tensor_torch,
            dims=self.dims,
            seq_len=seq_len,
            stride=max(1, stride),
            drop_last=drop_last,
            as_torch=as_torch,
        )
    
    def get_lr_dataset(self) -> xr.Dataset:
        """Return the low-resolution dataset."""
        return self.lr_dataset
    
    def get_hr_dataset(self) -> xr.Dataset:
        """Return the high-resolution dataset."""
        return self.hr_dataset
    
    def get_baseline_dataset(self) -> xr.Dataset:
        """Return the baseline dataset."""
        return self.baseline_prepared
    
    def get_residual_dataset(self) -> xr.Dataset:
        """Return the residual dataset."""
        return self.residual_dataset


class WebDatasetIterableDataset(IterableDataset):
    """
    Phase B3: Streaming dataset from WebDataset TAR shards.
    
    Reads samples from pre-processed TAR shards (created by preprocess_to_shards.py).
    Provides 5-10x better throughput than Zarr for sequential reading patterns.
    
    Each sample in the shard contains:
        * ``lr.pt``      : LR tensor (seq_len, channels_lr, lat_lr, lon_lr)
        * ``baseline.pt``: Baseline tensor (seq_len, channels_hr, lat_hr, lon_hr)
        * ``residual.pt``: Residual tensor (seq_len, channels_hr, lat_hr, lon_hr)
        * ``hr.pt``      : HR tensor (seq_len, channels_hr, lat_hr, lon_hr)
        * ``static.pt``  : Static tensor (channels_static, lat_hr, lon_hr) (optional)
        * ``time.json``  : Time metadata (JSON list of timestamps)
    """
    
    def __init__(
        self,
        shard_pattern: str | Path,
        *,
        metadata_path: Optional[str | Path] = None,
        shuffle: bool = False,
        shardshuffle: int = 100,
        shuffle_buffer_size: int = 1000,
    ) -> None:
        if IterableDataset is None:
            raise ImportError(
                "PyTorch IterableDataset unavailable. Install torch to use WebDatasetIterableDataset."
            )
        if not HAS_WEBDATASET:
            raise ImportError(
                "WebDataset is not installed. Install via: pip install webdataset"
            )
        
        self.shard_pattern = str(shard_pattern)
        self.metadata_path = Path(metadata_path) if metadata_path else None
        
        # Load metadata if provided
        self.metadata = {}
        if self.metadata_path and self.metadata_path.exists():
            import json
            with open(self.metadata_path, "r") as f:
                self.metadata = json.load(f)
        
        # Create WebDataset pipeline
        # Pattern: "data_{000000..000999}.tar" or "data_*.tar"
        dataset = wds.WebDataset(self.shard_pattern)
        
        # Shuffle shards if requested
        if shuffle:
            dataset = dataset.shuffle(shardshuffle)
        
        # Decode tensors and JSON
        dataset = dataset.decode("torch")  # Decode .pt files as PyTorch tensors
        
        # Map function to format samples
        dataset = dataset.map(self._format_sample)
        
        # Shuffle samples within buffer if requested
        if shuffle:
            dataset = dataset.shuffle(shuffle_buffer_size)
        
        self.dataset = dataset
    
    def _format_sample(self, sample: Dict) -> Dict[str, object]:
        """
        Format WebDataset sample to match ResDiffIterableDataset output format.
        """
        formatted = {}
        
        # Extract tensors (already decoded by WebDataset)
        if "lr.pt" in sample:
            formatted["lr"] = sample["lr.pt"]
        if "baseline.pt" in sample:
            formatted["baseline"] = sample["baseline.pt"]
        if "residual.pt" in sample:
            formatted["residual"] = sample["residual.pt"]
        if "hr.pt" in sample:
            formatted["hr"] = sample["hr.pt"]
        if "static.pt" in sample:
            formatted["static"] = sample["static.pt"]
        
        # Decode time metadata
        if "time.json" in sample:
            import json
            time_data = sample["time.json"]
            if isinstance(time_data, bytes):
                time_data = json.loads(time_data.decode('utf-8'))
            formatted["time"] = time_data
        
        return formatted
    
    def __iter__(self) -> Iterable[Dict[str, object]]:
        """Iterate over samples in the WebDataset."""
        return iter(self.dataset)
