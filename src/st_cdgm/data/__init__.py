"""
Modules de gestion des données pour ST-CDGM.
"""

from .pipeline import (
    NetCDFDataPipeline,
    ZarrDataPipeline,
    ResDiffIterableDataset,
    WebDatasetIterableDataset,
)
from .netcdf_utils import NetCDFToDataFrame

__all__ = [
    "NetCDFDataPipeline",
    "ZarrDataPipeline",
    "ResDiffIterableDataset",
    "WebDatasetIterableDataset",
    "NetCDFToDataFrame",
]

