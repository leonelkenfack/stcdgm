"""
Programme pour transformer des fichiers NetCDF (.nc) en DataFrame pandas
Inspiré du code du projet downscaling
"""

import xarray as xr
import pandas as pd
import numpy as np
from pathlib import Path
from dask.diagnostics import ProgressBar
from typing import Union, List, Optional, Dict, Any
import json
import os
from datetime import datetime

# Import netCDF4 pour accès direct aux métadonnées brutes
try:
    import netCDF4
    NETCDF4_AVAILABLE = True
except ImportError:
    NETCDF4_AVAILABLE = False


class NetCDFToDataFrame:
    """
    Classe pour convertir des fichiers NetCDF climatiques en DataFrame pandas
    """

    def __init__(self, filepath: Union[str, Path],
                 chunks: Optional[Dict[str, int]] = None,
                 load_in_memory: bool = False):
        """
        Initialise le convertisseur avec un fichier NetCDF

        Parameters:
        -----------
        filepath : str ou Path
            Chemin vers le fichier NetCDF
        chunks : dict, optional
            Dictionnaire pour le chargement en chunks (ex: {"time": 1000})
            Utile pour les gros fichiers
        load_in_memory : bool
            Si True, charge toutes les données en mémoire
        """
        self.filepath = filepath
        self.chunks = chunks
        self.dataset = None

        # Ouvrir le dataset
        if chunks:
            self.dataset = xr.open_dataset(filepath, chunks=chunks)
        else:
            self.dataset = xr.open_dataset(filepath)

        if load_in_memory:
            with ProgressBar():
                self.dataset = self.dataset.load()

    def format_time(self):
        """
        Formate la dimension temporelle au format datetime pandas
        Similaire à ce qui est fait dans process_input_training_data.py
        """
        if 'time' in self.dataset.coords:
            self.dataset['time'] = pd.to_datetime(
                self.dataset.time.dt.strftime("%Y-%m-%d")
            )

    def select_region(self,
                     lat_slice: Optional[slice] = None,
                     lon_slice: Optional[slice] = None,
                     lat_range: Optional[tuple] = None,
                     lon_range: Optional[tuple] = None):
        """
        Sélectionne une région géographique

        Parameters:
        -----------
        lat_slice : slice, optional
            Tranche de latitude (ex: slice(-65, -25))
        lon_slice : slice, optional
            Tranche de longitude (ex: slice(150, 220.5))
        lat_range : tuple, optional
            Tuple (lat_min, lat_max) pour créer automatiquement le slice
        lon_range : tuple, optional
            Tuple (lon_min, lon_max) pour créer automatiquement le slice
        """
        if lat_range:
            lat_slice = slice(lat_range[0], lat_range[1])
        if lon_range:
            lon_slice = slice(lon_range[0], lon_range[1])

        if lat_slice or lon_slice:
            sel_dict = {}
            if lat_slice:
                sel_dict['lat'] = lat_slice
            if lon_slice:
                sel_dict['lon'] = lon_slice
            self.dataset = self.dataset.sel(**sel_dict)

    def select_time_period(self,
                          time_slice: Optional[slice] = None,
                          start_date: Optional[str] = None,
                          end_date: Optional[str] = None):
        """
        Sélectionne une période temporelle

        Parameters:
        -----------
        time_slice : slice, optional
            Tranche temporelle (ex: slice("1975", "2014"))
        start_date : str, optional
            Date de début (format: "YYYY-MM-DD" ou "YYYY")
        end_date : str, optional
            Date de fin (format: "YYYY-MM-DD" ou "YYYY")
        """
        if start_date and end_date:
            time_slice = slice(start_date, end_date)
        elif start_date:
            time_slice = slice(start_date, None)
        elif end_date:
            time_slice = slice(None, end_date)

        if time_slice:
            self.dataset = self.dataset.sel(time=time_slice)

    def normalize_variables(self,
                           variable_names: List[str],
                           means_dataset: Optional[xr.Dataset] = None,
                           stds_dataset: Optional[xr.Dataset] = None,
                           means_filepath: Optional[str] = None,
                           stds_filepath: Optional[str] = None):
        """
        Normalise les variables climatiques

        Parameters:
        -----------
        variable_names : list
            Liste des noms de variables à normaliser
        means_dataset : xr.Dataset, optional
            Dataset contenant les moyennes
        stds_dataset : xr.Dataset, optional
            Dataset contenant les écarts-types
        means_filepath : str, optional
            Chemin vers le fichier des moyennes
        stds_filepath : str, optional
            Chemin vers le fichier des écarts-types
        """
        # Charger les moyennes et écarts-types si nécessaire
        if means_filepath:
            means_dataset = xr.open_dataset(means_filepath)
        if stds_filepath:
            stds_dataset = xr.open_dataset(stds_filepath)

        # Normaliser (comme dans prepare_training_data)
        if means_dataset and stds_dataset:
            for var in variable_names:
                if var in self.dataset.data_vars:
                    mean_val = means_dataset[var].mean() if var in means_dataset else means_dataset[var]
                    std_val = stds_dataset[var].mean() if var in stds_dataset else stds_dataset[var]
                    self.dataset[var] = (self.dataset[var] - mean_val) / std_val

    def to_dataframe(self,
                    variable_name: Optional[str] = None,
                    method: str = 'point',
                    drop_na: bool = True) -> pd.DataFrame:
        """
        Convertit le dataset en DataFrame pandas

        Parameters:
        -----------
        variable_name : str, optional
            Nom de la variable à convertir. Si None, convertit toutes les variables
        method : str
            'point' : chaque point spatial devient une colonne
            'stacked' : empile toutes les dimensions sauf time
            'mean' : moyenne spatiale par date
        drop_na : bool
            Si True, supprime les valeurs NaN

        Returns:
        --------
        pd.DataFrame
            DataFrame avec les données NetCDF
        """
        # Vérifier si la variable existe
        if variable_name:
            if variable_name not in self.dataset.data_vars:
                available_vars = list(self.dataset.data_vars.keys())
                raise KeyError(
                    f"Variable '{variable_name}' non trouvée dans le dataset. "
                    f"Variables disponibles: {available_vars}"
                )

        if method == 'point':
            # Conversion simple : chaque point spatial = colonne
            if variable_name:
                data = self.dataset[variable_name]
            else:
                # Prendre la première variable si non spécifiée
                available_vars = list(self.dataset.data_vars.keys())
                if not available_vars:
                    raise ValueError("Aucune variable trouvée dans le dataset")
                data = self.dataset[available_vars[0]]
                variable_name = available_vars[0]

            df = data.to_dataframe(name=variable_name or 'value')
            df = df.reset_index()

            if drop_na:
                df = df.dropna()
            return df

        elif method == 'stacked':
            # Empile toutes les dimensions sauf time
            if variable_name:
                data = self.dataset[variable_name]
            else:
                # Prendre la première variable si non spécifiée
                available_vars = list(self.dataset.data_vars.keys())
                if not available_vars:
                    raise ValueError("Aucune variable trouvée dans le dataset")
                data = self.dataset[available_vars[0]]
                variable_name = available_vars[0]

            # Stack les dimensions spatiales
            stacked = data.stack(point=('lat', 'lon'))
            df = stacked.to_dataframe(name=variable_name or 'value')
            df = df.reset_index()

            if drop_na:
                df = df.dropna()
            return df

        elif method == 'mean':
            # Moyenne spatiale par date
            if variable_name:
                data = self.dataset[variable_name]
            else:
                # Prendre la première variable si non spécifiée
                available_vars = list(self.dataset.data_vars.keys())
                if not available_vars:
                    raise ValueError("Aucune variable trouvée dans le dataset")
                data = self.dataset[available_vars[0]]
                variable_name = available_vars[0]

            # Moyenne sur les dimensions spatiales
            data_mean = data.mean(['lat', 'lon'])
            df = data_mean.to_pandas()
            df = df.reset_index()

            return df

        else:
            raise ValueError(f"Méthode '{method}' non reconnue. Utilisez 'point', 'stacked', ou 'mean'")

    def to_dataframe_multiple_vars(self,
                                   variable_names: List[str],
                                   method: str = 'stacked') -> pd.DataFrame:
        """
        Convertit plusieurs variables en un seul DataFrame

        Parameters:
        -----------
        variable_names : list
            Liste des noms de variables
        method : str
            Méthode de conversion (voir to_dataframe)

        Returns:
        --------
        pd.DataFrame
            DataFrame avec plusieurs variables comme colonnes
        """
        dfs = []
        for var in variable_names:
            if var in self.dataset.data_vars:
                df_var = self.to_dataframe(variable_name=var, method=method, drop_na=False)
                dfs.append(df_var)

        # Fusionner les DataFrames
        if dfs:
            df_final = dfs[0]
            for df in dfs[1:]:
                df_final = pd.merge(df_final, df, on=['time', 'lat', 'lon'], how='outer')
            return df_final
        else:
            raise ValueError("Aucune variable trouvée dans le dataset")

    def extract_all_metadata(self) -> Dict[str, Any]:
        """
        Extrait TOUTES les métadonnées possibles du fichier NetCDF sans exception

        Returns:
        --------
        dict
            Dictionnaire contenant toutes les métadonnées extraites
        """
        metadata = {
            'file_info': {},
            'dimensions': {},
            'coordinates': {},
            'data_variables': {},
            'global_attributes': {},
            'cf_standard_attributes': {},  # Attributs CF-Conventions standards
            'encoding': {},
            'file_structure': {},
            'statistics': {},
            'cf_conventions': {}
        }

        # Informations sur le fichier
        file_path = Path(self.filepath)
        metadata['file_info'] = {
            'filepath': str(self.filepath),
            'filename': file_path.name,
            'file_size_bytes': os.path.getsize(self.filepath) if os.path.exists(self.filepath) else None,
            'file_size_mb': round(os.path.getsize(self.filepath) / (1024*1024), 2) if os.path.exists(self.filepath) else None,
            'file_extension': file_path.suffix,
            'file_format': self.dataset.encoding.get('source', 'unknown') if hasattr(self.dataset, 'encoding') else 'unknown'
        }

        # Dimensions avec détails complets
        for dim_name, dim_size in self.dataset.sizes.items():
            dim_info = {
                'size': int(dim_size),
                'unlimited': False  # sera mis à jour avec netCDF4 si disponible
            }

            # Extraire les coordonnées si elles existent
            if dim_name in self.dataset.coords:
                coord = self.dataset.coords[dim_name]

                # Fonction helper pour convertir les valeurs en format sérialisable
                def safe_convert_to_value(val):
                    """Convertit une valeur en format sérialisable"""
                    if val is None:
                        return None
                    try:
                        # Essayer de convertir en float
                        return float(val)
                    except (TypeError, ValueError):
                        # Si c'est une date/temps, convertir en string
                        try:
                            return str(val)
                        except:
                            return repr(val)

                dim_info.update({
                    'has_coordinates': True,
                    'dtype': str(coord.dtype),
                    'shape': list(coord.shape) if hasattr(coord, 'shape') else None,
                    'attributes': dict(coord.attrs) if coord.attrs else {},
                })

                # Extraire min/max/mean avec gestion des erreurs
                try:
                    if hasattr(coord, 'min') and coord.size > 0:
                        dim_info['min_value'] = safe_convert_to_value(coord.min().values)
                except Exception as e:
                    dim_info['min_value_error'] = str(e)

                try:
                    if hasattr(coord, 'max') and coord.size > 0:
                        dim_info['max_value'] = safe_convert_to_value(coord.max().values)
                except Exception as e:
                    dim_info['max_value_error'] = str(e)

                try:
                    if hasattr(coord, 'mean') and coord.size > 0:
                        dim_info['mean_value'] = safe_convert_to_value(coord.mean().values)
                except Exception as e:
                    dim_info['mean_value_error'] = str(e)

                # Échantillon de valeurs avec conversion safe
                try:
                    if hasattr(coord, 'values') and len(coord.values) > 0:
                        sample = coord.values[:10]
                        # Convertir en liste avec conversion safe
                        dim_info['values_sample'] = [safe_convert_to_value(v) for v in sample]
                except Exception as e:
                    dim_info['values_sample_error'] = str(e)
            else:
                dim_info['has_coordinates'] = False

            metadata['dimensions'][dim_name] = dim_info

        # Fonction helper pour convertir les valeurs
        def safe_convert_to_value(val):
            """Convertit une valeur en format sérialisable"""
            if val is None:
                return None
            try:
                return float(val)
            except (TypeError, ValueError):
                try:
                    return str(val)
                except:
                    return repr(val)

        # Coordonnées avec tous leurs attributs
        for coord_name in self.dataset.coords:
            if coord_name not in self.dataset.dims:  # Coordonnées non-dimensionnelles
                coord = self.dataset.coords[coord_name]
                coord_info = {
                    'dims': list(coord.dims),
                    'shape': list(coord.shape),
                    'dtype': str(coord.dtype),
                    'size': int(coord.size),
                    'attributes': dict(coord.attrs)
                }

                # Extraire min/max/mean avec gestion des erreurs
                try:
                    if coord.size > 0:
                        coord_info['min_value'] = safe_convert_to_value(coord.min().values)
                except Exception as e:
                    coord_info['min_value_error'] = str(e)

                try:
                    if coord.size > 0:
                        coord_info['max_value'] = safe_convert_to_value(coord.max().values)
                except Exception as e:
                    coord_info['max_value_error'] = str(e)

                try:
                    if coord.size > 0:
                        coord_info['mean_value'] = safe_convert_to_value(coord.mean().values)
                except Exception as e:
                    coord_info['mean_value_error'] = str(e)

                metadata['coordinates'][coord_name] = coord_info

        # Variables de données avec TOUS leurs attributs
        for var_name in self.dataset.data_vars:
            var = self.dataset.data_vars[var_name]
            var_info = {
                'dims': list(var.dims),
                'shape': list(var.shape),
                'dtype': str(var.dtype),
                'size': int(var.size),
                'nbytes': int(var.nbytes) if hasattr(var, 'nbytes') else None,
                'attributes': dict(var.attrs),
                'encoding': dict(var.encoding) if hasattr(var, 'encoding') and var.encoding else {}
            }

            # Statistiques sur les données (si possible)
            def safe_float_conversion(val):
                """Convertit en float si possible, sinon retourne None"""
                try:
                    return float(val)
                except (TypeError, ValueError):
                    return None

            try:
                if var.size > 0:
                    min_val = var.min().values
                    max_val = var.max().values
                    mean_val = var.mean().values

                    var_info['statistics'] = {
                        'min': safe_float_conversion(min_val),
                        'max': safe_float_conversion(max_val),
                        'mean': safe_float_conversion(mean_val),
                        'min_raw': str(min_val) if safe_float_conversion(min_val) is None else None,
                        'max_raw': str(max_val) if safe_float_conversion(max_val) is None else None,
                        'mean_raw': str(mean_val) if safe_float_conversion(mean_val) is None else None,
                    }

                    # STD et NaN seulement pour les types numériques
                    try:
                        if np.issubdtype(var.dtype, np.number):
                            var_info['statistics']['std'] = safe_float_conversion(var.std().values) if hasattr(var, 'std') else None
                            if hasattr(var.values, '__iter__'):
                                var_info['statistics']['has_nan'] = bool(np.isnan(var.values).any())
                                var_info['statistics']['nan_count'] = int(np.isnan(var.values).sum())
                    except Exception:
                        pass
            except Exception as e:
                var_info['statistics_error'] = str(e)

            # Informations sur les chunks si disponibles
            if hasattr(var, 'encoding') and 'chunksizes' in var.encoding:
                var_info['chunksizes'] = var.encoding['chunksizes']

            # Fill value
            if hasattr(var, 'encoding') and '_FillValue' in var.encoding:
                var_info['fill_value'] = var.encoding['_FillValue']
            elif hasattr(var, 'attrs') and '_FillValue' in var.attrs:
                var_info['fill_value'] = var.attrs['_FillValue']

            metadata['data_variables'][var_name] = var_info

        # Attributs globaux
        metadata['global_attributes'] = dict(self.dataset.attrs)

        # Extraire les attributs CF-Conventions standards spécifiques
        cf_standard_attrs = ['title', 'institution', 'source_id', 'experiment_id',
                             'grid_label', 'history', 'references', 'Conventions']
        for attr in cf_standard_attrs:
            if attr in self.dataset.attrs:
                metadata['cf_standard_attributes'][attr] = self.dataset.attrs[attr]

        # Encodings du dataset
        if hasattr(self.dataset, 'encoding'):
            metadata['encoding'] = dict(self.dataset.encoding)

        # Structure du fichier
        metadata['file_structure'] = {
            'has_groups': False,  # sera mis à jour avec netCDF4
            'number_of_dimensions': len(self.dataset.dims),
            'number_of_coordinates': len(self.dataset.coords),
            'number_of_data_variables': len(self.dataset.data_vars),
            'total_variables': len(self.dataset.variables)
        }

        # Informations CF Conventions si présentes
        cf_attrs = ['Conventions', 'featureType', 'history', 'institution', 'source',
                   'references', 'comment', 'title', 'summary', 'keywords',
                   'keywords_vocabulary', 'platform', 'product_version', 'date_created']
        for attr in cf_attrs:
            if attr in self.dataset.attrs:
                metadata['cf_conventions'][attr] = self.dataset.attrs[attr]

        # Extraire les informations CF-Conventions spécifiques pour chaque variable
        for var_name in self.dataset.data_vars:
            var = self.dataset.data_vars[var_name]
            var_attrs = var.attrs

            # Attributs CF-Conventions spécifiques
            cf_var_info = {}

            # Informations sur les cell methods
            if 'cell_methods' in var_attrs:
                cf_var_info['cell_methods'] = var_attrs['cell_methods']

            # Plages de valeurs valides
            if 'valid_range' in var_attrs:
                cf_var_info['valid_range'] = var_attrs['valid_range']
            if 'valid_min' in var_attrs:
                cf_var_info['valid_min'] = var_attrs['valid_min']
            if 'valid_max' in var_attrs:
                cf_var_info['valid_max'] = var_attrs['valid_max']

            # Flags pour données catégorielles
            if 'flag_values' in var_attrs:
                cf_var_info['flag_values'] = var_attrs['flag_values']
            if 'flag_meanings' in var_attrs:
                cf_var_info['flag_meanings'] = var_attrs['flag_meanings']
            if 'flag_masks' in var_attrs:
                cf_var_info['flag_masks'] = var_attrs['flag_masks']

            # Coordonnées auxiliaires
            if 'coordinates' in var_attrs:
                cf_var_info['coordinates'] = var_attrs['coordinates']
            if 'bounds' in var_attrs:
                cf_var_info['bounds'] = var_attrs['bounds']
            if 'ancillary_variables' in var_attrs:
                cf_var_info['ancillary_variables'] = var_attrs['ancillary_variables']

            # Direction pour coordonnées verticales
            if 'positive' in var_attrs:
                cf_var_info['positive'] = var_attrs['positive']

            # Grid mapping
            if 'grid_mapping' in var_attrs:
                cf_var_info['grid_mapping'] = var_attrs['grid_mapping']

            # Formula terms pour coordonnées
            if 'formula_terms' in var_attrs:
                cf_var_info['formula_terms'] = var_attrs['formula_terms']

            # Compression et scaling
            if 'scale_factor' in var_attrs:
                cf_var_info['scale_factor'] = var_attrs['scale_factor']
            if 'add_offset' in var_attrs:
                cf_var_info['add_offset'] = var_attrs['add_offset']

            if cf_var_info:
                if 'cf_conventions_attributes' not in metadata['data_variables'][var_name]:
                    metadata['data_variables'][var_name]['cf_conventions_attributes'] = {}
                metadata['data_variables'][var_name]['cf_conventions_attributes'] = cf_var_info

        # Accès direct avec netCDF4 pour informations supplémentaires
        if NETCDF4_AVAILABLE:
            try:
                with netCDF4.Dataset(self.filepath, 'r') as nc:
                    # Vérifier les dimensions illimitées
                    for dim_name, dim in nc.dimensions.items():
                        if dim_name in metadata['dimensions']:
                            metadata['dimensions'][dim_name]['unlimited'] = dim.isunlimited()
                            metadata['dimensions'][dim_name]['netcdf4_id'] = dim.__class__.__name__

                    # Informations sur les groupes
                    metadata['file_structure']['has_groups'] = hasattr(nc, 'groups') and len(nc.groups) > 0
                    if hasattr(nc, 'groups') and len(nc.groups) > 0:
                        metadata['file_structure']['groups'] = list(nc.groups.keys())
                        # Informations détaillées sur les groupes
                        group_info = {}
                        for group_name, group in nc.groups.items():
                            group_info[group_name] = {
                                'variables': list(group.variables.keys()),
                                'dimensions': list(group.dimensions.keys()),
                                'groups': list(group.groups.keys()) if hasattr(group, 'groups') else []
                            }
                        if group_info:
                            metadata['file_structure']['groups_detail'] = group_info

                    # Informations sur les variables brutes (accès direct)
                    for var_name in nc.variables:
                        if var_name in metadata['data_variables']:
                            nc_var = nc.variables[var_name]
                            nc4_info = {
                                'storage': nc_var.chunking() if hasattr(nc_var, 'chunking') else None,
                                'endianness': nc_var.endian() if hasattr(nc_var, 'endian') else None,
                                'all_attributes': {attr: getattr(nc_var, attr) for attr in nc_var.ncattrs()}
                            }

                            # Informations sur les filtres de compression
                            try:
                                if hasattr(nc_var, 'filters'):
                                    filters = nc_var.filters()
                                    if filters:
                                        nc4_info['compression'] = {
                                            'filters': filters,
                                            'is_compressed': True
                                        }
                                    else:
                                        nc4_info['compression'] = {'is_compressed': False}
                            except Exception:
                                pass

                            # Type de données spécial
                            if hasattr(nc_var, 'datatype'):
                                dt = nc_var.datatype
                                nc4_info['datatype'] = {
                                    'name': dt.name if hasattr(dt, 'name') else str(dt),
                                    'is_compound': dt.iscompound if hasattr(dt, 'iscompound') else False,
                                    'is_enum': dt.isenum if hasattr(dt, 'isenum') else False,
                                    'is_vlen': dt.isvlen if hasattr(dt, 'isvlen') else False,
                                    'is_opaque': dt.isopaquetype if hasattr(dt, 'isopaquetype') else False
                                }

                            metadata['data_variables'][var_name]['netcdf4_info'] = nc4_info

                    # Informations sur les types de données composés et enums
                    if hasattr(nc, 'datatypes'):
                        metadata['file_structure']['datatypes'] = {}
                        for dt_name, dt in nc.datatypes.items():
                            dt_info = {
                                'name': dt_name,
                                'is_compound': dt.iscompound if hasattr(dt, 'iscompound') else False,
                                'is_enum': dt.isenum if hasattr(dt, 'isenum') else False,
                                'size': dt.size if hasattr(dt, 'size') else None
                            }
                            if hasattr(dt, 'fields') and dt.fields:
                                dt_info['fields'] = {field: {'name': field, 'dtype': str(dt.fields[field][1])}
                                                    for field in dt.fields.keys()}
                            metadata['file_structure']['datatypes'][dt_name] = dt_info

                    # Informations sur les dimensions unlimited
                    unlimited_dims = [dim for dim in nc.dimensions.values() if dim.isunlimited()]
                    if unlimited_dims:
                        metadata['file_structure']['unlimited_dimensions'] = [dim.name for dim in unlimited_dims]

                    # Informations sur les chemin des variables dans les groupes
                    if hasattr(nc, 'path'):
                        metadata['file_structure']['root_path'] = nc.path

                    # Informations supplémentaires sur la structure
                    if hasattr(nc, 'isncfile'):
                        metadata['file_structure']['is_valid_netcdf'] = nc.isncfile()
            except Exception as e:
                metadata['netcdf4_extraction_error'] = str(e)

        return metadata

    def get_info(self, detailed: bool = True):
        """
        Affiche les informations sur le dataset

        Parameters:
        -----------
        detailed : bool
            Si True, affiche toutes les métadonnées détaillées
        """
        if detailed:
            metadata = self.extract_all_metadata()

            print("=" * 80)
            print("INFORMATIONS COMPLÈTES DU FICHIER NETCDF")
            print("=" * 80)

            # Informations fichier
            print("\n📁 INFORMATIONS FICHIER")
            print("-" * 80)
            for key, value in metadata['file_info'].items():
                print(f"  {key}: {value}")

            # Dimensions
            print("\n📏 DIMENSIONS")
            print("-" * 80)
            for dim_name, dim_info in metadata['dimensions'].items():
                print(f"\n  Dimension: {dim_name}")
                for key, value in dim_info.items():
                    if key != 'values_sample' or (key == 'values_sample' and value is not None):
                        if key == 'values_sample':
                            print(f"    {key}: {value} (échantillon)")
                        else:
                            print(f"    {key}: {value}")

            # Coordonnées
            if metadata['coordinates']:
                print("\n🗺️  COORDONNÉES NON-DIMENSIONNELLES")
                print("-" * 80)
                for coord_name, coord_info in metadata['coordinates'].items():
                    print(f"\n  Coordonnée: {coord_name}")
                    for key, value in coord_info.items():
                        print(f"    {key}: {value}")

            # Variables de données
            print("\n📊 VARIABLES DE DONNÉES")
            print("-" * 80)
            for var_name, var_info in metadata['data_variables'].items():
                print(f"\n  Variable: {var_name}")
                print(f"    Dimensions: {var_info['dims']}")
                print(f"    Shape: {var_info['shape']}")
                print(f"    Type: {var_info['dtype']}")
                print(f"    Taille: {var_info['size']} éléments")
                if var_info.get('nbytes'):
                    print(f"    Mémoire: {var_info['nbytes'] / (1024*1024):.2f} MB")

                if 'statistics' in var_info:
                    print(f"    Statistiques:")
                    for stat_key, stat_value in var_info['statistics'].items():
                        if stat_value is not None:
                            print(f"      {stat_key}: {stat_value}")

                if var_info.get('attributes'):
                    print(f"    Attributs ({len(var_info['attributes'])}):")
                    for attr_key, attr_value in var_info['attributes'].items():
                        if isinstance(attr_value, (list, tuple)) and len(str(attr_value)) > 100:
                            print(f"      {attr_key}: {str(attr_value)[:100]}...")
                        else:
                            print(f"      {attr_key}: {attr_value}")

                if var_info.get('encoding'):
                    print(f"    Encodage:")
                    for enc_key, enc_value in var_info['encoding'].items():
                        print(f"      {enc_key}: {enc_value}")

                # Attributs CF-Conventions spécifiques
                if var_info.get('cf_conventions_attributes'):
                    print(f"    Attributs CF-Conventions:")
                    for cf_key, cf_value in var_info['cf_conventions_attributes'].items():
                        print(f"      {cf_key}: {cf_value}")

                # Informations netCDF4 brutes si disponibles
                if var_info.get('netcdf4_info'):
                    print(f"    Informations NetCDF4:")
                    nc4 = var_info['netcdf4_info']
                    if nc4.get('compression'):
                        print(f"      Compression: {nc4['compression']}")
                    if nc4.get('datatype'):
                        print(f"      Type de données: {nc4['datatype']}")
                    if nc4.get('endianness'):
                        print(f"      Endianness: {nc4['endianness']}")
                    if nc4.get('storage'):
                        print(f"      Chunking: {nc4['storage']}")

            # Attributs globaux
            if metadata['global_attributes']:
                print("\n🌐 ATTRIBUTS GLOBAUX")
                print("-" * 80)
                for attr_key, attr_value in metadata['global_attributes'].items():
                    if isinstance(attr_value, (list, tuple)) and len(str(attr_value)) > 200:
                        print(f"  {attr_key}: {str(attr_value)[:200]}...")
                    else:
                        print(f"  {attr_key}: {attr_value}")

            # Structure
            print("\n🏗️  STRUCTURE DU FICHIER")
            print("-" * 80)
            for key, value in metadata['file_structure'].items():
                print(f"  {key}: {value}")

            # Attributs CF-Conventions standards
            if metadata['cf_standard_attributes']:
                print("\n📋 ATTRIBUTS CF-CONVENTIONS STANDARDS")
                print("-" * 80)
                for key, value in metadata['cf_standard_attributes'].items():
                    if isinstance(value, (list, tuple)) and len(str(value)) > 200:
                        print(f"  {key}: {str(value)[:200]}...")
                    else:
                        print(f"  {key}: {value}")

            # Conventions CF (autres attributs)
            if metadata['cf_conventions']:
                print("\n📋 AUTRES CONVENTIONS CF")
                print("-" * 80)
                for key, value in metadata['cf_conventions'].items():
                    if isinstance(value, (list, tuple)) and len(str(value)) > 200:
                        print(f"  {key}: {str(value)[:200]}...")
                    else:
                        print(f"  {key}: {value}")
        else:
            # Version simple (comme avant)
            print("=== Informations du Dataset NetCDF ===")
            print(f"Fichier: {self.filepath}")
            print(f"\nDimensions: {dict(self.dataset.sizes)}")
            print(f"\nCoordonnées: {list(self.dataset.coords.keys())}")
            print(f"\nVariables: {list(self.dataset.data_vars.keys())}")
            print(f"\nAttributs: {self.dataset.attrs}")

            if 'time' in self.dataset.coords:
                print(f"\nPériode temporelle: {self.dataset.time.min().values} à {self.dataset.time.max().values}")
            if 'lat' in self.dataset.coords:
                print(f"Latitude: {float(self.dataset.lat.min().values)} à {float(self.dataset.lat.max().values)}")
            if 'lon' in self.dataset.coords:
                print(f"Longitude: {float(self.dataset.lon.min().values)} à {float(self.dataset.lon.max().values)}")

    def export_metadata_to_json(self, output_file: Optional[str] = None) -> str:
        """
        Exporte toutes les métadonnées au format JSON

        Parameters:
        -----------
        output_file : str, optional
            Chemin du fichier de sortie. Si None, utilise le nom du fichier avec extension .json

        Returns:
        --------
        str
            Chemin du fichier JSON créé
        """
        metadata = self.extract_all_metadata()

        # Convertir les types non sérialisables
        def convert_to_serializable(obj):
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (datetime, pd.Timestamp)):
                return str(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_to_serializable(item) for item in obj]
            elif isinstance(obj, (bytes, bytearray)):
                return obj.decode('utf-8', errors='ignore')
            else:
                try:
                    json.dumps(obj)
                    return obj
                except (TypeError, ValueError):
                    return str(obj)

        metadata_serializable = convert_to_serializable(metadata)

        if output_file is None:
            file_path = Path(self.filepath)
            output_file = str(file_path.with_suffix('.metadata.json'))

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(metadata_serializable, f, indent=2, ensure_ascii=False)

        return output_file

    def export_metadata_to_csv(self, output_file: Optional[str] = None) -> str:
        """
        Exporte les métadonnées pertinentes des variables au format CSV
        (Les métadonnées statiques restent dans le JSON)

        Parameters:
        -----------
        output_file : str, optional
            Chemin du fichier de sortie

        Returns:
        --------
        str
            Chemin du fichier CSV créé
        """
        metadata = self.extract_all_metadata()

        # Fonction pour convertir les structures complexes en JSON string
        def to_csv_value(value):
            """Convertit une valeur en format approprié pour CSV"""
            if value is None:
                return ''
            elif isinstance(value, (dict, list)):
                # Convertir les structures complexes en JSON string
                try:
                    return json.dumps(value, ensure_ascii=False, default=str)
                except:
                    return str(value)
            elif isinstance(value, (np.integer, np.floating)):
                return float(value)
            elif isinstance(value, (bool, np.bool_)):
                return bool(value)
            elif isinstance(value, (bytes, bytearray)):
                return value.decode('utf-8', errors='ignore')
            else:
                return str(value)

        # Créer un DataFrame avec les métadonnées pertinentes des variables (une ligne par variable)
        rows = []
        for var_name, var_info in metadata['data_variables'].items():
            row = {
                'variable_name': var_name,
                'dims': str(var_info.get('dims', '')),
                'shape': str(var_info.get('shape', '')),
                'dtype': var_info.get('dtype', ''),
                'size': var_info.get('size', ''),
                'nbytes': var_info.get('nbytes', ''),
            }

            # Ajouter les statistiques si disponibles
            if 'statistics' in var_info and var_info['statistics']:
                for stat_key, stat_value in var_info['statistics'].items():
                    row[f'stat_{stat_key}'] = to_csv_value(stat_value)

            # Ajouter les attributs principaux (métadonnées pertinentes)
            attrs = var_info.get('attributes', {})
            if attrs:
                for attr_key, attr_value in attrs.items():
                    row[f'attr_{attr_key}'] = to_csv_value(attr_value)

            # Ajouter les informations d'encodage principales
            encoding = var_info.get('encoding', {})
            if encoding:
                for enc_key, enc_value in encoding.items():
                    row[f'encoding_{enc_key}'] = to_csv_value(enc_value)

            # Ajouter les attributs CF-Conventions
            cf_attrs = var_info.get('cf_conventions_attributes', {})
            if cf_attrs:
                for cf_key, cf_value in cf_attrs.items():
                    row[f'cf_{cf_key}'] = to_csv_value(cf_value)

            # Ajouter d'autres champs pertinents si présents
            if 'fill_value' in var_info:
                row['fill_value'] = to_csv_value(var_info['fill_value'])
            if 'chunksizes' in var_info:
                row['chunksizes'] = to_csv_value(var_info['chunksizes'])
            if 'statistics_error' in var_info:
                row['statistics_error'] = to_csv_value(var_info['statistics_error'])

            rows.append(row)

        df = pd.DataFrame(rows)

        if output_file is None:
            file_path = Path(self.filepath)
            output_file = str(file_path.with_suffix('.metadata.csv'))

        # Sauvegarder avec toutes les colonnes
        df.to_csv(output_file, index=False, encoding='utf-8')
        return output_file


def process_netcdf_file(filepath: str,
                        variable_name: Optional[str] = None,
                        variable_names: Optional[List[str]] = None,
                        lat_range: Optional[tuple] = None,
                        lon_range: Optional[tuple] = None,
                        time_slice: Optional[slice] = None,
                        normalize: bool = False,
                        means_filepath: Optional[str] = None,
                        stds_filepath: Optional[str] = None,
                        method: str = 'stacked',
                        chunks: Optional[Dict[str, int]] = None) -> pd.DataFrame:
    """
    Fonction utilitaire pour traiter un fichier NetCDF et le convertir en DataFrame

    Parameters:
    -----------
    filepath : str
        Chemin vers le fichier NetCDF
    variable_name : str, optional
        Nom d'une variable à extraire
    variable_names : list, optional
        Liste de variables à extraire
    lat_range : tuple, optional
        (lat_min, lat_max) pour sélectionner une région
    lon_range : tuple, optional
        (lon_min, lon_max) pour sélectionner une région
    time_slice : slice, optional
        Tranche temporelle à sélectionner
    normalize : bool
        Si True, normalise les variables
    means_filepath : str, optional
        Chemin vers le fichier des moyennes
    stds_filepath : str, optional
        Chemin vers le fichier des écarts-types
    method : str
        Méthode de conversion ('point', 'stacked', 'mean')
    chunks : dict, optional
        Chunks pour le chargement

    Returns:
    --------
    pd.DataFrame
        DataFrame avec les données
    """
    # Initialiser le convertisseur
    converter = NetCDFToDataFrame(filepath, chunks=chunks)

    # Formater le temps
    converter.format_time()

    # Sélectionner une région si nécessaire
    if lat_range or lon_range:
        converter.select_region(lat_range=lat_range, lon_range=lon_range)

    # Sélectionner une période si nécessaire
    if time_slice:
        converter.select_time_period(time_slice=time_slice)

    # Normaliser si nécessaire
    if normalize:
        vars_to_norm = variable_names or ([variable_name] if variable_name else [])
        converter.normalize_variables(
            vars_to_norm,
            means_filepath=means_filepath,
            stds_filepath=stds_filepath
        )

    # Convertir en DataFrame
    if variable_names:
        df = converter.to_dataframe_multiple_vars(variable_names, method=method)
    elif variable_name:
        df = converter.to_dataframe(variable_name, method=method)
    else:
        df = converter.to_dataframe(method=method)

    return df


# Exemple d'utilisation
if __name__ == "__main__":
    import sys
    import os

    # Exemple 1: Traitement simple d'un fichier
    # Utiliser le fichier fourni en argument ou un fichier par défaut
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        # Chercher un fichier NetCDF dans le répertoire courant
        nc_files = ['./NorESM2-MM_histupdated_compressed.nc']
        if nc_files:
            filepath = nc_files[0]
            print(f"Utilisation du fichier trouvé: {filepath}")
        else:
            print("Aucun fichier NetCDF trouvé. Veuillez spécifier un fichier en argument.")
            sys.exit(1)

    # Créer un convertisseur
    converter = NetCDFToDataFrame(filepath)

    # Extraire et afficher TOUTES les métadonnées
    print("\n" + "="*80)
    print("EXTRACTION COMPLÈTE DES MÉTADONNÉES")
    print("="*80)
    converter.get_info(detailed=True)

    # Exporter les métadonnées en JSON
    print("\n" + "="*80)
    print("EXPORT DES MÉTADONNÉES")
    print("="*80)
    try:
        json_file = converter.export_metadata_to_json()
        print(f"✓ Métadonnées exportées en JSON: {json_file}")

        csv_file = converter.export_metadata_to_csv()
        print(f"✓ Résumé exporté en CSV: {csv_file}")
    except Exception as e:
        print(f"Erreur lors de l'export: {e}")

    print("\n" + "="*80)
    print("FIN DU TRAITEMENT DES MÉTADONNÉES")
    print("="*80)
    print("Les données brutes ne sont plus exportées dans un CSV combiné.")

