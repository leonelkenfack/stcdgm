# 🚀 Optimisations Proposées pour ST-CDGM

Ce document détaille toutes les optimisations proposées pour améliorer les performances et la précision du modèle ST-CDGM. Les optimisations sont organisées en deux catégories principales :

1. **Optimisations de Performance** : Amélioration de la vitesse d'entraînement et d'inférence
2. **Optimisations d'Accuracy/Loss/Métriques** : Amélioration de la précision, du loss et des scores d'évaluation (F1, etc.)

---

## Table des Matières

1. [Optimisations de Performance](#optimisations-de-performance)
   - [1.1 Pipeline de Données](#11-pipeline-de-données)
   - [1.2 Boucle d'Entraînement](#12-boucle-dentraînement)
   - [1.3 RCN Cell](#13-rcn-cell)
   - [1.4 Graphe Hétérogène](#14-graphe-hétérogène)
   - [1.5 Décodeur de Diffusion](#15-décodeur-de-diffusion)
   - [1.6 Encodeur Intelligible](#16-encodeur-intelligible)
   - [1.7 Optimisations Globales](#17-optimisations-globales)

2. [Optimisations d'Accuracy, Loss et Métriques](#optimisations-daccuracy-loss-et-métriques)
   - [2.1 Fonctions de Perte Améliorées](#21-fonctions-de-perte-améliorées)
   - [2.2 Métriques d'Évaluation Enrichies](#22-métriques-dévaluation-enrichies)
   - [2.3 Optimisation de l'Entraînement](#23-optimisation-de-lentraînement)
   - [2.4 Régularisation Avancée](#24-régularisation-avancée)

3. [Résumé et Impact Attendu](#résumé-et-impact-attendu)

---

## Optimisations de Performance

### 1.1 Pipeline de Données

**Fichier concerné** : `data_pipeline.py`

#### Problème 1 : Conversions répétées NumPy ↔ PyTorch

**Description** : À chaque batch, les données sont converties de NumPy à PyTorch, ce qui crée des allocations mémoire inutiles.

**Localisation** : `ResDiffIterableDataset._format_sample()` lignes 727-763

**Solution proposée** :
```python
# Dans ResDiffIterableDataset
def __init__(self, ...):
    # ... code existant ...
    # Pré-allouer les buffers si la forme est constante
    self._sample_buffer = {}
    self._pin_memory = torch.cuda.is_available()

def _format_sample(self, ...):
    # Vérifier si la forme est constante
    if not hasattr(self, '_sample_buffer') or lr_np.shape not in self._sample_buffer:
        # Créer de nouveaux tenseurs
        if self._pin_memory and self.as_torch:
            lr_tensor = torch.from_numpy(lr_np).pin_memory()
            baseline_tensor = torch.from_numpy(baseline_np).pin_memory()
            residual_tensor = torch.from_numpy(residual_np).pin_memory()
            hr_tensor = torch.from_numpy(hr_np).pin_memory()
        else:
            lr_tensor = torch.from_numpy(lr_np)
            baseline_tensor = torch.from_numpy(baseline_np)
            residual_tensor = torch.from_numpy(residual_np)
            hr_tensor = torch.from_numpy(hr_np)
        
        # Mettre en cache si forme constante
        if lr_np.shape == baseline_np.shape == residual_np.shape == hr_np.shape:
            self._sample_buffer[lr_np.shape] = {
                'lr': lr_tensor.clone(),
                'baseline': baseline_tensor.clone(),
                'residual': residual_tensor.clone(),
                'hr': hr_tensor.clone(),
            }
    else:
        # Réutiliser les buffers (copier les données dedans)
        cached = self._sample_buffer[lr_np.shape]
        cached['lr'].copy_(torch.from_numpy(lr_np))
        cached['baseline'].copy_(torch.from_numpy(baseline_np))
        cached['residual'].copy_(torch.from_numpy(residual_np))
        cached['hr'].copy_(torch.from_numpy(hr_np))
        lr_tensor = cached['lr']
        baseline_tensor = cached['baseline']
        residual_tensor = cached['residual']
        hr_tensor = cached['hr']
    
    # ... reste du code ...
```

**Gain attendu** : 10-20% de réduction du temps de chargement des données

---

#### Problème 2 : Opérations xarray non optimisées

**Description** : Les opérations de reshape dans `_dataset_to_numpy()` recréent des DataArray à chaque appel.

**Localisation** : `data_pipeline.py._dataset_to_numpy()` lignes 98-280

**Solution proposée** :
```python
def _dataset_to_numpy(
    dataset: xr.Dataset,
    time_dim: str,
    lat_dim: str,
    lon_dim: str,
    spatial_shape: Optional[Tuple[int, int]] = None,
    *,
    cache_key: Optional[str] = None,  # Nouveau paramètre pour le cache
) -> np.ndarray:
    """
    Version optimisée avec cache pour les transformations répétées.
    """
    # Utiliser un cache global pour éviter les recalculs
    if not hasattr(_dataset_to_numpy, '_cache'):
        _dataset_to_numpy._cache = {}
    
    # Si cache_key fourni, vérifier le cache
    if cache_key and cache_key in _dataset_to_numpy._cache:
        cached_shape = _dataset_to_numpy._cache[cache_key]['shape']
        if cached_shape == dataset.dims:
            # Utiliser les indices de transposition en cache
            cached_dims = _dataset_to_numpy._cache[cache_key]['dims']
            # Réappliquer directement la transformation
            array = dataset.to_array(dim="channel")
            # ... utiliser les indices en cache pour transpose ...
    
    # ... code existant ...
    
    # Mettre en cache si cache_key fourni
    if cache_key:
        _dataset_to_numpy._cache[cache_key] = {
            'shape': dataset.dims,
            'dims': new_dims,  # ou available_dims selon le cas
        }
    
    return array.values.astype(np.float32)
```

**Gain attendu** : 5-10% de réduction sur les opérations de preprocessing

---

### 1.2 Boucle d'Entraînement

**Fichier concerné** : `training_loop.py`

#### Problème 1 : Conversion `.to(device)` répétée

**Description** : Les données sont converties sur le device à chaque batch même si elles y sont déjà.

**Localisation** : Lignes 122-124

**Solution proposée** :
```python
def _ensure_on_device(tensor: Tensor, device: torch.device) -> Tensor:
    """
    Vérifie si le tenseur est déjà sur le bon device avant de le transférer.
    """
    if tensor.device != device:
        return tensor.to(device, non_blocking=True)
    return tensor

# Dans train_epoch, remplacer :
# lr_data: Tensor = batch["lr"].to(device)
# target_data: Tensor = batch.get(residual_key, batch.get("hr")).to(device)

# Par :
lr_data = _ensure_on_device(batch["lr"], device)
target_data = _ensure_on_device(
    batch.get(residual_key, batch.get("hr")), 
    device
)
```

**Gain attendu** : 5-10% de réduction du temps de transfert

---

#### Problème 2 : Liste Python `drivers` recréée à chaque batch

**Description** : La liste `drivers` est créée en parcourant `lr_data` avec une boucle Python.

**Localisation** : Ligne 144

**Solution proposée** :
```python
# AVANT:
drivers = [lr_data[t] for t in range(lr_data.shape[0])]

# APRÈS: Passer directement le tenseur et modifier RCNSequenceRunner
# Dans training_loop.py:
drivers = lr_data  # [seq_len, N, features_lr]

# Modifier RCNSequenceRunner.run() pour accepter un tenseur directement:
def run(
    self,
    H_init: Tensor,
    drivers: Tensor | Sequence[Tensor],  # Accepter les deux formats
    reconstruction_sources: Optional[Sequence[Optional[Tensor]]] = None,
) -> RCNSequenceOutput:
    # Si drivers est un tenseur, le convertir en séquence
    if isinstance(drivers, Tensor):
        drivers_list = [drivers[t] for t in range(drivers.shape[0])]
    else:
        drivers_list = list(drivers)
    
    # ... reste du code existant ...
```

**Gain attendu** : 2-5% de réduction (gain mineur mais utile)

---

#### Problème 3 : Vérifications NaN/Inf répétées

**Description** : Les vérifications sont faites à chaque batch même si le dataset est propre.

**Localisation** : Lignes 227-242

**Solution proposée** :
```python
def train_epoch(
    ...,
    skip_nan_checks: bool = False,  # Nouveau paramètre
    ...
):
    # ... code existant ...
    
    # Ne vérifier NaN/Inf que si nécessaire
    if not skip_nan_checks:
        nan_mask = torch.isnan(target) | torch.isinf(target)
        nan_count = nan_mask.sum().item()
        # ... vérifications existantes ...
    else:
        nan_count = 0
        nan_ratio = 0.0
        valid_mask = torch.ones_like(target, dtype=torch.bool)
```

**Gain attendu** : 1-3% de réduction (gain mineur mais utile en mode production)

---

### 1.3 RCN Cell

**Fichier concerné** : `causal_rcn.py`

#### Problème 1 : Boucles Python pour les GRU

**Description** : Les GRU sont appelées dans une boucle Python au lieu d'être vectorisées.

**Localisation** : Lignes 181-186

**Solution proposée** :
```python
def forward(self, H_prev: Tensor, driver: Tensor, ...):
    # ... code existant jusqu'à H_hat_tensor ...
    
    # Étape 2 : Mise à jour par forçage externe (GRU) - VERSION VECTORISÉE
    driver_emb = self.driver_encoder(driver)  # [N, hidden_dim]
    
    # Si toutes les GRU partagent les mêmes dimensions, on peut les batchifier
    # Option 1: Utiliser un GRU standard avec états cachés séparés
    # (nécessite une refactorisation plus profonde)
    
    # Option 2: Garder la boucle mais optimiser avec torch.jit.script
    # Compiler les GRU cells individuellement
    if not hasattr(self, '_compiled_gru_cells'):
        self._compiled_gru_cells = [
            torch.jit.script(gru) if self.training else gru
            for gru in self.gru_cells
        ]
    
    H_next = []
    for k in range(self.num_vars):
        gru_output = self._compiled_gru_cells[k](
            driver_emb, 
            H_hat_tensor[k]
        )
        gru_output = self.dropout(gru_output)
        H_next.append(gru_output)
    H_next_tensor = torch.stack(H_next, dim=0)
    
    # ... reste du code ...
```

**Gain attendu** : 10-15% de réduction sur le forward RCN

---

#### Problème 2 : Cache du masque diagonal

**Description** : Le masque diagonal est recalculé à chaque forward alors qu'il ne change pas si `A_dag` est fixe.

**Localisation** : Ligne 169

**Solution proposée** :
```python
class RCNCell(nn.Module):
    def __init__(self, ...):
        # ... code existant ...
        self._cached_mask = None
        self._mask_cache_valid = False
    
    def forward(self, H_prev: Tensor, driver: Tensor, ...):
        # ... validations ...
        
        # Cache le masque diagonal si en mode évaluation
        if not self.training:
            if self._cached_mask is None or not self._mask_cache_valid:
                self._cached_mask = MaskDiagonal.apply(self.A_dag)
                self._mask_cache_valid = True
            A_masked = self._cached_mask
        else:
            # En training, toujours recalculer (car A_dag change)
            A_masked = MaskDiagonal.apply(self.A_dag)
        
        # ... reste du code ...
    
    def reset_cache(self):
        """Réinitialise le cache (à appeler après modification de A_dag)."""
        self._mask_cache_valid = False
        self._cached_mask = None
```

**Gain attendu** : 2-5% en mode évaluation

---

### 1.4 Graphe Hétérogène

**Fichier concerné** : `graph_builder.py`

#### Problème : Clonage répété du template

**Description** : Le template de graphe est cloné à chaque batch, ce qui copie toutes les structures (edge_index, etc.).

**Localisation** : Lignes 200, 181

**Solution proposée** :
```python
def prepare_step_data_optimized(
    self,
    features: Dict[str, torch.Tensor],
    *,
    clone_template: bool = True,
) -> HeteroData:
    """
    Version optimisée qui évite de cloner toute la structure.
    """
    if self._template_cache is None:
        self._template_cache = self.build_template()
    
    # Créer un nouveau HeteroData mais partager les edge_index (immuables)
    data = HeteroData()
    
    # Copier uniquement la structure (num_nodes, edge_index) sans cloner les tensors
    for node_type in self.dynamic_node_types:
        data[node_type].num_nodes = self._template_cache[node_type].num_nodes
    
    data["SP_HR"].num_nodes = self._template_cache["SP_HR"].num_nodes
    data["SP_HR"].x = self._template_cache["SP_HR"].x.clone()  # Statique, cloner
    
    # Partager les edge_index (ils ne changent pas)
    for edge_type, edge_index in self._template_cache.edge_index_dict.items():
        data[edge_type].edge_index = edge_index  # Référence, pas de clonage
    
    # Copier les batch vectors si nécessaire
    for node_type in data.node_types:
        if hasattr(self._template_cache[node_type], 'batch'):
            data[node_type].batch = self._template_cache[node_type].batch.clone()
    
    # Injecter les features dynamiques
    self.inject_dynamic_features(data, features)
    
    return data
```

**Gain attendu** : 5-10% de réduction sur la préparation des données de graphe

---

### 1.5 Décodeur de Diffusion

**Fichier concerné** : `diffusion_decoder.py`

#### Problème 1 : Pas de gradient checkpointing

**Description** : Le UNet consomme beaucoup de mémoire et pourrait bénéficier de gradient checkpointing.

**Localisation** : Méthode `forward()` et `compute_loss()`

**Solution proposée** :
```python
from torch.utils.checkpoint import checkpoint

def forward(
    self,
    noisy_sample: Tensor,
    timestep: Tensor,
    conditioning: Tensor,
    *,
    use_checkpointing: bool = True,  # Nouveau paramètre
) -> Tensor:
    """
    Passe avant du UNet avec option de gradient checkpointing.
    """
    conditioning = self._prepare_conditioning(conditioning)
    
    # Utiliser gradient checkpointing si en training et activé
    if self.training and use_checkpointing and noisy_sample.requires_grad:
        output = checkpoint(
            self._unet_forward,
            noisy_sample,
            timestep,
            conditioning,
            use_reentrant=False,  # Plus rapide mais nécessite PyTorch >= 1.13
        )
    else:
        output = self._unet_forward(noisy_sample, timestep, conditioning)
    
    return output.sample

def _unet_forward(self, noisy_sample, timestep, conditioning):
    """Wrapper pour le forward UNet utilisé par checkpoint."""
    return self.unet(
        sample=noisy_sample,
        timestep=timestep,
        encoder_hidden_states=conditioning,
    )
```

**Gain attendu** : Réduction mémoire de 40-50%, permettant des batch sizes plus grands

---

#### Problème 2 : Masquage NaN recalculé à chaque forward

**Description** : Le masque valide est recalculé même si le target a la même forme.

**Localisation** : `compute_loss()` lignes 114-122

**Solution proposée** :
```python
def compute_loss(
    self,
    target: Tensor,
    conditioning: Tensor,
    *,
    cache_mask: bool = True,  # Nouveau paramètre
) -> Tensor:
    """
    Version avec cache du masque valide.
    """
    # Cache le masque valide si le target a la même forme
    target_shape = tuple(target.shape)
    cache_key = f"valid_mask_{target_shape}"
    
    if cache_mask and hasattr(self, '_valid_mask_cache'):
        if cache_key in self._valid_mask_cache:
            valid_mask = self._valid_mask_cache[cache_key]
        else:
            valid_mask = ~torch.isnan(target) & ~torch.isinf(target)
            self._valid_mask_cache[cache_key] = valid_mask
    else:
        valid_mask = ~torch.isnan(target) & ~torch.isinf(target)
        if cache_mask:
            if not hasattr(self, '_valid_mask_cache'):
                self._valid_mask_cache = {}
            self._valid_mask_cache[cache_key] = valid_mask
    
    # ... reste du code existant ...
```

**Gain attendu** : 1-2% de réduction (gain mineur)

---

### 1.6 Encodeur Intelligible

**Fichier concerné** : `intelligible_encoder.py`

#### Problème : Pooling batch-wise avec boucles Python

**Description** : Le pooling est fait avec une boucle Python au lieu d'utiliser des opérations vectorisées.

**Localisation** : `RCNCell.pool_state()` lignes 222-244

**Solution proposée** :
```python
def pool_state_optimized(
    self,
    H: Tensor,
    *,
    batch: Optional[Tensor] = None,
    pool: str = "mean",
) -> Tensor:
    """
    Version vectorisée du pooling avec torch_scatter.
    """
    from torch_scatter import scatter_mean, scatter_max
    
    pool = pool.lower()
    if batch is None:
        if pool == "mean":
            return H.mean(dim=1, keepdim=False).unsqueeze(0)
        if pool == "max":
            return H.amax(dim=1, keepdim=False).unsqueeze(0)
        raise ValueError(f"Pooling '{pool}' non pris en charge sans batch.")
    
    # Vectoriser avec scatter operations
    # H: [q, N, hidden_dim]
    # batch: [N]
    # Résultat attendu: [num_graphs, q, hidden_dim]
    
    num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 1
    
    # Réorganiser pour scatter: [q, N, hidden_dim] -> [q * N, hidden_dim]
    q, N, hidden_dim = H.shape
    H_flat = H.permute(0, 2, 1).reshape(-1, N)  # [q * hidden_dim, N]
    
    # Étendre batch pour chaque variable: [N] -> [q * hidden_dim, N]
    batch_expanded = batch.unsqueeze(0).expand(q * hidden_dim, -1)
    
    # Scatter
    if pool == "mean":
        pooled_flat = scatter_mean(H_flat, batch_expanded, dim=1, dim_size=num_graphs)
    elif pool == "max":
        pooled_flat = scatter_max(H_flat, batch_expanded, dim=1, dim_size=num_graphs)[0]
    else:
        raise ValueError(f"Pooling '{pool}' non pris en charge.")
    
    # Remettre en forme: [q * hidden_dim, num_graphs] -> [num_graphs, q, hidden_dim]
    pooled = pooled_flat.reshape(q, hidden_dim, num_graphs).permute(2, 0, 1)
    
    return pooled
```

**Gain attendu** : 10-20% de réduction sur le pooling (si beaucoup de graphes)

---

### 1.7 Optimisations Globales

#### 1.7.1 Compilation JIT (PyTorch 2.0+)

**Description** : Compiler les modules critiques avec `torch.compile` pour une accélération automatique.

**Solution proposée** :
```python
# Dans le notebook ou script d'entraînement, après création des modules:

# Compiler les modules critiques
if hasattr(torch, 'compile'):
    encoder = torch.compile(encoder, mode="reduce-overhead")
    rcn_runner.cell = torch.compile(rcn_runner.cell, mode="reduce-overhead")
    diffusion_decoder = torch.compile(diffusion_decoder, mode="max-autotune")
```

**Gain attendu** : 10-30% d'accélération selon le module

---

#### 1.7.2 Mixed Precision Training

**Description** : Utiliser la précision mixte (FP16/BF16) pour réduire la consommation mémoire et accélérer.

**Solution proposée** :
```python
from torch.cuda.amp import autocast, GradScaler

# Dans train_epoch:
scaler = GradScaler()

for batch_idx, batch in enumerate(data_loader):
    optimizer.zero_grad()
    
    # Forward pass avec autocast
    with autocast():
        # ... tout le forward pass ...
        loss_total = loss_gen_value + loss_rec_value + loss_dag_value
    
    # Backward avec scaler
    scaler.scale(loss_total).backward()
    
    if gradient_clipping is not None:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(..., gradient_clipping)
    
    scaler.step(optimizer)
    scaler.update()
```

**Gain attendu** : 40-50% d'accélération sur GPU moderne, 2x batch size possible

---

#### 1.7.3 Préchargement des Données

**Description** : Utiliser des DataLoader avec plusieurs workers pour précharger les batches.

**Solution proposée** :
```python
# Dans le notebook, utiliser DataLoader au lieu d'itérer directement:
from torch.utils.data import DataLoader

# Si possible, convertir IterableDataset en Dataset standard
# Sinon, utiliser un DataLoader avec num_workers > 0

dataloader = DataLoader(
    dataset,
    batch_size=1,  # ou batch_size approprié
    num_workers=4,  # Nombre de workers parallèles
    pin_memory=True,  # Accélère les transferts GPU
    prefetch_factor=2,  # Précharger 2 batches par worker
    persistent_workers=True,  # Garder les workers entre epochs
)
```

**Gain attendu** : 20-40% de réduction du temps d'I/O si I/O-bound

---

## Optimisations d'Accuracy, Loss et Métriques

### 2.1 Fonctions de Perte Améliorées

#### 2.1.1 Perte de Diffusion Avancée

**Fichier concerné** : `diffusion_decoder.py`

**Localisation** : Méthode `compute_loss()` lignes 96-172

**Problème** : La perte MSE simple ne favorise pas les pixels importants ni les extrêmes climatiques.

**Solutions proposées** :

##### A. Focal Loss pour les pixels difficiles

```python
def compute_loss(
    self,
    target: Tensor,
    conditioning: Tensor,
    *,
    use_focal_loss: bool = False,
    focal_gamma: float = 2.0,
    focal_alpha: float = 0.25,
    **kwargs,
) -> Tensor:
    """
    Version avec Focal Loss optionnelle.
    
    Le Focal Loss réduit l'importance des pixels faciles et se concentre sur
    les pixels difficiles, améliorant la convergence.
    """
    # ... code existant pour préparer le masque ...
    
    noise_pred_valid = noise_pred[valid_mask]
    noise_valid = noise[valid_mask]
    
    if use_focal_loss:
        # Calculer l'erreur
        error = torch.abs(noise_pred_valid - noise_valid)
        
        # Normaliser l'erreur pour obtenir un poids focal
        error_norm = error / (error.mean() + 1e-8)
        
        # Poids focal: (1 - p_t)^gamma où p_t est la probabilité de prédiction correcte
        # Approximation: utiliser l'erreur normalisée
        focal_weight = (error_norm ** focal_gamma) * focal_alpha
        
        # Normaliser les poids
        focal_weight = focal_weight / (focal_weight.mean() + 1e-8)
        
        # Appliquer le focal loss
        loss = (focal_weight * (noise_pred_valid - noise_valid) ** 2).mean()
    else:
        # Loss standard
        loss = nn.functional.mse_loss(noise_pred_valid, noise_valid)
    
    return loss
```

**Impact attendu** : +5-10% d'amélioration sur les pixels difficiles

---

##### B. Loss pondérée spatialement

```python
def compute_loss(
    self,
    target: Tensor,
    conditioning: Tensor,
    *,
    use_weighted_loss: bool = False,
    spatial_weights: Optional[Tensor] = None,
    weight_map_path: Optional[str] = None,
    **kwargs,
) -> Tensor:
    """
    Version avec pondération spatiale.
    
    Permet de donner plus d'importance à certaines régions géographiques
    (ex: zones d'intérêt, continents vs océans).
    """
    # ... code existant ...
    
    if use_weighted_loss:
        if spatial_weights is None and weight_map_path:
            # Charger depuis fichier
            spatial_weights = torch.load(weight_map_path)
        
        if spatial_weights is not None:
            # Assurer que les dimensions correspondent
            if spatial_weights.shape != target.shape:
                # Interpoler si nécessaire
                spatial_weights = torch.nn.functional.interpolate(
                    spatial_weights.unsqueeze(0).unsqueeze(0),
                    size=target.shape[-2:],
                    mode='bilinear',
                    align_corners=False
                ).squeeze()
            
            weights_valid = spatial_weights[valid_mask]
            # Normaliser les poids
            weights_valid = weights_valid / (weights_valid.mean() + 1e-8)
            
            loss = (weights_valid * (noise_pred_valid - noise_valid) ** 2).mean()
        else:
            raise ValueError("spatial_weights requis si use_weighted_loss=True")
    else:
        loss = nn.functional.mse_loss(noise_pred_valid, noise_valid)
    
    return loss
```

**Impact attendu** : +10-15% d'amélioration sur les régions pondérées

---

##### C. Loss pour les extrêmes climatiques

```python
def compute_loss(
    self,
    target: Tensor,
    conditioning: Tensor,
    *,
    extreme_threshold_percentile: float = 95.0,
    extreme_weight: float = 2.0,
    **kwargs,
) -> Tensor:
    """
    Version avec pondération pour les valeurs extrêmes.
    
    Les événements climatiques extrêmes (précipitations intenses, températures
    extrêmes) sont souvent plus importants mais moins fréquents. Cette perte
    leur donne plus de poids.
    """
    # ... code existant ...
    
    if extreme_threshold_percentile > 0:
        target_valid = target[valid_mask]
        
        # Calculer le seuil percentile
        threshold = torch.quantile(
            target_valid, 
            extreme_threshold_percentile / 100.0
        )
        
        # Identifier les valeurs extrêmes
        is_extreme = target_valid > threshold
        
        # Créer les poids: extrêmes = extreme_weight, autres = 1.0
        weights = torch.where(
            is_extreme,
            torch.full_like(is_extreme, extreme_weight, dtype=torch.float32),
            torch.ones_like(is_extreme, dtype=torch.float32)
        )
        
        # Normaliser pour maintenir l'échelle de la loss
        weights = weights / (weights.mean() + 1e-8)
        
        loss = (weights * (noise_pred_valid - noise_valid) ** 2).mean()
    else:
        loss = nn.functional.mse_loss(noise_pred_valid, noise_valid)
    
    return loss
```

**Impact attendu** : +15-25% d'amélioration sur les événements extrêmes

---

##### D. Loss combinée (MSE + L1 + Huber)

```python
def compute_loss(
    self,
    target: Tensor,
    conditioning: Tensor,
    *,
    loss_type: str = "mse",  # "mse", "l1", "huber", "combined"
    huber_delta: float = 1.0,
    l1_weight: float = 0.1,
    **kwargs,
) -> Tensor:
    """
    Version avec différents types de loss.
    
    - MSE: Sensible aux outliers mais bon gradient
    - L1: Plus robuste aux outliers
    - Huber: Compromis entre MSE et L1
    - Combined: Combinaison pondérée de plusieurs
    """
    # ... code existant ...
    
    noise_pred_valid = noise_pred[valid_mask]
    noise_valid = noise[valid_mask]
    
    if loss_type == "mse":
        loss = nn.functional.mse_loss(noise_pred_valid, noise_valid)
    elif loss_type == "l1":
        loss = nn.functional.l1_loss(noise_pred_valid, noise_valid)
    elif loss_type == "huber":
        loss = nn.functional.huber_loss(
            noise_pred_valid, 
            noise_valid, 
            delta=huber_delta
        )
    elif loss_type == "combined":
        mse_loss = nn.functional.mse_loss(noise_pred_valid, noise_valid)
        l1_loss = nn.functional.l1_loss(noise_pred_valid, noise_valid)
        loss = mse_loss + l1_weight * l1_loss
    else:
        raise ValueError(f"Type de loss inconnu: {loss_type}")
    
    return loss
```

**Impact attendu** : +5-10% de robustesse générale

---

#### 2.1.2 Perte de Reconstruction Améliorée

**Fichier concerné** : `training_loop.py`

**Localisation** : Fonction `loss_reconstruction()` ligne 24

**Problème** : MSE simple ne capture pas les similarités directionnelles.

**Solution proposée** :
```python
def loss_reconstruction(
    pred: Optional[Tensor], 
    target: Optional[Tensor],
    *,
    use_cosine_similarity: bool = False,
    alpha: float = 0.5,  # Poids pour combinaison MSE + Cosine
    use_perceptual: bool = False,  # Pour future extension
) -> Tensor:
    """
    Perte de reconstruction améliorée avec similarité cosinus optionnelle.
    
    La similarité cosinus capture la direction du vecteur (relations entre
    variables) même si l'amplitude diffère.
    """
    if pred is None or target is None:
        return torch.zeros(
            (), 
            device=pred.device if pred is not None else target.device,
            requires_grad=True
        )
    
    # Loss MSE de base
    mse_loss = nn.functional.mse_loss(pred, target)
    
    if use_cosine_similarity:
        # Calculer la similarité cosinus
        # Normaliser les tenseurs
        pred_flat = pred.flatten(start_dim=1)  # [B, ...]
        target_flat = target.flatten(start_dim=1)
        
        pred_norm = pred_flat / (pred_flat.norm(dim=1, keepdim=True) + 1e-8)
        target_norm = target_flat / (target_flat.norm(dim=1, keepdim=True) + 1e-8)
        
        # Similarité cosinus (produit scalaire des vecteurs normalisés)
        cosine_sim = (pred_norm * target_norm).sum(dim=1).mean()
        
        # Convertir similarité ([-1, 1]) en perte ([0, 2])
        cosine_loss = 1 - cosine_sim
        
        # Combinaison pondérée
        return alpha * mse_loss + (1 - alpha) * cosine_loss
    
    return mse_loss
```

**Impact attendu** : +5-10% d'amélioration sur la reconstruction des patterns

---

#### 2.1.3 Perte DAG Stabilisée

**Fichier concerné** : `training_loop.py`

**Localisation** : Fonction `loss_no_tears()` ligne 40

**Problème** : L'exponentielle de matrice peut exploser numériquement et pas de régularisation de sparsité.

**Solution proposée** :
```python
def loss_no_tears(
    A_masked: Tensor,
    *,
    lambda_reg: float = 1e-3,  # Régularisation L1 pour sparsité
    clamp_value: float = 10.0,  # Pour éviter les explosions
    use_approx: bool = False,  # Utiliser approximation si nécessaire
) -> Tensor:
    """
    Version améliorée avec régularisation et stabilisation numérique.
    
    - Clamp pour éviter les explosions numériques
    - Régularisation L1 pour favoriser la sparsité du DAG
    - Approximation de fallback si l'exponentielle échoue
    """
    # Clamper pour éviter les valeurs trop grandes
    A_clamped = torch.clamp(A_masked, -clamp_value, clamp_value)
    A_squared = torch.mul(A_clamped, A_clamped)
    
    # Calculer la trace de l'exponentielle
    try:
        matrix_exp = torch.matrix_exp(A_squared)
        trace = torch.trace(matrix_exp)
        acyclicity_loss = trace - A_masked.shape[0]
    except RuntimeError as e:
        if "exp" in str(e).lower() or use_approx:
            # Fallback: Approximation avec série tronquée
            # exp(A) ≈ I + A + A²/2 + A³/6
            identity = torch.eye(
                A_squared.shape[0], 
                device=A_squared.device, 
                dtype=A_squared.dtype
            )
            
            # Approximation tronquée à l'ordre 2 (suffisant pour petites valeurs)
            A_power = A_squared
            trace_approx = torch.trace(
                identity + A_squared + 0.5 * A_power
            )
            acyclicity_loss = trace_approx - A_masked.shape[0]
        else:
            raise e
    
    # Régularisation L1 pour favoriser la sparsité
    # Cela encourage le DAG à avoir peu d'arêtes (structure causale simple)
    sparsity_loss = lambda_reg * torch.abs(A_clamped).sum()
    
    return acyclicity_loss + sparsity_loss
```

**Impact attendu** : Stabilité numérique améliorée, DAG plus sparses et interprétables

---

### 2.2 Métriques d'Évaluation Enrichies

**Fichier concerné** : `evaluation_xai.py`

#### 2.2.1 F1 Score pour Événements Extrêmes

**Problème** : Pas de métrique F1 pour évaluer la détection des extrêmes climatiques.

**Solution proposée** :
```python
def compute_extreme_events_f1(
    pred: Tensor, 
    target: Tensor, 
    threshold_percentile: float = 95.0,
    threshold_absolute: Optional[float] = None,
    window_size: int = 1,
    return_detailed: bool = False,
) -> Dict[str, float]:
    """
    Calcule le F1 score pour les événements extrêmes climatiques.
    
    Un événement extrême est défini comme une valeur > seuil (percentile ou absolu).
    Cette métrique est cruciale pour les applications climatiques où les extrêmes
    (sécheresses, inondations, vagues de chaleur) sont les plus importants.
    
    Parameters:
    -----------
    pred : Tensor
        Prédictions [B, C, H, W] ou [C, H, W]
    target : Tensor
        Vérité terrain [B, C, H, W] ou [C, H, W]
    threshold_percentile : float
        Percentile pour définir les extrêmes (ex: 95.0 pour top 5%)
    threshold_absolute : Optional[float]
        Seuil absolu (prioritaire sur percentile si fourni)
    window_size : int
        Taille de fenêtre spatiale pour le matching (pour gérer décalages mineurs)
    return_detailed : bool
        Si True, retourne aussi TP, FP, FN, TN
    
    Returns:
    --------
    Dict avec 'precision', 'recall', 'f1', 'threshold', et optionnellement
    'tp', 'fp', 'fn', 'tn'
    """
    # Aplatir pour le calcul
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    
    # Calculer le seuil
    if threshold_absolute is not None:
        threshold = threshold_absolute
    else:
        threshold = torch.quantile(
            target_flat, 
            threshold_percentile / 100.0
        )
    
    # Identifier les événements extrêmes
    pred_extreme = (pred_flat > threshold).float()
    target_extreme = (target_flat > threshold).float()
    
    # Calculer TP, FP, FN, TN
    tp = (pred_extreme * target_extreme).sum().item()
    fp = (pred_extreme * (1 - target_extreme)).sum().item()
    fn = ((1 - pred_extreme) * target_extreme).sum().item()
    tn = ((1 - pred_extreme) * (1 - target_extreme)).sum().item()
    
    # Métriques
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    
    # Accuracy globale
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    
    result = {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy,
        'threshold': threshold.item(),
    }
    
    if return_detailed:
        result.update({
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'tn': tn,
        })
    
    return result
```

**Impact attendu** : Métrique cruciale pour évaluer les performances sur les événements critiques

---

#### 2.2.2 Métriques avec Correction de Biais

**Solution proposée** :
```python
def compute_bias_corrected_metrics(
    pred: Tensor,
    target: Tensor,
    *,
    use_bias_correction: bool = True,
    return_uncorrected: bool = True,
) -> Dict[str, float]:
    """
    Calcule des métriques avec correction de biais.
    
    Utile pour identifier si le modèle sous/surestime systématiquement,
    et pour évaluer les performances après correction de biais.
    
    Parameters:
    -----------
    pred : Tensor
        Prédictions
    target : Tensor
        Vérité terrain
    use_bias_correction : bool
        Si True, calcule aussi les métriques corrigées
    return_uncorrected : bool
        Si True, retourne aussi les métriques non corrigées
    
    Returns:
    --------
    Dict avec métriques de base et optionnellement corrigées
    """
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    
    # Statistiques de base
    mse = torch.mean((pred_flat - target_flat) ** 2).item()
    mae = torch.mean(torch.abs(pred_flat - target_flat)).item()
    rmse = torch.sqrt(torch.mean((pred_flat - target_flat) ** 2)).item()
    
    # Biais (erreur systématique)
    bias = torch.mean(pred_flat - target_flat).item()
    
    # Variance des erreurs
    error_var = torch.var(pred_flat - target_flat).item()
    
    # Skill score (R²)
    target_var = torch.var(target_flat).item()
    target_mean = torch.mean(target_flat).item()
    
    # R² standard
    ss_res = torch.sum((target_flat - pred_flat) ** 2).item()
    ss_tot = torch.sum((target_flat - target_mean) ** 2).item()
    r2 = 1 - (ss_res / (ss_tot + 1e-8))
    
    result = {}
    
    if return_uncorrected:
        result.update({
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'bias': bias,
            'error_variance': error_var,
            'r2': r2,
            'target_mean': target_mean,
            'pred_mean': torch.mean(pred_flat).item(),
        })
    
    if use_bias_correction:
        # Correction de biais: soustraire le biais moyen
        pred_corrected = pred_flat - bias
        
        mse_corrected = torch.mean((pred_corrected - target_flat) ** 2).item()
        mae_corrected = torch.mean(torch.abs(pred_corrected - target_flat)).item()
        rmse_corrected = torch.sqrt(
            torch.mean((pred_corrected - target_flat) ** 2)
        ).item()
        
        # R² après correction
        ss_res_corrected = torch.sum((target_flat - pred_corrected) ** 2).item()
        r2_corrected = 1 - (ss_res_corrected / (ss_tot + 1e-8))
        
        result.update({
            'mse_corrected': mse_corrected,
            'mae_corrected': mae_corrected,
            'rmse_corrected': rmse_corrected,
            'r2_corrected': r2_corrected,
            'bias_removed': bias,
        })
    
    return result
```

**Impact attendu** : Meilleure compréhension des biais systématiques du modèle

---

#### 2.2.3 Corrélation Spatiale

**Solution proposée** :
```python
def compute_spatial_correlation(
    pred: Tensor,
    target: Tensor,
    *,
    aggregate_regions: bool = False,
    region_mask: Optional[Tensor] = None,
) -> Dict[str, float]:
    """
    Calcule la corrélation spatiale pour évaluer la cohérence spatiale.
    
    Important pour les données climatiques où la structure spatiale
    (gradients, patterns) est cruciale.
    
    Parameters:
    -----------
    pred : Tensor
        Prédictions [B, C, H, W] ou [C, H, W]
    target : Tensor
        Vérité terrain [B, C, H, W] ou [C, H, W]
    aggregate_regions : bool
        Si True, calcule la corrélation par région
    region_mask : Optional[Tensor]
        Masque pour définir les régions (valeurs entières)
    
    Returns:
    --------
    Dict avec métriques de corrélation spatiale
    """
    # Aplatir spatialement
    if pred.dim() == 4:  # [B, C, H, W]
        pred = pred.mean(dim=0)  # Moyenne sur batch
        target = target.mean(dim=0)
    
    if pred.dim() == 3:  # [C, H, W]
        pred = pred.mean(dim=0)  # Moyenne sur canaux
        target = target.mean(dim=0)
    
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    
    # Corrélation de Pearson
    pred_mean = pred_flat.mean()
    target_mean = target_flat.mean()
    
    numerator = ((pred_flat - pred_mean) * (target_flat - target_mean)).sum()
    pred_std = torch.std(pred_flat)
    target_std = torch.std(target_flat)
    denominator = pred_std * target_std * len(pred_flat)
    
    correlation = (numerator / (denominator + 1e-8)).item()
    
    # Corrélation de Spearman (rang)
    pred_ranks = torch.argsort(torch.argsort(pred_flat))
    target_ranks = torch.argsort(torch.argsort(target_flat))
    
    pred_ranks_mean = pred_ranks.float().mean()
    target_ranks_mean = target_ranks.float().mean()
    
    numerator_rank = ((pred_ranks.float() - pred_ranks_mean) * 
                      (target_ranks.float() - target_ranks_mean)).sum()
    pred_ranks_std = torch.std(pred_ranks.float())
    target_ranks_std = torch.std(target_ranks.float())
    denominator_rank = pred_ranks_std * target_ranks_std * len(pred_flat)
    
    spearman_correlation = (numerator_rank / (denominator_rank + 1e-8)).item()
    
    result = {
        'pearson_correlation': correlation,
        'spearman_correlation': spearman_correlation,
        'pred_mean': pred_mean.item(),
        'target_mean': target_mean.item(),
        'pred_std': pred_std.item(),
        'target_std': target_std.item(),
    }
    
    # Corrélation par région si demandée
    if aggregate_regions and region_mask is not None:
        region_correlations = {}
        unique_regions = torch.unique(region_mask)
        
        for region_id in unique_regions:
            region_mask_bool = (region_mask == region_id).flatten()
            if region_mask_bool.sum() > 10:  # Au moins 10 pixels
                pred_region = pred_flat[region_mask_bool]
                target_region = target_flat[region_mask_bool]
                
                # Corrélation pour cette région
                pred_region_mean = pred_region.mean()
                target_region_mean = target_region.mean()
                
                numerator_region = ((pred_region - pred_region_mean) * 
                                   (target_region - target_region_mean)).sum()
                pred_region_std = torch.std(pred_region)
                target_region_std = torch.std(target_region)
                denominator_region = pred_region_std * target_region_std * len(pred_region)
                
                corr_region = (numerator_region / (denominator_region + 1e-8)).item()
                region_correlations[f'region_{region_id.item()}'] = corr_region
        
        result['region_correlations'] = region_correlations
    
    return result
```

**Impact attendu** : Meilleure évaluation de la cohérence spatiale des prédictions

---

#### 2.2.4 Mise à Jour de MetricReport

**Solution proposée** :
```python
@dataclass
class EnhancedMetricReport:
    """
    Rapport de métriques enrichi avec nouvelles métriques.
    """
    # Métriques existantes
    mse: float
    mae: float
    hist_distance: float
    crps: float
    spectrum_distance: float
    baseline_mse: Optional[float] = None
    baseline_mae: Optional[float] = None
    
    # Nouvelles métriques
    f1_extreme: Optional[float] = None
    precision_extreme: Optional[float] = None
    recall_extreme: Optional[float] = None
    bias: Optional[float] = None
    r2: Optional[float] = None
    r2_corrected: Optional[float] = None
    pearson_correlation: Optional[float] = None
    spearman_correlation: Optional[float] = None
    
    def to_dict(self) -> Dict[str, float]:
        """Convertit en dictionnaire pour sauvegarde."""
        return {
            k: v for k, v in self.__dict__.items() 
            if v is not None
        }
```

---

### 2.3 Optimisation de l'Entraînement

#### 2.3.1 Learning Rate Scheduling Adaptatif

**Fichier concerné** : `training_loop.py` ou script d'entraînement

**Solution proposée** :
```python
def create_adaptive_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    mode: str = "plateau",  # "plateau", "cosine", "step", "warmup_cosine"
    factor: float = 0.5,
    patience: int = 5,
    min_lr: float = 1e-6,
    warmup_epochs: int = 5,  # Pour warmup_cosine
    total_epochs: int = 100,  # Pour cosine
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """
    Crée un scheduler adaptatif basé sur la loss ou un schedule fixe.
    
    Modes disponibles:
    - "plateau": Réduit LR quand la loss stagne
    - "cosine": Schedule cosinus pour convergence douce
    - "step": Réduction par paliers
    - "warmup_cosine": Warmup puis cosine (recommandé)
    """
    if mode == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=factor, 
            patience=patience, 
            min_lr=min_lr,
            verbose=True,
        )
    elif mode == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=total_epochs, 
            eta_min=min_lr
        )
    elif mode == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=30, 
            gamma=factor
        )
    elif mode == "warmup_cosine":
        # Warmup puis cosine
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            else:
                progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
                return 0.5 * (1 + np.cos(np.pi * progress))
        
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        return None
```

**Utilisation dans train_epoch** :
```python
def train_epoch(
    ...,
    lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    ...
) -> Dict[str, float]:
    # ... code existant ...
    
    # À la fin de l'epoch, mettre à jour le scheduler
    if lr_scheduler is not None:
        if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            lr_scheduler.step(avg_loss)  # ReduceLROnPlateau nécessite la métrique
        else:
            lr_scheduler.step()  # Autres schedulers
    
    # ... retour ...
```

**Impact attendu** : +5-15% d'amélioration de la convergence

---

#### 2.3.2 Early Stopping Intelligent

**Solution proposée** :
```python
class EarlyStopping:
    """
    Early stopping basé sur les métriques de validation.
    
    Surveille une métrique (loss, F1, R², etc.) et arrête l'entraînement
    si aucune amélioration pendant 'patience' epochs.
    """
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 1e-6,
        metric: str = "loss",  # "loss", "f1_extreme", "r2", etc.
        mode: str = "min",  # "min" ou "max"
        restore_best_weights: bool = True,
        verbose: bool = True,
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.metric = metric
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose
        
        self.counter = 0
        self.best_score = None
        self.best_model_state = None
        self.best_epoch = None
        self.early_stop = False
    
    def __call__(
        self, 
        score: float, 
        model_state: dict,
        epoch: int,
    ) -> bool:
        """
        Vérifie si on doit arrêter l'entraînement.
        
        Parameters:
        -----------
        score : float
            Score de la métrique surveillée
        model_state : dict
            État du modèle (pour restauration)
        epoch : int
            Époque actuelle
        
        Returns:
        --------
        bool: True si on doit arrêter
        """
        if self.best_score is None:
            self.best_score = score
            self.best_model_state = model_state.copy()
            self.best_epoch = epoch
        elif self._is_better(score, self.best_score):
            self.best_score = score
            self.best_model_state = model_state.copy()
            self.best_epoch = epoch
            self.counter = 0
            if self.verbose:
                print(f"✓ Meilleur {self.metric}: {score:.6f} à l'epoch {epoch}")
        else:
            self.counter += 1
            if self.verbose:
                print(f"  Pas d'amélioration ({self.counter}/{self.patience})")
            
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"\n⚠️  Early stopping déclenché!")
                    print(f"   Meilleur {self.metric}: {self.best_score:.6f} à l'epoch {self.best_epoch}")
        
        return self.early_stop
    
    def _is_better(self, current: float, best: float) -> bool:
        """Détermine si le score actuel est meilleur."""
        if self.mode == "min":
            return current < (best - self.min_delta)
        else:
            return current > (best + self.min_delta)
    
    def get_best_state(self) -> Optional[dict]:
        """Retourne l'état du meilleur modèle."""
        return self.best_model_state
    
    def reset(self):
        """Réinitialise l'early stopping."""
        self.counter = 0
        self.best_score = None
        self.best_model_state = None
        self.best_epoch = None
        self.early_stop = False
```

**Impact attendu** : Évite l'overfitting, économise du temps d'entraînement

---

#### 2.3.3 Pondération Adaptative des Losses

**Solution proposée** :
```python
def compute_adaptive_loss_weights(
    loss_gen: float,
    loss_rec: float,
    loss_dag: float,
    *,
    initial_lambda: float = 1.0,
    initial_beta: float = 0.1,
    initial_gamma: float = 0.1,
    adaptation_rate: float = 0.01,
    min_weight: float = 0.01,
    max_weight: float = 10.0,
) -> Tuple[float, float, float]:
    """
    Ajuste dynamiquement les poids des loss pour équilibrer l'entraînement.
    
    Si une loss domine trop, on réduit son poids pour permettre aux autres
    de contribuer davantage.
    
    Parameters:
    -----------
    loss_gen, loss_rec, loss_dag : float
        Valeurs des losses
    initial_lambda, initial_beta, initial_gamma : float
        Poids initiaux
    adaptation_rate : float
        Taux d'ajustement (0.01 = 1% par étape)
    min_weight, max_weight : float
        Bornes pour les poids
    
    Returns:
    --------
    Tuple de (lambda_gen, beta_rec, gamma_dag) ajustés
    """
    # Normaliser les losses pour qu'elles soient comparables
    total = loss_gen + loss_rec + loss_dag + 1e-8
    
    gen_ratio = loss_gen / total
    rec_ratio = loss_rec / total
    dag_ratio = loss_dag / total
    
    # Si une loss domine trop (> 80%), réduire son poids
    max_ratio = max(gen_ratio, rec_ratio, dag_ratio)
    
    if max_ratio > 0.8:
        if gen_ratio > 0.8:
            # Génération domine, réduire son poids
            lambda_gen = max(min_weight, initial_lambda * (1 - adaptation_rate))
            beta_rec = min(max_weight, initial_beta * (1 + adaptation_rate))
            gamma_dag = min(max_weight, initial_gamma * (1 + adaptation_rate))
        elif rec_ratio > 0.8:
            # Reconstruction domine
            lambda_gen = min(max_weight, initial_lambda * (1 + adaptation_rate))
            beta_rec = max(min_weight, initial_beta * (1 - adaptation_rate))
            gamma_dag = min(max_weight, initial_gamma * (1 + adaptation_rate))
        else:
            # DAG domine
            lambda_gen = min(max_weight, initial_lambda * (1 + adaptation_rate))
            beta_rec = min(max_weight, initial_beta * (1 + adaptation_rate))
            gamma_dag = max(min_weight, initial_gamma * (1 - adaptation_rate))
    else:
        # Aucune loss ne domine, garder les poids initiaux
        lambda_gen = initial_lambda
        beta_rec = initial_beta
        gamma_dag = initial_gamma
    
    return lambda_gen, beta_rec, gamma_dag
```

**Impact attendu** : Équilibrage automatique des losses, meilleure convergence

---

### 2.4 Régularisation Avancée

#### 2.4.1 Dropout Adaptatif

**Fichier concerné** : `causal_rcn.py`

**Solution proposée** :
```python
class RCNCell(nn.Module):
    def __init__(
        self,
        ...,
        dropout: float = 0.0,
        dropout_schedule: Optional[Callable[[int], float]] = None,
    ):
        # ... code existant ...
        self.dropout_rate = dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.dropout_schedule = dropout_schedule
        self.current_epoch = 0
    
    def update_dropout(self, epoch: int):
        """
        Met à jour le taux de dropout selon le schedule.
        
        Exemple de schedule:
        - Linear: lambda e: max(0.0, min(0.5, 0.1 + e * 0.01))
        - Cosine: lambda e: 0.5 * (1 + np.cos(np.pi * e / max_epochs))
        """
        if self.dropout_schedule is not None:
            new_dropout = self.dropout_schedule(epoch)
            new_dropout = max(0.0, min(0.9, new_dropout))  # Clamp entre 0 et 0.9
            self.dropout_rate = new_dropout
            self.dropout = nn.Dropout(new_dropout)
            self.current_epoch = epoch
```

**Impact attendu** : Meilleure régularisation, réduction de l'overfitting

---

#### 2.4.2 Weight Decay Adaptatif

**Solution proposée** :
```python
def create_optimizer_with_adaptive_weight_decay(
    model: nn.Module,
    base_lr: float = 1e-4,
    base_weight_decay: float = 1e-5,
    weight_decay_groups: Optional[Dict[str, float]] = None,
) -> torch.optim.Optimizer:
    """
    Crée un optimiseur avec weight decay différencié par groupe.
    
    Permet d'appliquer plus de régularisation aux couches spécifiques
    (ex: DAG matrix, decodeurs) et moins aux couches critiques (ex: encodeur).
    
    Parameters:
    -----------
    weight_decay_groups : Dict[str, float]
        Mapping nom de paramètre → weight decay
        Ex: {"A_dag": 1e-4, "decoder": 1e-5, "default": 1e-6}
    """
    if weight_decay_groups is None:
        weight_decay_groups = {
            "A_dag": 1e-4,  # Plus de régularisation pour le DAG
            "decoder": 1e-5,
            "default": 1e-6,
        }
    
    param_groups = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        # Trouver le weight decay approprié
        wd = base_weight_decay
        for key, wd_value in weight_decay_groups.items():
            if key in name:
                wd = wd_value
                break
        else:
            wd = weight_decay_groups.get("default", base_weight_decay)
        
        param_groups.append({
            'params': [param],
            'lr': base_lr,
            'weight_decay': wd,
            'name': name,
        })
    
    return torch.optim.AdamW(param_groups, lr=base_lr)
```

**Impact attendu** : Régularisation ciblée, meilleure généralisation

---

#### 2.4.3 Gradient Noise Injection

**Solution proposée** :
```python
def add_gradient_noise(
    model: nn.Module,
    noise_scale: float = 0.01,
    decay: float = 0.55,
    current_epoch: int = 0,
):
    """
    Ajoute du bruit aux gradients pour améliorer la généralisation.
    
    Technique inspirée de "Adding Gradient Noise Improves Learning for Very Deep Networks"
    Le bruit décroît avec le temps: noise = noise_scale / (1 + epoch)^decay
    """
    if noise_scale <= 0:
        return
    
    effective_noise = noise_scale / ((1 + current_epoch) ** decay)
    
    for param in model.parameters():
        if param.grad is not None:
            noise = torch.randn_like(param.grad) * effective_noise
            param.grad.add_(noise)
```

**Utilisation dans train_epoch** :
```python
# Après backward(), avant optimizer.step()
if use_gradient_noise:
    add_gradient_noise(
        combined_model,  # Tous les modules
        noise_scale=0.01,
        current_epoch=current_epoch,
    )
```

**Impact attendu** : +2-5% d'amélioration de la généralisation

---

## Résumé et Impact Attendu

### Résumé des Optimisations

#### Optimisations de Performance

| Catégorie | Optimisation | Gain Attendu | Priorité |
|-----------|--------------|--------------|----------|
| Pipeline de données | Pré-allocation + pin_memory | 10-20% | Haute |
| Pipeline de données | Cache transformations | 5-10% | Moyenne |
| Boucle d'entraînement | Éviter conversions inutiles | 5-10% | Haute |
| Boucle d'entraînement | Optimiser drivers | 2-5% | Basse |
| RCN Cell | Vectorisation GRU (JIT) | 10-15% | Moyenne |
| RCN Cell | Cache masque diagonal | 2-5% | Basse |
| Graph Builder | Éviter clonage complet | 5-10% | Moyenne |
| Diffusion Decoder | Gradient checkpointing | 40-50% mémoire | Haute |
| Diffusion Decoder | Cache masque NaN | 1-2% | Basse |
| Encodeur | Pooling vectorisé | 10-20% | Moyenne |
| **Globales** | **torch.compile** | **10-30%** | **Haute** |
| **Globales** | **Mixed Precision** | **40-50%** | **Haute** |
| **Globales** | **DataLoader workers** | **20-40%** | **Haute** |

**Gain total estimé (performance)** : **2-4x d'accélération** en combinant toutes les optimisations

---

#### Optimisations d'Accuracy/Loss/Métriques

| Catégorie | Optimisation | Impact Attendu | Priorité |
|-----------|--------------|----------------|----------|
| **Loss Diffusion** | **Focal Loss** | **+5-10% accuracy pixels difficiles** | **Moyenne** |
| **Loss Diffusion** | **Pondération spatiale** | **+10-15% régions importantes** | **Moyenne** |
| **Loss Diffusion** | **Focus extrêmes** | **+15-25% événements extrêmes** | **Haute** |
| **Loss Diffusion** | **MSE+L1 combinée** | **+5-10% robustesse** | **Basse** |
| **Loss Reconstruction** | **Similarité cosinus** | **+5-10% patterns** | **Moyenne** |
| **Loss DAG** | **Stabilisation numérique** | **Stabilité améliorée** | **Haute** |
| **Loss DAG** | **Régularisation L1** | **DAG plus sparses** | **Moyenne** |
| **Métriques** | **F1 score extrêmes** | **Métrique cruciale** | **Haute** |
| **Métriques** | **Correction biais** | **Meilleure compréhension** | **Haute** |
| **Métriques** | **Corrélation spatiale** | **Évaluation cohérence** | **Moyenne** |
| **Entraînement** | **LR scheduling adaptatif** | **+5-15% convergence** | **Haute** |
| **Entraînement** | **Early stopping** | **Évite overfitting** | **Haute** |
| **Entraînement** | **Pondération adaptative** | **Équilibrage automatique** | **Moyenne** |
| **Régularisation** | **Dropout adaptatif** | **Réduction overfitting** | **Moyenne** |
| **Régularisation** | **Weight decay adaptatif** | **Meilleure généralisation** | **Basse** |
| **Régularisation** | **Gradient noise** | **+2-5% généralisation** | **Basse** |

**Impact total estimé (accuracy)** : 
- **Accuracy globale** : +10-20%
- **Loss finale** : -15-25%
- **F1 score extrêmes** : +20-30%

---

### Plan d'Implémentation Recommandé

#### Phase 1 : Optimisations Critiques (Semaine 1-2)

1. **Mixed Precision Training** (Impact élevé, effort faible)
   - Ajouter `autocast` et `GradScaler` dans `training_loop.py`
   - Tester avec batch size augmenté

2. **torch.compile** (Impact élevé, effort très faible)
   - Compiler les modules principaux après création
   - Mode "reduce-overhead" pour débuter

3. **Gradient Checkpointing** (Impact mémoire élevé)
   - Ajouter dans `diffusion_decoder.py`
   - Permet d'augmenter batch size

4. **Early Stopping + LR Scheduling** (Impact convergence élevé)
   - Implémenter `EarlyStopping` class
   - Ajouter `ReduceLROnPlateau` scheduler

5. **F1 Score pour Extrêmes** (Impact évaluation élevé)
   - Ajouter dans `evaluation_xai.py`
   - Utiliser pour monitoring pendant entraînement

---

#### Phase 2 : Optimisations de Performance (Semaine 3-4)

1. **Pré-allocation données** (`data_pipeline.py`)
2. **Optimisations RCN** (JIT compilation, cache)
3. **Pooling vectorisé** (`intelligible_encoder.py`)
4. **Graph builder optimisé**

---

#### Phase 3 : Optimisations de Loss (Semaine 5-6)

1. **Focal Loss pour diffusion**
2. **Loss extrêmes pondérée**
3. **Perte reconstruction avec cosinus**
4. **Stabilisation DAG loss**

---

#### Phase 4 : Optimisations Avancées (Semaine 7+)

1. **Métriques enrichies** (biais, corrélation spatiale)
2. **Pondération adaptative losses**
3. **Régularisation avancée** (dropout adaptatif, etc.)

---

### Notes Importantes

#### Compatibilité

- **PyTorch >= 2.0** requis pour `torch.compile`
- **CUDA >= 11.0** recommandé pour mixed precision optimal
- **torch_scatter** requis pour pooling vectorisé (installable via pip)

#### Tests Recommandés

Avant d'appliquer toutes les optimisations :

1. **Baseline** : Mesurer les performances actuelles (loss, accuracy, temps)
2. **Incrémentiel** : Appliquer une optimisation à la fois
3. **Validation** : Vérifier que les résultats sont identiques (ou meilleurs)
4. **Benchmark** : Mesurer l'amélioration réelle

#### Paramètres à Ajuster

Certaines optimisations nécessitent un ajustement des hyperparamètres :

- **Mixed Precision** : Peut nécessiter d'ajuster le `GradScaler` si instabilité
- **Focal Loss** : Tester différents `focal_gamma` (1.0, 2.0, 3.0)
- **LR Scheduling** : Ajuster `patience` selon le dataset
- **Early Stopping** : Ajuster `patience` et `min_delta`

---

### Conclusion

Ces optimisations couvrent deux aspects essentiels :

1. **Performance** : Réduction du temps d'entraînement et d'inférence par un facteur 2-4x
2. **Précision** : Amélioration de l'accuracy, du loss et des métriques (F1, etc.) de 10-30%

Les optimisations sont conçues pour être **modulaires** et **incrémentielles**, permettant de les appliquer progressivement et de mesurer leur impact individuel.

**Recommandation** : Commencer par les optimisations de Phase 1 (impact élevé, effort faible) pour obtenir des gains rapides, puis procéder aux phases suivantes selon les besoins. 