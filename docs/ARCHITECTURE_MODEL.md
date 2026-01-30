# Architecture Détaillée du Modèle AI (ST-CDGM)

Ce document décrit l'architecture technique et le flux de données du modèle **ST-CDGM** (Spatio-Temporal Conditional Deep Generative Model) utilisé dans ce projet.

Ce modèle est conçu pour le **downscaling climatique**, c'est-à-dire la génération de champs climatiques haute résolution (HR) à partir de données basse résolution (LR), en respectant les contraintes physiques et la causalité temporelle.

---

## 1. Vue d'Ensemble

Le modèle est une architecture hybride composée de trois modules principaux interconnectés :

1.  **Encodeur Intelligible (Graph Neural Network)** : Transforme les données spatiales brutes en variables latentes interprétables via des convolutions de graphes.
2.  **Réseau Causal Récurrent (RCN - Causal RNN)** : Modélise la dynamique temporelle et les relations causales entre ces variables latentes.
3.  **Décodeur de Diffusion (Conditional Diffusion Model)** : Génère les détails haute fréquence de l'image finale, conditionné par l'état prédit par le RCN.

### Schéma Conceptuel Détaillé

```mermaid
graph TD
    subgraph INPUT ["Données d'Entrée"]
        LR[Données Basse Résolution (LR)]
        Drivers[Drivers Externes (Forçage)]
        Noise[Bruit Gaussien Aléatoire]
    end

    subgraph ENCODER ["Module 1: Encodeur Intelligible (GNN)"]
        Graph[Construction Graphe Hétérogène]
        GNN[Convolutions Hétérogènes (SAGEConv)]
        Pool[Pooling par Méta-Chemin]
        
        LR --> Graph
        Graph --> GNN
        GNN --> Pool
    end

    subgraph RCN ["Module 2: Réseau Causal Récurrent"]
        DAG[Matrice Causale A_dag (Apprise)]
        SCM[MLP Structurels]
        GRU[Cellules GRU Temporelles]
        Hidden[État Caché H(t)]

        Pool --> Hidden
        Drivers --> GRU
        DAG -.-> SCM
        Hidden --> SCM
        SCM --> GRU
        GRU --> Hidden
    end

    subgraph DIFFUSION ["Module 3: Décodeur de Diffusion"]
        Cond[Projection Conditionnelle]
        UNet[UNet 2D Conditionnel]
        CrossAttn[Cross-Attention Layers]
        
        Hidden --> Cond
        Cond --> CrossAttn
        Noise --> UNet
        CrossAttn -.-> UNet
    end

    subgraph OUTPUT ["Sortie & Reconstruction"]
        Residual[Résidu HR Prédit]
        Baseline[Baseline Interpolée]
        Phys[Contraintes Physiques]
        Final[Champ Haute Résolution]

        UNet --> Residual
        LR -.-> Baseline
        Residual --> Phys
        Baseline --> Phys
        Phys --> Final
    end
```

---

## 2. Flux de Données (Data Pipeline)

### Étape 1 : Ingestion et Prétraitement (`NetCDFDataPipeline`)
- **Entrées** : Fichiers NetCDF contenant les variables climatiques (ex: Précipitations, Température) en Basse Résolution (LR) et Haute Résolution (HR).
- **Normalisation** : Les données sont normalisées (Min-Max ou Z-Score) pour faciliter l'entraînement.
- **Séquençage** : Les données sont découpées en fenêtres glissantes de longueur `seq_len` (ex: 6 pas de temps).
- **Baseline** : Une version interpolée (ex: bicubique) de la basse résolution est calculée pour servir de base. Le modèle apprendra uniquement à prédire le *résidu* (la différence entre la vérité terrain HR et cette baseline).

### Étape 2 : Construction du Graphe (`HeteroGraphBuilder`)
Les grilles spatiales LR sont transformées en objets `HeteroData` (PyTorch Geometric) :
- **Nœuds** : Chaque point de la grille LR devient un nœud.
- **Arêtes** : Des relations sont établies entre les nœuds pour représenter la physique :
    - `spat_adj` : Adjacence spatiale (voisins géographiques).
    - `vert_adj` : Relations verticales (si données atmosphériques multicouches).
    - `causes` : Relations causales définies par le domaine (ex: Pression -> Vent).

### Étape 3 : Encodage Intelligible (`IntelligibleVariableEncoder`)
- **Module** : `IntelligibleVariableEncoder` utilise des couches `HeteroConv` (convolutions hétérogènes, souvent `SAGEConv`).
- **Opération** : Il agrège l'information locale autour de chaque nœud en suivant des "méta-chemins" spécifiques (ex: "Surface" = GP850 -> spat_adj -> GP850).
- **Sortie** : Un ensemble de vecteurs latents (embeddings) pour chaque variable "intelligible" configurée.

### Étape 4 : Dynamique Temporelle et Causale (`RCNSequenceRunner`)
- **Module** : `RCNCell` combine un modèle causal structurel (SCM) et un réseau récurrent (GRU).
- **Matrice DAG** : Une matrice d'adjacence `A_dag` *apprenable* pondère les interactions instantanées entre les variables latentes.
- **Mise à jour** :
    1.  **Causalité Instantanée** : Les variables s'influencent mutuellement via des MLPs structurels guidés par `A_dag`.
    2.  **Mémoire Temporelle** : Des cellules GRU mettent à jour l'état de chaque variable en fonction de son passé et des *drivers* externes.
- **Sortie** : Une séquence d'états cachés `H(t)` qui capturent l'évolution du système.

### Étape 5 : Génération par Diffusion (`CausalDiffusionDecoder`)
- **Module** : `CausalDiffusionDecoder` (basé sur `UNet2DConditionModel` de HuggingFace Diffusers).
- **Conditionnement** : L'état caché `H(t)` issu du RCN est projeté et injecté dans le UNet via des couches de **Cross-Attention**.
- **Processus** :
    - Le modèle prédit le bruit ajouté à une image "propre" (le résidu).
    - Lors de l'inférence, on part d'un bruit aléatoire et on le dénuise itérativement sur $T$ étapes (ex: 1000 pas), guidé par le conditionnement causal.
- **Reconstruction Physique** :
    - Sortie du UNet = Résidu prédit.
    - Sortie Finale = Baseline + Résidu.
    - **Contraintes** : Une couche finale applique des contraintes physiques strictes (ex: `T_min <= T <= T_max`) via des fonctions ReLU pour garantir la validité physique des valeurs.

---

## 3. Détails Techniques des Composants

### A. Intelligible Encoder
L'encodeur ne se contente pas de compresser l'image. Il sépare l'information en canaux sémantiques définis par la configuration `metapaths` :
*   *Exemple* : Un canal peut représenter l'advection, un autre la convection.
*   Cela rend l'espace latent partiellement interprétable.

### B. RCN (Recurrent Causal Network)
Le cœur "intelligent" du système. Contrairement à un LSTM standard qui mélange tout dans un vecteur caché dense, le RCN maintient $N$ variables latentes distinctes et apprend explicitement comment elles interagissent (graphe causal).
*   **L_dag (Perte DAG)** : Une pénalité est appliquée pour forcer la matrice `A_dag` à être un graphe acyclique (ou proche), favorisant des relations causales claires.

### C. Diffusion Decoder
Utilise la puissance des modèles génératifs modernes pour halluciner les détails haute fréquence réalistes que l'interpolation lisse (floue).
*   En conditionnant sur le RCN, la diffusion ne génère pas n'importe quelle texture réaliste, mais une texture cohérente avec la dynamique temporelle et physique observée en basse résolution.

---

## 4. Entraînement et Fonctions de Coût

L'entraînement minimise une somme pondérée de trois pertes :

1.  **Loss Générative (`lambda_gen`)** : MSE (Mean Squared Error) sur le bruit prédit par le modèle de diffusion. C'est la perte standard des DDPM.
2.  **Loss de Reconstruction (`beta_rec`)** : MSE auxiliaire appliquée directement sur la sortie du RCN (via un petit décodeur linéaire) pour forcer l'espace latent à conserver l'information pertinente de l'entrée.
3.  **Loss DAG (`gamma_dag`)** : Contrainte de sparsité et d'acyclicité sur la matrice `A_dag` du RCN pour découvrir la structure causale.

$$ L_{total} = \lambda_{gen} L_{diff} + \beta_{rec} L_{rec} + \gamma_{dag} L_{dag} $$

---

## 5. Synthèse des Entrées/Sorties

| Type | Dimensions | Description |
| :--- | :--- | :--- |
| **Input LR** | `[Batch, Time, C, H_lr, W_lr]` | Données climatiques grossières (ex: 23x26). |
| **Input HR** | `[Batch, Time, C, H_hr, W_hr]` | Vérité terrain fine (ex: 172x179) pour l'entraînement. |
| **Latent** | `[Batch, Time, Q, Hidden]` | `Q` variables causales latentes évoluant dans le temps. |
| **Output** | `[Batch, Time, 3, H_hr, W_hr]` | Prédiction HR (Canaux : Min, Moyenne, Max ou similaire). |

---

## 6. Structures de Données Détaillées

Cette section décrit les formes exactes des tenseurs à chaque étape du pipeline, avec des exemples numériques concrets basés sur les données réelles du projet (NorESM2-MM : LR 23×26, 15 variables, 7300 timesteps).

### 6.1 Pipeline d'Entrée (`NetCDFDataPipeline`)

#### Données NetCDF Brutes
- **Format** : Fichiers NetCDF avec variables climatiques
- **Structure** : `xarray.Dataset` avec dimensions `(time, lat, lon)`
- **Exemple concret** : 
  - `time`: 7300 pas de temps (1986-01-01 à 2005-12-31)
  - `lat`: 23 points (de -59.38° à -26.38°)
  - `lon`: 26 points (de 150.6° à 188.1°)
  - Variables : 15 variables (t_850, t_500, t_250, u_850, u_500, u_250, v_850, v_500, v_250, w_850, w_500, w_250, q_850, q_500, q_250)

#### Conversion en Tenseurs PyTorch
- **Forme NetCDF** : `[time, lat, lon]` pour chaque variable
- **Forme après `_dataset_to_numpy`** : `[time, channel, lat, lon]`
  - Exemple : `[7300, 15, 23, 26]` (toutes variables combinées)
- **Forme après séquençage** : `[seq_len, channel, lat, lon]`
  - Exemple avec `seq_len=6` : `[6, 15, 23, 26]`
- **Forme dans le DataLoader** : Dictionnaire avec clés :
  - `"lr"` : `[seq_len, C_lr, H_lr, W_lr]` → `[6, 15, 23, 26]`
  - `"baseline"` : `[seq_len, C_hr, H_hr, W_hr]` → `[6, 3, 172, 179]` (exemple HR)
  - `"residual"` : `[seq_len, C_hr, H_hr, W_hr]` → `[6, 3, 172, 179]`
  - `"hr"` : `[seq_len, C_hr, H_hr, W_hr]` → `[6, 3, 172, 179]`
  - `"static"` (optionnel) : `[C_static, H_hr, W_hr]` → `[1, 172, 179]`

### 6.2 Construction du Graphe (`HeteroGraphBuilder`)

#### Structure HeteroData
- **Type de nœuds dynamiques** : `GP850`, `GP500`, `GP250` (si `include_mid_layer=True`)
- **Type de nœuds statiques** : `SP_HR` (variables statiques haute résolution)
- **Nombre de nœuds LR** : `num_nodes_lr = H_lr × W_lr`
  - Exemple : `598 = 23 × 26`
- **Nombre de nœuds HR** : `num_nodes_hr = H_hr × W_hr`
  - Exemple : `30788 = 172 × 179`

#### Features Nodales
- **Nœuds dynamiques** : Features injectées à chaque pas de temps
  - Forme : `[N_lr, C_lr]` → `[598, 15]` (exemple avec 15 canaux)
- **Nœuds statiques** : Features fixes (topographie, etc.)
  - Forme : `[N_hr, C_static]` → `[30788, 1]` (exemple avec topographie)

#### Arêtes (Edge Index)
- **Arêtes spatiales (`spat_adj`)** : Connectivité 8-voisins sur la grille LR
  - Forme : `[2, E_spatial]` où `E_spatial ≈ 8 × N_lr` (moins les bords)
  - Exemple : `[2, ~4700]` pour une grille 23×26
- **Arêtes verticales (`vert_adj`)** : Mapping identique entre couches
  - Forme : `[2, N_lr]` → `[2, 598]`
- **Arêtes statiques (`causes`)** : Mapping HR → LR parent
  - Forme : `[2, N_hr]` → `[2, 30788]`
  - Chaque nœud HR est connecté à son nœud LR parent via un facteur d'échelle entier

### 6.3 Encodeur Intelligible (`IntelligibleVariableEncoder`)

#### Transformation : Grille → Nœuds
- **Entrée** : `lr_grid` de forme `[C, H_lr, W_lr]` → `[15, 23, 26]`
- **Conversion** : `lr_grid_to_nodes()` → `[N_lr, C]` → `[598, 15]`

#### Convolution Hétérogène
- **Entrée HeteroData** : Features nodales par type de nœud
  - `GP850.x` : `[598, 15]`
  - `SP_HR.x` : `[30788, C_static]`
- **Opération** : `HeteroConv` avec `SAGEConv` par méta-chemin
- **Sortie** : Embeddings par type de nœud cible
  - Forme : `[N_target, hidden_dim]` → `[598, 128]` (exemple avec `hidden_dim=128`)

#### État Initial H(0)
- **Méthode** : `init_state()` agrège les embeddings par méta-chemin
- **Forme** : `[q, N_lr, hidden_dim]`
  - `q` : Nombre de variables intelligibles (nombre de méta-chemins configurés)
  - Exemple avec 3 méta-chemins : `[3, 598, 128]`
- **Signification** : Chaque variable intelligible a un embedding par nœud spatial

### 6.4 Réseau Causal Récurrent (RCN)

#### État Caché H(t)
- **Forme** : `[q, N_lr, hidden_dim]`
  - Exemple : `[3, 598, 128]` (3 variables intelligibles, 598 nœuds, dimension cachée 128)
- **Évolution temporelle** : `H(t)` est mis à jour à chaque pas de temps `t ∈ [0, seq_len-1]`

#### Drivers Externes
- **Forme** : `[N_lr, driver_dim]` à chaque pas de temps
  - `driver_dim` : Nombre de canaux LR (ex: 15)
  - Exemple : `[598, 15]`
- **Séquence** : Liste de `seq_len` tenseurs → `[6] × [598, 15]`

#### Transformations Internes
1. **Matrice DAG** : `A_dag` de forme `[q, q]` → `[3, 3]`
2. **Prédiction SCM** : 
   - Entrée : `H_prev` `[3, 598, 128]`
   - Opération : `torch.einsum("ik,inj->knj", A_masked, H_prev)` → `[3, 598, 128]`
   - MLP structurel par variable → `[3, 598, 128]`
3. **Mise à jour GRU** :
   - Driver encodé : `[598, 128]`
   - GRU par variable → `[3, 598, 128]`

#### Reconstruction (Optionnelle)
- **Entrée** : `H_next` ou `driver` de forme `[598, 15]`
- **Sortie** : `[N_lr, reconstruction_dim]` → `[598, 15]` (si `reconstruction_dim=15`)

#### Conditionnement pour Diffusion
- **Pooling spatial** : `H(t)` `[3, 598, 128]` → `[1, 3, 128]` (mean pooling)
- **Projection** : `[1, 3, 128]` → `[1, 3, conditioning_dim]` → `[1, 3, 128]`
- **Forme finale** : `[batch, q, conditioning_dim]` → `[1, 3, 128]`

### 6.5 Décodeur de Diffusion (`CausalDiffusionDecoder`)

#### Conditionnement
- **Entrée** : État causal projeté de forme `[batch, q, conditioning_dim]`
  - Exemple : `[1, 3, 128]`
- **Préparation** : Aplatissement pour cross-attention → `[batch, q, conditioning_dim]`

#### Processus de Diffusion (Entraînement)
- **Cible** : Résidu HR de forme `[batch, C_hr, H_hr, W_hr]`
  - Exemple : `[1, 3, 172, 179]`
- **Bruit** : `noise` `[1, 3, 172, 179]` (gaussien)
- **Timesteps** : `[batch]` → `[1]` (aléatoire entre 0 et 999)
- **Échantillon bruité** : `[1, 3, 172, 179]`
- **Prédiction du bruit** : UNet → `[1, 3, 172, 179]`

#### Processus de Diffusion (Inférence)
- **Conditionnement** : `[1, 3, 128]`
- **Échantillon initial** : Bruit gaussien `[1, 3, 172, 179]`
- **Dénoising itératif** : 1000 pas (ou moins avec scheduler)
- **Résidu final** : `[1, 3, 172, 179]`

#### Reconstruction Physique
- **Résidu** : `[1, 3, 172, 179]`
- **Baseline** : `[1, 3, 172, 179]`
- **Composite** : `baseline + residual` → `[1, 3, 172, 179]`
- **Contraintes physiques** :
  - Canal 0 : `T_min` → `[1, 1, 172, 179]`
  - Canal 1 : `Δ1` → `T = T_min + ReLU(Δ1)` → `[1, 1, 172, 179]`
  - Canal 2 : `Δ2` → `T_max = T + ReLU(Δ2)` → `[1, 1, 172, 179]`
- **Sortie finale** : `DiffusionOutput` avec `t_min`, `t_mean`, `t_max` chacun `[1, 1, 172, 179]`

---

## 7. Architecture des Couches par Module

Cette section détaille chaque couche de chaque module avec ses paramètres exacts.

### 7.1 IntelligibleVariableEncoder

#### Structure Complète
```python
IntelligibleVariableEncoder(
    configs: List[IntelligibleVariableConfig],
    hidden_dim: int = 128,
    conditioning_dim: Optional[int] = 128,
    activation: Optional[nn.Module] = ReLU(),
    use_layer_norm: bool = True,
    default_pool: str = "mean"
)
```

#### Couches Internes

1. **HeteroConv**
   - **Type** : `torch_geometric.nn.HeteroConv`
   - **Agrégation** : `aggr="sum"`
   - **Convolutions par méta-chemin** :
     - `SAGEConv(in_channels=-1, out_channels=hidden_dim)`
     - `-1` signifie auto-inférence depuis les features d'entrée
     - Exemple : `SAGEConv(in_channels=(-1, -1), out_channels=128)`

2. **Layer Normalization**
   - **Type** : `nn.LayerNorm(hidden_dim)` si `use_layer_norm=True`
   - **Paramètres** : `normalized_shape=128`, `eps=1e-5`
   - **Application** : Après chaque convolution, avant activation

3. **Activation**
   - **Type** : `nn.ReLU()` (par défaut)
   - **Application** : Après LayerNorm

4. **Projection Conditionnelle** (si `conditioning_dim ≠ hidden_dim`)
   - **Type** : `nn.Linear(hidden_dim, conditioning_dim)`
   - **Paramètres** : 
     - `in_features=128`
     - `out_features=128` (si identique, `nn.Identity()`)
   - **Application** : Lors de `project_conditioning()` ou `project_state_tensor()`

5. **Pooling Global** (par méta-chemin)
   - **Types** : `global_mean_pool()` ou `global_max_pool()`
   - **Application** : Lors de `pooled_state()` ou `project_conditioning()`

#### Exemple de Configuration
```python
configs = [
    IntelligibleVariableConfig(
        name="surface",
        meta_path=("GP850", "spat_adj", "GP850"),
        conv_class=SAGEConv,
        pool="mean"
    ),
    IntelligibleVariableConfig(
        name="vertical",
        meta_path=("GP500", "vert_adj", "GP850"),
        pool="mean"
    ),
    IntelligibleVariableConfig(
        name="static",
        meta_path=("SP_HR", "causes", "GP850"),
        pool="mean"
    )
]
```

### 7.2 RCNCell

#### Structure Complète
```python
RCNCell(
    num_vars: int,              # q (nombre de variables intelligibles)
    hidden_dim: int = 128,
    driver_dim: int,            # C_lr (nombre de canaux LR)
    reconstruction_dim: Optional[int] = None,
    activation: Optional[nn.Module] = ReLU(),
    dropout: float = 0.0
)
```

#### Couches Internes

1. **Matrice DAG Apprenable**
   - **Type** : `nn.Parameter`
   - **Forme** : `[num_vars, num_vars]` → `[3, 3]` (exemple)
   - **Initialisation** : `nn.init.xavier_uniform_()`
   - **Contrainte** : Diagonale masquée à zéro via `MaskDiagonal` (autograd function)

2. **MLPs Structurels** (un par variable)
   - **Type** : `nn.ModuleList` de `nn.Sequential`
   - **Structure** : `[Linear(hidden_dim, hidden_dim), ReLU, Linear(hidden_dim, hidden_dim)]`
   - **Nombre** : `num_vars` (ex: 3)
   - **Paramètres par MLP** :
     - `Linear(128, 128)` : poids `[128, 128]`, biais `[128]`
     - `ReLU()`
     - `Linear(128, 128)` : poids `[128, 128]`, biais `[128]`
   - **Initialisation** : Xavier uniform pour les Linear

3. **Encodeur de Driver**
   - **Type** : `nn.Sequential`
   - **Structure** : `[Linear(driver_dim, hidden_dim), ReLU]`
   - **Paramètres** :
     - `Linear(15, 128)` : poids `[128, 15]`, biais `[128]` (exemple avec 15 canaux)
   - **Initialisation** : Xavier uniform

4. **Cellules GRU** (une par variable)
   - **Type** : `nn.ModuleList` de `nn.GRUCell`
   - **Structure** : `GRUCell(input_size=hidden_dim, hidden_size=hidden_dim)`
   - **Nombre** : `num_vars` (ex: 3)
   - **Paramètres par GRU** :
     - Poids d'entrée : `[3*hidden_dim, hidden_dim]` → `[384, 128]`
     - Poids caché : `[3*hidden_dim, hidden_dim]` → `[384, 128]`
     - Biais : `[3*hidden_dim]` → `[384]`
   - **Initialisation** : Standard GRU (uniforme)

5. **Dropout** (optionnel)
   - **Type** : `nn.Dropout(dropout)` si `dropout > 0`, sinon `nn.Identity()`
   - **Application** : Après chaque GRU

6. **Décodeur de Reconstruction** (optionnel)
   - **Type** : `nn.Linear` si `reconstruction_dim is not None`
   - **Paramètres** :
     - `in_features = num_vars * hidden_dim` → `3 * 128 = 384`
     - `out_features = reconstruction_dim` → `15` (exemple)
   - **Initialisation** : Xavier uniform pour poids, zéros pour biais

#### Flux de Données dans RCNCell
1. **Entrée** : `H_prev [3, 598, 128]`, `driver [598, 15]`
2. **SCM** : `A_dag [3, 3]` × `H_prev` → MLPs → `H_hat [3, 598, 128]`
3. **Driver Encoding** : `driver [598, 15]` → `[598, 128]`
4. **GRU** : `GRU(driver_emb, H_hat[k])` pour chaque `k` → `H_next [3, 598, 128]`
5. **Reconstruction** (si activé) : `H_next` → `[598, 15]`

### 7.3 CausalDiffusionDecoder

#### Structure Complète
```python
CausalDiffusionDecoder(
    in_channels: int = 3,                    # C_hr
    conditioning_dim: int = 128,
    height: int = 172,                       # H_hr
    width: int = 179,                        # W_hr
    num_diffusion_steps: int = 1000,
    unet_kwargs: Optional[dict] = None
)
```

#### Couches Internes

1. **UNet2DConditionModel** (HuggingFace Diffusers)
   - **Type** : `diffusers.UNet2DConditionModel`
   - **Paramètres principaux** :
     - `sample_size`: `(height, width)` → `(172, 179)`
     - `in_channels`: `3` (canaux d'entrée = canaux de sortie)
     - `out_channels`: `3`
     - `cross_attention_dim`: `128` (dimension du conditionnement)
   - **Architecture interne** (par défaut Diffusers) :
     - **Encodeur** : Blocs de convolution downsampling avec attention
     - **Bottleneck** : Blocs avec self-attention et cross-attention
     - **Décodeur** : Blocs de convolution upsampling avec attention
     - **Timestep embedding** : Projection sinusoïdale + MLP
   - **Cross-Attention** : Injecte le conditionnement causal à plusieurs niveaux

2. **DDPMScheduler**
   - **Type** : `diffusers.DDPMScheduler`
   - **Paramètres** :
     - `num_train_timesteps`: `1000`
     - `beta_schedule`: `"linear"` (par défaut)
     - `beta_start`: `0.0001`
     - `beta_end`: `0.02`
   - **Fonction** : Gère le bruitage progressif et le dénoising

3. **Adaptateur de Conditionnement** (optionnel)
   - **Type** : `Callable[[Tensor], Tensor]` (peut être défini via `set_condition_adapter()`)
   - **Application** : Avant l'injection dans le UNet
   - **Par défaut** : `None` (pas de transformation)

4. **Contraintes Physiques** (méthode statique)
   - **Fonction** : `apply_physical_constraints(raw_output)`
   - **Entrée** : `[batch, 3, H, W]` (canaux : `T_min`, `Δ1`, `Δ2`)
   - **Opérations** :
     - `T = T_min + ReLU(Δ1)`
     - `T_max = T + ReLU(Δ2)`
   - **Sortie** : Tuple `(T_min, T, T_max)` chacun `[batch, 1, H, W]`

#### Flux de Données dans CausalDiffusionDecoder

**Entraînement** (`compute_loss`):
1. Cible : `target [1, 3, 172, 179]`
2. Bruit : `noise [1, 3, 172, 179]` (gaussien)
3. Timestep : `t [1]` (aléatoire 0-999)
4. Échantillon bruité : `scheduler.add_noise(target, noise, t)` → `[1, 3, 172, 179]`
5. Conditionnement : `[1, 3, 128]` → préparé pour cross-attention
6. Prédiction : `UNet(noisy_sample, t, conditioning)` → `[1, 3, 172, 179]`
7. Perte : `MSE(predicted_noise, noise)`

**Inférence** (`sample`):
1. Conditionnement : `[1, 3, 128]`
2. Échantillon initial : `[1, 3, 172, 179]` (bruit gaussien)
3. Boucle de dénoising (1000 pas) :
   - Prédiction : `UNet(sample, t, conditioning)`
   - Mise à jour : `scheduler.step(prediction, t, sample)`
4. Résidu final : `[1, 3, 172, 179]`
5. Reconstruction : `baseline + residual` → `[1, 3, 172, 179]`
6. Contraintes : `apply_physical_constraints()` → `(T_min, T, T_max)`

---

## 8. Tableaux Récapitulatifs

### 8.1 Tableau des Transformations de Formes

| Étape | Module | Entrée | Sortie | Exemple Numérique |
|:------|:-------|:-------|:-------|:------------------|
| **1. Chargement NetCDF** | `NetCDFDataPipeline` | Fichier `.nc` | `xarray.Dataset` | `(7300, 23, 26)` par variable |
| **2. Conversion NumPy** | `_dataset_to_numpy` | `Dataset` | `np.ndarray` | `[7300, 15, 23, 26]` |
| **3. Séquençage** | `ResDiffIterableDataset` | `[T, C, H, W]` | `[seq_len, C, H, W]` | `[6, 15, 23, 26]` |
| **4. Conversion Graphe** | `lr_grid_to_nodes` | `[C, H_lr, W_lr]` | `[N_lr, C]` | `[15, 23, 26]` → `[598, 15]` |
| **5. Encodage GNN** | `IntelligibleVariableEncoder` | `HeteroData` | `[q, N_lr, hidden_dim]` | `[3, 598, 128]` |
| **6. RCN - État Initial** | `RCNCell` | `H(0) [q, N, d]` | `H(0) [q, N, d]` | `[3, 598, 128]` |
| **7. RCN - Driver** | `driver_encoder` | `[N, driver_dim]` | `[N, hidden_dim]` | `[598, 15]` → `[598, 128]` |
| **8. RCN - SCM** | `structural_mlps` | `[q, N, d]` | `[q, N, d]` | `[3, 598, 128]` → `[3, 598, 128]` |
| **9. RCN - GRU** | `gru_cells` | `[N, d]` × q | `[q, N, d]` | `[598, 128]` → `[3, 598, 128]` |
| **10. RCN - Pooling** | `pool_state` | `[q, N, d]` | `[1, q, d]` | `[3, 598, 128]` → `[1, 3, 128]` |
| **11. Diffusion - Conditionnement** | `project_conditioning` | `[1, q, d]` | `[1, q, cond_dim]` | `[1, 3, 128]` → `[1, 3, 128]` |
| **12. Diffusion - UNet** | `UNet2DConditionModel` | `[B, C, H, W]` + cond | `[B, C, H, W]` | `[1, 3, 172, 179]` → `[1, 3, 172, 179]` |
| **13. Reconstruction** | `baseline + residual` | `[B, C, H, W]` × 2 | `[B, C, H, W]` | `[1, 3, 172, 179]` |
| **14. Contraintes Physiques** | `apply_physical_constraints` | `[B, 3, H, W]` | `(T_min, T, T_max)` | `[1, 1, 172, 179]` × 3 |

### 8.2 Tableau des Hyperparamètres par Défaut

| Catégorie | Paramètre | Valeur par Défaut | Description |
|:----------|:----------|:------------------|:------------|
| **Data** | `seq_len` | `6` | Longueur de la séquence temporelle |
| | `stride` | `1` | Pas de la fenêtre glissante |
| | `baseline_strategy` | `"hr_smoothing"` | Stratégie de calcul du baseline |
| | `baseline_factor` | `4` | Facteur de lissage pour baseline |
| | `normalize` | `True` | Normalisation des données LR |
| **Graph** | `lr_shape` | `(23, 26)` | Forme de la grille basse résolution |
| | `hr_shape` | `(172, 179)` | Forme de la grille haute résolution |
| | `include_mid_layer` | `False` | Inclure les couches GP500/GP250 |
| **Encoder** | `hidden_dim` | `128` | Dimension des embeddings cachés |
| | `conditioning_dim` | `128` | Dimension du conditionnement |
| | `metapaths` | `[("GP850", "spat_adj", "GP850"), ...]` | Méta-chemins configurés |
| **RCN** | `hidden_dim` | `128` | Dimension de l'état caché |
| | `driver_dim` | `8` (ou C_lr) | Dimension du forçage externe |
| | `reconstruction_dim` | `8` (ou None) | Dimension de reconstruction |
| | `dropout` | `0.0` | Taux de dropout |
| | `detach_interval` | `None` | Intervalle de détachement du graphe |
| **Diffusion** | `in_channels` | `3` | Nombre de canaux d'entrée/sortie |
| | `conditioning_dim` | `128` | Dimension du conditionnement |
| | `height` | `172` | Hauteur de l'image HR |
| | `width` | `179` | Largeur de l'image HR |
| | `num_diffusion_steps` | `1000` | Nombre de pas de diffusion |
| **Loss** | `lambda_gen` | `1.0` | Poids de la perte générative |
| | `beta_rec` | `0.1` | Poids de la perte de reconstruction |
| | `gamma_dag` | `0.1` | Poids de la contrainte DAG |
| **Training** | `epochs` | `1` | Nombre d'époques |
| | `lr` | `1e-4` | Taux d'apprentissage |
| | `gradient_clipping` | `1.0` | Seuil de clipping des gradients |
| | `log_every` | `1` | Fréquence de logging |

### 8.3 Tableau des Couches avec Paramètres

| Module | Couche | Type | Paramètres | Forme des Poids |
|:-------|:-------|:-----|:-----------|:----------------|
| **IntelligibleVariableEncoder** | HeteroConv | `HeteroConv` | `aggr="sum"` | - |
| | SAGEConv (par méta-chemin) | `SAGEConv` | `in_channels=(-1, -1)`, `out_channels=128` | Auto-inféré |
| | LayerNorm | `LayerNorm` | `normalized_shape=128` | - |
| | Activation | `ReLU` | - | - |
| | Conditioning Projection | `Linear` | `in_features=128`, `out_features=128` | `[128, 128]` + biais `[128]` |
| **RCNCell** | Matrice DAG | `Parameter` | `[q, q]` | `[3, 3]` (exemple) |
| | Structural MLP (×q) | `Sequential` | `[Linear(128,128), ReLU, Linear(128,128)]` | `[128,128]` × 2 + biais × 2 |
| | Driver Encoder | `Sequential` | `[Linear(driver_dim, 128), ReLU]` | `[128, driver_dim]` + biais `[128]` |
| | GRU Cell (×q) | `GRUCell` | `input_size=128`, `hidden_size=128` | `[384, 128]` × 2 + biais `[384]` |
| | Dropout | `Dropout` | `p=0.0` (par défaut) | - |
| | Reconstruction Decoder | `Linear` | `in_features=q*128`, `out_features=reconstruction_dim` | `[reconstruction_dim, 384]` + biais |
| **CausalDiffusionDecoder** | UNet | `UNet2DConditionModel` | `sample_size=(172,179)`, `in_channels=3`, `out_channels=3`, `cross_attention_dim=128` | Architecture complexe (Diffusers) |
| | Scheduler | `DDPMScheduler` | `num_train_timesteps=1000` | - |
| | Condition Adapter | `Callable` (optionnel) | - | - |

**Note** : Les formes exactes des poids du UNet dépendent de l'architecture interne de Diffusers (blocs de convolution, attention, etc.) et ne sont pas détaillées ici pour des raisons de concision.

