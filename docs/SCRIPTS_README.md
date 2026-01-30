# Scripts d'Exécution ST-CDGM

Ce guide documente tous les scripts d'exécution disponibles pour ST-CDGM.

## Scripts Principaux

### `run_preprocessing.py`

Convertit les données NetCDF en format optimisé (Zarr ou WebDataset).

**Usage :**
```bash
python scripts/run_preprocessing.py \
    --lr_path data/raw/lr.nc \
    --hr_path data/raw/hr.nc \
    --format zarr \
    --output_dir data/processed \
    --seq_len 10 \
    --normalize
```

**Options principales :**
- `--lr_path` : Chemin vers fichier NetCDF basse résolution
- `--hr_path` : Chemin vers fichier NetCDF haute résolution
- `--static_path` : Chemin vers fichier statique (optionnel)
- `--format` : Format de sortie (`zarr` ou `webdataset`)
- `--output_dir` : Répertoire de sortie
- `--seq_len` : Longueur de séquence temporelle
- `--normalize` : Activer la normalisation

### `run_training.py`

Lance l'entraînement complet avec checkpointing et callbacks.

**Usage :**
```bash
python scripts/run_training.py \
    --config config/training_config.yaml \
    --checkpoint_dir models \
    --save_every 5 \
    --max_checkpoints 5
```

**Options principales :**
- `--config` : Fichier de configuration Hydra
- `--checkpoint_dir` : Répertoire pour sauvegarder les checkpoints
- `--save_every` : Sauvegarder tous les N epochs
- `--max_checkpoints` : Nombre maximum de checkpoints à conserver
- `--resume_from` : Reprendre depuis un checkpoint spécifique

**Fonctionnalités :**
- Checkpointing automatique
- Early Stopping (si activé dans config)
- LR Scheduling (si activé dans config)
- Mixed Precision Training (si activé)
- Toutes les optimisations Phase C, D, E

### `run_evaluation.py`

Évalue un modèle entraîné avec calcul de métriques.

**Usage :**
```bash
python scripts/run_evaluation.py \
    --checkpoint models/best_model.pt \
    --lr_path data/raw/lr.nc \
    --hr_path data/raw/hr.nc \
    --output_dir results/evaluation \
    --num_samples 10 \
    --scheduler_type edm
```

**Options principales :**
- `--checkpoint` : Chemin vers le checkpoint du modèle
- `--lr_path` / `--hr_path` : Données pour évaluation
- `--output_dir` : Répertoire de sortie
- `--num_samples` : Nombre d'échantillons pour les métriques
- `--scheduler_type` : Type de scheduler (`ddpm`, `edm`, `dpm_solver++`)
- `--num_inference_steps` : Nombre de pas pour l'inférence

**Métriques calculées :**
- MSE, MAE
- CRPS (Continuous Ranked Probability Score)
- FSS (Fraction Skill Score)
- Wasserstein Distance
- Energy Score
- F1 Score pour événements extrêmes
- Distance spectrale

### `run_full_pipeline.py`

Orchestre le pipeline complet : preprocessing → training → evaluation.

**Usage :**
```bash
python scripts/run_full_pipeline.py \
    --lr_path data/raw/lr.nc \
    --hr_path data/raw/hr.nc \
    --config config/training_config.yaml \
    --format zarr
```

**Options principales :**
- `--lr_path` / `--hr_path` : Données d'entrée
- `--config` : Fichier de configuration
- `--format` : Format de preprocessing (`zarr` ou `webdataset`)
- `--skip_preprocessing` : Ignorer le preprocessing
- `--skip_training` : Ignorer l'entraînement
- `--skip_evaluation` : Ignorer l'évaluation

## Scripts Utilitaires

### `save_model.py`

Sauvegarde un modèle avec métadonnées.

**Usage :**
```bash
python scripts/save_model.py \
    --encoder encoder.pt \
    --rcn rcn.pt \
    --diffusion diffusion.pt \
    --output checkpoint.pt \
    --config config.yaml
```

### `load_model.py`

Charge un checkpoint de modèle.

**Usage :**
```bash
python scripts/load_model.py models/best_model.pt --device cuda
```

## Scripts de Test

### `test_installation.py`

Vérifie l'installation et les dépendances.

**Usage :**
```bash
python scripts/test_installation.py
```

**Vérifie :**
- Version Python
- Installation PyTorch et CUDA
- Dépendances requises
- Imports des modules ST-CDGM
- Opérations GPU/CPU de base

### `test_pipeline.py`

Test end-to-end avec données synthétiques.

**Usage :**
```bash
python scripts/test_pipeline.py
```

**Teste :**
- Création de données synthétiques
- Création des modèles
- Forward pass de base

## Utilisation dans Docker

Tous les scripts peuvent être exécutés dans Docker :

```bash
# Avec docker-compose
docker-compose exec st-cdgm-training python scripts/run_preprocessing.py ...

# Ou en mode interactif
docker-compose exec st-cdgm-training /bin/bash
python scripts/run_preprocessing.py ...
```

## Utilisation sans Docker

Les scripts peuvent aussi être exécutés directement dans l'environnement Python :

```bash
# Installer les dépendances
pip install -r requirements.txt
pip install -e .

# Exécuter les scripts
python scripts/run_preprocessing.py ...
```

## Configuration

### Fichier de Configuration Hydra

Le fichier `config/training_config.yaml` contient toutes les options d'entraînement :

- **Data** : Chemins, séquence, normalisation
- **Graph** : Formes LR/HR, variables statiques
- **Encoder** : Meta-paths, dimensions
- **RCN** : Dimensions cachées, dropout
- **Diffusion** : Canaux, résolution, scheduler
- **Loss** : Poids des pertes, optimisations (Focal Loss, etc.)
- **Training** : Epochs, LR, Mixed Precision, Early Stopping
- **Checkpoint** : Sauvegarde automatique
- **Evaluation** : Métriques, visualisations

### Variables d'Environnement

Les variables dans `config/docker.env` sont utilisées par Docker Compose.

## Exemples Complets

### Workflow Standard

```bash
# 1. Preprocessing
python scripts/run_preprocessing.py \
    --lr_path data/raw/lr.nc \
    --hr_path data/raw/hr.nc \
    --format zarr \
    --output_dir data/processed

# 2. Training
python scripts/run_training.py \
    --config config/training_config.yaml \
    --checkpoint_dir models

# 3. Evaluation
python scripts/run_evaluation.py \
    --checkpoint models/best_model.pt \
    --lr_path data/raw/lr.nc \
    --hr_path data/raw/hr.nc \
    --output_dir results/evaluation
```

### Pipeline Complet en Une Commande

```bash
python scripts/run_full_pipeline.py \
    --lr_path data/raw/lr.nc \
    --hr_path data/raw/hr.nc \
    --config config/training_config.yaml \
    --format zarr
```

## Notes Importantes

1. **Chemins** : Tous les chemins sont relatifs au répertoire de travail ou absolus
2. **GPU** : Les scripts détectent automatiquement la disponibilité CUDA
3. **Checkpoints** : Les modèles sont sauvegardés automatiquement pendant l'entraînement
4. **Configuration** : Utilisez Hydra pour surcharger les paramètres via ligne de commande
5. **Logs** : Les logs sont affichés dans la console et peuvent être redirigés

## Troubleshooting

### Erreur d'Import

Vérifiez que le package est installé :
```bash
pip install -e .
```

### GPU Non Disponible

Les scripts fonctionnent en mode CPU si CUDA n'est pas disponible.

### Checkpoint Corrompu

Utilisez `--resume_from` pour reprendre depuis un checkpoint précédent.

### Mémoire Insuffisante

- Réduisez `batch_size` ou `seq_len`
- Activez gradient checkpointing
- Utilisez Mixed Precision (réduit l'utilisation mémoire)

