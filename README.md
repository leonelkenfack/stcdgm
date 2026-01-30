# ST-CDGM: Spatio-Temporal Causal Diffusion Generative Model

**ST-CDGM** est un modèle d'intelligence artificielle avancé conçu pour le **downscaling climatique**. Il génère des champs climatiques haute résolution (HR) à partir de données basse résolution (LR) en respectant les contraintes physiques et la causalité temporelle.

## 📋 Vue d'Ensemble

**ST-CDGM** combine trois techniques avancées:
- **Graph Neural Networks** (PyTorch Geometric) pour l'encodage spatial
- **Réseaux Récurrents Causaux** (RCN) pour la dynamique temporelle
- **Modèles de Diffusion** (HuggingFace Diffusers) pour la génération haute résolution

**Cas d'usage**: Transformation de grilles climatiques de 23×26 points (LR) en grilles de 172×179 points (HR), avec un facteur d'amélioration de résolution d'environ **4-8x**.

## 🚀 Installation Rapide

### Installation Locale

```bash
# Cloner le repository
git clone <repo-url> climate_data
cd climate_data

# Installer les dépendances
pip install -r requirements.txt

# Installer le package
pip install -e .

# Vérifier l'installation
python scripts/test_installation.py
```

### Installation dans CyVerse VICE

Pour installer dans l'environnement CyVerse Discovery Environment (VICE), voir le guide complet:

📖 **[CYVERSE_VICE_SETUP.md](CYVERSE_VICE_SETUP.md)** - Guide complet d'installation pour CyVerse VICE

**Installation rapide VICE**:
```bash
# Dans le terminal Jupyter Lab de VICE
cd ~/
git clone <repo-url> climate_data
cd climate_data
pip install -r requirements.txt
pip install -e .
python scripts/test_installation.py
```

## 🎯 Quick Start

### 1. Préparation des Données

Les données doivent être au format NetCDF avec des coordonnées temporelles communes:

```bash
# Preprocessing (conversion NetCDF → Zarr pour meilleure performance)
python scripts/run_preprocessing.py \
    --lr_path data/raw/lr_data.nc \
    --hr_path data/raw/hr_data.nc \
    --output_dir data/processed \
    --format zarr
```

### 2. Entraînement

```bash
# Training avec configuration par défaut
python scripts/run_training.py \
    --config config/training_config.yaml \
    --checkpoint_dir models \
    --save_every 5

# Training avec configuration VICE (pour CyVerse)
python scripts/run_training.py \
    --config config/training_config_vice.yaml \
    --checkpoint_dir models \
    --save_every 5
```

### 3. Évaluation

```bash
# Evaluation du modèle
python scripts/run_evaluation.py \
    --lr_path data/raw/lr_data.nc \
    --hr_path data/raw/hr_data.nc \
    --checkpoint models/best_model.pt \
    --output_dir results
```

### 4. Pipeline Complet

```bash
# Exécuter le pipeline complet (preprocessing + training + evaluation)
python scripts/run_full_pipeline.py \
    --lr_path data/raw/lr_data.nc \
    --hr_path data/raw/hr_data.nc \
    --config config/training_config.yaml \
    --format zarr
```

## 📚 Documentation

- **[CYVERSE_VICE_SETUP.md](CYVERSE_VICE_SETUP.md)** - Guide d'installation et utilisation pour CyVerse VICE
- **[docs/st_cdgm_quickstart.md](docs/st_cdgm_quickstart.md)** - Guide de démarrage rapide
- **[docs/ARCHITECTURE_MODEL.md](docs/ARCHITECTURE_MODEL.md)** - Architecture détaillée du modèle
- **[docs/OPTIMISATION.md](docs/OPTIMISATION.md)** - Guide d'optimisation et de performance
- **[ANALYSE_PROJET_COMPLETE.md](ANALYSE_PROJET_COMPLETE.md)** - Analyse complète du projet

## 🛠️ Configuration

### Configuration Locale

La configuration par défaut se trouve dans `config/training_config.yaml`:

```yaml
data:
  lr_path: "data/raw/predictor_ACCESS-CM2_hist.nc"
  hr_path: "data/raw/pr_ACCESS-CM2_hist.nc"
  seq_len: 6
  stride: 1

training:
  device: "cuda"  # ou "cpu"
  epochs: 100
  lr: 0.0001
```

### Configuration CyVerse VICE

Pour CyVerse VICE, utilisez `config/training_config_vice.yaml` qui inclut:
- Chemins adaptés pour Data Store
- Configuration GPU/CPU automatique
- Recommandations pour performance I/O

## 📦 Structure du Projet

```
climate_data/
├── src/st_cdgm/          # Code source principal
│   ├── data/             # Pipeline de données
│   ├── models/           # Modèles (GNN, RCN, Diffusion)
│   ├── training/         # Boucle d'entraînement
│   └── evaluation/       # Métriques d'évaluation
├── scripts/              # Scripts d'exécution
│   ├── run_training.py
│   ├── run_evaluation.py
│   ├── sync_datastore.py # Utilitaires CyVerse VICE
│   └── vice_utils.py     # Détection VICE
├── config/               # Fichiers de configuration
│   ├── training_config.yaml
│   └── training_config_vice.yaml
├── docs/                 # Documentation technique
├── tests/                # Tests unitaires
└── README.md             # Ce fichier
```

## 🔧 Dépendances Principales

- **PyTorch** (≥2.0.0) - Framework principal
- **PyTorch Geometric** (≥2.3.0) - Graph Neural Networks
- **HuggingFace Diffusers** (≥0.21.0) - Modèles de diffusion
- **xarray** (≥2023.1.0) - Manipulation NetCDF
- **Hydra** (≥1.3.0) - Gestion de configuration

Voir `requirements.txt` pour la liste complète.

## 🌐 CyVerse VICE

Pour les utilisateurs **CyVerse Discovery Environment (VICE)**:

### Utilitaires VICE

- **`scripts/vice_utils.py`** - Détection automatique de l'environnement VICE
- **`scripts/sync_datastore.py`** - Synchronisation données entre local et Data Store

### Utilisation dans VICE

```bash
# Détecter l'environnement VICE
python -c "from scripts.vice_utils import is_vice_environment; print(is_vice_environment())"

# Copier des données depuis Data Store (pour performance)
python scripts/sync_datastore.py --copy-from-datastore \
    ~/data-store/home/<username>/data/raw/*.nc \
    ~/climate_data/data/raw/

# Sauvegarder des résultats dans Data Store
python scripts/sync_datastore.py --save-to-datastore \
    ~/climate_data/models/ \
    ~/data-store/home/<username>/st-cdgm/models/
```

**Important**: Les containers VICE sont éphémères. Sauvegardez régulièrement vos résultats dans le Data Store!

📖 Voir **[CYVERSE_VICE_SETUP.md](CYVERSE_VICE_SETUP.md)** pour plus de détails.

## 🧪 Tests

```bash
# Test d'installation
python scripts/test_installation.py

# Tests unitaires (si pytest installé)
pytest tests/

# Smoke test du modèle
pytest tests/test_st_cdgm_smoke.py
```

## 📊 Métriques d'Évaluation

Le modèle supporte plusieurs métriques pour l'évaluation:

- **CRPS** (Continuous Ranked Probability Score) - Métrique probabiliste standard
- **FSS** (Fractional Skill Score) - Score de compétence fractionnel
- **Wasserstein Distance** - Distance entre distributions
- **Energy Score** - Score d'énergie pour cohérence multivariée
- **SHD** (Structural Hamming Distance) - Distance pour graphes causaux

## 🔬 Architecture

Le pipeline de traitement suit cette séquence:

```
Données NetCDF (LR) 
  ↓
Normalisation & Séquençage temporel
  ↓
Construction Graphe Hétérogène (relations spatiales/verticales)
  ↓
Encodage Intelligible (GNN) → Variables latentes interprétables
  ↓
Dynamique Causale Récurrente (RCN) → Évolution temporelle
  ↓
Décodeur de Diffusion Conditionnel → Génération HR
  ↓
Reconstruction Physique + Contraintes
  ↓
Champ HR Final (172×179)
```

## 🤝 Contribution

Les contributions sont les bienvenues! Veuillez ouvrir une issue ou une pull request pour proposer des améliorations.

## 📝 Licence

[À compléter selon votre licence]

## 🙏 Remerciements

- PyTorch Geometric pour les Graph Neural Networks
- HuggingFace Diffusers pour les modèles de diffusion
- CyVerse pour l'environnement VICE

## 📧 Support

Pour des questions ou du support:
- Ouvrir une issue sur GitHub
- Consulter la documentation dans `docs/`
- Pour CyVerse VICE: voir [CYVERSE_VICE_SETUP.md](CYVERSE_VICE_SETUP.md)

---

**Version**: 0.1.0  
**Dernière mise à jour**: 2026-01-16

