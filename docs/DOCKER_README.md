# Docker Compose Guide for ST-CDGM

Ce guide explique comment utiliser Docker Compose pour exécuter ST-CDGM dans un environnement containerisé.

## Prérequis

- Docker avec support GPU (nvidia-docker2) OU Docker Desktop avec GPU support
- Docker Compose (généralement inclus avec Docker)
- Fichiers de données NetCDF dans `data/raw/`

## Configuration

### 1. Fichiers Docker

- `docker-compose.yml` : Configuration principale Docker Compose
- `.dockerignore` : Fichiers à exclure de l'image Docker
- `config/docker.env` : Variables d'environnement Docker

### 2. Structure des Volumes

Les volumes suivants sont montés dans le container :

```
./data          → /workspace/data      # Données NetCDF et données traitées
./models        → /workspace/models    # Modèles sauvegardés
./results       → /workspace/results   # Résultats d'évaluation
./src           → /workspace/src       # Code source
./ops           → /workspace/ops       # Scripts d'opérations
./scripts       → /workspace/scripts   # Scripts d'exécution
./config        → /workspace/config    # Fichiers de configuration
```

## Utilisation

### Démarrer le Container

```bash
# Démarrer en arrière-plan
docker-compose up -d

# Ou démarrer et voir les logs
docker-compose up
```

### Accéder au Container

```bash
# Ouvrir un shell interactif
docker-compose exec st-cdgm-training /bin/bash

# Exécuter une commande directement
docker-compose exec st-cdgm-training python scripts/test_installation.py
```

### Exécuter le Preprocessing

```bash
docker-compose exec st-cdgm-training python scripts/run_preprocessing.py \
    --lr_path data/raw/lr.nc \
    --hr_path data/raw/hr.nc \
    --format zarr \
    --output_dir data/processed
```

### Exécuter l'Entraînement

```bash
docker-compose exec st-cdgm-training python scripts/run_training.py \
    --config config/training_config.yaml \
    --checkpoint_dir models
```

### Exécuter l'Évaluation

```bash
docker-compose exec st-cdgm-training python scripts/run_evaluation.py \
    --checkpoint models/best_model.pt \
    --lr_path data/raw/lr.nc \
    --hr_path data/raw/hr.nc \
    --output_dir results/evaluation
```

### Pipeline Complet

```bash
docker-compose exec st-cdgm-training python scripts/run_full_pipeline.py \
    --lr_path data/raw/lr.nc \
    --hr_path data/raw/hr.nc \
    --config config/training_config.yaml
```

## Configuration GPU

### Vérifier le Support GPU

```bash
docker-compose exec st-cdgm-training python -c "import torch; print(torch.cuda.is_available())"
```

### Spécifier un GPU Spécifique

Modifiez `config/docker.env` :
```env
CUDA_VISIBLE_DEVICES=0  # Utiliser uniquement le GPU 0
```

Ou utilisez la variable d'environnement lors du démarrage :
```bash
CUDA_VISIBLE_DEVICES=0 docker-compose up
```

## Commandes Utiles

### Arrêter le Container

```bash
docker-compose stop
```

### Redémarrer le Container

```bash
docker-compose restart
```

### Voir les Logs

```bash
docker-compose logs -f st-cdgm-training
```

### Supprimer le Container

```bash
docker-compose down
```

### Nettoyer (supprime aussi les volumes)

```bash
docker-compose down -v
```

## Troubleshooting

### GPU Non Détecté

1. Vérifiez que nvidia-docker2 est installé :
   ```bash
   docker run --rm --gpus all nvidia/cuda:13.1-base-ubuntu22.04 nvidia-smi
   ```

2. Vérifiez la configuration Docker Compose :
   ```yaml
   deploy:
     resources:
       reservations:
         devices:
           - driver: nvidia
             count: all
             capabilities: [gpu]
   ```

### Erreurs de Permission

Si vous avez des erreurs de permission avec les volumes :
```bash
# Linux/Mac: Ajuster les permissions
sudo chown -R $USER:$USER data/ models/ results/
```

### Mémoire Insuffisante

Si le container manque de mémoire :
1. Réduisez le batch size dans la configuration
2. Activez gradient checkpointing dans `config/training_config.yaml`
3. Réduisez le nombre de workers dans le DataLoader

### Container Ne Démarre Pas

Vérifiez les logs :
```bash
docker-compose logs st-cdgm-training
```

## Notes

- Le container utilise l'image PyTorch officielle avec CUDA 13.1
- Les données et modèles sont persistés sur l'hôte via les volumes
- Les modifications du code source sont immédiatement visibles dans le container (bind mount)
- Pour une image personnalisée, créez un `Dockerfile` et modifiez `docker-compose.yml`

