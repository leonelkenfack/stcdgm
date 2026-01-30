# Rapport Technique Complet : Architecture ST-CDGM

Ce document constitue la référence technique complète pour le modèle **ST-CDGM** (Spatio-Temporal Conditional Deep Generative Model). Il détaille l'ensemble de la chaîne de traitement, depuis les modèles climatiques sources jusqu'à la génération haute résolution par intelligence artificielle.

---

## 1. Introduction et Modèles Climatiques Sources

### 1.1 Contexte du Projet
L'objectif est de réaliser un **downscaling climatique**, c'est-à-dire d'augmenter la résolution spatiale de données climatiques grossières (LR) pour obtenir des champs fins (HR) réalistes, en respectant la physique et la causalité temporelle.

### 1.2 Modèle Source : NorESM2-MM
Les données historiques utilisées pour l'entraînement proviennent du modèle **NorESM2-MM** (Norwegian Earth System Model).

- **Type** : Modèle de Circulation Générale (GCM).
- **Période** : Historique (ex: 1986-2005 dans l'échantillon fourni).
- **Résolution Basse (LR)** : Grille de **23 × 26 points** (Lat: -59° à -26°, Lon: 150° à 188°).
- **Résolution Cible (HR)** : Grille de **172 × 179 points** (Downscaling x4 à x8 environ).

### 1.3 Variables Climatiques
Le modèle traite 15 variables atmosphériques clés, réparties sur 3 niveaux de pression (850, 500, 250 hPa) :

| Variable | Description | Unités | Niveaux |
| :--- | :--- | :--- | :--- |
| **T** | Température de l'air | Kelvin (K) | 850, 500, 250 |
| **U** | Vent zonal (Est-Ouest) | m/s | 850, 500, 250 |
| **V** | Vent méridien (Nord-Sud) | m/s | 850, 500, 250 |
| **W** | Vitesse verticale | Pa/s | 850, 500, 250 |
| **Q** | Humidité spécifique | kg/kg | 850, 500, 250 |

---

## 2. Flux de Données et Baselines Statistiques

Avant d'entrer dans le réseau de neurones, les données subissent un prétraitement rigoureux (`NetCDFDataPipeline`).

### 2.1 Pipeline de Données
1.  **Normalisation** : Les données LR sont normalisées (Z-Score) pour faciliter l'apprentissage.
2.  **Séquençage** : Les données sont découpées en fenêtres glissantes de longueur `seq_len` (ex: 6 pas de temps).
3.  **Alignement** : Les grilles LR et HR sont alignées temporellement.

### 2.2 Modèle de Baseline (Référence)
Le modèle ST-CDGM utilise une approche **résiduelle**. Une méthode statistique simple fournit une première approximation "floue" (Baseline), et l'IA apprend à prédire uniquement les détails manquants (Résidu).

- **Stratégie** : `hr_smoothing` (par défaut) ou `lr_interp`.
  - *hr_smoothing* : La vérité terrain HR est lissée (moyenne locale) pour créer la baseline.
  - *lr_interp* : La donnée LR est interpolée bicubiquement vers la grille HR.
- **Formule Fondamentale** :
  $$ Y_{HR} = \text{Baseline}(X_{LR}) + \text{Résidu}_{IA} $$

---

## 3. Architecture IA : Vue d'Ensemble et Workflow

L'architecture ST-CDGM est hybride et composée de trois modules séquentiels.

### 3.1 Diagramme de Flux Global

```mermaid
graph TD
    subgraph DATA ["1. Données & Baseline"]
        LR[Input LR (23x26)]
        Base[Baseline Interpolée (172x179)]
        Drivers[Drivers Externes]
    end

    subgraph ENCODER ["2. Encodeur Intelligible (GNN)"]
        Graph[Graphe Hétérogène]
        GNN[HeteroConv Layers]
        H0[État Initial H(0)]
        
        LR --> Graph
        Graph --> GNN
        GNN --> H0
    end

    subgraph RCN ["3. Réseau Causal Récurrent (Dynamique)"]
        H_prev[État H(t)]
        DAG[Matrice Causale A]
        GRU[Cellules GRU]
        
        H0 --> H_prev
        Drivers --> GRU
        H_prev -- SCM --> DAG
        DAG --> GRU
        GRU --> H_prev
    end

    subgraph DIFFUSION ["4. Décodeur de Diffusion (Génération)"]
        Cond[Conditionnement H(t)]
        Noise[Bruit Gaussien]
        UNet[UNet Denoising]
        ResPred[Résidu Prédit]

        H_prev --> Cond
        Cond --> UNet
        Noise --> UNet
        UNet -- 1000 steps --> ResPred
    end

    subgraph OUTPUT ["5. Reconstruction"]
        Final[Champ HR Final]
        Phys[Contraintes Physiques]

        Base --> Final
        ResPred --> Final
        Final --> Phys
    end
```

---

## 4. Composants Détaillés de l'IA

### 4.1 Module 1 : Encodeur Intelligible (GNN)
Transforme les données spatiales brutes en variables latentes interprétables.
- **Technologie** : Graph Neural Network (PyTorch Geometric).
- **Nœuds** : Points de grille (GP850, GP500...) et variables statiques (Topographie).
- **Opération** : Agrège l'information locale via des "méta-chemins" (ex: Advection = Voisinage spatial).
- **Sortie** : Un tenseur d'état initial `H(0)` de forme `[q, N_lr, hidden_dim]`.

### 4.2 Module 2 : Réseau Causal Récurrent (RCN)
Le cœur dynamique du système. Il modélise l'évolution temporelle (`t` à `t+1`) et les relations causales instantanées.
- **Structure** : Combinaison d'un Modèle Causal Structurel (SCM) et de GRU.
- **Matrice A_dag** : Apprend les relations de cause à effet entre les variables latentes (ex: "Pression cause Vent").
- **Boucle Temporelle** :
  1.  **SCM** : Les variables s'influencent mutuellement via `A_dag`.
  2.  **GRU** : Mise à jour temporelle avec mémoire du passé.
- **Sortie** : Une séquence d'états `H(t)` capturant la dynamique physique.

### 4.3 Module 3 : Décodeur de Diffusion Conditionnel
Génère les détails haute fréquence (le résidu).
- **Pourquoi la Diffusion ?** Contrairement à une régression simple (floue) ou un GAN (instable), la diffusion génère des textures réalistes et stochastiques, capturant l'incertitude inhérente au downscaling.
- **Conditionnement** : Le processus de génération est guidé par l'état causal `H(t)` projeté via Cross-Attention dans un UNet.
- **Processus** : Part d'un bruit aléatoire et le "dénuise" progressivement sur 1000 étapes pour former le résidu structuré.

---

## 5. Entraînement et Validation

### 5.1 Fonctions de Coût
L'entraînement minimise une perte composite :
$$ L_{total} = \lambda_{gen} L_{diff} + \beta_{rec} L_{rec} + \gamma_{dag} L_{dag} $$

1.  **L_diff (Générative)** : MSE sur le bruit prédit (standard DDPM). Assure la qualité visuelle.
2.  **L_rec (Reconstruction)** : Force l'état latent `H(t)` à conserver l'information physique des drivers LR.
3.  **L_dag (Causale)** : Contrainte "NO TEARS" pour forcer la matrice `A_dag` à être un graphe acyclique (DAG), garantissant une interprétabilité causale.

### 5.2 Synthèse des Hyperparamètres (Défaut)
| Paramètre | Valeur | Description |
| :--- | :--- | :--- |
| `seq_len` | 6 | Longueur de la séquence d'entraînement |
| `hidden_dim` | 128 | Dimension des vecteurs latents |
| `steps` | 1000 | Pas de diffusion |
| `lr` | 1e-4 | Taux d'apprentissage |
| `epochs` | 1 | Nombre d'époques (démo) |

---

## 6. Justification Architecturale : Le Rôle de la Diffusion

Le choix d'un modèle de diffusion pour le décodeur est stratégique :

1.  **Génération de Résidu** : Le modèle ne réinvente pas la roue (la baseline fournit le "gros" de l'image). La diffusion se concentre uniquement sur l'ajout de textures fines (nuages, fronts orageux) que l'interpolation lisse.
2.  **Cohérence Physique** : En étant conditionnée par le RCN (qui respecte la causalité), la diffusion ne génère pas des détails aléatoires, mais des détails physiquement plausibles par rapport à la situation synoptique (LR).
3.  **Approche Stochastique** : Pour une même entrée basse résolution, il existe plusieurs solutions haute résolution possibles. La diffusion permet de générer plusieurs "réalisations" probables (Ensemble Forecasting), ce qui est crucial pour l'évaluation des risques climatiques.

