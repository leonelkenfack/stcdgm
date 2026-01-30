# Guide Pédagogique : Comprendre le Modèle ST-CDGM

**Pour les non-initiés**

---

## 🎯 Introduction : Qu'est-ce que nous essayons de faire ?

Imaginez que vous avez une photo floue d'un paysage. Vous pouvez deviner les grandes lignes (montagnes, ciel, forêt), mais les détails sont manquants. Notre modèle ST-CDGM fait exactement cela, mais avec des données météorologiques !

**Le problème** : Les modèles climatiques nous donnent des cartes météo "grossières" (basse résolution), comme une photo pixelisée. Par exemple, une grille de 23×26 points pour toute une région.

**Notre solution** : Transformer ces cartes grossières en cartes détaillées (haute résolution) de 172×179 points, avec tous les détails fins comme les nuages, les fronts météo, etc.

**Pourquoi c'est important ?** Les prévisions locales précises sont essentielles pour l'agriculture, la gestion des catastrophes naturelles, et la planification urbaine.

---

## 📦 Étape 1 : DATA (Préparation des Données)

### 🎯 But
Préparer les données météo brutes pour qu'elles soient utilisables par l'intelligence artificielle.

### 🔍 Analogie
C'est comme préparer des ingrédients avant de cuisiner : laver, couper, mesurer. Vous ne pouvez pas cuisiner avec des légumes non lavés et des quantités approximatives !

### 📝 Exemple Concret

**Avant (données brutes)** :
```
Fichier NetCDF : NorESM2-MM_histupdated.nc
├─ Température à 850 hPa : entre 251K et 299K
├─ Vent U à 850 hPa : entre -37 m/s et +39 m/s
├─ 7300 jours de données (1986-2005)
└─ Grille 23×26 points
```

**Après normalisation** :
```
Températures normalisées : entre -2.5 et +2.5 (écarts-types)
Vents normalisés : entre -2.5 et +2.5
```

### 🔧 Ce qui se passe étape par étape

1. **Normalisation** : 
   - On calcule la moyenne et l'écart-type de chaque variable.
   - On transforme les valeurs pour qu'elles soient centrées autour de 0.
   - **Pourquoi ?** L'IA apprend mieux avec des nombres de taille similaire.

2. **Séquençage** :
   - On découpe les données en "fenêtres" de 6 jours consécutifs.
   - **Analogie** : Comme regarder un film par séquences de 6 images pour comprendre le mouvement.

3. **Création de la Baseline** :
   - On crée une version "floue" de la haute résolution par interpolation.
   - **Exemple** : Si on a 23°C au point A et 25°C au point B, on estime 24°C au milieu.
   - **Important** : L'IA apprendra uniquement à ajouter les détails manquants (le "résidu"), pas à tout refaire !

### ⚙️ Hyperparamètres (Réglages)

| Paramètre | Valeur | À quoi ça sert ? |
|:----------|:-------|:-----------------|
| `seq_len` | 6 jours | Combien de jours consécutifs on regarde à la fois |
| `normalize` | Oui | Mettre les données à la même échelle |
| `baseline_strategy` | "hr_smoothing" | Comment créer la version floue de référence |

---

## 🕸️ Étape 2 : GRAPH BUILDER (Construction du Réseau)

### 🎯 But
Transformer une simple grille de points en un réseau intelligent où chaque point "connaît" ses voisins.

### 🔍 Analogie
Imaginez une carte de France avec toutes les villes. Au lieu de voir juste des points isolés, on crée un réseau où Paris est connecté à ses villes voisines (Versailles, Orléans, etc.). Cela permet de comprendre que ce qui se passe à Paris peut influencer Versailles !

### 📝 Exemple Pas-à-Pas

**Entrée** : Une grille météo de 23 lignes × 26 colonnes = 598 points

**Étape 1 - Créer les nœuds** :
```
Point (0,0) → Nœud #0
Point (0,1) → Nœud #1
Point (0,2) → Nœud #2
...
Point (22,25) → Nœud #597
```

**Étape 2 - Créer les connexions** :

Pour chaque point, on le connecte à ses 8 voisins (comme les cases autour d'une case d'échecs) :

```
Exemple pour le Point (5, 10) :

        [4,9]  [4,10]  [4,11]
          ↖      ↑      ↗
        [5,9] ← [5,10] → [5,11]
          ↙      ↓      ↘
        [6,9]  [6,10]  [6,11]

Ce point a donc 8 connexions
```

**Étape 3 - Ajouter les relations causales** :
- On connecte aussi les variables statiques (topographie) aux variables dynamiques (météo).
- **Exemple** : Une montagne influence le vent et les précipitations autour d'elle.

**Sortie** : 
- 598 nœuds (points météo)
- ~4,700 connexions spatiales (voisins)
- 30,788 nœuds statiques pour la haute résolution

### ⚙️ Hyperparamètres

| Paramètre | Valeur | À quoi ça sert ? |
|:----------|:-------|:-----------------|
| `lr_shape` | (23, 26) | Taille de la grille basse résolution |
| `hr_shape` | (172, 179) | Taille de la grille haute résolution cible |
| `include_mid_layer` | Non | Inclure ou non les niveaux atmosphériques intermédiaires |

---

## 🧠 Étape 3 : ENCODEUR (Extraction de Patterns Météo)

### 🎯 But
Identifier les structures météorologiques importantes dans les données brutes, comme un météorologue expérimenté qui regarde une carte.

### 🔍 Analogie
Un médecin expérimenté peut regarder une radio et immédiatement identifier "fracture", "inflammation", etc. L'encodeur fait pareil avec les cartes météo : il détecte "anticyclone", "front froid", "zone de basse pression", etc.

### 📝 Exemple Concret

**Entrée** : 598 points avec 15 variables chacun (température, vent, humidité...)
```
Point #0 : T=275K, U=5m/s, V=-2m/s, Q=0.004...
Point #1 : T=274K, U=6m/s, V=-1m/s, Q=0.005...
...
```

**Processus - Le GNN (Graph Neural Network) analyse** :
1. Il regarde chaque point et ses voisins.
2. Il détecte des patterns :
   - "Zone de haute pression au centre" (températures élevées, vents divergents)
   - "Front froid qui descend du nord" (gradient de température fort)
   - "Humidité élevée près de l'océan" (gradient d'humidité)

**Sortie - H(0) : État Initial** :
```
3 variables intelligibles (résumés) :
├─ Variable 1 "Advection" : [598 valeurs de 128 dimensions]
├─ Variable 2 "Convection" : [598 valeurs de 128 dimensions]
└─ Variable 3 "Influence statique" : [598 valeurs de 128 dimensions]

Forme finale : [3, 598, 128]
```

### 💡 Ce que fait vraiment l'encodeur

Au lieu de garder 15 variables brutes désorganisées, il crée 3 "résumés intelligents" qui capturent :
- **Advection** : Le transport horizontal (vent qui déplace l'air chaud/froid)
- **Convection** : Les mouvements verticaux (air qui monte/descend)
- **Influence statique** : L'effet de la topographie (montagnes, océans)

### ⚙️ Hyperparamètres

| Paramètre | Valeur | À quoi ça sert ? |
|:----------|:-------|:-----------------|
| `hidden_dim` | 128 | Taille des "résumés" pour chaque point |
| `metapaths` | 3 chemins | Combien de types de relations météo on veut capturer |

---

## 🔄 Étape 4 : RCN (Prédiction de l'Évolution Temporelle)

### 🎯 But
Comprendre comment la météo évolue dans le temps, en respectant les lois de cause à effet.

### 🔍 Analogie
Imaginez un jeu d'échecs : vous devez prédire le coup suivant en comprenant comment les pièces s'influencent mutuellement (le fou menace la tour, le cavalier protège le roi, etc.) ET en vous souvenant de ce qui s'est passé avant. Le RCN fait exactement ça avec la météo !

### 📝 Les Concepts Clés Expliqués

#### 1. **H(0) - L'État Initial**
**C'est quoi ?** La "photo" de la situation météo au départ (temps 0).
**Exemple** : 
```
H(0) = [
  Advection : "Vent d'ouest dominant",
  Convection : "Air stable, peu de mouvements verticaux",
  Statique : "Influence des Alpes au sud"
]
```

#### 2. **Driver - Les Nouvelles Observations**
**C'est quoi ?** Les nouvelles données qui arrivent à chaque instant.
**Exemple** :
```
Driver au jour 1 : "La température a augmenté de 2°C"
Driver au jour 2 : "Le vent s'est renforcé à 15 m/s"
```
**Analogie** : Comme les nouvelles informations qu'un médecin reçoit pour ajuster son diagnostic.

#### 3. **H(t) → H(t+1) - L'Évolution**
**C'est quoi ?** Comment on passe de l'état actuel à l'état suivant.
**Exemple** :
```
Hier (t=0) : "Anticyclone stable"
      ↓
Aujourd'hui (t=1) : "Anticyclone qui se déplace vers l'est"
      ↓
Demain (t=2) : "Début de dépression atlantique"
```

#### 4. **SCM - Les Règles de Cause à Effet**
**C'est quoi ?** SCM = Structural Causal Model (Modèle Causal Structurel)
**Son rôle** : Apprendre les relations de cause à effet entre phénomènes météo.

**Exemple concret** :
```
Cause → Effet :
"Haute pression" → "Temps sec"
"Basse pression" → "Risque de pluie"
"Différence de pression" → "Vent fort"
```

**Comment ça marche ?** Le SCM utilise une matrice A_dag (comme un tableau de relations) :
```
            Pression   Vent   Température
Pression    [   0      0.8      0.3     ]
Vent        [   0       0       0.2     ]
Température [  0.1      0        0      ]

Lecture : "La pression influence le vent (0.8) et la température (0.3)"
```

#### 5. **GRU - La Mémoire du Système**
**C'est quoi ?** GRU = Gated Recurrent Unit (Unité Récurrente à Portes)
**Son rôle** : Se souvenir de ce qui s'est passé avant.

**Analogie** : Imaginez que vous prédisez la météo de demain :
- Sans mémoire : Vous regardez juste aujourd'hui.
- Avec GRU : Vous vous souvenez que depuis 3 jours il fait de plus en plus chaud → tendance à la hausse !

**Exemple** :
```
Jour 1 : T = 15°C → GRU retient "tendance stable"
Jour 2 : T = 17°C → GRU retient "tendance à la hausse"
Jour 3 : T = 19°C → GRU prédit "probablement 21°C demain"
```

#### 6. **Pooling - Résumer l'Information Spatiale**
**C'est quoi ?** Réduire 598 points en un résumé unique.
**Exemple** :
```
Avant pooling (598 points) :
[12.5, 12.3, 12.7, 13.1, 12.9, ... 597 autres valeurs]

Après pooling (1 valeur) :
Moyenne = 12.8°C → "Température moyenne de la région"
```

**Pourquoi ?** La diffusion (étape suivante) a besoin d'un résumé compact, pas de 598 valeurs séparées.

#### 7. **Projection - Adapter le Format**
**C'est quoi ?** Transformer le résumé pour qu'il soit compatible avec le module suivant.
**Analogie** : Comme convertir un fichier Word en PDF pour l'envoyer par email.
**Technique** : Passage de [3, 598, 128] → [1, 3, 128] (via pooling + projection linéaire)

### 🔄 Le Cycle Complet du RCN

```
┌──────────────────────────────────────────────────┐
│  Jour 0 : H(0) "Anticyclone stable"             │
│     ↓                                             │
│  Driver Jour 1 : "Nouvelle observation"          │
│     ↓                                             │
│  SCM : "La pression influence le vent"           │
│     ↓                                             │
│  GRU : "Je me souviens que hier il faisait beau" │
│     ↓                                             │
│  Jour 1 : H(1) "Anticyclone qui se déplace"     │
│     ↓                                             │
│  [Boucle continue pour jours 2, 3, 4, 5...]     │
└──────────────────────────────────────────────────┘
```

### 📉 Fonctions de Perte (Comment le RCN Apprend)

#### L_rec (Reconstruction Loss)
**But** : S'assurer que le RCN garde l'information importante des données d'entrée.
**Comment ?** On demande au RCN de reconstruire les observations originales à partir de son état interne.
**Analogie** : Comme un test où un étudiant doit restituer ce qu'il a appris pour prouver qu'il a compris.

```
Observation réelle : Température = 15°C
Reconstruction RCN : Température = 14.8°C
Erreur = |15 - 14.8| = 0.2°C
→ Plus l'erreur est petite, mieux c'est !
```

#### L_dag (Contrainte Causale)
**But** : Forcer le modèle à apprendre des relations causales cohérentes (pas de cycles).
**Exemple de problème** :
```
❌ MAUVAIS (cycle) :
"A cause B" → "B cause C" → "C cause A" 
(impossible en physique !)

✅ BON (pas de cycle) :
"Pression cause Vent" → "Vent cause Vagues"
```

### ⚙️ Hyperparamètres

| Paramètre | Valeur | À quoi ça sert ? |
|:----------|:-------|:-----------------|
| `hidden_dim` | 128 | Taille de la mémoire par variable |
| `driver_dim` | 15 | Nombre de variables d'entrée |
| `num_vars` | 3 | Nombre de phénomènes météo suivis (advection, convection, statique) |
| `dropout` | 0.0 | Régularisation (0 = pas d'oubli volontaire) |

---

## 🎨 Étape 5 : DIFFUSION (Ajout des Détails Réalistes)

### 🎯 But
Ajouter la "texture" et les détails fins à l'image météo, comme des nuages, des tourbillons, des gradients subtils.

### 🔍 Analogie du Squelette et de la Chair

Imaginez un dessinateur qui travaille en deux étapes :
1. **Le RCN dessine le squelette** : Les grandes lignes, la structure générale ("il y a une montagne ici, un nuage là")
2. **La diffusion ajoute la chair** : Les détails réalistes (texture du nuage, ombres, dégradés)

### 📝 Exemple Concret

**Entrée 1 - Le Conditionnement (du RCN)** :
```
Instructions du RCN : 
"Zone de basse pression au centre"
"Humidité élevée"
"Vent du sud-ouest"
```

**Entrée 2 - Bruit Aléatoire** :
```
Image 172×179 remplie de bruit aléatoire (comme la neige sur un vieux téléviseur)
```

**Processus - 1000 Étapes de "Débruitage"** :

```
Étape 0 : ▓▓▓▓▓▓▓▓ (bruit pur)
Étape 100 : ▓▓▒▒░░▓▓ (vagues formes)
Étape 500 : ▒▒░░  ▒▒ (structures apparaissent)
Étape 1000 : ☁️ ⛅ 🌤️ (nuages détaillés !)
```

À chaque étape, le modèle :
1. Regarde le bruit actuel
2. Consulte les "instructions" du RCN
3. Enlève un peu de bruit en suivant ces instructions
4. Répète 1000 fois

**Sortie - Le Résidu** :
```
Image haute résolution 172×179 avec :
- Structures de nuages réalistes
- Gradients de température subtils
- Tourbillons et fronts météo détaillés
```

### 🔧 Pourquoi la Diffusion et pas juste un CNN ?

| Méthode | Problème | Avantage Diffusion |
|:--------|:---------|:-------------------|
| **CNN Simple** | Résultats flous, manque de détails | Génère des textures fines |
| **GAN** | Instable, difficile à entraîner | Plus stable, convergence garantie |
| **Interpolation** | Trop lisse, pas réaliste | Capture la complexité naturelle |

### 🌟 Le Rôle du Conditionnement

**Sans conditionnement** : La diffusion générerait n'importe quels nuages (aléatoires).

**Avec conditionnement** : Les nuages générés sont cohérents avec la météo prédite par le RCN.

**Exemple** :
```
RCN dit : "Haute pression, temps sec"
→ Diffusion génère : Peu de nuages, ciel dégagé

RCN dit : "Basse pression, humidité élevée"
→ Diffusion génère : Beaucoup de nuages, structures complexes
```

### 📉 Fonction de Perte

#### L_diff (Diffusion Loss)
**But** : Apprendre à prédire le bruit qu'on a ajouté à une image.

**Comment ça marche ?** :
1. On prend une vraie image météo haute résolution.
2. On lui ajoute du bruit (on connaît exactement ce bruit).
3. On demande au modèle de deviner quel bruit on a ajouté.
4. On compare sa prédiction avec le vrai bruit.

```
Bruit réel ajouté : [0.5, -0.3, 0.2, ...]
Bruit prédit : [0.48, -0.31, 0.19, ...]
Erreur = MSE = 0.0012
→ Plus l'erreur est petite, mieux le modèle apprend !
```

### ⚙️ Hyperparamètres

| Paramètre | Valeur | À quoi ça sert ? |
|:----------|:-------|:-----------------|
| `num_diffusion_steps` | 1000 | Combien d'étapes de débruitage |
| `conditioning_dim` | 128 | Taille du "message" venant du RCN |
| `in_channels` | 3 | Nombre de variables météo à générer (T_min, T_mean, T_max) |

---

## 📊 Étape 6 : LOSS (Fonctions de Perte) - Comment le Modèle Apprend

### 🎯 Vue d'Ensemble

Le modèle apprend en minimisant 3 types d'erreurs simultanément. C'est comme un étudiant évalué sur 3 critères différents.

### 🔍 Analogie du Professeur

Imaginez un professeur qui corrige un devoir de géographie avec 3 critères :
1. **Esthétique** : Le dessin de la carte est-il joli et détaillé ? → **L_diff**
2. **Exactitude** : Les données correspondent-elles à la réalité ? → **L_rec**
3. **Cohérence** : Les explications logiques sont-elles correctes ? → **L_dag**

### 📝 Détail des 3 Pertes

#### 1. L_diff (Loss de Diffusion) = Qualité Visuelle

**Formule** : MSE (Mean Squared Error) entre le bruit prédit et le bruit réel

**Exemple chiffré** :
```
Pixel 1 : Bruit réel = 0.5, Prédit = 0.48 → Erreur = (0.5-0.48)² = 0.0004
Pixel 2 : Bruit réel = -0.3, Prédit = -0.31 → Erreur = (-0.3+0.31)² = 0.0001
...
Moyenne sur tous les pixels = 0.0012
```

**Poids** : λ_gen = 1.0 (priorité normale)

#### 2. L_rec (Loss de Reconstruction) = Fidélité aux Données

**But** : Vérifier que le RCN peut reconstruire les observations originales.

**Exemple chiffré** :
```
Variable originale : Température = [15.2, 14.8, 16.1, ...]
Reconstruite par RCN : [15.1, 14.9, 16.0, ...]
Erreur par point : [0.1², 0.1², 0.1², ...] 
Moyenne = 0.01
```

**Poids** : β_rec = 0.1 (10% de l'importance totale)

#### 3. L_dag (Loss de Causalité) = Respect des Lois Physiques

**Formule** : Trace(e^(A²)) - q (Contrainte NO TEARS)

**Explication simple** :
- Le modèle apprend une matrice A qui dit "qui cause quoi".
- Cette perte punit les cycles impossibles (A cause B, B cause C, C cause A).
- Plus la valeur est proche de 0, mieux les relations causales sont respectées.

**Exemple** :
```
Matrice A_dag (3×3) :
       P    V    T
P   [ 0   0.5  0.2 ]  (Pression cause Vent et Température)
V   [ 0    0   0.1 ]  (Vent cause Température)
T   [ 0    0    0  ]  (Température ne cause rien d'autre)

L_dag = 0.02 (proche de 0 → bon !)
```

**Poids** : γ_dag = 0.1 (10% de l'importance totale)

### 🧮 La Formule Totale

$$L_{total} = 1.0 \times L_{diff} + 0.1 \times L_{rec} + 0.1 \times L_{dag}$$

**Exemple de calcul** :
```
L_diff = 0.0012
L_rec = 0.01
L_dag = 0.02

L_total = (1.0 × 0.0012) + (0.1 × 0.01) + (0.1 × 0.02)
        = 0.0012 + 0.001 + 0.002
        = 0.0042

→ Le modèle essaie de réduire ce nombre à chaque itération !
```

---

## 🎓 Étape 7 : TRAINING (Boucle d'Apprentissage)

### 🎯 But
Répéter le processus d'apprentissage jusqu'à ce que le modèle devienne excellent.

### 🔍 Analogie
Comme apprendre à jouer du piano : vous jouez un morceau, le professeur vous dit ce qui ne va pas, vous ajustez, et vous rejouez. Répétez 10,000 fois !

### 🔄 Le Cycle d'Entraînement

```
Itération 1 :
  1. Prendre un batch de données (ex: 6 jours de météo)
  2. Forward Pass : DATA → GRAPH → ENCODER → RCN → DIFFUSION
  3. Calculer L_total (comparer prédiction vs réalité)
  4. Backward Pass : Calculer les gradients (où améliorer ?)
  5. Mettre à jour les poids du modèle
  
Itération 2 :
  [Répéter...]
  
Itération 10,000 :
  [Modèle de plus en plus précis !]
```

### 📉 Évolution Typique de la Loss

```
Epoch 1 : L_total = 0.5000 (modèle très mauvais)
Epoch 5 : L_total = 0.0500 (commence à apprendre)
Epoch 10 : L_total = 0.0080 (beaucoup mieux)
Epoch 20 : L_total = 0.0042 (bon résultat)
```

### ⚙️ Hyperparamètres d'Entraînement

| Paramètre | Valeur | À quoi ça sert ? |
|:----------|:-------|:-----------------|
| `learning_rate` | 0.0001 | Vitesse d'apprentissage (trop grand = instable, trop petit = lent) |
| `epochs` | 20 | Combien de fois on parcourt toutes les données |
| `gradient_clipping` | 1.0 | Empêche les mises à jour trop brutales |
| `optimizer` | Adam | Algorithme d'optimisation utilisé |

---

## 📋 Tableau Récapitulatif des Hyperparamètres

### 🎛️ Tous les Réglages en un Coup d'Œil

| Module | Paramètre | Valeur Défaut | Explication Simple |
|:-------|:----------|:--------------|:-------------------|
| **DATA** | `seq_len` | 6 | Longueur de la séquence temporelle (jours) |
| | `normalize` | Oui | Mettre les données à la même échelle |
| | `baseline_strategy` | "hr_smoothing" | Méthode pour créer l'image floue de référence |
| | `baseline_factor` | 4 | Combien on lisse l'image de base |
| **GRAPH** | `lr_shape` | (23, 26) | Taille de la grille basse résolution |
| | `hr_shape` | (172, 179) | Taille de la grille haute résolution |
| | `include_mid_layer` | Non | Ajouter des niveaux atmosphériques intermédiaires |
| **ENCODER** | `hidden_dim` | 128 | Taille des résumés intelligents |
| | `conditioning_dim` | 128 | Taille du message vers la diffusion |
| | `metapaths` | 3 | Nombre de types de relations météo |
| **RCN** | `num_vars` | 3 | Nombre de phénomènes suivis (advection, convection, statique) |
| | `hidden_dim` | 128 | Taille de la mémoire |
| | `driver_dim` | 15 | Nombre de variables d'entrée |
| | `dropout` | 0.0 | Taux d'oubli volontaire (régularisation) |
| **DIFFUSION** | `num_diffusion_steps` | 1000 | Nombre d'étapes de débruitage |
| | `in_channels` | 3 | Nombre de variables à générer |
| | `conditioning_dim` | 128 | Taille du message reçu du RCN |
| **LOSS** | `lambda_gen` | 1.0 | Importance de la qualité visuelle |
| | `beta_rec` | 0.1 | Importance de la fidélité aux données |
| | `gamma_dag` | 0.1 | Importance du respect de la causalité |
| **TRAINING** | `learning_rate` | 0.0001 | Vitesse d'apprentissage |
| | `epochs` | 20 | Nombre de passages sur toutes les données |
| | `gradient_clipping` | 1.0 | Limite des mises à jour |

---

## 🎬 Conclusion : Le Voyage Complet des Données

Récapitulons le voyage d'une donnée météo, du début à la fin :

1. **DATA** : "Fichier NetCDF brut" → "Données normalisées prêtes"
2. **GRAPH** : "Grille 23×26" → "Réseau de 598 nœuds connectés"
3. **ENCODER** : "15 variables brutes" → "3 résumés intelligents H(0)"
4. **RCN** : "H(0)" → "Évolution temporelle H(1), H(2)... H(6)"
5. **DIFFUSION** : "Instructions + Bruit" → "Image détaillée 172×179"
6. **LOSS** : "Comparer avec la réalité" → "Ajuster le modèle"
7. **TRAINING** : Répéter 10,000 fois → "Modèle expert !"

**Résultat Final** : Une carte météo détaillée et réaliste, respectant la physique, avec des textures fines que les méthodes classiques ne peuvent pas générer !

---

## 💡 Questions Fréquentes

**Q : Pourquoi ne pas juste interpoler (agrandir) l'image basse résolution ?**
R : L'interpolation donne des résultats flous. Elle ne peut pas inventer les détails fins comme les structures de nuages ou les tourbillons.

**Q : Ça prend combien de temps à entraîner ?**
R : Avec un GPU moderne, quelques heures à quelques jours selon la quantité de données.

**Q : Le modèle peut-il se tromper ?**
R : Oui ! C'est pour ça qu'on utilise 3 fonctions de perte pour le surveiller. Plus on l'entraîne, moins il se trompe.

**Q : Pourquoi c'est si compliqué ?**
R : Parce que la météo est un système complexe avec de la mémoire temporelle (ce qui s'est passé hier compte) et des relations causales (la pression cause le vent). Un modèle simple ne peut pas capturer tout ça.

---

**📖 Pour aller plus loin, consultez :**
- `RAPPORT_TECHNIQUE_COMPLET.md` : Version technique détaillée
- `ARCHITECTURE_MODEL.md` : Architecture complète avec formules mathématiques



