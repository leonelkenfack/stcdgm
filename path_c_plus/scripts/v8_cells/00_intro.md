# ST-CDGM **V8** — run GPU complet

Mêmes jeux de données que depuis le début. Aucun prétraitement à lancer : les
**13 nœuds libres** sont dérivés à la volée depuis les 15 canaux bruts.

## Ce qui change par rapport à tous les runs précédents

| | avant | V8 |
|---|---|---|
| nœuds du DAG | plongements de métachemins, **mêmes entrées pour tous** | 13 quantités physiques, **un canal chacun** |
| drivers | un embedding partagé + un biais appris | **routage diagonal** : l'info inter-variable ne passe que par `A[u,v]` |
| structure | `A_dag` mono-lag | `A_inst` (τ=0) **et** `A_dag` (τ≥1) |
| prior | MSE vers la matrice complète | **C7** : 3 niveaux, λ annelé par niveau, hors-prior LIBRE |
| décodeur | requêtes apprises, aveugles à l'entrée | **A2a** : requêtes amorcées par l'état + encodage positionnel partagé |
| perte étage 1 | MSE sur log1p | **A3** : vraisemblance Bernoulli-Gamma, ancre `μ = p·α·β` |
| retour en mm | `expm1(μ)` | **A1** : `expm1(μ + s²/2)`, `s²` hétéroscédastique |

## Pourquoi ces changements

Trois audits ont trouvé le même défaut à trois endroits : **les variables ne se
distinguaient jamais par leurs entrées**, seulement par des poids appris. Les
types de nœuds du builder recevaient tous les mêmes 15 canaux ; `driver_encoder`
distribuait le même vecteur aux q variables ; et aucune ligne du dépôt ne
calculait l'IVT. Un DAG dans ces conditions est décoratif : rien ne le rend
load-bearing. V8 corrige les trois.

## Interrupteurs

Chaque brique s'éteint séparément (`V8` en Cell 2). P1 demande **un seul
changement par run** pour pouvoir attribuer l'effet. Tout à `False` ≈ pile V5.

## Cible

**Parité in-distribution + gain OOD.** Une régression ID de quelques pour cent
est prévue et acceptée : le DAG gelé est une feature OOD assumée, pas un
avantage ID. Le verdict se joue sur EC-Earth3, puis **une seule fois** sur le
holdout.

## Comparaison aux autres modèles

Les Cells 10-11 mesurent V8 **dans le protocole du 3-way** (K=32, 24 pas,
cfg 0.0, split de test complet, composition mm par membre, `evaluate_ensemble`)
puis relisent la table déjà produite pour V6', ORACLE (V5) et CorrDiff : ces
trois-là ne sont **pas réévalués**. C'est aussi pourquoi l'étage 2 est l'UNet
CorrDiff-Normal (~43 M paramètres) et non l'UNet minimal du YAML de base : à
1 M paramètres, la table comparerait des tailles de réseau.

Si la table de référence est absente, la Cell 11 le dit et marque la
comparaison non valide plutôt que de s'appuyer sur les valeurs
pré-enregistrées, que le prérégistre V6' déclare lui-même calculées avec une
Convention A buggée.

## Règle absolue

**NorESM2-MM est le holdout OOD.** Aucune cellule ne l'ouvre. Un garde lève à
la moindre tentative.
