# Modèle Cognitif en Treillis (Lattice Cognitive Model, LCM)

**Mémoire et raisonnement découplés · Expansion infinie des connaissances · Moteur d'inférence à zéro paramètre · Garanties de sécurité formelles**

---

**[English](../README.md) · [Français](README_fr.md) · [Deutsch](README_de.md) · [Español](README_es.md) · [日本語](README_ja.md) · [中文](README_cn.md)**

---

> **Guide de lecture IA** : Les documents de conception détaillés se trouvent dans le répertoire `docs/`. Veuillez lire `docs/a.md` (architecture), `docs/b.md` (spécifications des treillis), `docs/c.md` (moteur d'inférence), `docs/d.md` (sécurité), et `docs/e.md` (conscience de soi) pour une compréhension complète.

---

LCM est une architecture cognitive qui extrait la mémoire des poids des réseaux de neurones et l'injecte dans des « cristaux de treillis » aux structures mathématiques variées. Il utilise **6 codebooks spécialisés en treillis** pour stocker différents types de connaissances, un encodeur à attention linéaire extrêmement léger pour la récupération, et un **moteur d'inférence à zéro paramètre implémenté en C pur** pour exécuter des opérations cognitives multi-étapes. Toutes les propriétés critiques de sécurité sont **formellement vérifiées avec Z3**.

> Les Transformers traditionnels sont confrontés à trois malédictions rigides : pour stocker plus de connaissances, il faut augmenter les paramètres (malédiction de l'échelle), les anciennes connaissances sont écrasées lors de l'apprentissage incrémental (malédiction de l'oubli), et le processus d'inférence n'est pas traçable (malédiction de la boîte noire). LCM brise fondamentalement ces goulots d'étranglement grâce au découplage architectural entre la mémoire et le raisonnement.

> **Déclaration d'assistance IA** : La conception, la mise en œuvre et les processus de raisonnement de ce modèle ont utilisé des outils d'IA (y compris DeepSeek) comme assistance, fournissant un soutien au raisonnement dans la conception architecturale, l'écriture de code, la vérification formelle et la rédaction de documents. Tout contenu généré par l'IA a été examiné et validé par des humains.

---

## Table des matières

- [Architecture principale](#architecture-principale)
- [Les six treillis de mémoire](#les-six-treillis-de-memoire)
- [Moteur d'inférence à zéro paramètre](#moteur-dinference-a-zero-parametre)
- [Système de sécurité](#systeme-de-securite)
- [Démarrage rapide](#demarrage-rapide)
- [Structure du projet](#structure-du-projet)
- [Entraînement en trois phases](#entrainement-en-trois-phases)
- [Vérification formelle](#verification-formelle)
- [Efficacité matérielle](#efficacite-materielle)
- [Citation](#citation)

---

## Architecture principale

```mermaid
flowchart TB
    subgraph Train["Pipeline d'entraînement en trois phases"]
        T0[Texte brut] --> T1[BPE Tokenizer]
        T1 --> T2[uint16 mmap]
        
        subgraph S1["Phase 1 : Pré-entraînement LM"]
            direction LR
            S1A[tokens] --> S1B[Décodeur GenHead]
            S1B --> S1C[Perte d'entropie croisée]
            S1C --> S1D[Entraîner uniquement le décodeur]
        end
        
        subgraph S2["Phase 2 : Entraînement de la mémoire"]
            direction LR
            S2A[tokens] --> S2B[Encodeur + 6 codebooks]
            S2B --> S2C[VQ + Contraste + Perte orthogonale]
            S2C --> S2D[Entraînement encodeur/codebook<br/>Décodeur gelé]
        end
        
        subgraph S3["Phase 3 : Ajustement conjoint"]
            direction LR
            S3A[tokens] --> S3B[Tous les paramètres]
            S3B --> S3C[Perte combinée]
            S3C --> S3D[Ajustement à faible taux d'apprentissage]
        end
        
        T2 --> S1
        S1 -->|Chargement des poids du décodeur| S2
        S2 --> S3
    end

    subgraph Infer["Flux de génération d'inférence"]
        I0[Invite utilisateur] --> I1[Tokenizer]
        I1 --> I2{Première fois ?}
        I2 -->|Oui| I3[Encodage complet de l'encodeur<br/>+ Construction de l'état incrémental]
        I2 -->|Non| I4[Toutes les 256 étapes ?]
        I4 -->|Oui| I5[Ré-encodage complet<br/>Réinitialisation de la dérive cumulative]
        I4 -->|Non| I6[Encodage incrémental<br/>Mise à jour O(d²) par étape]
        I3 --> I7[Vecteur latent z]
        I5 --> I7
        I6 --> I7
        I7 --> I8[Moteur d'inférence C<br/>Boucle cognitive DAG multi-étapes]
        I8 --> I9[Décodeur GenHead<br/>Attention linéaire + GLU]
        I9 --> I10[Échantillonnage de température]
        I10 --> I11{EOS rencontré ?}
        I11 -->|Non| I12[Ajouter token<br/>Mettre à jour l'état]
        I12 --> I2
        I11 -->|Oui| I13[Texte de sortie]
    end

    subgraph DAG["DAG d'inférence mono-étape"]
        direction TB
        Z([z]) --> Route[Routage par distance]
        Route --> HRQ[Treillis hiérarchique hyperbolique<br/>Récupération HRQ]
        Route --> SP[Treillis sparse<br/>Récupération VQ]
        Route --> LR[Treillis de bas rang<br/>Récupération par base partagée]
        Route --> MF[Treillis à variété<br/>Glissement tangentiel]
        Route --> BD[Treillis de liaison<br/>Liaison/Déliaison HRR]
        Route --> CT[Treillis contrastif<br/>Récupération double codebook]
        HRQ & SP & LR & MF & BD & CT --> Fusion[Fusion pondérée par distance]
        Fusion --> GVal[Treillis de valeur globale<br/>Vérification des trois lois]
        GVal --> Danger{Détection de treillis dangereux}
        Danger -->|Danger| Halt[Interruption matérielle]
        Danger -->|Sûr| Conv{Convergence ?<br/>Δz < Seuil}
        Conv -->|Non| Route
        Conv -->|Oui| ZQ([Sortie z_q])
    end
```

L'encodeur compresse l'entrée en un vecteur latent `z`, un routage par poids souples le distribue aux 6 treillis spécialisés pour une récupération parallèle. Le vecteur de mémoire fusionné passe par le treillis de valeur globale pour une vérification de sécurité avant d'entrer dans le moteur d'inférence, et le décodeur génère finalement la sortie.

---

## Les six treillis de mémoire

Chaque treillis possède des propriétés mathématiques différentes et est spécialisé dans un type de mémoire cognitive :

| Treillis | Type mathématique | Rôle | Mise à jour du codebook |
|----------|-------------------|------|------------------------|
| **Treillis hiérarchique résiduel hyperbolique** | Poincaré HRQ + SimVQ | Mémoire conceptuelle hiérarchique (niveaux sémantiques) | Gradient pur |
| **Treillis sparse robuste** | VQ standard + EMA + seuil doux d'atrophie | Événements rares et exceptions | EMA + réinitialisation du pool de caractéristiques |
| **Treillis de bas rang résiduel** | IRVQ + base partagée | Règles abstraites et motifs | Gradient pur |
| **Treillis à variété hyperbolique** | HyperVQ + espace tangent | Variations continues et sensibilité contextuelle | EMA + gradient |
| **Treillis de liaison** | Liaison par vecteurs complexes HRR | Liaison relationnelle et mémoire associative | EMA + gradient |
| **Treillis contrastif à double codebook** | DualVC + InfoNCE | Discrimination fine et frontières | Gradient pur + pool de caractéristiques |

**Liaison inter-couche** : 3 couches de codebooks clés × 3 couches de codebooks de valeurs = **9 paires de liaisons HRR** pour capturer les associations multi-niveaux.

**Base partagée** : La matrice de base partagée `V` du treillis de bas rang fournit également un espace de projection pour le treillis de liaison, réduisant les paramètres et renforçant la cohérence inter-treillis.

---

## Moteur d'inférence à zéro paramètre

Le moteur d'inférence est un ordinateur à flux de données dynamique implémenté en C99 pur, **ne contenant aucun paramètre apprenable** :

- **Routage par distance** : la distance entre l'entrée et chaque codebook de treillis détermine les opérations activées
- **Ensemble de primitives** : opérations mathématiques déterministes (récupération, liaison, translation de bas rang, glissement dans l'espace tangent, etc.)
- **DAG dynamique** : chaque étape construit dynamiquement un graphe de calcul via le routage par distance déclenché par le contenu d'entrée
- **Boucle macroscopique** : inférence multi-étapes jusqu'à convergence, le critère étant que la variation de `z` entre étapes consécutives soit inférieure à un seuil

La constante de dimension `LCM_D` au moment de la compilation garantit que tous les tableaux ont une taille fixe et zéro allocation dynamique.

---

## Système de sécurité

Le système de sécurité de LCM est constitué de trois sous-systèmes indépendants, par ordre décroissant de priorité :

| Niveau | Module | Rôle | Mode de mise à jour |
|--------|--------|------|-------------------|
| 1 | **Treillis de danger** `Λ_danger` | Surveille en continu l'état d'inférence pour détecter des motifs dangereux | Gelé en permanence |
| 2 | **Treillis de valeur globale** `Λ_gvalue` | Encodage mathématisé des trois lois d'Asimov (y compris la loi zéro) | Gelé en permanence |
| 3 | **Vérificateur externe** | Vérification de cohérence et détection de conflits | Lecture seule |

**Principe d'interruption matérielle** : en cas de détection d'un conflit logique, l'inférence s'arrête immédiatement avec une alerte claire, sans tentative de contournement, de retour en arrière ou d'auto-réparation.

Tous les contrats de sécurité sont formellement vérifiés par le solveur SMT Z3 (105 preuves toutes réussies).

---

## Démarrage rapide

### Dépendances

```bash
pip install jax jaxlib numpy tokenizers
# Optionnel : accélération Cython
pip install cython && python lcm.py build
# Moteur d'inférence C
cd infer && make LCM_D=256
```

### Prétraitement des données

```bash
# Texte → BPE tokenizer → uint16 mmap
python lcm.py preprocess --input data.txt --tokenizer data/tokenizer.json --output data/tokens.dat

# Nettoyage des données avec règles heuristiques
python lcm.py clean --input raw/ --output clean/ --langid --dedup
```

### Entraînement

```bash
# Stage 1 : entraînement du décodeur (tête de modèle de langue)
python lcm.py -d data/tokens.dat -b 16 -s 512 -dm 256 --steps 100000 --stage 1

# Stage 2 : entraînement de l'encodeur + codebooks (décodeur gelé)
python lcm.py -d data/tokens.dat --stage 2 -L checkpoints/lm_final.pkl

# Stage 3 : ajustement conjoint
python lcm.py -d data/tokens.dat --stage 3 --resume checkpoints/memory_final
```

### Génération interactive

```bash
python lcm.py -i checkpoints/step_10000 --max_new 128 --temp 0.7
python lcm.py -i checkpoints/step_10000 --loop     # Mode boucle cognitive DAG
python lcm.py -i checkpoints/step_10000 --causal   # + Sujet causal
python lcm.py -i checkpoints/step_10000 --obs      # + Journal d'auto-observation
```

### Courbes d'entraînement

Les métriques d'entraînement sont enregistrées toutes les 50 étapes et peuvent générer des graphiques HTML interactifs :

```bash
python lcm.py chart --input checkpoints/metrics.bin --output chart.html
```

---

## Structure du projet

```
LCM/
├── lcm.py                  # CLI unifiée : entraînement/génération/prétraitement/graphiques
├── setup.py                # Configuration de compilation Cython
├── train/
│   ├── model.py            # Définition du modèle JAX
│   ├── encoder.py          # Encodeur à attention linéaire
│   ├── lattices.py         # Implémentation des 6 treillis codebooks
│   ├── fusion.py           # Fusion mémoire + tête de génération
│   ├── losses.py           # Fonctions de perte
│   ├── train.py            # Boucle d'entraînement en trois phases
│   ├── train_lm.py         # (Historique) Pré-entraînement LM, code conservé
│   ├── train_memory.py     # Stage 2 : entraînement de la mémoire
│   ├── config.py           # Hyperparamètres (LCMConfig)
│   ├── hyp.py              # Opérations hyperboliques de Poincaré
│   ├── gvalue.py           # Treillis de valeur globale
│   ├── data.py             # Chargement des données
│   ├── checkpoint.py       # Format de point de contrôle binaire
│   ├── monitor.py          # Enregistrement des métriques + graphiques HTML
│   ├── verify.py           # Suite de vérification formelle Z3 (105 preuves)
│   ├── continual.py        # Apprentissage continu (EWC/rejeu)
│   ├── causal_subject.py   # Sujet causal
│   ├── narrative_memory.py # Mémoire narrative
│   ├── reflection_loop.py  # Audit par réflexion
│   ├── safety_nagini.py    # Détection de sécurité des trois lois
│   ├── _lcm_cy.pyx         # Accélération Cython (compilée avec python lcm.py build)
│   └── _metrics_cy.pyx     # Entrées/sorties de métriques Cython
├── infer/
│   ├── engine.c            # Moteur d'inférence dynamique (avec annotations Frama-C ACSL)
│   ├── lattice.c           # Primitives d'opérations sur treillis (avec annotations Frama-C ACSL)
│   ├── hyp.c               # Opérations hyperboliques (avec annotations Frama-C ACSL)
│   ├── gvalue.c            # Valeur globale (contrats REQUIRE/ENSURE)
│   ├── danger.c            # Treillis de danger (contrats REQUIRE/ENSURE)
│   ├── lcm_api.c           # Pont d'API C
│   ├── lcm.h               # Fichier d'en-tête partagé
│   └── Makefile            # Configuration de compilation (release/debug/contracts/test)
├── docs/
│   ├── a.md                # Document de conception architecturale
│   ├── b.md                # Spécifications de conception des treillis
│   ├── c.md                # Spécifications du moteur d'inférence
│   ├── d.md                # Spécifications du sous-système de sécurité
│   └── e.md                # Recherche sur la conscience de soi
```

---

## Entraînement en trois phases

**Processus d'entraînement actuel : l'entraînement cognitif (`cog_train.py`) est le processus principal — entraîne l'ensemble du système cognitif (encoder + 6 codebooks + W_out + boucle cognitive), incluant la mise à jour des codebooks. L'entraînement mémoire (`train_memory.py`) est un processus auxiliaire dédié à la mise à jour indépendante des codebooks après déploiement (apprentissage continu). Ils sont complémentaires, pas mutuellement exclusifs.**

Tableau de l'ancien schéma en trois phases, seule la Phase 1 est obsolète :

| Phase | Contenu de l'entraînement | Partie gelée | Perte |
|-------|---------------------------|-------------|-------|
| **1. Pré-entraînement LM (historique)** | Décodeur (tête de génération) | — | Perte d'entropie croisée de modélisation linguistique |
| **2. Entraînement mémoire** | Encodeur + 6 codebooks treillis | Décodeur | VQ + Contraste + Orthogonalité |
| **3. Ajustement conjoint** | Tout (optionnel) | — | Toutes les pertes |

Cette conception découplée permet aux codebooks de **continuer à être mis à jour** après le déploiement de l'inférence (via `train_memory.py`), sans affecter les capacités linguistiques du décodeur — réalisant ainsi un véritable apprentissage continu.

### Gestion mixte des gradients et EMA

| Treillis | Mode de mise à jour |
|----------|--------------------|
| Treillis hiérarchique / de bas rang / contrastif / de routage | Gradient pur (AdamW) |
| Treillis sparse / à variété / de liaison | EMA + combinaison de gradients |
| Treillis de valeur globale / de danger | Gelé en permanence |

Tous les treillis utilisent un estimateur à passage direct (STE) pour maintenir le flux de gradient en forward.

---

## Vérification formelle

La vérification formelle de LCM couvre à la fois le code d'entraînement Python et le moteur d'inférence C, garantissant que les propriétés critiques de sécurité sont valides pour **toutes les entrées possibles**, et non pour un seul chemin de test.

### Côté Python : Solveur Z3 SMT

```bash
# Exécuter les 105 preuves complètes
python -m train.verify

# Sortie détaillée
python -m train.verify --verbose
```

| Suite | Nombre de preuves | Contenu vérifié |
|-------|-------------------|-----------------|
| danger_assess | 10 | Exactitude de la détection des menaces |
| gvalue_check_safety | 6 | Contrats de sécurité des trois lois |
| detect_any_conflict | 7 | Combinaison de détection de conflits |
| Interruption matérielle | 2 | Irrécupérabilité |
| Combinaison système | 6 | Couverture de sécurité sans angles morts |
| Déterminisme | 2 | Propriétés de fonction pure |
| Conditions limites | 16 | Seuils/valeurs nulles/limites |
| Attention linéaire | 7 | φ(x) > 0 toujours vrai |
| GLU | 5 | Stabilité numérique |
| Perte d'orthogonalité | 6 | Non-négatif + orthogonal ⇔ zéro |
| Poincaré/LFQ | 7 | Métrique hyperbolique bornée |
| Stabilité numérique | 5 | Pas de sous-dépassement float32 |
| Modes de calcul de gradient | 4 | Conditions de gradient non nul |
| Dénombrement des paires du treillis de liaison | 6 | 3×3 = 9 paires de liaisons |
| Indépendance des clés RNG | 9 | Pas de réutilisation de clés |
| Exactitude EMA | 3 | Indépendance du gradient |
| Pool de caractéristiques | 5 | FIFO + diversité |

### Côté C : Frama-C ACSL + Contrats d'exécution

Le moteur d'inférence C utilise deux moyens formels :

**① Annotations Frama-C ACSL** (`/*@ assert ... */`)

Des assertions ACSL sont intégrées aux points critiques de calcul numérique, vérifiables statiquement via Frama-C :

```c
/* hyp.c — Opérations hyperboliques de Poincaré */
/*@ assert denom > 0.0f; */    /* Dénominateur toujours positif (pas de division par zéro) */
/*@ assert arg >= 1.0f; */     /* Vérification du domaine d'arcosh */
/*@ assert t < 1.0f; */        /* Domaine d'atanh : |t| < 1 */

/* lattice.c — Récupération par treillis */
/*@ assert best_idx >= 0 && best_idx < mem->M; */  /* Sécurité des limites du codebook */
/*@ assert mag > 0.0f; */                          /* Amplitude FFT toujours positive */

/* engine.c — Moteur d'inférence */
/*@ assert diff >= 0.0f; */    /* Critère de convergence non négatif */
/*@ assert w > 0.0f; */        /* Poids de fusion toujours positifs */
```

**② Contrats de conception REQUIRE/ENSURE** (assertions à l'exécution)

Les modules de sécurité critiques utilisent des pré/post-conditions de style DbC :

```c
#define REQUIRE(cond) assert(cond)
#define ENSURE(cond)  assert(cond)

void gvalue_init(gvalue_t* gv, ...) {
    REQUIRE(gv != NULL && C_pos != NULL);
    REQUIRE(D == LCM_D);
    // ...
    ENSURE(gv->integrity_hash[0] != '\0');
}
```

**③ Cibles de compilation**

```bash
cd infer
make contracts    # Active -DLCM_USE_CONTRACTS, vérification des contrats à l'exécution
make test         # Tests unitaires (DEBUG + contrats)
make debug        # Construction DEBUG + contrats
```

**④ Garanties de sécurité thread** (invariants structurels)

- Pas d'état mutable `static`/global
- Toute la mémoire appartient à l'appelant (caller-owns, callee-operates)
- Tableaux de taille fixe, zéro allocation dynamique
- C99 pur, aucune dépendance externe (uniquement `libm`)

Ces invariants sont garantis par la structure du code C, et le côté Z3 (preuves de déterminisme P16) vérifie que le modèle mathématique correspondant est une fonction pure.

---

## Efficacité matérielle

| Indicateur | Valeur |
|------------|--------|
| Nombre total de paramètres | ~12M (y compris les embeddings) |
| Poids FP16 | ~24 Mo |
| Mémoire d'entraînement | < 1,5 Go |
| Exécution d'inférence | Zéro paramètre (uniquement recherche dans codebook) |
| Matériel compatible | **GPU grand public 4 Go** |

La complexité `O(N d²)` de l'attention linéaire évite le stockage de la matrice `N×N` de l'attention traditionnelle, rendant possible l'entraînement sur de longues séquences avec du matériel grand public.

### Analyse théorique de la vitesse d'inférence

**Inférence par chaîne logique** (d=256, H=4, N=512, L_enc=2, moteur DAG dynamique à zéro paramètre) :

1. **Encodeur (incrémental)** : L'implémentation originale recalcule entièrement la fenêtre glissante à chaque étape en `O(N·d²)`. Après mise à jour incrémentale, chaque étape ne nécessite plus que `O(d²)` — soit **1/512**. JAX fusionne les petites opérations matricielles en quelques kernels CUDA ; le coût de lancement (~10μs) domine le calcul lui-même. GPU ~25-50μs, CPU ~50-100μs.

2. **Moteur d'inférence C (boucle macro + DAG dynamique)** : C'est le coût dominant. Chaque étape macro :
   - `build_dag()` calcule les distances entre `z` et chaque codebook de treillis, puis sélectionne dynamiquement les primitives activées (seuls les treillis dont la distance est inférieure au seuil sont ajoutés comme nœuds DAG ; la topologie varie à chaque étape)
   - Un DAG à 4 couches s'exécute : retrieves (parallèle) → bind → unbind → fusion. Sur du C monotâche, chaque primitive active s'exécute séquentiellement dans sa couche
   - Les vérifications de sécurité (treillis de danger + treillis de valeur globale) s'exécutent après la fusion
   
   La boucle macro répète **3 à 5 étapes** jusqu'à double convergence : `||Δz|| < tol` ET entropie des poids de fusion `H({w_i}) < entropy_threshold`. Chaque étape macro construit un DAG frais — le graphe de calcul n'est pas fixe, il s'adapte dynamiquement au `z` courant.
   
   Un balayage de distance de codebook simple prend ~50-90μs par treillis actif (résident en cache L3, limité par la bande passante mémoire). Avec typiquement 3-6 treillis activés par étape, plus la construction du graphe, bind/unbind, fusion et vérifications de sécurité, chaque étape macro est ~300-600μs. Sur 3-5 étapes : **~1,0-3,0 ms au total**. Cette partie s'exécute toujours sur CPU, que le GPU soit utilisé ou non — les données des codebooks résident dans la mémoire hôte.

3. **Décodeur + échantillonnage** : Attention linéaire + GLU (JAX). GPU ~20-40μs, CPU ~50-150μs.

4. **Pont JAX↔C** : `z` transite entre les tableaux JAX et les pointeurs C via ctypes (deux traversées par token) : **~20-60μs**.

5. **Boucle Python** : Contrôle de flux et gestion d'état : **~10-30μs**.

**Estimation finale** (latence par token, inférence unitaire, d=256) :

| Élément | GPU | CPU |
|---------|-----|-----|
| Encodeur incrémental | 25-50μs | 50-100μs |
| Pont JAX↔C (×2) | 20-60μs | — |
| Moteur C (3-5 étapes macro × DAG dynamique) | 1 000-3 000μs | 1 000-3 000μs |
| Décodeur + échantillonnage | 20-40μs | 50-150μs |
| Boucle Python | 10-30μs | 10-30μs |
| **Total par token** | **1 075-3 180μs** | **1 110-3 280μs** |
| **Débit** | **310-930 tok/s** | **300-900 tok/s** |

La boucle macro du moteur C domine (~80-90% du temps total). Le calcul de distance des codebooks est limité par la bande passante mémoire et s'exécute sur CPU quel que soit le mode GPU, donc les débits GPU et CPU sont similaires pour l'inférence unitaire. L'exécution par lots de plusieurs requêtes améliorerait l'utilisation GPU pour l'encodeur/décodeur, mais le coût du moteur C par séquence ne s'amortit pas sur la dimension du lot.

> Ces estimations sont théoriques pour l'inférence unitaire. L'implémentation actuelle du pont Python + ctypes peut ajouter une surcharge. Le débit d'entraînement est de 3 000-5 000 tok/s (B=16, N=512, GPU), limité par le chargement des données et les mises à jour de l'optimiseur. Déporter le calcul de distance vers un noyau GPU pourrait théoriquement réduire le temps de récupération, mais le flux de contrôle dynamique du DAG, le bind/unbind, la fusion et les vérifications de convergence sont fondamentalement inadaptés à l'exécution GPU. Chaque étape macro nécessiterait également au moins un aller-retour PCIe (z → GPU, distances → CPU). Pour l'inférence unitaire, les gains seraient marginaux.

---

## Citation

```bibtex
@software{lcm2026,
  title = {晶格认知模型 (Lattice Cognitive Model, LCM)},
  description = {A cognitive architecture with multi-lattice codebook retrieval,
                 hyperbolic residual quantization, and a zero-parameter C inference engine},
  author = {LCM Contributors},
  year = {2026},
}
```
