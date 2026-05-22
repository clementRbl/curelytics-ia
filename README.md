# 🧠 BrainScanAI — Détection de tumeurs cérébrales par apprentissage semi-supervisé

Projet R&D **CurelyticsIA** — Data Science / Computer Vision

Détection automatisée de tumeurs cérébrales sur IRM, en exploitant un grand corpus d'images **non labellisées** grâce à l'apprentissage semi-supervisé.

---

## 📋 Contexte

CurelyticsIA, startup en e-santé, souhaite automatiser la détection de tumeurs cérébrales sur IRM. Le dataset brut comporte 1 506 images (100 labellisées + 1 406 non labellisées). Après **déduplication** (cf. plus bas), il reste :

- **99 images fortement labellisées** (50 cancer / 49 normal) — annotées par des radiologues experts
- **1 311 images non labellisées** — abondantes mais sans annotation

Annoter manuellement coûte cher (budget de 300 €, insuffisant). L'objectif est donc d'exploiter au maximum les données non labellisées via le **semi-supervisé**, et d'évaluer la faisabilité d'un passage à l'échelle (4M images, budget 5 000 €).

---

## 🎯 Objectifs

1. **Extraire des features visuelles** via un modèle pré-entraîné (ResNet-50)
2. **Clustering non-supervisé** pour identifier des regroupements naturels et générer des "labels faibles"
3. **Apprentissage semi-supervisé** : pré-entraînement sur labels faibles + fine-tuning sur labels forts
4. **Synthétiser** les résultats et formuler des recommandations de passage à l'échelle

---

## 🗂️ Structure du projet

```
curelytics-ia/
├── notebooks/
│   ├── 01_exploration_et_features.ipynb      # EDA + extraction features ResNet-50
│   └── 02_clustering_et_semisupervise.ipynb  # Clustering + semi-supervisé
├── data/                          # Données brutes (non versionnées)
│   ├── avec_labels/
│   │   ├── cancer/                # 50 images
│   │   └── normal/                # 50 images
│   └── sans_label/                # 1 406 images
├── outputs/                       # Embeddings + résultats (non versionnés sauf CSV)
├── requirements.txt
└── README.md
```

> ℹ️ Les dossiers `data/`, les fichiers `.npy`/`.png` d'`outputs/` et `presentation/` sont exclus du git (`.gitignore`) car volumineux ou locaux.

---

## ⚙️ Installation

### Prérequis
- Python 3.10
- conda (recommandé) ou venv

### Avec conda (recommandé)

```bash
# Créer l'environnement
conda create -n brainscanai python=3.10 -y

# Installer PyTorch (CPU)
conda run -n brainscanai pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Installer les autres dépendances
conda run -n brainscanai pip install -r requirements.txt

# Enregistrer le kernel Jupyter
conda run -n brainscanai python -m ipykernel install --user --name brainscanai --display-name "Python (brainscanai)"
```

### Avec venv + pip

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## ▶️ Lancement

1. Placer les données dans `data/` (structure ci-dessus).
2. Ouvrir les notebooks dans VS Code ou Jupyter :

```bash
conda run -n brainscanai jupyter lab
```

3. **Sélectionner le kernel** `Python (brainscanai)`.
4. Exécuter les notebooks **dans l'ordre** :
   - `01_exploration_et_features.ipynb` → génère les embeddings dans `outputs/`
   - `02_clustering_et_semisupervise.ipynb` → utilise ces embeddings pour le clustering et le semi-supervisé

> ⚠️ Le Notebook 2 dépend des fichiers `.npy` produits par le Notebook 1. Exécuter le Notebook 1 en premier.

---

## 🔬 Pipeline technique

| Étape | Méthode | Outil |
|---|---|---|
| **1. Exploration** | Analyse qualité, distribution, biais | Pillow, pandas, matplotlib |
| **2. Features** | ResNet-50 gelé → embeddings 2048D | PyTorch, torchvision |
| **3. Réduction dim.** | StandardScaler + PCA 50D + t-SNE 2D | scikit-learn |
| **4. Clustering** | K-Means, DBSCAN, Agglomerative, GMM | scikit-learn |
| **5. Semi-supervisé** | Pré-entraînement (weak) → fine-tuning (strong) | PyTorch |

---

## 🧹 Déduplication (étape critique anti-fuite)

Un audit MD5 du dataset brut a révélé **96 doublons**, dont **32 images labellisées également présentes dans le dossier non labellisé**. Sans correction, ces copies pouvaient faire fuiter des images de test dans l'entraînement et gonfler les scores.

Une étape de déduplication par hash MD5 (Notebook 1) retire ces doublons : **1 506 → 1 410 images** (99 labellisées + 1 311 non labellisées). La séparation labellisé / non-labellisé devient ainsi robuste à n'importe quel split.

---

## 📊 Résultats

### Clustering (score ARI sur données fortement labellisées)

| Algorithme | ARI |
|---|---|
| **GMM** | **0.48** ✓ |
| K-Means | 0.43 ✓ |
| Agglomerative (Ward) | 0.28 |
| DBSCAN | n/a (tous outliers) |

### Classification : Supervisé vs Semi-supervisé (test = 20 images)

| Métrique | Supervisé | Semi-supervisé |
|---|---|---|
| Accuracy | 0.95 | 0.95 |
| F1-Score | 0.95 | 0.95 |
| AUC-ROC | 0.99 | 0.98 |
| **Recall cancer** ★ | **1.00** | 0.90 |
| Precision cancer | 0.91 | **1.00** |

> ★ Le **Recall** est la métrique prioritaire en imagerie médicale : un faux négatif (tumeur non détectée) est plus grave qu'un faux positif.

**Conclusion :** sur ce dataset dédupliqué, les deux modèles sont **à égalité** sur Accuracy / F1 / AUC. Ils ne diffèrent que sur l'arbitrage recall/precision, soit **1 seule image de test** classée différemment — un écart dans le bruit statistique pour n=20. Le semi-supervisé est donc **compétitif** avec le supervisé ; une validation sur un test set plus large est nécessaire pour conclure à un gain net.

> 💡 Avant déduplication, le semi-supervisé semblait dominer avec des scores parfaits (1.00). Ces scores étaient **partiellement gonflés par la fuite de données** — un bon rappel de l'importance de l'hygiène des données.

---

## ✅ Definition of Done

- [x] ARI > 0.3 → atteint à **0.48** (GMM)
- [x] Recall cancer > 0.80 (supervisé **1.00**, semi-supervisé **0.90**)
- [x] Données dédupliquées, jeu de test isolé (aucune fuite)
- [x] Labels faibles et forts strictement séparés
- [x] Même jeu de test pour Phase A et Phase B
- [x] Comparaison supervisé vs semi-supervisé réalisée

---

## 📝 Règle critique respectée

> **Ne jamais mélanger** les données faiblement labellisées (clustering) et les données fortement labellisées (radiologues).

Vérifié : les 1 311 images "weak" et les 99 images "strong" sont **strictement disjointes** (par chemin ET par contenu MD5), et le jeu de test n'est utilisé nulle part pendant l'entraînement.

---

## 🛠️ Technologies

`Python 3.10` · `PyTorch` · `torchvision` · `scikit-learn` · `pandas` · `numpy` · `matplotlib` · `seaborn` · `Pillow`
