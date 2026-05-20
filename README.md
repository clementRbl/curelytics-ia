# 🧠 BrainScanAI — Détection de tumeurs cérébrales par apprentissage semi-supervisé

Projet R&D **CurelyticsIA** — Data Science / Computer Vision

Détection automatisée de tumeurs cérébrales sur IRM, en exploitant un grand corpus d'images **non labellisées** grâce à l'apprentissage semi-supervisé.

---

## 📋 Contexte

CurelyticsIA, startup en e-santé, souhaite automatiser la détection de tumeurs cérébrales sur IRM. Le dataset comporte :

- **100 images fortement labellisées** (50 cancer / 50 normal) — annotées par des radiologues experts
- **1 406 images non labellisées** — abondantes mais sans annotation

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

## 📊 Résultats

### Clustering (score ARI sur données fortement labellisées)

| Algorithme | ARI |
|---|---|
| **Agglomerative (Ward)** | **0.70** ✓ |
| GMM | 0.51 |
| K-Means | 0.43 |
| DBSCAN | n/a |

### Classification : Supervisé vs Semi-supervisé

| Métrique | Supervisé | Semi-supervisé | Gain |
|---|---|---|---|
| Accuracy | 0.90 | **1.00** | +10 % |
| F1-Score | 0.90 | **1.00** | +10 % |
| **Recall cancer** ★ | 0.90 | **1.00** | +10 % |
| AUC-ROC | 0.99 | **1.00** | +1 % |

> ★ Le **Recall** est la métrique prioritaire en imagerie médicale : un faux négatif (tumeur non détectée) est plus grave qu'un faux positif.

**Conclusion :** l'apprentissage semi-supervisé améliore les performances sur toutes les métriques. Résultats à confirmer sur un dataset plus large (test set de 20 images).

---

## ✅ Definition of Done

- [x] ARI > 0.3 → atteint à **0.70**
- [x] Recall cancer > 0.80 (supervisé) → **0.90**
- [x] F1 semi-supervisé ≥ F1 supervisé → **1.00 vs 0.90**
- [x] Labels faibles et forts strictement séparés
- [x] Même jeu de test pour Phase A et Phase B

---

## 📝 Règle critique respectée

> **Ne jamais mélanger** les données faiblement labellisées (clustering) et les données fortement labellisées (radiologues).

Vérifié : les 1 406 images "weak" et les 100 images "strong" sont **strictement disjointes**, et le jeu de test n'est utilisé nulle part pendant l'entraînement.

---

## 🛠️ Technologies

`Python 3.10` · `PyTorch` · `torchvision` · `scikit-learn` · `pandas` · `numpy` · `matplotlib` · `seaborn` · `Pillow`
