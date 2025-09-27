# 📁 Dossier des Modèles Pré-entraînés

Ce dossier contient les poids des modèles GNN pré-entraînés utilisés par l'application web.

## 🏗️ Structure

```
models/
├── README.md              # Ce fichier
├── stripe_lite_model.pth # Modèle STRIPE Lite (à créer)
└── tgn_model.pth         # Modèle TGN (copié depuis MAWI/)
```

## 🎯 Modèles Disponibles

### 1. **STRIPE Litet** (`stripe_lite_model.pth`)
- **Framework** : PyTorch + PyTorch Geometric
- **Architecture** : TemporalGNNAnomalyDetector
- **Usage** : Dataset de base (edge_events_clean.csv)
- **Status** : À créer/entraîner

### 2. **TGN** (`tgn_model.pth`)
- **Framework** : PyTorch + PyTorch Geometric
- **Architecture** : TemporalGNNAnomalyDetector (TGN version)
- **Usage** : Datasets uploadés et analyses avancées
- **Status** : ✅ Disponible (copié depuis MAWI/)

## 🚀 Comment Ajouter un Nouveau Modèle

### Étape 1: Entraînement sur Google Colab
```python
# Dans votre notebook Colab
import torch
from gnn_anomaly_detection import TemporalGNNAnomalyDetector

# Entraîner votre modèle
model = TemporalGNNAnomalyDetector(...)
# ... processus d'entraînement ...

# Sauvegarder les poids
torch.save(model.state_dict(), 'mon_modele.pth')
```

### Étape 2: Téléchargement et Placement
```bash
# Télécharger le fichier .pth depuis Colab
# Le placer dans ce dossier models/
# Renommer selon la convention : nom_du_modele.pth
```

### Étape 3: Mise à Jour du Code
```python
# Dans gnn_anomaly_detection.py - ModelFactory
model_paths = {
    'STRIPE_Lite': 'models/stripe_lite_model.pth',
    'TGN': 'models/tgn_model.pth',
    'MON_MODELE': 'models/mon_modele.pth'  # Nouveau modèle
}
```

### Étape 4: Utilisation dans l'Application
```python
# Dans le backend
pipeline = GraphAnomalyDetectionPipeline(
    model_name='MON_MODELE',
    model_path='models/mon_modele.pth'
)
```

## 🔧 Configuration des Modèles

### Paramètres du Modèle
```python
TemporalGNNAnomalyDetector(
    num_nodes=1000,      # Nombre de nœuds dans votre graphe
    edge_dim=64,         # Dimension des embeddings d'arêtes
    msg_dim=5,          # Nombre de types d'arêtes (HAS_PORT, DEPENDS_ON, etc.)
    hidden_dim=64       # Dimension cachée
)
```

### Types d'Arêtes Supportés
- `HAS_PORT` (index 0)
- `DEPENDS_ON` (index 1)
- `INSTALLED_AT` (index 2)
- `HAS` (index 3)
- `REFERS_TO` (index 4)

## 📊 Format des Données d'Entraînement

Les modèles attendent des données au format :
```python
{
    'src': [nœud_source, ...],      # Liste des nœuds source
    'dst': [nœud_destination, ...], # Liste des nœuds destination
    't': [timestamp, ...],          # Liste des timestamps
    'msg': [one_hot_label, ...]     # Encodage one-hot du label d'arête
}
```

## 🔍 Dépannage

### Erreur "Model file not found"
- Vérifiez que le fichier `.pth` existe dans le dossier `models/`
- Vérifiez les permissions du fichier
- Vérifiez que le nom du modèle correspond dans `ModelFactory`

### Erreur "KeyError" lors du chargement
- Vérifiez que les paramètres du modèle correspondent (num_nodes, msg_dim, etc.)
- Le modèle doit être compatible avec les données d'inférence

### Performance
- Les modèles sont chargés en mode `eval()` pour l'inférence
- Utilisez un device approprié (CPU/GPU) selon vos besoins

## 📈 Améliorations Possibles

- [ ] Support pour différents formats de modèles (ONNX, TensorFlow)
- [ ] Interface web pour upload de nouveaux modèles
- [ ] Comparaison de performance entre modèles
- [ ] Versionning automatique des modèles
