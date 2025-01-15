## 🤖 Présentation du Projet

Ce projet implémente un système avancé de simplification de texte utilisant l'apprentissage profond.

### 🎯 Objectif Principal
Réduire la complexité linguistique des textes tout en préservant leur sens original, rendant l'information plus accessible à un public plus large.

## 🧠 Architecture Technique

### Modèle
- Modèle T5 (Text-to-Text Transfer Transformer)
- PyTorch
- Hugging Face Transformers

### Métriques d'Évaluation
1. **SARI** (Simplification Awareness Reduced Information)
   - Mesure la qualité de simplification
   - Évalue:
     * Mots conservés
     * Mots supprimés
     * Mots ajoutés

## 🛠 Prérequis Techniques

### Dépendances
- Python 3.8+
- PyTorch
- Transformers
- NLTK
- Evaluate
- sacrebleu sacremoses

### Installation
```bash
pip install torch transformers nltk evaluate sacrebleu sacremoses
```

## 🔍 Détails 

### Preprocessing
- Tokenization (auto)
- Troncature (réduction de la longueur)
- Padding (uniformiser la longueur des séquences dans un lot (batch) )

### Optimisations
- Gradient Checkpointing : Technique pour réduire l'utilisation de la mémoire GPU
- Mixed Precision Training : Technique d'entraînement qui utilise à la fois des précisions de calcul 16 et 32 bits
- Adaptive Learning Rate : Adaptation du taux d'apprentissage

## 🧩 Personnalisation

### Paramètres Configurables
- Modèle de base
- Taille du modèle
- Hyperparamètres d'entraînement
- Métriques d'évaluation
---
