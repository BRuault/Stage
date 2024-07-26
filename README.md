# Stage Repository

Ce dépôt contient plusieurs scripts Python destinés à traiter et analyser des images médicales en utilisant des techniques de traitement d'images et des filtres spécifiques.

## Contenu du dépôt

- `evaluate.py` : Ce script évalue les résultats des traitements appliqués aux images. 
  
- `frangi.py` : Implémente le filtre de Frangi, utilisé pour le rehaussement des structures tubulaires dans les images. Ce filtre est couramment utilisé pour détecter les vaisseaux sanguins dans les images médicales.
  
- `hessian.py` : Contient des fonctions pour le calcul de la matrice de Hessian, qui est essentielle pour l'application du filtre de Frangi et d'autres techniques de détection de structures.
  
- `utils.py` : Regroupe des fonctions utilitaires utilisées par les autres scripts.
