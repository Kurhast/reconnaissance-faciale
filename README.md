# Reconnaissance faciale avec opencv

Ce projet implémente un système de reconnaissance faciale en utilisant la bibliothèque opencv et l'ensemble de données yaleface. Il propose plusieurs modèles de reconnaissance faciale (LBPH, Eigenfaces, Fisherfaces) et inclut également un système de reconnaissance en temps réel avec la webcam.

## Prérequis

Avant de commencer, assurez-vous d'avoir les bibliothèques suivantes installées :

- `opencv-python`
- `numpy`
- `Pillow`

Vous pouvez les installer via pip :

```bash
pip install opencv-python numpy Pillow
```
## Utilisation

### Exécution du code d'entraînement

Pour entraîner le modèle de reconnaissance facile, exécutez le fichier [`app.py`](#app.py).
Vous pouvez choisir le classificateur à utiliser parmi `LBP`, `Eigen`, ou `Fisher`:

```bash
python app.py --classifier lbp
```
### Exécution du code d'entraînement

Pour tester la reconnaissance faciale en temps réel avec la webcam, exécutez le fichier [`realtime.py`](#realtime.py):

```bash
python realtime.py
```

### Structure des dossiers

- `yalefaces/`: Ce dossier doit contenire les images faciales pour l'entraînement et les tests. Les fichiers contenant `.wink`dans leur nom sont utilisés pour le test, tandis que les autre sont utilisés pour l'entraînement.
- `haarcascade_frontalface_default.xml`: Fichier de classificateur Haar utilisé pour la détection de visage.

## Détails des fichiers

### app.py

Ce fichier est le point d'entrée pour l'entraînement du modèle de reconnaissance faciale. Il charge les données, entraîne un modèle selon le classificateur spécifié et évalue la précision du modèle.

- **Importation des modules** : Ce script utilise `argparse` pour gérer les paramètres en ligne de commande, `cv2` pour OpenCV et `numpy` pour les tableaux de données.
- **Choix du classificateur** : Le modèle peut utiliser trois types de classificateurs :
    - **LBP** (Local Binary Patterns)
    - **Eigenfaces**
    - **Fisherfaces**
- **Chargement des données** : Le fichier `dataset.py` est utilisé pour charger les données d'entraînement et de test à partir du dossier `yalefaces/`. La fonction `load_data()` renvoie des ensembles de données pour les visages détectés dans ces images.
- **Entraînement** : Le modèle sélectionné est entraîné avec les images de visages d'entraînement, puis testé sur un jeu de données séparé.
- **Évaluation** : Le taux de précision du modèle est calculé en comparant les prédictions du modèle avec les étiquettes réelles des visages dans le jeu de test.

### dataset.py

Ce fichier contient les fonctions nécessaires pour charger et préparer les données d'entraînement et de test.

- `_extract_face(filepath, face_cascade)` : Cette fonction prend une image et utilise un classificateur en cascade (Haar) pour détecter la zone du visage dans l'image. Le visage est ensuite extrait et renvoyé sous forme de tableau numpy. Si un visage n'est pas détecté, le programme se termine avec un message d'erreur.
- `load_data(face_cascade, data_dir='yalefaces')` : Cette fonction parcourt les fichiers dans le dossier `yalefaces/`, les sépare en données d'entraînement et de test, et extrait les visages de chaque image. Les images dont le nom se termine par `.wink` sont utilisées pour les tests, tandis que les autres sont utilisées pour l'entraînement.
    - **Entraînement** : Les images de visages sont converties en niveaux de gris, et le modèle est entraîné à partir de ces données.
    - **Test** : Les données de test servent à évaluer la précision du modèle.

### realtime.py

Ce fichier implémente la reconnaissance faciale en temps réel à l'aide de la webcam.

- **Chargement des données et entraînement du modèle** : Comme dans `app.py`, ce fichier utilise `load_data()` pour charger les visages d'entraînement et entraîne un modèle LBPH.
- **Capture vidéo** : Utilisation de `cv2.VideoCapture(0)` pour accéder à la webcam en temps réel.
- **Détection et prédiction** :
    - La vidéo en temps réel est convertie en niveaux de gris.
    - Le classificateur Haar détecte les visages dans chaque image capturée par la webcam.
    - Le modèle LBPH préalablement entraîné prédit l'identité du visage détecté, et le résultat est affiché à l'écran (étiquette de la personne et niveau de confiance).
- **Affichage** : Les résultats sont affichés directement sur l'image capturée de la webcam, avec un rectangle entourant chaque visage détecté et un texte indiquant l'identité prédite ainsi que la confiance associée. 
- **Arrêt du programme** : Appuyez sur `q` pour fermer la fenêtre et arrêter le programme.
